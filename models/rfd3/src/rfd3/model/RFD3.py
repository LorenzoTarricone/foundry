import os

import hydra
import torch
from omegaconf import DictConfig
from rfd3.model.cfg_utils import (
    strip_f,
)
from rfd3.model.inference_sampler import ConditionalDiffusionSampler
from rfd3.model.layers.encoders import TokenInitializer
from rfd3.model.RFD3_diffusion_module import RFD3DiffusionModule
from torch import nn

from foundry.utils.ddp import RankedLogger

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


class RFD3(nn.Module):
    """
    Simplified model for generation
    This module level serves to wrap the diffusion module of AF3
    to be roughly equivalent to the AF3 model w/o trunk processing.

    Allows the same sampler to be used
    """

    def __init__(
        self,
        *,
        # Channel dimensions ('global' features)
        c_s: int,
        c_z: int,
        c_atom: int,
        c_atompair: int,
        # Arguments for modules that will be instantiated
        token_initializer: DictConfig | dict,
        diffusion_module: DictConfig | dict,
        inference_sampler: DictConfig | dict,
        **_,
    ):
        super().__init__()
        # Check for memory optimization modes via environment variables
        # These two modes are ORTHOGONAL and can be combined:
        #
        # LOW_MEMORY_MODE: Sparse P_LL via chunked_pairwise_embedder
        #   - Computes P_LL only for k sparse neighbors per atom
        #   - Reduces memory from O(L²) to O(L·k)
        #
        # ATTENTION_PARALLEL: Multi-GPU parallel cross-attention
        #   - Splits queries across GPUs (each GPU handles L/n atoms)
        #   - Z_II is [I_par, I] per GPU instead of [I, I]
        #   - P_LL computation is split across GPUs
        #
        # Combinations:
        #   - Neither: Full O(L² + I²) tensors
        #   - LOW_MEM only: Sparse P_LL, full Z_II
        #   - PARALLEL only: Cross-attention with full P_chunk [L_par, L]
        #   - Both: Sparse P_LL split across GPUs (most memory efficient)
        #
        low_mem = os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"
        # Parallel mode: =0 or unset → standard, any non-zero value → parallel
        attn_par_val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
        attn_parallel = attn_par_val not in ("0", "", "false", "False")
        
        # chunked_pll is INDEPENDENT of attn_parallel
        use_chunked_pll = low_mem
        
        ranked_logger.info(
            f"RFD3 memory modes: LOW_MEMORY={low_mem}, ATTENTION_PARALLEL={attn_parallel}"
        )
        if low_mem:
            ranked_logger.info("  -> Sparse P_LL via chunked_pairwise_embedder")
        if attn_parallel:
            ranked_logger.info("  -> Multi-GPU cross-attention (no full I×I tensors)")
        if low_mem and attn_parallel:
            ranked_logger.info("  -> Combined: Sparse P_LL split across GPUs (maximum memory savings)")

        # Factory pattern: instantiate parallel vs standard classes based on attn_parallel
        if attn_parallel:
            # Import parallel classes only when needed
            from rfd3.model.parallel.layers.encoders import ParallelTokenInitializer
            from rfd3.model.parallel.diffusion_module import ParallelDiffusionModule

            ranked_logger.info("  -> Using parallel classes: ParallelTokenInitializer, ParallelDiffusionModule")

            # Use parallel token initializer
            self.token_initializer = ParallelTokenInitializer(
                c_s=c_s,
                c_z=c_z,
                c_atom=c_atom,
                c_atompair=c_atompair,
                use_chunked_pll=use_chunked_pll,
                **token_initializer,
            )

            # Manually instantiate parallel diffusion module
            # (can't use hydra.utils.instantiate because we need ParallelDiffusionModule class)
            diffusion_module_config = dict(diffusion_module)
            diffusion_module_config.pop("_target_", None)  # Remove _target_ if present
            self.diffusion_module = ParallelDiffusionModule(
                c_atom=c_atom,
                c_atompair=c_atompair,
                c_s=c_s,
                c_z=c_z,
                **diffusion_module_config,
            )
        else:
            # Use standard classes
            ranked_logger.info("  -> Using standard classes: TokenInitializer, RFD3DiffusionModule")

            # Simple constant-feature initializer
            self.token_initializer = TokenInitializer(
                c_s=c_s,
                c_z=c_z,
                c_atom=c_atom,
                c_atompair=c_atompair,
                use_chunked_pll=use_chunked_pll,
                **token_initializer,
            )

            # Diffusion module instantiated to allow for config scripting
            self.diffusion_module = hydra.utils.instantiate(
                diffusion_module, c_atom=c_atom, c_atompair=c_atompair, c_s=c_s, c_z=c_z
            )

        self.use_classifier_free_guidance = (
            inference_sampler["use_classifier_free_guidance"]
            and inference_sampler["cfg_scale"] != 1.0
        )
        self.cfg_features = inference_sampler.pop("cfg_features", [])

        # ... initialize the inference sampler, which performs a full diffusion rollout during inference
        self.inference_sampler = ConditionalDiffusionSampler(**inference_sampler)

    def forward(
        self,
        input: dict,
        coord_atom_lvl_to_be_noised: torch.Tensor = None,
        n_cycle=None,
        **_,
    ) -> dict:
        # Ensure model is on the correct device for this rank (CRITICAL for distributed training)
        import torch.distributed as dist
        import os
        
        if dist.is_initialized():
            # In distributed mode, use LOCAL_RANK to ensure each rank uses its assigned GPU
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        else:
            # In single-GPU mode, use device from input tensors
            device = coord_atom_lvl_to_be_noised.device if coord_atom_lvl_to_be_noised is not None else input["f"]["restype"].device
        
        self.to(device)
        
        initializer_outputs = self.token_initializer(input["f"])

        if self.training:
            # Single denoising step
            return self.diffusion_module(
                X_noisy_L=input["X_noisy_L"],
                t=input["t"],
                f=input["f"],
                n_recycle=n_cycle,
                **initializer_outputs,
            )  # [D, L, 3]
        else:
            if self.use_classifier_free_guidance:
                f_ref = strip_f(input["f"], self.cfg_features)
                ref_initializer_outputs = self.token_initializer(f_ref)
            else:
                f_ref = None
                ref_initializer_outputs = None

            return self.inference_sampler.sample_diffusion_like_af3(
                f=input["f"],
                f_ref=f_ref,  # for cfg
                diffusion_module=self.diffusion_module,
                diffusion_batch_size=coord_atom_lvl_to_be_noised.shape[0],
                coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised,
                # Forwarded as **kwargs:
                initializer_outputs=initializer_outputs,
                ref_initializer_outputs=ref_initializer_outputs,  # for cfg
            )
