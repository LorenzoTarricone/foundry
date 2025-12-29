import inspect
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple, List

import torch
import torch.distributed as dist
from jaxtyping import Float
from tqdm import tqdm
from rfd3.inference.symmetry.symmetry_utils import apply_symmetry_to_xyz_atomwise
from rfd3.model.cfg_utils import strip_X

from foundry.common import exists
from foundry.utils.alignment import weighted_rigid_align
from foundry.utils.ddp import RankedLogger
from foundry.utils.rotation_augmentation import (
    rot_vec_mul,
    uniform_random_rotation,
)

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


# =============================================================================
# Multi-GPU Parallel Inference Utilities
# =============================================================================

def _get_n_parallel() -> int:
    """Get parallelism factor from environment, or 0 if not set."""
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", None)
    if val is None:
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


def _is_streaming_mode() -> bool:
    """Check if streaming/parallel attention mode is enabled."""
    return _get_n_parallel() > 1


def _get_gpu_rank_and_world_size() -> Tuple[int, int]:
    """Get current GPU rank and total world size for distributed processing."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def _compute_gpu_query_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Compute the query index range for a specific GPU.
    
    Each GPU handles a contiguous chunk of queries. Queries are split evenly,
    with earlier ranks getting any remainder.
    
    Args:
        total: Total number of queries (I or L)
        rank: This GPU's rank (0 to world_size-1)
        world_size: Total number of GPUs
        
    Returns:
        (start_idx, end_idx): Query range for this GPU [start, end)
    """
    chunk_size = total // world_size
    remainder = total % world_size
    
    # Earlier ranks get one extra if there's remainder
    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size
    
    return start, end


def _all_gather_variable_size(
    tensor: torch.Tensor, 
    dim: int, 
    sizes: List[int],
) -> torch.Tensor:
    """
    Gather tensors of potentially different sizes along a dimension.
    
    Args:
        tensor: Local tensor chunk
        dim: Dimension along which chunks vary
        sizes: List of sizes for each GPU's chunk
        
    Returns:
        Concatenated tensor from all GPUs
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    # Create placeholder tensors for each GPU's contribution
    gathered = []
    for i, size in enumerate(sizes):
        # Create tensor with correct size for GPU i
        shape = list(tensor.shape)
        shape[dim] = size
        gathered.append(torch.zeros(shape, dtype=tensor.dtype, device=tensor.device))
    
    # Gather all tensors
    dist.all_gather(gathered, tensor)
    
    return torch.cat(gathered, dim=dim)


def _all_gather_concat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Gather tensors from all GPUs and concatenate along specified dimension.
    
    Assumes all tensors have the same size along the gather dimension.
    
    Args:
        tensor: Local tensor to gather
        dim: Dimension to concatenate along
        
    Returns:
        Concatenated tensor from all GPUs
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    # Gather all tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    return torch.cat(gathered, dim=dim)


def _broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank to all GPUs."""
    if not dist.is_initialized():
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


@dataclass(kw_only=True)
class SampleDiffusionConfig:
    kind: Literal["default", "symmetry"] = "default"

    # Standard EDM args
    num_timesteps: int = 200
    min_t: int = 0
    max_t: int = 1
    sigma_data: int = 16
    s_min: float = 4e-4
    s_max: int = 160
    p: int = 7
    gamma_0: float = 0.6
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    solver: Literal["af3"] = "af3"

    # RFD3 / design args
    center_option: str = "all"
    s_trans: float = 1.0
    s_jitter_origin: float = 0.0
    fraction_of_steps_to_fix_motif: float = 0.0
    skip_few_diffusion_steps: bool = False
    allow_realignment: bool = False
    insert_motif_at_end: bool = True
    use_classifier_free_guidance: bool = False
    cfg_scale: float = 2.0
    cfg_t_max: float | None = None


class SampleDiffusionWithMotif(SampleDiffusionConfig):
    """Diffusion sampler that supports optional motif alignment."""

    def _construct_inference_noise_schedule(
        self, device: torch.device, partial_t: float = None
    ) -> torch.Tensor:
        """Constructs a noise schedule for use during inference.

        The inference noise schedule is defined in the AF-3 supplement as:

            t_hat = sigma_data * (s_max**(1/p) + t * (s_min**(1/p) - s_max**(1/p)))**p

        Returns:
            torch.Tensor: A tensor representing the noise schedule `t_hat`.

        Reference:
            AlphaFold 3 Supplement, Section 3.7.1.
        """
        # Create a linearly spaced tensor of timesteps between min_t and max_t
        t = torch.linspace(self.min_t, self.max_t, self.num_timesteps, device=device)

        # Construct the noise schedule, using the formula provided in the reference
        t_hat = (
            self.sigma_data
            * (
                (self.s_max) ** (1 / self.p)
                + t * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )

        if partial_t is not None:
            # For now, partial t is a global parameter
            partial_t = float(partial_t.mean())
            noise_schedule = t_hat
            ranked_logger.info("Using partial diffusion with t={}".format(partial_t))

            # Debug the noise schedule filtering
            original_schedule_len = len(noise_schedule)
            original_max = noise_schedule.max().item()
            original_min = noise_schedule.min().item()

            noise_schedule = noise_schedule[noise_schedule <= partial_t]

            new_schedule_len = len(noise_schedule)
            if new_schedule_len > 0:
                new_max = noise_schedule.max().item()
                new_min = noise_schedule.min().item()
                ranked_logger.info(
                    f"Noise schedule: {original_schedule_len} → {new_schedule_len} steps"
                )
                ranked_logger.info(
                    f"Original range: [{original_min:.3f}, {original_max:.3f}]"
                )
                ranked_logger.info(f"Filtered range: [{new_min:.3f}, {new_max:.3f}]")
            else:
                ranked_logger.warning(
                    f"No noise schedule steps found with t <= {partial_t}!"
                )
                ranked_logger.info(
                    f"Original schedule range: [{original_min:.3f}, {original_max:.3f}]"
                )
                # Fallback to smallest available step
                noise_schedule_original = self._construct_inference_noise_schedule(
                    device=device
                )
                noise_schedule = noise_schedule_original[-1:]  # Just use the final step
                ranked_logger.info(
                    f"Using fallback: final step with t={noise_schedule[0].item():.6f}"
                )

        return t_hat

    def _get_initial_structure(
        self,
        c0: torch.Tensor,
        D: int,
        L: int,
        coord_atom_lvl_to_be_noised: torch.Tensor,
        is_motif_atom_with_fixed_coord,
    ) -> torch.Tensor:
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        noise[..., is_motif_atom_with_fixed_coord, :] = 0  # Zero out noise going in
        X_L = noise + coord_atom_lvl_to_be_noised
        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Diffusion sampling loop with optional multi-GPU parallel processing.
        
        When RFD3_ATTENTION_PARALLEL is set and multiple GPUs are available,
        this uses DistriFusion-style parallel inference where:
        - Queries are split across GPUs (each GPU processes I/n_parallel tokens)
        - Each GPU computes its chunk of the output
        - Results are gathered and reassembled after each diffusion step
        
        Args:
            f: Feature dictionary
            diffusion_module: The diffusion model
            diffusion_batch_size: Number of parallel diffusion samples (D)
            coord_atom_lvl_to_be_noised: [D, L, 3] coordinates to denoise
            initializer_outputs: Outputs from TokenInitializer
            ref_initializer_outputs: Reference outputs for CFG (optional)
            f_ref: Reference features for CFG (optional)
            
        Returns:
            dict with X_L, trajectories, sequence predictions
        """
        # Check for streaming/parallel mode
        streaming_mode = initializer_outputs.get("streaming_mode", False)
        n_parallel = _get_n_parallel()
        gpu_rank, world_size = _get_gpu_rank_and_world_size()
        
        if streaming_mode and world_size > 1:
            ranked_logger.info(
                f"Parallel diffusion sampling: GPU {gpu_rank}/{world_size}, n_parallel={n_parallel}"
            )
        
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]

        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]                      # Number of atoms
        D = diffusion_batch_size                           # Diffusion batch size

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )                                                  # [D, L, 3]

        if self.s_jitter_origin > 0.0:
            X_L[:, is_motif_atom_with_fixed_coord, :] += torch.normal(
                mean=0.0,
                std=self.s_jitter_origin,
                size=(D, 1, 3),
                device=X_L.device,
            )

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        threshold_step = (len(noise_schedule) - 1) * self.fraction_of_steps_to_fix_motif

        # Only show progress bar on rank 0 to avoid duplicate output
        is_main_rank = not dist.is_initialized() or dist.get_rank() == 0
        num_steps = len(noise_schedule) - 1
        pbar = tqdm(
            enumerate(zip(noise_schedule, noise_schedule[1:])),
            total=num_steps,
            desc="Sampling",
            disable=not is_main_rank,
        )
        
        for step_num, (c_t_minus_1, c_t) in pbar:
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, _ = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                    center_option=self.center_option,
                    centering_affects_motif=(max(step_num - 1, 0)) >= threshold_step,
                    s_trans=self.s_trans if step_num >= threshold_step else 0.0,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)
            
            # Update progress bar with current noise level
            if is_main_rank:
                t_val = t_hat.item() if isinstance(t_hat, torch.Tensor) else t_hat
                pbar.set_postfix({"t": f"{t_val:.3f}"})

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )                                              # [D, L, 3]
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = 0  # No noise for fixed atoms
            X_noisy_L = X_L + epsilon_L                    # [D, L, 3]

            # ================================================================
            # Denoise the coordinates - handle chunked/streaming mode
            # ================================================================
            tic = time.time()
            
            # Prepare common arguments
            chunked_embedder = initializer_outputs.get("chunked_pairwise_embedder", None)
            
            if chunked_embedder is not None or streaming_mode:
                # Chunked/streaming mode: explicitly provide P_LL=None
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k not in ("chunked_pairwise_embedder", "streaming_mode")
                }
                
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,                   # [D, L, 3]
                    t=t_hat.tile(D),                       # [D]
                    f=f,
                    P_LL=None,                             # Not used in chunked/streaming mode
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    streaming_mode=streaming_mode,         # Pass streaming flag!
                    **other_outputs,
                )
                
                toc = time.time()
                if step_num == 0:  # Log only first step to avoid spam
                    ranked_logger.info(
                        f"{'Streaming' if streaming_mode else 'Chunked'} mode step time: {toc - tic:.2f}s"
                    )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,                   # [D, L, 3]
                    t=t_hat.tile(D),                       # [D]
                    f=f,
                    **initializer_outputs,
                )

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs  # [D, L, 3]

            # ================================================================
            # Multi-GPU synchronization
            # ================================================================
            # In true multi-GPU parallel mode:
            # - Token features (S_I, A_I) are all_gathered within the encoder
            # - Each GPU computes full X_L from the synced token features
            # - X_L should be identical across GPUs, but we sync to ensure consistency
            if streaming_mode and world_size > 1:
                # Barrier to ensure all GPUs have finished this step
                dist.barrier()
                
                # Broadcast X_L from rank 0 to ensure exact consistency
                # (should be identical, but floating point differences can accumulate)
                _broadcast_tensor(X_denoised_L, src=0)
                
                # Sync sequence predictions
                if "sequence_logits_I" in outs and outs["sequence_logits_I"] is not None:
                    _broadcast_tensor(outs["sequence_logits_I"], src=0)

            # Compute the delta between the noisy and denoised coordinates, scaled by t_hat
            delta_L = (X_noisy_L - X_denoised_L) / t_hat   # [D, L, 3]
            d_t = c_t - t_hat

            # ================================================================
            # Classifier-free guidance (optional)
            # ================================================================
            if self.use_classifier_free_guidance and (
                self.cfg_t_max is None or c_t > self.cfg_t_max
            ):
                X_noisy_L_stripped = strip_X(X_noisy_L, f_ref)

                # Unconditional forward pass
                if chunked_embedder is not None or streaming_mode:
                    ref_other = {
                        k: v
                        for k, v in ref_initializer_outputs.items()
                        if k not in ("chunked_pairwise_embedder", "streaming_mode")
                    }
                    outs_ref = diffusion_module(
                        X_noisy_L=X_noisy_L_stripped,
                        t=t_hat.tile(D),
                        f=f_ref,
                        P_LL=None,
                        chunked_pairwise_embedder=ref_initializer_outputs.get(
                            "chunked_pairwise_embedder"
                        ),
                        streaming_mode=ref_initializer_outputs.get("streaming_mode", False),
                        **ref_other,
                    )
                else:
                    outs_ref = diffusion_module(
                        X_noisy_L=X_noisy_L_stripped,
                        t=t_hat.tile(D),
                        f=f_ref,
                        **ref_initializer_outputs,
                    )

                X_denoised_L_stripped = outs_ref["X_L"]

                # Sync CFG outputs if distributed
                if streaming_mode and world_size > 1:
                    dist.all_reduce(X_denoised_L_stripped, op=dist.ReduceOp.AVG)

                delta_L_ref = (X_noisy_L_stripped - X_denoised_L_stripped) / t_hat

                # Pad delta_L_ref with zeros to match delta_L
                if delta_L_ref.shape[1] < delta_L.shape[1]:
                    delta_L_ref = torch.cat(
                        [
                            delta_L_ref,
                            torch.zeros_like(delta_L[:, delta_L_ref.shape[1] :, :]),
                        ],
                        dim=1,
                    )

                # Apply CFG
                delta_L = delta_L + (self.cfg_scale - 1) * (delta_L - delta_L_ref)

            # ================================================================
            # Sequence entropy tracking
            # ================================================================
            if exists(outs.get("sequence_logits_I")):
                p = torch.softmax(outs["sequence_logits_I"], dim=-1).cpu()  # [D, I, vocab]
                seq_entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # [D, I]
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L   # [D, L, 3]

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )                                              # [D, L, 3]
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        # ================================================================
        # Post-processing: motif alignment
        # ================================================================
        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, _ = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,                                       # [D, L, 3]
            X_noisy_L_traj=X_noisy_L_traj,                 # list[[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,           # list[[D, L, 3]]
            t_hats=t_hats,                                 # list[Tensor]
            sequence_logits_I=outs.get("sequence_logits_I"),   # [D, I, vocab]
            sequence_indices_I=outs.get("sequence_indices_I"), # [D, I]
            sequence_entropy_traj=sequence_entropy_traj,   # list[[D, I]]
        )


class SampleDiffusionWithSymmetry(SampleDiffusionWithMotif):
    """
    This class is a wrapper around the SampleDiffusionWithMotif class.
    It is used to sample diffusion with symmetry.
    """

    def __init__(self, sym_step_frac: float = 0.9, **kwargs):
        assert (
            kwargs.get("gamma_0") > 0.5
        ), "gamma_0 must be greater than 0.5 for symmetry sampling"
        self.sym_step_frac = sym_step_frac
        super().__init__(**kwargs)

    def apply_symmetry_to_X_L(self, X_L, f):
        # check that we are doing symmetric inference

        assert "sym_transform" in f.keys(), "Symmetry transform not found in f"

        # update symmetric frames to correct for change in global frame
        symmetry_feats = {k: v for k, v in f.items() if "sym" in k}

        # apply symmetry frame shift to X_L
        X_L = apply_symmetry_to_xyz_atomwise(
            X_L, symmetry_feats, partial_diffusion=("partial_t" in f)
        )

        return X_L

    def sample_diffusion_like_af3(
        self,
        *,
        f: dict[str, Any],
        diffusion_module: torch.nn.Module,
        diffusion_batch_size: int,
        coord_atom_lvl_to_be_noised: Float[torch.Tensor, "D L 3"],
        initializer_outputs,
        ref_initializer_outputs: dict[str, Any] | None,
        f_ref: dict[str, Any] | None,
        **_,
    ) -> dict[str, Any]:
        """
        Symmetry-aware diffusion sampling with optional multi-GPU parallel processing.
        
        Same as parent class but applies symmetry constraints during denoising.
        
        Args:
            f: Feature dictionary (must contain sym_transform)
            diffusion_module: The diffusion model
            diffusion_batch_size: Number of parallel diffusion samples (D)
            coord_atom_lvl_to_be_noised: [D, L, 3] coordinates to denoise
            initializer_outputs: Outputs from TokenInitializer
            ref_initializer_outputs: Not used (CFG disabled for symmetry)
            f_ref: Not used
            
        Returns:
            dict with X_L, trajectories, sequence predictions
        """
        # Check for streaming/parallel mode
        streaming_mode = initializer_outputs.get("streaming_mode", False)
        n_parallel = _get_n_parallel()
        gpu_rank, world_size = _get_gpu_rank_and_world_size()
        
        if streaming_mode and world_size > 1:
            ranked_logger.info(
                f"Parallel symmetry diffusion: GPU {gpu_rank}/{world_size}, n_parallel={n_parallel}"
            )
        
        # Motif setup to recenter the motif at every step
        is_motif_atom_with_fixed_coord = f["is_motif_atom_with_fixed_coord"]
        
        # Book-keeping
        noise_schedule = self._construct_inference_noise_schedule(
            device=coord_atom_lvl_to_be_noised.device,
            partial_t=f.get("partial_t", None),
        )

        L = f["ref_element"].shape[0]                      # Number of atoms
        D = diffusion_batch_size                           # Diffusion batch size
        
        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )                                                  # [D, L, 3]

        X_noisy_L_traj = []
        X_denoised_L_traj = []
        sequence_entropy_traj = []
        t_hats = []

        # Symmetrize X_L until the step gamma = gamma_min_sym
        gamma_min_sym_idx = min(
            int(len(noise_schedule) * self.sym_step_frac), len(noise_schedule) - 1
        )
        gamma_min_sym = noise_schedule[gamma_min_sym_idx]

        ranked_logger.info(f"gamma_min_sym: {gamma_min_sym}")
        ranked_logger.info(f"gamma_min: {self.gamma_min}")
        
        # Only show progress bar on rank 0 to avoid duplicate output
        is_main_rank = not dist.is_initialized() or dist.get_rank() == 0
        num_steps = len(noise_schedule) - 1
        pbar = tqdm(
            enumerate(zip(noise_schedule, noise_schedule[1:])),
            total=num_steps,
            desc="Sampling",
            disable=not is_main_rank,
        )
        
        for step_num, (c_t_minus_1, c_t) in pbar:
            # Assert no grads on X_L
            assert not torch.is_grad_enabled(), "Computation graph should not be active"
            assert not X_L.requires_grad, "X_L should not require gradients"
            
            # Ensure all tensors are on the same device as X_L (for multi-GPU)
            # In single-GPU mode, all tensors are already on cuda:0
            # In multi-GPU mode, after parallel processing, X_L ends up on the local device
            # but tensors computed at the start (noise_schedule, gamma_min_sym) remain on cuda:0
            device = X_L.device
            c_t_minus_1 = c_t_minus_1.to(device) if isinstance(c_t_minus_1, torch.Tensor) else torch.tensor(c_t_minus_1, device=device)
            c_t = c_t.to(device) if isinstance(c_t, torch.Tensor) else torch.tensor(c_t, device=device)
            gamma_min_sym = gamma_min_sym.to(device) if isinstance(gamma_min_sym, torch.Tensor) else torch.tensor(gamma_min_sym, device=device)
            coord_atom_lvl_to_be_noised = coord_atom_lvl_to_be_noised.to(device)
            is_motif_atom_with_fixed_coord = is_motif_atom_with_fixed_coord.to(device)

            # Apply a random rotation and translation to the structure
            if self.allow_realignment:
                X_L, R = centre_random_augment_around_motif(
                    X_L,
                    coord_atom_lvl_to_be_noised,
                    is_motif_atom_with_fixed_coord,
                )

            # Update gamma & step scale
            gamma = self.gamma_0 if c_t > self.gamma_min else 0
            step_scale = self.step_scale

            # Compute the value of t_hat
            t_hat = c_t_minus_1 * (gamma + 1)
            
            # Update progress bar with current noise level
            if is_main_rank:
                t_val = t_hat.item() if isinstance(t_hat, torch.Tensor) else t_hat
                pbar.set_postfix({"t": f"{t_val:.3f}"})

            # Noise the coordinates with scaled Gaussian noise
            epsilon_L = (
                self.noise_scale
                * torch.sqrt(torch.square(t_hat) - torch.square(c_t_minus_1))
                * torch.normal(mean=0.0, std=1.0, size=X_L.shape, device=X_L.device)
            )                                              # [D, L, 3]
            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = 0  # No noise for fixed atoms

            # NOTE: no symmetry applied to the noisy structure
            X_noisy_L = X_L + epsilon_L                    # [D, L, 3]

            # ================================================================
            # Denoise the coordinates - handle chunked/streaming mode
            # ================================================================
            tic = time.time()
            
            chunked_embedder = initializer_outputs.get("chunked_pairwise_embedder", None)
            
            if chunked_embedder is not None or streaming_mode:
                # Chunked/streaming mode: explicitly provide P_LL=None
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k not in ("chunked_pairwise_embedder", "streaming_mode")
                }
                
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,                   # [D, L, 3]
                    t=t_hat.tile(D),                       # [D]
                    f=f,
                    P_LL=None,
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    streaming_mode=streaming_mode,         # Pass streaming flag!
                    **other_outputs,
                )
                
                toc = time.time()
                if step_num == 0:
                    ranked_logger.info(
                        f"{'Streaming' if streaming_mode else 'Chunked'} symmetry mode step time: {toc - tic:.2f}s"
                    )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,                   # [D, L, 3]
                    t=t_hat.tile(D),                       # [D]
                    f=f,
                    **initializer_outputs,
                )

            # ================================================================
            # Multi-GPU synchronization
            # ================================================================
            if streaming_mode and world_size > 1:
                dist.barrier()
                if "X_L" in outs:
                    _broadcast_tensor(outs["X_L"], src=0)
                if "sequence_logits_I" in outs and outs["sequence_logits_I"] is not None:
                    _broadcast_tensor(outs["sequence_logits_I"], src=0)

            # Apply symmetry to X_denoised_L
            if "X_L" in outs and c_t > gamma_min_sym:
                outs["X_L"] = self.apply_symmetry_to_X_L(outs["X_L"], f)

            X_denoised_L = outs["X_L"] if "X_L" in outs else outs  # [D, L, 3]

            # Ensure all tensors are on the same device as X_denoised_L (for multi-GPU)
            device = X_denoised_L.device
            X_noisy_L = X_noisy_L.to(device)
            t_hat = t_hat.to(device)
            c_t = c_t.to(device) if isinstance(c_t, torch.Tensor) else torch.tensor(c_t, device=device)

            # Compute the delta between the noisy and denoised coordinates
            delta_L = (X_noisy_L - X_denoised_L) / t_hat   # [D, L, 3]
            d_t = c_t - t_hat

            # NOTE: no classifier-free guidance for symmetry

            # Sequence entropy tracking
            if exists(outs.get("sequence_logits_I")):
                p = torch.softmax(outs["sequence_logits_I"], dim=-1).cpu()  # [D, I, vocab]
                seq_entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)  # [D, I]
                sequence_entropy_traj.append(seq_entropy)

            # Update the coordinates, scaled by the step size
            X_L = X_noisy_L + step_scale * d_t * delta_L   # [D, L, 3]

            # Append the results to the trajectory
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )                                              # [D, L, 3]
            X_noisy_L_traj.append(X_noisy_L_scaled)
            X_denoised_L_traj.append(X_denoised_L)
            t_hats.append(t_hat)

        # ================================================================
        # Post-processing: motif alignment with symmetry
        # ================================================================
        if torch.any(is_motif_atom_with_fixed_coord) and self.allow_realignment:
            # Insert the gt motif at the end
            X_L, R = centre_random_augment_around_motif(
                X_L,
                coord_atom_lvl_to_be_noised,
                is_motif_atom_with_fixed_coord,
                reinsert_motif=self.insert_motif_at_end,
            )

            # Apply symmetry frame shift to X_L
            X_L = self.apply_symmetry_to_X_L(X_L, f)

            # Align prediction to original motif
            X_L = weighted_rigid_align(
                coord_atom_lvl_to_be_noised,
                X_L,
                X_exists_L=is_motif_atom_with_fixed_coord,
            )

        return dict(
            X_L=X_L,                                       # [D, L, 3]
            X_noisy_L_traj=X_noisy_L_traj,                 # list[[D, L, 3]]
            X_denoised_L_traj=X_denoised_L_traj,           # list[[D, L, 3]]
            t_hats=t_hats,                                 # list[Tensor]
            sequence_logits_I=outs.get("sequence_logits_I"),   # [D, I, vocab]
            sequence_indices_I=outs.get("sequence_indices_I"), # [D, I]
            sequence_entropy_traj=sequence_entropy_traj,   # list[[D, I]]
        )


class ConditionalDiffusionSampler:
    """
    Conditional diffusion sampler, chooses at construction time which sampler to use,
    then forwards `sample_diffusion_like_af3` to the chosen sampler.
    If you write a new sampler, you best add it to the registry below
    and inference_sampler.kind in inference_engine config.
    """

    _registry = {
        "default": SampleDiffusionWithMotif,
        "symmetry": SampleDiffusionWithSymmetry,
    }

    def __init__(self, kind="default", **kwargs):
        ranked_logger.info(
            f"Initializing ConditionalDiffusionSampler with kind: {kind}"
        )
        try:
            SamplerCls = self._registry[kind]
            # remove kwargs that the sampler cannot take
            init_args = self.get_class_init_args(SamplerCls)
            kwargs = {k: v for k, v in kwargs.items() if k in init_args}
        except KeyError:
            raise ValueError(
                f"Invalid sampler kind: {kind}, must be one of {list(self._registry.keys())}"
            )
        self.sampler = SamplerCls(**kwargs)

    def sample_diffusion_like_af3(self, **kwargs):
        return self.sampler.sample_diffusion_like_af3(**kwargs)

    def get_class_init_args(self, cls):
        arg_names = []
        if hasattr(cls, "__init__") and callable(cls.__init__):
            for p_cls in cls.__mro__:
                if "__init__" in p_cls.__dict__ and p_cls is not object:
                    signature = inspect.signature(p_cls.__init__)
                    arg_names.extend(
                        [param.name for param in signature.parameters.values()]
                    )
        return arg_names


def centre_random_augment_around_motif(
    X_L: torch.Tensor,  # (D, L, 3) noisy diffused coordinates
    coord_atom_lvl_to_be_noised: torch.Tensor,  # (D, L, 3) original coordinates
    is_motif_atom_with_fixed_coord: torch.Tensor,  # (D, L) indices in original coordinates to be kept constant
    s_trans: float = 1.0,
    center_option: str = "all",
    centering_affects_motif: bool = True,
    reinsert_motif=True,
):
    D, L, _ = X_L.shape

    if reinsert_motif and torch.any(is_motif_atom_with_fixed_coord):
        # ... Align original coordinates to the prediction
        coords_with_gt_aligned = weighted_rigid_align(
            X_L[..., is_motif_atom_with_fixed_coord, :],
            coord_atom_lvl_to_be_noised[..., is_motif_atom_with_fixed_coord, :],
        )

        # ... Insert original coordinates into X_L
        X_L[..., is_motif_atom_with_fixed_coord, :] = coords_with_gt_aligned

    # ... Centering
    if torch.any(is_motif_atom_with_fixed_coord):
        if center_option == "motif":
            center = torch.mean(
                X_L[..., is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of motif atoms
        elif center_option == "diffuse":
            center = torch.mean(
                X_L[..., ~is_motif_atom_with_fixed_coord, :], dim=-2, keepdim=True
            )  # (D, 1, 3) - COM of diffused atoms

        else:
            center = torch.mean(X_L, dim=-2, keepdim=True)
    else:
        center = torch.mean(X_L, dim=-2, keepdim=True)

    # ... Center
    if centering_affects_motif:
        X_L = X_L - center
    else:
        X_L[..., ~is_motif_atom_with_fixed_coord, :] = (
            X_L[..., ~is_motif_atom_with_fixed_coord, :] - center
        )

    # ... Random augmentation
    R = uniform_random_rotation((D,)).to(X_L.device)
    noise = (
        torch.normal(mean=0, std=1, size=(D, 1, 3), device=X_L.device) * s_trans
    )  # (D, 1, 3)
    X_L = rot_vec_mul(R[:, None], X_L) + noise

    return X_L, R
