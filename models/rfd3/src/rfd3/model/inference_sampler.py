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
from rfd3.model.debug_context import debug_ctx, debug_log_all_ranks, debug_time, debug_time_log, TimingInstrument
from rfd3.model.parallel.utils import compute_chunk_ranges

ranked_logger = RankedLogger(__name__, rank_zero_only=True)


# =============================================================================
# Multi-GPU Parallel Inference Utilities
# =============================================================================

def _is_parallel_mode() -> bool:
    """
    Check if streaming/parallel attention mode is enabled.

    Env var scheme:
      - RFD3_ATTENTION_PARALLEL=0 or unset → standard mode (False)
      - RFD3_ATTENTION_PARALLEL=1 or any non-zero value → parallel mode (True)
    """
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    return val not in ("0", "", "false", "False")


def _get_gpu_rank_and_world_size() -> Tuple[int, int]:
    """Get current GPU rank and total world size for distributed processing."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    else:
        return 0, 1


def _compute_gpu_query_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Compute the query index range for a specific GPU.

    Uses compute_chunk_ranges to ensure consistent chunk distribution
    across all code paths (encoder, transformer, decoder).

    Args:
        total: Total number of queries (I or L)
        rank: This GPU's rank (0 to world_size-1)
        world_size: Total number of GPUs

    Returns:
        (start_idx, end_idx): Query range for this GPU [start, end)
    """
    chunk_ranges = compute_chunk_ranges(total, world_size)
    if rank >= len(chunk_ranges):
        return total, total  # No tokens for this rank
    return chunk_ranges[rank]


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


def _all_gather_concat(tensor: torch.Tensor, dim: int = 0, total_size: int = None) -> torch.Tensor:
    """
    Gather tensors from all GPUs and concatenate along specified dimension.

    Handles uneven chunk sizes: when dividing N elements across W GPUs,
    the last GPU may have fewer elements. This function pads smaller chunks
    before gathering and slices to the correct total size.

    Args:
        tensor: Local tensor to gather
        dim: Dimension to concatenate along
        total_size: Expected total size after gathering (handles uneven chunks)

    Returns:
        Concatenated tensor from all GPUs
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Handle uneven chunk sizes by finding max size and padding
    local_size = tensor.shape[dim]
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size_tensor)
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)
    actual_total = sum(all_sizes)

    # Pad tensor if needed
    if local_size < max_size:
        pad_size = max_size - local_size
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)

    # Gather all tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    result = torch.cat(gathered, dim=dim)

    # Slice to correct size
    final_size = total_size if total_size is not None else actual_total
    if result.shape[dim] > final_size:
        indices = [slice(None)] * result.dim()
        indices[dim] = slice(0, final_size)
        result = result[tuple(indices)].contiguous()

    return result


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
        """Generates the initial noisy structure for diffusion sampling.
        
        Args:
            c0: Initial noise scale from the noise schedule
            D: Diffusion batch size
            L: Number of atoms
            coord_atom_lvl_to_be_noised: [L, 3] coordinates to be noised
            is_motif_atom_with_fixed_coord: Boolean mask for motif atoms with fixed coords

        Returns:
            X_L: Initial noisy coordinates [D, L, 3]
        """
        noise = c0 * torch.normal(mean=0.0, std=1.0, size=(D, L, 3), device=c0.device)
        # NOTE: Initial noise is NOT broadcast here because tensors may not be on
        # correct local devices yet. Initial noise synchronization relies on
        # set_seed() being called with the same seed on all ranks. The per-step
        # epsilon_L noise IS broadcast in the diffusion loop where device placement
        # is correct.
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
        parallel_mode = initializer_outputs.get("parallel_mode", False)
        gpu_rank, world_size = _get_gpu_rank_and_world_size()

        # RNG diagnostic: log state at start of diffusion sampling
        # _rng_diagnostic("SampleDiffusionWithMotif.sample_start")

        if parallel_mode and world_size > 1:
            ranked_logger.info(
                f"Parallel diffusion sampling: GPU {gpu_rank}/{world_size}"
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

        # Initialize comprehensive timing instrumentation
        timing = TimingInstrument(
            enabled=debug_ctx.time_enabled or debug_ctx.memory_enabled,
            rank=gpu_rank,
            world_size=world_size,
            num_steps=len(noise_schedule) - 1
        )

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
            # Update debug context with current step
            debug_ctx.set_step(step_num)

            # Mark start of timing for this step
            timing.mark_step_start(step_num)

            # =======================================================================
            # MEMORY OPTIMIZATION: Clean memory at start of each step
            # Force garbage collection and CUDA cache clear to ensure consistent
            # memory state. This prevents fragmentation from accumulating.
            # =======================================================================
            if parallel_mode and step_num > 0:
                import gc
                gc.collect()

                timing.mark_sync_point("cuda_sync")
                with timing.time_operation("cuda_sync", use_cuda_events=False):
                    torch.cuda.synchronize()

                with timing.time_operation("empty_cache", use_cuda_events=False):
                    torch.cuda.empty_cache()

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

            # CRITICAL: In multi-GPU mode, broadcast noise from rank 0 to ensure
            # all GPUs use identical noise. Otherwise each GPU generates different
            # random noise, causing X_noisy_L to diverge across GPUs.
            if parallel_mode and world_size > 1:
                _broadcast_tensor(epsilon_L, src=0)

            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = 0  # No noise for fixed atoms
            X_noisy_L = X_L + epsilon_L                    # [D, L, 3]

            # ================================================================
            # Denoise the coordinates - handle chunked/streaming mode
            # ================================================================
            # Prepare common arguments
            chunked_embedder = initializer_outputs.get("chunked_pairwise_embedder", None)

            with debug_time("SAMPLER", f"diffusion_step_{step_num}"):
                with timing.time_operation("model_forward", use_cuda_events=True):
                    if chunked_embedder is not None or parallel_mode:
                        # Chunked/streaming mode: explicitly provide P_LL=None
                        other_outputs = {
                            k: v
                            for k, v in initializer_outputs.items()
                            if k not in ("chunked_pairwise_embedder", "parallel_mode")
                        }

                        outs = diffusion_module(
                            X_noisy_L=X_noisy_L,                   # [D, L, 3]
                            t=t_hat.tile(D),                       # [D]
                            f=f,
                            P_LL=None,                             # Not used in chunked/streaming mode
                            chunked_pairwise_embedder=chunked_embedder,
                            initializer_outputs=other_outputs,
                            parallel_mode=parallel_mode,         # Pass streaming flag!
                            **other_outputs,
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
            if parallel_mode and world_size > 1:
                # Barrier to ensure all GPUs have finished this step
                timing.mark_sync_point("barrier")
                with timing.time_operation("barrier", use_cuda_events=False):
                    dist.barrier()

                # Broadcast X_L from rank 0 to ensure exact consistency
                # (should be identical, but floating point differences can accumulate)
                _broadcast_tensor(X_denoised_L, src=0)

                # Sync sequence predictions
                if "sequence_logits_I" in outs and outs["sequence_logits_I"] is not None:
                    _broadcast_tensor(outs["sequence_logits_I"], src=0)

            # Compute the delta (will be updated by CFG if enabled)
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
                if chunked_embedder is not None or parallel_mode:
                    ref_other = {
                        k: v
                        for k, v in ref_initializer_outputs.items()
                        if k not in ("chunked_pairwise_embedder", "parallel_mode")
                    }
                    outs_ref = diffusion_module(
                        X_noisy_L=X_noisy_L_stripped,
                        t=t_hat.tile(D),
                        f=f_ref,
                        P_LL=None,
                        chunked_pairwise_embedder=ref_initializer_outputs.get(
                            "chunked_pairwise_embedder"
                        ),
                        parallel_mode=ref_initializer_outputs.get("parallel_mode", False),
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
                if parallel_mode and world_size > 1:
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
            with timing.time_operation("ode_update", use_cuda_events=True):
                X_L = X_noisy_L + step_scale * d_t * delta_L   # [D, L, 3]

            # Append the results to the trajectory (for visualization of the diffusion process)
            X_noisy_L_scaled = (
                self.sigma_data * X_noisy_L / torch.sqrt(t_hat**2 + self.sigma_data**2)
            )                                              # [D, L, 3]
            # =======================================================================
            # MEMORY OPTIMIZATION: Move trajectory tensors to CPU to prevent GPU
            # memory accumulation. Over 99 steps, keeping these on GPU causes
            # ~300MB+ accumulation that contributes to fragmentation and OOM.
            # =======================================================================
            if parallel_mode:
                X_noisy_L_traj.append(X_noisy_L_scaled.cpu())
                X_denoised_L_traj.append(X_denoised_L.cpu())
                t_hats.append(t_hat.cpu())
            else:
                X_noisy_L_traj.append(X_noisy_L_scaled)
                X_denoised_L_traj.append(X_denoised_L)
                t_hats.append(t_hat)

            # =======================================================================
            # MEMORY OPTIMIZATION: Aggressive cleanup between diffusion steps
            # In multi-GPU streaming mode, memory fragmentation can cause OOM even
            # when total free memory is sufficient. The key is to:
            # 1. Delete all temporary tensors explicitly
            # 2. Force Python GC to release references
            # 3. Synchronize CUDA to complete all pending ops
            # 4. Clear CUDA cache to defragment memory
            # This ensures that if step N succeeds, step N+1 will too.
            # NOTE: Do NOT delete 'outs' - it's used after the loop for return values
            # =======================================================================
            if parallel_mode:
                import gc
                # Delete temporary tensors from this step
                del epsilon_L, X_noisy_L_scaled, delta_L, X_denoised_L
                # Force Python garbage collection to release any dangling references
                gc.collect()

                # Synchronize CUDA to ensure all operations are complete
                timing.mark_sync_point("cuda_sync")
                with timing.time_operation("cuda_sync", use_cuda_events=False):
                    torch.cuda.synchronize()

                # Clear CUDA cache to defragment and release unused memory
                with timing.time_operation("empty_cache", use_cuda_events=False):
                    torch.cuda.empty_cache()

            # Mark end of timing for this step
            timing.mark_step_end()

        # ================================================================
        # Print timing summary at end of diffusion loop
        # ================================================================
        debug_ctx.print_timing_summary()

        # ================================================================
        # Comprehensive timing analysis
        # ================================================================
        timing.finalize_all_steps()
        timing.print_comprehensive_report()

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

        # Print all_gather synchronization summary (if timing enabled)
        debug_ctx.print_allgather_summary()
        debug_ctx.print_allgather_per_step_summary(num_steps=self.num_timesteps)

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
        parallel_mode = initializer_outputs.get("parallel_mode", False)
        gpu_rank, world_size = _get_gpu_rank_and_world_size()

        # RNG diagnostic: log state at start of symmetry diffusion sampling
        # _rng_diagnostic("SampleDiffusionWithSymmetry.sample_start")

        if parallel_mode and world_size > 1:
            ranked_logger.info(
                f"Parallel symmetry diffusion: GPU {gpu_rank}/{world_size}"
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

        # DIAGNOSTIC: Log input shapes (only if stats logging enabled)
        if debug_ctx.stats_enabled:
            attn_parallel = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
            print(f"[DIAG] SymmetryInferenceSampler START: L={L}, D={D}, attn_parallel={attn_parallel}", flush=True)
            print(f"[DIAG]   coord_atom_lvl_to_be_noised.shape={coord_atom_lvl_to_be_noised.shape}", flush=True)
            print(f"[DIAG]   f['is_ca'].sum()={f['is_ca'].sum().item()} (number of tokens I)", flush=True)

        # RNG diagnostic: log state before initial structure generation
        # _rng_diagnostic("SampleDiffusionWithSymmetry.before_initial_noise")

        X_L = self._get_initial_structure(
            c0=noise_schedule[0],
            D=D,
            L=L,
            coord_atom_lvl_to_be_noised=coord_atom_lvl_to_be_noised.clone(),
            is_motif_atom_with_fixed_coord=is_motif_atom_with_fixed_coord,
        )                                                  # [D, L, 3]

        # CRITICAL: In multi-GPU mode, broadcast initial X_L from rank 0 to all ranks
        # Each rank generates different random noise, so we must synchronize the initial structure
        if parallel_mode and world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", gpu_rank))
            local_device = torch.device(f"cuda:{local_rank}")
            # Move X_L to local device before broadcast (NCCL requires tensors on local GPU)
            X_L = X_L.to(local_device)
            debug_log_all_ranks("SAMPLER", "BEFORE_INIT_BROADCAST", f"X_L.shape={X_L.shape}, device={X_L.device}, X_L[0,0,:3]={X_L[0,0,:3].tolist()}")
            dist.broadcast(X_L, src=0)
            debug_log_all_ranks("SAMPLER", "AFTER_INIT_BROADCAST", f"X_L[0,0,:3]={X_L[0,0,:3].tolist()}")

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
            # Update debug context with current step
            debug_ctx.set_step(step_num)

            # =======================================================================
            # MEMORY OPTIMIZATION: Clean memory at start of each step
            # Force garbage collection and CUDA cache clear to ensure consistent
            # memory state. This prevents fragmentation from accumulating.
            # =======================================================================
            if parallel_mode and step_num > 0:
                import gc
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # CRITICAL: In multi-GPU mode, ensure X_L is on the correct LOCAL device
            # X_L is initially created on cuda:0 for all ranks, but NCCL requires each rank
            # to have tensors on its own device (rank 0 → cuda:0, rank 1 → cuda:1)
            if parallel_mode and world_size > 1:
                local_rank = int(os.environ.get("LOCAL_RANK", gpu_rank))
                local_device = torch.device(f"cuda:{local_rank}")
                if X_L.device != local_device:
                    X_L = X_L.to(local_device)

            # DEBUG: Track where each rank is at start of loop
            debug_log_all_ranks("SAMPLER", "LOOP_START", f"step={step_num}, device={X_L.device}")

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

            # DEBUG: Track epsilon generation
            debug_log_all_ranks("SAMPLER", "EPSILON_GENERATED", f"device={epsilon_L.device}, mean={epsilon_L.mean().item():.6f}")

            # CRITICAL: In multi-GPU mode, broadcast noise from rank 0 to ensure
            # all GPUs use identical noise. Otherwise each GPU generates different
            # random noise, causing X_noisy_L to diverge across GPUs.
            if parallel_mode and world_size > 1:
                debug_log_all_ranks("SAMPLER", "BEFORE_BROADCAST", f"epsilon_L.device={epsilon_L.device}")
                _broadcast_tensor(epsilon_L, src=0)
                debug_log_all_ranks("SAMPLER", "AFTER_BROADCAST", f"epsilon_L.mean={epsilon_L.mean().item():.6f}")

            epsilon_L[..., is_motif_atom_with_fixed_coord, :] = 0  # No noise for fixed atoms

            # NOTE: no symmetry applied to the noisy structure
            X_noisy_L = X_L + epsilon_L                    # [D, L, 3]

            # DEBUG: Track X_noisy_L
            debug_log_all_ranks("SAMPLER", "X_NOISY_READY", f"device={X_noisy_L.device}, mean={X_noisy_L.mean().item():.6f}")

            # ================================================================
            # Denoise the coordinates - handle chunked/streaming mode
            # ================================================================
            tic = time.time()

            chunked_embedder = initializer_outputs.get("chunked_pairwise_embedder", None)

            # DEBUG: Track before model forward
            debug_log_all_ranks("SAMPLER", "BEFORE_MODEL_FORWARD", f"streaming={parallel_mode}, chunked={chunked_embedder is not None}")

            if chunked_embedder is not None or parallel_mode:
                # Chunked/streaming mode: explicitly provide P_LL=None
                other_outputs = {
                    k: v
                    for k, v in initializer_outputs.items()
                    if k not in ("chunked_pairwise_embedder", "parallel_mode")
                }
                
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,                   # [D, L, 3]
                    t=t_hat.tile(D),                       # [D]
                    f=f,
                    P_LL=None,
                    chunked_pairwise_embedder=chunked_embedder,
                    initializer_outputs=other_outputs,
                    parallel_mode=parallel_mode,         # Pass streaming flag!
                    **other_outputs,
                )
                
                toc = time.time()
                if step_num == 0:
                    ranked_logger.info(
                        f"{'Streaming' if parallel_mode else 'Chunked'} symmetry mode step time: {toc - tic:.2f}s"
                    )
            else:
                # Standard mode: P_LL is included in initializer_outputs
                outs = diffusion_module(
                    X_noisy_L=X_noisy_L,                   # [D, L, 3]
                    t=t_hat.tile(D),                       # [D]
                    f=f,
                    **initializer_outputs,
                )

            # DEBUG: Track after model forward
            debug_log_all_ranks("SAMPLER", "AFTER_MODEL_FORWARD", f"X_L_shape={outs.get('X_L', 'N/A')}")

            # ================================================================
            # Multi-GPU synchronization
            # ================================================================
            if parallel_mode and world_size > 1:
                debug_log_all_ranks("SAMPLER", "BEFORE_BARRIER", "entering barrier")
                dist.barrier()
                debug_log_all_ranks("SAMPLER", "AFTER_BARRIER", "barrier complete")
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
            # =======================================================================
            # MEMORY OPTIMIZATION: Move trajectory tensors to CPU to prevent GPU
            # memory accumulation. Over 99 steps, keeping these on GPU causes
            # ~300MB+ accumulation that contributes to fragmentation and OOM.
            # =======================================================================
            if parallel_mode:
                X_noisy_L_traj.append(X_noisy_L_scaled.cpu())
                X_denoised_L_traj.append(X_denoised_L.cpu())
                t_hats.append(t_hat.cpu())
            else:
                X_noisy_L_traj.append(X_noisy_L_scaled)
                X_denoised_L_traj.append(X_denoised_L)
                t_hats.append(t_hat)

            # =======================================================================
            # MEMORY OPTIMIZATION: Aggressive cleanup between diffusion steps
            # In multi-GPU streaming mode, memory fragmentation can cause OOM even
            # when total free memory is sufficient. The key is to:
            # 1. Delete all temporary tensors explicitly
            # 2. Force Python GC to release references
            # 3. Synchronize CUDA to complete all pending ops
            # 4. Clear CUDA cache to defragment memory
            # This ensures that if step N succeeds, step N+1 will too.
            # NOTE: Do NOT delete 'outs' - it's used after the loop for return values
            # =======================================================================
            if parallel_mode:
                import gc
                # Delete temporary tensors from this step
                del epsilon_L, X_noisy_L_scaled, delta_L, X_denoised_L
                # Force Python garbage collection to release any dangling references
                gc.collect()
                # Synchronize CUDA to ensure all operations are complete
                torch.cuda.synchronize()
                # Clear CUDA cache to defragment and release unused memory
                torch.cuda.empty_cache()

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

        # DIAGNOSTIC: Log output shapes (only if stats logging enabled)
        if debug_ctx.stats_enabled:
            print(f"[DIAG] SymmetryInferenceSampler END: X_L.shape={X_L.shape}", flush=True)

        # Print all_gather synchronization summary (if timing enabled)
        debug_ctx.print_allgather_summary()
        debug_ctx.print_allgather_per_step_summary(num_steps=self.num_timesteps)

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
