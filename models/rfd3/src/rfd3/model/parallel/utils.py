"""
Shared utilities for parallel (multi-GPU) inference.

Consolidates utility functions previously duplicated across the codebase,
encoders.py, and RFD3_diffusion_module.py into a single module.
"""

import logging
import os
from typing import List, Tuple

import torch
import torch.distributed as dist

from rfd3.model.debug_context import (
    debug_ctx,
    debug_chunking,
    debug_gpu_memory_snapshot,
    timed_all_gather,
)

logger = logging.getLogger(__name__)

# =============================================================================
# EXTRA_CHUNKING: Control sequential loop behavior in parallel mode
# =============================================================================
# Default False = vectorized operation (fast, O(1) scaling, more memory)
# Set True = chunked operation (slower, O(N) scaling, less memory)
#
# In parallel mode, the chunked functions loop over the full I dimension which
# causes O(N) time scaling. Setting EXTRA_CHUNKING=False (default) bypasses
# these loops and uses vectorized operations instead.
# =============================================================================
EXTRA_CHUNKING = os.environ.get("RFD3_EXTRA_CHUNKING", "0") == "1"


def is_parallel_mode() -> bool:
    """
    Check if parallel mode is enabled.

    Env var scheme:
      - RFD3_ATTENTION_PARALLEL=0 or unset -> standard mode (False)
      - RFD3_ATTENTION_PARALLEL=1 or any non-zero value -> parallel mode (True)

    NOTE: design_parallel.py sets this to world_size (e.g., "4" for 4 GPUs),
    so we check for any truthy non-zero value, not just "1".
    """
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    return val not in ("0", "", "false", "False")


def get_world_size() -> int:
    """Get actual GPU count from distributed runtime."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_gpu_rank_and_world_size() -> Tuple[int, int]:
    """
    Get current GPU rank and world size for distributed processing.

    Returns:
        (rank, world_size): Tuple of (GPU rank, total GPUs).
                           Returns (0, 1) if not in distributed mode.
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def compute_chunk_ranges(total: int, n_par: int) -> List[Tuple[int, int]]:
    """
    Compute start/end indices for chunked processing.

    Uses floor division with remainder distributed to early ranks.
    For 13800 tokens / 7 GPUs: ranks 0-2 get 1972, ranks 3-6 get 1971.

    Args:
        total: Total number of elements (I or L)
        n_par: Number of parallel chunks

    Returns:
        List of (start, end) tuples for each chunk
    """
    chunk_size = total // n_par
    remainder = total % n_par

    ranges = []
    for rank in range(n_par):
        if rank < remainder:
            start = rank * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = rank * chunk_size + remainder
            end = start + chunk_size
        if start < total:
            ranges.append((start, end))
    return ranges


def compute_gpu_query_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Compute the query index range for a specific GPU.

    Uses compute_chunk_ranges to ensure consistent chunk distribution
    across all code paths (encoder, transformer, decoder).

    Args:
        total: Total number of queries (I tokens)
        rank: This GPU's rank (0 to world_size-1)
        world_size: Total number of GPUs

    Returns:
        (start_idx, end_idx): Query range [start, end) for this GPU
    """
    chunk_ranges = compute_chunk_ranges(total, world_size)
    if rank >= len(chunk_ranges):
        return total, total  # No tokens for this rank
    return chunk_ranges[rank]


def all_gather_concat(tensor: torch.Tensor, dim: int = 0, total_size: int = None) -> torch.Tensor:
    """
    Gather tensors from all GPUs and concatenate along specified dimension.

    Handles uneven chunk sizes across GPUs. When dividing N elements
    across W GPUs, the last GPU may have fewer elements (N % W != 0). This function
    pads the smaller chunk before gathering and slices to the correct total size.

    Args:
        tensor: Local tensor chunk to gather
        dim: Dimension to concatenate along
        total_size: The expected total size along `dim` after gathering.
                    If provided, the output is sliced to this exact size.
                    This handles uneven chunk sizes (e.g., 13800 / 7 GPUs).

    Returns:
        Full tensor reassembled from all GPUs (size = total_size along dim)
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()

    if world_size == 1:
        return tensor

    # Ensure tensor is on the correct device for this rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    expected_device = torch.device(f"cuda:{local_rank}")
    if tensor.device != expected_device:
        tensor = tensor.to(expected_device)

    # Handle uneven chunk sizes: find max chunk size, pad smaller chunks
    local_size = tensor.shape[dim]
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    timed_all_gather(all_sizes, local_size_tensor,
                     "ALL_GATHER", "parallel.utils.all_gather_concat.sizes",
                     gather_type="all_gather")
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)
    actual_total = sum(all_sizes)

    # Pad tensor if needed (only the last rank typically needs padding)
    if local_size < max_size:
        pad_size = max_size - local_size
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)

    # Ensure tensor is contiguous (required for NCCL)
    tensor = tensor.contiguous()

    # Now all tensors have the same size (max_size), gather them
    if hasattr(dist, 'all_gather_into_tensor'):
        # Memory-efficient: use single flat output tensor
        flat_input = tensor.contiguous().view(-1)
        flat_output = torch.empty(flat_input.numel() * world_size, dtype=tensor.dtype, device=tensor.device)
        timed_all_gather(flat_output, flat_input,
                         "ALL_GATHER", "parallel.utils.all_gather_concat.data",
                         gather_type="all_gather_into_tensor")

        # Reshape: split into world_size chunks, then merge along dim
        chunk_shape = list(tensor.shape)
        reshaped = flat_output.view(world_size, *chunk_shape)

        if dim == 0:
            output_tensor = reshaped.view(-1, *chunk_shape[1:])
        else:
            perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, len(chunk_shape) + 1))
            transposed = reshaped.permute(*perm)
            final_shape = list(tensor.shape)
            final_shape[dim] = final_shape[dim] * world_size
            output_tensor = transposed.contiguous().view(*final_shape)

        del flat_input, flat_output, reshaped
    else:
        # Fallback for older PyTorch
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        timed_all_gather(gathered, tensor,
                         "ALL_GATHER", "parallel.utils.all_gather_concat.data_fallback",
                         gather_type="all_gather")
        output_tensor = torch.cat(gathered, dim=dim)
        del gathered

    # Slice to correct size: remove padding to get exact total_size
    final_size = total_size if total_size is not None else actual_total
    current_size = output_tensor.shape[dim]

    if current_size > final_size:
        indices = [slice(None)] * output_tensor.dim()
        indices[dim] = slice(0, final_size)
        output_tensor = output_tensor[tuple(indices)].contiguous()

    return output_tensor


def all_gather_along_dim(
    tensor: torch.Tensor,
    world_size: int,
    dim: int = 0,
    total_size: int = None,
) -> torch.Tensor:
    """
    Gather tensor chunks from all GPUs and concatenate along specified dimension.

    Handles uneven chunk sizes: with floor-division-with-remainder algorithm,
    early ranks have larger chunks. This function pads smaller chunks before
    gathering and slices to the correct total size.

    Args:
        tensor: Local tensor chunk
        world_size: Total number of GPUs
        dim: Dimension to concatenate along
        total_size: Expected total size (if known, for precise slicing)

    Returns:
        Concatenated tensor from all GPUs [... sum(chunk_sizes) ...]
    """
    if world_size == 1:
        return tensor

    # Gather chunk sizes from all ranks to handle uneven distribution
    local_size = tensor.shape[dim]
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    timed_all_gather(all_sizes, local_size_tensor,
                     "ALL_GATHER", "parallel.utils.all_gather_along_dim.sizes",
                     gather_type="all_gather")
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)
    actual_total = sum(all_sizes)

    # Pad tensor to max_size if this rank has a smaller chunk
    if local_size < max_size:
        pad_size = max_size - local_size
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)

    # Ensure tensor is contiguous (required for NCCL)
    tensor = tensor.contiguous()

    # Gather from all ranks (all tensors now have same size = max_size)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    timed_all_gather(gathered, tensor,
                     "ALL_GATHER", "parallel.utils.all_gather_along_dim.data",
                     gather_type="all_gather")
    result = torch.cat(gathered, dim=dim)

    # Slice to correct total size (remove padding)
    final_size = total_size if total_size is not None else actual_total
    if result.shape[dim] > final_size:
        indices = [slice(None)] * result.dim()
        indices[dim] = slice(0, final_size)
        result = result[tuple(indices)].contiguous()

    return result


def make_all_gather_fn(world_size: int):
    """
    Create a memory-efficient all_gather closure for use in parallel forward passes.

    This replaces the inline closure previously defined in RFD3DiffusionModule.forward().

    Args:
        world_size: Number of GPUs

    Returns:
        Callable that gathers and concatenates tensors from all GPUs
    """
    def _all_gather_fn(tensor, ws, dim):
        tensor = tensor.contiguous()

        if hasattr(dist, 'all_gather_into_tensor'):
            flat_input = tensor.view(-1)
            flat_output = torch.empty(flat_input.numel() * ws, dtype=tensor.dtype, device=tensor.device)
            timed_all_gather(flat_output, flat_input,
                             "ALL_GATHER", "parallel.utils.make_all_gather_fn.data",
                             gather_type="all_gather_into_tensor")

            chunk_shape = list(tensor.shape)
            reshaped = flat_output.view(ws, *chunk_shape)

            if dim == 0:
                result = reshaped.view(-1, *chunk_shape[1:])
            else:
                perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, len(chunk_shape) + 1))
                transposed = reshaped.permute(*perm)
                final_shape = list(tensor.shape)
                final_shape[dim] = final_shape[dim] * ws
                result = transposed.contiguous().view(*final_shape)

            del flat_input, flat_output, reshaped
            return result
        else:
            gathered = [torch.zeros_like(tensor) for _ in range(ws)]
            timed_all_gather(gathered, tensor,
                             "ALL_GATHER", "parallel.utils.make_all_gather_fn.data_fallback",
                             gather_type="all_gather")
            result = torch.cat(gathered, dim=dim)
            del gathered
            return result

    return _all_gather_fn


def z_transition_chunked(Z: torch.Tensor, transition_fn, key_chunk: int = 512, extra_chunking: bool = None) -> torch.Tensor:
    """
    Apply z_transition, optionally with chunking for memory efficiency.

    By default (extra_chunking=False), uses a single vectorized operation for O(1) scaling.
    If extra_chunking=True, processes in sub-chunks to reduce peak memory (O(N) scaling).

    Args:
        Z: [I_par, I, c_z] or [B, I_par, I, c_z]
        transition_fn: z_transition module
        key_chunk: chunk size along key dimension (only used if extra_chunking=True)
        extra_chunking: If True, use chunked processing. If None, uses EXTRA_CHUNKING env var.

    Returns:
        Z + transition_fn(Z)
    """
    use_chunking = extra_chunking if extra_chunking is not None else EXTRA_CHUNKING

    # DEFAULT: Vectorized operation (fast, O(1) scaling)
    if not use_chunking or Z.dim() not in [3, 4]:
        if debug_ctx.memory_enabled and Z.dim() in [3, 4]:
            I = Z.shape[1] if Z.dim() == 3 else Z.shape[2]
            z_mem_gb = Z.numel() * Z.element_size() / 1e9
            swiglu_peak_gb = Z.numel() * 4 * Z.element_size() / 1e9
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                free_mem, total_mem = torch.cuda.mem_get_info()
                free_gb = free_mem / 1e9
                debug_chunking(f"z_transition_chunked: VECTORIZED, I={I}")
                debug_chunking(f"  Z shape={list(Z.shape)}, Z_mem={z_mem_gb:.2f}GB")
                debug_chunking(f"  SwiGLU 4x expansion will need ~{swiglu_peak_gb:.2f}GB")
                debug_chunking(f"  GPU: allocated={allocated:.2f}GB, free={free_gb:.2f}GB")
                if swiglu_peak_gb > free_gb:
                    debug_chunking(f"  WARNING: SwiGLU peak ({swiglu_peak_gb:.2f}GB) > free ({free_gb:.2f}GB) - OOM likely!")
                    debug_gpu_memory_snapshot("OOM_DEBUG", f"before_z_transition_I={I}", min_size_mb=50.0)
            else:
                debug_chunking(f"z_transition_chunked: VECTORIZED, I={I}, Z_mem={z_mem_gb:.2f}GB, SwiGLU_peak={swiglu_peak_gb:.2f}GB")
        return Z + transition_fn(Z)

    # OPTIONAL: Chunked operation (slow, O(N) scaling, less memory)
    if Z.dim() == 3:
        I = Z.shape[1]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"z_transition_chunked(3D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, k_start:k_end, :]
            out_chunks.append(Z_sub + transition_fn(Z_sub))
        result = torch.cat(out_chunks, dim=1)
        del out_chunks
        return result
    else:  # Z.dim() == 4
        I = Z.shape[2]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"z_transition_chunked(4D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, :, k_start:k_end, :]
            out_chunks.append(Z_sub + transition_fn(Z_sub))
        result = torch.cat(out_chunks, dim=2)
        del out_chunks
        return result


def process_z_chunked(Z: torch.Tensor, process_fn, key_chunk: int = 512, extra_chunking: bool = None) -> torch.Tensor:
    """
    Apply process_z (Linear), optionally with chunking for memory efficiency.

    By default (extra_chunking=False), uses a single vectorized operation for O(1) scaling.
    If extra_chunking=True, processes in sub-chunks to reduce peak memory (O(N) scaling).

    Args:
        Z: [I_par, I, c_in] or [B, I_par, I, c_in]
        process_fn: process_z module (typically nn.Sequential with RMSNorm + Linear)
        key_chunk: chunk size along key dimension (only used if extra_chunking=True)
        extra_chunking: If True, use chunked processing. If None, uses EXTRA_CHUNKING env var.

    Returns:
        process_fn(Z)
    """
    use_chunking = extra_chunking if extra_chunking is not None else EXTRA_CHUNKING

    # DEFAULT: Vectorized operation (fast, O(1) scaling)
    if not use_chunking or Z.dim() not in [3, 4]:
        if debug_ctx.memory_enabled and Z.dim() in [3, 4]:
            I = Z.shape[1] if Z.dim() == 3 else Z.shape[2]
            z_mem_gb = Z.numel() * Z.element_size() / 1e9
            peak_gb = z_mem_gb * 2
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                free_mem, total_mem = torch.cuda.mem_get_info()
                free_gb = free_mem / 1e9
                debug_chunking(f"process_z_chunked: VECTORIZED, I={I}")
                debug_chunking(f"  Z shape={list(Z.shape)}, Z_mem={z_mem_gb:.2f}GB")
                debug_chunking(f"  Estimated peak ~{peak_gb:.2f}GB")
                debug_chunking(f"  GPU: allocated={allocated:.2f}GB, free={free_gb:.2f}GB")
            else:
                debug_chunking(f"process_z_chunked: VECTORIZED, I={I}, Z_mem={z_mem_gb:.2f}GB")
        return process_fn(Z)

    # OPTIONAL: Chunked operation (slow, O(N) scaling, less memory)
    if Z.dim() == 3:
        I = Z.shape[1]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"process_z_chunked(3D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, k_start:k_end, :]
            out_chunks.append(process_fn(Z_sub))
            del Z_sub
        result = torch.cat(out_chunks, dim=1)
        del out_chunks
        return result
    else:  # Z.dim() == 4
        I = Z.shape[2]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"process_z_chunked(4D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, :, k_start:k_end, :]
            out_chunks.append(process_fn(Z_sub))
            del Z_sub
        result = torch.cat(out_chunks, dim=2)
        del out_chunks
        return result
