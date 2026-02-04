"""
Debug context for RFD3 parallel/standard comparison.

This module provides a centralized way to track the current execution mode
and sampling step, making debug logs much easier to grep and compare.

Usage:
    from rfd3.model.debug_context import debug_ctx, debug_log

    # Set context at start of sampling
    debug_ctx.set_mode("PARALLEL")  # or "STANDARD"
    debug_ctx.set_step(42)

    # Log with automatic context prefix
    debug_log("PAIRWISE", "Z_pairs", f"mean={z.mean():.6f}")
    # Output: [PARALLEL-STEP042-PAIRWISE] Z_pairs: mean=0.123456

    # Or get prefix for custom formatting
    prefix = debug_ctx.prefix("ENCODER")
    print(f"{prefix} Custom message")
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Any


class DebugContext:
    """Thread-safe debug context for tracking mode and step."""

    def __init__(self):
        self._mode: str = "UNKNOWN"
        self._step: int = -1
        self._enabled: bool = True

    def set_mode(self, mode: str):
        """Set current execution mode (STANDARD or PARALLEL)."""
        self._mode = mode.upper()

    def set_step(self, step: int):
        """Set current sampling step number."""
        self._step = step

    def enable(self):
        """Enable debug logging."""
        self._enabled = True

    def disable(self):
        """Disable debug logging."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def step(self) -> int:
        return self._step

    def prefix(self, category: str) -> str:
        """
        Get formatted prefix for debug messages.

        Args:
            category: Category name (e.g., "PAIRWISE", "ENCODER", "MODEL")

        Returns:
            Formatted prefix like "[PARALLEL-STEP042-PAIRWISE]"
        """
        step_str = f"STEP{self._step:03d}" if self._step >= 0 else "INIT"
        return f"[{self._mode}-{step_str}-{category}]"

    def auto_detect_mode(self):
        """Auto-detect mode from environment variable."""
        attn_parallel = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
        # Enable parallel mode for any non-zero value (1, 2, 4, etc.)
        if attn_parallel not in ("0", "", "false", "False"):
            self._mode = "PARALLEL"
        else:
            self._mode = "STANDARD"


# Global singleton instance
debug_ctx = DebugContext()


def debug_log(category: str, name: str, message: str, rank: int = 0):
    """
    Log a debug message with automatic context prefix.

    Args:
        category: Category name (e.g., "PAIRWISE", "ENCODER")
        name: Specific item name (e.g., "Z_pairs", "S_I")
        message: The message content
        rank: Only print on this rank (default 0)
    """
    if not debug_ctx.enabled:
        return

    # Only print on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return

    prefix = debug_ctx.prefix(category)
    print(f"{prefix} {name}: {message}", flush=True)


def debug_tensor(category: str, name: str, tensor: torch.Tensor, rank: int = 0):
    """
    Log tensor statistics with automatic context prefix.

    Args:
        category: Category name
        name: Tensor name
        tensor: The tensor to log stats for
        rank: Only print on this rank (default 0)
    """
    if not debug_ctx.enabled:
        return

    if dist.is_initialized() and dist.get_rank() != rank:
        return

    if tensor is None:
        debug_log(category, name, "None", rank)
        return

    with torch.no_grad():
        t = tensor.float()
        stats = {
            "shape": list(tensor.shape),
            "mean": f"{t.mean().item():.6f}",
            "std": f"{t.std().item():.6f}",
            "min": f"{t.min().item():.6f}",
            "max": f"{t.max().item():.6f}",
        }

    prefix = debug_ctx.prefix(category)
    print(f"{prefix} {name}: {stats}", flush=True)


def debug_elements(category: str, name: str, tensor: torch.Tensor, indices: list, rank: int = 0):
    """
    Log specific element values for comparison.

    Args:
        category: Category name
        name: Tensor name
        tensor: The tensor
        indices: List of index tuples to print, e.g., [(0,0), (100,50)]
        rank: Only print on this rank
    """
    if not debug_ctx.enabled:
        return

    if dist.is_initialized() and dist.get_rank() != rank:
        return

    prefix = debug_ctx.prefix(category)
    for idx in indices:
        try:
            if len(idx) == 2:
                val = tensor[idx[0], idx[1], :5].tolist()
                print(f"{prefix} {name}[{idx[0]},{idx[1]},:5]={val}", flush=True)
            elif len(idx) == 3:
                val = tensor[idx[0], idx[1], idx[2], :5].tolist()
                print(f"{prefix} {name}[{idx[0]},{idx[1]},{idx[2]},:5]={val}", flush=True)
        except (IndexError, RuntimeError) as e:
            print(f"{prefix} {name}[{idx}]: ERROR - {e}", flush=True)


def debug_log_all_ranks(category: str, name: str, message: str):
    """
    Log a debug message from ALL ranks (for multi-GPU debugging).

    Args:
        category: Category name (e.g., "PAIRWISE", "ENCODER")
        name: Specific item name
        message: The message content
    """
    if not debug_ctx.enabled:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    prefix = debug_ctx.prefix(category)
    print(f"{prefix} [RANK{rank}/{world_size}] {name}: {message}", flush=True)


def debug_tensor_all_ranks(category: str, name: str, tensor: torch.Tensor):
    """
    Log tensor statistics from ALL ranks (for multi-GPU debugging).

    Args:
        category: Category name
        name: Tensor name
        tensor: The tensor to log stats for
    """
    if not debug_ctx.enabled:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if tensor is None:
        debug_log_all_ranks(category, name, "None")
        return

    with torch.no_grad():
        t = tensor.float()
        stats = {
            "shape": list(tensor.shape),
            "mean": f"{t.mean().item():.6f}",
            "std": f"{t.std().item():.6f}",
            "min": f"{t.min().item():.6f}",
            "max": f"{t.max().item():.6f}",
            "device": str(tensor.device),
        }

    prefix = debug_ctx.prefix(category)
    print(f"{prefix} [RANK{rank}/{world_size}] {name}: {stats}", flush=True)


def verify_tensor_sync(category: str, name: str, tensor: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-5):
    """
    Verify that a tensor is synchronized across all ranks.

    Broadcasts tensor from rank 0 and compares with local tensor on each rank.
    Prints mismatch warnings if tensors differ.

    Args:
        category: Category name
        name: Tensor name
        tensor: Local tensor to verify
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    if not debug_ctx.enabled:
        return

    if not dist.is_initialized():
        return  # Nothing to verify in single-process mode

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Clone tensor for comparison (avoid modifying original)
    local_tensor = tensor.clone()

    # Broadcast reference tensor from rank 0
    ref_tensor = tensor.clone() if rank == 0 else torch.zeros_like(tensor)
    dist.broadcast(ref_tensor, src=0)

    # Compare
    with torch.no_grad():
        diff = (local_tensor.float() - ref_tensor.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_close = torch.allclose(local_tensor.float(), ref_tensor.float(), rtol=rtol, atol=atol)

    prefix = debug_ctx.prefix(category)
    if is_close:
        print(f"{prefix} [RANK{rank}/{world_size}] {name}: SYNC_OK (max_diff={max_diff:.2e})", flush=True)
    else:
        print(f"{prefix} [RANK{rank}/{world_size}] {name}: SYNC_MISMATCH! max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", flush=True)
        # Print some specific values for debugging
        if tensor.numel() > 0:
            local_val = local_tensor.flatten()[:3].tolist()
            ref_val = ref_tensor.flatten()[:3].tolist()
            print(f"{prefix} [RANK{rank}/{world_size}] {name}: local[:3]={local_val}, ref[:3]={ref_val}", flush=True)


def debug_memory(category: str, stage: str):
    """
    Log GPU memory usage from all ranks.

    Args:
        category: Category name (e.g., "ENCODER", "DECODER")
        stage: Stage description (e.g., "before_Z_chunk", "after_distogram")
    """
    if not debug_ctx.enabled:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get memory stats
    allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
    reserved = torch.cuda.memory_reserved(local_rank) / 1024**3
    free_mem, total_mem = torch.cuda.mem_get_info(local_rank)
    free_gb = free_mem / 1024**3
    total_gb = total_mem / 1024**3
    used_gb = total_gb - free_gb

    prefix = debug_ctx.prefix(category)
    print(
        f"{prefix} [RANK{rank}/{world_size}] MEMORY@{stage}: "
        f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, "
        f"global_used={used_gb:.2f}GB, global_free={free_gb:.2f}GB",
        flush=True
    )
