"""
Debug context for RFD3 parallel/standard comparison.

This module provides a centralized way to track the current execution mode
and sampling step, making debug logs much easier to grep and compare.

Verbosity Levels (set via environment variables or configure() method):
    - RFD3_VERBOSE=1          : Enable ALL logging
    - RFD3_VERBOSE_STATS=1    : Tensor statistics ([DEBUG-STREAMING], [PARALLEL-...], etc.)
    - RFD3_VERBOSE_MEMORY=1   : Memory and chunking logs ([CHUNKING], [MEMORY], etc.)
    - RFD3_VERBOSE_TIME=1     : Timing logs ([TIME], etc.)

Usage:
    from rfd3.model.debug_context import debug_ctx, debug_log, debug_time

    # Configure from config dict (e.g., from YAML)
    debug_ctx.configure({
        'verbose': True,           # Enable all
        'verbose_stats': True,     # Or enable specific categories
        'verbose_memory': True,
        'verbose_time': True,
    })

    # Set context at start of sampling
    debug_ctx.set_mode("PARALLEL")  # or "STANDARD"
    debug_ctx.set_step(42)

    # Log with automatic context prefix
    debug_log("PAIRWISE", "Z_pairs", f"mean={z.mean():.6f}")
    # Output: [PARALLEL-STEP042-PAIRWISE] Z_pairs: mean=0.123456

    # Time a code block
    with debug_time("ENCODER", "pairformer_block"):
        # ... code to time ...
"""

import os
import time
import torch
import torch.distributed as dist
from typing import Optional, Any, Dict
from contextlib import contextmanager
from functools import wraps


class DebugContext:
    """Thread-safe debug context for tracking mode, step, and verbosity levels."""

    def __init__(self):
        self._mode: str = "UNKNOWN"
        self._step: int = -1

        # Verbosity flags - default to environment variables
        self._verbose: bool = os.environ.get("RFD3_VERBOSE", "0") == "1"
        self._verbose_stats: bool = os.environ.get("RFD3_VERBOSE_STATS", "0") == "1"
        self._verbose_memory: bool = os.environ.get("RFD3_VERBOSE_MEMORY", "0") == "1"
        self._verbose_time: bool = os.environ.get("RFD3_VERBOSE_TIME", "0") == "1"

        # Legacy enabled flag (for backwards compatibility)
        self._enabled: bool = os.environ.get("RFD3_DEBUG", "0") == "1"

        # Timing accumulator for summary
        self._timing_data: Dict[str, list] = {}

        # All-gather timing accumulator for synchronization overhead analysis
        self._allgather_timings: Dict[str, list] = {}
        self._allgather_counts: Dict[str, int] = {}

    def configure(self, config: Dict[str, Any]):
        """
        Configure debug context from a config dictionary (e.g., from YAML).

        Args:
            config: Dictionary with keys like 'verbose', 'verbose_stats', etc.
        """
        if config.get('verbose', False):
            self._verbose = True
            self._verbose_stats = True
            self._verbose_memory = True
            self._verbose_time = True
            self._enabled = True
        else:
            self._verbose_stats = config.get('verbose_stats', self._verbose_stats)
            self._verbose_memory = config.get('verbose_memory', self._verbose_memory)
            self._verbose_time = config.get('verbose_time', self._verbose_time)
            # Enable legacy flag if any verbosity is on
            self._enabled = self._verbose_stats or self._verbose_memory or self._verbose_time

    def set_mode(self, mode: str):
        """Set current execution mode (STANDARD or PARALLEL)."""
        self._mode = mode.upper()

    def set_step(self, step: int):
        """Set current sampling step number."""
        self._step = step

    def enable(self):
        """Enable all debug logging (legacy method)."""
        self._enabled = True
        self._verbose = True
        self._verbose_stats = True
        self._verbose_memory = True
        self._verbose_time = True

    def disable(self):
        """Disable all debug logging."""
        self._enabled = False
        self._verbose = False
        self._verbose_stats = False
        self._verbose_memory = False
        self._verbose_time = False

    @property
    def enabled(self) -> bool:
        """Legacy property for backwards compatibility."""
        return self._enabled or self._verbose or self._verbose_stats or self._verbose_memory or self._verbose_time

    @property
    def stats_enabled(self) -> bool:
        """Check if stats logging is enabled."""
        return self._verbose or self._verbose_stats

    @property
    def memory_enabled(self) -> bool:
        """Check if memory logging is enabled."""
        return self._verbose or self._verbose_memory

    @property
    def time_enabled(self) -> bool:
        """Check if timing logging is enabled."""
        return self._verbose or self._verbose_time

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

    def record_timing(self, category: str, stage: str, duration_ms: float):
        """Record timing data for later summary."""
        key = f"{category}/{stage}"
        if key not in self._timing_data:
            self._timing_data[key] = []
        self._timing_data[key].append(duration_ms)

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all recorded timings."""
        summary = {}
        for key, times in self._timing_data.items():
            if times:
                summary[key] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                }
        return summary

    def print_timing_summary(self):
        """Print a summary of all recorded timings."""
        if not self.time_enabled:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        summary = self.get_timing_summary()
        if not summary:
            return

        print(f"\n{'='*80}", flush=True)
        print(f"[TIMING SUMMARY] Step {self._step}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  {'Stage':<50} {'Count':<8} {'Total(ms)':<12} {'Avg(ms)':<12}", flush=True)
        print(f"  {'-'*50} {'-'*8} {'-'*12} {'-'*12}", flush=True)

        # Sort by total time (descending)
        for key, stats in sorted(summary.items(), key=lambda x: -x[1]['total_ms']):
            print(f"  {key:<50} {stats['count']:<8} {stats['total_ms']:<12.1f} {stats['avg_ms']:<12.2f}", flush=True)

        print(f"{'='*80}\n", flush=True)

    def reset_timing(self):
        """Reset timing data for new step."""
        self._timing_data.clear()

    def record_allgather_time(self, category: str, name: str, elapsed_ms: float):
        """Record timing for an all_gather operation."""
        key = f"{category}/{name}"
        if key not in self._allgather_timings:
            self._allgather_timings[key] = []
            self._allgather_counts[key] = 0
        self._allgather_timings[key].append(elapsed_ms)
        self._allgather_counts[key] += 1

    def print_allgather_summary(self):
        """Print summary of all all_gather operations."""
        if not (self.time_enabled or self.memory_enabled):
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        if not self._allgather_timings:
            return

        print("\n" + "="*100, flush=True)
        print(f"ALL_GATHER SYNCHRONIZATION SUMMARY", flush=True)
        print("="*100, flush=True)

        # Sort by total time (descending)
        items = []
        for key, timings in self._allgather_timings.items():
            total_ms = sum(timings)
            count = len(timings)
            avg_ms = total_ms / count if count > 0 else 0
            items.append((key, count, total_ms, avg_ms, min(timings), max(timings)))

        items.sort(key=lambda x: x[2], reverse=True)  # Sort by total time

        print(f"{'Operation':<60} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}", flush=True)
        print("-" * 100, flush=True)

        grand_total = 0
        grand_count = 0
        for key, count, total, avg, min_t, max_t in items:
            print(f"{key:<60} {count:>8} {total:>12.2f} {avg:>10.2f} {min_t:>10.2f} {max_t:>10.2f}", flush=True)
            grand_total += total
            grand_count += count

        print("-" * 100, flush=True)
        print(f"{'TOTAL':<60} {grand_count:>8} {grand_total:>12.2f} {'':>10} {'':>10} {'':>10}", flush=True)
        print("="*100, flush=True)
        print(f"\nTotal time in all_gather: {grand_total/1000:.2f} seconds", flush=True)
        print(f"Average per all_gather: {grand_total/grand_count:.2f} ms" if grand_count > 0 else "No all_gather calls recorded", flush=True)
        print("="*100 + "\n", flush=True)

    def print_allgather_per_step_summary(self, num_steps: int = None):
        """Print per-step breakdown of all_gather overhead."""
        if not (self.time_enabled or self.memory_enabled):
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return

        if not self._allgather_timings:
            return

        if num_steps is None:
            num_steps = self.step + 1 if self.step is not None and self.step >= 0 else 1

        # Group by operation
        step_totals = {}
        for key, timings in self._allgather_timings.items():
            # Estimate step distribution (assumes uniform distribution)
            per_step = sum(timings) / num_steps
            step_totals[key] = per_step

        print("\n" + "="*80, flush=True)
        print(f"ALL_GATHER OVERHEAD PER DIFFUSION STEP (estimated)", flush=True)
        print("="*80, flush=True)
        print(f"{'Operation':<50} {'Per-Step(ms)':>12} {'% of Total':>15}", flush=True)
        print("-" * 80, flush=True)

        total_per_step = sum(step_totals.values())
        for key, per_step_ms in sorted(step_totals.items(), key=lambda x: x[1], reverse=True):
            pct = (per_step_ms / total_per_step * 100) if total_per_step > 0 else 0
            print(f"{key:<50} {per_step_ms:>12.2f} {pct:>14.1f}%", flush=True)

        print("-" * 80, flush=True)
        print(f"{'TOTAL PER STEP':<50} {total_per_step:>12.2f} {'100.0%':>15}", flush=True)
        print("="*80 + "\n", flush=True)


# Global singleton instance
debug_ctx = DebugContext()


# =============================================================================
# STATS LOGGING (verbose_stats)
# =============================================================================

def debug_log(category: str, name: str, message: str, rank: int = 0):
    """
    Log a debug message with automatic context prefix.
    Requires: verbose or verbose_stats

    Args:
        category: Category name (e.g., "PAIRWISE", "ENCODER")
        name: Specific item name (e.g., "Z_pairs", "S_I")
        message: The message content
        rank: Only print on this rank (default 0)
    """
    if not debug_ctx.stats_enabled:
        return

    # Only print on specified rank
    if dist.is_initialized() and dist.get_rank() != rank:
        return

    prefix = debug_ctx.prefix(category)
    print(f"{prefix} {name}: {message}", flush=True)


def debug_tensor(category: str, name: str, tensor: torch.Tensor, rank: int = 0):
    """
    Log tensor statistics with automatic context prefix.
    Requires: verbose or verbose_stats

    Args:
        category: Category name
        name: Tensor name
        tensor: The tensor to log stats for
        rank: Only print on this rank (default 0)
    """
    if not debug_ctx.stats_enabled:
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


def log_tensor_stats(name: str, tensor: torch.Tensor, rank: int = 0):
    """Log tensor statistics for debugging parallel blocks. Controlled by verbose_stats flag."""
    if not debug_ctx.stats_enabled:
        return
    debug_tensor("BLOCKS", name, tensor, rank)


def debug_elements(category: str, name: str, tensor: torch.Tensor, indices: list, rank: int = 0):
    """
    Log specific element values for comparison.
    Requires: verbose or verbose_stats
    """
    if not debug_ctx.stats_enabled:
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
    Requires: verbose or verbose_stats
    """
    if not debug_ctx.stats_enabled:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    prefix = debug_ctx.prefix(category)
    print(f"{prefix} [RANK{rank}/{world_size}] {name}: {message}", flush=True)


def debug_tensor_all_ranks(category: str, name: str, tensor: torch.Tensor):
    """
    Log tensor statistics from ALL ranks (for multi-GPU debugging).
    Requires: verbose or verbose_stats
    """
    if not debug_ctx.stats_enabled:
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
    Requires: verbose or verbose_stats
    """
    if not debug_ctx.stats_enabled:
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


# =============================================================================
# MEMORY LOGGING (verbose_memory)
# =============================================================================

def debug_memory(category: str, stage: str):
    """
    Log GPU memory usage from all ranks.
    Requires: verbose or verbose_memory

    Args:
        category: Category name (e.g., "ENCODER", "DECODER")
        stage: Stage description (e.g., "before_Z_chunk", "after_distogram")
    """
    if not debug_ctx.memory_enabled:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get memory stats for THIS rank's device
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


def debug_tensor_memory(category: str, tensor_name: str, tensor: torch.Tensor):
    """
    Log memory usage for a specific tensor.
    Requires: verbose or verbose_memory

    Args:
        category: Category name (e.g., "DIFFUSION", "ENCODER")
        tensor_name: Name of the tensor (e.g., "Q_L", "Z_II")
        tensor: The tensor to log memory for
    """
    if not debug_ctx.memory_enabled:
        return

    if tensor is None:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    numel = tensor.numel()
    bytes_per_elem = tensor.element_size()
    total_mb = (numel * bytes_per_elem) / 1e6

    prefix = debug_ctx.prefix(category)
    print(
        f"{prefix} [RANK{rank}/{world_size}] TENSOR {tensor_name}: "
        f"shape={list(tensor.shape)}, memory={total_mb:.1f}MB, dtype={tensor.dtype}, device={tensor.device}",
        flush=True
    )


def debug_gpu_memory_snapshot(category: str, stage: str, min_size_mb: float = 50.0):
    """
    Take a comprehensive snapshot of GPU memory, listing all large tensors.
    Requires: verbose or verbose_memory

    IMPORTANT: Only shows tensors on THIS rank's local device to avoid confusion.

    Args:
        category: Category name (e.g., "OOM_DEBUG")
        stage: Stage description (e.g., "before_transition")
        min_size_mb: Only show tensors larger than this (MB)
    """
    import gc

    if not debug_ctx.memory_enabled:
        return

    if not torch.cuda.is_available():
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_device = f"cuda:{local_rank}"

    prefix = debug_ctx.prefix(category)

    # Get overall memory stats for THIS rank's device
    allocated = torch.cuda.memory_allocated(local_rank) / 1e9
    reserved = torch.cuda.memory_reserved(local_rank) / 1e9
    free_mem, total_mem = torch.cuda.mem_get_info(local_rank)
    free_gb = free_mem / 1e9
    total_gb = total_mem / 1e9

    print(f"\n{'='*80}", flush=True)
    print(f"{prefix} [RANK{rank}/{world_size}] GPU MEMORY SNAPSHOT @ {stage}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  Local Device:  {local_device}", flush=True)
    print(f"  Total GPU:     {total_gb:.2f} GB", flush=True)
    print(f"  PyTorch alloc: {allocated:.2f} GB", flush=True)
    print(f"  PyTorch rsrvd: {reserved:.2f} GB", flush=True)
    print(f"  Free (global): {free_gb:.2f} GB", flush=True)
    print(f"{'='*80}", flush=True)

    # Collect all tensors from Python's garbage collector
    # ONLY include tensors on THIS rank's local device
    gc.collect()
    tensor_info = []
    tensors_on_wrong_device = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                device_str = str(obj.device)
                size_mb = obj.numel() * obj.element_size() / 1e6

                # Only include tensors on THIS rank's device
                if device_str == local_device:
                    if size_mb >= min_size_mb:
                        tensor_info.append({
                            'shape': list(obj.shape),
                            'dtype': str(obj.dtype),
                            'size_mb': size_mb,
                            'device': device_str,
                            'requires_grad': obj.requires_grad,
                        })
                else:
                    # Count tensors on wrong device (potential bug)
                    if size_mb >= min_size_mb:
                        tensors_on_wrong_device += 1
        except (ReferenceError, RuntimeError):
            # Object was deleted or inaccessible
            pass

    # Sort by size (largest first)
    tensor_info.sort(key=lambda x: x['size_mb'], reverse=True)

    print(f"  Large tensors on {local_device} (>={min_size_mb:.0f}MB):", flush=True)
    print(f"  {'Shape':<40} {'Size':<12} {'Dtype':<15} {'Grad'}", flush=True)
    print(f"  {'-'*40} {'-'*12} {'-'*15} {'-'*5}", flush=True)

    total_tracked = 0.0
    for info in tensor_info[:30]:  # Top 30 largest
        shape_str = str(info['shape'])
        if len(shape_str) > 38:
            shape_str = shape_str[:35] + "..."
        size_str = f"{info['size_mb']:.1f}MB" if info['size_mb'] < 1000 else f"{info['size_mb']/1000:.2f}GB"
        grad_str = "Y" if info['requires_grad'] else "N"
        print(f"  {shape_str:<40} {size_str:<12} {info['dtype']:<15} {grad_str}", flush=True)
        total_tracked += info['size_mb']

    if len(tensor_info) > 30:
        print(f"  ... and {len(tensor_info) - 30} more tensors", flush=True)

    print(f"  {'-'*80}", flush=True)
    print(f"  Total on {local_device}:  {total_tracked/1000:.2f} GB in {len(tensor_info)} tensors", flush=True)
    print(f"  Untracked:       {allocated - total_tracked/1000:.2f} GB (fragmentation, intermediates)", flush=True)

    if tensors_on_wrong_device > 0:
        print(f"  WARNING: {tensors_on_wrong_device} large tensors found on OTHER devices (potential bug!)", flush=True)

    print(f"{'='*80}\n", flush=True)


def debug_chunking(message: str):
    """
    Log chunking-related messages.
    Requires: verbose or verbose_memory
    """
    if not debug_ctx.memory_enabled:
        return

    print(f"[CHUNKING] {message}", flush=True)


# =============================================================================
# TIMING LOGGING (verbose_time)
# =============================================================================

@contextmanager
def debug_time(category: str, stage: str, rank: int = 0):
    """
    Context manager for timing code blocks.
    Requires: verbose or verbose_time

    Usage:
        with debug_time("ENCODER", "pairformer_block"):
            # ... code to time ...

    Args:
        category: Category name (e.g., "ENCODER", "DIFFUSION")
        stage: Stage name (e.g., "pairformer_block", "attention")
        rank: Only print on this rank (default 0)
    """
    if not debug_ctx.time_enabled:
        yield
        return

    current_rank = dist.get_rank() if dist.is_initialized() else 0

    # Synchronize before timing (to get accurate GPU time)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    yield

    # Synchronize after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    # Record timing data
    debug_ctx.record_timing(category, stage, duration_ms)

    # Print only on specified rank
    if current_rank == rank:
        prefix = debug_ctx.prefix(category)
        print(f"{prefix} [TIME] {stage}: {duration_ms:.2f}ms", flush=True)


def debug_time_func(category: str, stage: str = None, rank: int = 0):
    """
    Decorator for timing functions.
    Requires: verbose or verbose_time

    Usage:
        @debug_time_func("ENCODER", "process_tokens")
        def process_tokens(x):
            ...

    Args:
        category: Category name
        stage: Stage name (defaults to function name)
        rank: Only print on this rank
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_stage = stage or func.__name__
            with debug_time(category, func_stage, rank):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def debug_time_log(category: str, stage: str, duration_ms: float, rank: int = 0):
    """
    Manually log a timing measurement.
    Requires: verbose or verbose_time

    Args:
        category: Category name
        stage: Stage name
        duration_ms: Duration in milliseconds
        rank: Only print on this rank
    """
    if not debug_ctx.time_enabled:
        return

    current_rank = dist.get_rank() if dist.is_initialized() else 0

    # Record timing data
    debug_ctx.record_timing(category, stage, duration_ms)

    # Print only on specified rank
    if current_rank == rank:
        prefix = debug_ctx.prefix(category)
        print(f"{prefix} [TIME] {stage}: {duration_ms:.2f}ms", flush=True)


class TimingContext:
    """
    Reusable timing context for repeated measurements.

    Usage:
        timer = TimingContext("ENCODER", "attention")

        for block in blocks:
            timer.start()
            # ... code ...
            timer.stop()  # Prints and records timing
    """

    def __init__(self, category: str, stage: str, rank: int = 0):
        self.category = category
        self.stage = stage
        self.rank = rank
        self._start_time = None

    def start(self):
        """Start the timer."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer and log the result. Returns duration in ms."""
        if self._start_time is None:
            return 0.0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration_ms = (time.perf_counter() - self._start_time) * 1000
        debug_time_log(self.category, self.stage, duration_ms, self.rank)
        self._start_time = None
        return duration_ms

    def elapsed(self) -> float:
        """Get elapsed time without stopping. Returns duration in ms."""
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_local_device() -> torch.device:
    """Get the correct local device for this rank."""
    if not torch.cuda.is_available():
        return torch.device("cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return torch.device(f"cuda:{local_rank}")


def ensure_on_local_device(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Ensure a tensor is on the correct local device. Move it if necessary.
    Logs a warning if the tensor was on the wrong device.

    Args:
        tensor: The tensor to check/move
        name: Name of the tensor (for logging)

    Returns:
        Tensor on the correct local device
    """
    if tensor is None:
        return None

    local_device = get_local_device()

    if tensor.device != local_device:
        if debug_ctx.memory_enabled:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[WARNING] [RANK{rank}] Tensor '{name}' on wrong device: "
                f"{tensor.device} -> {local_device}",
                flush=True
            )
        return tensor.to(local_device)

    return tensor


# =============================================================================
# COMPREHENSIVE TIMING INSTRUMENTATION
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class StepTiming:
    """Stores all timing data for a single diffusion step."""
    step_num: int
    total_time: float = 0.0
    model_forward_time: float = 0.0
    ode_update_time: float = 0.0
    all_gather_time: float = 0.0
    sync_overhead: float = 0.0
    barrier_overhead: float = 0.0
    memory_overhead: float = 0.0
    device_transfer_overhead: float = 0.0
    other_time: float = 0.0


@dataclass
class EventPair:
    """Stores a CUDA event pair for deferred timing measurement."""
    category: str
    start: Any  # torch.cuda.Event
    end: Any  # torch.cuda.Event
    step_num: int


class TimingInstrument:
    """
    Zero-overhead comprehensive timing for bottleneck analysis.

    Uses CUDA events to record timings without adding synchronization overhead
    during execution. All timing computations are deferred until finalize_all_steps().
    """

    def __init__(self, enabled: bool = False, rank: int = 0, world_size: int = 1, num_steps: int = 200):
        """
        Initialize timing instrumentation.

        Args:
            enabled: Whether timing is active (controlled by verbose_time flag)
            rank: Current GPU rank
            world_size: Total number of GPUs
            num_steps: Total number of diffusion steps (for pre-allocation)
        """
        self.enabled = enabled
        self.rank = rank
        self.world_size = world_size
        self.num_steps = num_steps

        # Per-step storage
        self.step_timings = []

        # CUDA event pairs (processed at end to avoid sync during execution)
        self._pending_events = []

        # CPU-side timing accumulator (for non-CUDA operations)
        self._cpu_timings = {}

        # Current step state
        self.current_step = -1
        self._step_start_events = []
        self._step_end_events = []

        # Compute interval tracking
        self.last_sync_timestamp = None
        self.compute_intervals = []

    @contextmanager
    def time_operation(self, category: str, use_cuda_events: bool = True):
        """
        Context manager for timing with zero overhead during execution.

        Args:
            category: Operation category (e.g., "model_forward", "cuda_sync")
            use_cuda_events: If True, use CUDA events (GPU ops). If False, use CPU timing.
        """
        if not self.enabled:
            yield
            return

        if use_cuda_events:
            # GPU operation timing - use CUDA events (no synchronization)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            yield

            end.record()
            # Don't synchronize! Just record for later processing
            self._pending_events.append(EventPair(
                category=category,
                start=start,
                end=end,
                step_num=self.current_step
            ))
        else:
            # CPU operation timing - use wall-clock time
            start_time = time.perf_counter()

            yield

            elapsed = time.perf_counter() - start_time
            self._record_cpu_timing(category, elapsed * 1000)  # Convert to ms

    def mark_step_start(self, step_num: int):
        """Mark the beginning of a diffusion step."""
        if not self.enabled:
            return

        self.current_step = step_num
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._step_start_events.append((step_num, event))

    def mark_step_end(self):
        """Mark the end of a diffusion step."""
        if not self.enabled:
            return

        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._step_end_events.append((self.current_step, event))

    def mark_sync_point(self, sync_type: str):
        """
        Track when synchronization occurs to measure compute intervals.

        Args:
            sync_type: Type of synchronization (e.g., "all_gather", "barrier", "cuda_sync")
        """
        if not self.enabled:
            return

        current_time = time.perf_counter()
        if self.last_sync_timestamp is not None:
            # Time between syncs = pure compute time
            interval = (current_time - self.last_sync_timestamp) * 1000  # ms
            self.compute_intervals.append((self.current_step, sync_type, interval))
        self.last_sync_timestamp = current_time

    def _record_cpu_timing(self, category: str, elapsed_ms: float):
        """Record CPU-side timing."""
        if category not in self._cpu_timings:
            self._cpu_timings[category] = []
        self._cpu_timings[category].append(elapsed_ms)

    def finalize_all_steps(self):
        """
        Process all recorded events ONCE at the very end.

        This is when we finally synchronize to get all timing results.
        """
        if not self.enabled:
            return

        # NOW synchronize to get all timing results
        torch.cuda.synchronize()

        # Initialize step timings
        self.step_timings = [StepTiming(step_num=i) for i in range(self.num_steps)]

        # Process step start/end events
        step_times_map = {}
        for step_num, start_event in self._step_start_events:
            for end_step_num, end_event in self._step_end_events:
                if step_num == end_step_num:
                    elapsed_ms = start_event.elapsed_time(end_event)
                    step_times_map[step_num] = elapsed_ms
                    if step_num < len(self.step_timings):
                        self.step_timings[step_num].total_time = elapsed_ms
                    break

        # Process all pending CUDA events
        category_times_per_step = {}
        for event_pair in self._pending_events:
            elapsed_ms = event_pair.start.elapsed_time(event_pair.end)
            step_num = event_pair.step_num
            category = event_pair.category

            if step_num not in category_times_per_step:
                category_times_per_step[step_num] = {}
            if category not in category_times_per_step[step_num]:
                category_times_per_step[step_num][category] = 0.0
            category_times_per_step[step_num][category] += elapsed_ms

        # Assign category times to step timings
        for step_num, categories in category_times_per_step.items():
            if step_num < 0 or step_num >= len(self.step_timings):
                continue

            step_timing = self.step_timings[step_num]
            for category, total_ms in categories.items():
                if category == "model_forward":
                    step_timing.model_forward_time = total_ms
                elif category == "ode_update":
                    step_timing.ode_update_time = total_ms
                elif category == "all_gather":
                    step_timing.all_gather_time = total_ms
                elif category == "cuda_sync":
                    step_timing.sync_overhead += total_ms
                elif category == "barrier":
                    step_timing.barrier_overhead += total_ms
                elif category == "empty_cache":
                    step_timing.memory_overhead += total_ms
                elif category == "device_transfer":
                    step_timing.device_transfer_overhead += total_ms
                else:
                    step_timing.other_time += total_ms

        # Add CPU timings (distributed across all steps)
        for category, timings in self._cpu_timings.items():
            total_ms = sum(timings)
            per_step_ms = total_ms / self.num_steps if self.num_steps > 0 else 0

            for step_timing in self.step_timings:
                if category == "cuda_sync":
                    step_timing.sync_overhead += per_step_ms
                elif category == "barrier":
                    step_timing.barrier_overhead += per_step_ms
                elif category == "empty_cache":
                    step_timing.memory_overhead += per_step_ms
                else:
                    step_timing.other_time += per_step_ms

    def print_comprehensive_report(self):
        """Generate detailed bottleneck analysis report."""
        if not self.enabled:
            return

        # Only print from rank 0
        if self.rank != 0:
            return

        if not self.step_timings:
            print("[WARNING] No timing data collected. Call finalize_all_steps() first.")
            return

        # Compute statistics
        valid_steps = [s for s in self.step_timings if s.total_time > 0]
        if not valid_steps:
            print("[WARNING] No valid step timings recorded.")
            return

        num_valid = len(valid_steps)

        # Aggregate by category
        def compute_stats(values):
            if not values:
                return 0, 0, 0, 0, 0
            import statistics
            total = sum(values)
            mean = total / len(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0
            min_val = min(values)
            max_val = max(values)
            return mean, std_dev, min_val, max_val, total

        # Extract values for each category
        total_times = [s.total_time for s in valid_steps]
        model_forward_times = [s.model_forward_time for s in valid_steps]
        ode_update_times = [s.ode_update_time for s in valid_steps]
        all_gather_times = [s.all_gather_time for s in valid_steps]
        sync_times = [s.sync_overhead for s in valid_steps]
        barrier_times = [s.barrier_overhead for s in valid_steps]
        memory_times = [s.memory_overhead for s in valid_steps]
        transfer_times = [s.device_transfer_overhead for s in valid_steps]
        other_times = [s.other_time for s in valid_steps]

        # Compute stats
        total_stats = compute_stats(total_times)
        model_forward_stats = compute_stats(model_forward_times)
        ode_update_stats = compute_stats(ode_update_times)
        all_gather_stats = compute_stats(all_gather_times)
        sync_stats = compute_stats(sync_times)
        barrier_stats = compute_stats(barrier_times)
        memory_stats = compute_stats(memory_times)
        transfer_stats = compute_stats(transfer_times)
        other_stats = compute_stats(other_times)

        # Print report
        print("\n" + "="*100)
        print("=== RFD3 COMPREHENSIVE TIMING ANALYSIS ===")
        print("="*100)
        print(f"Configuration:")
        print(f"  GPUs:                  {self.world_size}")
        print(f"  Steps measured:        {num_valid}/{self.num_steps}")

        print(f"\n=== PER-STEP STATISTICS ({num_valid} steps) ===")
        header = f"{'Category':<22} | {'Mean':>8} | {'Std Dev':>8} | {'Min':>8} | {'Max':>8} | {'Total':>10} | {'% Total':>8}"
        print(header)
        print("-" * len(header))

        def print_row(name, stats):
            mean, std_dev, min_val, max_val, total = stats
            total_sec = total / 1000
            pct = (total / (total_stats[4] + 1e-9)) * 100
            print(f"{name:<22} | {mean:>7.0f}ms | {std_dev:>7.0f}ms | {min_val:>7.0f}ms | {max_val:>7.0f}ms | {total_sec:>9.1f}s | {pct:>7.1f}%")

        print_row("Model Forward", model_forward_stats)
        print_row("ODE Update", ode_update_stats)
        print_row("All-Gather", all_gather_stats)
        print_row("CUDA Synchronize", sync_stats)
        print_row("Dist Barrier", barrier_stats)
        print_row("Empty Cache", memory_stats)
        print_row("Device Transfer", transfer_stats)
        print_row("Other", other_stats)
        print("-" * len(header))
        print_row("Total Step Time", total_stats)

        # Compute interval analysis
        if self.compute_intervals:
            intervals = [interval for _, _, interval in self.compute_intervals]
            avg_interval = sum(intervals) / len(intervals)
            syncs_per_step = len(self.compute_intervals) / num_valid

            total_compute = sum(intervals)
            total_sync = sum(all_gather_times) + sum(sync_times) + sum(barrier_times)
            compute_efficiency = (total_compute / (total_compute + total_sync + 1e-9)) * 100

            print(f"\n=== COMPUTE INTERVAL ANALYSIS ===")
            print(f"Average time between syncs:     {avg_interval:.1f}ms")
            print(f"Sync frequency:                 {syncs_per_step:.1f} syncs/step")
            print(f"Pure compute efficiency:        {compute_efficiency:.1f}% (time not blocked on sync)")

        # Bottleneck identification
        print(f"\n=== BOTTLENECK IDENTIFICATION ===")

        # Find primary bottleneck
        categories = [
            ("Model Forward", model_forward_stats[4]),
            ("ODE Update", ode_update_stats[4]),
            ("All-Gather", all_gather_stats[4]),
            ("CUDA Synchronize", sync_stats[4]),
            ("Dist Barrier", barrier_stats[4]),
            ("Empty Cache", memory_stats[4]),
            ("Device Transfer", transfer_stats[4]),
        ]
        categories.sort(key=lambda x: x[1], reverse=True)

        primary = categories[0]
        secondary = categories[1]

        print(f"PRIMARY BOTTLENECK: {primary[0]} ({primary[1]/1000:.1f}s, {(primary[1]/(total_stats[4]+1e-9))*100:.1f}% of time)")
        print(f"SECONDARY BOTTLENECK: {secondary[0]} ({secondary[1]/1000:.1f}s, {(secondary[1]/(total_stats[4]+1e-9))*100:.1f}% of time)")

        # Sync overhead summary
        total_sync_ms = all_gather_stats[4] + sync_stats[4] + barrier_stats[4]
        sync_pct = (total_sync_ms / (total_stats[4] + 1e-9)) * 100

        print(f"\nSYNC OVERHEAD: {sync_pct:.1f}% total")
        print(f"  All-gather:      {(all_gather_stats[4]/(total_stats[4]+1e-9))*100:.1f}%")
        print(f"  Explicit syncs:  {(sync_stats[4]/(total_stats[4]+1e-9))*100:.1f}%")
        print(f"  Barriers:        {(barrier_stats[4]/(total_stats[4]+1e-9))*100:.1f}%")

        # Recommendations
        print(f"\n=== NEXT STEPS FOR INVESTIGATION ===")
        if model_forward_stats[4] > total_stats[4] * 0.5:
            print("Since model_forward dominates:")
            print("1. Check debug_ctx.print_timing_summary() for encoder/decoder/transformer breakdown")
            print("2. Profile individual operations within model forward")
            print("3. Check for hidden O(N) loops in parallel blocks")
            print("4. Measure GPU utilization during model forward")

        if sync_pct > 10:
            print("\nSync overhead is significant:")
            print("1. Consider reducing all_gather frequency")
            print("2. Investigate async communication patterns")
            print("3. Check for unnecessary barriers")

        print("="*100 + "\n")


# ALL_GATHER TIMING (for parallel synchronization overhead analysis)
# =============================================================================

def timed_all_gather(
    output_or_list,
    input_tensor,
    category: str,
    name: str,
    gather_type: str = "all_gather",
    timing_instrument=None,
) -> float:
    """
    Wrapper for dist.all_gather operations with timing.
    Only activates when debug_ctx.time_enabled or memory_enabled is True.

    Args:
        output_or_list: Output tensor or list for all_gather
        input_tensor: Input tensor to gather
        category: Debug category (e.g., "ENCODER", "TRANSFORMER")
        name: Operation name (e.g., "S_I_gather_block3", "Q_L_chunk_sync")
        gather_type: "all_gather" or "all_gather_into_tensor"
        timing_instrument: Optional TimingInstrument for zero-overhead timing

    Returns:
        elapsed_ms: Time taken for the all_gather operation in milliseconds
    """
    # Skip timing if debug not enabled
    if not (debug_ctx.time_enabled or debug_ctx.memory_enabled):
        if gather_type == "all_gather":
            dist.all_gather(output_or_list, input_tensor)
        elif gather_type == "all_gather_into_tensor":
            dist.all_gather_into_tensor(output_or_list, input_tensor)
        return 0.0

    # Use CUDA events for accurate timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    # Execute the actual all_gather
    if gather_type == "all_gather":
        dist.all_gather(output_or_list, input_tensor)
    elif gather_type == "all_gather_into_tensor":
        dist.all_gather_into_tensor(output_or_list, input_tensor)
    else:
        raise ValueError(f"Unknown gather_type: {gather_type}")

    end.record()

    # If using comprehensive timing instrument, don't synchronize here!
    # Record events for deferred processing to avoid double sync overhead
    if timing_instrument and timing_instrument.enabled:
        timing_instrument._pending_events.append(EventPair(
            category="all_gather",
            start=start,
            end=end,
            step_num=timing_instrument.current_step
        ))
        # Return 0 since we don't have the timing yet (deferred)
        return 0.0

    # Legacy path: synchronize immediately (doubles sync overhead)
    torch.cuda.synchronize()  # FIXME: This doubles sync overhead!
    elapsed_ms = start.elapsed_time(end)

    # Record timing
    debug_ctx.record_allgather_time(category, name, elapsed_ms)

    # Optionally log immediately (if verbose_stats)
    if debug_ctx.stats_enabled:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        debug_log(category, f"all_gather_timing",
                  f"{name}: {elapsed_ms:.2f}ms [rank {rank}/{world_size}]")

    return elapsed_ms
