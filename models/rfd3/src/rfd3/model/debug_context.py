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
