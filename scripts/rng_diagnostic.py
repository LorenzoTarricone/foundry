#!/usr/bin/env python3
"""
RNG State Diagnostic Utility

This module tracks torch RNG state at various checkpoints to diagnose
divergence between standard and parallel modes.

Usage:
    from scripts.rng_diagnostic import rng_checkpoint

    rng_checkpoint("before_engine_init")
    engine = RFD3InferenceEngine(...)
    rng_checkpoint("after_engine_init")
"""

import hashlib
import torch
import torch.distributed as dist
import os
from typing import Optional

# Global storage for RNG state checksums
_rng_checkpoints: dict[str, str] = {}


def _get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _rng_state_checksum() -> str:
    """
    Compute a checksum of the current torch RNG state.

    This allows comparing RNG states between different runs without
    storing the full state (which is large).
    """
    # Get CPU generator state
    cpu_state = torch.get_rng_state()

    # Get CUDA generator state if available
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
    else:
        cuda_state = b""

    # Combine and hash
    combined = cpu_state.numpy().tobytes() + cuda_state.numpy().tobytes() if torch.cuda.is_available() else cpu_state.numpy().tobytes()
    return hashlib.md5(combined).hexdigest()[:16]


def _sample_random_numbers(n: int = 5) -> list[float]:
    """
    Sample a few random numbers to show actual RNG output.

    WARNING: This consumes RNG state! Only use for debugging.
    """
    # Save state
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    # Sample
    samples = torch.randn(n).tolist()

    # Restore state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)

    return samples


def rng_checkpoint(name: str, sample: bool = True) -> dict:
    """
    Record the current RNG state at a named checkpoint.

    Args:
        name: Identifier for this checkpoint
        sample: If True, also sample random numbers (without advancing state)

    Returns:
        dict with checkpoint information
    """
    rank = _get_rank()
    mode = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    mode_str = "PARALLEL" if mode == "1" else "STANDARD"

    checksum = _rng_state_checksum()
    samples = _sample_random_numbers(5) if sample else []

    info = {
        "name": name,
        "mode": mode_str,
        "rank": rank,
        "checksum": checksum,
        "samples": samples,
    }

    _rng_checkpoints[name] = checksum

    # Format samples for printing
    samples_str = ", ".join(f"{s:.6f}" for s in samples[:3]) if samples else "N/A"

    print(f"[RNG-{mode_str}] {name}: checksum={checksum}, samples=[{samples_str}, ...]", flush=True)

    return info


def compare_checkpoints(name1: str, name2: str) -> bool:
    """Compare two previously recorded checkpoints."""
    if name1 not in _rng_checkpoints or name2 not in _rng_checkpoints:
        print(f"Warning: Checkpoint {name1} or {name2} not found")
        return False

    match = _rng_checkpoints[name1] == _rng_checkpoints[name2]
    status = "MATCH" if match else "DIFFER"
    print(f"[RNG] Compare {name1} vs {name2}: {status}")
    return match


def get_checkpoints() -> dict[str, str]:
    """Return all recorded checkpoints."""
    return dict(_rng_checkpoints)


def clear_checkpoints():
    """Clear recorded checkpoints."""
    _rng_checkpoints.clear()


def print_all_checkpoints():
    """Print all recorded checkpoints."""
    mode = "PARALLEL" if os.environ.get("RFD3_ATTENTION_PARALLEL", "0") == "1" else "STANDARD"
    print(f"\n[RNG-{mode}] All checkpoints:")
    for name, checksum in _rng_checkpoints.items():
        print(f"  {name}: {checksum}")


if __name__ == "__main__":
    # Test the diagnostic utility
    import random
    import numpy as np

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    print("Testing RNG diagnostic utility...")

    set_seed(42)
    rng_checkpoint("after_seed_42")

    # Consume some random numbers
    _ = torch.randn(10)
    rng_checkpoint("after_10_randn")

    # Re-seed
    set_seed(42)
    rng_checkpoint("after_reseed_42")

    print_all_checkpoints()
