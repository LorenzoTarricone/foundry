"""
Parallel (multi-GPU) inference implementations for RFD3.

This package contains parallel-specific implementations of model components
that enable scaling beyond single-GPU memory limits through distributed inference.
"""

# Export main parallel classes
from .layers.encoders import ParallelTokenInitializer, ParallelDiffusionTokenEncoder
from .diffusion_module import ParallelDiffusionModule

__all__ = [
    "ParallelTokenInitializer",
    "ParallelDiffusionTokenEncoder",
    "ParallelDiffusionModule",
]
