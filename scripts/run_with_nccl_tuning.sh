#!/bin/bash
# NCCL tuning for RFD3 parallel inference
# This script sets optimized NCCL environment variables for multi-GPU communication

# =============================================================================
# InfiniBand Optimization (if available)
# =============================================================================
export NCCL_IB_DISABLE=0                 # Enable InfiniBand
export NCCL_IB_HCA=mlx5                  # InfiniBand HCA (adjust for your system)
export NCCL_IB_GID_INDEX=3               # RoCE v2 for InfiniBand

# If no InfiniBand, comment out above and use Ethernet:
# export NCCL_IB_DISABLE=1               # Disable InfiniBand
# export NCCL_SOCKET_IFNAME=eth0         # Use high-speed Ethernet

# =============================================================================
# Algorithm Selection
# =============================================================================
export NCCL_ALGO=Ring,Tree               # Ring for large messages, Tree for small
export NCCL_PROTO=Simple                 # Simple protocol for lower latency

# =============================================================================
# Buffer Tuning for S_I Tensors (~500KB each)
# =============================================================================
export NCCL_BUFFSIZE=2097152             # 2MB buffer size
export NCCL_MIN_NCHANNELS=4              # Minimum channels for parallel transfers
export NCCL_MAX_NCHANNELS=16             # Maximum channels for bandwidth

# =============================================================================
# Thread Optimization
# =============================================================================
export NCCL_NSOCKS_PERTHREAD=4           # Sockets per thread for parallelism
export NCCL_SOCKET_NTHREADS=2            # Number of socket threads

# =============================================================================
# Multi-Node Optimization (if running across nodes)
# =============================================================================
export NCCL_CROSS_NIC=1                  # Use multiple NICs if available
export NCCL_NET_GDR_LEVEL=5              # GPU Direct RDMA (requires hardware support)
export NCCL_P2P_LEVEL=SYS                # Enable P2P communication within system

# =============================================================================
# Small Message Optimization
# =============================================================================
export NCCL_LL_THRESHOLD=16384           # Use Low-Latency protocol for <16KB messages
export NCCL_LL128_THRESHOLD=65536        # Use LL128 for 16-64KB messages

# =============================================================================
# Debug Settings (DISABLE in production for performance)
# =============================================================================
# export NCCL_DEBUG=INFO                 # Show NCCL algorithm choices
# export NCCL_DEBUG_SUBSYS=ALL           # Detailed debug information
# export NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.log  # Per-host, per-process logs

# =============================================================================
# Execute the command passed as arguments
# =============================================================================
exec "$@"
