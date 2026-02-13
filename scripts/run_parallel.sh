#!/bin/bash
#=============================================================================
# SLURM Batch Script for Parallel Multi-GPU RFDiffusion3 Design
#
# Isambard-AI cluster configuration for distributed PyTorch inference.
# Supports both single-node (1-4 GPUs) and multi-node (>4 GPUs) runs.
#
# Usage:
#   sbatch scripts/run_parallel.sh                    # Use defaults
#   sbatch --gpus=8 scripts/run_parallel.sh           # Override GPU count
#   sbatch -J my_job scripts/run_parallel.sh          # Override job name
#
# To modify GPU count without editing this file:
#   sbatch --gpus=6 --nodes=2 scripts/run_parallel.sh
#=============================================================================

#SBATCH --job-name=RFD3_parallel
#SBATCH --gpus=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/slurm_%j.out
#SBATCH --error=logs/slurm/slurm_%j.err

# NOTE: Nodes are auto-calculated from GPU count (4 GPUs per node on Isambard)
# Override with: sbatch --gpus=5 scripts/run_parallel.sh

# =============================================================================
# Configuration - Modify these as needed
# =============================================================================
CONFIG_FILE="${1:-configs/design_parallel.yaml}"
CONDA_ENV="foundry"  # Your conda environment name

# =============================================================================
# Setup environment
# =============================================================================
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Node list: $SLURM_NODELIST"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules for NCCL communication
module load brics/nccl brics/aws-ofi-nccl

# Activate conda environment
# Try common conda initialization paths
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Warning: Could not find conda.sh, trying direct activation"
fi

conda activate $CONDA_ENV
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python)"

# =============================================================================
# Distributed training environment setup
# =============================================================================

# Set master address (first node in allocation)
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29600

echo "Master address: $MASTER_ADDR:$MASTER_PORT"

# Memory optimization for PyTorch CUDA allocator
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Disable internal torchrun launch since we're using srun
export RFD3_USE_SRUN=1

# =============================================================================
# Launch distributed job
# =============================================================================
echo ""
echo "Starting distributed RFD3 inference..."
echo "Config: $CONFIG_FILE"
echo ""

# Isambard has 4 GPUs per node
GPUS_PER_NODE=4

# Try to read attention_parallel_factor from config to determine exact GPU count
REQUESTED_GPUS=${SLURM_GPUS:-8}
if [ -f "$CONFIG_FILE" ]; then
    # Extract attention_parallel_factor from YAML (if present)
    APF=$(grep -E "^attention_parallel_factor:" "$CONFIG_FILE" 2>/dev/null | awk '{print $2}')
    if [ -n "$APF" ] && [ "$APF" != "null" ] && [ "$APF" -gt 0 ] 2>/dev/null; then
        REQUESTED_GPUS=$APF
        echo "Using attention_parallel_factor=$APF from config"
    fi
fi

# Calculate nodes needed (round up)
NUM_NODES=${SLURM_JOB_NUM_NODES:-$(( (REQUESTED_GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))}
TOTAL_ALLOCATED=$((NUM_NODES * GPUS_PER_NODE))

echo "GPUs requested: $REQUESTED_GPUS"
echo "Nodes allocated: $NUM_NODES (with $TOTAL_ALLOCATED GPUs total)"

# Calculate tasks per node for distribution across nodes
# For NCCL to work correctly, we use the TRADITIONAL approach:
# - Each node sees all its GPUs (not --gpus-per-task=1)
# - LOCAL_RANK = SLURM_LOCALID selects which GPU each task uses
TOTAL_TASKS=$REQUESTED_GPUS
REMAINDER=$((REQUESTED_GPUS % NUM_NODES))

# GPUs per node: ceiling division to ensure enough GPUs on each node
GPUS_PER_NODE_ACTUAL=$(( (REQUESTED_GPUS + NUM_NODES - 1) / NUM_NODES ))

if [ "$REMAINDER" -eq 0 ]; then
    # Even distribution: e.g., 6 GPUs on 2 nodes = 3 tasks per node
    TASKS_PER_NODE=$((REQUESTED_GPUS / NUM_NODES))
    echo "Even distribution: $TASKS_PER_NODE tasks/GPUs per node"
    SRUN_TASK_ARGS="--ntasks-per-node=$TASKS_PER_NODE"
else
    # Uneven distribution: e.g., 5 GPUs on 2 nodes
    # Cannot use --ntasks-per-node for uneven, use --ntasks only
    echo "Uneven distribution: $REQUESTED_GPUS tasks across $NUM_NODES nodes (max $GPUS_PER_NODE_ACTUAL GPUs per node)"
    SRUN_TASK_ARGS="--ntasks=$TOTAL_TASKS"
fi

echo "Launching $TOTAL_TASKS tasks across $NUM_NODES nodes"

# TRADITIONAL MULTI-GPU APPROACH:
# - --gpus-per-node: each node sees all its allocated GPUs
# - Each task uses SLURM_LOCALID to select its GPU
# This works correctly with NCCL which expects to see the full GPU topology.
srun --nodes=$NUM_NODES \
     $SRUN_TASK_ARGS \
     --gpus-per-node=$GPUS_PER_NODE_ACTUAL \
     --mpi=pmi2 \
     --export=ALL \
     bash -c '
        # CRITICAL: Set CUDA memory allocator config BEFORE any torch import
        # This reduces fragmentation by using expandable memory segments
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        export PYTORCH_ALLOC_CONF="expandable_segments:True"

        # Set distributed environment variables
        export WORLD_SIZE=$SLURM_NTASKS
        export RANK=$PMI_RANK
        # LOCAL_RANK = which GPU on this node (0, 1, 2, ...)
        # SLURM_LOCALID gives the local task ID on the node
        export LOCAL_RANK=$SLURM_LOCALID

        # Workaround for Lightning Fabric SLURM validation
        # Use ceiling of tasks/nodes for uneven distribution
        export SLURM_NTASKS_PER_NODE='"$GPUS_PER_NODE_ACTUAL"'

        # Debug output
        echo "[Rank $RANK] Starting on $(hostname), LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

        # Run the design script as a worker (skip internal torchrun)
        python scripts/design_parallel.py \
            --config '"$CONFIG_FILE"' \
            --launch_mode worker
     '

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
