#!/bin/bash
#=============================================================================
# SLURM Batch Script for RFD3 Parallel Inference with NCCL Tuning
#
# Isambard-AI configuration: 4 GPUs per node, 8 GPUs total = 2 nodes
#
# Usage:
#   sbatch scripts/benchmark_with_nccl.sh
#
# Expected speedup: 1.2-1.5× over untuned baseline
#=============================================================================

#SBATCH --job-name=RFD3_NCCL_tuned
#SBATCH --gpus=8                      # Total GPUs (Isambard auto-allocates 2 nodes)
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/slurm_%j.out
#SBATCH --error=logs/slurm/slurm_%j.err

# =============================================================================
# Configuration
# =============================================================================
CONFIG_FILE="configs/design_parallel.yaml"
CONDA_ENV="foundry"
GPUS_PER_NODE=4  # Isambard-AI has 4 GPUs per node

# =============================================================================
# Setup Environment
# =============================================================================
echo "=============================================="
echo "RFD3 Parallel Inference with NCCL Tuning"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=============================================="

# Create logs directory
mkdir -p logs/slurm

# Load required modules for NCCL
module load brics/nccl brics/aws-ofi-nccl

# Activate conda environment
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate $CONDA_ENV
echo "Conda environment: $CONDA_DEFAULT_ENV"

# =============================================================================
# NCCL Tuning Configuration
# =============================================================================
echo ""
echo "Applying NCCL Optimizations:"
source scripts/run_with_nccl_tuning.sh
echo "  NCCL_ALGO=$NCCL_ALGO"
echo "  NCCL_PROTO=$NCCL_PROTO"
echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "  NCCL_BUFFSIZE=$NCCL_BUFFSIZE"
echo "  NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS"
echo ""

# =============================================================================
# Distributed Setup
# =============================================================================
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29600
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export RFD3_USE_SRUN=1

echo "Master address: $MASTER_ADDR:$MASTER_PORT"

# Calculate distribution
REQUESTED_GPUS=8
NUM_NODES=${SLURM_JOB_NUM_NODES:-2}
TASKS_PER_NODE=$(( REQUESTED_GPUS / NUM_NODES ))

echo "Launching $REQUESTED_GPUS tasks across $NUM_NODES nodes ($TASKS_PER_NODE tasks/node)"
echo ""

# =============================================================================
# Launch Distributed Job with NCCL Tuning
# =============================================================================
srun --nodes=$NUM_NODES \
     --ntasks-per-node=$TASKS_PER_NODE \
     --gpus-per-node=$GPUS_PER_NODE \
     --mpi=pmi2 \
     --export=ALL \
     bash -c '
        # Apply NCCL tuning in each worker
        source scripts/run_with_nccl_tuning.sh

        # Set memory allocator config
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        export PYTORCH_ALLOC_CONF="expandable_segments:True"

        # Distributed environment variables
        export WORLD_SIZE=$SLURM_NTASKS
        export RANK=$PMI_RANK
        export LOCAL_RANK=$SLURM_LOCALID
        export SLURM_NTASKS_PER_NODE='"$TASKS_PER_NODE"'

        echo "[Rank $RANK] Starting on $(hostname), LOCAL_RANK=$LOCAL_RANK, NCCL_ALGO=$NCCL_ALGO"

        # Run design script
        python scripts/design_parallel.py \
            --config '"$CONFIG_FILE"' \
            --launch_mode worker
     '

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output: inference_outputs/design_I_173AAASU_parallel"
echo "=============================================="

exit $EXIT_CODE
