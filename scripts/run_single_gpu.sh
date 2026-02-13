#!/bin/bash
#=============================================================================
# SLURM Batch Script for Single-GPU RFDiffusion3 Design
#
# Isambard-AI cluster configuration for standard (non-parallel) inference.
# Used for N=1 baseline benchmarks.
#
# Usage:
#   sbatch scripts/run_single_gpu.sh                           # Use defaults
#   sbatch scripts/run_single_gpu.sh path/to/config.yaml       # Custom config
#   sbatch --time=04:00:00 scripts/run_single_gpu.sh           # Override time
#
#=============================================================================

#SBATCH --job-name=RFD3_single
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/slurm_%j.out
#SBATCH --error=logs/slurm/slurm_%j.err

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
echo "Config: $CONFIG_FILE"
echo "=============================================="

# Create logs directory if it doesn't exist
mkdir -p logs/slurm

# Load required modules
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
# Memory optimization
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# =============================================================================
# Launch single-GPU job
# =============================================================================
echo ""
echo "Starting single-GPU RFD3 inference..."
echo "Config: $CONFIG_FILE"
echo ""

# Run the design script in single-GPU mode (no distributed)
# Set WORLD_SIZE=1 and RANK=0 to indicate non-distributed mode
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

python scripts/design_parallel.py \
    --config "$CONFIG_FILE" \
    --launch_mode worker

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
