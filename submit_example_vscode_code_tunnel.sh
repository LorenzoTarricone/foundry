#!/bin/bash
#SBATCH --job-name=code_tunnel
#SBATCH --gpus=1            # this also allocates 72 CPU cores and 115GB memory
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/code_tunnel_%j.out

# Start named VS Code tunnel for remote connection to compute node
~/opt/vscode_cli/code tunnel --name "i-ai_compute"