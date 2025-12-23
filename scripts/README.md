# RFDiffusion3 Scripts

This folder contains scripts for running RFDiffusion3 protein design.

## Scripts

### `design_annotate.py`
Standard design script with optional memory optimizations.

```bash
# Standard mode (single GPU):
python scripts/design_annotate.py --config config/design_annotate.yaml

# Low memory mode (sequential chunking):
python scripts/design_annotate.py --config config/design_low_memory.yaml

# Override config values:
python scripts/design_annotate.py --config config/design_annotate.yaml --length 200
```

### `design_parallel.py`
Multi-GPU parallel inference that splits attention across GPUs.

```bash
# Step 1: Allocate GPUs via srun
srun --gpus=4 --time=03:00:00 --pty /bin/bash --login

# Step 2: Run the script
python scripts/design_parallel.py --config config/design_parallel.yaml
```

## Config Files

All configs are in the `config/` folder:

| Config | Description |
|--------|-------------|
| `design_annotate.yaml` | Standard single-GPU inference |
| `design_low_memory.yaml` | Single-GPU with sequential chunking |
| `design_parallel.yaml` | Multi-GPU parallel inference |

## Memory Modes

| Mode | GPUs | Memory per GPU | Speed |
|------|------|----------------|-------|
| Standard | 1 | O(L² + I²) | Fast |
| Low Memory | 1 | O(L·chunk + I·chunk) | Slower |
| Parallel | N | O(L²/N + I²/N) | Fast |

## Usage Examples

### Design 100-residue protein (standard)
```bash
python scripts/design_annotate.py --config config/design_annotate.yaml
```

### Design 500-residue protein with 4 GPUs
```bash
# Allocate GPUs
srun --gpus=4 --time=03:00:00 --pty /bin/bash --login

# Run parallel design
python scripts/design_parallel.py --config config/design_parallel.yaml --length 500
```

### Design symmetric protein (C3)
```bash
python scripts/design_annotate.py --config config/design_annotate.yaml --symmetry C3
```

### Design long protein on single GPU (low memory)
```bash
python scripts/design_annotate.py --config config/design_low_memory.yaml --length 800
```

## Command Line Arguments

All scripts support these arguments:

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config file |
| `--out_dir` | Output directory |
| `--length` | Protein length |
| `--num_designs` | Number of designs |
| `--symmetry` | Symmetry type (C2, C3, D2, etc.) |
| `--mpnn_batch_size` | MPNN batch size |
| `--wandb` | Enable W&B logging |

CLI arguments override config file values.

