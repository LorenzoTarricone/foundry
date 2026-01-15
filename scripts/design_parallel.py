#!/usr/bin/env python3
"""
Parallel Multi-GPU RFDiffusion3 Design Script

This script runs RFD3 inference in parallel across multiple GPUs.
Each GPU processes a chunk of queries, avoiding full L×L and I×I tensors.

SETUP:
======

1. Allocate GPUs via srun:
   srun --gpus=4 --time=03:00:00 --pty /bin/bash --login

2. Run this script:
   python scripts/design_parallel.py --config config/design_parallel.yaml

The script will:
- Auto-detect allocated GPUs from CUDA_VISIBLE_DEVICES
- Launch distributed processes using torchrun internally
- Split attention queries across GPUs
- Gather results after each attention layer

ALTERNATIVE (direct torchrun):
==============================

If you prefer manual control, you can also launch directly:
   torchrun --nproc_per_node=4 scripts/design_parallel.py --config config/design_parallel.yaml

HOW IT WORKS:
=============

Standard attention:  Q[L] × K[L]^T → [L, L] tensor (OOM for large L)
Parallel attention:  Q[L/N] × K[L]^T → [L/N, L] tensor per GPU

Memory savings: O(L²/N) per GPU instead of O(L²)
"""

# Set environment variables BEFORE any imports
import os
os.environ.setdefault('CCD_MIRROR_PATH', '')
os.environ.setdefault('PDB_MIRROR_PATH', '')

import sys
import argparse
import subprocess
from pathlib import Path
import time
import yaml

import torch
import torch.distributed as dist
import random
import numpy as np


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.
    Ensures consistent seeds across all distributed processes.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Use the same seed for all processes (for reproducibility)
    # In distributed mode, all ranks use the same base seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Only print on rank 0 (if distributed is initialized) or main process
    try:
        rank = dist.get_rank() if dist.is_initialized() else 0
    except:
        rank = 0
    if rank == 0:
        print(f"Random seed set to: {seed}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge config file with command line arguments (CLI takes precedence)."""
    result = dict(config)
    
    # Flatten wandb config
    if 'wandb' in result:
        wandb_config = result.pop('wandb')
        result['use_wandb'] = wandb_config.get('enabled', False)
        result['wandb_project'] = wandb_config.get('project', 'fast-rfd3')
        result['wandb_run_name'] = wandb_config.get('run_name')
    
    # Override with CLI arguments
    for key, value in vars(args).items():
        if key in ('config', 'launch_mode'):
            continue
        if value is not None:
            result[key] = value
    
    return result


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process():
    """Check if this is the main process."""
    return get_rank() == 0


def print_rank0(msg):
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


def setup_distributed():
    """Initialize PyTorch distributed processing."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Debug: show CUDA environment
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    device_count = torch.cuda.device_count()
    print(f"[Rank {rank}] CUDA env: CUDA_VISIBLE_DEVICES={cuda_visible}, "
          f"device_count={device_count}, local_rank={local_rank}")
    
    if local_rank >= 0 and world_size > 1:
        if local_rank >= device_count:
            raise RuntimeError(
                f"local_rank={local_rank} but only {device_count} GPUs visible! "
                f"CUDA_VISIBLE_DEVICES={cuda_visible}"
            )
        
        print(f"[Rank {rank}] Setting CUDA device to local_rank={local_rank}")
        torch.cuda.set_device(local_rank)
        
        # Verify the device was set correctly
        current = torch.cuda.current_device()
        print(f"[Rank {rank}] Current CUDA device after set_device: {current}")
        
        # Initialize with explicit device_id to avoid NCCL communicator issues
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}")
        )
        dist.barrier(device_ids=[local_rank])
        
        if is_main_process():
            print(f"Distributed initialized: {world_size} GPUs")
        
        return rank, world_size, local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_gpu_memory_stats():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    i = local_rank if local_rank >= 0 else 0
    
    if i < torch.cuda.device_count():
        free_b, total_b = torch.cuda.mem_get_info(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        peak = torch.cuda.max_memory_allocated(i) / 1024**3
        return {
            f"gpu_{i}/free_gb": free_b / 1024**3,
            f"gpu_{i}/allocated_gb": allocated,
            f"gpu_{i}/peak_gb": peak,
        }
    return {}


def log_memory(stage: str):
    """Log memory stats."""
    stats = get_gpu_memory_stats()
    if stats:
        rank = get_rank()
        for k, v in stats.items():
            print(f"[Rank {rank}] {stage}: {k}={v:.2f} GB")


def run_worker(
    out_dir: str,
    length: int,
    num_designs: int,
    symmetry: str = None,
    mpnn_batch_size: int = 5,
    attention_parallel_factor: int = None,
    use_wandb: bool = False,
    wandb_project: str = "fast-rfd3-parallel",
    wandb_run_name: str = None,
    seed: int = 42,
    **kwargs
):
    """Worker function that runs on each GPU."""
    # Set random seed BEFORE any random operations (including distributed setup)
    set_seed(seed)
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Debug: verify GPU assignment
    current_device = torch.cuda.current_device()
    print(f"[Rank {rank}] GPU assignment: local_rank={local_rank}, current_device={current_device}, "
          f"device_count={torch.cuda.device_count()}")
    
    # Set env vars BEFORE imports (modules check at import time)
    # In parallel mode, we also need LOW_MEMORY_MODE for encoder's chunked P_LL
    os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
    if world_size > 1:
        os.environ["RFD3_ATTENTION_PARALLEL"] = str(world_size)
    
    # Add paths for local imports
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root / "models/rfd3/src"))
    sys.path.insert(0, str(project_root / "models/mpnn/src"))
    sys.path.insert(0, str(project_root / "models/rf3/src"))
    sys.path.insert(0, str(project_root / "src"))
    
    # Import after env vars and paths are set
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import get_chains
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    
    # Optional W&B
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
    
    # Setup output (rank 0 only)
    out_path = Path(out_dir)
    if is_main_process():
        out_path.mkdir(parents=True, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()
    
    # W&B (rank 0 only)
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE and is_main_process():
        run_name = wandb_run_name or f"parallel_L{length}_N{num_designs}_GPUs{world_size}"
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "length": length,
                "num_designs": num_designs,
                "world_size": world_size,
                "symmetry": symmetry,
            }
        )
    
    # Print config
    print_rank0(f"\n{'='*60}")
    print_rank0(f"Parallel RFDiffusion3 Design")
    print_rank0(f"{'='*60}")
    print_rank0(f"  Length: {length}")
    print_rank0(f"  Num designs: {num_designs}")
    print_rank0(f"  GPUs: {world_size}")
    print_rank0(f"  Output: {out_path.resolve()}")
    if symmetry:
        print_rank0(f"  Symmetry: {symmetry}")
    
    # Configure RFD3
    rfd3_spec = {
        'length': length,
        'extra': {}  # Avoid KeyError in engine
    }
    rfd3_inference_sampler = {}
    
    if symmetry:
        rfd3_spec['symmetry'] = {'id': symmetry, 'is_symmetric_motif': False}
        rfd3_inference_sampler = {"kind": "symmetry"}
    
    # Path to local checkpoint
    ckpt_path = project_root / "ckpt" / "rfd3_latest.ckpt"
    if not ckpt_path.exists():
        print_rank0(f"Warning: Checkpoint not found at {ckpt_path}. Falling back to default 'rfd3' lookup.")
        ckpt_path = "rfd3"
    else:
        ckpt_path = str(ckpt_path)
    
    # In parallel mode, also enable low_memory_mode for chunked P_LL in encoder
    # (decoder has parallel path, but encoder still needs chunked embedder)
    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=num_designs,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=True,   # Required for encoder chunked P_LL
        attention_parallel=True,
        attention_parallel_factor=attention_parallel_factor,
    )
    
    # Initialize and run RFD3
    print_rank0("Initializing RFD3 engine...")
    rfd3_engine = RFD3InferenceEngine(**rfd3_config)
    log_memory("rfd3_init")
    
    if dist.is_initialized():
        dist.barrier()
    
    print_rank0("Running RFD3 inference...")
    rfd3_start = time.time()
    rfd3_outputs = rfd3_engine.run(inputs=None, out_dir=None, n_batches=1)
    rfd3_time = time.time() - rfd3_start
    
    log_memory("rfd3_inference")
    
    if dist.is_initialized():
        dist.barrier()
    
    print_rank0(f"  RFD3 took {rfd3_time:.2f}s")
    
    if wandb_run:
        wandb_run.log({"rfd3/inference_time_s": rfd3_time})
    
    # MPNN (rank 0 only)
    if is_main_process():
        print("\nInitializing MPNN Engine...")
        
        mpnn_ckpt = project_root / "ckpt" / "ligandmpnn_v_32_010_25.pt"
        mpnn_ckpt = str(mpnn_ckpt) if mpnn_ckpt.exists() else None
        
        mpnn_engine = MPNNInferenceEngine(
            model_type="ligand_mpnn",
            checkpoint_path=mpnn_ckpt,
            is_legacy_weights=True,
            out_directory=None,
            write_structures=False,
            write_fasta=False,
        )
        log_memory("mpnn_init")
        
        total_mpnn_time = 0.0
        for batch_id, output_list in rfd3_outputs.items():
            for i, rfd3_out in enumerate(output_list):
                name = f"design_{batch_id}_{i}"
                print(f"Processing {name}...")
                
                to_cif_file(rfd3_out.atom_array, str(out_path / f"{name}_backbone.cif"))
                
                mpnn_config = {"batch_size": mpnn_batch_size, "name": name}
                if symmetry:
                    chains = sorted(list(set(get_chains(rfd3_out.atom_array))))
                    mpnn_config["homo_oligomer_chains"] = [chains]
                
                mpnn_start = time.time()
                results = mpnn_engine.run(input_dicts=[mpnn_config], atom_arrays=[rfd3_out.atom_array])
                mpnn_time = time.time() - mpnn_start
                total_mpnn_time += mpnn_time
                print(f"  MPNN took {mpnn_time:.2f}s")
                
                for j, out in enumerate(results):
                    out.write_structure(base_path=str(out_path / f"{name}_seq_{j}"), file_type="cif")
        
        if wandb_run:
            wandb_run.log({"mpnn/total_time_s": total_mpnn_time})
            wandb_run.finish()
        
        print(f"\nDone! Outputs saved to {out_path.resolve()}")
    
    cleanup_distributed()


def launch_with_torchrun(args, params, n_gpus):
    """Launch the script with torchrun for multi-GPU."""
    import shutil
    import random
    
    script_path = Path(__file__).resolve()
    
    # Build command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus}",
        f"--master_port={random.randint(29500, 29999)}",
        str(script_path),
        "--launch_mode", "worker",  # Signal that we're a worker
    ]
    
    # Pass config if provided
    if args.config:
        cmd.extend(["--config", args.config])
    
    # Pass CLI overrides
    for key, value in vars(args).items():
        if key in ('config', 'launch_mode'):
            continue
        if value is not None:
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
    
    print(f"Launching with torchrun: {n_gpus} GPUs")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Execute
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Multi-GPU RFDiffusion3 Design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE:
======

1. Allocate GPUs:
   srun --gpus=4 --time=03:00:00 --pty /bin/bash --login

2. Run script:
   python scripts/design_parallel.py --config config/design_parallel.yaml

EXAMPLES:
=========

# With config:
python scripts/design_parallel.py --config config/design_parallel.yaml

# Override length:
python scripts/design_parallel.py --config config/design_parallel.yaml --length 1000

# Direct torchrun (alternative):
torchrun --nproc_per_node=4 scripts/design_parallel.py --config config/design_parallel.yaml
        """
    )
    
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--launch_mode", type=str, default="auto",
                        choices=["auto", "worker"],
                        help="Launch mode (auto=detect, worker=run as worker)")
    
    # Design parameters
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--num_designs", type=int, default=None)
    parser.add_argument("--symmetry", type=str, default=None)
    parser.add_argument("--mpnn_batch_size", type=int, default=None)
    parser.add_argument("--attention_parallel_factor", type=int, default=None)
    
    # W&B
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: 42, or from config file)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
        params = merge_config_with_args(config, args)
        # Ensure seed has a default value (config takes precedence, then CLI, then 42)
        if 'seed' not in params or params.get('seed') is None:
            params['seed'] = args.seed if args.seed is not None else 42
    else:
        # Defaults match design_annotate.py
        params = {
            'out_dir': args.out_dir or "inference_outputs/design_demo",
            'length': args.length or 100,
            'num_designs': args.num_designs or 1,
            'symmetry': args.symmetry,  # None = no symmetry
            'mpnn_batch_size': args.mpnn_batch_size or 5,
            'attention_parallel_factor': args.attention_parallel_factor,
            'use_wandb': args.use_wandb or False,
            'wandb_project': args.wandb_project or "fast-rfd3",
            'wandb_run_name': args.wandb_run_name,
            'seed': args.seed,
        }
    
    # Check if we're already in distributed mode (launched by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if args.launch_mode == "worker" or local_rank >= 0:
        # We're a worker process - run directly
        run_worker(**params)
    else:
        # We need to launch with torchrun
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        if n_gpus > 1:
            launch_with_torchrun(args, params, n_gpus)
        else:
            print("Only 1 GPU available - running in single-GPU mode")
            run_worker(**params)


if __name__ == "__main__":
    main()

