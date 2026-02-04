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

# =============================================================================
# MEMORY OPTIMIZATION: Configure PyTorch's CUDA memory allocator
# expandable_segments:True reduces memory fragmentation by allowing the allocator
# to use expandable memory segments instead of fixed-size blocks. This prevents
# OOM errors that occur when memory is fragmented (lots of "reserved but unallocated"
# memory). Must be set BEFORE importing torch.
# Set BOTH old and new env var names for compatibility across PyTorch versions
# =============================================================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
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
import threading
from lightning.fabric import seed_everything

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.
    Ensures consistent seeds across all distributed processes.
    
    Args:
        seed: Random seed value (default: 42)
    """
    # Use the same seed for all processes (for reproducibility)
    # In distributed mode, all ranks use the same base seed
    seed_everything(seed)
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
          f"device_count={device_count}, local_rank={local_rank}", flush=True)

    if local_rank >= 0 and world_size > 1:
        if local_rank >= device_count:
            raise RuntimeError(
                f"local_rank={local_rank} but only {device_count} GPUs visible! "
                f"CUDA_VISIBLE_DEVICES={cuda_visible}"
            )

        print(f"[Rank {rank}] Setting CUDA device to local_rank={local_rank}", flush=True)
        torch.cuda.set_device(local_rank)

        # Verify the device was set correctly
        current = torch.cuda.current_device()
        print(f"[Rank {rank}] Current CUDA device after set_device: {current}", flush=True)

        # Initialize distributed with explicit device_id matching LOCAL_RANK
        # Using --ntasks-per-node + --gpus-per-node, each task sees all GPUs on the node
        # and LOCAL_RANK determines which GPU this task uses
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        dist.barrier(device_ids=[local_rank])

        if is_main_process():
            print(f"Distributed initialized: {world_size} GPUs", flush=True)

        return rank, world_size, local_rank
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_gpu_memory_stats(gpu_id: int = None):
    """
    Get current GPU memory usage statistics.

    Args:
        gpu_id: GPU device ID to query. If None, uses LOCAL_RANK from env.

    Returns:
        Dict of memory stats for the specified GPU.
    """
    if not torch.cuda.is_available():
        return {}

    if gpu_id is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        gpu_id = local_rank if local_rank >= 0 else 0

    if gpu_id >= torch.cuda.device_count():
        return {}

    # Use global rank for consistent naming across processes
    rank = get_rank()

    # Global (CUDA) view — includes other processes
    free_b, total_b = torch.cuda.mem_get_info(gpu_id)
    free_gb = free_b / 1024**3
    total_gb = total_b / 1024**3
    used_gb = total_gb - free_gb

    # PyTorch (this process) view
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
    peak_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3

    # cached = reserved - allocated (PyTorch's internal cache, but FRAGMENTED!)
    cached = max(reserved - allocated, 0.0)

    return {
        f"gpu_{rank}/global_total_gb": total_gb,
        f"gpu_{rank}/global_free_gb": free_gb,      # KEY: OOM when single alloc > this!
        f"gpu_{rank}/global_used_gb": used_gb,
        f"gpu_{rank}/allocated_gb": allocated,
        f"gpu_{rank}/reserved_gb": reserved,
        f"gpu_{rank}/cached_gb": cached,            # WARNING: fragmented, can't use for large allocs
        f"gpu_{rank}/peak_allocated_gb": peak_allocated,
    }


def gather_all_gpu_stats():
    """
    Gather GPU memory stats from all ranks to rank 0.

    Returns:
        On rank 0: combined dict of stats from all GPUs
        On other ranks: local stats only
    """
    local_stats = get_gpu_memory_stats()

    if not dist.is_initialized():
        return local_stats

    world_size = get_world_size()
    rank = get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Gather stats from all ranks
    # Must set device context for NCCL to work properly
    gathered = [None] * world_size
    with torch.cuda.device(local_rank):
        dist.all_gather_object(gathered, local_stats)

    # Combine all stats on rank 0
    if rank == 0:
        combined = {}
        for stats in gathered:
            if stats:
                combined.update(stats)
        return combined

    return local_stats


class GPUMemoryMonitor:
    """
    Background thread that continuously monitors GPU memory usage.

    In distributed mode:
    - Only rank 0 runs the monitor and logs to W&B
    - Rank 0 monitors all visible GPUs directly (no distributed gather needed)
    - Stage-based logging (log_memory) still gathers from all ranks at sync points
    """

    def __init__(self, wandb_run=None, interval=1.0):
        """
        Args:
            wandb_run: W&B run object for logging (should only be set on rank 0)
            interval: Sampling interval in seconds
        """
        self.wandb_run = wandb_run
        self.interval = interval
        self.running = False
        self.thread = None
        self.current_stage = "init"
        self.start_time = None

    def _get_all_gpu_stats(self):
        """
        Get memory stats for all visible GPUs on this node.
        Called only from rank 0's monitor thread.
        """
        if not torch.cuda.is_available():
            return {}

        stats = {}
        for i in range(torch.cuda.device_count()):
            # Global (CUDA) view — includes other processes
            free_b, total_b = torch.cuda.mem_get_info(i)
            free_gb = free_b / 1024**3
            total_gb = total_b / 1024**3
            used_gb = total_gb - free_gb

            # PyTorch view - note: only accurate for current process's device
            # For other devices, we show global stats only
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            cached = max(reserved - allocated, 0.0)

            stats[f"gpu_{i}/global_total_gb"] = total_gb
            stats[f"gpu_{i}/global_free_gb"] = free_gb
            stats[f"gpu_{i}/global_used_gb"] = used_gb
            stats[f"gpu_{i}/allocated_gb"] = allocated
            stats[f"gpu_{i}/reserved_gb"] = reserved
            stats[f"gpu_{i}/cached_gb"] = cached
            stats[f"gpu_{i}/peak_allocated_gb"] = peak_allocated

        return stats

    def _monitor_loop(self):
        """Main monitoring loop - runs only on rank 0."""
        while self.running:
            # Get stats for all GPUs visible on this node
            stats = self._get_all_gpu_stats()

            if stats and self.wandb_run is not None:
                log_data = {}
                for k, v in stats.items():
                    log_data[f"monitor/{k}"] = v
                log_data["monitor/stage"] = self.current_stage

                self.wandb_run.log(log_data)

            time.sleep(self.interval)

    def start(self):
        """Start the monitoring thread (only on rank 0)."""
        if self.running:
            return
        # Only start on rank 0
        if not is_main_process():
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"  GPU memory monitor started (interval: {self.interval}s)")

    def stop(self):
        """Stop the monitoring thread."""
        if not self.running:
            return
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if is_main_process():
            print("  GPU memory monitor stopped")

    def set_stage(self, stage: str):
        """Update the current pipeline stage."""
        self.current_stage = stage


def log_memory(stage: str, wandb_run=None, monitor=None, gather: bool = True):
    """
    Log memory stats for a given stage.

    Args:
        stage: Name of the pipeline stage
        wandb_run: W&B run object for logging
        monitor: GPUMemoryMonitor instance
        gather: If True (default), gather stats from all ranks. Set to False when
                only rank 0 is executing (e.g., during MPNN phase) to avoid hangs.
    """
    if gather and dist.is_initialized():
        stats = gather_all_gpu_stats()
    else:
        # Local stats only - use device index for naming when not gathering
        stats = get_gpu_memory_stats()

    if stats:
        rank = get_rank()
        # Show the KEY metric for this rank's GPU
        local_free = stats.get(f'gpu_{rank}/global_free_gb', 0)
        local_allocated = stats.get(f'gpu_{rank}/allocated_gb', 0)
        print(f"[Rank {rank}] [{stage}] GPU: {local_allocated:.1f} GB allocated, {local_free:.1f} GB free")

        # Log to W&B on rank 0 only
        if wandb_run is not None and is_main_process():
            logged_stats = {f"{stage}/{k}": v for k, v in stats.items()}
            wandb_run.log(logged_stats)

        # Update monitor stage
        if monitor is not None:
            monitor.set_stage(stage)

    return stats


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

    # Setup log file for debugging - each rank writes to its own log
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = logs_dir / f"design_parallel_rank{rank}.log"
    log_file = open(log_file_path, "w", buffering=1)  # Line buffered

    # Create a tee-like class to write to both stdout and log file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    # Redirect stdout to tee (both stdout and log file)
    import sys as _sys
    original_stdout = _sys.stdout
    _sys.stdout = TeeOutput(original_stdout, log_file)

    print(f"[Rank {rank}] Log file: {log_file_path}")

    # Reset peak memory stats on all ranks
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # W&B (rank 0 only)
    wandb_run = None
    monitor = None
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
                "pipeline": "design_parallel",
            }
        )
        print(f"W&B run initialized: {wandb_run.url}")

    # Start GPU memory monitor (only on rank 0)
    if use_wandb and WANDB_AVAILABLE and is_main_process():
        monitor = GPUMemoryMonitor(wandb_run=wandb_run, interval=1.0)
        monitor.start()
    
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
        seed=seed,
    )
    
    # Initialize and run RFD3
    print_rank0("Initializing RFD3 engine...")
    rfd3_engine = RFD3InferenceEngine(**rfd3_config)
    log_memory("rfd3_init", wandb_run, monitor)

    if dist.is_initialized():
        dist.barrier()

    print_rank0("Running RFD3 inference...")
    rfd3_start = time.time()
    rfd3_outputs = rfd3_engine.run(inputs=None, out_dir=None, n_batches=1)
    rfd3_time = time.time() - rfd3_start

    log_memory("rfd3_inference", wandb_run, monitor)

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
        # gather=False because only rank 0 is in this block
        log_memory("mpnn_init", wandb_run, monitor, gather=False)

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
                # gather=False because only rank 0 is in this block
                log_memory(f"mpnn_{name}", wandb_run, monitor, gather=False)
                print(f"  MPNN took {mpnn_time:.2f}s")

                for j, out in enumerate(results):
                    out.write_structure(base_path=str(out_path / f"{name}_seq_{j}"), file_type="cif")

        # Stop monitor and log final summary
        if monitor is not None:
            monitor.stop()

        # gather=False because only rank 0 is in this block
        final_stats = log_memory("final", wandb_run, monitor, gather=False)
        if wandb_run:
            wandb_run.log({
                "mpnn/total_time_s": total_mpnn_time,
                "summary/peak_memory_gb": final_stats.get("gpu_0/peak_allocated_gb", 0),
            })
            wandb_run.finish()
            print(f"W&B run completed.")

        print(f"\nDone! Outputs saved to {out_path.resolve()}")

    # Close log file and restore stdout
    print(f"[Rank {rank}] Closing log file: {log_file_path}")
    _sys.stdout = original_stdout
    log_file.close()

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

