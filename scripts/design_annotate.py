#!/usr/bin/env python3
"""
RFDiffusion3 Design + MPNN Annotation Script

Supports:
- Standard inference (single GPU)
- Low memory mode (sequential chunked processing)
- Attention parallel mode (multi-GPU, requires srun allocation)
- Optional symmetry constraints

USAGE:
======

# With config file:
python scripts/design_annotate.py --config config/design_annotate.yaml

# Override config values:
python scripts/design_annotate.py --config config/design_annotate.yaml --length 200

# Low memory mode:
python scripts/design_annotate.py --config config/design_low_memory.yaml

# Command line only (no config):
python scripts/design_annotate.py --length 100 --num_designs 4 --out_dir outputs/
"""

# Set environment variables BEFORE any imports
import os
os.environ.setdefault('CCD_MIRROR_PATH', '')
os.environ.setdefault('PDB_MIRROR_PATH', '')

import sys
import argparse
from pathlib import Path
import time
import threading
import yaml

import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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
    
    # Override with CLI arguments (if provided and not None)
    for key, value in vars(args).items():
        if key == 'config':
            continue
        if value is not None:
            result[key] = value
    
    return result


# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GPUMemoryMonitor:
    """Background thread that continuously monitors GPU memory usage."""
    
    def __init__(self, wandb_run=None, interval=1.0):
        self.wandb_run = wandb_run
        self.interval = interval
        self.running = False
        self.thread = None
        self.current_stage = "init"
        self.start_time = None
    
    def _get_gpu_stats(self):
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(i)
            free_gb = free_b / 1024**3
            total_gb = total_b / 1024**3
            
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            
            stats[f"gpu_{i}/global_free_gb"] = free_gb
            stats[f"gpu_{i}/allocated_gb"] = allocated
            stats[f"gpu_{i}/peak_allocated_gb"] = peak_allocated
        
        return stats
    
    def _monitor_loop(self):
        while self.running:
            stats = self._get_gpu_stats()
            if stats and self.wandb_run is not None:
                log_data = {f"monitor/{k}": v for k, v in stats.items()}
                self.wandb_run.log(log_data)
            time.sleep(self.interval)
    
    def start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"  GPU memory monitor started (interval: {self.interval}s)")
    
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("  GPU memory monitor stopped")


def get_gpu_memory_stats():
    """Get current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gb = free_b / 1024**3
        
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        
        stats[f"gpu_{i}/global_free_gb"] = free_gb
        stats[f"gpu_{i}/allocated_gb"] = allocated
        stats[f"gpu_{i}/peak_allocated_gb"] = peak_allocated
    
    return stats


def log_memory(stage: str, wandb_run=None, monitor=None):
    """Log memory stats for a given stage."""
    stats = get_gpu_memory_stats()
    if stats:
        for k, v in stats.items():
            if "free" in k or "peak" in k:
                print(f"  [{stage}] {k}: {v:.2f} GB")
        
        if wandb_run is not None:
            log_data = {f"stage/{stage}/{k}": v for k, v in stats.items()}
            wandb_run.log(log_data)
    
    return stats


def run_design(
    out_dir: str = "inference_outputs/design_demo",
    length: int = 100,
    num_designs: int = 1,
    symmetry: str = None,
    mpnn_batch_size: int = 5,
    low_memory_mode: bool = False,
    attention_parallel: bool = False,
    attention_parallel_factor: int = None,
    use_wandb: bool = False,
    wandb_project: str = "fast-rfd3",
    wandb_run_name: str = None,
    seed: int = 42,
    **kwargs  # Ignore unknown config keys
):
    """
    Run RFD3 design + MPNN annotation pipeline.
    """
    # Set random seed for reproducibility
    set_seed(seed)
    # Add paths for local imports
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sys.path.insert(0, str(project_root / "models/rfd3/src"))
    sys.path.insert(0, str(project_root / "models/mpnn/src"))
    sys.path.insert(0, str(project_root / "models/rf3/src"))
    sys.path.insert(0, str(project_root / "src"))
    
    # Import here to allow env vars and paths to be set first
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import get_chains
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    
    # Setup output
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # W&B setup
    wandb_run = None
    monitor = None
    
    if use_wandb and WANDB_AVAILABLE:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        mode_str = "parallel" if attention_parallel else ("low_mem" if low_memory_mode else "standard")
        run_name = wandb_run_name or f"{mode_str}_L{length}_N{num_designs}_GPUs{n_gpus}"
        
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "length": length,
                "num_designs": num_designs,
                "symmetry": symmetry,
                "low_memory_mode": low_memory_mode,
                "attention_parallel": attention_parallel,
                "n_gpus": n_gpus,
            }
        )
        monitor = GPUMemoryMonitor(wandb_run=wandb_run, interval=1.0)
        monitor.start()
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"RFDiffusion3 Design + Annotation")
    print(f"{'='*60}")
    print(f"  Length: {length}")
    print(f"  Num designs: {num_designs}")
    print(f"  Output: {out_path.resolve()}")
    
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"  Available GPUs: {n_gpus}")
    
    if attention_parallel:
        factor = attention_parallel_factor or n_gpus
        print(f"  Mode: ATTENTION PARALLEL (factor={factor})")
        if n_gpus <= 1:
            print(f"    -> Falling back to LOW_MEMORY_MODE (only {n_gpus} GPU)")
    elif low_memory_mode:
        print(f"  Mode: LOW MEMORY (sequential chunked)")
    else:
        print(f"  Mode: STANDARD")
    
    if symmetry:
        print(f"  Symmetry: {symmetry}")
    
    # Configure RFD3
    rfd3_batch_size = num_designs
    rfd3_spec = {
        'length': length,
        'extra': {}  # Avoid KeyError in engine
    }
    rfd3_inference_sampler = {}
    
    if symmetry:
        rfd3_spec['symmetry'] = {
            'id': symmetry,
            'is_symmetric_motif': False
        }
        rfd3_inference_sampler = {"kind": "symmetry"}
    
    # Path to local checkpoint
    ckpt_path = project_root / "ckpt" / "rfd3_latest.ckpt"
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint not found at {ckpt_path}. Falling back to default 'rfd3' lookup.")
        ckpt_path = "rfd3"
    else:
        ckpt_path = str(ckpt_path)
    
    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=rfd3_batch_size,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=low_memory_mode,
        attention_parallel=attention_parallel,
        attention_parallel_factor=attention_parallel_factor,
    )
    
    print(f"\nInitializing RFD3 engine...")
    rfd3_engine = RFD3InferenceEngine(**rfd3_config)
    log_memory("rfd3_init", wandb_run, monitor)
    
    print(f"Running RFD3 inference...")
    rfd3_start = time.time()
    rfd3_outputs = rfd3_engine.run(
        inputs=None,
        out_dir=None,
        n_batches=1,
    )
    rfd3_time = time.time() - rfd3_start
    log_memory("rfd3_inference", wandb_run, monitor)
    print(f"  RFD3 inference took {rfd3_time:.2f}s")
    
    if wandb_run:
        wandb_run.log({"rfd3/inference_time_s": rfd3_time})
    
    # Configure MPNN
    print(f"\nInitializing MPNN Engine...")
    
    mpnn_ckpt_path = project_root / "ckpt" / "ligandmpnn_v_32_010_25.pt"
    if not mpnn_ckpt_path.exists():
        mpnn_ckpt_path = None
    else:
        mpnn_ckpt_path = str(mpnn_ckpt_path)
    
    mpnn_engine = MPNNInferenceEngine(
        model_type="ligand_mpnn",
        checkpoint_path=mpnn_ckpt_path,
        is_legacy_weights=True,
        out_directory=None,
        write_structures=False,
        write_fasta=False,
    )
    log_memory("mpnn_init", wandb_run, monitor)
    
    # Process designs
    total_mpnn_time = 0.0
    for batch_id, output_list in rfd3_outputs.items():
        for i, rfd3_out in enumerate(output_list):
            backbone_name = f"design_{batch_id}_{i}"
            print(f"Processing {backbone_name}...")
            
            to_cif_file(rfd3_out.atom_array, str(out_path / f"{backbone_name}_backbone.cif"))
            
            mpnn_input_config = {
                "batch_size": mpnn_batch_size,
                "name": backbone_name
            }
            
            if symmetry:
                chains = get_chains(rfd3_out.atom_array)
                chain_list = sorted(list(set(chains)))
                mpnn_input_config["homo_oligomer_chains"] = [chain_list]
            
            mpnn_start = time.time()
            mpnn_results = mpnn_engine.run(
                input_dicts=[mpnn_input_config],
                atom_arrays=[rfd3_out.atom_array]
            )
            mpnn_time = time.time() - mpnn_start
            total_mpnn_time += mpnn_time
            log_memory(f"mpnn_{backbone_name}", wandb_run, monitor)
            print(f"  MPNN took {mpnn_time:.2f}s")
            
            for j, mpnn_out in enumerate(mpnn_results):
                seq_name = f"{backbone_name}_seq_{j}"
                mpnn_out.write_structure(base_path=str(out_path / seq_name), file_type="cif")
    
    # Cleanup
    if monitor is not None:
        monitor.stop()
    
    final_stats = log_memory("final", wandb_run, monitor)
    if wandb_run:
        wandb_run.log({
            "mpnn/total_inference_time_s": total_mpnn_time,
            "summary/peak_memory_gb": final_stats.get("gpu_0/peak_allocated_gb", 0),
        })
        wandb_run.finish()
    
    print(f"\nDone! Outputs saved to {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="RFDiffusion3 Design + MPNN Annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
=========

# With config file:
python scripts/design_annotate.py --config config/design_annotate.yaml

# Override config values:
python scripts/design_annotate.py --config config/design_annotate.yaml --length 200

# Low memory mode:
python scripts/design_annotate.py --config config/design_low_memory.yaml

# Command line only:
python scripts/design_annotate.py --length 100 --num_designs 4
        """
    )
    
    # Config file argument
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Design parameters (can override config)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--num_designs", type=int, default=None)
    parser.add_argument("--symmetry", type=str, default=None)
    parser.add_argument("--mpnn_batch_size", type=int, default=None)
    
    # Memory modes
    parser.add_argument("--low_memory_mode", action="store_true", default=None)
    parser.add_argument("--attention_parallel", action="store_true", default=None)
    parser.add_argument("--attention_parallel_factor", type=int, default=None)
    
    # W&B
    parser.add_argument("--wandb", dest="use_wandb", action="store_true", default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (default: 42, or from config file)")
    
    args = parser.parse_args()
    
    # Load config and merge with CLI args
    if args.config:
        config = load_config(args.config)
        params = merge_config_with_args(config, args)
        # Ensure seed has a default value (config takes precedence, then CLI, then 42)
        if 'seed' not in params or params.get('seed') is None:
            params['seed'] = args.seed if args.seed is not None else 42
    else:
        # Use CLI args only with defaults (matching design_annotate.py)
        params = {
            'out_dir': args.out_dir or "inference_outputs/design_demo",
            'length': args.length or 100,
            'num_designs': args.num_designs or 1,
            'symmetry': args.symmetry,  # None = no symmetry
            'mpnn_batch_size': args.mpnn_batch_size or 5,
            'low_memory_mode': args.low_memory_mode or False,
            'attention_parallel': args.attention_parallel or False,
            'attention_parallel_factor': args.attention_parallel_factor,
            'use_wandb': args.use_wandb or False,
            'wandb_project': args.wandb_project or "fast-rfd3",
            'wandb_run_name': args.wandb_run_name,
            'seed': args.seed,
        }
    
    run_design(**params)


if __name__ == "__main__":
    main()

