#!/usr/bin/env python3
"""
Diagnostic Compare Script - Tracks RNG state to identify divergence point

This script runs the same comparison as compare_parallel.py but adds
RNG state checkpoints to pinpoint where standard and parallel modes diverge.

Usage:
    python scripts/compare_parallel_diagnostic.py --config config/compare_parallel.yaml --num-gpus 2

Output is logged to: logs/compare_parallel_diagnostic.log (overwritten each run)
"""

import os
os.environ.setdefault('CCD_MIRROR_PATH', '')
os.environ.setdefault('PDB_MIRROR_PATH', '')

import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
import yaml
import torch
import torch.distributed as dist
import random
import numpy as np
import pickle
import hashlib


class TeeLogger:
    """Write output to both stdout and a file."""

    def __init__(self, log_file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def rng_state_hash() -> str:
    """Get hash of current RNG state (CPU and CUDA)."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else torch.tensor([])
    combined = cpu_state.numpy().tobytes() + cuda_state.numpy().tobytes()
    return hashlib.md5(combined).hexdigest()[:12]


def rng_sample(n: int = 3) -> list:
    """Sample random numbers WITHOUT advancing state (saves/restores)."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    samples = torch.randn(n).tolist()
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    return samples


def log_rng(checkpoint_name: str, mode: str = ""):
    """Log RNG state at a checkpoint."""
    h = rng_state_hash()
    samples = rng_sample(3)
    samples_str = ", ".join(f"{s:.4f}" for s in samples)
    print(f"[RNG-{mode}] {checkpoint_name}: hash={h}, next_samples=[{samples_str}]", flush=True)
    return h


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_distributed():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def setup_imports(project_root: Path):
    sys.path.insert(0, str(project_root / "models/rfd3/src"))
    sys.path.insert(0, str(project_root / "models/mpnn/src"))
    sys.path.insert(0, str(project_root / "models/rf3/src"))
    sys.path.insert(0, str(project_root / "src"))


def get_checkpoint_path(project_root: Path) -> str:
    ckpt_path = project_root / "ckpt" / "rfd3_latest.ckpt"
    if not ckpt_path.exists():
        return "rfd3"
    return str(ckpt_path)


def build_rfd3_spec(length: int, symmetry: str = None):
    rfd3_spec = {'length': length, 'extra': {}}
    rfd3_inference_sampler = {}
    if symmetry:
        rfd3_spec['symmetry'] = {'id': symmetry, 'is_symmetric_motif': False}
        rfd3_inference_sampler = {"kind": "symmetry"}
    return rfd3_spec, rfd3_inference_sampler


def run_standard_mode(project_root: Path, rfd3_spec: dict, rfd3_inference_sampler: dict,
                       ckpt_path: str, seed: int) -> tuple:
    """Run standard mode with RNG checkpoints."""
    MODE = "STANDARD"
    checkpoints = {}

    print(f"\n{'='*60}")
    print(f"STANDARD MODE - RNG Diagnostic")
    print(f"{'='*60}")

    # Checkpoint 1: Before anything
    print(f"\n[{MODE}] Step 1: Setting initial seed={seed}")
    set_seed(seed)
    checkpoints["01_after_initial_seed"] = log_rng("01_after_initial_seed", MODE)

    # Checkpoint 2: Before imports
    print(f"\n[{MODE}] Step 2: Before module imports")
    checkpoints["02_before_imports"] = log_rng("02_before_imports", MODE)

    # Import modules
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.model.debug_context import debug_ctx

    # Checkpoint 3: After imports
    print(f"\n[{MODE}] Step 3: After module imports")
    checkpoints["03_after_imports"] = log_rng("03_after_imports", MODE)

    os.environ["RFD3_ATTENTION_PARALLEL"] = "0"
    os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
    debug_ctx.set_mode("STANDARD")

    # Checkpoint 4: Before engine creation
    print(f"\n[{MODE}] Step 4: Before engine creation")
    checkpoints["04_before_engine"] = log_rng("04_before_engine", MODE)

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=1,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=True,
        attention_parallel=False,
    )
    engine = RFD3InferenceEngine(**rfd3_config)

    # Checkpoint 5: After engine creation
    print(f"\n[{MODE}] Step 5: After engine creation")
    checkpoints["05_after_engine"] = log_rng("05_after_engine", MODE)

    # Checkpoint 6: Re-seeding
    print(f"\n[{MODE}] Step 6: Re-setting seed={seed}")
    set_seed(seed)
    checkpoints["06_after_reseed"] = log_rng("06_after_reseed", MODE)

    # Checkpoint 7: Before engine.run()
    print(f"\n[{MODE}] Step 7: Before engine.run()")
    checkpoints["07_before_run"] = log_rng("07_before_run", MODE)

    print(f"\n[{MODE}] Running inference...")
    with torch.no_grad():
        outputs = engine.run(inputs=None, out_dir=None, n_batches=1)

    # Checkpoint 8: After inference
    print(f"\n[{MODE}] Step 8: After engine.run()")
    checkpoints["08_after_run"] = log_rng("08_after_run", MODE)

    # Extract coordinates
    X = None
    if outputs and isinstance(outputs, dict):
        for example_id, output_list in outputs.items():
            if isinstance(output_list, list) and len(output_list) > 0:
                rfd3_output = output_list[0]
                if hasattr(rfd3_output, 'atom_array') and hasattr(rfd3_output.atom_array, 'coord'):
                    X = torch.from_numpy(rfd3_output.atom_array.coord).float()
                    break

    return X, checkpoints


def run_parallel_mode_distributed(project_root: Path, rfd3_spec: dict, rfd3_inference_sampler: dict,
                                   ckpt_path: str, seed: int, output_cache: str) -> int:
    """Run parallel mode with RNG checkpoints."""
    MODE = "PARALLEL"
    checkpoints = {}

    rank, world_size, local_rank = init_distributed()

    def log(msg):
        if is_main_process():
            print(msg, flush=True)

    log(f"\n{'='*60}")
    log(f"PARALLEL MODE - RNG Diagnostic (rank 0)")
    log(f"{'='*60}")

    # Checkpoint 1: Before seeding (after distributed init)
    log(f"\n[{MODE}] Step 1: Setting initial seed={seed}")
    set_seed(seed)
    if is_main_process():
        checkpoints["01_after_initial_seed"] = log_rng("01_after_initial_seed", MODE)

    if dist.is_initialized():
        dist.barrier()

    # Checkpoint 2: Before imports
    log(f"\n[{MODE}] Step 2: Before module imports")
    if is_main_process():
        checkpoints["02_before_imports"] = log_rng("02_before_imports", MODE)

    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.model.debug_context import debug_ctx

    # Checkpoint 3: After imports
    log(f"\n[{MODE}] Step 3: After module imports")
    if is_main_process():
        checkpoints["03_after_imports"] = log_rng("03_after_imports", MODE)

    os.environ["RFD3_ATTENTION_PARALLEL"] = "1"
    os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
    debug_ctx.set_mode("PARALLEL")

    # Checkpoint 4: Before engine creation
    log(f"\n[{MODE}] Step 4: Before engine creation")
    if is_main_process():
        checkpoints["04_before_engine"] = log_rng("04_before_engine", MODE)

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=1,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=True,
        attention_parallel=True,
    )
    engine = RFD3InferenceEngine(**rfd3_config)

    # Checkpoint 5: After engine creation
    log(f"\n[{MODE}] Step 5: After engine creation")
    if is_main_process():
        checkpoints["05_after_engine"] = log_rng("05_after_engine", MODE)

    # Checkpoint 6: Re-seeding
    log(f"\n[{MODE}] Step 6: Re-setting seed={seed}")
    set_seed(seed)
    if is_main_process():
        checkpoints["06_after_reseed"] = log_rng("06_after_reseed", MODE)

    if dist.is_initialized():
        dist.barrier()

    # Checkpoint 7: Before engine.run()
    log(f"\n[{MODE}] Step 7: Before engine.run()")
    if is_main_process():
        checkpoints["07_before_run"] = log_rng("07_before_run", MODE)

    log(f"\n[{MODE}] Running inference...")
    with torch.no_grad():
        outputs = engine.run(inputs=None, out_dir=None, n_batches=1)

    if dist.is_initialized():
        dist.barrier()

    # Checkpoint 8: After inference
    log(f"\n[{MODE}] Step 8: After engine.run()")
    if is_main_process():
        checkpoints["08_after_run"] = log_rng("08_after_run", MODE)

    # Extract coordinates and save (only rank 0)
    X = None
    if is_main_process():
        if outputs and isinstance(outputs, dict):
            for example_id, output_list in outputs.items():
                if isinstance(output_list, list) and len(output_list) > 0:
                    rfd3_output = output_list[0]
                    if hasattr(rfd3_output, 'atom_array') and hasattr(rfd3_output.atom_array, 'coord'):
                        X = torch.from_numpy(rfd3_output.atom_array.coord).float()
                        break

        if output_cache:
            with open(output_cache, 'wb') as f:
                pickle.dump({'X_par': X, 'checkpoints': checkpoints}, f)

    success = 0
    if dist.is_initialized():
        success_val = 1 if (is_main_process() and X is not None) else 0
        success_tensor = torch.tensor([success_val], device=f'cuda:{local_rank}')
        dist.broadcast(success_tensor, src=0)
        success = success_tensor.item()
        dist.barrier()
        dist.destroy_process_group()

    return 0 if success else 1


def run_orchestrator(config: dict, config_path: str, num_gpus: int, log_dir: Path) -> int:
    """Main orchestrator with diagnostic output."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Setup logging to both terminal and file (overwrites each run)
    log_file_path = log_dir / "compare_parallel_diagnostic.log"
    tee = TeeLogger(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        setup_imports(project_root)

        length = config.get('length', 100)
        symmetry = config.get('symmetry')
        seed = config.get('seed', 42)

        print(f"\n{'='*70}")
        print("RFD3 DIAGNOSTIC - RNG State Tracking")
        print(f"{'='*70}")
        print(f"Length: {length}")
        print(f"Symmetry: {symmetry}")
        print(f"Seed: {seed}")
        print(f"Parallel GPUs: {num_gpus}")
        print(f"Log file: {log_file_path}")

        ckpt_path = get_checkpoint_path(project_root)
        rfd3_spec, rfd3_inference_sampler = build_rfd3_spec(length, symmetry)

        # ========================================================================
        # PHASE 1: Standard Mode
        # ========================================================================
        X_std, std_checkpoints = run_standard_mode(
            project_root, rfd3_spec, rfd3_inference_sampler, ckpt_path, seed
        )

        if X_std is None:
            print("ERROR: Failed to get standard mode output")
            return 1

        # ========================================================================
        # PHASE 2: Parallel Mode (via torchrun)
        # ========================================================================
        par_cache = Path(tempfile.gettempdir()) / f"diag_par_{os.getpid()}.pkl"

        cmd = [
            'torchrun',
            f'--nproc_per_node={num_gpus}',
            str(Path(__file__).resolve()),
            '--parallel-subprocess',
            '--output-cache', str(par_cache),
            '--length', str(length),
            '--seed', str(seed),
        ]
        if symmetry:
            cmd.extend(['--symmetry', symmetry])
        if config_path:
            cmd.extend(['--config', config_path])

        print(f"\nLaunching parallel mode: {' '.join(cmd)}")

        # Run torchrun and capture output to log file
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end='')
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: torchrun failed with exit code {e.returncode}")
            return 1

        if not par_cache.exists():
            print(f"ERROR: Parallel output not found at {par_cache}")
            return 1

        with open(par_cache, 'rb') as f:
            par_data = pickle.load(f)
            X_par = par_data['X_par']
            par_checkpoints = par_data.get('checkpoints', {})

        par_cache.unlink(missing_ok=True)

        # ========================================================================
        # PHASE 3: Compare RNG Checkpoints
        # ========================================================================
        print(f"\n{'='*70}")
        print("RNG CHECKPOINT COMPARISON")
        print(f"{'='*70}")
        print(f"{'Checkpoint':<30} {'STANDARD':<15} {'PARALLEL':<15} {'Status'}")
        print("-" * 70)

        all_checkpoints = sorted(set(std_checkpoints.keys()) | set(par_checkpoints.keys()))
        divergence_point = None

        for cp in all_checkpoints:
            std_hash = std_checkpoints.get(cp, "N/A")
            par_hash = par_checkpoints.get(cp, "N/A")
            if std_hash == par_hash:
                status = "MATCH"
            else:
                status = "DIVERGE"
                if divergence_point is None:
                    divergence_point = cp

            print(f"{cp:<30} {std_hash:<15} {par_hash:<15} {status}")

        # ========================================================================
        # PHASE 4: Compare Final Outputs
        # ========================================================================
        print(f"\n{'='*70}")
        print("FINAL OUTPUT COMPARISON")
        print(f"{'='*70}")

        if X_std is not None and X_par is not None:
            diff = (X_std - X_par).abs()
            print(f"  X_std shape: {X_std.shape}")
            print(f"  X_par shape: {X_par.shape}")
            print(f"  Max diff:  {diff.max().item():.6e}")
            print(f"  Mean diff: {diff.mean().item():.6e}")

        # ========================================================================
        # Summary
        # ========================================================================
        print(f"\n{'='*70}")
        print("DIAGNOSIS")
        print(f"{'='*70}")

        if divergence_point:
            print(f"  First divergence at: {divergence_point}")
            if "after_imports" in divergence_point:
                print("  -> Module imports consume different random numbers")
            elif "after_engine" in divergence_point:
                print("  -> Engine initialization consumes different random numbers")
            elif "before_run" in divergence_point:
                print("  -> Re-seeding didn't help; operations between reseed and run() differ")
        else:
            print("  No divergence detected in checkpoints!")
            print("  Issue may be inside engine.run() after checkpoint 07")

        return 0

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"Log saved to: {log_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic Compare Standard vs Parallel RFD3")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--symmetry", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--parallel-subprocess", action="store_true")
    parser.add_argument("--output-cache", type=str, default=None)

    args = parser.parse_args()

    is_subprocess = args.parallel_subprocess or ('RANK' in os.environ)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    log_dir = Path(args.log_dir) if args.log_dir else project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    if args.length is not None:
        config['length'] = args.length
    if args.symmetry is not None:
        config['symmetry'] = args.symmetry
    if args.seed is not None:
        config['seed'] = args.seed

    config.setdefault('length', 100)
    config.setdefault('seed', 42)

    if is_subprocess:
        setup_imports(project_root)
        ckpt_path = get_checkpoint_path(project_root)
        rfd3_spec, rfd3_inference_sampler = build_rfd3_spec(
            config['length'], config.get('symmetry')
        )
        return run_parallel_mode_distributed(
            project_root, rfd3_spec, rfd3_inference_sampler,
            ckpt_path, config['seed'], args.output_cache
        )
    else:
        return run_orchestrator(config, args.config, args.num_gpus, log_dir)


if __name__ == "__main__":
    sys.exit(main())
