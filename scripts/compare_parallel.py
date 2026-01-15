#!/usr/bin/env python3
"""
Compare Standard vs Parallel RFD3 Outputs

This script compares standard (1 GPU) and parallel (N GPUs) RFD3 implementations.

Usage:
    # Run full comparison (standard on 1 GPU, parallel on 2 GPUs):
    python scripts/compare_parallel.py --config config/compare_parallel.yaml --num-gpus 2

    # The script automatically:
    # 1. Runs standard mode on GPU 0
    # 2. Spawns torchrun for parallel mode on N GPUs
    # 3. Compares the outputs
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


class TeeLogger:
    """Write output to both stdout and a file."""

    def __init__(self, log_file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def init_distributed():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Set CUDA device for this rank
        torch.cuda.set_device(local_rank)

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

        return rank, world_size, local_rank
    return 0, 1, 0


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Reset the default generator to ensure identical state
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.manual_seed(seed)
        torch.cuda.set_device(0)  # Reset to default


def compare_tensors(name: str, t1: torch.Tensor, t2: torch.Tensor, atol: float = 1e-5) -> dict:
    """Compare two tensors and return detailed statistics."""
    if t1 is None or t2 is None:
        return {"name": name, "error": "One or both tensors are None"}

    if t1.shape != t2.shape:
        return {
            "name": name,
            "error": f"Shape mismatch: {t1.shape} vs {t2.shape}"
        }

    t1 = t1.float()
    t2 = t2.float()
    diff = (t1 - t2).abs()

    result = {
        "name": name,
        "shape": list(t1.shape),
        "t1_mean": t1.mean().item(),
        "t1_std": t1.std().item(),
        "t2_mean": t2.mean().item(),
        "t2_std": t2.std().item(),
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "median_diff": diff.median().item(),
        "pct_close": (diff < atol).float().mean().item() * 100,
    }

    if result["max_diff"] < atol:
        result["status"] = "MATCH"
    elif result["max_diff"] < atol * 100:
        result["status"] = "CLOSE"
    else:
        result["status"] = "DIVERGE"

    return result


def print_comparison(result: dict):
    """Pretty print comparison result."""
    if "error" in result:
        print(f"  {result['name']}: ERROR - {result['error']}")
        return

    status = result['status']
    name = result['name']
    max_diff = result['max_diff']
    mean_diff = result['mean_diff']
    pct_close = result['pct_close']

    print(f"  [{status}] {name}")
    print(f"      shape: {result['shape']}")
    print(f"      standard: mean={result['t1_mean']:.6f}, std={result['t1_std']:.6f}")
    print(f"      parallel: mean={result['t2_mean']:.6f}, std={result['t2_std']:.6f}")
    print(f"      diff: max={max_diff:.2e}, mean={mean_diff:.2e}, {pct_close:.1f}% within tol")


def setup_imports(project_root: Path):
    """Setup import paths for local modules."""
    sys.path.insert(0, str(project_root / "models/rfd3/src"))
    sys.path.insert(0, str(project_root / "models/mpnn/src"))
    sys.path.insert(0, str(project_root / "models/rf3/src"))
    sys.path.insert(0, str(project_root / "src"))


def get_checkpoint_path(project_root: Path) -> str:
    """Get path to model checkpoint."""
    ckpt_path = project_root / "ckpt" / "rfd3_latest.ckpt"
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return "rfd3"
    return str(ckpt_path)


def build_rfd3_spec(length: int, symmetry: str = None):
    """Build RFD3 specification dict."""
    rfd3_spec = {'length': length, 'extra': {}}
    rfd3_inference_sampler = {}

    if symmetry:
        rfd3_spec['symmetry'] = {'id': symmetry, 'is_symmetric_motif': False}
        rfd3_inference_sampler = {"kind": "symmetry"}

    return rfd3_spec, rfd3_inference_sampler


def run_standard_mode(project_root: Path, rfd3_spec: dict, rfd3_inference_sampler: dict,
                       ckpt_path: str, seed: int) -> torch.Tensor:
    """Run standard (1 GPU) inference and return coordinates."""
    # CRITICAL: Set seed BEFORE any rfd3 imports to ensure deterministic behavior
    # Module imports can consume random numbers during initialization
    print(f"  Setting seed={seed} BEFORE imports...")
    set_seed(seed)

    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.model.debug_context import debug_ctx

    os.environ["RFD3_ATTENTION_PARALLEL"] = "0"
    os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
    debug_ctx.set_mode("STANDARD")
    debug_ctx.set_step(-1)

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=1,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=True,
        attention_parallel=False,
    )

    print("  Initializing standard RFD3 engine...")
    engine = RFD3InferenceEngine(**rfd3_config)

    # Set seed AGAIN after engine init to ensure diffusion sampling is identical
    # (engine init may consume random numbers non-deterministically)
    print(f"  Re-setting seed={seed} after engine init...")
    set_seed(seed)

    print("  Running standard forward pass...")
    with torch.no_grad():
        outputs = engine.run(inputs=None, out_dir=None, n_batches=1)

    # Extract coordinates
    X = None
    if outputs and isinstance(outputs, dict):
        for example_id, output_list in outputs.items():
            if isinstance(output_list, list) and len(output_list) > 0:
                rfd3_output = output_list[0]
                if hasattr(rfd3_output, 'atom_array') and hasattr(rfd3_output.atom_array, 'coord'):
                    X = torch.from_numpy(rfd3_output.atom_array.coord).float()
                    print(f"  Extracted X_std: shape {X.shape}")
                    break

    return X


def run_parallel_mode_distributed(project_root: Path, rfd3_spec: dict, rfd3_inference_sampler: dict,
                                   ckpt_path: str, seed: int, output_cache: str) -> int:
    """Run parallel (N GPU) inference in distributed mode and save results."""
    # CRITICAL: Set seed BEFORE any rfd3 imports to ensure deterministic behavior
    # Module imports can consume random numbers during initialization
    # NOTE: set_seed uses torch which is already imported at module level
    set_seed(seed)

    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from rfd3.model.debug_context import debug_ctx

    rank, world_size, local_rank = init_distributed()

    def log(msg):
        if is_main_process():
            print(msg)

    log(f"  Seed={seed} set BEFORE imports (all ranks)")

    # Synchronize after seeding to ensure all ranks start identically
    if dist.is_initialized():
        dist.barrier()

    log(f"  Parallel mode: rank={rank}, world_size={world_size}")

    os.environ["RFD3_ATTENTION_PARALLEL"] = "1"
    os.environ["RFD3_LOW_MEMORY_MODE"] = "1"
    debug_ctx.set_mode("PARALLEL")
    debug_ctx.set_step(-1)

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=1,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=True,
        attention_parallel=True,
    )

    log("  Initializing parallel RFD3 engine...")
    engine = RFD3InferenceEngine(**rfd3_config)

    # Set seed AGAIN after engine init to ensure diffusion sampling is identical
    log(f"  Re-setting seed={seed} after engine init (all ranks)...")
    set_seed(seed)

    # Synchronize after seeding to ensure all ranks are ready
    if dist.is_initialized():
        dist.barrier()

    log("  Running parallel forward pass...")
    with torch.no_grad():
        outputs = engine.run(inputs=None, out_dir=None, n_batches=1)

    # Synchronize after run
    if dist.is_initialized():
        dist.barrier()

    # Extract coordinates (only on rank 0)
    X = None
    if is_main_process():
        if outputs and isinstance(outputs, dict):
            for example_id, output_list in outputs.items():
                if isinstance(output_list, list) and len(output_list) > 0:
                    rfd3_output = output_list[0]
                    if hasattr(rfd3_output, 'atom_array') and hasattr(rfd3_output.atom_array, 'coord'):
                        X = torch.from_numpy(rfd3_output.atom_array.coord).float()
                        log(f"  Extracted X_par: shape {X.shape}")
                        break

        # Save to output cache
        if X is not None and output_cache:
            log(f"  Saving parallel output to: {output_cache}")
            with open(output_cache, 'wb') as f:
                pickle.dump({'X_par': X}, f)

    # Determine success on rank 0, broadcast to all ranks
    success = 0
    if dist.is_initialized():
        # Rank 0 determines success (did we extract coordinates?)
        success_val = 1 if (is_main_process() and X is not None) else 0
        success_tensor = torch.tensor([success_val], device=f'cuda:{local_rank}')
        dist.broadcast(success_tensor, src=0)
        success = success_tensor.item()

        # Clean up distributed
        dist.barrier()
        dist.destroy_process_group()

    return 0 if success else 1


def run_orchestrator(config: dict, config_path: str, num_gpus: int, log_dir: Path) -> int:
    """
    Main orchestrator: runs standard mode, spawns torchrun for parallel, compares results.
    All output is written to both terminal and log file.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Setup logging to both terminal and file
    log_file_path = log_dir / "compare_parallel.log"
    tee = TeeLogger(log_file_path)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        setup_imports(project_root)

        length = config.get('length', 100)
        symmetry = config.get('symmetry')
        seed = config.get('seed', 42)

        print(f"\n{'='*70}")
        print("RFD3 Standard vs Parallel Comparison (Unified)")
        print(f"{'='*70}")
        print(f"Length: {length}")
        print(f"Symmetry: {symmetry}")
        print(f"Seed: {seed}")
        print(f"Parallel GPUs: {num_gpus}")
        print(f"Log file: {log_file_path}")
        print("")

        ckpt_path = get_checkpoint_path(project_root)
        rfd3_spec, rfd3_inference_sampler = build_rfd3_spec(length, symmetry)

        # ========================================================================
        # PHASE 1: Run Standard Mode (1 GPU)
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 1: STANDARD mode (1 GPU)")
        print("="*70)

        X_std = run_standard_mode(project_root, rfd3_spec, rfd3_inference_sampler, ckpt_path, seed)

        if X_std is None:
            print("  ERROR: Failed to get standard mode output")
            return 1

        # Save standard output to temp file
        std_cache = Path(tempfile.gettempdir()) / f"compare_std_{os.getpid()}.pkl"
        with open(std_cache, 'wb') as f:
            pickle.dump({'X_std': X_std}, f)

        # ========================================================================
        # PHASE 2: Run Parallel Mode (N GPUs via torchrun)
        # ========================================================================
        print("\n" + "="*70)
        print(f"PHASE 2: PARALLEL mode ({num_gpus} GPUs via torchrun)")
        print("="*70)

        par_cache = Path(tempfile.gettempdir()) / f"compare_par_{os.getpid()}.pkl"

        # Build torchrun command
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

        print(f"  Running: {' '.join(cmd)}")

        # Run torchrun and stream output to both terminal and log file
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )
            # Stream output line by line to tee logger
            for line in process.stdout:
                print(line, end='')  # Already has newline
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            print("  Parallel mode completed.")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: torchrun failed with exit code {e.returncode}")
            return 1

        # Load parallel output
        if not par_cache.exists():
            print(f"  ERROR: Parallel output not found at {par_cache}")
            return 1

        with open(par_cache, 'rb') as f:
            X_par = pickle.load(f)['X_par']

        print(f"  Loaded X_par: shape {X_par.shape}")

        # Clean up temp files
        std_cache.unlink(missing_ok=True)
        par_cache.unlink(missing_ok=True)

        # ========================================================================
        # PHASE 3: Compare Results
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 3: Comparing Results")
        print("="*70)

        results = []

        if X_std is not None and X_par is not None:
            result = compare_tensors("atom_array.coord (final output)", X_std, X_par)
            results.append(result)
            print_comparison(result)
        else:
            print("  ERROR: Could not extract coordinates from one or both runs")
            print(f"    X_std is None: {X_std is None}")
            print(f"    X_par is None: {X_par is None}")
            return 1

        # ========================================================================
        # Summary
        # ========================================================================
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        n_match = sum(1 for r in results if r.get("status") == "MATCH")
        n_close = sum(1 for r in results if r.get("status") == "CLOSE")
        n_diverge = sum(1 for r in results if r.get("status") == "DIVERGE")
        n_error = sum(1 for r in results if "error" in r)

        print(f"  MATCH:   {n_match}")
        print(f"  CLOSE:   {n_close}")
        print(f"  DIVERGE: {n_diverge}")
        print(f"  ERRORS:  {n_error}")

        if n_diverge > 0 or n_error > 0:
            print("\n  Status: FAIL - Standard and parallel modes produce different outputs")
            return 1
        elif n_match == 0:
            print("\n  Status: FAIL - No comparisons were made")
            return 1
        else:
            print("\n  Status: PASS - Standard and parallel modes produce same outputs")
            return 0

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"Log saved to: {log_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Standard vs Parallel RFD3")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--length", type=int, default=None, help="Protein length (overrides config)")
    parser.add_argument("--symmetry", type=str, default=None, help="Symmetry type (e.g., I for Icosahedral)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs for parallel mode")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log files")

    # Internal flags for subprocess mode
    parser.add_argument("--parallel-subprocess", action="store_true",
                        help="[Internal] Run as torchrun subprocess for parallel mode")
    parser.add_argument("--output-cache", type=str, default=None,
                        help="[Internal] Path to save parallel output")

    args = parser.parse_args()

    # Determine if we're the orchestrator or a torchrun subprocess
    is_subprocess = args.parallel_subprocess or ('RANK' in os.environ)

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    log_dir = Path(args.log_dir) if args.log_dir else project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    # CLI args override config
    if args.length is not None:
        config['length'] = args.length
    if args.symmetry is not None:
        config['symmetry'] = args.symmetry
    if args.seed is not None:
        config['seed'] = args.seed

    # Set defaults
    config.setdefault('length', 100)
    config.setdefault('seed', 42)

    if is_subprocess:
        # We're inside torchrun - run parallel mode only
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
        # Main orchestrator mode
        return run_orchestrator(config, args.config, args.num_gpus, log_dir)


if __name__ == "__main__":
    sys.exit(main())