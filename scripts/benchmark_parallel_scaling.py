#!/usr/bin/env python3
"""
Benchmark parallel scaling for icosahedral protein design.

This script empirically validates the theoretical scaling predictions:
1. √N scaling for maximum ASU length before OOM
2. Constant time for generation (with extra_chunking=False)
3. Protein diameter as a function of ASU length

Usage:
    python scripts/benchmark_parallel_scaling.py --output results/scaling_benchmark.csv
    python scripts/benchmark_parallel_scaling.py --gpu-counts 1 2 4 8 --output results/test.csv
    python scripts/benchmark_parallel_scaling.py --dry-run  # Show commands without running
    python scripts/benchmark_parallel_scaling.py --resume   # Resume from existing results

This script:
1. For N = 1, 2, 3, 4, 5, 6, 7, 8 GPUs (configurable)
2. Binary search to find max ASU length before OOM
3. Records timing, diameter, memory for successful runs
4. Outputs CSV with all metrics
"""

import argparse
import subprocess
import time
import os
import sys
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set
import yaml

# Theoretical predictions for initial binary search bounds
# Based on formula: I_max ≈ 548 * sqrt(N * M) where M ≈ 1.3 GB effective memory
# ASU_max = I_max / 60 (for icosahedral symmetry)
PREDICTED_MAX_ASU = {
    1: 55,   
    2: 100,  
    3: 117,  
    4: 137,  
    5: 156,  
    6: 170,  
    7: 183,  
    8: 193,  
}

# Search bounds: [predicted - margin, predicted + margin]
SEARCH_MARGIN = 40

# Minimum ASU length to test
MIN_ASU = 30

# GPUs per node on Isambard
GPUS_PER_NODE = 4


class TeeLogger:
    """Write to both stdout and a log file for persistent logging."""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a", buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def setup_logging(output_dir: Path) -> Path:
    """
    Set up persistent logging to benchmark_outputs directory.

    Returns:
        Path to the log file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"benchmark_log_{timestamp}.txt"
    return log_path


def find_existing_results(output_base: Path = Path("benchmark_outputs")) -> Dict[Tuple[int, int], Dict]:
    """
    Scan benchmark_outputs for existing successful runs.

    Returns:
        Dict mapping (n_gpus, asu_length) -> metrics dict
    """
    existing = {}

    if not output_base.exists():
        return existing

    for folder in output_base.iterdir():
        if not folder.is_dir():
            continue

        # Parse folder name: gpus{N}_asu{L}
        match = re.match(r"gpus(\d+)_asu(\d+)", folder.name)
        if not match:
            continue

        n_gpus = int(match.group(1))
        asu_length = int(match.group(2))

        # Check for successful completion (benchmark_metrics.csv exists)
        csv_path = folder / "benchmark_metrics.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        existing[(n_gpus, asu_length)] = {
                            "job_id": row["job_id"],
                            "n_gpus": int(row["n_gpus"]),
                            "n_nodes": int(row["n_nodes"]),
                            "asu_length": int(row["asu_length"]),
                            "i_total": int(row["i_total"]),
                            "l_total": int(row["l_total"]),
                            "extra_chunking": row["extra_chunking"].lower() == "true",
                            "time_seconds": float(row["time_seconds"]),
                            "diameter_angstroms": float(row["diameter_angstroms"]),
                            "peak_memory_gb": float(row["peak_memory_gb"]),
                            "status": row["status"],
                        }
                        break  # Only need first row
            except Exception as e:
                print(f"  Warning: Could not parse {csv_path}: {e}")

    return existing


def find_failed_experiments(output_base: Path = Path("benchmark_outputs")) -> Set[Tuple[int, int]]:
    """
    Find experiments that have output folders but no benchmark_metrics.csv.
    These are likely OOM failures.

    Returns:
        Set of (n_gpus, asu_length) tuples that failed
    """
    failed = set()

    if not output_base.exists():
        return failed

    for folder in output_base.iterdir():
        if not folder.is_dir():
            continue

        match = re.match(r"gpus(\d+)_asu(\d+)", folder.name)
        if not match:
            continue

        n_gpus = int(match.group(1))
        asu_length = int(match.group(2))

        csv_path = folder / "benchmark_metrics.csv"
        if not csv_path.exists():
            failed.add((n_gpus, asu_length))

    return failed


def create_config_file(
    n_gpus: int,
    asu_length: int,
    extra_chunking: bool = False,
    out_dir: str = None,
    seed: int = 42
) -> str:
    """
    Create a YAML config file for a benchmark run.

    Args:
        n_gpus: Number of GPUs to use
        asu_length: ASU length to test
        extra_chunking: Whether to enable extra chunking (affects time scaling)
        out_dir: Output directory
        seed: Random seed

    Returns:
        Path to the created config file
    """
    if out_dir is None:
        out_dir = f"benchmark_outputs/gpus{n_gpus}_asu{asu_length}"

    config = {
        "out_dir": out_dir,
        "length": asu_length,
        "num_designs": 1,
        "symmetry": "I",  # Icosahedral symmetry
        "seed": seed,
        "wandb": {"enabled": False},
        # Enable time and memory logging for benchmarks
        "verbose_time": True,
        "verbose_memory": True,
    }

    # Only set attention_parallel_factor for multi-GPU runs
    if n_gpus > 1:
        config["attention_parallel_factor"] = n_gpus

    # Add extra_chunking to config if needed
    if extra_chunking:
        config["extra_chunking"] = True

    # Create config file in project directory (shared filesystem, accessible from compute nodes)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    configs_dir = project_root / "configs" / "benchmark"
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / f"benchmark_config_{n_gpus}_{asu_length}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


def submit_job(
    n_gpus: int,
    asu_length: int,
    extra_chunking: bool = False,
    timeout_hours: int = 2,
    dry_run: bool = False
) -> Tuple[str, str]:
    """
    Submit a Slurm job for a specific GPU count and ASU length.

    Args:
        n_gpus: Number of GPUs
        asu_length: ASU length to test
        extra_chunking: Whether to enable extra chunking
        timeout_hours: Job timeout in hours
        dry_run: If True, print command without executing

    Returns:
        (job_id, config_path) tuple
    """
    # Calculate nodes needed (round up)
    n_nodes = (n_gpus + GPUS_PER_NODE - 1) // GPUS_PER_NODE

    # Calculate GPU allocation for sbatch
    # CRITICAL: Slurm requires --gpus-per-node to be a divisor of total GPUs on node.
    # On Isambard, each node has 4 GPUs, so we always request full nodes (4 GPUs each)
    # and let run_parallel.sh use only the number of GPUs we actually need.
    if n_nodes == 1:
        # Single node: just request the GPUs we need
        gpu_args = [f"--gpus={n_gpus}"]
    else:
        # Multi-node: always request full nodes (4 GPUs per node)
        # run_parallel.sh reads attention_parallel_factor from config and uses only that many
        gpu_args = [f"--gpus-per-node={GPUS_PER_NODE}"]

    # Create config file
    out_dir = f"benchmark_outputs/gpus{n_gpus}_asu{asu_length}"
    config_path = create_config_file(n_gpus, asu_length, extra_chunking, out_dir)

    # Choose script based on GPU count
    script_dir = Path(__file__).parent
    if n_gpus == 1:
        script = script_dir / "run_single_gpu.sh"
    else:
        script = script_dir / "run_parallel.sh"

    # Build sbatch command
    cmd = [
        "sbatch",
        *gpu_args,
        f"--nodes={n_nodes}",
        f"--time={timeout_hours:02d}:00:00",
        f"--job-name=bench_N{n_gpus}_L{asu_length}",
        "--output=logs/slurm/benchmark_%j.out",
        "--error=logs/slurm/benchmark_%j.err",
        "--parsable",
        str(script),
        config_path,
    ]

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return "DRY_RUN", config_path

    # Submit job
    print(f"Submitting: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Error submitting job: {result.stderr}")
        return None, config_path

    job_id = result.stdout.strip()
    print(f"  Submitted job {job_id}")
    return job_id, config_path


def get_job_status(job_id: str) -> str:
    """
    Get the status of a Slurm job.

    Returns one of: PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED, OOM, UNKNOWN
    """
    cmd = ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return "UNKNOWN"

    # sacct can return multiple states for a job (one per step)
    # Take the first non-empty state
    states = [s.strip() for s in result.stdout.strip().split("\n") if s.strip()]
    if not states:
        return "UNKNOWN"

    state = states[0]

    # Map Slurm states to our simplified states
    if state in ("PENDING", "CONFIGURING", "REQUEUED"):
        return "PENDING"
    elif state in ("RUNNING", "COMPLETING"):
        return "RUNNING"
    elif state == "COMPLETED":
        return "COMPLETED"
    elif state in ("FAILED", "NODE_FAIL"):
        return "FAILED"
    elif state == "TIMEOUT":
        return "TIMEOUT"
    elif state in ("CANCELLED", "CANCELLED+"):
        return "CANCELLED"
    elif "OUT_OF_MEMORY" in state or state == "OUT_OF_ME+":
        return "OOM"
    else:
        return state


def check_oom_in_logs(job_id: str) -> bool:
    """
    Check if a job failed due to OOM by examining log files.

    Returns True if OOM was detected, False otherwise.
    """
    log_patterns = [
        f"logs/slurm/benchmark_{job_id}.err",
        f"logs/slurm/benchmark_{job_id}.out",
    ]

    oom_indicators = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "OOM",
        "out of memory",
        "Cannot allocate memory",
        "SIGKILL",  # Often indicates OOM killer
    ]

    for pattern in log_patterns:
        log_path = Path(pattern)
        if log_path.exists():
            try:
                content = log_path.read_text()
                for indicator in oom_indicators:
                    if indicator.lower() in content.lower():
                        return True
            except Exception:
                pass

    return False


def parse_benchmark_results(job_id: str, n_gpus: int, asu_length: int) -> Optional[Dict]:
    """
    Parse benchmark results from the output CSV file.

    Returns metrics dict if found, None otherwise.
    """
    out_dir = Path(f"benchmark_outputs/gpus{n_gpus}_asu{asu_length}")
    csv_path = out_dir / "benchmark_metrics.csv"

    if not csv_path.exists():
        return None

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                return {
                    "job_id": job_id,
                    "n_gpus": int(row["n_gpus"]),
                    "n_nodes": int(row["n_nodes"]),
                    "asu_length": int(row["asu_length"]),
                    "i_total": int(row["i_total"]),
                    "l_total": int(row["l_total"]),
                    "extra_chunking": row["extra_chunking"].lower() == "true",
                    "time_seconds": float(row["time_seconds"]),
                    "diameter_angstroms": float(row["diameter_angstroms"]),
                    "peak_memory_gb": float(row["peak_memory_gb"]),
                    "status": row["status"],
                }
    except Exception as e:
        print(f"  Error parsing results: {e}")
        return None

    return None


def wait_for_job(
    job_id: str,
    n_gpus: int,
    asu_length: int,
    timeout: int = 7200,
    poll_interval: int = 30
) -> Tuple[str, Optional[Dict]]:
    """
    Wait for a job to complete and return its status and metrics.

    Args:
        job_id: Slurm job ID
        n_gpus: Number of GPUs (for finding output)
        asu_length: ASU length (for finding output)
        timeout: Maximum wait time in seconds
        poll_interval: Seconds between status checks

    Returns:
        (status, metrics) tuple where status is one of: success, oom, failed, timeout
        metrics is a dict if successful, None otherwise
    """
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"  Job {job_id} timed out after {timeout}s")
            return "timeout", None

        status = get_job_status(job_id)

        if status == "COMPLETED":
            # Job completed - check for results
            metrics = parse_benchmark_results(job_id, n_gpus, asu_length)
            if metrics:
                return "success", metrics
            else:
                # Completed but no results - might have failed silently
                if check_oom_in_logs(job_id):
                    return "oom", None
                return "failed", None

        elif status == "OOM":
            return "oom", None

        elif status in ("FAILED", "TIMEOUT", "CANCELLED"):
            # Check logs for OOM
            if check_oom_in_logs(job_id):
                return "oom", None
            return "failed", None

        elif status in ("PENDING", "RUNNING"):
            # Still running - wait and poll again
            remaining = timeout - elapsed
            print(f"  Job {job_id}: {status} ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)")
            time.sleep(poll_interval)

        else:
            print(f"  Job {job_id}: Unknown status '{status}'")
            time.sleep(poll_interval)


def binary_search_max_asu(
    n_gpus: int,
    extra_chunking: bool = False,
    timeout_hours: int = 2,
    dry_run: bool = False,
    existing_results: Dict[Tuple[int, int], Dict] = None,
    failed_experiments: Set[Tuple[int, int]] = None
) -> Tuple[Optional[int], Optional[Dict]]:
    """
    Binary search to find maximum ASU length for N GPUs.

    Args:
        n_gpus: Number of GPUs
        extra_chunking: Whether to enable extra chunking
        timeout_hours: Per-job timeout in hours
        dry_run: If True, only show commands
        existing_results: Dict of (n_gpus, asu) -> metrics for completed runs
        failed_experiments: Set of (n_gpus, asu) tuples that are known failures

    Returns:
        (max_asu, metrics_dict) where metrics_dict contains timing, diameter, etc.
        Returns (None, None) if no successful run was found.
    """
    if existing_results is None:
        existing_results = {}
    if failed_experiments is None:
        failed_experiments = set()

    predicted = PREDICTED_MAX_ASU.get(n_gpus, 100)

    # Check if we already have results for this GPU count
    gpu_results = {asu: metrics for (gpus, asu), metrics in existing_results.items()
                   if gpus == n_gpus}
    gpu_failures = {asu for (gpus, asu) in failed_experiments if gpus == n_gpus}

    if gpu_results:
        # Find the highest successful ASU we already have
        max_existing = max(gpu_results.keys())
        print(f"[N={n_gpus}] Found {len(gpu_results)} existing successful results, max ASU={max_existing}")
        best_success = max_existing
        best_metrics = gpu_results[max_existing]
        # Continue searching from max_existing + 1
        low = max_existing + 1
    else:
        best_success = None
        best_metrics = None
        low = max(MIN_ASU, predicted - SEARCH_MARGIN)

    # Determine high bound - use min of failed experiments if available
    if gpu_failures:
        min_failure = min(gpu_failures)
        print(f"[N={n_gpus}] Found {len(gpu_failures)} known failures, min failed ASU={min_failure}")
        high = min_failure - 1
    else:
        high = predicted + SEARCH_MARGIN

    print(f"\n[N={n_gpus}] Binary search range: [{low}, {high}]")
    print(f"[N={n_gpus}] Predicted max ASU: {predicted}")

    if low > high:
        print(f"[N={n_gpus}] Search range exhausted (low={low} > high={high})")
        return best_success, best_metrics

    iteration = 0
    while low <= high:
        iteration += 1
        mid = (low + high) // 2

        # Skip if already tested (success or failure)
        if (n_gpus, mid) in existing_results:
            print(f"[N={n_gpus}] ASU={mid} already succeeded, skipping")
            best_success = mid
            best_metrics = existing_results[(n_gpus, mid)]
            low = mid + 1
            continue

        if (n_gpus, mid) in failed_experiments:
            print(f"[N={n_gpus}] ASU={mid} previously failed, skipping")
            high = mid - 1
            continue

        print(f"\n[N={n_gpus}] Iteration {iteration}: Testing ASU={mid} (range: [{low}, {high}])")

        if dry_run:
            job_id, _ = submit_job(n_gpus, mid, extra_chunking, timeout_hours, dry_run=True)
            # In dry run, assume success for small ASU, OOM for large
            if mid <= predicted:
                best_success = mid
                low = mid + 1
            else:
                high = mid - 1
            continue

        job_id, config_path = submit_job(n_gpus, mid, extra_chunking, timeout_hours)
        if job_id is None:
            print(f"[N={n_gpus}] Failed to submit job for ASU={mid}")
            high = mid - 1
            continue

        status, metrics = wait_for_job(job_id, n_gpus, mid, timeout=timeout_hours * 3600)

        if status == "success":
            print(f"[N={n_gpus}] ASU={mid} succeeded!")
            best_success = mid
            best_metrics = metrics
            low = mid + 1  # Try larger
        else:  # OOM or other failure
            print(f"[N={n_gpus}] ASU={mid} failed ({status})")
            high = mid - 1  # Try smaller

    print(f"\n[N={n_gpus}] Binary search complete. Max ASU: {best_success}")
    return best_success, best_metrics


def run_single_asu(
    n_gpus: int,
    asu_length: int,
    extra_chunking: bool = False,
    timeout_hours: int = 2,
    dry_run: bool = False,
    existing_results: Dict[Tuple[int, int], Dict] = None
) -> Tuple[str, Optional[Dict]]:
    """
    Run a single benchmark for a specific ASU length.

    Returns:
        (status, metrics) tuple
    """
    if existing_results is None:
        existing_results = {}

    # Check if already completed
    if (n_gpus, asu_length) in existing_results:
        print(f"[N={n_gpus}] ASU={asu_length} already completed, using cached result")
        return "cached", existing_results[(n_gpus, asu_length)]

    print(f"\n[N={n_gpus}] Testing ASU={asu_length}")

    if dry_run:
        submit_job(n_gpus, asu_length, extra_chunking, timeout_hours, dry_run=True)
        return "dry_run", None

    job_id, config_path = submit_job(n_gpus, asu_length, extra_chunking, timeout_hours)
    if job_id is None:
        return "submit_failed", None

    return wait_for_job(job_id, n_gpus, asu_length, timeout=timeout_hours * 3600)


@dataclass
class BinarySearchState:
    """Track state of binary search for a specific GPU count."""
    n_gpus: int
    low: int
    high: int
    best_success: Optional[int] = None
    best_metrics: Optional[Dict] = None
    current_asu: Optional[int] = None
    current_job_id: Optional[str] = None
    iteration: int = 0
    completed: bool = False


def parallel_binary_search_all(
    gpu_counts: List[int],
    extra_chunking: bool = False,
    timeout_hours: int = 2,
    dry_run: bool = False,
    existing_results: Dict[Tuple[int, int], Dict] = None,
    failed_experiments: Set[Tuple[int, int]] = None,
    poll_interval: int = 30
) -> Dict[int, Tuple[Optional[int], Optional[Dict]]]:
    """
    Run binary searches for all GPU counts in parallel.

    Jobs are submitted for all GPU counts simultaneously, and as jobs complete,
    new tests are submitted based on updated search bounds.

    Args:
        gpu_counts: List of GPU counts to test
        extra_chunking: Whether to enable extra chunking
        timeout_hours: Per-job timeout in hours
        dry_run: If True, only show commands
        existing_results: Dict of (n_gpus, asu) -> metrics for completed runs
        failed_experiments: Set of (n_gpus, asu) tuples that are known failures
        poll_interval: Seconds between status checks

    Returns:
        Dict mapping n_gpus -> (max_asu, metrics_dict)
    """
    if existing_results is None:
        existing_results = {}
    if failed_experiments is None:
        failed_experiments = set()

    # Initialize search state for each GPU count
    search_states: Dict[int, BinarySearchState] = {}

    for n_gpus in gpu_counts:
        predicted = PREDICTED_MAX_ASU.get(n_gpus, 100)

        # Check if we already have results for this GPU count
        gpu_results = {asu: metrics for (gpus, asu), metrics in existing_results.items()
                       if gpus == n_gpus}
        gpu_failures = {asu for (gpus, asu) in failed_experiments if gpus == n_gpus}

        if gpu_results:
            max_existing = max(gpu_results.keys())
            print(f"[N={n_gpus}] Found {len(gpu_results)} existing successful results, max ASU={max_existing}")
            best_success = max_existing
            best_metrics = gpu_results[max_existing]
            low = max_existing + 1
        else:
            best_success = None
            best_metrics = None
            low = max(MIN_ASU, predicted - SEARCH_MARGIN)

        if gpu_failures:
            min_failure = min(gpu_failures)
            print(f"[N={n_gpus}] Found {len(gpu_failures)} known failures, min failed ASU={min_failure}")
            high = min_failure - 1
        else:
            high = predicted + SEARCH_MARGIN

        print(f"[N={n_gpus}] Binary search range: [{low}, {high}], predicted max={predicted}")

        state = BinarySearchState(
            n_gpus=n_gpus,
            low=low,
            high=high,
            best_success=best_success,
            best_metrics=best_metrics,
        )

        # Check if already complete
        if low > high:
            print(f"[N={n_gpus}] Search range exhausted (low={low} > high={high})")
            state.completed = True

        search_states[n_gpus] = state

    def get_next_test_asu(state: BinarySearchState) -> Optional[int]:
        """Get the next ASU to test for this search, or None if done."""
        if state.completed or state.low > state.high:
            return None

        mid = (state.low + state.high) // 2

        # Skip if already tested
        while mid >= state.low and mid <= state.high:
            if (state.n_gpus, mid) in existing_results:
                print(f"[N={state.n_gpus}] ASU={mid} already succeeded, skipping")
                state.best_success = mid
                state.best_metrics = existing_results[(state.n_gpus, mid)]
                state.low = mid + 1
                mid = (state.low + state.high) // 2
            elif (state.n_gpus, mid) in failed_experiments:
                print(f"[N={state.n_gpus}] ASU={mid} previously failed, skipping")
                state.high = mid - 1
                mid = (state.low + state.high) // 2
            else:
                break

        if state.low > state.high:
            return None

        return mid

    def submit_next_job(state: BinarySearchState) -> bool:
        """Submit the next job for this search. Returns True if a job was submitted."""
        asu = get_next_test_asu(state)
        if asu is None:
            state.completed = True
            return False

        state.iteration += 1
        state.current_asu = asu

        print(f"\n[N={state.n_gpus}] Iteration {state.iteration}: Testing ASU={asu} (range: [{state.low}, {state.high}])")

        if dry_run:
            job_id, _ = submit_job(state.n_gpus, asu, extra_chunking, timeout_hours, dry_run=True)
            # In dry run, assume success for small ASU, OOM for large
            predicted = PREDICTED_MAX_ASU.get(state.n_gpus, 100)
            if asu <= predicted:
                state.best_success = asu
                state.low = asu + 1
            else:
                state.high = asu - 1
            return False  # No real job submitted

        job_id, _ = submit_job(state.n_gpus, asu, extra_chunking, timeout_hours)
        if job_id is None:
            print(f"[N={state.n_gpus}] Failed to submit job for ASU={asu}")
            state.high = asu - 1
            return submit_next_job(state)  # Try next ASU

        state.current_job_id = job_id
        return True

    # Submit initial jobs for all GPU counts
    print("\n" + "=" * 60)
    print("Submitting initial jobs for all GPU counts...")
    print("=" * 60)

    active_jobs: Dict[str, int] = {}  # job_id -> n_gpus

    for n_gpus, state in search_states.items():
        if not state.completed:
            if submit_next_job(state):
                active_jobs[state.current_job_id] = n_gpus

    if dry_run:
        # In dry run mode, searches were already completed in submit_next_job
        results = {}
        for n_gpus, state in search_states.items():
            # Continue simulation until convergence
            while not state.completed and state.low <= state.high:
                submit_next_job(state)
            results[n_gpus] = (state.best_success, state.best_metrics)
        return results

    # Poll for job completions and submit next tests
    print("\n" + "=" * 60)
    print(f"Polling for job completions ({len(active_jobs)} active jobs)...")
    print("=" * 60)

    timeout_seconds = timeout_hours * 3600
    start_time = time.time()

    while active_jobs:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds * 2:  # Allow 2x timeout for all jobs
            print(f"\nGlobal timeout reached after {elapsed:.0f}s")
            break

        # Check status of all active jobs
        completed_jobs = []

        for job_id, n_gpus in list(active_jobs.items()):
            status = get_job_status(job_id)
            state = search_states[n_gpus]

            if status in ("PENDING", "RUNNING"):
                continue  # Still running

            completed_jobs.append(job_id)

            if status == "COMPLETED":
                metrics = parse_benchmark_results(job_id, n_gpus, state.current_asu)
                if metrics:
                    print(f"[N={n_gpus}] ASU={state.current_asu} succeeded!")
                    state.best_success = state.current_asu
                    state.best_metrics = metrics
                    state.low = state.current_asu + 1
                else:
                    # Completed but no results - might have failed silently
                    if check_oom_in_logs(job_id):
                        print(f"[N={n_gpus}] ASU={state.current_asu} failed (OOM)")
                        state.high = state.current_asu - 1
                    else:
                        print(f"[N={n_gpus}] ASU={state.current_asu} failed (no results)")
                        state.high = state.current_asu - 1

            elif status == "OOM":
                print(f"[N={n_gpus}] ASU={state.current_asu} failed (OOM)")
                state.high = state.current_asu - 1

            else:  # FAILED, TIMEOUT, CANCELLED, etc.
                if check_oom_in_logs(job_id):
                    print(f"[N={n_gpus}] ASU={state.current_asu} failed (OOM in logs)")
                else:
                    print(f"[N={n_gpus}] ASU={state.current_asu} failed ({status})")
                state.high = state.current_asu - 1

        # Remove completed jobs and submit next tests
        for job_id in completed_jobs:
            n_gpus = active_jobs.pop(job_id)
            state = search_states[n_gpus]

            # Submit next job if search is not complete
            if state.low <= state.high:
                if submit_next_job(state):
                    active_jobs[state.current_job_id] = n_gpus
            else:
                state.completed = True
                print(f"[N={n_gpus}] Binary search complete. Max ASU: {state.best_success}")

        if active_jobs:
            # Show status summary
            active_summary = ", ".join(f"N={n}" for n in sorted(active_jobs.values()))
            print(f"  Active: {len(active_jobs)} jobs ({active_summary}), elapsed: {elapsed:.0f}s")
            time.sleep(poll_interval)

    # Collect results
    results = {}
    for n_gpus, state in search_states.items():
        results[n_gpus] = (state.best_success, state.best_metrics)
        if not state.completed:
            print(f"[N={n_gpus}] Search incomplete, best so far: ASU={state.best_success}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel scaling for icosahedral protein design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full benchmark with PARALLEL binary search for max ASU (default)
    python scripts/benchmark_parallel_scaling.py --output results/scaling.csv

    # Run with SEQUENTIAL binary search (one GPU count at a time)
    python scripts/benchmark_parallel_scaling.py --sequential-search --output results/scaling.csv

    # Resume from existing results (skip completed experiments)
    python scripts/benchmark_parallel_scaling.py --output results/scaling.csv --resume

    # Test specific GPU counts only
    python scripts/benchmark_parallel_scaling.py --gpu-counts 1 2 4 --output results/test.csv

    # Test a fixed ASU length across all GPU counts (no binary search)
    python scripts/benchmark_parallel_scaling.py --fixed-asu 100 --output results/fixed.csv

    # Dry run - show commands without executing
    python scripts/benchmark_parallel_scaling.py --dry-run

    # Check existing results only
    python scripts/benchmark_parallel_scaling.py --dry-run --resume

    # Enable extra chunking (O(N) time but fits in memory)
    python scripts/benchmark_parallel_scaling.py --extra-chunking --output results/chunked.csv
        """
    )
    parser.add_argument("--output", "-o", default="results/scaling_benchmark.csv",
                        help="Output CSV file path")
    parser.add_argument("--gpu-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="GPU counts to test (default: 1 2 3 4 5 6 7 8)")
    parser.add_argument("--fixed-asu", type=int, default=None,
                        help="Test a fixed ASU length instead of binary search")
    parser.add_argument("--extra-chunking", action="store_true",
                        help="Enable extra chunking (O(N) time, less memory)")
    parser.add_argument("--timeout-hours", type=int, default=2,
                        help="Per-job timeout in hours (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results, skip completed experiments")
    parser.add_argument("--parallel-search", action="store_true",
                        help="Run binary searches for all GPU counts in parallel (default: enabled)")
    parser.add_argument("--sequential-search", action="store_true",
                        help="Run binary searches sequentially (one GPU count at a time)")
    args = parser.parse_args()

    # Default to parallel search unless sequential is explicitly requested
    use_parallel_search = not args.sequential_search

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)

    # Create benchmark outputs directory
    benchmark_dir = Path("benchmark_outputs")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Setup persistent logging
    log_path = setup_logging(benchmark_dir)
    tee_logger = TeeLogger(log_path)
    sys.stdout = tee_logger

    print(f"Logging to: {log_path}")
    print(f"Started at: {datetime.now().isoformat()}")

    # Find existing results and failures
    existing_results = {}
    failed_experiments = set()

    if args.resume:
        existing_results = find_existing_results()
        failed_experiments = find_failed_experiments()
        print(f"\nFound {len(existing_results)} existing successful runs:")
        for (n_gpus, asu), metrics in sorted(existing_results.items()):
            print(f"  N={n_gpus}, ASU={asu}: {metrics['time_seconds']:.1f}s, "
                  f"diameter={metrics['diameter_angstroms']:.1f}Å")
        print(f"\nFound {len(failed_experiments)} known failures:")
        for (n_gpus, asu) in sorted(failed_experiments):
            print(f"  N={n_gpus}, ASU={asu}")

    results = []

    print("\n" + "=" * 60)
    print("Parallel Scaling Benchmark")
    print("=" * 60)
    print(f"GPU counts: {args.gpu_counts}")
    print(f"Extra chunking: {args.extra_chunking}")
    print(f"Output: {args.output}")
    print(f"Resume mode: {args.resume}")
    print(f"Search mode: {'parallel' if use_parallel_search else 'sequential'}")
    if args.fixed_asu:
        print(f"Fixed ASU: {args.fixed_asu}")
    else:
        print("Mode: Binary search for max ASU")
    print("=" * 60)

    if args.fixed_asu:
        # Fixed ASU mode - test each GPU count with the same ASU length
        for n_gpus in args.gpu_counts:
            print(f"\n{'='*60}")
            print(f"Benchmarking N={n_gpus} GPUs")
            print(f"{'='*60}")

            status, metrics = run_single_asu(
                n_gpus, args.fixed_asu, args.extra_chunking,
                args.timeout_hours, args.dry_run, existing_results
            )
            if metrics:
                results.append(metrics)
                print(f"[N={n_gpus}] ASU={args.fixed_asu}: {status}, "
                      f"Time={metrics['time_seconds']:.1f}s, "
                      f"Diameter={metrics['diameter_angstroms']:.1f}Å")
            else:
                print(f"[N={n_gpus}] ASU={args.fixed_asu}: {status}")

    elif use_parallel_search:
        # Parallel binary search - run all GPU counts simultaneously
        print("\n" + "=" * 60)
        print("Running PARALLEL binary search for all GPU counts")
        print("=" * 60)

        parallel_results = parallel_binary_search_all(
            gpu_counts=args.gpu_counts,
            extra_chunking=args.extra_chunking,
            timeout_hours=args.timeout_hours,
            dry_run=args.dry_run,
            existing_results=existing_results,
            failed_experiments=failed_experiments,
        )

        # Collect results
        for n_gpus in args.gpu_counts:
            max_asu, metrics = parallel_results.get(n_gpus, (None, None))
            if metrics:
                results.append(metrics)
                print(f"[N={n_gpus}] Max ASU: {max_asu}, "
                      f"Time: {metrics['time_seconds']:.1f}s, "
                      f"Diameter: {metrics['diameter_angstroms']:.1f}Å")
            elif max_asu:
                print(f"[N={n_gpus}] Max ASU: {max_asu} (no metrics available)")
            else:
                print(f"[N={n_gpus}] No successful runs")

    else:
        # Sequential binary search - run one GPU count at a time
        for n_gpus in args.gpu_counts:
            print(f"\n{'='*60}")
            print(f"Benchmarking N={n_gpus} GPUs")
            print(f"{'='*60}")

            max_asu, metrics = binary_search_max_asu(
                n_gpus, args.extra_chunking, args.timeout_hours, args.dry_run,
                existing_results, failed_experiments
            )

            if metrics:
                results.append(metrics)
                print(f"[N={n_gpus}] Max ASU: {max_asu}, "
                      f"Time: {metrics['time_seconds']:.1f}s, "
                      f"Diameter: {metrics['diameter_angstroms']:.1f}Å")
            elif max_asu:
                print(f"[N={n_gpus}] Max ASU: {max_asu} (no metrics available)")
            else:
                print(f"[N={n_gpus}] No successful runs")

    # Write results CSV
    if results and not args.dry_run:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "job_id", "n_gpus", "n_nodes", "asu_length", "i_total", "l_total",
                "extra_chunking", "time_seconds", "diameter_angstroms", "peak_memory_gb", "status"
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")

    # Print summary
    if results:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'N GPUs':>8} {'Max ASU':>10} {'Time (s)':>12} {'Diameter (Å)':>14}")
        print("-" * 60)
        for r in results:
            print(f"{r['n_gpus']:>8} {r['asu_length']:>10} {r['time_seconds']:>12.1f} {r['diameter_angstroms']:>14.1f}")

    print(f"\nCompleted at: {datetime.now().isoformat()}")
    print("Benchmark complete!")

    # Restore stdout and close log file
    sys.stdout = tee_logger.terminal
    tee_logger.close()


if __name__ == "__main__":
    main()
