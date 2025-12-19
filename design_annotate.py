#!/usr/bin/env python3
"""
Script to run RFDiffusion3 design and annotate with ProteinMPNN.
Supports optional symmetry constraints.
"""

# Set environment variables BEFORE any imports that use them
import os
os.environ.setdefault('CCD_MIRROR_PATH', '')
os.environ.setdefault('PDB_MIRROR_PATH', '')

import sys
import argparse
from pathlib import Path
import numpy as np
import time
import torch

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


import threading


class GPUMemoryMonitor:
    """Background thread that continuously monitors GPU memory usage."""
    
    def __init__(self, wandb_run=None, interval=1.0):
        """
        Args:
            wandb_run: W&B run object for logging
            interval: Sampling interval in seconds
        """
        self.wandb_run = wandb_run
        self.interval = interval
        self.running = False
        self.thread = None
        self.current_stage = "init"
        self.start_time = None
    
    def _get_gpu_stats(self):
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        stats = {}
        for i in range(torch.cuda.device_count()):
            # Global (CUDA) view — includes other processes
            free_b, total_b = torch.cuda.mem_get_info(i)  # cudaMemGetInfo
            free_gb = free_b / 1024**3
            total_gb = total_b / 1024**3
            used_gb = total_gb - free_gb
            
            # PyTorch (this process) view
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            
            # cached = reserved - allocated (PyTorch's internal cache, but FRAGMENTED!)
            cached = max(reserved - allocated, 0.0)
            
            stats[f"gpu_{i}/global_total_gb"] = total_gb
            stats[f"gpu_{i}/global_free_gb"] = free_gb      # KEY: OOM when single alloc > this!
            stats[f"gpu_{i}/global_used_gb"] = used_gb
            stats[f"gpu_{i}/allocated_gb"] = allocated
            stats[f"gpu_{i}/reserved_gb"] = reserved
            stats[f"gpu_{i}/cached_gb"] = cached            # WARNING: fragmented, can't use for large allocs
            stats[f"gpu_{i}/peak_allocated_gb"] = peak_allocated
        
        return stats
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            stats = self._get_gpu_stats()
            
            if stats and self.wandb_run is not None:
                log_data = {}
                for k, v in stats.items():
                    log_data[f"monitor/{k}"] = v
                
                self.wandb_run.log(log_data)
            
            time.sleep(self.interval)
    
    def start(self):
        """Start the monitoring thread."""
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"  GPU memory monitor started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        print("  GPU memory monitor stopped")
    
    def set_stage(self, stage: str):
        """Update the current pipeline stage."""
        self.current_stage = stage


def get_gpu_memory_stats():
    """Get current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {}
    
    stats = {}
    for i in range(torch.cuda.device_count()):
        # Global (CUDA) view — includes other processes
        free_b, total_b = torch.cuda.mem_get_info(i)
        free_gb = free_b / 1024**3
        total_gb = total_b / 1024**3
        used_gb = total_gb - free_gb
        
        # PyTorch (this process) view
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        
        # cached = reserved - allocated (PyTorch's internal cache, but FRAGMENTED!)
        cached = max(reserved - allocated, 0.0)
        
        stats[f"gpu_{i}/global_total_gb"] = total_gb
        stats[f"gpu_{i}/global_free_gb"] = free_gb
        stats[f"gpu_{i}/global_used_gb"] = used_gb
        stats[f"gpu_{i}/allocated_gb"] = allocated
        stats[f"gpu_{i}/reserved_gb"] = reserved
        stats[f"gpu_{i}/cached_gb"] = cached
        stats[f"gpu_{i}/peak_allocated_gb"] = peak_allocated
    
    return stats


def log_memory(stage: str, wandb_run=None, monitor=None):
    """Log memory stats for a given stage."""
    stats = get_gpu_memory_stats()
    if stats:
        # Show the KEY metric: global_free_gb determines if large allocations will OOM
        global_free = stats.get('gpu_0/global_free_gb', 0)
        allocated = stats.get('gpu_0/allocated_gb', 0)
        print(f"  [{stage}] GPU: {allocated:.1f} GB allocated, {global_free:.1f} GB free (OOM if alloc > free)")
        
        if wandb_run is not None:
            # Add stage prefix to all keys
            logged_stats = {f"{stage}/{k}": v for k, v in stats.items()}
            wandb_run.log(logged_stats)
        
        # Update monitor stage
        if monitor is not None:
            monitor.set_stage(stage)
    
    return stats

# Add project root to path if needed
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import get_chains
except ImportError:
    # Fallback for manual path setup if not installed as package
    sys.path.append(str(project_root / "models/rfd3/src"))
    sys.path.append(str(project_root / "models/mpnn/src"))
    sys.path.append(str(project_root / "models/rf3/src"))
    sys.path.append(str(project_root / "src"))
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import get_chains

def run_design(
    out_dir, 
    length=100, 
    num_designs=1, 
    symmetry_id=None,  # <--- CHANGE THIS: Set to "C3", "D2", etc. for symmetry. None for no symmetry.
    mpnn_batch_size=8,
    low_memory_mode=False,
    use_wandb=False,
    wandb_project="fast-rfd3",
    wandb_run_name=None
):
    """
    Run the design pipeline: RFD3 -> ProteinMPNN.
    
    Args:
        symmetry_id (str): Symmetry group ID (e.g. "C3", "D2"). 
                           If provided, enables symmetric backbone generation and sequence design.
                           If None, runs unconditionally without symmetry.
        low_memory_mode (bool): Enable low memory mode for RFD3 (memory efficient tokenization).
                                Useful for symmetric designs or large structures.
        use_wandb (bool): Enable W&B logging for memory tracking.
        wandb_project (str): W&B project name.
        wandb_run_name (str): W&B run name. Auto-generated if None.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Initialize W&B for memory tracking
    # =========================================================================
    wandb_run = None
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: W&B requested but not installed. Skipping W&B logging.")
        else:
            run_name = wandb_run_name or f"design_L{length}_sym{symmetry_id or 'none'}"
            wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "length": length,
                    "num_designs": num_designs,
                    "symmetry_id": symmetry_id,
                    "mpnn_batch_size": mpnn_batch_size,
                    "low_memory_mode": low_memory_mode,
                    "pipeline": "design_annotate",
                }
            )
            print(f"W&B run initialized: {wandb_run.url}")
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Start GPU memory monitor for continuous logging
    monitor = None
    if wandb_run is not None:
        monitor = GPUMemoryMonitor(wandb_run=wandb_run, interval=1.0)
        monitor.start()

    # =========================================================================
    # 1. Configure RFD3 (Backbone Generation)
    # =========================================================================
    print(f"Initializing RFD3 Engine...")
    
    # Base specification
    rfd3_spec = {
        'length': length,
        'extra': {} # Add this to avoid KeyError in engine
    }
    
    # Base inference configuration
    rfd3_inference_sampler = {}
    rfd3_batch_size = num_designs
    
    # Path to local checkpoint
    ckpt_path = "ckpt/rfd3_latest.ckpt"
    if not Path(ckpt_path).exists():
        print(f"Warning: Checkpoint not found at {ckpt_path}. Falling back to default 'rfd3' lookup.")
        ckpt_path = "rfd3"

    # --- SYMMETRY CONFIGURATION (RFD3) ---
    if symmetry_id:
        print(f"  Enabling symmetry: {symmetry_id}")
        # Add symmetry spec
        rfd3_spec['symmetry'] = {
            'id': symmetry_id,
            'is_symmetric_motif': False # Important for unconditional symmetric design (no motif)
        }
        
        # Enable symmetry sampler
        rfd3_inference_sampler = {"kind": "symmetry"}
        
        # Recommended to reduce batch size for symmetry to avoid OOM
        # You can try increasing this if you have enough VRAM
        if num_designs > 1:
            print("  Note: Symmetry requested. It is often recommended to use batch_size=1 or run sequentially for memory reasons.")
        if not low_memory_mode:
            print("  Note: Consider using --low_memory_mode for symmetric designs to reduce memory usage.")
    # -------------------------------------
    
    if low_memory_mode:
        print(f"  Low memory mode enabled (memory efficient tokenization)")

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=rfd3_batch_size,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path,
        low_memory_mode=low_memory_mode
    )
    
    rfd3_engine = RFD3InferenceEngine(**rfd3_config)
    log_memory("rfd3_init", wandb_run, monitor)
    
    print(f"Running RFD3 to generate {num_designs} designs...")
    rfd3_start = time.time()
    rfd3_outputs = rfd3_engine.run(
        inputs=None,      # Unconditional generation
        out_dir=None,     # Return results in memory
        n_batches=1,
    )
    rfd3_time = time.time() - rfd3_start
    log_memory("rfd3_inference", wandb_run, monitor)
    print(f"  RFD3 inference took {rfd3_time:.2f}s")
    if wandb_run:
        wandb_run.log({"rfd3/inference_time_s": rfd3_time})

    # =========================================================================
    # 2. Configure MPNN (Sequence Design)
    # =========================================================================
    print(f"Initializing MPNN Engine...")
    
    # Check for local MPNN checkpoint in ckpt/ directory
    mpnn_ckpt_filename = "ligandmpnn_v_32_010_25.pt"
    mpnn_ckpt_path = Path("ckpt") / mpnn_ckpt_filename
    if not mpnn_ckpt_path.exists():
        print(f"Warning: MPNN checkpoint not found at {mpnn_ckpt_path}. Using default lookup.")
        mpnn_ckpt_path = None # Let foundry find default
    else:
        mpnn_ckpt_path = str(mpnn_ckpt_path)

    mpnn_engine = MPNNInferenceEngine(
        model_type="ligand_mpnn",
        checkpoint_path=mpnn_ckpt_path,
        is_legacy_weights=True,
        out_directory=None,
        write_structures=False,
        write_fasta=False, # Default to not writing FASTA
    )
    log_memory("mpnn_init", wandb_run, monitor)

    # =========================================================================
    # 3. Process Designs
    # =========================================================================
    total_mpnn_time = 0.0
    for batch_id, output_list in rfd3_outputs.items():
        for i, rfd3_out in enumerate(output_list):
            backbone_name = f"design_{batch_id}_{i}"
            print(f"Processing {backbone_name}...")
            
            # Save the backbone
            to_cif_file(rfd3_out.atom_array, str(out_path / f"{backbone_name}_backbone.cif"))
            
            # Prepare MPNN input options
            mpnn_input_config = {
                "batch_size": mpnn_batch_size,
                "name": backbone_name
            }

            # --- SYMMETRY CONFIGURATION (MPNN) ---
            if symmetry_id:
                # Detect chains in the generated backbone
                chains = get_chains(rfd3_out.atom_array)
                chain_list = sorted(list(set(chains)))
                print(f"  Detected chains for symmetry: {chain_list}")
                
                # Tell MPNN to treat these chains as a homo-oligomer
                # This ensures the designed sequence is identical across symmetric subunits
                mpnn_input_config["homo_oligomer_chains"] = [chain_list]
            # -------------------------------------

            # Run MPNN
            mpnn_start = time.time()
            mpnn_results = mpnn_engine.run(
                input_dicts=[mpnn_input_config], 
                atom_arrays=[rfd3_out.atom_array]
            )
            mpnn_time = time.time() - mpnn_start
            total_mpnn_time += mpnn_time
            log_memory(f"mpnn_inference_{backbone_name}", wandb_run, monitor)
            print(f"  MPNN inference took {mpnn_time:.2f}s")
            
            # Save annotated outputs
            for j, mpnn_out in enumerate(mpnn_results):
                seq_name = f"{backbone_name}_seq_{j}"
                out_file = out_path / f"{seq_name}"
                
                # Save structure (CIF) with MPNN annotations
                mpnn_out.write_structure(base_path=str(out_file), file_type="cif")
                # Save sequence (FASTA)
                # mpnn_out.write_fasta(base_path=str(out_file))

    # Stop monitor and log final summary
    if monitor is not None:
        monitor.stop()
    
    final_stats = log_memory("final", wandb_run, monitor)
    if wandb_run:
        wandb_run.log({
            "mpnn/total_inference_time_s": total_mpnn_time,
            "summary/peak_memory_gb": final_stats.get("gpu_0/memory_max_allocated_gb", 0),
        })
        wandb_run.finish()
        print(f"W&B run completed.")
    
    print(f"Done! Outputs saved to {out_path.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="inference_outputs/design_demo")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("--symmetry", type=str, default=None, help="Symmetry ID (e.g. C3, D2). Default None.")
    parser.add_argument("--mpnn_batch_size", type=int, default=5)
    parser.add_argument("--low_memory_mode", action="store_true", help="Enable low memory mode for RFD3 (memory efficient tokenization)")
    # W&B arguments
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging for memory tracking")
    parser.add_argument("--wandb_project", type=str, default="fast-rfd3", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (auto-generated if not provided)")
    args = parser.parse_args()

    run_design(
        out_dir=args.out_dir,
        length=args.length,
        num_designs=args.num_designs,
        symmetry_id=args.symmetry, # Pass symmetry argument here
        mpnn_batch_size=args.mpnn_batch_size,
        low_memory_mode=args.low_memory_mode,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )

if __name__ == "__main__":
    main()

