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
    mpnn_batch_size=8
):
    """
    Run the design pipeline: RFD3 -> ProteinMPNN.
    
    Args:
        symmetry_id (str): Symmetry group ID (e.g. "C3", "D2"). 
                           If provided, enables symmetric backbone generation and sequence design.
                           If None, runs unconditionally without symmetry.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

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
    # -------------------------------------

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=rfd3_batch_size,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path
    )
    
    rfd3_engine = RFD3InferenceEngine(**rfd3_config)
    
    print(f"Running RFD3 to generate {num_designs} designs...")
    rfd3_outputs = rfd3_engine.run(
        inputs=None,      # Unconditional generation
        out_dir=None,     # Return results in memory
        n_batches=1,
    )

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

    # =========================================================================
    # 3. Process Designs
    # =========================================================================
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
            mpnn_results = mpnn_engine.run(
                input_dicts=[mpnn_input_config], 
                atom_arrays=[rfd3_out.atom_array]
            )
            
            # Save annotated outputs
            for j, mpnn_out in enumerate(mpnn_results):
                seq_name = f"{backbone_name}_seq_{j}"
                out_file = out_path / f"{seq_name}"
                
                # Save structure (CIF) with MPNN annotations
                mpnn_out.write_structure(base_path=str(out_file), file_type="cif")
                # Save sequence (FASTA)
                # mpnn_out.write_fasta(base_path=str(out_file))

    print(f"Done! Outputs saved to {out_path.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="inference_outputs/design_demo")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("--symmetry", type=str, default=None, help="Symmetry ID (e.g. C3, D2). Default None.")
    parser.add_argument("--mpnn_batch_size", type=int, default=5)
    args = parser.parse_args()

    run_design(
        out_dir=args.out_dir,
        length=args.length,
        num_designs=args.num_designs,
        symmetry_id=args.symmetry, # Pass symmetry argument here
        mpnn_batch_size=args.mpnn_batch_size
    )

if __name__ == "__main__":
    main()

