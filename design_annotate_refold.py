#!/usr/bin/env python3
"""
Script to run RFDiffusion3 -> ProteinMPNN -> RF3 (validation)
Based on design_with_symmetry.py
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
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput
    from rf3.utils.predicted_error import annotate_atom_array_b_factor_with_plddt
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import get_chains, rmsd, superimpose
    from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES
except ImportError:
    # Fallback for manual path setup if not installed as package
    sys.path.append(str(project_root / "models/rfd3/src"))
    sys.path.append(str(project_root / "models/mpnn/src"))
    sys.path.append(str(project_root / "models/rf3/src"))
    sys.path.append(str(project_root / "src"))
    from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
    from mpnn.inference_engines.mpnn import MPNNInferenceEngine
    from rf3.inference_engines.rf3 import RF3InferenceEngine
    from rf3.utils.inference import InferenceInput
    from rf3.utils.predicted_error import annotate_atom_array_b_factor_with_plddt
    from atomworks.io.utils.io_utils import to_cif_file
    from biotite.structure import get_chains, rmsd, superimpose
    from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES

def run_design_validate(
    out_dir, 
    length=100, 
    num_designs=1, 
    symmetry_id=None,
    mpnn_batch_size=8,
    write_combined=True
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Configure RFD3 (Backbone Generation)
    # =========================================================================
    print(f"Initializing RFD3 Engine...")
    rfd3_spec = {
        'length': length,
        'extra': {}
    }
    rfd3_inference_sampler = {}
    ckpt_path = "ckpt/rfd3_latest.ckpt"
    if not Path(ckpt_path).exists():
        ckpt_path = "rfd3"

    if symmetry_id:
        print(f"  Enabling symmetry: {symmetry_id}")
        rfd3_spec['symmetry'] = {
            'id': symmetry_id,
            'is_symmetric_motif': False
        }
        rfd3_inference_sampler = {"kind": "symmetry"}
        if num_designs > 1:
            print("  Note: Symmetry requested. It is often recommended to use batch_size=1 or run sequentially for memory reasons.")

    rfd3_config = RFD3InferenceConfig(
        specification=rfd3_spec,
        diffusion_batch_size=num_designs,
        inference_sampler=rfd3_inference_sampler,
        ckpt_path=ckpt_path
    )
    
    rfd3_engine = RFD3InferenceEngine(**rfd3_config)
    
    print(f"Running RFD3 to generate {num_designs} designs...")
    rfd3_outputs = rfd3_engine.run(
        inputs=None,
        out_dir=None,
        n_batches=1,
    )

    # =========================================================================
    # 2. Configure MPNN (Sequence Design)
    # =========================================================================
    print(f"Initializing MPNN Engine...")
    mpnn_ckpt_path = Path("ckpt") / "ligandmpnn_v_32_010_25.pt"
    mpnn_ckpt = str(mpnn_ckpt_path) if mpnn_ckpt_path.exists() else None

    mpnn_engine = MPNNInferenceEngine(
        model_type="ligand_mpnn",
        checkpoint_path=mpnn_ckpt,
        is_legacy_weights=True,
        out_directory=None,
        write_structures=False,
        write_fasta=False,
    )

    # =========================================================================
    # 3. Configure RF3 (Validation)
    # =========================================================================
    print(f"Initializing RF3 Engine for validation...")
    # Try finding local checkpoint for RF3
    # Look for ckpt starting with rf3_foundry...
    rf3_ckpts = list(Path("ckpt").glob("rf3_foundry*.ckpt"))
    rf3_ckpt = str(rf3_ckpts[0]) if rf3_ckpts else "rf3" # default lookup if not found
    
    rf3_engine = RF3InferenceEngine(ckpt_path=rf3_ckpt, verbose=False)

    # =========================================================================
    # 4. Pipeline Loop
    # =========================================================================
    for batch_id, output_list in rfd3_outputs.items():
        for i, rfd3_out in enumerate(output_list):
            backbone_name = f"design_{batch_id}_{i}"
            print(f"\nProcessing {backbone_name}...")
            
            # Save the backbone
            to_cif_file(rfd3_out.atom_array, str(out_path / f"{backbone_name}_backbone.cif"))
            
            # Prepare MPNN
            mpnn_input_config = {
                "batch_size": mpnn_batch_size,
                "name": backbone_name
            }
            if symmetry_id:
                chains = get_chains(rfd3_out.atom_array)
                chain_list = sorted(list(set(chains)))
                mpnn_input_config["homo_oligomer_chains"] = [chain_list]

            # Run MPNN
            mpnn_results = mpnn_engine.run(
                input_dicts=[mpnn_input_config], 
                atom_arrays=[rfd3_out.atom_array]
            )
            
            # Validation Loop per sequence
            for j, mpnn_out in enumerate(mpnn_results):
                seq_name = f"{backbone_name}_seq_{j}"
                out_file = out_path / f"{seq_name}"
                
                # Save MPNN design
                mpnn_out.write_structure(base_path=str(out_file), file_type="cif")
                
                print(f"  Validating {seq_name} with RF3...")
                
                # Prepare RF3 Input from the MPNN output (which has the new sequence)
                # This refolds the sequence predicted by MPNN
                rf3_input = InferenceInput.from_atom_array(mpnn_out.atom_array, example_id=seq_name)
                
                # Run RF3
                rf3_outputs = rf3_engine.run(inputs=rf3_input)
                rf3_result = rf3_outputs[seq_name][0] # Get best/first model
                
                # Metrics
                # 1. pLDDT
                plddt = rf3_result.summary_confidences['overall_plddt']
                
                # 2. RMSD (Backbone vs Refolded)
                # Get structures
                aa_designed = mpnn_out.atom_array
                aa_refolded = rf3_result.atom_array
                
                # Filter to backbone (and ensure canonical order/size matches if possible)
                # Note: RF3 output might have slightly different atoms (e.g. termini handling)
                # Ideally we filter by chain/residue ID intersection
                
                # Simple filter by name
                bb_designed = aa_designed[np.isin(aa_designed.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
                bb_refolded = aa_refolded[np.isin(aa_refolded.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)]
                
                # Ensure equal length for superimpose by trimming to the shorter one (naive)
                # or by common residue IDs (better)
                
                # Get common residue IDs
                res_ids_designed = set(zip(bb_designed.chain_id, bb_designed.res_id, bb_designed.atom_name))
                res_ids_refolded = set(zip(bb_refolded.chain_id, bb_refolded.res_id, bb_refolded.atom_name))
                common_ids = sorted(list(res_ids_designed.intersection(res_ids_refolded)))
                
                if not common_ids:
                    print("    -> Warning: No common backbone atoms found for RMSD calculation.")
                    continue

                # Create masks for common atoms
                # This is slow but robust
                mask_designed = np.array([
                    (c, r, a) in common_ids 
                    for c, r, a in zip(bb_designed.chain_id, bb_designed.res_id, bb_designed.atom_name)
                ])
                mask_refolded = np.array([
                    (c, r, a) in common_ids 
                    for c, r, a in zip(bb_refolded.chain_id, bb_refolded.res_id, bb_refolded.atom_name)
                ])
                
                bb_designed_common = bb_designed[mask_designed]
                bb_refolded_common = bb_refolded[mask_refolded]
                
                if len(bb_designed_common) != len(bb_refolded_common):
                     # Fallback if something is weird with duplicates, just trim to min length
                     min_len = min(len(bb_designed_common), len(bb_refolded_common))
                     bb_designed_common = bb_designed_common[:min_len]
                     bb_refolded_common = bb_refolded_common[:min_len]

                # Superimpose and calc RMSD
                bb_refolded_fitted, _ = superimpose(bb_designed_common, bb_refolded_common)
                rmsd_val = rmsd(bb_designed_common, bb_refolded_fitted)
                
                print(f"    -> pLDDT: {plddt:.2f}")
                print(f"    -> RMSD (vs backbone): {rmsd_val:.2f} A")
                
                # Annotate pLDDT into B-factor for visualization
                if 'atom_plddts' in rf3_result.confidences:
                    # Need to construct is_real_atom mask if possible, or modify usage
                    # But wait, 'atom_plddts' in rf3_result.confidences is already a list of floats (per atom).
                    # The function annotate_atom_array_b_factor_with_plddt expects raw tensors.
                    
                    # If we have the simple list, we can just assign it directly since atom counts match
                    # (RF3 output atom_array should match the length of atom_plddts)
                    
                    plddt_values = np.array(rf3_result.confidences['atom_plddts'])
                    if len(plddt_values) == aa_refolded.array_length():
                        aa_refolded.set_annotation("b_factor", plddt_values)
                        print("    -> Annotated refolded structure with pLDDT in B-factor column")
                    else:
                        print(f"    -> Warning: pLDDT length ({len(plddt_values)}) mismatch with atoms ({aa_refolded.array_length()})")

                # Save refolded structure
                to_cif_file(aa_refolded, str(out_path / f"{seq_name}_refolded.cif"))

                # Combined file for visual inspection
                if write_combined:
                    combined_path = out_path / f"{seq_name}_combined.cif"
                    
                    # Create a stack or simply modify chain IDs to distinguish them in one file
                    # Modifying chain IDs is safer for viewers that don't handle multi-model CIFs well
                    
                    # Deep copy to avoid mutating originals
                    aa_designed_view = aa_designed.copy()
                    aa_refolded_view = aa_refolded.copy()
                    
                    # Add binary annotation to distinguish designed vs refolded
                    # Method 1: Using label_entity_id: 0 = designed, 1 = refolded
                    # This allows easy selection in Mol* via "Entity" coloring/selection
                    designed_entity_ids = np.zeros(len(aa_designed_view), dtype=int)
                    refolded_entity_ids = np.ones(len(aa_refolded_view), dtype=int)
                    aa_designed_view.set_annotation("label_entity_id", designed_entity_ids)
                    aa_refolded_view.set_annotation("label_entity_id", refolded_entity_ids)
                    
                    # Method 2: Using occupancy: 1.0 = designed, 0.5 = refolded
                    # This works with any viewer and can be used for coloring
                    aa_designed_view.set_annotation("occupancy", np.ones(len(aa_designed_view)))
                    aa_refolded_view.set_annotation("occupancy", np.full(len(aa_refolded_view), 0.5))
                    
                    # For alignment: we already computed RMSD on common backbone atoms
                    # We can use the full superposition on the filtered set to align the view
                    # Re-do superposition on the view copies to apply transformation
                    
                    # Recalculate common set for the full view alignment
                    res_ids_des = set(zip(aa_designed_view.chain_id, aa_designed_view.res_id, aa_designed_view.atom_name))
                    res_ids_ref = set(zip(aa_refolded_view.chain_id, aa_refolded_view.res_id, aa_refolded_view.atom_name))
                    common_ids_view = sorted(list(res_ids_des.intersection(res_ids_ref)))
                    
                    if common_ids_view:
                        # Filter to common atoms for superimpose calculation
                        mask_des = np.array([(c,r,a) in common_ids_view for c,r,a in zip(aa_designed_view.chain_id, aa_designed_view.res_id, aa_designed_view.atom_name)])
                        mask_ref = np.array([(c,r,a) in common_ids_view for c,r,a in zip(aa_refolded_view.chain_id, aa_refolded_view.res_id, aa_refolded_view.atom_name)])
                        
                        # Further filter to just backbone for the alignment calculation (more robust)
                        bb_mask_des = np.isin(aa_designed_view.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
                        bb_mask_ref = np.isin(aa_refolded_view.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
                        
                        align_mask_des = mask_des & bb_mask_des
                        align_mask_ref = mask_ref & bb_mask_ref
                        
                        atoms_fixed = aa_designed_view[align_mask_des]
                        atoms_mob = aa_refolded_view[align_mask_ref]
                        
                        # Ensure same length
                        min_len_align = min(len(atoms_fixed), len(atoms_mob))
                        atoms_fixed = atoms_fixed[:min_len_align]
                        atoms_mob = atoms_mob[:min_len_align]
                        
                        if len(atoms_fixed) > 3 and len(atoms_mob) > 3:
                            # Calculate superimposition on the matching subset
                            _, transformation = superimpose(atoms_fixed, atoms_mob)
                            
                            # Apply the same transformation to the full refolded structure
                            aa_refolded_view = transformation.apply(aa_refolded_view)
                    
                    # To save in one file, we can append them.
                    # A trick for viewers: shift chain IDs of the second structure
                    # E.g. A -> B, B -> C ... or just ensure uniqueness.
                    # For simplicity here, we will save as a two-model CIF file if atom counts match,
                    # OR if they don't, we append them as different chains in one model.
                    
                    # Method: Append as new chains.
                    # Get existing chains
                    existing_chains = set(aa_designed_view.chain_id)
                    
                    # Rename chains in refolded structure to avoid collision
                    # Simple strategy: suffix 'R' or shift letters
                    new_chains = []
                    used_chains = set(existing_chains)
                    
                    import string
                    alphabet = string.ascii_uppercase + string.ascii_lowercase
                    
                    # Mapping for old chain -> new chain
                    chain_map = {}
                    for c in sorted(list(set(aa_refolded_view.chain_id))):
                        # Find a new chain ID
                        for char in alphabet:
                            if char not in used_chains:
                                chain_map[c] = char
                                used_chains.add(char)
                                break
                        else:
                            # Fallback if run out of chars
                            chain_map[c] = "X" 
                            
                    # Apply mapping
                    for i in range(len(aa_refolded_view)):
                        c = aa_refolded_view.chain_id[i]
                        if c in chain_map:
                            aa_refolded_view.chain_id[i] = chain_map[c]
                            
                    # Combine
                    combined_array = aa_designed_view + aa_refolded_view
                    
                    to_cif_file(combined_array, str(combined_path))
                    print(f"    -> Combined view saved: {combined_path.name}")

    print(f"\nDone! Outputs saved to {out_path.resolve()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="inference_outputs/validation_demo")
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--num_designs", type=int, default=1)
    parser.add_argument("--symmetry", type=str, default=None)
    parser.add_argument("--mpnn_batch_size", type=int, default=5) # Reduced default for validation speed
    parser.add_argument("--designed_folded_file", type=bool, default=True, help="Generate a combined CIF of designed and refolded structures")
    args = parser.parse_args()

    run_design_validate(
        out_dir=args.out_dir,
        length=args.length,
        num_designs=args.num_designs,
        symmetry_id=args.symmetry,
        mpnn_batch_size=args.mpnn_batch_size,
        write_combined=args.designed_folded_file
    )

if __name__ == "__main__":
    main()

