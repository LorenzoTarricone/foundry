# Flow_diagram.md — RFDiffusion3 Architecture Visual Guide

**Auto-generated:** 2026-02-03
**Purpose:** Visual ASCII diagrams explaining the preprocessing pipeline, TokenInitializer, and diffusion module operations.

---

## Table of Contents

1. [High-Level Data Flow](#1-high-level-data-flow)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [TokenInitializer Deep Dive](#3-tokeninitializer-deep-dive)
4. [Diffusion Module Architecture](#4-diffusion-module-architecture)
5. [Upcast/Downcast Operations](#5-upcastdowncast-operations)
6. [Complete Single Denoising Step](#6-complete-single-denoising-step)

---

## 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RFDiffusion3 Data Flow                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

  INPUT                     PREPROCESSING                    MODEL                OUTPUT
  ─────                     ─────────────                    ─────                ──────

┌─────────────┐          ┌──────────────────┐          ┌──────────────┐      ┌──────────┐
│ JSON/YAML   │          │ Transform        │          │              │      │          │
│ Spec +      │─────────▶│ Pipeline         │─────────▶│ RFD3 Model   │─────▶│ .cif.gz  │
│ PDB/CIF     │          │ (20+ transforms) │          │ (200 steps)  │      │ .json    │
└─────────────┘          └──────────────────┘          └──────────────┘      └──────────┘
       │                          │                           │
       │                          │                           │
       ▼                          ▼                           ▼
┌─────────────┐          ┌──────────────────┐          ┌──────────────┐
│DesignInput  │          │ Features Dict    │          │ Denoised     │
│Specification│          │ f = {restype,    │          │ Coordinates  │
│ (Pydantic)  │          │  chain_id, ...}  │          │ X_L [D,L,3]  │
└─────────────┘          └──────────────────┘          └──────────────┘
```

---

## 2. Preprocessing Pipeline

The transform pipeline converts raw input structures into model-ready features.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Transform Pipeline Overview                               │
│                     (pipelines.py:build_atom14_base_pipeline)                   │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────┐
                    │   Raw Input       │
                    │   (AtomArray)     │
                    └─────────┬─────────┘
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │         PHASE 1: PRE-CROP TRANSFORMS              │
    │         (Structure cleaning & preparation)        │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │  1. RemoveHydrogens()              - Strip H atoms                │
    │  2. FilterToSpecifiedPNUnits()     - Remove clashing chains       │
    │  3. RemoveTerminalOxygen()         - Clean termini                │
    │  4. RemoveUnresolvedPNUnits()      - Remove unresolved residues   │
    │  5. MaskPolymerResiduesWithUnresolvedFrameAtoms()                 │
    │  6. FlagAndReassignCovalentModifications()                        │
    │  7. AtomizeByCCDName()             - Handle ligands               │
    │  8. AddWithinChainInstanceResIdx() - Add chain indexing           │
    │  9. AddWithinPolyResIdxAnnotation()                               │
    │ 10. AddProteinTerminiAnnotation()  - Mark N/C termini             │
    └───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────┼─────────────────────────┐
    │         PHASE 2: CONDITIONING TRANSFORMS          │
    │         (Add conditioning annotations)            │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │ 11. CalculateRASA()                - Solvent accessibility        │
    │ 12. CalculateHbondsPlus()          - H-bond networks              │
    │ 13. AddPPIHotspotFeature()         - PPI hotspots                 │
    │ 14. Add1DSSFeature()               - Secondary structure          │
    │ 15. AddGlobalIsNonLoopyFeature()   - Non-loopy conditioning       │
    └───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────┼─────────────────────────┐
    │         PHASE 3: FEATURE ENGINEERING              │
    │         (Build model input features)              │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │ 16. EncodeAF3TokenLevelFeatures()  - Token-level encoding         │
    │ 17. CreateDesignReferenceFeatures()- Reference conformers         │
    │ 18. FeaturizeAtoms()               - Atom-level features          │
    │ 19. FeaturizepLDDT()               - pLDDT conditioning           │
    │ 20. AddAF3TokenBondFeatures()      - Bond information             │
    │ 21. AddSymmetryFeats()             - Symmetry information         │
    └───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────┼─────────────────────────┐
    │         PHASE 4: VIRTUAL ATOMS & BATCHING         │
    │         (Padding and diffusion preparation)       │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │ 22. PadTokensWithVirtualAtoms()    - Pad to n_atoms_per_token     │
    │ 23. ComputeAtomToTokenMap()        - Build tok_idx mapping        │
    │ 24. ConvertToTorch()               - NumPy → PyTorch tensors      │
    │ 25. AggregateFeaturesLikeAF3WithoutMSA()                          │
    │ 26. BatchStructuresForDiffusionNoising()                          │
    │ 27. SampleEDMNoise()               - Sample noise schedule        │
    └───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   Features Dict   │
                    │   f = {...}       │
                    └───────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Features Dictionary (f)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────────────────────────────────────────────────────────┐
  │ IDENTITY FEATURES (per-atom)                                                │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ restype          : [L]      Residue type index (0-31)                      │
  │ ref_element      : [L]      Element type                                    │
  │ chain_id         : [L]      Chain index                                     │
  │ res_id           : [L]      Residue number                                  │
  └────────────────────────────────────────────────────────────────────────────┘
                              │
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ STRUCTURAL FEATURES                                                         │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ is_ca            : [L]      CA atom mask (defines tokens: I = sum(is_ca))  │
  │ is_backbone      : [L]      Backbone atom mask                              │
  │ ref_pos          : [L, 3]   Reference coordinates                           │
  │ ref_space_uid    : [L]      Unique ID for spatial grouping                  │
  │ token_bonds      : [I, I]   Token-level bond connectivity                   │
  └────────────────────────────────────────────────────────────────────────────┘
                              │
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ CONDITIONING FEATURES                                                       │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ is_motif_atom_with_fixed_coord : [L]   Fixed 3D position mask              │
  │ is_motif_atom_with_fixed_seq   : [L]   Fixed sequence mask                 │
  │ is_motif_atom_unindexed        : [L]   Unindexed motif mask                │
  │ rasa_bin                       : [L]   RASA category (0-3)                 │
  │ active_donor                   : [L]   H-bond donor conditioning           │
  │ active_acceptor                : [L]   H-bond acceptor conditioning        │
  │ ref_plddt                      : [L]   Reference pLDDT values              │
  │ partial_t                      : [L]   Partial diffusion noise level       │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. TokenInitializer Deep Dive

The TokenInitializer converts raw features into the initial embeddings for the model.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TokenInitializer Architecture                               │
│                         (encoders.py:TokenInitializer)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────┐
                         │  Features Dict f  │
                         │                   │
                         │ L atoms, I tokens │
                         └─────────┬─────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│ atom_1d_embedder│     │ token_1d_embedder│    │ atom_1d_embedder_2 │
│       _1        │     │                  │     │                    │
└────────┬────────┘     └────────┬────────┘     └─────────┬──────────┘
         │                       │                         │
         ▼                       ▼                         │
    [L, c_s]                [I, c_s]                       │
         │                       │                         │
         │                       ▼                         │
         │              ┌─────────────────┐                │
         │              │transition_post_ │                │
         │              │     token       │                │
         │              │  (Transition)   │                │
         │              └────────┬────────┘                │
         │                       │                         │
         │                       ▼                         │
         │                  [I, c_s]                       │
         │                       │                         │
         └───────────┬───────────┘                         │
                     │                                     │
                     ▼                                     │
            ┌─────────────────┐                            │
            │   downcast_atom │                            │
            │   (Downcast)    │                            │
            │                 │                            │
            │ Atom→Token Pool │                            │
            └────────┬────────┘                            │
                     │                                     │
                     ▼                                     │
                [I, c_s]                                   │
                     │                                     │
                     ▼                                     │
            ┌─────────────────┐                            │
            │transition_post_ │                            │
            │     atom        │                            │
            └────────┬────────┘                            │
                     │                                     │
                     ▼                                     │
            ┌─────────────────┐                            │
            │  process_s_init │                            │
            │ (RMSNorm+Linear)│                            │
            └────────┬────────┘                            │
                     │                                     │
                     ▼                                     │
              S_I [I, c_s]  ◀──── Initial Single Features  │
                     │                                     │
    ┌────────────────┼────────────────┐                    │
    │                │                │                    │
    ▼                ▼                ▼                    ▼
┌────────┐    ┌────────────┐    ┌────────────────┐   ┌──────────┐
│to_z_   │    │to_z_init_j │    │ relative_      │   │atom_1d_  │
│init_i  │    │            │    │ position_      │   │embedder_2│
└───┬────┘    └─────┬──────┘    │ encoding       │   └────┬─────┘
    │               │           └───────┬────────┘        │
    ▼               ▼                   │                 ▼
[I,1,c_z]      [1,I,c_z]               ▼            Q_L_init
    │               │           [I, I, c_z]         [L, c_atom]
    │               │                   │                 │
    └───────┬───────┘                   │                 │
            │                           │                 │
            ▼                           │                 │
     ┌─────────────┐                    │                 │
     │  Z_i + Z_j  │                    │                 │
     │ (broadcast) │                    │                 │
     └──────┬──────┘                    │                 │
            │                           │                 │
            ▼                           │                 │
      [I, I, c_z]                       │                 │
            │                           │                 │
            └───────────┬───────────────┘                 │
                        │                                 │
                        ▼                                 │
               ┌─────────────────┐                        │
               │    Z_init_II    │                        │
               │ = Z_i + Z_j +   │                        │
               │   RPE + bonds   │                        │
               │   + ref_pos     │                        │
               └────────┬────────┘                        │
                        │                                 │
                        ▼                                 │
         ┌──────────────────────────┐                     │
         │   transformer_stack      │                     │
         │   (PairformerBlock ×N)   │                     │
         │                          │                     │
         │ ┌──────────────────────┐ │                     │
         │ │ for block in stack:  │ │                     │
         │ │  S_I, Z_II = block(  │ │                     │
         │ │    S_I, Z_II)        │ │                     │
         │ └──────────────────────┘ │                     │
         └────────────┬─────────────┘                     │
                      │                                   │
         ┌────────────┴────────────┐                      │
         │                         │                      │
         ▼                         ▼                      │
    S_I [I, c_s]           Z_init_II [I, I, c_z]         │
         │                         │                      │
         │                         ▼                      │
         │              ┌──────────────────┐              │
         │              │ + RPE2 + concat  │              │
         │              │ + process_z_init │              │
         │              │ + transitions    │              │
         │              └────────┬─────────┘              │
         │                       │                        │
         │                       ▼                        │
         │              Z_II [I, I, c_z]                  │
         │                       │                        │
         ▼                       │                        │
┌─────────────────┐              │                        │
│ process_s_trunk │              │                        │
│ (RMSNorm+Linear)│              │                        │
└────────┬────────┘              │                        │
         │                       │                        │
         ▼                       │                        │
    [I, c_atom]                  │                        │
         │                       │                        │
         │  ┌────────────────────┘                        │
         │  │                                             │
         │  │  ┌──────────────────────────────────────────┘
         │  │  │
         ▼  ▼  ▼
    ┌─────────────────────────────────────────────────────┐
    │            C_L = Q_L_init + S_trunk[tok_idx]        │
    │                                                     │
    │  Atom conditioning: project token features to atoms │
    └───────────────────────────┬─────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────────────────┐
                │         TokenInitializer Output       │
                │                                       │
                │  Q_L_init : [L, c_atom]  Atom feats   │
                │  C_L      : [L, c_atom]  Conditioned  │
                │  S_I      : [I, c_s]     Single feats │
                │  Z_II     : [I, I, c_z]  Pair feats   │
                │  P_LL     : [L, L, c_atompair] (opt)  │
                └───────────────────────────────────────┘
```

### PairformerBlock Detail

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PairformerBlock                                          │
│                    (pairformer_layers.py)                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

           S_I [I, c_s]              Z_II [I, I, c_z]
                │                          │
                │                          ▼
                │               ┌──────────────────────┐
                │               │   z_transition       │
                │               │   (SwiGLU MLP)       │
                │               │                      │
                │               │ Z = Z + MLP(Z)       │
                │               └───────────┬──────────┘
                │                           │
                │                           ▼
                │               Z_II [I, I, c_z] (updated)
                │                           │
                ▼                           │
    ┌───────────────────────┐               │
    │ attention_pair_bias   │               │
    │                       │◀──────────────┘
    │ Multi-head attention  │   (Z as pair bias)
    │ with pair bias        │
    │                       │
    │ Q, K, V from S_I      │
    │ Bias from Z_II        │
    └───────────┬───────────┘
                │
                ▼
         S_I [I, c_s]
                │
                ▼
    ┌───────────────────────┐
    │    s_transition       │
    │    (SwiGLU MLP)       │
    │                       │
    │ S = S + MLP(S)        │
    └───────────┬───────────┘
                │
                ▼
    S_I [I, c_s] (updated)

    ─────────────────────────────────────────────────────

    OUTPUT: (S_I, Z_II) both updated
```

---

## 4. Diffusion Module Architecture

The RFD3DiffusionModule processes noisy coordinates through encoder, transformer, and decoder.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      RFD3DiffusionModule Architecture                            │
│                      (RFD3_diffusion_module.py)                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

    INPUTS
    ──────
    X_noisy_L : [D, L, 3]     Noisy atom coordinates
    t         : [D]           Noise level (sigma)
    f         : dict          Features

    FROM TokenInitializer:
    S_I       : [I, c_s]      Single token features
    Z_II      : [I, I, c_z]   Pair token features
    Q_L_init  : [L, c_atom]   Initial atom features
    C_L       : [L, c_atom]   Conditioned atom features


                    X_noisy_L [D, L, 3]
                            │
                            ▼
                ┌───────────────────────┐
                │   scale_positions_in  │
                │                       │
                │ if EDM:               │
                │   R = X / sqrt(t²+σ²) │
                └───────────┬───────────┘
                            │
                            ▼
                    R_noisy_L [D, L, 3]
                            │
                            ▼
                ┌───────────────────────┐
                │      process_r        │
                │   Linear(3 → c_atom)  │
                └───────────┬───────────┘
                            │
                            ▼
                     [D, L, c_atom]
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  │                  ▼
    ┌─────────────┐         │         ┌─────────────────┐
    │ process_c   │         │         │  process_time_  │
    │ (RMSNorm +  │         │         │                 │
    │  Linear)    │         │         │ Fourier embed   │
    └──────┬──────┘         │         │ + Linear        │
           │                │         └────────┬────────┘
           ▼                │                  │
    C_L [D, L, c_atom]      │                  ▼
           │                │           [D, 1, c_atom]
           │                │                  │
           │                │                  │
           │                ▼                  │
           │      Q_L [D, L, c_atom]           │
           │                │                  │
           │                │                  │
           │     ┌──────────┴──────────┐       │
           │     │                     │       │
           ▼     ▼                     ▼       ▼
    ┌──────────────────────────────────────────────────┐
    │                    ENCODER                        │
    │              (LocalAtomTransformer)               │
    │                                                   │
    │  Q_L = Q_L + time_embed                          │
    │  Q_L = attention(Q_L, C_L, P_LL)                 │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐ │
    │  │ LocalAttentionPairBias                      │ │
    │  │   - Query: Q_L [D, L, c_atom]              │ │
    │  │   - Key/Value: C_L [D, L, c_atom]          │ │
    │  │   - Pair bias: P_LL [L, L, c_atompair]     │ │
    │  │   - Local attention window: n_seq_neighbors │ │
    │  └─────────────────────────────────────────────┘ │
    └───────────────────────┬──────────────────────────┘
                            │
                            ▼
                Q_L [D, L, c_atom] (encoded atoms)
                            │
                            │
    ┌───────────────────────┴───────────────────────────┐
    │                    DOWNCAST_Q                      │
    │               (Atom → Token pooling)              │
    │                                                   │
    │  A_I = mean_pool(project(Q_L), tok_idx)          │
    │                                                   │
    │  Pool atoms within each token, then project       │
    └───────────────────────┬───────────────────────────┘
                            │
                            ▼
                   A_I [D, I, c_token]
                            │
    ┌───────────────────────┴───────────────────────────┐
    │              DIFFUSION_TOKEN_ENCODER              │
    │             (DiffusionTokenEncoder)               │
    │                                                   │
    │  Refine S_I and Z_II with A_I context            │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐ │
    │  │ - Process distogram D_II (self-conditioning)│ │
    │  │ - Pairformer blocks with A_I + S_I          │ │
    │  │ - Update Z_II with pair features            │ │
    │  └─────────────────────────────────────────────┘ │
    └───────────────────────┬───────────────────────────┘
                            │
                            ▼
        S_I [I, c_s], Z_II [I, I, c_z] (refined)
                            │
    ┌───────────────────────┴───────────────────────────┐
    │              DIFFUSION_TRANSFORMER                │
    │            (LocalTokenTransformer)                │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐ │
    │  │ for block in transformer_blocks:            │ │
    │  │   A_I = A_I + attention(A_I, S_I, Z_II)    │ │
    │  │   A_I = A_I + transition(A_I)              │ │
    │  └─────────────────────────────────────────────┘ │
    │                                                   │
    │  Uses conditioned transition blocks with S_I     │
    │  and attention with Z_II as pair bias            │
    └───────────────────────┬───────────────────────────┘
                            │
                            ▼
                   A_I [D, I, c_token] (transformed)
                            │
    ┌───────────────────────┴───────────────────────────┐
    │                      DECODER                       │
    │            (CompactStreamingDecoder)              │
    │                                                   │
    │  ┌─────────────────────────────────────────────┐ │
    │  │ for i in range(n_blocks):                   │ │
    │  │   Q_L = upcast(Q_L, A_I, tok_idx)          │ │
    │  │   Q_L = atom_transformer(Q_L, C_L, P_LL)   │ │
    │  │                                             │ │
    │  │ A_I = downcast(Q_L, A_I, S_I, tok_idx)     │ │
    │  └─────────────────────────────────────────────┘ │
    │                                                   │
    │  Iterative upcast → attention → downcast         │
    └───────────────────────┬───────────────────────────┘
                            │
                            ▼
                Q_L [D, L, c_atom] (decoded atoms)
                            │
    ┌───────────────────────┴───────────────────────────┐
    │                  OUTPUT HEADS                      │
    └───────────────────────┬───────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   to_r_update   │ │  sequence_head  │ │  (other heads)  │
│                 │ │                 │ │                 │
│ RMSNorm+Linear  │ │ Linear to vocab │ │                 │
│ (c_atom → 3)    │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └─────────────────┘
         │                   │
         ▼                   ▼
  R_update_L [D,L,3]  seq_logits [D,I,vocab]
         │
         ▼
┌─────────────────────────────┐
│     scale_positions_out     │
│                             │
│ if EDM:                     │
│   X_out = (σ²/(σ²+t²))·X +  │
│           (σt/√(σ²+t²))·R   │
└─────────────┬───────────────┘
              │
              ▼
       X_denoised_L [D, L, 3]
```

---

## 5. Upcast/Downcast Operations

These operations transfer information between atom-level (L) and token-level (I) representations.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Upcast / Downcast Operations                                │
└─────────────────────────────────────────────────────────────────────────────────┘


    DOWNCAST: Atoms → Tokens                  UPCAST: Tokens → Atoms
    ─────────────────────────                 ─────────────────────────

    Q_L [B, L, c_atom]                        Q_L [B, L, c_atom]
    A_I [B, I, c_token]                       A_I [B, I, c_token]
    tok_idx [L] (maps atom → token)           tok_idx [L] (maps atom → token)

         Q_L [B, L, c_atom]                        A_I [B, I, c_token]
                │                                         │
                ▼                                         ▼
    ┌───────────────────────┐                 ┌───────────────────────┐
    │  Reshape by tok_idx   │                 │  Index by tok_idx     │
    │                       │                 │                       │
    │  Group atoms per token│                 │  A_I[:, tok_idx, :]   │
    │  Q_IA [B, I, A, c]    │                 │  → [B, L, c_token]    │
    └───────────┬───────────┘                 └───────────┬───────────┘
                │                                         │
                ▼                                         ▼
    ┌───────────────────────┐                 ┌───────────────────────┐
    │     Pooling           │                 │     Projection        │
    │                       │                 │                       │
    │ if method=="mean":    │                 │ Linear(c_token,c_atom)│
    │   project + mean_pool │                 │                       │
    │                       │                 │                       │
    │ if method=="cross_attn│                 │                       │
    │   GatedCrossAttention │                 │                       │
    │   (A_I queries atoms) │                 │                       │
    └───────────┬───────────┘                 └───────────┬───────────┘
                │                                         │
                ▼                                         ▼
    ┌───────────────────────┐                 ┌───────────────────────┐
    │   Residual + S_I      │                 │      Residual         │
    │                       │                 │                       │
    │ A_I = A_I + A_I_update│                 │ Q_L = Q_L + Q_update  │
    │ A_I = A_I + process_s │                 │                       │
    └───────────┬───────────┘                 └───────────┬───────────┘
                │                                         │
                ▼                                         ▼
    A_I [B, I, c_token] (updated)             Q_L [B, L, c_atom] (updated)



    VISUAL: Token-Atom Relationship
    ────────────────────────────────

    Tokens (I=4):      T0          T1          T2          T3
                       │           │           │           │
                       │           │           │           │
    Atoms (L=11):    A0 A1 A2    A3 A4 A5    A6 A7      A8 A9 A10
                     └──┬──┘     └──┬──┘     └─┬─┘      └──┬──┘
                        │           │          │           │
    tok_idx:     [0,0,0,       1,1,1,       2,2,        3,3,3]

    DOWNCAST: Pool atoms A0,A1,A2 → T0
              Pool atoms A3,A4,A5 → T1
              etc.

    UPCAST:   Broadcast T0 → A0,A1,A2
              Broadcast T1 → A3,A4,A5
              etc.
```

---

## 6. Complete Single Denoising Step

This shows the full data flow for one denoising step (one of 200 steps).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Complete Single Denoising Step                               │
│                    (inference_sampler.py:421-627)                               │
└─────────────────────────────────────────────────────────────────────────────────┘


INPUTS AT STEP n:
─────────────────
    X_L           : [D, L, 3]     Current coordinates (from step n-1)
    c_t_minus_1   : scalar        Previous noise level
    c_t           : scalar        Target noise level
    f             : dict          Features (constant)
    initializer_outputs : dict    From TokenInitializer (constant)


                    X_L [D, L, 3]
                         │
    ┌────────────────────┼────────────────────┐
    │       STEP 1: Compute noise level       │
    └────────────────────┼────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ gamma = 0.6 if c_t > │
              │         gamma_min    │
              │         else 0       │
              │                      │
              │ t_hat = c_t_minus_1  │
              │       * (gamma + 1)  │
              └───────────┬──────────┘
                          │
                          ▼
                   t_hat (effective noise)
                          │
    ┌─────────────────────┼─────────────────────┐
    │       STEP 2: Add stochastic noise        │
    └─────────────────────┼─────────────────────┘
                          │
                          ▼
              ┌───────────────────────────────────┐
              │ epsilon_L = noise_scale *         │
              │   sqrt(t_hat² - c_t_minus_1²) *   │
              │   randn(D, L, 3)                  │
              │                                   │
              │ epsilon_L[is_fixed] = 0           │
              └───────────────┬───────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │ X_noisy_L = X_L + epsilon_L       │
              └───────────────┬───────────────────┘
                              │
                              ▼
                    X_noisy_L [D, L, 3]
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │         STEP 3: Neural network forward            │
    │              (diffusion_module)                   │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │       RFD3DiffusionModule         │
              │                                   │
              │  X_noisy_L ──┐                    │
              │  t_hat ──────┼─▶ X_denoised_L    │
              │  f ──────────┤                    │
              │  S_I, Z_II ──┘                    │
              │                                   │
              │  (See Section 4 for internals)    │
              └───────────────┬───────────────────┘
                              │
                              ▼
                  X_denoised_L [D, L, 3]
                              │
    ┌─────────────────────────┼─────────────────────────┐
    │         STEP 4: ODE update step                   │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────────┐
              │ delta_L = (X_noisy - X_denoised)  │
              │           / t_hat                 │
              │                                   │
              │ d_t = c_t - t_hat                 │
              │                                   │
              │ X_L = X_noisy_L +                 │
              │       step_scale * d_t * delta_L  │
              └───────────────┬───────────────────┘
                              │
                              ▼
                    X_L [D, L, 3]
                         │
    ┌────────────────────┼────────────────────┐
    │   STEP 5: Save trajectory (optional)    │
    └────────────────────┼────────────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │ if dump_trajectories:    │
              │   X_noisy_traj.append()  │
              │   X_denoised_traj.append()│
              └──────────────────────────┘
                         │
                         │
                         ▼
               ┌─────────────────────┐
               │  Continue to next   │
               │  step (n+1)         │
               └─────────────────────┘


    ═══════════════════════════════════════════════════════════════════

    NOISE SCHEDULE VISUALIZATION (200 steps):

    Step:     0 ─────────────────────────────────────────────────▶ 199
    t_hat:  160.0 ─────────────────────────────────────────────▶ 0.0004
              │                                                     │
              │   High noise                         Low noise      │
              │   (random)                           (clean)        │
              │                                                     │
              ▼                                                     ▼

    X_L:   random ──────── gradually denoised ──────────▶ structure

    ═══════════════════════════════════════════════════════════════════
```

---

## Appendix: Tensor Shape Reference

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Quick Tensor Shape Reference                            │
└─────────────────────────────────────────────────────────────────────────────────┘

    Symbol    Default    Description
    ──────    ───────    ───────────────────────────────────────
    D         8-16       Diffusion batch size (parallel samples)
    L         varies     Number of atoms in structure
    I         varies     Number of tokens (≈ number of residues)
    c_s       256        Single token feature dimension
    c_z       128        Pair token feature dimension
    c_atom    128        Atom feature dimension
    c_atompair 128       Atom pair feature dimension
    c_token   256        Token activation dimension


    TENSOR                 SHAPE              DESCRIPTION
    ──────────────────────────────────────────────────────────────

    Coordinates:
    X_L                    [D, L, 3]          Atom coordinates
    X_noisy_L              [D, L, 3]          Noised coordinates
    X_denoised_L           [D, L, 3]          Model output coords

    Single Features:
    S_I                    [I, c_s]           Token single features

    Pair Features:
    Z_II                   [I, I, c_z]        Token pair features
    P_LL                   [L, L, c_atompair] Atom pair features

    Atom Features:
    Q_L                    [D, L, c_atom]     Atom query features
    C_L                    [D, L, c_atom]     Atom condition features
    Q_L_init               [L, c_atom]        Initial atom features

    Token Activations:
    A_I                    [D, I, c_token]    Token activations

    Mappings:
    tok_idx                [L]                Atom → Token index

    Noise:
    t                      [D]                Noise levels
    epsilon_L              [D, L, 3]          Random noise

    Sequence:
    sequence_logits_I      [D, I, vocab]      Sequence predictions
```

---

*End of Flow_diagram.md*
