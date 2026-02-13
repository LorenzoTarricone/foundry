# Flow_diagram_parallel.md — RFDiffusion3 Parallel Mode Visual Guide

**Auto-generated:** 2026-02-11
**Purpose:** Visual ASCII diagrams explaining the parallel (multi-GPU streaming) mode architecture, contrasting with the standard mode documented in Flow_diagram.md.

---

## Table of Contents

1. [High-Level Parallel Data Flow](#1-high-level-parallel-data-flow)
2. [Stripe Cross-Attention: The Core Idea](#2-stripe-cross-attention-the-core-idea)
3. [TokenInitializer — Streaming Mode](#3-tokeninitializer--streaming-mode)
4. [DiffusionTokenEncoder — Streaming Mode](#4-diffusiontokenencoder--streaming-mode)
5. [Diffusion Module — Parallel Forward](#5-diffusion-module--parallel-forward)
6. [All-Gather Synchronization](#6-all-gather-synchronization)
7. [Complete Single Denoising Step (Parallel)](#7-complete-single-denoising-step-parallel)
8. [Tensor Shape Reference (Parallel Mode)](#8-tensor-shape-reference-parallel-mode)

---

## 1. High-Level Parallel Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     RFDiffusion3 Parallel Mode Data Flow                         │
│                     (N GPUs, NCCL-synchronized)                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

  STANDARD (single GPU)                    PARALLEL (N GPUs)
  ─────────────────────                    ──────────────────

  Z_II: [I, I, c_z]                       Z_chunk: [I_par, I, c_z] per GPU
  P_LL: [L, L, c_atompair]                P_chunk: [L_par, L, c_atompair] per GPU
  Attention: self-attention                Attention: cross-attention (chunk Q → all K)
  Sync: None                              Sync: all_gather after each Pairformer block

  Memory: O(I² + L²)                      Memory: O(I²/N + L²/N) per GPU


┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│  ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌───────┐ │
│  │ JSON/YAML   │     │ Transform        │     │ RFD3 Model   │     │.cif.gz│ │
│  │ Spec +      │────▶│ Pipeline         │────▶│ (200 steps)  │────▶│.json  │ │
│  │ PDB/CIF     │     │ (same as std)    │     │              │     │       │ │
│  └─────────────┘     └──────────────────┘     └──────────────┘     └───────┘ │
│                                                       │                       │
│                                                       ▼                       │
│                              ┌──────────────────────────────────────────┐     │
│                              │  Parallel Dispatch:                       │     │
│                              │                                          │     │
│                              │  GPU 0: queries [0, I_par)               │     │
│                              │  GPU 1: queries [I_par, 2·I_par)         │     │
│                              │  ...                                     │     │
│                              │  GPU N-1: queries [(N-1)·I_par, I)       │     │
│                              │                                          │     │
│                              │  Each GPU computes its chunk, then       │     │
│                              │  all_gather to reconstruct full result   │     │
│                              └──────────────────────────────────────────┘     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stripe Cross-Attention: The Core Idea

The fundamental parallel strategy is to split **self-attention queries** across GPUs while keeping **all keys** on every GPU. Each GPU computes a "stripe" of the attention matrix.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     Standard vs Parallel Attention                               │
└─────────────────────────────────────────────────────────────────────────────────┘

  STANDARD MODE (1 GPU):                    PARALLEL MODE (N=4 GPUs):
  ──────────────────────                    ─────────────────────────

  Queries (I)                               Queries split across GPUs
  ┌─────────────────────┐                   ┌─────────────────────┐
  │ Q₀  Q₁  Q₂ ... Qᵢ  │                   │ Q₀...Qₚ │ GPU 0    │ I_par = I/N
  │                     │                   ├──────────┤          │
  │ Full self-attention │                   │Qₚ...Q₂ₚ │ GPU 1    │
  │ Q[I] × K[I]        │                   ├──────────┤          │
  │ → Attn[I, I]       │                   │Q₂ₚ..Q₃ₚ │ GPU 2    │
  │                     │                   ├──────────┤          │
  │ Memory: O(I²)      │                   │Q₃ₚ...Qᵢ │ GPU 3    │
  └─────────────────────┘                   └──────────┘

                                            Keys (ALL on every GPU)
                                            ┌─────────────────────┐
                                            │ K₀  K₁  K₂ ... Kᵢ  │ Full keys
                                            └─────────────────────┘

                                            Each GPU computes:
                                            Q[I_par] × K[I] → Attn[I_par, I]
                                            Memory per GPU: O(I²/N)

  ═══════════════════════════════════════════════════════════════════

  Attention Matrix Visualization (I=12, N=3 GPUs):

  Standard: Full I×I                        Parallel: Stripes

  K → K₀ K₁ K₂ K₃ K₄ K₅ K₆ K₇ K₈ K₉ K₁₀K₁₁    K → K₀ K₁ K₂ K₃ K₄ K₅ K₆ K₇ K₈ K₉ K₁₀K₁₁
  Q₀  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₀  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ┐
  Q₁  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₁  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  │ GPU 0
  Q₂  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₂  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  │
  Q₃  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₃  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ┘
  Q₄  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₄  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ┐
  Q₅  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₅  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  │ GPU 1
  Q₆  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₆  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  │
  Q₇  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₇  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ┘
  Q₈  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₈  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ┐
  Q₉  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₉  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  │ GPU 2
  Q₁₀ ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₁₀ ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  │
  Q₁₁ ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■         Q₁₁ ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ■  ┘

  1 GPU: 144 entries                        3 GPUs: 48 entries each = 144 total
```

---

## 3. TokenInitializer — Streaming Mode

In parallel mode, the TokenInitializer never materializes the full `[I, I, c_z]` tensor. Instead, each GPU computes only its query rows `[I_par, I, c_z]`.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                TokenInitializer — Streaming Mode (_forward_streaming)            │
│                         (encoders.py:764)                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

                         ┌───────────────────┐
                         │  Features Dict f  │
                         │  L atoms, I tokens│
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
         └───────────┬───────────┘                         │
                     │                                     │
                     ▼                                     │
            ┌─────────────────┐                            │
            │   downcast_atom │                            │
            │   + transitions │                            │
            │   + process_s   │                            │
            └────────┬────────┘                            │
                     │                                     │
                     ▼                                     │
              S_I [I, c_s]  (FULL, on all GPUs)            │
                     │                                     │
  ┌──────────────────┴─────────────────────────────────┐   │
  │                                                    │   │
  │  ╔════════════════════════════════════════════╗     │   │
  │  ║  PARALLEL: Each GPU computes its Z chunk   ║     │   │
  │  ║  GPU k computes Z[start_k:end_k, :, c_z]  ║     │   │
  │  ╚════════════════════════════════════════════╝     │   │
  │                                                    │   │
  │  Step 1: Z_i = to_z_init_i(S_I[start:end])        │   │
  │          Z_j = to_z_init_j(S_I) (pre-computed)     │   │
  │          Z_chunk = Z_i + Z_j                       │   │
  │                   [I_par,1,c_z] + [1,I,c_z]        │   │
  │                   = [I_par, I, c_z]                │   │
  │                                                    │   │
  │  Step 2: + RPE.forward_chunk(f, start, end)        │   │
  │  Step 3: + process_token_bonds(bonds[start:end,:]) │   │
  │  Step 4: + ref_pos_embedder_tok.forward_chunk()    │   │
  │                                                    │   │
  │  ┌──────────────────────────────────────────────┐  │   │
  │  │  _process_s_through_transformer_stack        │  │   │
  │  │                                              │  │   │
  │  │  for block in transformer_stack:             │  │   │
  │  │    Z_chunk = Z_chunk + z_transition(Z_chunk) │  │   │
  │  │    S_I_chunk = S_I[start:end]                │  │   │
  │  │    S_I_chunk += attn.forward_chunked(        │  │   │
  │  │        Q=S_I_chunk,     [I_par, c_s]         │  │   │
  │  │        K=S_I,           [I, c_s]             │  │   │
  │  │        Z_bias=Z_chunk)  [I_par, I, c_z]      │  │   │
  │  │    S_I_chunk += s_transition(S_I_chunk)      │  │   │
  │  │                                              │  │   │
  │  │    ┌─────────────────────────────────────┐   │  │   │
  │  │    │ ALL_GATHER S_I_chunk → full S_I     │   │  │   │
  │  │    │ (synchronize across GPUs)           │   │  │   │
  │  │    └─────────────────────────────────────┘   │  │   │
  │  └──────────────────────────────────────────────┘  │   │
  │                                                    │   │
  │  Step 5: + RPE2.forward_chunk()                    │   │
  │          Z_chunk = cat([Z_chunk, RPE2], dim=-1)    │   │
  │          Z_chunk = process_z_init(Z_chunk)         │   │
  │  Step 6: Z_chunk += transition_1[0](Z_chunk)       │   │
  │          Z_chunk += transition_1[1](Z_chunk)       │   │
  │                                                    │   │
  └──────────────────────┬─────────────────────────────┘   │
                         │                                 │
                         ▼                                 ▼
        ┌───────────────────────────────────────────────────────┐
        │              TokenInitializer Output (Streaming)      │
        │                                                       │
        │  Q_L_init          : [L, c_atom]     Atom features    │
        │  C_L               : [L, c_atom]     Conditioned      │
        │  S_I               : [I, c_s]        Single (FULL)    │
        │  Z_II (= Z_chunk)  : [I_par, I, c_z] THIS GPU ONLY   │
        │  streaming_mode    : True                             │
        │  z_chunk_range     : (start_i, end_i)                 │
        └───────────────────────────────────────────────────────┘
```

### Key Difference: Z_II Shape

```
  Standard Mode:                            Streaming Mode:
  ──────────────                            ───────────────

  Z_II: [I, I, c_z]                        Z_chunk: [I_par, I, c_z]

  ┌──────────────────┐                      GPU 0: ┌──────────────────┐
  │                  │                             │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ I_par rows
  │                  │                             └──────────────────┘
  │   Full I×I       │                      GPU 1: ┌──────────────────┐
  │   on 1 GPU       │                             │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ I_par rows
  │                  │                             └──────────────────┘
  │                  │                      GPU 2: ┌──────────────────┐
  │                  │                             │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ I_par rows
  └──────────────────┘                             └──────────────────┘
  Memory: I × I × c_z                      Memory per GPU: (I/N) × I × c_z
```

---

## 4. DiffusionTokenEncoder — Streaming Mode

The DiffusionTokenEncoder refines Z and S through Pairformer blocks. In streaming mode, each GPU holds only its Z chunk and synchronizes S_I via `all_gather` after each block.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│           DiffusionTokenEncoder — Streaming Mode (_forward_streaming)            │
│                     (encoders.py:1356)                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

    INPUTS (per GPU):
    ─────────────────
    Z_init_chunk  : [I_par, I, c_z]   This GPU's Z rows from TokenInitializer
    S_init_I      : [I, c_s]          Full single features (replicated)
    R_L           : [B, L, 3]         Scaled positions (replicated)
    z_chunk_range : (gpu_start, gpu_end)


                    Z_init_chunk [I_par, I, c_z]         S_init_I [I, c_s]
                              │                                  │
                              ▼                                  ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │                                                                               │
  │  Step 1: S_I transitions (operates on full I, no I×I needed)                 │
  │    S_I += transition_1[0](S_I)                                               │
  │    S_I += transition_1[1](S_I)                                               │
  │                                                                               │
  └──────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │  Step 2-4: Build Z_chunk [B, I_par, I, cat_c_z] via pre-allocation           │
  │                                                                               │
  │  ╔════════════════════════════════════════════════════════════════════╗        │
  │  ║ MEMORY OPTIMIZATION: Pre-allocate Z_chunk and copy slices         ║        │
  │  ║ in-place instead of torch.cat() to avoid 20+ GB memory spikes    ║        │
  │  ╚════════════════════════════════════════════════════════════════════╝        │
  │                                                                               │
  │  Z_chunk = empty(B, I_par, I, cat_c_z)    # Pre-allocate full size           │
  │  Z_chunk[:,:,:, 0:c_z]         = Z_init_chunk   # Copy Z_init                │
  │  Z_chunk[:,:,:, c_z:c_z+d]     = D_chunk        # Distogram (chunked)        │
  │  Z_chunk[:,:,:, c_z+d:c_z+d+s] = D_self_chunk   # Self-conditioning          │
  │                                                                               │
  │  Distogram: dist_embedder.forward_chunk(R_ca_query, R_ca)                    │
  │             [B, I_par, I, c_z] — only this GPU's query rows                  │
  │                                                                               │
  │  Self-conditioning: D_II_self[:, gpu_start:gpu_end, :]                       │
  │             [B, I_par, I, n_bins] — sliced from recycled output              │
  │                                                                               │
  └──────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
                    Z_chunk [B, I_par, I, cat_c_z]
                              │
  ┌───────────────────────────┴───────────────────────────────────────────────────┐
  │  Step 5: process_z (chunked for memory)                                       │
  │                                                                               │
  │  Z_chunk = _process_z_chunked(Z_chunk, process_z)  → [B, I_par, I, c_z]     │
  │  Z_chunk += _z_transition_chunked(Z_chunk, transition_2[0])                  │
  │  Z_chunk += _z_transition_chunked(Z_chunk, transition_2[1])                  │
  │                                                                               │
  │  ╔════════════════════════════════════════════════════════════════════╗        │
  │  ║ _z_transition_chunked: Applies SwiGLU MLP to Z_chunk.             ║        │
  │  ║ If extra_chunking=True: loops over key dim in 512-wide chunks    ║        │
  │  ║   → O(I/512) time but O(I²/N²) memory                           ║        │
  │  ║ If extra_chunking=False (default): single vectorized op          ║        │
  │  ║   → O(1) time but O(I²/N) memory                                ║        │
  │  ╚════════════════════════════════════════════════════════════════════╝        │
  │                                                                               │
  └──────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
  ┌───────────────────────────────────────────────────────────────────────────────┐
  │  Step 6: Pairformer Stack (18 blocks) — THE MAIN BOTTLENECK                  │
  │                                                                               │
  │  for block in pairformer_stack:   ◀── 18 iterations                          │
  │    │                                                                          │
  │    │  ┌─────────────────────────────────────────────────────────────┐         │
  │    │  │ Z_chunk = _z_transition_chunked(Z_chunk, block.z_transition)│         │
  │    │  │          [B, I_par, I, c_z] — SwiGLU applied per GPU       │         │
  │    │  └─────────────────────────────────────────────────────────────┘         │
  │    │                                                                          │
  │    │  ┌─────────────────────────────────────────────────────────────┐         │
  │    │  │ S_I_chunk += attention_pair_bias.forward_chunked(           │         │
  │    │  │     A_I_query = S_I_chunk,     [I_par, c_s]  (Q from chunk)│         │
  │    │  │     A_I_key   = S_I,           [I, c_s]      (K from all) │         │
  │    │  │     Z_chunk   = Z_chunk[0],    [I_par, I, c_z] (bias)     │         │
  │    │  │ )                                                          │         │
  │    │  │ S_I_chunk += s_transition(S_I_chunk)                       │         │
  │    │  └─────────────────────────────────────────────────────────────┘         │
  │    │                                                                          │
  │    │  ┌─────────────────────────────────────────────────────────────┐         │
  │    │  │     ╔═══════════════════════════════════════════╗           │         │
  │    │  │     ║  ALL_GATHER: S_I_chunk → full S_I         ║           │         │
  │    │  │     ║  (NCCL synchronization across N GPUs)     ║           │         │
  │    │  │     ║  18 blocks × all_gather per block         ║           │         │
  │    │  │     ║  = 18 sync points per diffusion step      ║           │         │
  │    │  │     ╚═══════════════════════════════════════════╝           │         │
  │    │  │                                                            │         │
  │    │  │  GPU 0: S_I_chunk [I_par, c_s] ──┐                        │         │
  │    │  │  GPU 1: S_I_chunk [I_par, c_s] ──┤── all_gather ──▶ S_I [I, c_s]   │
  │    │  │  ...                             │                        │         │
  │    │  │  GPU N: S_I_chunk [I_par, c_s] ──┘                        │         │
  │    │  └─────────────────────────────────────────────────────────────┘         │
  │    │                                                                          │
  │    └── next block (uses gathered S_I as keys)                                │
  │                                                                               │
  └──────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │  OUTPUT:                                │
                │    S_I     : [I, c_s]     (all_gathered)│
                │    Z_chunk : [I_par, I, c_z] (per GPU) │
                └────────────────────────────────────────┘
```

### PairformerBlock — Standard vs Parallel

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PairformerBlock: Standard vs Parallel                          │
└─────────────────────────────────────────────────────────────────────────────────┘

  STANDARD (1 GPU):                         PARALLEL (N GPUs, per GPU):
  ──────────────────                        ───────────────────────────

  S_I [I, c_s]   Z_II [I, I, c_z]          S_I [I, c_s]   Z_chunk [I_par, I, c_z]
       │              │                          │               │
       │              ▼                          │               ▼
       │    Z += z_transition(Z)                 │     Z_chunk += z_transition(Z_chunk)
       │    [I, I, c_z]                          │     [I_par, I, c_z]
       │              │                          │               │
       ▼              │                          │               │
  self-attention      │                     ┌────┴────┐          │
  Q,K,V from S_I ◀───┘                     │         │          │
  bias from Z_II                            ▼         ▼          │
       │                               cross-attention           │
       ▼                               Q=S_I_chunk ◀────────────┘
  S += attn(S,S,Z)                     K=S_I (full)
  S += s_trans(S)                      bias=Z_chunk
       │                                    │
       ▼                                    ▼
  S_I [I, c_s]                         S_I_chunk [I_par, c_s]
  (updated, done)                            │
                                             ▼
                                    ╔═══════════════════╗
                                    ║   ALL_GATHER      ║
                                    ║   S_I_chunk →     ║
                                    ║   full S_I [I,c_s]║
                                    ╚═══════════════════╝
                                             │
                                             ▼
                                        S_I [I, c_s]
                                        (synced, next block)
```

---

## 5. Diffusion Module — Parallel Forward

The `RFD3DiffusionModule.forward()` orchestrates encoder, token encoder, transformer, and decoder. In parallel mode, each stage splits work across GPUs and synchronizes.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                RFD3DiffusionModule — Parallel Forward Path                       │
│                     (RFD3_diffusion_module.py:742)                              │
└─────────────────────────────────────────────────────────────────────────────────┘

    INPUTS:
    ───────
    X_noisy_L    : [B, L, 3]          Noisy atom coordinates (replicated)
    t            : [B]                Noise levels (replicated)
    S_I          : [I, c_s]           Single features (replicated)
    Z_II         : [I_par, I, c_z]    Z chunk (THIS GPU ONLY)
    Q_L_init     : [L, c_atom]        Atom features (replicated)
    C_L          : [L, c_atom]        Conditioned features (replicated)
    streaming_mode = True


                    X_noisy_L [B, L, 3]
                            │
                            ▼
                ┌───────────────────────┐
                │   scale_positions_in  │     Same as standard
                │   process_r, time     │
                └───────────┬───────────┘
                            │
                            ▼
              Q_L [B, L, c_atom], C_L [B, L, c_atom], A_I [B, I, c_token]
                            │
    ════════════════════════╪════════════════════════════════════
    ENCODER (LocalAtomTransformer) — PARALLEL ATOM PROCESSING
    ════════════════════════╪════════════════════════════════════
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────┐
    │  encoder.forward_parallel()                               │
    │                                                           │
    │  Each GPU processes L_par = L/N atom queries:             │
    │                                                           │
    │  GPU 0: Q_L[:, 0:L_par, :]     ──┐                       │
    │  GPU 1: Q_L[:, L_par:2L_par, :] ──┤── local attention    │
    │  ...                              │   with P_LL_chunk     │
    │  GPU N: Q_L[:, (N-1)L_par:L, :] ──┘   [L_par, L, c]     │
    │                                                           │
    │  ╔═════════════════════════════════════════╗              │
    │  ║  ALL_GATHER: Q_L chunks → full Q_L      ║              │
    │  ╚═════════════════════════════════════════╝              │
    └───────────────────────┬───────────────────────────────────┘
                            │
                            ▼
                Q_L [B, L, c_atom] (all_gathered, replicated)
                            │
    ┌───────────────────────┴───────────────────────────────────┐
    │                    DOWNCAST_Q                              │
    │               (Atom → Token pooling)                      │
    │               Same as standard mode                       │
    └───────────────────────┬───────────────────────────────────┘
                            │
                            ▼
                   A_I [B, I, c_token]
                            │
    ════════════════════════╪════════════════════════════════════
    DIFFUSION TOKEN ENCODER — PARALLEL Z PROCESSING
    ════════════════════════╪════════════════════════════════════
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────┐
    │  diffusion_token_encoder._forward_streaming()             │
    │                                                           │
    │  Z_chunk stays [I_par, I, c_z] throughout                │
    │  S_I all_gathered after each Pairformer block             │
    │  (See Section 4 for details)                              │
    │                                                           │
    │  18 Pairformer blocks × all_gather = main bottleneck     │
    └───────────────────────┬───────────────────────────────────┘
                            │
                            ▼
        S_I [I, c_s] (synced), Z_chunk [I_par, I, c_z] (per GPU)
                            │
    ════════════════════════╪════════════════════════════════════
    DIFFUSION TRANSFORMER — PARALLEL TOKEN PROCESSING
    ════════════════════════╪════════════════════════════════════
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────┐
    │  _diffusion_transformer_parallel()                        │
    │                                                           │
    │  Cross-attention for each GPU's token chunk:              │
    │                                                           │
    │  A_I_chunk = A_I[:, start:end, :]   [B, I_par, c_token]  │
    │                                                           │
    │  diffusion_transformer.forward_cross_attn(                │
    │      A_I_chunk = A_I_chunk,          [B, I_par, c_token]  │
    │      A_I_full  = A_I,                [B, I, c_token]      │
    │      S_I_chunk = S_I[start:end],     [I_par, c_s]         │
    │      S_I_full  = S_I,                [I, c_s]             │
    │      Z_chunk   = Z_II_chunk,         [I_par, I, c_z]      │
    │  )                                                        │
    │                                                           │
    │  ╔═════════════════════════════════════════╗              │
    │  ║  ALL_GATHER: A_I chunks → full A_I      ║              │
    │  ╚═════════════════════════════════════════╝              │
    └───────────────────────┬───────────────────────────────────┘
                            │
                            ▼
               A_I [B, I, c_token] (all_gathered)
                            │
    ════════════════════════╪════════════════════════════════════
    DECODER — PARALLEL ATOM READOUT
    ════════════════════════╪════════════════════════════════════
                            │
                            ▼
    ┌───────────────────────────────────────────────────────────┐
    │  _decoder_parallel() or _decoder_parallel_sparse()        │
    │                                                           │
    │  Each GPU decodes L_par atom queries:                     │
    │                                                           │
    │  P_LL_chunk [L_par, L, c_atompair]                       │
    │    computed on-the-fly (never full L×L)                   │
    │                                                           │
    │  OR (with low_memory_mode):                               │
    │  Sparse P_LL [L_par, k, c_atompair] via embedder         │
    │                                                           │
    │  ╔═════════════════════════════════════════╗              │
    │  ║  ALL_GATHER: Q_L chunks → full Q_L      ║              │
    │  ╚═════════════════════════════════════════╝              │
    └───────────────────────┬───────────────────────────────────┘
                            │
                            ▼
              Q_L [B, L, c_atom] (decoded, all_gathered)
                            │
    ┌───────────────────────┴───────────────────────────────────┐
    │                  OUTPUT HEADS (same as standard)           │
    │                                                           │
    │  R_update_L = to_r_update(Q_L)       [B, L, 3]           │
    │  X_out_L = scale_positions_out(...)   [B, L, 3]           │
    │  seq_logits = sequence_head(A_I)      [B, I, vocab]       │
    │                                                           │
    │  D_II_self: Computed as CHUNK [B, I_par, I, n_bins]       │
    │    (NOT gathered to full I×I — memory optimization)        │
    └───────────────────────┬───────────────────────────────────┘
                            │
                            ▼
                  X_denoised_L [B, L, 3]
```

---

## 6. All-Gather Synchronization

The `all_gather` operation is the primary communication primitive. Each GPU sends its chunk and receives all others'.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       All-Gather Pattern                                        │
│                    (_all_gather_concat in encoders.py:256)                      │
└─────────────────────────────────────────────────────────────────────────────────┘

  BEFORE all_gather:                        AFTER all_gather:

  GPU 0: [chunk_0]                          GPU 0: [chunk_0 | chunk_1 | chunk_2]
  GPU 1: [chunk_1]          ──────▶         GPU 1: [chunk_0 | chunk_1 | chunk_2]
  GPU 2: [chunk_2]                          GPU 2: [chunk_0 | chunk_1 | chunk_2]


  ═══════════════════════════════════════════════════════════════════

  UNEVEN CHUNK HANDLING (floor-division with remainder):
  ─────────────────────────────────────────────────────

  Example: I=13800 tokens, N=7 GPUs

  chunk_size = 13800 // 7 = 1971
  remainder  = 13800 % 7  = 3

  GPU 0: 1972 tokens (chunk_size + 1)  ◀─ gets extra
  GPU 1: 1972 tokens (chunk_size + 1)  ◀─ gets extra
  GPU 2: 1972 tokens (chunk_size + 1)  ◀─ gets extra
  GPU 3: 1971 tokens
  GPU 4: 1971 tokens
  GPU 5: 1971 tokens
  GPU 6: 1971 tokens
  ────────────────────
  Total: 13800 tokens  ✓

  all_gather pads smaller chunks to max_size (1972),
  gathers, then slices to total_size (13800).


  ═══════════════════════════════════════════════════════════════════

  WHERE all_gather OCCURS (per denoising step × recycle):

  ┌────────────────────────────────┬──────────────┬──────────────────┐
  │ Location                       │ # of calls   │ What is gathered │
  ├────────────────────────────────┼──────────────┼──────────────────┤
  │ TokenInitializer               │ N_init_blocks│ S_I [I, c_s]     │
  │   transformer_stack            │ (e.g., 4)    │                  │
  ├────────────────────────────────┼──────────────┼──────────────────┤
  │ Encoder                        │ 1            │ Q_L [B, L, c_atom│
  │   forward_parallel             │              │                  │
  ├────────────────────────────────┼──────────────┼──────────────────┤
  │ DiffusionTokenEncoder          │ 18           │ S_I [I, c_s]     │
  │   pairformer_stack (per block) │              │                  │
  ├────────────────────────────────┼──────────────┼──────────────────┤
  │ DiffusionTransformer           │ 1            │ A_I [B, I, c_tok]│
  │   _diffusion_transformer_par   │              │                  │
  ├────────────────────────────────┼──────────────┼──────────────────┤
  │ Decoder                        │ N_dec_blocks │ Q_L [B, L, c_atom│
  │   forward_parallel             │ (e.g., 3)    │                  │
  ├────────────────────────────────┼──────────────┼──────────────────┤
  │ TOTAL per step                 │ ~27          │                  │
  │ TOTAL per generation           │ ~27 × 200    │ = ~5,400         │
  │   (with n_recycle=3)           │ × 3 = ~16,200│                  │
  └────────────────────────────────┴──────────────┴──────────────────┘
```

---

## 7. Complete Single Denoising Step (Parallel)

This shows the full parallel data flow for one denoising step. Compare with Section 6 of Flow_diagram.md for the standard mode.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              Complete Single Denoising Step — Parallel Mode                      │
│                    (inference_sampler.py:421-627)                               │
└─────────────────────────────────────────────────────────────────────────────────┘


INPUTS AT STEP n (replicated on all GPUs):
──────────────────────────────────────────
    X_L              : [D, L, 3]              Current coordinates
    c_t_minus_1      : scalar                 Previous noise level
    c_t              : scalar                 Target noise level
    f                : dict                   Features (constant)
    initializer_outputs : dict                From TokenInitializer (constant)
      ├── S_I        : [I, c_s]              Single features (FULL, replicated)
      ├── Z_II       : [I_par, I, c_z]       Z chunk (PER GPU, different on each)
      ├── z_chunk_range : (start, end)        This GPU's query range
      └── streaming_mode : True


    ┌────────────────────────────────────────────┐
    │    STEPS 1-2: Noise & stochasticity        │
    │    (IDENTICAL to standard mode)            │
    │                                            │
    │    t_hat, epsilon, X_noisy_L computation   │
    │    Same on all GPUs (same random seed)      │
    └──────────────────┬─────────────────────────┘
                       │
                       ▼
            X_noisy_L [D, L, 3]
                       │
    ┌──────────────────┼──────────────────────────┐
    │    STEP 3: Parallel diffusion_module        │
    └──────────────────┼──────────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │  RFD3DiffusionModule.forward()                                         │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ RECYCLE LOOP (n_recycle=3 iterations):                          │   │
    │  │                                                                 │   │
    │  │  for i in range(3):                                             │   │
    │  │    │                                                            │   │
    │  │    │  ┌───── ENCODER (parallel) ─────────────────────────┐     │   │
    │  │    │  │ Split L atoms across N GPUs                       │     │   │
    │  │    │  │ Each: Q_L[:, L_par_start:L_par_end, :]            │     │   │
    │  │    │  │ Local attention with P_LL_chunk or sparse P_LL    │     │   │
    │  │    │  │ all_gather → Q_L [B, L, c_atom]                  │     │   │
    │  │    │  └──────────────────────────────────────────────────┘     │   │
    │  │    │                                                            │   │
    │  │    │  ┌───── DOWNCAST_Q ────────────────────────────────┐      │   │
    │  │    │  │ Standard pooling (no I×I needed)                 │      │   │
    │  │    │  │ A_I = pool(Q_L, tok_idx)  [B, I, c_token]       │      │   │
    │  │    │  └─────────────────────────────────────────────────┘      │   │
    │  │    │                                                            │   │
    │  │    │  ┌───── TOKEN ENCODER (parallel, 18 blocks) ──────┐      │   │
    │  │    │  │ Z_chunk processed locally [B, I_par, I, c_z]    │      │   │
    │  │    │  │ S_I_chunk attention → all_gather each block     │      │   │
    │  │    │  │ = 18 × all_gather (MAIN BOTTLENECK)             │      │   │
    │  │    │  └─────────────────────────────────────────────────┘      │   │
    │  │    │                                                            │   │
    │  │    │  ┌───── TRANSFORMER (parallel) ────────────────────┐      │   │
    │  │    │  │ Cross-attention: A_I_chunk [B, I_par] → all [I] │      │   │
    │  │    │  │ all_gather → A_I [B, I, c_token]                │      │   │
    │  │    │  └─────────────────────────────────────────────────┘      │   │
    │  │    │                                                            │   │
    │  │    │  ┌───── DECODER (parallel) ────────────────────────┐      │   │
    │  │    │  │ Split L atoms across N GPUs                      │      │   │
    │  │    │  │ P_LL_chunk [L_par, L] or sparse [L_par, k]      │      │   │
    │  │    │  │ all_gather → Q_L [B, L, c_atom]                 │      │   │
    │  │    │  └─────────────────────────────────────────────────┘      │   │
    │  │    │                                                            │   │
    │  │    │  ┌───── OUTPUT HEADS ──────────────────────────────┐      │   │
    │  │    │  │ R_update = to_r_update(Q_L)    [B, L, 3]        │      │   │
    │  │    │  │ X_out = scale_positions_out()   [B, L, 3]        │      │   │
    │  │    │  │                                                  │      │   │
    │  │    │  │ D_II_self = distogram_chunked() [B, I_par, I, n]│      │   │
    │  │    │  │   (kept as chunk — NOT gathered!)                │      │   │
    │  │    │  └─────────────────────────────────────────────────┘      │   │
    │  │    │                                                            │   │
    │  │    └── next recycle (uses D_II_self for conditioning)          │   │
    │  │                                                                 │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └──────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                           X_denoised_L [D, L, 3]
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────────┐
    │    STEP 4: ODE update (IDENTICAL to standard mode)                      │
    │                                                                         │
    │    delta_L = (X_noisy_L - X_denoised_L) / t_hat                       │
    │    X_L = X_noisy_L + step_scale * d_t * delta_L                        │
    │                                                                         │
    │    (Same result on all GPUs — outputs are synchronized)                │
    └──────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                              X_L [D, L, 3]
                                       │
                                       ▼
                             Continue to step n+1
```

---

## 8. Tensor Shape Reference (Parallel Mode)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Tensor Shape Reference — Parallel Mode                        │
│                    (N GPUs, I_par ≈ I/N, L_par ≈ L/N)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

    Symbol     Meaning
    ──────     ──────────────────────────────────────────
    N          Number of GPUs (attention_parallel_factor)
    I_par      I / N (this GPU's token query count)
    L_par      L / N (this GPU's atom query count)


    TENSOR                  STANDARD SHAPE        PARALLEL SHAPE (per GPU)
    ───────────────────────────────────────────────────────────────────────

    Coordinates (replicated on all GPUs):
    X_L                     [D, L, 3]             [D, L, 3]       (same)
    X_noisy_L               [D, L, 3]             [D, L, 3]       (same)
    X_denoised_L            [D, L, 3]             [D, L, 3]       (same)

    Single Features:
    S_I                     [I, c_s]              [I, c_s]         (FULL, replicated)
    S_I_chunk               N/A                   [I_par, c_s]     (this GPU's chunk)

    Pair Features:
    Z_II                    [I, I, c_z]           [I_par, I, c_z]  (this GPU's rows)
    Z_chunk (batched)       [B, I, I, c_z]        [B, I_par, I, c_z]

    Atom Pair Features:
    P_LL                    [L, L, c_atompair]    [L_par, L, c_atompair] (chunk)
    P_LL (sparse)           [L, k, c_atompair]    [L_par, k, c_atompair]

    Atom Features (replicated after all_gather):
    Q_L                     [D, L, c_atom]        [D, L, c_atom]   (same after gather)
    C_L                     [D, L, c_atom]        [D, L, c_atom]   (same)
    Q_L_init                [L, c_atom]           [L, c_atom]      (same)

    Token Activations:
    A_I                     [D, I, c_token]       [D, I, c_token]  (FULL after gather)
    A_I_chunk               N/A                   [D, I_par, c_token]

    Self-Conditioning:
    D_II_self               [B, I, I, n_bins]     [B, I_par, I, n_bins]  (CHUNK!)

    Mappings:
    tok_idx                 [L]                   [L]              (same)
    z_chunk_range           N/A                   (start_i, end_i)


    ═══════════════════════════════════════════════════════════════════

    MEMORY COMPARISON (I=10500, L≈63000, N=6 GPUs, c_z=128):

    TENSOR              STANDARD (1 GPU)    PARALLEL (per GPU)    SAVINGS
    ─────────────────────────────────────────────────────────────────────
    Z_II                25.2 GB             4.2 GB                6×
    P_LL                ~48.0 GB            ~8.0 GB               6×
    SwiGLU peak         ~65 GB              ~11 GB                6×
    D_II_self           ~13.6 GB            ~2.3 GB               6×

    Total peak          ~150+ GB            ~25 GB per GPU        6×


    ═══════════════════════════════════════════════════════════════════

    COMMUNICATION COST:

    all_gather operations per generation (200 steps × 3 recycles):
    - TokenInitializer transformer_stack:  N_init × 600 = ~2,400
    - Encoder forward_parallel:            1 × 600       = 600
    - DiffusionTokenEncoder pairformer:    18 × 600      = 10,800
    - DiffusionTransformer:                1 × 600       = 600
    - Decoder:                             N_dec × 600   = ~1,800
                                                    TOTAL ≈ 16,200

    Per all_gather data volume (I=10500, c_s=256, bfloat16):
    S_I gather: 10500 × 256 × 2 bytes ≈ 5.1 MB
    A_I gather: D × 10500 × 256 × 2 bytes ≈ D × 5.1 MB
```

---

## Appendix A: Configuration & Environment Variables

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  Parallel Mode Configuration                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

  Environment Variables:
  ──────────────────────

  RFD3_ATTENTION_PARALLEL=N     Enable parallel mode with N GPUs
                                (any non-zero value enables; GPU count from dist)
                                Set by design_parallel.py launch script

  RFD3_EXTRA_CHUNKING=1         Enable key-dimension chunking in z_transition
                                Default: 0 (vectorized, faster, more memory)
                                Set to 1 for O(I²/N²) memory at O(N) time cost

  RFD3_LOW_MEMORY_MODE=1        Enable sparse P_LL (structure-local attention)
                                Can be combined with ATTENTION_PARALLEL


  Mode Combinations:
  ──────────────────

  ┌──────────────────────┬────────────────┬──────────────────────────────────┐
  │ ATTENTION_PARALLEL   │ LOW_MEMORY_MODE│ Behavior                          │
  ├──────────────────────┼────────────────┼──────────────────────────────────┤
  │ 0 (or unset)         │ 0 (or unset)  │ Standard: full Z[I,I], P[L,L]    │
  │ 0                    │ 1             │ Low-mem: full Z[I,I], sparse P    │
  │ N (>0)               │ 0             │ Parallel: Z[I_par,I], P[L_par,L] │
  │ N (>0)               │ 1             │ Both: Z[I_par,I], sparse P split  │
  └──────────────────────┴────────────────┴──────────────────────────────────┘


  Design Config (configs/design_parallel.yaml):
  ─────────────────────────────────────────────

  attention_parallel_factor: 6    # Number of GPUs
  low_memory_mode: true           # Sparse P_LL
  extra_chunking: false           # Key-dimension chunking
```

---

## Appendix B: Memory vs Time Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Memory vs Time Trade-offs                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

  extra_chunking=False (DEFAULT):
  ───────────────────────────────
  z_transition operates on full Z_chunk [I_par, I, c_z] at once.

  Time:   O(1) — single vectorized operation
  Memory: O(I²/N) per GPU — key dimension I is NOT split

  SwiGLU peak memory: 13 × (I/N) × I × c_z × 2 bytes
  For I=10500, N=6: 13 × 1750 × 10500 × 128 × 2 ≈ 61 GB  (may OOM!)


  extra_chunking=True:
  ────────────────────
  z_transition loops over key dimension in chunks of 512.

  Time:   O(I/512) — loops over key chunks
  Memory: O(I × 512 / N) per GPU — much smaller peak

  SwiGLU peak: 13 × (I/N) × 512 × c_z × 2 bytes
  For I=10500, N=6: 13 × 1750 × 512 × 128 × 2 ≈ 3 GB  (safe)


  ═══════════════════════════════════════════════════════════════════

  Maximum ASU Length (95 GB GPUs, ~70 GB effective):

  ┌────────┬───────────────┬───────────────────────────────────────┐
  │ N GPUs │ Max I (tokens)│ Max ASU (I symmetry, 60 copies)      │
  ├────────┼───────────────┼───────────────────────────────────────┤
  │ 1      │ 4,583         │ 76                                   │
  │ 2      │ 6,481         │ 108                                  │
  │ 4      │ 9,166         │ 152                                  │
  │ 6      │ 11,227        │ 187                                  │
  │ 8      │ 12,964        │ 216                                  │
  └────────┴───────────────┴───────────────────────────────────────┘

  Scaling: Max I ∝ sqrt(N) (because memory per GPU = O(I²/N))
```

---

*End of Flow_diagram_parallel.md*
