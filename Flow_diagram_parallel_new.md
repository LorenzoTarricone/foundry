# Flow_diagram_parallel_new.md — Post-Refactoring Parallel Mode Call Stack

**Created:** 2026-02-16
**Purpose:** Documents the parallel (multi-GPU) inference call stack after the refactoring that separated parallel code into the `model/parallel/` package. This replaces the pre-refactoring Flow_diagram_parallel.md.

---

## Table of Contents

1. [How Parallel Mode is Activated](#1-how-parallel-mode-is-activated)
2. [File Layout After Refactoring](#2-file-layout-after-refactoring)
3. [Full Call Stack (Flat List)](#3-full-call-stack-flat-list)
4. [ASCII Call-Stack Tree](#4-ascii-call-stack-tree)
5. [Detailed Data Flow per Component](#5-detailed-data-flow-per-component)
6. [All-Gather Synchronization Points](#6-all-gather-synchronization-points)
7. [Free Function Dispatch Table](#7-free-function-dispatch-table)
8. [Tensor Shape Reference](#8-tensor-shape-reference)
9. [Class Hierarchy](#9-class-hierarchy)

---

## 1. How Parallel Mode is Activated

Parallel mode follows the same pattern as low-memory mode: **an environment variable is checked at runtime** to decide which code path to take.

```
┌────────────────────────────────┬──────────────────────────────────────────────┐
│ Environment Variable           │ Effect                                       │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ RFD3_ATTENTION_PARALLEL=0      │ Standard mode (single GPU)                   │
│ RFD3_ATTENTION_PARALLEL=N      │ Parallel mode (N GPUs, any non-zero value)   │
│ RFD3_LOW_MEMORY_MODE=1         │ Sparse P_LL (orthogonal, combinable)         │
│ RFD3_EXTRA_CHUNKING=1          │ Key-dimension chunking (slower, less memory) │
└────────────────────────────────┴──────────────────────────────────────────────┘
```

### Activation Flow

```
RFD3.__init__()  [RFD3.py:28]
│
├── Reads env vars:
│     attn_parallel = os.environ.get("RFD3_ATTENTION_PARALLEL", "0") not in ("0","","false","False")
│     low_mem       = os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1"
│
├── if attn_parallel:
│   │  # Import parallel classes (lazy, only when needed)
│   │  from rfd3.model.parallel.layers.encoders import ParallelTokenInitializer
│   │  from rfd3.model.parallel.diffusion_module import ParallelDiffusionModule
│   │
│   ├── self.token_initializer = ParallelTokenInitializer(...)
│   └── self.diffusion_module  = ParallelDiffusionModule(...)
│         └── __init__: swaps self.diffusion_token_encoder.__class__ = ParallelDiffusionTokenEncoder
│
└── else:
    ├── self.token_initializer = TokenInitializer(...)
    └── self.diffusion_module  = hydra.utils.instantiate(diffusion_module, ...)
```

### Where `is_parallel_mode()` is Checked at Runtime

| Location | File | Line | Purpose |
|----------|------|------|---------|
| `is_parallel_mode()` | `parallel/utils.py:37` | Canonical check | Used by parallel subclasses |
| `_is_parallel_mode()` | `RFD3_diffusion_module.py:42` | Base class check | Controls `use_full_attention` flag |
| `_is_parallel_mode()` | `inference_sampler.py:31` | Sampler check | Controls noise broadcast, memory cleanup |
| Factory check | `RFD3.py:63-64` | Init time | Decides which classes to instantiate |

### Comparison with Low-Memory Mode

Both modes use the same pattern:

```
LOW_MEMORY_MODE:
  env var → RFD3.__init__ sets use_chunked_pll=True
  → TokenInitializer creates chunked_pairwise_embedder
  → Passed through initializer_outputs to diffusion_module
  → process_() checks if chunked_pairwise_embedder is not None

ATTENTION_PARALLEL:
  env var → RFD3.__init__ instantiates Parallel* classes
  → ParallelTokenInitializer.forward() returns parallel_mode=True
  → Passed through initializer_outputs to diffusion_module
  → ParallelDiffusionModule.process_() checks parallel_mode flag
```

---

## 2. File Layout After Refactoring

```
models/rfd3/src/rfd3/model/
├── RFD3.py                              # Factory: standard vs parallel classes
├── RFD3_diffusion_module.py             # Base: RFD3DiffusionModule (standard only)
├── inference_sampler.py                 # Diffusion loop (shared, mode-aware)
│
├── layers/                              # Base classes (standard mode only)
│   ├── encoders.py                      # TokenInitializer, DiffusionTokenEncoder
│   ├── blocks.py                        # CompactDecoder, LocalTokenTransformer, etc.
│   ├── attention.py                     # LocalAttentionPairBias
│   ├── pairformer_layers.py             # AttentionPairBiasPairformerDeepspeed
│   ├── chunked_pairwise.py             # ChunkedPairwiseEmbedder
│   └── layer_utils.py                   # Transition (SwiGLU), RMSNorm, linearNoBias
│
└── parallel/                            # Parallel-specific code
    ├── __init__.py                      # Exports: ParallelTokenInitializer, ParallelDiffusionTokenEncoder, ParallelDiffusionModule
    ├── utils.py                         # Consolidated utilities: is_parallel_mode, all_gather_concat, z_transition_chunked, etc.
    ├── diffusion_module.py              # ParallelDiffusionModule(RFD3DiffusionModule)
    └── layers/
        ├── __init__.py
        ├── encoders.py                  # ParallelTokenInitializer(TokenInitializer), ParallelDiffusionTokenEncoder(DiffusionTokenEncoder)
        ├── blocks.py                    # Free functions: local_token_transformer_cross_attn, compact_decoder_parallel, etc.
        ├── attention.py                 # Free functions: local_attention_cross_attn, local_attention_sparse_cross_attn, sparse_cross_attention
        ├── pairformer_layers.py         # Free function: attention_pair_bias_forward_chunked
        └── chunked_pairwise.py          # Free function: chunked_pairwise_forward_parallel
```

---

## 3. Full Call Stack (Flat List)

| # | Function | File:Line | Description | Key Tensors |
|---|----------|-----------|-------------|-------------|
| 0a | `main()` | `scripts/design_parallel.py:802` | CLI entry: argparse, load_config, detect GPUs | — |
| 0b | `launch_with_torchrun(args, params, n_gpus)` | `scripts/design_parallel.py:762` | Spawns N worker processes via `torchrun` | — |
| 0c | `run_worker(out_dir, length, num_designs, symmetry, seed, ...)` | `scripts/design_parallel.py:441` | Per-GPU worker: sets env vars, creates engine, runs inference | — |
| 0d | `RFD3InferenceEngine(**rfd3_config)` | `engine.py:242` | Engine init → `_construct_trainer()` → `RFD3(...)` | — |
| 1 | `RFD3.__init__()` | `RFD3.py:28` | Factory: creates `ParallelTokenInitializer` + `ParallelDiffusionModule` | — |
| 2 | `ParallelDiffusionModule.__init__()` | `parallel/diffusion_module.py:46` | Calls `super().__init__()`, swaps encoder class to `ParallelDiffusionTokenEncoder` | — |
| 3 | `RFD3.forward()` | `RFD3.py:136` | Entry point for inference | `input["f"]` |
| 4 | `ParallelTokenInitializer.forward()` | `parallel/layers/encoders.py:227` | Computes Z_chunk [I_par, I, c_z], returns `parallel_mode=True` | `S_I`, `Z_II (chunk)`, `Q_L_init`, `C_L` |
| 5 | `ParallelTokenInitializer._process_s_through_transformer_stack()` | `parallel/layers/encoders.py:34` | Chunked Pairformer on S_I with all_gather per block | `S_I`, `Z_chunk` |
| 6 | `attention_pair_bias_forward_chunked()` | `parallel/layers/pairformer_layers.py:10` | Free function: chunked cross-attention Q[I_par] × K[I] | `S_I_chunk`, `S_I`, `Z_chunk` |
| 7 | `all_gather_concat()` | `parallel/utils.py` | Gathers S_I chunks from all GPUs → full S_I | `S_I_chunk` → `S_I` |
| 8 | `ConditionalDiffusionSampler.sample_diffusion_like_af3()` | `inference_sampler.py:306` | Dispatches to `SampleDiffusionWithMotif` | — |
| 9 | `SampleDiffusionWithMotif.sample_diffusion_like_af3()` | `inference_sampler.py:306` | Main diffusion loop (200 steps), passes `parallel_mode` to `diffusion_module()` | `X_L`, `X_noisy_L` |
| 10 | `RFD3DiffusionModule.forward()` | `RFD3_diffusion_module.py:198` | Inherited by ParallelDiffusionModule — shared code: scale_positions, encoder, downcast | `X_noisy_L`, `Z_II (chunk)` |
| 11 | `LocalAtomTransformer.forward()` | `blocks.py` | Atom-level encoder — standard call, no parallel override | `Q_L`, `C_L`, `P_LL` |
| 12 | `RFD3DiffusionModule.forward_with_recycle()` | `RFD3_diffusion_module.py:365` | Inherited — loops `process_()` n_recycle times, passes `parallel_mode` | — |
| 13 | `ParallelDiffusionModule.process_()` | `parallel/diffusion_module.py:460` | **Override** — parallel token encoder, transformer, decoder | `S_I`, `Z_II (chunk)`, `A_I` |
| 14 | `ParallelDiffusionTokenEncoder.forward()` | `parallel/layers/encoders.py:376` | Process Z_chunk with distogram, Pairformer, all_gather S_I | `Z_chunk [B,I_par,I,c_z]` |
| 15 | `process_z_chunked()` | `parallel/utils.py` | Key-chunked linear projection of concatenated Z features | `Z_chunk` |
| 16 | `z_transition_chunked()` | `parallel/utils.py` | Key-chunked SwiGLU transition (avoids 4× memory spike) | `Z_chunk` |
| 17 | `attention_pair_bias_forward_chunked()` | `parallel/layers/pairformer_layers.py:10` | Pairformer cross-attention Q[I_par] × K[I] with Z_chunk bias | `S_I_chunk`, `S_I` |
| 18 | `all_gather_concat()` | `parallel/utils.py` | Gathers S_I chunks after each Pairformer block | `S_I_chunk` → `S_I` |
| 19 | `ParallelDiffusionModule._diffusion_transformer_parallel()` | `parallel/diffusion_module.py:55` | Cross-attention A_I transformer | `A_I`, `Z_II_chunk` |
| 20 | `local_token_transformer_cross_attn()` | `parallel/layers/blocks.py:25` | Free function: chunk queries attend to all keys | `A_I_chunk`, `A_I`, `Z_chunk` |
| 21 | `structure_local_atom_block_cross_attn()` | `parallel/layers/blocks.py` | Free function: atom-level cross-attention block | — |
| 22 | `local_attention_cross_attn()` | `parallel/layers/attention.py` | Free function: local attention with cross-attention bias | — |
| 23 | `all_gather_along_dim()` | `parallel/utils.py` | Gathers A_I chunks → full A_I | `A_I_chunk` → `A_I` |
| 24 | `ParallelDiffusionModule._decoder_parallel()` or `_decoder_parallel_sparse()` | `parallel/diffusion_module.py:168` / `:260` | Decoder with parallel P_LL computation | `A_I`, `Q_L`, `P_LL_chunk` |
| 25 | `compact_decoder_parallel()` or `compact_decoder_parallel_sparse()` | `parallel/layers/blocks.py:385` / `:470` | Free functions: decoder cross-attention blocks | — |
| 26 | `local_attention_sparse_cross_attn()` | `parallel/layers/attention.py` | Free function: sparse cross-attention with indices | — |
| 27 | `chunked_pairwise_forward_parallel()` | `parallel/layers/chunked_pairwise.py:13` | Free function: sparse P_LL for GPU chunk only | `P_LL_sparse [L_par, k, c]` |
| 28 | `all_gather_along_dim()` | `parallel/utils.py` | Gathers Q_L, A_I chunks after decoder | `Q_L_chunk` → `Q_L` |
| 29 | `bucketize_scaled_distogram_chunked()` | `layers/block_utils.py` | D_II_self computed as chunk [B, I_par, I, n_bins] | `D_II_self_chunk` |
| 30 | `scale_positions_out()` | `RFD3_diffusion_module.py:170` | Inherited — denormalize positions | `X_out_L` |
| 31 | `_broadcast_tensor()` | `inference_sampler.py:164` | Sync X_denoised_L from rank 0 | `X_denoised_L` |

---

## 4. ASCII Call-Stack Tree

```
design_parallel.py                                           # scripts/design_parallel.py
│
├── main()                                                   # :802
│   ├── argparse + load_config(config_path)                  # :117
│   ├── merge_config_with_args(config, args)                 # :123
│   │
│   ├── if n_gpus > 1:
│   │   └── launch_with_torchrun(args, params, n_gpus)       # :762
│   │       └── subprocess.run(["torchrun", "--nproc_per_node=N", ...])
│   │           └── spawns N worker processes → run_worker(**params)
│   │
│   └── else: run_worker(**params)                           # :891
│
└── run_worker(                                              # :441
      out_dir, length, num_designs, symmetry,
      mpnn_batch_size, attention_parallel_factor,
      seed, **kwargs
    )
    ├── set_seed(seed)                                       # :88  (seed_everything + torch/numpy/random)
    ├── rank, world_size, local_rank = setup_distributed()   # :169 (init_process_group, nccl)
    │
    ├── os.environ["RFD3_LOW_MEMORY_MODE"] = "1"             # Always in parallel script
    ├── os.environ["RFD3_ATTENTION_PARALLEL"] = str(world_size)  # Triggers parallel classes
    │
    ├── rfd3_config = RFD3InferenceConfig(                   # engine.py:42
    │     specification={'length': length, ...},
    │     diffusion_batch_size=num_designs,
    │     ckpt_path=...,
    │     low_memory_mode=True,
    │     attention_parallel=True,
    │     attention_parallel_factor=...,
    │     seed=seed,
    │   )
    │
    ├── rfd3_engine = RFD3InferenceEngine(**rfd3_config)     # engine.py:242
    │   └── BaseInferenceEngine.__init__()                   # base.py:38
    │       └── _resolve_checkpoint_path()
    │
    ├── rfd3_outputs = rfd3_engine.run(                      # engine.py:343
    │     inputs=None, out_dir=None, n_batches=1
    │   )
    │   ├── _canonicalize_inputs()                           # Builds spec from config
    │   ├── initialize()                                     # base.py:125
    │   │   ├── torch.load(ckpt_path)
    │   │   ├── _construct_pipeline()                        # Builds transform pipeline
    │   │   └── _construct_trainer()                         # base.py:166
    │   │       ├── trainer.construct_model()
    │   │       │   └── RFD3(...)  ← FACTORY PATTERN (see below)
    │   │       └── trainer.load_checkpoint()
    │   │
    │   └── _run_multi(specs)                                # engine.py:382
    │       └── _model_forward(pipeline_output)              # engine.py:446
    │           ├── trainer.validation_step()                 # trainer/rfd3.py:189
    │           │   └── RFD3.forward()  ← ENTRY TO MODEL (see below)
    │           ├── _build_predicted_atom_array_stack()
    │           └── RFD3Output.dump()
    │
    ├── # Post-processing (rank 0 only):
    ├── MPNNInferenceEngine.run()                            # Sequence design
    ├── to_cif_file(...)                                     # Save structures
    └── cleanup_distributed()

─────────────────────────────────────────────────────────────
MODEL FORWARD (from RFD3.forward() downward):
─────────────────────────────────────────────────────────────

RFD3.__init__()                                              # RFD3.py:28
    │   ┌─ if RFD3_ATTENTION_PARALLEL != 0:
    │   │    from rfd3.model.parallel.layers.encoders import ParallelTokenInitializer
    │   │    from rfd3.model.parallel.diffusion_module import ParallelDiffusionModule
    │   │
    │   ├── self.token_initializer = ParallelTokenInitializer(...)
    │   └── self.diffusion_module = ParallelDiffusionModule(...)
    │         └── __init__()                                 # parallel/diffusion_module.py:46
    │             ├── super().__init__(**)                    # Creates standard sub-modules
    │             └── self.diffusion_token_encoder.__class__ = ParallelDiffusionTokenEncoder
    │
    └── RFD3.forward()                                       # RFD3.py:136
        │
        ├── ParallelTokenInitializer.forward(f)              # parallel/layers/encoders.py:227
        │   ├── # S_I embedding (5 steps):
        │   │   token_1d_embedder → transition_post_token → downcast_atom → transition_post_atom → process_s_init
        │   │                                                → S_I [I, c_s]
        │   │
        │   ├── if world_size > 1:                           # MULTI-GPU PATH
        │   │   ├── S_I, Z_chunk = _process_s_through_transformer_stack()  # :34
        │   │   │   ├── Compute base Z_chunk from initial S_I  [I_par, I, c_z]
        │   │   │   └── for block in self.transformer_stack:
        │   │   │       ├── z_transition_chunked(Z_chunk)    # parallel/utils.py
        │   │   │       ├── attention_pair_bias_forward_chunked()  # parallel/layers/pairformer_layers.py:10
        │   │   │       ├── block.s_transition(S_I_chunk)
        │   │   │       └── all_gather_concat(S_I_chunk)     # parallel/utils.py → S_I [I, c_s]
        │   │   │
        │   │   └── Post-transformer Z: cat(Z_chunk, RPE2) → process_z_chunked → z_transition ×2
        │   │                                                → Z_parallel [I_par, I, c_z]
        │   │
        │   ├── Q_L_init = self.atom_1d_embedder_2(f, L)    → [L, c_atom]
        │   ├── C_L = Q_L_init + process_s_trunk(S_I)[tok_idx]  → [L, c_atom]
        │   │
        │   └── return {S_I, Z_II=Z_parallel, Q_L_init, C_L, parallel_mode=True, z_chunk_range}
        │
        └── inference_sampler.sample_diffusion_like_af3()    # inference_sampler.py:306
            │   parallel_mode = initializer_outputs["parallel_mode"]  (= True)
            │
            └── for step in noise_schedule (200 steps):
                │
                ├── broadcast epsilon_L from rank 0           # Ensures identical X_noisy_L
                │
                ├── diffusion_module(X_noisy_L, t, f, parallel_mode=True, ...)
                │   │
                │   │── RFD3DiffusionModule.forward()        # RFD3_diffusion_module.py:198  (INHERITED)
                │   │   ├── scale_positions_in()             → R_L_uniform, R_noisy_L
                │   │   ├── process_time_()                  → time embeddings
                │   │   │
                │   │   ├── encoder(Q_L, C_L, P_LL/chunked)  # LocalAtomTransformer (standard)
                │   │   │   └── LocalAttentionPairBias        # Standard self-attention, NOT parallel
                │   │   │
                │   │   ├── downcast_q(Q_L → A_I)
                │   │   │
                │   │   └── forward_with_recycle(parallel_mode=True)  # :365  (INHERITED)
                │   │       └── for i in range(n_recycle):     # Default 3 iterations
                │   │           │
                │   │           └── ParallelDiffusionModule.process_()  # parallel/diffusion_module.py:460  (OVERRIDE)
                │   │               │
                │   │               ├── ParallelDiffusionTokenEncoder.forward()  # parallel/layers/encoders.py:376
                │   │               │   ├── S_I transitions (full I, no I×I)
                │   │               │   ├── Pre-allocate Z_chunk [B, I_par, I, final_dim]
                │   │               │   ├── Copy Z_init_chunk into first slice
                │   │               │   ├── Compute distogram D_chunk, copy into slice
                │   │               │   ├── Compute D_II_self chunk, copy into slice
                │   │               │   ├── process_z_chunked(Z_chunk)            # parallel/utils.py
                │   │               │   ├── z_transition_chunked(Z_chunk) ×2      # parallel/utils.py
                │   │               │   │
                │   │               │   ├── for block in self.pairformer_stack:    # 18 blocks
                │   │               │   │   ├── z_transition_chunked(Z_chunk)
                │   │               │   │   ├── attention_pair_bias_forward_chunked()  # parallel/layers/pairformer_layers.py
                │   │               │   │   │     Q=S_I_chunk [I_par, c_s]
                │   │               │   │   │     K=S_I [I, c_s]
                │   │               │   │   │     Bias=Z_chunk [I_par, I, c_z]
                │   │               │   │   ├── block.s_transition(S_I_chunk)
                │   │               │   │   └── all_gather_concat(S_I_chunk → S_I)  # ← 18 all_gathers per recycle!
                │   │               │   │
                │   │               │   └── return S_I [I, c_s], Z_chunk [I_par, I, c_z]
                │   │               │
                │   │               ├── _diffusion_transformer_parallel()         # parallel/diffusion_module.py:55
                │   │               │   ├── Slice A_I_chunk = A_I[:, start:end, :]
                │   │               │   ├── local_token_transformer_cross_attn()  # parallel/layers/blocks.py:25
                │   │               │   │   └── for block in self.blocks:
                │   │               │   │       ├── structure_local_atom_block_cross_attn()  # parallel/layers/blocks.py
                │   │               │   │       │   └── local_attention_cross_attn()         # parallel/layers/attention.py
                │   │               │   │       │         Q=A_I_chunk [B, I_par, c]
                │   │               │   │       │         K,V=A_I [B, I, c]
                │   │               │   │       │         Bias=Z_chunk [I_par, I, c_z]
                │   │               │   │       └── block transition layers
                │   │               │   │
                │   │               │   └── all_gather_along_dim(A_I_chunk → A_I)
                │   │               │
                │   │               ├── DECODER (4 modes based on z_is_chunked × chunked_embedder):
                │   │               │   │
                │   │               │   ├── z_is_chunked + chunked_embedder:     # BOTH modes combined
                │   │               │   │   └── _decoder_parallel_sparse()       # parallel/diffusion_module.py:260
                │   │               │   │       ├── compute_chunk_ranges(L)
                │   │               │   │       ├── compact_decoder_parallel_sparse()  # parallel/layers/blocks.py:470
                │   │               │   │       │   └── for block in self.blocks:
                │   │               │   │       │       ├── chunked_pairwise_forward_parallel()  # parallel/layers/chunked_pairwise.py:13
                │   │               │   │       │       │     → P_sparse [L_par, k, c_atompair]
                │   │               │   │       │       └── local_attention_sparse_cross_attn()   # parallel/layers/attention.py
                │   │               │   │       └── all_gather_along_dim(Q_L_chunk → Q_L)
                │   │               │   │
                │   │               │   ├── z_is_chunked only:                   # PARALLEL only
                │   │               │   │   └── _decoder_parallel()              # parallel/diffusion_module.py:168
                │   │               │   │       ├── _compute_P_LL_chunk()        → P_LL_chunk [L_par, L, c]
                │   │               │   │       ├── compact_decoder_parallel()   # parallel/layers/blocks.py:385
                │   │               │   │       │   └── local_attention_cross_attn()
                │   │               │   │       └── all_gather_along_dim(Q_L_chunk → Q_L)
                │   │               │   │
                │   │               │   ├── chunked_embedder only:               # LOW_MEM only
                │   │               │   │   └── self.decoder(... chunked_pairwise_embedder=...)  # Standard forward
                │   │               │   │
                │   │               │   └── neither:                             # Standard
                │   │               │       └── self.decoder(... P_LL=P_LL ...)
                │   │               │
                │   │               ├── self.to_r_update(Q_L)                    → R_update_L [B, L, 3]
                │   │               ├── self.scale_positions_out()               → X_out_L [B, L, 3]
                │   │               ├── self.sequence_head(A_I)                  → sequence_logits_I
                │   │               │
                │   │               └── D_II_self = bucketize_scaled_distogram_chunked()  → [B, I_par, I, n_bins]
                │   │                    (kept chunked — NOT all_gathered to save ~21 GB)
                │   │
                │   └── return {X_L, sequence_logits_I, ...}
                │
                ├── broadcast X_denoised_L from rank 0        # Ensures consistent ODE update
                ├── ODE update: X_L = X_noisy_L + step_scale * d_t * delta_L
                └── Move trajectories to CPU (parallel mode memory optimization)
```

---

## 5. Detailed Data Flow per Component

### 5a. ParallelTokenInitializer

```
ParallelTokenInitializer.forward(f)                 # parallel/layers/encoders.py:227
│
├── # S_I embedding pipeline (5 steps, matches standard path exactly):
├── S_I = self.token_1d_embedder(f, I)              [I, c_s]        # Step 1: residue type + conditioning
├── S_I = S_I + self.transition_post_token(S_I)     [I, c_s]        # Step 2: post-token SwiGLU
├── S_I = self.downcast_atom(                                        # Step 3: atom→token pooling
│       Q_L=self.atom_1d_embedder_1(f, L),          [L, c_s]
│       A_I=S_I,                                     [I, c_s]
│       tok_idx=tok_idx                              [L]
│   )                                                [I, c_s]
├── S_I = S_I + self.transition_post_atom(S_I)      [I, c_s]        # Step 4: post-atom SwiGLU
├── S_I = self.process_s_init(S_I)                  [I, c_s]        # Step 5: linear projection
│
├── if world_size == 1:   ────────────────────────── SINGLE GPU FALLBACK
│   ├── Compute full Z_II [I, I, c_z]
│   ├── transformer_stack(S_I, Z_II, f)             [I, c_s], [I, I, c_z]
│   └── z_chunk_range = (0, I)
│
├── if world_size > 1:    ────────────────────────── MULTI-GPU PATH
│   ├── _process_s_through_transformer_stack()
│   │   ├── compute_chunk_ranges(I, world_size)
│   │   ├── Compute base Z_chunk from initial S_I:
│   │   │     Z_i = to_z_init_i(S_I_chunk)          [I_par, 1, c_z]
│   │   │     Z_j = to_z_init_j(S_I)                [1, I, c_z]
│   │   │     Z_chunk = Z_i + Z_j                   [I_par, I, c_z]
│   │   │     + RPE via forward_chunk()
│   │   │     + token_bonds[start_i:end_i, :]
│   │   │     + ref_pos_embedder_tok.forward_chunk()
│   │   │
│   │   └── for block in transformer_stack:
│   │       Z_chunk ← z_transition_chunked(Z_chunk, block.z_transition)
│   │       S_I_chunk ← attention_pair_bias_forward_chunked(block.attention_pair_bias, ...)
│   │       S_I_chunk ← S_I_chunk + block.s_transition(S_I_chunk)
│   │       S_I ← all_gather_concat(S_I_chunk)      ← SYNC POINT
│   │   → returns (S_I, Z_chunk)                     ← Z_chunk has ALL z_transitions!
│   │
│   └── Post-transformer Z processing (uses Z_chunk from above, NOT recomputed):
│         Z_chunk = cat([Z_chunk, RPE2.forward_chunk(f)], dim=-1)  [I_par, I, 2*c_z]
│         Z_chunk = process_z_chunked(Z_chunk, self.process_z_init)  [I_par, I, c_z]
│         Z_chunk = z_transition_chunked(Z_chunk, self.transition_1[0])
│         Z_chunk = z_transition_chunked(Z_chunk, self.transition_1[1])
│         Z_parallel = Z_chunk                       [I_par, I, c_z]
│
├── Q_L_init = self.atom_1d_embedder_2(f, L)        [L, c_atom]
├── C_L = Q_L_init + self.process_s_trunk(S_I)[..., tok_idx, :]  [L, c_atom]
│
└── return {
      S_I:             [I, c_s],
      Z_II:            Z_parallel [I_par, I, c_z],   ← NOT full I×I!
      Q_L_init:        [L, c_atom],                  ← NOTE: Q_L_init, not Q_L
      C_L:             [L, c_atom],
      parallel_mode:   True,
      z_chunk_range:   (start_i, end_i),
    }
```

### 5b. ParallelDiffusionTokenEncoder

```
ParallelDiffusionTokenEncoder.forward(f, R_L, S_init_I, Z_init_II=Z_chunk, ...)
│
├── S_I transitions (full I, no I×I):
│     for b in range(2): S_I = S_I + self.transition_1[b](S_I)
│
├── Pre-allocate Z_chunk [B, I_par, I, final_dim]:
│     final_dim = c_z + distogram_dim + self_cond_dim
│
├── Fill Z_chunk slices (avoids torch.cat memory spikes):
│     [:,:,:, 0:c_z]                 ← Z_init_chunk [I_par, I, c_z] expanded to batch
│     [:,:,:, c_z:c_z+dist_dim]      ← D_chunk from dist_embedder.forward_chunk() or bucketized
│     [:,:,:, c_z+dist_dim:end]       ← D_II_self (already chunked [B, I_par, I, n_bins])
│
├── Z_chunk = process_z_chunked(Z_chunk, self.process_z)    [B, I_par, I, c_z]
├── Z_chunk = z_transition_chunked(Z_chunk) × 2             [B, I_par, I, c_z]
│
├── Pairformer (18 blocks):
│     for block in self.pairformer_stack:
│       Z_chunk ← z_transition_chunked(Z_chunk, block.z_transition)
│       S_I_chunk ← attention_pair_bias_forward_chunked(
│           block.attention_pair_bias,
│           A_I_query=S_I_chunk,    [I_par, c_s]
│           A_I_key=S_I,            [I, c_s]
│           Z_chunk=Z_chunk[0],     [I_par, I, c_z]
│       )
│       S_I_chunk += block.s_transition(S_I_chunk)
│       S_I = all_gather_concat(S_I_chunk)               ← SYNC POINT (×18)
│
└── return S_I [I, c_s], Z_chunk [I_par, I, c_z]
```

### 5c. ParallelDiffusionModule.process_()

```
process_(D_II_self, X_L_self, ..., parallel_mode=True)
│
├── 1. TOKEN ENCODER:
│   S_I, Z_II = self.diffusion_token_encoder(         # ParallelDiffusionTokenEncoder
│       Z_init_II=Z_II,              [I_par, I, c_z]   ← chunk, NOT full!
│       D_II_self=D_II_self,          [B, I_par, I, n_bins] ← chunk too!
│       parallel_mode=True,
│       z_chunk_range=(start_i, end_i),
│   )
│   → S_I [I, c_s] (all_gathered), Z_II [I_par, I, c_z] (still chunked)
│
├── 2. TRANSFORMER (if z_is_chunked):
│   A_I = _diffusion_transformer_parallel(
│       A_I,                          [B, I, c_token]
│       Z_II_chunk=Z_II,             [I_par, I, c_z]
│   )
│   → Uses local_token_transformer_cross_attn()  (free function)
│   → all_gather_along_dim(A_I_chunk) → A_I [B, I, c_token]
│
├── 3. DECODER (4-way branch):
│   ┌─────────────────┬───────────────────────┬─────────────────────────────────────┐
│   │ z_is_chunked    │ chunked_embedder      │ Code path                          │
│   ├─────────────────┼───────────────────────┼─────────────────────────────────────┤
│   │ True            │ Present               │ _decoder_parallel_sparse()         │
│   │ True            │ None                  │ _decoder_parallel()                │
│   │ False           │ Present               │ self.decoder(chunked=...)          │
│   │ False           │ None                  │ self.decoder(P_LL=P_LL)            │
│   └─────────────────┴───────────────────────┴─────────────────────────────────────┘
│   → A_I [B, I, c_token], Q_L [B, L, c_atom] (all_gathered if parallel)
│
├── 4. OUTPUTS:
│   R_update_L = self.to_r_update(Q_L)                [B, L, 3]
│   X_out_L = self.scale_positions_out(R_update_L)    [B, L, 3]
│   sequence_logits_I = self.sequence_head(A_I)
│
└── 5. SELF-CONDITIONING (if z_is_chunked):
    D_II_self = bucketize_scaled_distogram_chunked(
        X_out_L[..., is_ca, :],
        query_start=start_i, query_end=end_i,
    )                                                  [B, I_par, I, n_bins]
    ← Kept chunked! NOT all_gathered! Saves ~21 GB.
```

---

## 6. All-Gather Synchronization Points

### Per Diffusion Step (within one recycle iteration)

| Stage | Where | What is Gathered | Shape Change | Count |
|-------|-------|------------------|--------------|-------|
| TokenInitializer transformer_stack | `parallel/layers/encoders.py:196` | S_I | [I_par, c_s] → [I, c_s] | N_blocks (varies) |
| DiffusionTokenEncoder pairformer | `parallel/layers/encoders.py:747` | S_I | [I_par, c_s] → [I, c_s] | 18 |
| DiffusionTokenEncoder final | `parallel/layers/encoders.py:758` | S_I | [I_par, c_s] → [I, c_s] | 1 |
| Diffusion transformer | `parallel/diffusion_module.py:160` | A_I | [B, I_par, c_token] → [B, I, c_token] | 1 |
| Decoder (parallel) | `parallel/layers/blocks.py` | Q_L, A_I | [B, L_par, c] → [B, L, c] | 2 |

### Per Complete Generation (99 steps × 3 recycles)

```
Total all_gathers ≈ 99 × 3 × (18 + 1 + 1 + 2 + N_encoder_blocks) per GPU
                  ≈ 99 × 3 × ~25 ≈ 7,425 all_gather operations
```

This is the primary performance bottleneck (see CLAUDE.md timing analysis).

---

## 7. Free Function Dispatch Table

After refactoring, parallel methods were extracted from base classes as **free functions** that take the base class instance as the first argument.

| Free Function | Location | Replaces (removed from base) | Called by |
|--------------|----------|------------------------------|-----------|
| `local_token_transformer_cross_attn(self, ...)` | `parallel/layers/blocks.py:25` | `LocalTokenTransformer.forward_cross_attn()` | `ParallelDiffusionModule._diffusion_transformer_parallel()` |
| `structure_local_atom_block_cross_attn(block, ...)` | `parallel/layers/blocks.py` | `StructureLocalAtomTransformerBlock.forward_cross_attn()` | `local_token_transformer_cross_attn()` |
| `local_atom_transformer_parallel(self, ...)` | `parallel/layers/blocks.py` | `LocalAtomTransformer.forward_parallel()` | `compact_decoder_parallel()` |
| `compact_decoder_parallel(self, ...)` | `parallel/layers/blocks.py:385` | `CompactDecoder.forward_parallel()` | `ParallelDiffusionModule._decoder_parallel()` |
| `compact_decoder_parallel_sparse(self, ...)` | `parallel/layers/blocks.py:470` | `CompactDecoder.forward_parallel_sparse()` | `ParallelDiffusionModule._decoder_parallel_sparse()` |
| `local_attention_cross_attn(self, ...)` | `parallel/layers/attention.py` | `LocalAttentionPairBias.forward_cross_attn()` | `structure_local_atom_block_cross_attn()` |
| `local_attention_sparse_cross_attn(self, ...)` | `parallel/layers/attention.py` | `LocalAttentionPairBias.forward_sparse_cross_attn()` | `compact_decoder_parallel_sparse()` |
| `sparse_cross_attention(Q, K, V, B, ...)` | `parallel/layers/attention.py:16` | (new helper) | `local_attention_sparse_cross_attn()` |
| `attention_pair_bias_forward_chunked(self, ...)` | `parallel/layers/pairformer_layers.py:10` | `AttentionPairBiasPairformerDeepspeed.forward_chunked()` | `ParallelTokenInitializer`, `ParallelDiffusionTokenEncoder` |
| `chunked_pairwise_forward_parallel(self, ...)` | `parallel/layers/chunked_pairwise.py:13` | `ChunkedPairwiseEmbedder.forward_chunked_parallel()` | `compact_decoder_parallel_sparse()` via `local_atom_transformer_parallel()` |

### `forward_chunk` Methods Kept on Base Classes

These small methods remain on base classes (used only by parallel code, but too tightly coupled to `self` attributes):

| Method | Class | File | Note |
|--------|-------|------|------|
| `forward_chunk()` | `PositionPairDistEmbedder` | `layers/blocks.py` | Used by parallel mode only |
| `forward_chunk()` | `SinusoidalDistEmbed` | `layers/blocks.py` | Used by parallel mode only |
| `forward_chunk()` | `RelativePositionEncodingWithIndexRemoval` | `layers/blocks.py` | Used by parallel mode only |
| `forward_chunked()` | `ChunkedPairwiseEmbedder` | `layers/chunked_pairwise.py` | Standard low-mem method (not parallel-specific) |

---

## 8. Tensor Shape Reference

### Standard vs Parallel

| Tensor | Standard | Parallel (per GPU) | Where Computed |
|--------|----------|-------------------|----------------|
| `Z_II` | `[I, I, c_z]` | `[I_par, I, c_z]` | `ParallelTokenInitializer.forward()` |
| `D_II_self` | `[B, I, I, n_bins]` | `[B, I_par, I, n_bins]` | `ParallelDiffusionModule.process_()` |
| `P_LL` | `[L, L, c_atompair]` | `[L_par, L, c_atompair]` | `_compute_P_LL_chunk()` |
| `P_LL_sparse` | `[L, k, c_atompair]` | `[L_par, k, c_atompair]` | `chunked_pairwise_forward_parallel()` |
| `S_I` | `[I, c_s]` | `[I, c_s]` (all_gathered) | After each Pairformer block |
| `A_I` | `[B, I, c_token]` | `[B, I, c_token]` (all_gathered) | After transformer/decoder |
| `Q_L` | `[B, L, c_atom]` | `[B, L, c_atom]` (all_gathered) | After decoder |
| `X_L` | `[D, L, 3]` | `[D, L, 3]` (broadcast from rank 0) | After ODE update |

### Chunk Division

```
I_par = I / world_size  (approximately, with remainder on last GPU)

compute_chunk_ranges(I=10500, world_size=6):
  GPU 0: [0, 1750)       I_par = 1750
  GPU 1: [1750, 3500)     I_par = 1750
  GPU 2: [3500, 5250)     I_par = 1750
  GPU 3: [5250, 7000)     I_par = 1750
  GPU 4: [7000, 8750)     I_par = 1750
  GPU 5: [8750, 10500)    I_par = 1750
```

---

## 9. Class Hierarchy

```
TokenInitializer                          (layers/encoders.py)
└── ParallelTokenInitializer              (parallel/layers/encoders.py)
    Overrides: forward()
    Adds: _process_s_through_transformer_stack()

DiffusionTokenEncoder                     (layers/encoders.py)
└── ParallelDiffusionTokenEncoder         (parallel/layers/encoders.py)
    Overrides: forward()
    No __init__ override (class-swapped by ParallelDiffusionModule.__init__)

RFD3DiffusionModule                       (RFD3_diffusion_module.py)
└── ParallelDiffusionModule               (parallel/diffusion_module.py)
    Overrides: __init__(), process_()
    Adds: _diffusion_transformer_parallel(), _decoder_parallel(), _decoder_parallel_sparse(), _compute_P_LL_chunk()
    Inherits: forward(), forward_with_recycle(), scale_positions_in/out(), process_time_()

Standard classes (NOT subclassed — used via free functions):
  LocalTokenTransformer                   (layers/blocks.py)
  CompactDecoder                          (layers/blocks.py)
  LocalAttentionPairBias                  (layers/attention.py)
  AttentionPairBiasPairformerDeepspeed    (layers/pairformer_layers.py)
  ChunkedPairwiseEmbedder                (layers/chunked_pairwise.py)
```

### Import Direction (no circular imports)

```
parallel/ → layers/          ✓  (parallel imports from base)
layers/   → parallel/        ✗  (NEVER — base never imports from parallel)
RFD3.py   → parallel/        ✓  (factory pattern, conditional import)
RFD3.py   → layers/          ✓  (standard classes)
```

---

## Summary of Changes from Pre-Refactoring

| Aspect | Before | After |
|--------|--------|-------|
| Parallel code location | Interleaved in base classes | Dedicated `parallel/` package |
| Method dispatch | Instance methods on base classes (e.g., `self.decoder.forward_parallel()`) | Free functions (e.g., `compact_decoder_parallel(self.decoder, ...)`) |
| Class selection | Runtime `if` checks in every `forward()` | Factory pattern in `RFD3.__init__()` + class hierarchy |
| Utility functions | Duplicated across 3 files | Consolidated in `parallel/utils.py` |
| `DiffusionTokenEncoder` | Had both `_forward_standard()` and `_forward_streaming()` | Base has only standard; `ParallelDiffusionTokenEncoder` overrides `forward()` |
| `forward_cross_attn` etc. | Methods on base classes | Removed from base; exist as free functions in `parallel/layers/` |
| Base class complexity | 926+ lines in `RFD3_diffusion_module.py` | ~552 lines (standard only) |
