"""
Parallel implementations of encoder classes for multi-GPU inference.

Classes extracted from encoders.py for parallel (multi-GPU) mode.
"""

import gc
import os
import torch
import torch.distributed as dist

from rfd3.model.layers.encoders import TokenInitializer, DiffusionTokenEncoder
from rfd3.model.debug_context import debug_ctx, debug_tensor_all_ranks, verify_tensor_sync, debug_memory
from rfd3.model.parallel.utils import (
    get_gpu_rank_and_world_size,
    compute_chunk_ranges,
    all_gather_concat,
    z_transition_chunked,
    process_z_chunked,
)
from rfd3.model.layers.block_utils import bucketize_scaled_distogram_chunked


class ParallelTokenInitializer(TokenInitializer):
    """
    Parallel (multi-GPU) version of TokenInitializer.

    Overrides forward() to compute Z_chunk [I_par, I, c_z] per GPU instead of
    materializing full Z_II [I, I, c_z].

    Memory: O(I²/N) per GPU instead of O(I²).
    """

    def _process_s_through_transformer_stack(
        self,
        S_I: torch.Tensor,     # [I, c_s]
        f: dict,
        I: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Process S_I through transformer_stack using chunked attention.

        In standard mode, transformer_stack.forward does:
            Z_II = Z_II + z_transition(Z_II)
            S_I = S_I + attention_pair_bias(S_I, None, Z_II, ...)
            S_I = S_I + s_transition(S_I)

        IMPORTANT: Z_II accumulates z_transition updates across blocks!
        Block 0 applies z_transition_0, Block 1 applies z_transition_1, etc.
        The attention bias at block N uses Z with all previous transitions applied.

        In parallel mode, we can't materialize full Z_II, so we:
        1. Compute base Z_chunk for this GPU's query range
        2. Maintain Z_chunk across blocks, applying z_transition at each block
        3. Apply chunked attention to S_I
        4. All-gather S_I chunks to reconstruct full S_I

        Args:
            S_I: Single features [I, c_s]
            f: Feature dictionary
            I: Number of tokens
            device: Device for tensors
            dtype: Dtype for tensors

        Returns:
            S_I: Updated single features [I, c_s]
        """
        # Ensure S_I is on the correct device for this rank
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            expected_device = torch.device(f"cuda:{local_rank}")
            if S_I.device != expected_device:
                S_I = S_I.to(expected_device)
            # Override device parameter with computed expected_device for consistency
            device = expected_device
        else:
            expected_device = device

        # Get GPU range for chunking
        # CRITICAL: Use actual S_I.shape[0] instead of I parameter!
        # I comes from len(f["restype"]) which may include special tokens,
        # but S_I comes from token_1d_embedder which produces length × symmetry tokens.
        actual_I = S_I.shape[0]
        gpu_rank, world_size = get_gpu_rank_and_world_size()
        chunk_ranges = compute_chunk_ranges(actual_I, world_size)

        if gpu_rank >= len(chunk_ranges):
            return S_I  # Edge case: more GPUs than tokens

        start_i, end_i = chunk_ranges[gpu_rank]
        I_par = end_i - start_i

        # DEBUG: Helper function for printing tensor stats (define before use)
        def _stat_tensor(t):
            return {
                "shape": list(t.shape),
                "mean": float(t.float().mean().item()),
                "std": float(t.float().std().item()),
                "min": float(t.float().min().item()),
                "max": float(t.float().max().item()),
            }

        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

        # Pre-compute Z_j once (used for base Z)
        Z_j = self.to_z_init_j(S_I)                           # [I, c_z]

        # Get reference positions for Z computation
        ref_pos = f["ref_pos"][f["is_ca"]]                    # [I, 3]
        ref_space_uid = f["ref_space_uid"][f["is_ca"]]        # [I]

        # ================================================================
        # Compute base Z_chunk ONCE (before any z_transition)
        # CRITICAL: This matches standard mode where Z_init_II is computed
        # from INITIAL S_I before the transformer_stack loop
        # ================================================================
        S_I_chunk = S_I[start_i:end_i]                        # [I_par, c_s]

        # DEBUG: Print S_I_chunk used for Z_chunk computation
        if is_rank0 and debug_ctx.stats_enabled:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} Z_CHUNK_COMPUTATION: S_I_chunk={_stat_tensor(S_I_chunk)}, Z_j={_stat_tensor(Z_j)}", flush=True)

        Z_i = self.to_z_init_i(S_I_chunk).unsqueeze(-2)       # [I_par, 1, c_z]
        Z_j_full = Z_j.unsqueeze(0)                           # [1, I, c_z]
        Z_chunk = Z_i + Z_j_full                              # [I_par, I, c_z]

        # DEBUG: Print initial Z_chunk
        if is_rank0 and debug_ctx.stats_enabled:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} Z_CHUNK_INITIAL: Z_chunk={_stat_tensor(Z_chunk)}", flush=True)

        # Add RPE
        Z_chunk = Z_chunk + self.relative_position_encoding.forward_chunk(
            f, start_i, end_i
        )                                                      # [I_par, I, c_z]

        # Add token bonds
        token_bonds_chunk = f["token_bonds"][start_i:end_i, :]  # [I_par, I]
        Z_chunk = Z_chunk + self.process_token_bonds(
            token_bonds_chunk.unsqueeze(-1).float()
        )                                                      # [I_par, I, c_z]

        # Add reference position embedding
        ref_pos_chunk = ref_pos[start_i:end_i]                # [I_par, 3]
        ref_space_uid_chunk = ref_space_uid[start_i:end_i]    # [I_par]
        valid_mask = (
            ref_space_uid_chunk.unsqueeze(-1) == ref_space_uid.unsqueeze(0)
        ).unsqueeze(-1)                                        # [I_par, I, 1]
        ref_pos_embed = self.ref_pos_embedder_tok.forward_chunk(
            ref_pos_chunk, ref_pos, valid_mask
        )                                                      # [I_par, I, c_z]
        Z_chunk = Z_chunk + ref_pos_embed

        # ================================================================
        # Process through transformer_stack (Z_chunk accumulates updates!)
        # ================================================================
        # Note: _stat_tensor and is_rank0 are already defined above

        for block_idx, block in enumerate(self.transformer_stack):
            # DEBUG: Print S_I stats at start of each block to verify it's being updated
            if is_rank0 and debug_ctx.stats_enabled:
                print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} START: S_I={_stat_tensor(S_I)}, device={S_I.device}", flush=True)

            # Step 1: Apply z_transition (ACCUMULATES across blocks)
            Z_chunk = z_transition_chunked(Z_chunk, block.z_transition)

            # Step 2: Apply attention_pair_bias to S_I using Z_chunk as bias
            if hasattr(block, 'attention_pair_bias'):
                # CRITICAL: Slice S_I at the start of each iteration to get updated values
                # S_I should have been updated from the previous iteration's all_gather
                S_I_chunk = S_I[start_i:end_i].clone()        # [I_par, c_s] - clone to ensure fresh tensor

                # DEBUG: Print S_I_chunk stats after slicing
                if is_rank0 and debug_ctx.stats_enabled:
                    print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} AFTER_SLICE: S_I_chunk={_stat_tensor(S_I_chunk)}", flush=True)

                from rfd3.model.parallel.layers.pairformer_layers import attention_pair_bias_forward_chunked
                S_I_chunk = S_I_chunk + attention_pair_bias_forward_chunked(
                    block.attention_pair_bias,                  # AttentionPairBiasPairformerDeepspeed instance
                    A_I_query=S_I_chunk,                       # [I_par, c_s]
                    A_I_key=S_I,                               # [I, c_s]
                    Z_chunk=Z_chunk,                           # [I_par, I, c_z]
                    Beta_II=torch.tensor([0.0], device=device),
                )                                              # [I_par, c_s]
                S_I_chunk = S_I_chunk + block.s_transition(S_I_chunk)

                # DEBUG: Print S_I_chunk stats after processing
                if is_rank0 and debug_ctx.stats_enabled:
                    print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} AFTER_PROCESS: S_I_chunk={_stat_tensor(S_I_chunk)}", flush=True)

                # Step 3: All-gather S_I chunks to update full S_I for next block
                # CRITICAL: This updates S_I for the next iteration
                # Pass total_size=actual_I to handle uneven chunk sizes across GPUs
                if world_size > 1:
                    S_I_gathered = all_gather_concat(S_I_chunk, dim=0, total_size=actual_I)  # [I, c_s]
                    # Ensure S_I is on the correct device after all_gather
                    if S_I_gathered.device != expected_device:
                        S_I_gathered = S_I_gathered.to(expected_device)

                    # DEBUG: Print S_I stats from ALL ranks after all_gather to verify sync
                    debug_tensor_all_ranks("ENCODER-S_I", f"block{block_idx}_AFTER_ALLGATHER", S_I_gathered)

                    # MULTI-GPU DIAGNOSTIC: Verify S_I is synchronized across all ranks
                    # This broadcasts from rank 0 and compares on all ranks
                    verify_tensor_sync("ENCODER-S_I", f"block{block_idx}_S_I_SYNC", S_I_gathered)

                    S_I = S_I_gathered
                else:
                    S_I = S_I_chunk
                    # DEBUG: Single GPU case
                    if is_rank0 and debug_ctx.stats_enabled:
                        print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} SINGLE_GPU: S_I={_stat_tensor(S_I)}", flush=True)

                # Memory cleanup after each block to prevent excessive accumulation
                # (parallel mode memory optimization)
                del S_I_chunk
                torch.cuda.empty_cache()

        # MULTI-GPU DIAGNOSTIC: Final S_I sync check
        if world_size > 1:
            debug_tensor_all_ranks("ENCODER-S_I", "FINAL_S_I", S_I)
            verify_tensor_sync("ENCODER-S_I", "FINAL_S_I_SYNC", S_I)

        return S_I, Z_chunk

    def forward(self, f):
        """
        Parallel forward: computes Z_chunk [I_par, I, c_z] per GPU.

        Returns dict with parallel_mode=True and z_chunk_range metadata.
        """
        # Determine correct device for this rank
        import torch.distributed as dist
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = f["restype"].device
        dtype = torch.bfloat16 if "dtype" not in f else f["dtype"]

        # CRITICAL: Set CUDA device context for this rank
        torch.cuda.set_device(device)

        rank = dist.get_rank() if dist.is_initialized() else 0

        self.to(device)
        torch.cuda.synchronize(device)

        # Move feature dict tensors to correct device
        for key, val in f.items():
            if isinstance(val, torch.Tensor) and val.device != device:
                f[key] = val.to(device)

        # Get GPU range for chunking
        gpu_rank, world_size = get_gpu_rank_and_world_size()

        # Compute dimensions (needed for S_I embedding below)
        tok_idx = f["atom_to_token_map"]                      # [L]
        L = len(tok_idx)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        # S_I embedding (must match standard path in _forward_standard/init_tokens exactly)
        S_I = self.token_1d_embedder(f, I)                    # [I, c_s]
        S_I = S_I + self.transition_post_token(S_I)           # [I, c_s]
        S_I = self.downcast_atom(
            Q_L=self.atom_1d_embedder_1(f, L),                # [L, c_s]
            A_I=S_I,                                           # [I, c_s]
            tok_idx=tok_idx                                    # [L]
        )                                                      # [I, c_s]
        S_I = S_I + self.transition_post_atom(S_I)            # [I, c_s]
        S_I = self.process_s_init(S_I)                        # [I, c_s]

        # SINGLE GPU FALLBACK: If only 1 GPU, use standard transformer_stack
        if world_size == 1:
            # Standard path: materialize full Z_II [I, I, c_z]
            Z_i = self.to_z_init_i(S_I).unsqueeze(-2)         # [I, 1, c_z]
            Z_j = self.to_z_init_j(S_I).unsqueeze(0)          # [1, I, c_z]
            Z_II = Z_i + Z_j                                  # [I, I, c_z]

            # Add RPE
            Z_II = Z_II + self.relative_position_encoding(f)  # [I, I, c_z]

            # Add token bonds
            token_bonds = f["token_bonds"]                    # [I, I]
            Z_II = Z_II + self.process_token_bonds(
                token_bonds.unsqueeze(-1).float()
            )                                                 # [I, I, c_z]

            # Add reference position embedding
            ref_pos = f["ref_pos"][f["is_ca"]]               # [I, 3]
            ref_space_uid = f["ref_space_uid"][f["is_ca"]]   # [I]
            valid_mask = (
                ref_space_uid.unsqueeze(-1) == ref_space_uid.unsqueeze(0)
            ).unsqueeze(-1)                                   # [I, I, 1]
            ref_pos_embed = self.ref_pos_embedder_tok(
                ref_pos, valid_mask
            )                                                 # [I, I, c_z]
            Z_II = Z_II + ref_pos_embed

            # Process through transformer_stack (iterate over blocks, matching standard path)
            for block in self.transformer_stack:
                S_I, Z_II = block(S_I, Z_II)                  # [I, c_s], [I, I, c_z]

            # Post-transformer Z processing (matches standard path exactly)
            Z_II = torch.cat(
                [Z_II, self.relative_position_encoding2(f)],
                dim=-1,
            )                                                  # [I, I, 2*c_z]
            Z_II = self.process_z_init(Z_II)                  # [I, I, c_z]
            for b in range(2):
                Z_II = Z_II + self.transition_1[b](Z_II)      # [I, I, c_z]

            Z_parallel = Z_II
            z_chunk_range = (0, I)
        else:
            # MULTI-GPU PATH: Compute Z_chunk [I_par, I, c_z]
            chunk_ranges = compute_chunk_ranges(I, world_size)
            if gpu_rank >= len(chunk_ranges):
                raise RuntimeError(f"GPU rank {gpu_rank} >= {len(chunk_ranges)} chunks")

            start_i, end_i = chunk_ranges[gpu_rank]
            I_par = end_i - start_i

            # Process S_I through transformer_stack using chunked attention.
            # Returns BOTH updated S_I and Z_chunk with accumulated z_transitions.
            # Z_chunk was computed from initial S_I (before transformer_stack),
            # matching the standard path where Z goes through block(S_I, Z_II).
            S_I, Z_chunk = self._process_s_through_transformer_stack(S_I, f, I, device, dtype)

            # Post-transformer Z processing (matches standard path exactly)
            # Standard path does: cat(Z, RPE2) → process_z_init → transition_1[0..1]
            rpe2_chunk = self.relative_position_encoding2.forward_chunk(
                f, start_i, end_i
            )                                                 # [I_par, I, c_z]
            Z_chunk = torch.cat(
                [Z_chunk, rpe2_chunk], dim=-1
            )                                                 # [I_par, I, 2*c_z]
            Z_chunk = process_z_chunked(Z_chunk, self.process_z_init)  # [I_par, I, c_z]
            for b in range(2):
                Z_chunk = z_transition_chunked(
                    Z_chunk, self.transition_1[b]
                )                                             # [I_par, I, c_z]

            Z_parallel = Z_chunk
            z_chunk_range = (start_i, end_i)

            # Memory cleanup
            gc.collect()
            torch.cuda.empty_cache()

        # Compute atom-level features (shared with standard mode)
        # tok_idx and L already computed at top of forward()
        Q_L_init = self.atom_1d_embedder_2(f, L)             # [L, c_atom]
        C_L = Q_L_init + self.process_s_trunk(S_I)[..., tok_idx, :]  # [L, c_atom]

        # Return dict with parallel_mode=True flag
        # Key names must match RFD3DiffusionModule.forward() signature
        result = {
            "S_I": S_I,                                       # [I, c_s]
            "Z_II": Z_parallel,                              # [I_par, I, c_z] or [I, I, c_z]
            "Q_L_init": Q_L_init,                             # [L, c_atom]
            "C_L": C_L,                                       # [L, c_atom]
            "parallel_mode": True,                            # Parallel mode flag
            "z_chunk_range": z_chunk_range,                   # (start_i, end_i)
        }
        if self.use_chunked_pll:
            result["chunked_pairwise_embedder"] = self.chunked_pairwise_embedder
        return result


class ParallelDiffusionTokenEncoder(DiffusionTokenEncoder):
    """
    Parallel (multi-GPU) version of DiffusionTokenEncoder.

    Overrides forward() to process Z_chunk [I_par, I, c_z] without materializing
    full Z_II [I, I, c_z].

    Memory: O(I²/N) per GPU instead of O(I²).
    """

    def forward(self, f, R_L, S_init_I, Z_init_II, **kwargs):
        """
        Multi-GPU parallel forward: each GPU processes only its query chunk.

        Args:
            Z_init_II: Pre-computed Z chunk tensor [I_par, I, c_z]
            **kwargs: Must include z_chunk_range=(start_i, end_i)

        Multi-GPU parallelism:
        - Each GPU processes Z rows for its assigned query range [gpu_start, gpu_end)
        - This produces [I_par, I, c_z] per GPU, never full [I, I, c_z]
        - Single features S_I are computed per-GPU then all_gathered

        Returns:
            S_I: [I, c_s] - updated single features (all_gathered)
            Z_II: [I_par, I, c_z] - THIS GPU's Z rows only (NOT full I×I!)
        """
        B = R_L.shape[0]
        device = R_L.device
        dtype = R_L.dtype

        # CRITICAL: Set CUDA device context for this rank BEFORE any operations
        # In multi-GPU setups, operations may default to cuda:0 unless explicitly set.
        # This ensures all tensor allocations go to the correct device.
        torch.cuda.set_device(device)

        debug_memory("DIFF_ENCODER", "start_parallel")

        # CRITICAL: Ensure encoder modules are on the correct device for this rank
        # In multi-node setups, the model may have been placed on cuda:0 by Fabric
        # but each rank needs parameters on its local device (cuda:LOCAL_RANK)
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        self.to(device)

        # Verify move was successful and synchronize
        torch.cuda.synchronize(device)

        # CRITICAL: Move feature dict tensors to correct device (multi-node fix)
        # In multi-node setups, f tensors may have been created on a different device
        # They are used at lines 1360 (f["is_ca"]) and 1366 (f["is_motif_atom_with_fixed_coord"])
        for key, val in f.items():
            if isinstance(val, torch.Tensor) and val.device != device:
                f[key] = val.to(device)

        # Z_init_II is a pre-computed chunk tensor [I_par, I, c_z]
        Z_init_chunk = Z_init_II  # Rename to clarify it's the input

        # CRITICAL: Ensure Z_init_chunk is on the same device as R_L
        # In multi-node setups, Z_init_II may have been left on a different device
        if Z_init_chunk.device != device:
            Z_init_chunk = Z_init_chunk.to(device)

        z_chunk_range = kwargs.get("z_chunk_range")
        if z_chunk_range is None:
            raise ValueError("parallel_mode=True requires z_chunk_range in kwargs")
        gpu_start, gpu_end = z_chunk_range
        I = Z_init_chunk.shape[1]                       # Second dim is full I
        _, world_size = get_gpu_rank_and_world_size()
        base_z_dim = Z_init_chunk.shape[-1]             # c_z from tensor

        I_par = gpu_end - gpu_start

        # Step 1: Update S_I (operates on full I, no I×I)
        # Ensure S_init_I is on the correct device (may differ in multi-node)
        S_I = S_init_I.to(device) if S_init_I.device != device else S_init_I
        has_batch_S = S_I.dim() == 3
        for b in range(2):
            S_I = S_I + self.transition_1[b](S_I)

        # =======================================================================
        # MEMORY OPTIMIZATION: Pre-allocate Z_final and copy slices instead of cat
        #
        # The old approach used torch.cat() which causes massive memory spikes:
        #   - cat([Z_chunk, D_chunk]): peak = 13.3 + 13.3 + 26.6 = ~53 GB
        #   - cat([Z_chunk, D_self]): peak = 26.6 + 6.7 + 26.8 = ~60 GB
        #
        # New approach: pre-allocate final tensor, copy slices in-place:
        #   - Peak = Z_final (26.8 GB) + D_chunk (13.3 GB) = ~40 GB
        # This saves ~20 GB of peak memory!
        # =======================================================================

        # Calculate final feature dimension
        final_dim = base_z_dim  # c_z from TokenInitializer
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                final_dim += self.c_z  # sinusoidal uses DiffusionTokenEncoder.c_z
            else:
                final_dim += self.n_bins_distogram
        if self.use_self:
            final_dim += self.n_bins_distogram

        # Pre-allocate the final Z tensor (avoids cat operations)
        Z_chunk = torch.empty(B, I_par, I, final_dim, device=device, dtype=dtype)
        debug_memory("DIFF_ENCODER", f"after_preallocate_Zfinal_shape{list(Z_chunk.shape)}")

        # Copy Z_init_chunk into the first slice (expand for batch)
        # NOTE: Z_init_chunk is [I_par, I, c_z], expand to [B, I_par, I, c_z]
        offset = 0

        # DIAGNOSTIC: Verify device consistency before copy
        if Z_init_chunk.device != device:
            print(f"[RANK{rank}] ERROR: Z_init_chunk on {Z_init_chunk.device}, Z_chunk on {device}", flush=True)
            Z_init_chunk = Z_init_chunk.to(device)

        # Perform the copy with explicit synchronization for multi-GPU stability
        try:
            Z_chunk[:, :, :, :base_z_dim] = Z_init_chunk.unsqueeze(0).expand(B, -1, -1, -1)
            torch.cuda.synchronize(device)  # Ensure copy completes before proceeding
            debug_memory("DIFF_ENCODER", "after_Z_init_copy")
        except RuntimeError as e:
            print(f"[RANK{rank}] COPY_ERROR: {e}", flush=True)
            print(f"[RANK{rank}] Z_chunk.device={Z_chunk.device}, Z_init_chunk.device={Z_init_chunk.device}", flush=True)
            raise

        offset += base_z_dim
        del Z_init_chunk  # Free Z_init_chunk - no longer needed after copy
        torch.cuda.empty_cache()  # Clean up before computing distogram

        # Step 3: Compute distogram and copy into pre-allocated slice
        if self.use_distogram:
            # DIAGNOSTIC: Log before boolean indexing (common source of cross-device errors)
            debug_memory("DIFF_ENCODER", "before_distogram")

            # Verify f["is_ca"] device before boolean indexing
            is_ca = f["is_ca"]
            if is_ca.device != device:
                print(f"[RANK{rank}] DEVICE_FIX: is_ca on {is_ca.device}, moving to {device}", flush=True)
                is_ca = is_ca.to(device)
                f["is_ca"] = is_ca

            try:
                R_ca = R_L[..., is_ca, :]                 # [B, I, 3]
                torch.cuda.synchronize(device)  # Ensure indexing completes
                debug_memory("DIFF_ENCODER", "after_R_ca_indexing")
            except RuntimeError as e:
                print(f"[RANK{rank}] R_ca_INDEXING_ERROR: {e}", flush=True)
                print(f"[RANK{rank}] R_L.device={R_L.device}, is_ca.device={is_ca.device}", flush=True)
                raise

            if self.use_sinusoidal_distogram_embedder:
                R_ca_query = R_ca[:, gpu_start:gpu_end, :] # [B, I_par, 3]

                # Mask: [I_par, I, 1] - query chunk vs all keys
                # Verify motif mask device
                motif_full = f["is_motif_atom_with_fixed_coord"]
                if motif_full.device != device:
                    print(f"[RANK{rank}] DEVICE_FIX: motif on {motif_full.device}, moving to {device}", flush=True)
                    motif_full = motif_full.to(device)
                    f["is_motif_atom_with_fixed_coord"] = motif_full

                motif_mask = motif_full[is_ca]  # [I]
                motif_query = motif_mask[gpu_start:gpu_end]                    # [I_par]
                mask_chunk = (motif_query[:, None] != motif_mask[None, :]).unsqueeze(-1)

                debug_memory("DIFF_ENCODER", "after_mask_computation")

                # CRITICAL: Verify dist_embedder is on the correct device (multi-GPU fix)
                # In multi-node setups, self.to(device) may not fully move all parameters
                # due to Fabric wrapping or other issues. Explicitly move and verify.
                if hasattr(self, 'dist_embedder'):
                    # Check the output_proj Linear layer's weight device
                    embedder_device = self.dist_embedder.output_proj.weight.device
                    if embedder_device != device:
                        print(f"[RANK{rank}] WARNING: dist_embedder on {embedder_device}, moving to {device}", flush=True)
                        self.dist_embedder = self.dist_embedder.to(device)
                        # Synchronize to ensure move is complete before use
                        torch.cuda.synchronize(device)

                # Sinusoidal distance embedding for this chunk
                try:
                    D_chunk = self.dist_embedder.forward_chunk(
                        R_ca_query, R_ca, ~mask_chunk
                    )                                          # [B, I_par, I, c_z]
                    torch.cuda.synchronize(device)  # Ensure forward completes
                    debug_memory("DIFF_ENCODER", "after_dist_embedder")
                except RuntimeError as e:
                    print(f"[RANK{rank}] DIST_EMBEDDER_ERROR: {e}", flush=True)
                    print(f"[RANK{rank}] R_ca_query.device={R_ca_query.device}, R_ca.device={R_ca.device}, mask_chunk.device={mask_chunk.device}", flush=True)
                    raise

                distogram_dim = self.c_z
            else:
                # Bucketized distogram - compute for query chunk vs all atoms
                # DIAGNOSTIC: Log before bucketized distogram
                debug_memory("DIFF_ENCODER", f"before_bucketized_distogram_Rca{list(R_ca.shape)}")
                if R_ca.device != device:
                    print(f"[RANK{rank}] WARNING: R_ca on {R_ca.device}, expected {device}", flush=True)
                    R_ca = R_ca.to(device)
                    torch.cuda.synchronize(device)

                try:
                    D_chunk = bucketize_scaled_distogram_chunked(
                        R_ca, gpu_start, gpu_end,
                        min_dist=1, max_dist=30, sigma_data=16,  # default sigma_data
                        n_bins=self.n_bins_distogram
                    )                                          # [B, I_par, I, n_bins]
                    torch.cuda.synchronize(device)  # Ensure distogram computation completes
                    debug_memory("DIFF_ENCODER", f"after_bucketized_distogram_Dchunk{list(D_chunk.shape)}")
                except RuntimeError as e:
                    print(f"[RANK{rank}] BUCKETIZED_DISTOGRAM_ERROR: {e}", flush=True)
                    print(f"[RANK{rank}] R_ca.device={R_ca.device}, shape={R_ca.shape}", flush=True)
                    raise

                distogram_dim = self.n_bins_distogram

            # DIAGNOSTIC: Log before copy into Z_chunk
            debug_memory("DIFF_ENCODER", "before_D_chunk_copy")
            if D_chunk.device != device:
                print(f"[RANK{rank}] WARNING: D_chunk on {D_chunk.device}, Z_chunk on {device}", flush=True)
                D_chunk = D_chunk.to(device)
                torch.cuda.synchronize(device)

            # Copy into pre-allocated slice (avoids cat)
            try:
                Z_chunk[:, :, :, offset:offset+distogram_dim] = D_chunk
                torch.cuda.synchronize(device)  # Ensure copy completes
                debug_memory("DIFF_ENCODER", "after_D_chunk_copy")
            except RuntimeError as e:
                print(f"[RANK{rank}] D_CHUNK_COPY_ERROR: {e}", flush=True)
                print(f"[RANK{rank}] Z_chunk.device={Z_chunk.device}, D_chunk.device={D_chunk.device}", flush=True)
                raise

            offset += distogram_dim
            del D_chunk  # Free distogram tensor immediately
            torch.cuda.empty_cache()

        # Step 4: Compute self-conditioning and copy into pre-allocated slice
        if self.use_self:
            debug_memory("DIFF_ENCODER", "before_self_cond")
            D_II_self = kwargs.get("D_II_self")
            if D_II_self is not None:
                # CRITICAL: Ensure D_II_self is on the correct device (multi-node fix)
                if D_II_self.device != device:
                    print(f"[RANK{rank}] WARNING: D_II_self on {D_II_self.device}, moving to {device}", flush=True)
                    D_II_self = D_II_self.to(device)
                    torch.cuda.synchronize(device)
                # =======================================================================
                # MEMORY OPTIMIZATION: D_II_self may already be chunked [B, I_par, I, n_bins]
                # In multi-GPU mode, RFD3_diffusion_module now returns the chunk directly
                # instead of gathering to full [B, I, I, n_bins] to save 12-21 GB per GPU.
                # =======================================================================
                if D_II_self.shape[1] == I_par:
                    # Already chunked [B, I_par, I, n_bins] - use directly
                    D_self_chunk = D_II_self
                else:
                    # Full tensor [B, I, I, n_bins] (standard mode) - slice it
                    D_self_chunk = D_II_self[:, gpu_start:gpu_end, :]  # [B, I_par, I, n_bins]
                debug_memory("DIFF_ENCODER", f"D_II_self_provided_shape{list(D_self_chunk.shape)}")
            else:
                debug_memory("DIFF_ENCODER", f"before_zeros_B{B}_Ipar{I_par}_I{I}_nbins{self.n_bins_distogram}")
                try:
                    D_self_chunk = torch.zeros(
                        B, I_par, I, self.n_bins_distogram,
                        device=device, dtype=dtype
                    )                                          # [B, I_par, I, n_bins]
                    torch.cuda.synchronize(device)
                    debug_memory("DIFF_ENCODER", f"after_zeros_D_self_chunk{list(D_self_chunk.shape)}")
                except RuntimeError as e:
                    print(f"[RANK{rank}] ZEROS_ERROR: {e}", flush=True)
                    print(f"[RANK{rank}] device={device}, dtype={dtype}", flush=True)
                    raise

            # Copy into pre-allocated slice (avoids cat)
            debug_memory("DIFF_ENCODER", "before_D_self_copy")
            try:
                Z_chunk[:, :, :, offset:offset+self.n_bins_distogram] = D_self_chunk
                torch.cuda.synchronize(device)
                debug_memory("DIFF_ENCODER", "after_D_self_copy")
            except RuntimeError as e:
                print(f"[RANK{rank}] D_SELF_COPY_ERROR: {e}", flush=True)
                print(f"[RANK{rank}] Z_chunk.device={Z_chunk.device}, D_self_chunk.device={D_self_chunk.device}", flush=True)
                raise

            offset += self.n_bins_distogram
            del D_self_chunk  # Free self-conditioning tensor immediately
            torch.cuda.empty_cache()

        debug_memory("DIFF_ENCODER", f"after_fill_Zchunk_offset{offset}")

        # Verify dimensions before process_z (diagnostic)
        actual_dim = Z_chunk.shape[-1]

        # CRITICAL: process_z expects cat_c_z = self.c_z + distogram + self_cond
        # If TokenInitializer.c_z != DiffusionTokenEncoder.c_z, dimensions won't match
        process_z_expected = self.c_z
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                process_z_expected += self.c_z
            else:
                process_z_expected += self.n_bins_distogram
        if self.use_self:
            process_z_expected += self.n_bins_distogram

        if actual_dim != process_z_expected:
            raise RuntimeError(
                f"DiffusionTokenEncoder.forward (parallel): dimension mismatch!\n"
                f"  process_z expects: {process_z_expected} (based on DiffusionTokenEncoder.c_z={self.c_z})\n"
                f"  Z_chunk actual: {actual_dim}\n"
                f"  Z_chunk shape: {Z_chunk.shape}\n"
                f"  TokenInitializer.c_z (Z_parallel): {base_z_dim}\n"
                f"  use_distogram={self.use_distogram}, use_sinusoidal={self.use_sinusoidal_distogram_embedder}, "
                f"use_self={self.use_self}, n_bins={self.n_bins_distogram}\n"
                f"  Parallel mode requires TokenInitializer.c_z == DiffusionTokenEncoder.c_z"
            )

        # Step 5: Process concatenated Z features
        # Match standard: Z_II = self.process_z(Z_II)
        # =======================================================================
        # MEMORY OPTIMIZATION: Use key-chunking for process_z to avoid 22+ GB tensor
        # Z_chunk is [B, I_par, I, c_in] where c_in = c_z + distogram + self_cond
        # For I=8100, c_in=258, full tensor = 22 GB which causes OOM.
        # Processing in key-chunks of 512: [B, I_par, 512, c_in] = ~1.4 GB
        # =======================================================================
        debug_memory("DIFF_ENCODER", f"before_process_z_Zshape{list(Z_chunk.shape)}")
        Z_chunk = process_z_chunked(Z_chunk, self.process_z)  # [B, I_par, I, c_z]
        debug_memory("DIFF_ENCODER", "after_process_z")

        # Match standard: Z_II = Z_II + self.transition_2[b](Z_II)
        # Use key-chunking to reduce peak memory from SwiGLU 4x expansion
        for b in range(2):
            Z_chunk = z_transition_chunked(Z_chunk, self.transition_2[b])  # [B, I_par, I, c_z]

        # Step 6: Pairformer with chunked attention
        # S_I attention: this GPU's query chunk [I_par] attends to all keys [I]
        # Output is [I_par, c_s] per GPU, then all_gathered to [I, c_s]

        # Handle batch dimension in S_I
        if has_batch_S:
            # S_I is [B, I, c_s] - slice along dim 1, squeeze batch for attention
            S_I_chunk = S_I[:, gpu_start:gpu_end, :].squeeze(0)  # [I_par, c_s]
            S_I_unbatched = S_I.squeeze(0)                        # [I, c_s]
        else:
            # S_I is [I, c_s] - slice directly
            S_I_chunk = S_I[gpu_start:gpu_end]                    # [I_par, c_s]
            S_I_unbatched = S_I                                   # [I, c_s]

        # =======================================================================
        # MEMORY OPTIMIZATION: Clean memory before pairformer loop
        # The pairformer's ln_0(Z_chunk) creates a full copy of Z_chunk (~14 GB).
        # Ensure maximum free memory before this operation.
        # =======================================================================
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        debug_memory("DIFF_ENCODER", "before_pairformer_after_cleanup")

        for block in self.pairformer_stack:
            # Z transition (key-chunked to reduce memory)
            Z_chunk = z_transition_chunked(Z_chunk, block.z_transition)  # [B, I_par, I, c_z]

            # Attention: queries [I_par] attend to all keys [I] using Z_chunk as bias
            if hasattr(block, 'attention_pair_bias'):
                # Chunked attention: S_I_chunk queries, S_I keys, Z_chunk bias
                # forward_chunked expects 2D inputs [I_par, c_s] and [I, c_s]
                from rfd3.model.parallel.layers.pairformer_layers import attention_pair_bias_forward_chunked
                S_I_chunk = S_I_chunk + attention_pair_bias_forward_chunked(
                    block.attention_pair_bias,              # AttentionPairBiasPairformerDeepspeed instance
                    A_I_query=S_I_chunk,                   # [I_par, c_s]
                    A_I_key=S_I_unbatched,                 # [I, c_s]
                    Z_chunk=Z_chunk[0],                    # [I_par, I, c_z]
                    Beta_II=torch.tensor([0.0], device=device),
                )                                          # [I_par, c_s]
                S_I_chunk = S_I_chunk + block.s_transition(S_I_chunk)

            # CRITICAL: All-gather S_I_chunk to update keys for next block!
            # In standard mode, S_I is updated in each iteration and used as both
            # queries and keys in the next block. We must do the same here.
            # Pass total_size=I to handle uneven chunk sizes across GPUs
            if world_size > 1:
                S_I_unbatched = all_gather_concat(S_I_chunk, dim=0, total_size=I)  # [I, c_s]
            else:
                S_I_unbatched = S_I_chunk

            # Clean up after each pairformer block to release temp tensors
            torch.cuda.empty_cache()

        # Step 7: All-gather S_I chunks from all GPUs (final)
        # Each GPU has [I_par, c_s], gather to get full [I, c_s]
        # Pass total_size=I to handle uneven chunk sizes across GPUs
        if world_size > 1:
            S_I = all_gather_concat(S_I_chunk, dim=0, total_size=I)     # [I, c_s]
        else:
            # Single GPU: just use the chunk (which is full I in this case)
            S_I = S_I_chunk

        # Return this GPU's Z chunk (NOT full I×I!)
        # Downstream modules must also support chunked Z
        return S_I, Z_chunk[0] if B == 1 else Z_chunk      # S_I: [I, c_s], Z: [I_par, I, c_z]
