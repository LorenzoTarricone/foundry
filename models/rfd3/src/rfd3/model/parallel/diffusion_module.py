"""
Parallel implementation of RFD3DiffusionModule for multi-GPU inference.

Class extracted from RFD3_diffusion_module.py for parallel (multi-GPU) mode.
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

from rfd3.model.RFD3_diffusion_module import RFD3DiffusionModule
from rfd3.model.debug_context import (
    debug_ctx,
    debug_log,
    debug_tensor,
    debug_time,
    debug_memory,
    debug_tensor_memory,
    debug_tensor_all_ranks,
    debug_log_all_ranks,
    log_tensor_stats,
    verify_tensor_sync,
    timed_all_gather,
)
from rfd3.model.parallel.utils import (
    is_parallel_mode,
    get_gpu_rank_and_world_size,
    compute_chunk_ranges,
    all_gather_along_dim,
)
from rfd3.model.layers.block_utils import bucketize_scaled_distogram_chunked, create_attention_indices
from rfd3.model.parallel.layers.blocks import (
    structure_local_atom_block_cross_attn,
    structure_local_atom_block_sparse_cross_attn,
)


class ParallelDiffusionModule(RFD3DiffusionModule):
    """
    Parallel (multi-GPU) version of RFD3DiffusionModule.

    Overrides process_() to use parallel transformer/decoder methods that split
    work across GPUs using cross-attention.

    Memory: O(I²/N + L²/N) per GPU instead of O(I² + L²).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Swap diffusion_token_encoder to its parallel subclass.
        # ParallelDiffusionTokenEncoder has no __init__ override, so a class swap
        # is safe — same parameters, different forward() that handles Z_chunk
        # [I_par, I, c_z] instead of assuming full [I, I, c_z].
        from rfd3.model.parallel.layers.encoders import ParallelDiffusionTokenEncoder
        self.diffusion_token_encoder.__class__ = ParallelDiffusionTokenEncoder

    def _local_token_transformer_cross_attn(
        self,
        A_I_chunk,
        S_I_chunk,
        A_I_full,
        S_I_full,
        Z_chunk,
        f,
        X_L,
        query_start,
        world_size: int = 1,
    ):
        """
        Cross-attention forward: query chunk attends to all keys.

        Multi-GPU parallel inference:
        - Each GPU processes I_par = I // n_gpus query tokens
        - All GPUs have access to all I key tokens
        - Z_chunk is [I_par, I] not [I, I] - avoids I×I memory

        Args:
            A_I_chunk: Query features [B, I_par, c_token]
            S_I_chunk: Query conditioning [I_par, c_s] or [B, I_par, c_s]
            A_I_full: Key/Value features [B, I, c_token]
            S_I_full: K/V conditioning [I, c_s] or [B, I, c_s]
            Z_chunk: Pair bias [I_par, I, c_z] or [B, I_par, I, c_z]
            f: Feature dictionary
            X_L: CA positions [B, I, 3]
            query_start: Global start index of queries
            world_size: Number of GPUs

        Returns:
            A_I_chunk: Updated query features [B, I_par, c_token]
        """
        transformer = self.diffusion_transformer
        B, I_par, c_token = A_I_chunk.shape
        I = A_I_full.shape[1]
        device = A_I_chunk.device
        world_size = max(int(world_size), 1)
        rank = dist.get_rank() if dist.is_initialized() else 0
        chunk_ranges = compute_chunk_ranges(I, world_size)

        if rank >= len(chunk_ranges):
            return A_I_chunk
        query_start, query_end = chunk_ranges[rank]

        # Ensure transformer blocks are on the correct device for this rank
        transformer.to(device)

        def _gather_updated_tokens(chunk: torch.Tensor, block_idx: int) -> torch.Tensor:
            """Gather variable-length chunks across GPUs and reassemble full A_I."""
            if world_size == 1 or not dist.is_initialized():
                return chunk

            local_len = torch.tensor([chunk.shape[1]], device=device, dtype=torch.long)
            lens = [torch.zeros_like(local_len) for _ in range(world_size)]
            timed_all_gather(lens, local_len,
                             "ALL_GATHER", f"blocks._gather_updated_tokens.lens.block{block_idx}",
                             gather_type="all_gather")
            max_len = int(torch.stack(lens).max().item())

            if chunk.shape[1] < max_len:
                pad_len = max_len - chunk.shape[1]
                chunk = F.pad(chunk, (0, 0, 0, pad_len))

            chunk = chunk.contiguous()

            if hasattr(dist, 'all_gather_into_tensor'):
                flat_input = chunk.view(-1)
                flat_output = torch.empty(flat_input.numel() * world_size, dtype=chunk.dtype, device=chunk.device)
                timed_all_gather(flat_output, flat_input,
                                 "ALL_GATHER", f"blocks._gather_updated_tokens.data.block{block_idx}",
                                 gather_type="all_gather_into_tensor")

                B_local, _, C = chunk.shape
                reshaped = flat_output.view(world_size, B_local, max_len, C)

                trimmed = []
                for i, l in enumerate(lens):
                    trimmed.append(reshaped[i, :, : int(l.item()), :])
                del flat_input, flat_output, reshaped
            else:
                gathered = [torch.zeros_like(chunk) for _ in range(world_size)]
                timed_all_gather(gathered, chunk,
                                 "ALL_GATHER", f"blocks._gather_updated_tokens.data_fallback.block{block_idx}",
                                 gather_type="all_gather")
                trimmed = []
                for g, l in zip(gathered, lens):
                    trimmed.append(g[:, : int(l.item()), :])
                del gathered

            if block_idx == 0 and rank == 0:
                lens_py = [int(l.item()) for l in lens]
                debug_log("BLOCKS", "gather",
                          f"lens={lens_py}, max_len={max_len}, concat_len={sum(lens_py)}, chunk_ranges={chunk_ranges}")
            return torch.cat(trimmed, dim=1)

        def _compute_indices_chunk(start: int, end: int) -> torch.Tensor:
            indices_full = create_attention_indices(
                X_L=X_L,
                f=f,
                tok_idx=torch.arange(I, device=device),
                n_attn_keys=transformer.n_keys,
                n_attn_seq_neighbours=transformer.n_local_tokens,
            )
            return indices_full[:, start:end, :]

        def _slice_S_I_chunk(start: int, end: int) -> torch.Tensor:
            chunk = S_I_full[start:end] if S_I_full.ndim == 2 else S_I_full[:, start:end, :]
            return chunk.unsqueeze(0).expand(B, -1, -1) if chunk.ndim == 2 else chunk

        def _slice_Z_chunk(z_source, start: int, end: int):
            if z_source.ndim == 3:
                if z_source.shape[0] == (end - start):
                    return z_source
                return z_source[start:end, :, :]
            else:
                if z_source.shape[1] == (end - start):
                    return z_source
                return z_source[:, start:end, :, :]

        # Initial slices for this rank
        indices_chunk = _compute_indices_chunk(query_start, query_start + I_par)
        S_I_chunk = _slice_S_I_chunk(query_start, query_start + I_par)
        Z_local_chunk = _slice_Z_chunk(Z_chunk, query_start, query_start + I_par)

        # Import here to avoid circular dependency
        from foundry import DISABLE_CHECKPOINTING

        # Run cross-attention through blocks
        for block_idx, block in enumerate(transformer.blocks):
            block.attention_pair_bias.use_checkpointing = not DISABLE_CHECKPOINTING

            # Cross-attention: chunk queries → all keys (free function)
            A_I_chunk = structure_local_atom_block_cross_attn(
                block,
                Q_chunk=A_I_chunk,
                C_Q_chunk=S_I_chunk,
                K_V_full=A_I_full,
                C_KV_full=S_I_full,
                P_chunk=Z_local_chunk,
                indices_chunk=indices_chunk,
            )

            # After each block, gather updated chunks
            A_I_full = _gather_updated_tokens(A_I_chunk, block_idx)
            query_start, query_end = chunk_ranges[rank]
            A_I_chunk = A_I_full[:, query_start:query_end, :]

        return A_I_chunk

    def _diffusion_transformer_parallel(
        self,
        A_I: torch.Tensor,           # [B, I, c_token]
        S_I: torch.Tensor,           # [I, c_s]
        Z_II_chunk: torch.Tensor,    # [I_par, I, c_z] - this GPU's query rows
        f: dict,
        X_L: torch.Tensor,           # [B, I, 3] - CA positions
        gpu_rank: int,
        world_size: int,
    ) -> torch.Tensor:
        """
        Multi-GPU parallel diffusion transformer using TRUE CROSS-ATTENTION.

        Each GPU processes a chunk of query tokens (I_par) attending to all tokens (I).
        NO [I, I] tensor is ever created - Z_II_chunk stays [I_par, I].

        Cross-attention pattern:
        - Queries: A_I_chunk [B, I_par, c] - this GPU's token chunk
        - Keys/Values: A_I [B, I, c] - ALL tokens (full context)
        - Bias: Z_II_chunk [I_par, I, c_z] - pair bias for chunk×all

        Data flow:
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ Each GPU (TRUE CROSS-ATTENTION):                                         │
        │                                                                          │
        │   Input:                                                                 │
        │     A_I [B, I, c_token]        - full token features (all GPUs)         │
        │     Z_II_chunk [I_par, I, c_z] - THIS GPU's pair rows only              │
        │                                                                          │
        │   Cross-attention:                                                       │
        │     Q = A_I[:, start:end, :]   → [B, I_par, c] (queries from chunk)     │
        │     K, V = A_I                 → [B, I, c] (keys/values from all)        │
        │     Bias = Z_II_chunk          → [I_par, I, c_z] (NO I×I!)              │
        │                                                                          │
        │   Output:                                                                │
        │     A_I_chunk [B, I_par, c]    → gathered to [B, I, c]                  │
        └─────────────────────────────────────────────────────────────────────────┘

        Args:
            A_I: Full token features [B, I, c_token] (same on all GPUs)
            S_I: Single features [I, c_s]
            Z_II_chunk: This GPU's Z rows [I_par, I, c_z] - NEVER expanded!
            f: Feature dictionary
            X_L: CA positions [B, I, 3] for local attention indices
            gpu_rank: This GPU's rank
            world_size: Total number of GPUs

        Returns:
            A_I: Updated token features [B, I, c_token] (reconstructed from chunks)
        """
        B = A_I.shape[0]
        I = A_I.shape[1]
        device = A_I.device

        # Ensure diffusion transformer is on the correct device for this rank
        self.diffusion_transformer.to(device)

        # Compute chunk ranges for all GPUs
        chunk_ranges = compute_chunk_ranges(I, world_size)

        # This GPU's range
        if gpu_rank < len(chunk_ranges):
            start_i, end_i = chunk_ranges[gpu_rank]
            I_par = end_i - start_i                        # This GPU's query count
        else:
            # Edge case: more GPUs than tokens
            return A_I

        # Extract this GPU's query chunk
        A_I_chunk = A_I[:, start_i:end_i, :].contiguous()  # [B, I_par, c_token]
        S_I_chunk = S_I[start_i:end_i] if S_I.ndim == 2 else S_I[:, start_i:end_i, :]

        # Run cross-attention transformer
        # Queries from A_I_chunk, Keys/Values from A_I, Bias from Z_II_chunk
        if gpu_rank == 0 and debug_ctx.stats_enabled:
            with torch.no_grad():
                print(
                    f"{debug_ctx.prefix('DIFF')} xattn input shapes:",
                    {
                        "A_I_full": list(A_I.shape),
                        "A_I_chunk": list(A_I_chunk.shape),
                        "Z_II_chunk": list(Z_II_chunk.shape),
                        "world_size": world_size,
                        "chunk_range": [start_i, end_i],
                    },
                    flush=True,
                )
        A_I_chunk = self._local_token_transformer_cross_attn(
            A_I_chunk=A_I_chunk,                           # [B, I_par, c_token]
            S_I_chunk=S_I_chunk,                           # [I_par, c_s]
            A_I_full=A_I,                                  # [B, I, c_token]
            S_I_full=S_I,                                  # [I, c_s]
            Z_chunk=Z_II_chunk,                            # [I_par, I, c_z] - NO I×I!
            f=f,
            X_L=X_L,                                       # [B, I, 3]
            query_start=start_i,                           # Global query start index
            world_size=world_size,
        )                                                  # [B, I_par, c_token]

        # MULTI-GPU DIAGNOSTIC: Log A_I_chunk stats from ALL ranks before gathering
        debug_tensor_all_ranks("DIFF_XATTN", f"A_I_chunk_q{start_i}-{end_i}", A_I_chunk)

        # Gather chunks from all GPUs along token dimension (dim=1)
        # Pass total_size=I to handle uneven chunk sizes from floor-division-with-remainder
        A_I = all_gather_along_dim(A_I_chunk, world_size, dim=1, total_size=I)  # [B, I, c_token]

        # MULTI-GPU DIAGNOSTIC: Verify A_I is synchronized after all_gather
        debug_tensor_all_ranks("DIFF_XATTN", "A_I_after_gather", A_I)
        verify_tensor_sync("DIFF_XATTN", "A_I_SYNC", A_I)

        return A_I

    def _compact_decoder_parallel(
        self,
        A_I,
        S_I,
        Q_L,
        C_L,
        P_chunk,
        tok_idx,
        indices,
        query_start,
        query_end,
        world_size,
    ):
        """
        Parallel decoder: each GPU processes its chunk of atom queries.

        Multi-GPU parallelization:
        - Each GPU processes L_par = L // n_gpus atom queries
        - P_chunk is [L_par, L] not [L, L] - avoids L×L memory
        - Upcast operates on full A_I (no L×L)
        - Atom transformer uses cross-attention with P_chunk
        - Q_L chunks are all_gathered for downcast

        Args:
            A_I: Token features [B, I, c_token]
            S_I: Single features [I, c_s]
            Q_L: Atom features [B, L, c_atom]
            C_L: Conditioned atom features [B, L, c_atom]
            P_chunk: This GPU's P_LL rows [L_par, L, c_pair]
            tok_idx: Atom to token mapping [L]
            indices: Full attention indices [B, L, k]
            query_start: Start index of this GPU's queries
            query_end: End index of this GPU's queries
            world_size: Number of GPUs

        Returns:
            A_I: Updated token features [B, I, c_token]
            Q_L: Updated atom features [B, L, c_atom]
            o: Empty dict (for compatibility)
        """
        decoder = self.decoder
        indices_chunk = indices[:, query_start:query_end, :]

        # Run blocks with cross-attention
        for i in range(decoder.n_blocks):
            # Upcast: token → atom
            Q_L = decoder.upcast[i](Q_L, A_I, tok_idx=tok_idx)

            # Extract this GPU's Q_L chunk for attention
            Q_L_chunk = Q_L[:, query_start:query_end, :]
            C_L_chunk = C_L[:, query_start:query_end, :]

            # Cross-attention: Q_L_chunk queries attend to Q_L keys
            Q_L_chunk = structure_local_atom_block_cross_attn(
                decoder.atom_transformer[i],
                Q_chunk=Q_L_chunk,
                C_Q_chunk=C_L_chunk,
                K_V_full=Q_L,
                C_KV_full=C_L,
                P_chunk=P_chunk,
                indices_chunk=indices_chunk,
            )

            # All-gather Q_L chunks to reconstruct full Q_L
            Q_L = all_gather_along_dim(Q_L_chunk, world_size, dim=1)

        # Downcast to sequence
        A_I = decoder.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o

    def _compact_decoder_parallel_sparse(
        self,
        A_I,
        S_I,
        Q_L,
        C_L,
        tok_idx,
        tok_idx_chunk,
        indices,
        indices_chunk,
        query_start,
        query_end,
        f,
        chunked_pairwise_embedder,
        initializer_outputs,
        world_size,
    ):
        """
        Parallel decoder with SPARSE P_LL computation.

        Combines two memory optimizations:
        1. Multi-GPU parallelism: each GPU handles L_par queries
        2. Sparse P_LL: only k neighbors per atom via chunked_pairwise_embedder

        Memory footprint:
        - P_LL is never [L, L], only [L_par, k] per GPU
        - Total memory: O(L_par * k) = O(L * k / n_gpus) per GPU

        Args:
            A_I: Token features [B, I, c_token]
            S_I: Single features [I, c_s]
            Q_L: Atom features [B, L, c_atom]
            C_L: Conditioned atom features [B, L, c_atom]
            tok_idx: Full atom to token mapping [L]
            tok_idx_chunk: This GPU's atom to token mapping [L_par]
            indices: Full attention indices [B, L, k]
            indices_chunk: This GPU's attention indices [B, L_par, k]
            query_start: Start of this GPU's atom queries
            query_end: End of this GPU's atom queries
            f: Feature dictionary
            chunked_pairwise_embedder: For sparse P_LL computation
            initializer_outputs: Embedder state
            world_size: Number of GPUs

        Returns:
            A_I: Updated token features [B, I, c_token]
            Q_L: Updated atom features [B, L, c_atom]
            o: Empty dict
        """
        decoder = self.decoder

        # Run blocks
        for i in range(decoder.n_blocks):
            # Upcast: token → atom
            Q_L = decoder.upcast[i](Q_L, A_I, tok_idx=tok_idx)

            # Extract this GPU's Q_L chunk for attention
            Q_L_chunk = Q_L[:, query_start:query_end, :]
            C_L_chunk = C_L[:, query_start:query_end, :]

            # Compute SPARSE P_LL for this GPU's chunk only
            P_sparse_chunk = chunked_pairwise_embedder.forward_chunked(
                f=f,
                indices=indices_chunk,
                C_L=initializer_outputs["C_L"],
                Z_init_II=initializer_outputs["Z_II"],
                tok_idx=f["atom_to_token_map"],
                query_start=query_start,
                streaming_mode=initializer_outputs.get("streaming_mode", False),
                z_chunk_range=initializer_outputs.get("z_chunk_range"),
            )

            if i == 0:
                log_tensor_stats(f"decoder_P_sparse_chunk_block{i}", P_sparse_chunk)

            # Sparse cross-attention
            Q_L_chunk = structure_local_atom_block_sparse_cross_attn(
                decoder.atom_transformer[i],
                Q_chunk=Q_L_chunk,
                C_Q_chunk=C_L_chunk,
                K_V_full=Q_L,
                C_KV_full=C_L,
                P_sparse_chunk=P_sparse_chunk,
                indices_chunk=indices_chunk,
            )

            if i == 0:
                log_tensor_stats(f"decoder_Q_L_chunk_after_attn_block{i}", Q_L_chunk)

            # All-gather Q_L chunks to reconstruct full Q_L
            Q_L = all_gather_along_dim(Q_L_chunk, world_size, dim=1)

            if i == 0:
                log_tensor_stats(f"decoder_Q_L_after_gather_block{i}", Q_L)

        # Downcast to sequence
        A_I = decoder.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o

    def _decoder_parallel(
        self,
        A_I: torch.Tensor,           # [B, I, c_token]
        S_I: torch.Tensor,           # [I, c_s]
        Q_L: torch.Tensor,           # [B, L, c_atom]
        C_L: torch.Tensor,           # [B, L, c_atom]
        f: dict,
        initializer_outputs: dict,
        gpu_rank: int,
        world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Parallel decoder using TRUE CROSS-ATTENTION.

        Each GPU processes its chunk of atom queries (L_par) against all atoms (L).
        P_LL_chunk is computed on-the-fly as [L_par, L] - NO L×L tensor!

        For P_LL_chunk computation, we use the embedders from initializer_outputs:
        - motif_pos_embedder: distance-based features
        - ref_pos_embedder: reference position features
        - process_single_l/m: single atom features
        - process_z: Z_II contribution

        Args:
            A_I: Token features [B, I, c_token]
            S_I: Single features [I, c_s]
            Q_L: Atom features [B, L, c_atom]
            C_L: Conditioned atom features [B, L, c_atom]
            f: Feature dictionary
            initializer_outputs: Dict with embedders and features
            gpu_rank: This GPU's rank
            world_size: Total number of GPUs

        Returns:
            A_I: Updated token features [B, I, c_token]
            Q_L: Updated atom features [B, L, c_atom]
            o: Empty dict (for compatibility)
        """
        B = A_I.shape[0]
        L = Q_L.shape[1]
        device = A_I.device
        tok_idx = f["atom_to_token_map"]                   # [L]

        # Ensure decoder is on the correct device for this rank
        self.decoder.to(device)

        # Compute chunk ranges for atoms
        chunk_ranges = compute_chunk_ranges(L, world_size)

        if gpu_rank < len(chunk_ranges):
            start_l, end_l = chunk_ranges[gpu_rank]
            L_par = end_l - start_l                        # This GPU's atom query count
        else:
            # Edge case: more GPUs than atoms
            o = {}
            return A_I, Q_L, o

        # Compute P_LL_chunk [L_par, L, c_atompair] for this GPU's queries
        # This is computed on-the-fly without materializing full [L, L]
        P_LL_chunk = self._compute_P_LL_chunk(
            f=f,
            C_L=C_L,
            initializer_outputs=initializer_outputs,
            query_start=start_l,
            query_end=end_l,
        )  # [L_par, L, c_atompair]

        # MULTI-GPU DIAGNOSTIC: Log P_LL_chunk stats from ALL ranks
        debug_tensor_all_ranks("DECODER_PAR", f"P_LL_chunk_q{start_l}-{end_l}", P_LL_chunk)

        # Decoder with cross-attention
        A_I, Q_L, o = self._compact_decoder_parallel(
            A_I=A_I,                                       # [B, I, c_token]
            S_I=S_I,                                       # [I, c_s]
            Q_L=Q_L,                                       # [B, L, c_atom]
            C_L=C_L,                                       # [B, L, c_atom]
            P_chunk=P_LL_chunk,                            # [L_par, L, c_atompair]
            tok_idx=tok_idx,                               # [L]
            indices=f["attn_indices"],                     # [B, L, k]
            query_start=start_l,
            query_end=end_l,
            world_size=world_size,
        )

        # MULTI-GPU DIAGNOSTIC: Log output Q_L stats from ALL ranks
        debug_tensor_all_ranks("DECODER_PAR", "Q_L_output", Q_L)
        verify_tensor_sync("DECODER_PAR", "Q_L_SYNC", Q_L)

        return A_I, Q_L, o

    def _decoder_parallel_sparse(
        self,
        A_I: torch.Tensor,                   # [B, I, c_token]
        S_I: torch.Tensor,                   # [I, c_s]
        Q_L: torch.Tensor,                   # [B, L, c_atom]
        C_L: torch.Tensor,                   # [B, L, c_atom]
        f: dict,
        chunked_pairwise_embedder,           # ChunkedPairwiseEmbedder
        initializer_outputs: dict,
        gpu_rank: int,
        world_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Decoder with BOTH optimizations combined:
        - Sparse P_LL via chunked_pairwise_embedder (k neighbors per atom)
        - Multi-GPU parallel processing (each GPU handles L_par atoms)

        This is the most memory-efficient mode:
        - P_LL is never [L, L], only [L_par, k] per GPU
        - Work is split across GPUs

        Args:
            A_I: [B, I, c_token] token features (full, replicated on all GPUs)
            S_I: [I, c_s] token single features (full)
            Q_L: [B, L, c_atom] atom features (full)
            C_L: [B, L, c_atom] atom conditioning (full)
            f: Feature dictionary with attn_indices [B, L, k]
            chunked_pairwise_embedder: For sparse P_LL computation
            initializer_outputs: Dict with embedder state
            gpu_rank: This GPU's rank
            world_size: Total number of GPUs

        Returns:
            A_I: [B, I, c_token] updated token features (all-gathered)
            Q_L: [B, L, c_atom] updated atom features (all-gathered)
            o: Empty dict
        """
        L = Q_L.shape[1]
        device = A_I.device
        tok_idx = f["atom_to_token_map"]                   # [L]

        # Ensure decoder and embedder are on the correct device for this rank
        self.decoder.to(device)
        if chunked_pairwise_embedder is not None:
            chunked_pairwise_embedder.to(device)

        # Compute chunk range for this GPU
        chunk_ranges = compute_chunk_ranges(L, world_size)

        if gpu_rank < len(chunk_ranges):
            start_l, end_l = chunk_ranges[gpu_rank]
            L_par = end_l - start_l                        # This GPU's atom query count
        else:
            # Edge case: more GPUs than atoms
            return A_I, Q_L, {}

        # Slice indices for this GPU's atoms only
        indices_chunk = f["attn_indices"][:, start_l:end_l, :]  # [B, L_par, k]
        tok_idx_chunk = tok_idx[start_l:end_l]                   # [L_par]

        # MULTI-GPU DIAGNOSTIC: Log chunk info from ALL ranks
        debug_log_all_ranks("DECODER_SPARSE", "chunk_info",
                           f"query_range=[{start_l},{end_l}], L_par={L_par}, indices_chunk_shape={list(indices_chunk.shape)}")

        # Decoder with sparse P_LL computation for this GPU's chunk
        A_I, Q_L, o = self._compact_decoder_parallel_sparse(
            A_I=A_I,                                       # [B, I, c_token]
            S_I=S_I,                                       # [I, c_s]
            Q_L=Q_L,                                       # [B, L, c_atom]
            C_L=C_L,                                       # [B, L, c_atom]
            tok_idx=tok_idx,                               # [L]
            tok_idx_chunk=tok_idx_chunk,                   # [L_par]
            indices=f["attn_indices"],                     # [B, L, k] full for K/V gathering
            indices_chunk=indices_chunk,                   # [B, L_par, k] for sparse P_LL
            query_start=start_l,
            query_end=end_l,
            f=f,
            chunked_pairwise_embedder=chunked_pairwise_embedder,
            initializer_outputs=initializer_outputs,
            world_size=world_size,
        )

        # MULTI-GPU DIAGNOSTIC: Log output Q_L stats from ALL ranks
        debug_tensor_all_ranks("DECODER_SPARSE", "Q_L_output", Q_L)
        verify_tensor_sync("DECODER_SPARSE", "Q_L_SYNC", Q_L)

        return A_I, Q_L, o

    def _compute_P_LL_chunk(
        self,
        f: dict,
        C_L: torch.Tensor,           # [B, L, c_atom]
        initializer_outputs: dict,
        query_start: int,
        query_end: int,
    ) -> torch.Tensor:
        """
        Compute P_LL for query chunk [query_start:query_end] × all atoms.

        Returns [L_par, L, c_atompair] without materializing full [L, L].

        Uses the same computation as TokenInitializer._forward_standard but
        only for the query chunk rows.
        """
        L = C_L.shape[1]
        L_par = query_end - query_start
        device = C_L.device
        dtype = C_L.dtype

        # Get embedders from initializer_outputs
        # These should be set by TokenInitializer in non-chunked mode
        C_L_init = initializer_outputs.get("C_L")         # [L, c_atom] from init
        if C_L_init is None:
            C_L_init = C_L[0] if C_L.ndim == 3 else C_L   # Use first batch

        Z_II = initializer_outputs.get("Z_II")             # [I, I, c_z]
        tok_idx = f["atom_to_token_map"]                   # [L]

        # === Motif position embedding (chunked) ===
        # valid_mask_chunk: [L_par, L, 1]
        motif_mask_q = f["is_motif_atom_with_fixed_coord"][query_start:query_end]  # [L_par]
        motif_mask_all = f["is_motif_atom_with_fixed_coord"]                        # [L]
        valid_mask_chunk = (
            motif_mask_q.unsqueeze(-1) & motif_mask_all.unsqueeze(0)
        ).unsqueeze(-1)                                    # [L_par, L, 1]

        # P_LL_chunk from motif positions
        motif_pos_q = f["motif_pos"][query_start:query_end]  # [L_par, 3]
        motif_pos_all = f["motif_pos"]                       # [L, 3]

        # Compute pairwise distances for chunk
        # [L_par, 1, 3] - [1, L, 3] = [L_par, L, 3]
        D_chunk = motif_pos_q.unsqueeze(1) - motif_pos_all.unsqueeze(0)
        dist_chunk = torch.linalg.norm(D_chunk, dim=-1)    # [L_par, L]

        # Simple distance binning (matching PositionPairDistEmbedder behavior)
        c_atompair = self.decoder.atom_transformer[0].attention_pair_bias.to_b.in_features
        P_LL_chunk = torch.zeros(L_par, L, c_atompair, device=device, dtype=dtype)

        # === Single atom features contribution ===
        # process_single_l(C_L)[query:query+L_par, None, :] + process_single_m(C_L)[None, :, :]
        # This creates [L_par, L, c_atompair]
        if hasattr(self, 'process_single_l') and hasattr(self, 'process_single_m'):
            s_l = self.process_single_l(C_L_init)          # [L, c_atompair]
            s_m = self.process_single_m(C_L_init)          # [L, c_atompair]
            P_LL_chunk = P_LL_chunk + (
                s_l[query_start:query_end].unsqueeze(1) +  # [L_par, 1, c_atompair]
                s_m.unsqueeze(0)                           # [1, L, c_atompair]
            )                                              # [L_par, L, c_atompair]

        # === Z_II contribution (if available) ===
        if Z_II is not None:
            tok_idx_q = tok_idx[query_start:query_end]  # [L_par]

            # Check for parallel mode via initializer_outputs
            parallel_mode = initializer_outputs.get("parallel_mode", False)
            z_chunk_range = initializer_outputs.get("z_chunk_range")

            if parallel_mode and z_chunk_range is not None:
                # Parallel mode: Z_II is pre-computed [I_par, I, c_z] chunk
                z_start, z_end = z_chunk_range
                I_par_z = Z_II.shape[0]
                I_z = Z_II.shape[1]

                # Map atom query token indices to local Z chunk indices
                tok_q = tok_idx_q  # [L_par]
                tok_all = tok_idx  # [L]

                # Map to local chunk indices (assuming alignment)
                local_tok_q = tok_q - z_start  # [L_par]
                local_tok_q = torch.clamp(local_tok_q, 0, I_par_z - 1)
                tok_all_clamped = torch.clamp(tok_all, 0, I_z - 1)

                # Index into chunked Z
                Z_chunk = Z_II[local_tok_q][:, tok_all_clamped]  # [L_par, L, c_z]

                # Process through linear layer
                if hasattr(self.diffusion_token_encoder, 'process_z'):
                    Z_chunk = self.diffusion_token_encoder.process_z(Z_chunk)
            else:
                # Standard mode: Z_II is [I, I, c_z] tensor
                processed_Z = self.diffusion_token_encoder.process_z(Z_II) if hasattr(self.diffusion_token_encoder, 'process_z') else Z_II
                # Gather: Z_II[tok_idx_q, tok_idx_all, :] → [L_par, L, c_z]
                Z_chunk = processed_Z[tok_idx_q][:, tok_idx]   # [L_par, L, c_z]

            # Project to c_atompair if dimensions differ
            if Z_chunk.shape[-1] != c_atompair:
                # Simple linear projection (in practice, use a learned layer)
                Z_chunk = Z_chunk[..., :c_atompair]

            P_LL_chunk = P_LL_chunk + Z_chunk

        # Apply mask
        P_LL_chunk = P_LL_chunk * valid_mask_chunk.float()

        return P_LL_chunk.contiguous()                     # [L_par, L, c_atompair]

    def process_(
        self,
        D_II_self,
        X_L_self,
        *,
        R_L_uniform,
        X_noisy_L,
        t_L,
        f,
        Q_L,
        C_L,
        P_LL,
        A_I,
        S_I,
        Z_II,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        parallel_mode=False,
        **kwargs,
    ):
        """
        Parallel version of process_() - single recycling step.

        Overrides base class to use parallel transformer/decoder methods when
        parallel_mode=True.

        Args:
            D_II_self: [B, I, I, n_bins] or [B, I_par, I, n_bins] self-conditioning
            X_L_self: [B, L, 3] previous step positions or None
            R_L_uniform: [B, L, 3] scaled positions
            X_noisy_L: [B, L, 3] noisy positions
            t_L: [B, L] per-atom noise levels
            f: Feature dictionary
            Q_L: [B, L, c_atom] atom features
            C_L: [B, L, c_atom] conditioned atom features
            P_LL: [L, L, c_atompair] or None
            A_I: [B, I, c_token] token features
            S_I: [B, I, c_s] single features
            Z_II: [I, I, c_z] (full) OR [I_par, I, c_z] (chunked)
            chunked_pairwise_embedder: For sparse P_LL computation
            initializer_outputs: Additional outputs from TokenInitializer
            parallel_mode: If True, use parallel Z processing

        Returns:
            dict with X_L, D_II_self, sequence_logits_I, sequence_indices_I
        """
        # Determine if Z_II is parallel (Z_II is [I_par, I, c_z] instead of [I, I, c_z])
        is_parallel = parallel_mode

        # Get z_chunk_range from kwargs or initializer_outputs
        z_chunk_range = kwargs.get("z_chunk_range")
        if z_chunk_range is None and initializer_outputs is not None:
            z_chunk_range = initializer_outputs.get("z_chunk_range")

        # ... Embed token level features with atom level encodings
        with debug_time("DIFFUSION", "diffusion_token_encoder"):
            S_I, Z_II = self.diffusion_token_encoder(
                f=f,
                R_L=R_L_uniform,                               # [B, L, 3]
                D_II_self=D_II_self,                           # [B, I, I, n_bins] or [B, I_par, I, n_bins] or None
                S_init_I=S_I,                                  # [B, I, c_s]
                Z_init_II=Z_II,                                # [I, I, c_z] or [I_par, I, c_z]
                C_L=C_L,                                       # [B, L, c_atom]
                P_LL=P_LL,                                     # [L, L, c_atompair] or None
                parallel_mode=parallel_mode,
                z_chunk_range=z_chunk_range,
            )                                                  # Returns: [I, c_s], [I, I, c_z] or [I_par, I, c_z]

        # MEMORY TRACKING: Log token tensor memory after diffusion_token_encoder
        debug_memory("DIFFUSION", "after_token_encoder")
        debug_tensor_memory("DIFFUSION", "S_I_after_encoder", S_I)
        debug_tensor_memory("DIFFUSION", "Z_II_after_encoder", Z_II)

        # Determine full mode for transformer
        gpu_rank, world_size = get_gpu_rank_and_world_size()
        use_full_attention = not (
            os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1" or is_parallel_mode()
        )

        # Check if Z_II is chunked (from parallel encoder)
        # In multi-GPU mode, Z_II is [I_par, I, c_z] per GPU, not full [I, I, c_z]
        z_is_chunked = is_parallel and world_size > 1

        # CRITICAL: Ensure parallel_mode and z_chunk_range are in initializer_outputs
        # so that downstream code (decoder, chunked_pairwise) can access them
        if initializer_outputs is None:
            initializer_outputs = {}
        if is_parallel:
            initializer_outputs["parallel_mode"] = True
            initializer_outputs["z_chunk_range"] = z_chunk_range

        # ... Diffusion transformer with GPU-parallel attention
        with debug_time("DIFFUSION", "diffusion_transformer"):
            if z_is_chunked:
                # Multi-GPU parallel: each GPU processes its query chunk
                A_I = self._diffusion_transformer_parallel(
                    A_I=A_I,                                   # [B, I, c_token]
                    S_I=S_I,                                   # [I, c_s]
                    Z_II_chunk=Z_II,                           # [I_par, I, c_z] - this GPU's rows
                    f=f,
                    X_L=(
                        X_noisy_L[..., f["is_ca"], :]
                        if X_L_self is None
                        else X_L_self[..., f["is_ca"], :]
                    ),
                    gpu_rank=gpu_rank,
                    world_size=world_size,
                )                                              # [B, I, c_token] (all_gathered)
            else:
                # Standard mode - call parent class method
                A_I = self.diffusion_transformer(
                    A_I,                                       # [B, I, c_token]
                    S_I,                                       # [I, c_s] or [B, I, c_s]
                    Z_II,                                      # [I, I, c_z] or [B, I, I, c_z]
                    f=f,
                    X_L=(
                        X_noisy_L[..., f["is_ca"], :]
                        if X_L_self is None
                        else X_L_self[..., f["is_ca"], :]
                    ),
                    full=use_full_attention,
                )                                              # [B, I, c_token]

        # ... Decoder readout
        # Handle all combinations of LOW_MEMORY_MODE and ATTENTION_PARALLEL:
        #
        # | z_is_chunked | chunked_embedder | Mode                              |
        # |--------------|------------------|-----------------------------------|
        # | False        | None             | Standard: full P_LL               |
        # | False        | Present          | LOW_MEM: sparse P_LL via embedder |
        # | True         | None             | PARALLEL: P_chunk cross-attention |
        # | True         | Present          | BOTH: sparse P_LL across GPUs     |
        #
        with debug_time("DIFFUSION", "decoder"):
            if z_is_chunked and chunked_pairwise_embedder is not None:
                # BOTH modes: Sparse P_LL computation split across GPUs
                # Most memory efficient: each GPU computes sparse P_LL for its L_par atoms
                A_I, Q_L, o = self._decoder_parallel_sparse(
                    A_I=A_I,                                   # [B, I, c_token]
                    S_I=S_I,                                   # [I, c_s]
                    Q_L=Q_L,                                   # [B, L, c_atom]
                    C_L=C_L,                                   # [B, L, c_atom]
                    f=f,
                    chunked_pairwise_embedder=chunked_pairwise_embedder,
                    initializer_outputs=initializer_outputs,
                    gpu_rank=gpu_rank,
                    world_size=world_size,
                )                                              # A_I: [B, I, c_token], Q_L: [B, L, c_atom]
            elif z_is_chunked:
                # PARALLEL only: cross-attention with P_chunk [L_par, L]
                # Each GPU computes full P for its query chunk
                A_I, Q_L, o = self._decoder_parallel(
                    A_I=A_I,                                   # [B, I, c_token]
                    S_I=S_I,                                   # [I, c_s]
                    Q_L=Q_L,                                   # [B, L, c_atom]
                    C_L=C_L,                                   # [B, L, c_atom]
                    f=f,
                    initializer_outputs=initializer_outputs,
                    gpu_rank=gpu_rank,
                    world_size=world_size,
                )                                              # A_I: [B, I, c_token], Q_L: [B, L, c_atom]
            elif chunked_pairwise_embedder is not None:
                # LOW_MEM only: sparse P_LL via chunked_pairwise_embedder
                A_I, Q_L, o = self.decoder(
                    A_I,                                       # [B, I, c_token]
                    S_I,                                       # [I, c_s] or [B, I, c_s]
                    None,                                      # Decoder doesn't use Z_II directly
                    Q_L,                                       # [B, L, c_atom]
                    C_L,                                       # [B, L, c_atom]
                    P_LL=None,
                    tok_idx=f["atom_to_token_map"],            # [L]
                    indices=f["attn_indices"],                 # [B, L, k]
                    f=f,
                    chunked_pairwise_embedder=chunked_pairwise_embedder,
                    initializer_outputs=initializer_outputs,
                )
            else:
                # Standard mode: full P_LL [L, L, c_atompair]
                A_I, Q_L, o = self.decoder(
                    A_I,
                    S_I,
                    Z_II,
                    Q_L,
                    C_L,
                    P_LL=P_LL,
                    tok_idx=f["atom_to_token_map"],
                    indices=f["attn_indices"],
                )

        # ... Process outputs to positions update
        R_update_L = self.to_r_update(Q_L)                 # [B, L, 3]

        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t_L)  # [B, L, 3]

        sequence_logits_I, sequence_indices_I = self.sequence_head(A_I=A_I)

        # Self-conditioning distogram for recycling
        # In multi-GPU parallel mode: compute ONLY this GPU's chunk of D_II_self
        # to avoid creating full [I, I] tensor
        if z_is_chunked:
            # Compute chunk ranges for tokens (same as atoms via is_ca)
            I = A_I.shape[1]
            chunk_ranges = compute_chunk_ranges(I, world_size)
            if gpu_rank < len(chunk_ranges):
                start_i, end_i = chunk_ranges[gpu_rank]
                # Compute D_II_self ONLY for this GPU's query rows
                # D_II_self_chunk: [B, I_par, I, n_bins] - NOT [B, I, I, n_bins]!
                D_II_self_chunk = bucketize_scaled_distogram_chunked(
                    X_out_L[..., f["is_ca"], :].detach(),
                    query_start=start_i,
                    query_end=end_i,
                    sigma_data=1,
                    n_bins=self.n_bins,
                )                                          # [B, I_par, I, n_bins]

                # =======================================================================
                # MEMORY OPTIMIZATION: DON'T all_gather D_II_self back to full [I, I]
                # The downstream code (encoders.forward) immediately slices
                # D_II_self back to [I_par, I], so gathering is wasteful.
                # For I=9000 (length 150), full D_II_self = 21 GB per GPU!
                # Keeping it chunked saves massive memory.
                # =======================================================================

                # Each GPU keeps its own [B, I_par, I, n_bins] chunk
                # encoders.forward will use it directly without slicing
                D_II_self = D_II_self_chunk  # [B, I_par, I, n_bins] - NOT gathered!
            else:
                # Edge case: more GPUs than tokens
                D_II_self = None
        else:
            # Standard mode: compute full D_II_self
            D_II_self = self.bucketize_fn(X_out_L[..., f["is_ca"], :].detach())  # [B, I, I, n_bins]

        # MULTI-GPU DIAGNOSTIC: Verify final outputs are synchronized across all ranks
        if z_is_chunked:
            debug_tensor_all_ranks("FINAL_OUTPUT", "X_out_L", X_out_L)
            verify_tensor_sync("FINAL_OUTPUT", "X_out_L_SYNC", X_out_L)
            if D_II_self is not None:
                debug_tensor_all_ranks("FINAL_OUTPUT", "D_II_self", D_II_self)

        return {
            "X_L": X_out_L,                                # [B, L, 3]
            "D_II_self": D_II_self,                        # [B, I, I, n_bins] or [B, I_par, I, n_bins] or None
            "sequence_logits_I": sequence_logits_I,        # [B, I, vocab]
            "sequence_indices_I": sequence_indices_I,      # [B, I]
        } | o
