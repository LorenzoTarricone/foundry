"""
Parallel implementations of block forward methods for multi-GPU inference.

Free functions extracted from blocks.py parallel methods.
"""

import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

from rfd3.model.layers.block_utils import create_attention_indices
from rfd3.model.parallel.utils import compute_chunk_ranges
from rfd3.model.debug_context import debug_ctx, debug_tensor, debug_log, timed_all_gather


def _log_tensor_stats(name: str, tensor: torch.Tensor, rank: int = 0):
    """Log tensor statistics for debugging. Controlled by verbose_stats config flag."""
    if not debug_ctx.stats_enabled:
        return
    debug_tensor("BLOCKS", name, tensor, rank)


def local_token_transformer_cross_attn(
    self,  # LocalTokenTransformer instance
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
        self: LocalTokenTransformer instance
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
    self.to(device)

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
            n_attn_keys=self.n_keys,
            n_attn_seq_neighbours=self.n_local_tokens,
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
    for block_idx, block in enumerate(self.blocks):
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


def local_atom_transformer_parallel(
    self,  # LocalAtomTransformer instance
    Q_L,
    C_L,
    indices,
    query_start,
    query_end,
    f,
    chunked_pairwise_embedder,
    initializer_outputs,
    all_gather_fn,
    world_size,
):
    """
    Parallel forward: each GPU processes L_par = L/n_par atom queries.

    No L×L tensor is materialized:
    - P_LL_sparse computed on-the-fly for L_par queries only
    - Each GPU computes [L_par, k, c] instead of full [L, k, c]
    - Q_L chunks gathered after each block

    Args:
        self: LocalAtomTransformer instance
        Q_L: Full atom features [B, L, c_atom] (for K/V)
        C_L: Conditioned features [B, L, c_atom]
        indices: Full attention indices [B, L, k]
        query_start, query_end: This GPU's query range
        f, chunked_pairwise_embedder, initializer_outputs: For P_LL computation
        all_gather_fn: Function to gather Q_L chunks across GPUs
        world_size: Number of GPUs

    Returns:
        Q_L: Updated atom features [B, L, c_atom]
    """
    B = Q_L.shape[0]
    L = Q_L.shape[1]
    c_atom = Q_L.shape[2]
    L_par = query_end - query_start

    max_L_par = (L + world_size - 1) // world_size

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # Ensure all tensors are on the correct device
    Q_L = Q_L.to(device)
    C_L = C_L.to(device)
    indices = indices.to(device)

    indices_chunk = indices[:, query_start:query_end, :]

    # Ensure encoder blocks and embedder are on the correct device
    self.to(device)
    if chunked_pairwise_embedder is not None:
        chunked_pairwise_embedder.to(device)

    for block_idx, block in enumerate(self.blocks):
        Q_L_chunk = Q_L[:, query_start:query_end, :]
        C_L_chunk = C_L[:, query_start:query_end, :]

        # Compute P_LL_sparse for THIS GPU's queries only (free function)
        from rfd3.model.parallel.layers.chunked_pairwise import chunked_pairwise_forward_parallel
        P_sparse_chunk = chunked_pairwise_forward_parallel(
            chunked_pairwise_embedder,
            indices_chunk=indices_chunk,
            query_start=query_start,
            query_end=query_end,
            f=f,
            initializer_outputs=initializer_outputs,
        )

        # Sparse cross-attention (free function)
        Q_L_chunk = structure_local_atom_block_sparse_cross_attn(
            block,
            Q_chunk=Q_L_chunk,
            C_Q_chunk=C_L_chunk,
            K_V_full=Q_L,
            C_KV_full=C_L,
            P_sparse_chunk=P_sparse_chunk,
            indices_chunk=indices_chunk,
        )

        if block_idx == 0:
            _log_tensor_stats(f"encoder_Q_L_chunk_after_attn_block{block_idx}", Q_L_chunk)

        # Pad to max_L_par for all_gather
        if L_par < max_L_par:
            pad_size = max_L_par - L_par
            Q_L_chunk = torch.nn.functional.pad(Q_L_chunk, (0, 0, 0, pad_size))

        # All-gather padded chunks
        Q_L_gathered = all_gather_fn(Q_L_chunk, world_size, dim=1)

        # Trim back to actual L
        Q_L = Q_L_gathered[:, :L, :].to(device)

        if block_idx == 0:
            _log_tensor_stats(f"encoder_Q_L_after_gather_block{block_idx}", Q_L)

    return Q_L


def structure_local_atom_block_cross_attn(
    self,  # StructureLocalAtomTransformerBlock instance
    Q_chunk,
    C_Q_chunk,
    K_V_full,
    C_KV_full,
    P_chunk,
    indices_chunk,
):
    """
    Cross-attention forward: queries from chunk attend to all keys.

    For multi-GPU parallel inference:
    - Each GPU processes L_par = L // n_gpus queries
    - All GPUs have access to all L keys
    - P_chunk is [L_par, L] not [L, L] - avoids L×L memory

    Args:
        self: StructureLocalAtomTransformerBlock instance
        Q_chunk: Query features [D, L_par, c_a]
        C_Q_chunk: Query conditioning [D, L_par, c_s]
        K_V_full: Key/Value features [D, L, c_a]
        C_KV_full: K/V conditioning [D, L, c_s]
        P_chunk: Pair bias [L_par, L, c_pair] or [D, L_par, L, c_pair]
        indices_chunk: Attention indices [D, L_par, k]

    Returns:
        Q_chunk: Updated query features [D, L_par, c_a]
    """
    # Import here to avoid circular dependency
    from rfd3.model.parallel.layers.attention import local_attention_cross_attn

    # Cross-attention: Q_chunk queries attend to K_V_full keys
    attn_out = local_attention_cross_attn(
        self.attention_pair_bias,
        Q_chunk=Q_chunk,
        C_Q_chunk=C_Q_chunk,
        K_V_full=K_V_full,
        C_KV_full=C_KV_full,
        P_chunk=P_chunk,
        indices_chunk=indices_chunk,
    )

    Q_chunk = Q_chunk + self.dropout(attn_out)

    # Transition block on query chunk
    from foundry.common import exists
    if exists(C_Q_chunk):
        Q_chunk = Q_chunk + self.transition_block(Q_chunk, C_Q_chunk)
    else:
        Q_chunk = Q_chunk + self.transition_block(Q_chunk)

    return Q_chunk


def structure_local_atom_block_sparse_cross_attn(
    self,  # StructureLocalAtomTransformerBlock instance
    Q_chunk,
    C_Q_chunk,
    K_V_full,
    C_KV_full,
    P_sparse_chunk,
    indices_chunk,
):
    """
    Sparse cross-attention: queries attend to k sparse neighbors.

    Combines parallelism (L_par queries) with sparse attention (k neighbors):
    - P_sparse_chunk is [L_par, k] not [L_par, L] or [L, L]
    - K/V are gathered from indices, not dense
    - Memory: O(L_par * k) instead of O(L_par * L) or O(L²)

    Args:
        self: StructureLocalAtomTransformerBlock instance
        Q_chunk: Query features [D, L_par, c_a]
        C_Q_chunk: Query conditioning [D, L_par, c_s]
        K_V_full: Key/Value features [D, L, c_a]
        C_KV_full: K/V conditioning [D, L, c_s]
        P_sparse_chunk: Sparse pair bias [D, L_par, k, c_pair]
        indices_chunk: Sparse attention indices [D, L_par, k]

    Returns:
        Q_chunk: Updated query features [D, L_par, c_a]
    """
    # Import here to avoid circular dependency
    from rfd3.model.parallel.layers.attention import local_attention_sparse_cross_attn

    # Sparse cross-attention: Q_chunk queries attend to sparse K/V
    attn_out = local_attention_sparse_cross_attn(
        self.attention_pair_bias,
        Q_chunk=Q_chunk,
        C_Q_chunk=C_Q_chunk,
        K_V_full=K_V_full,
        C_KV_full=C_KV_full,
        P_sparse_chunk=P_sparse_chunk,
        indices_chunk=indices_chunk,
    )

    Q_chunk = Q_chunk + self.dropout(attn_out)

    # Transition block on query chunk
    from foundry.common import exists
    if exists(C_Q_chunk):
        Q_chunk = Q_chunk + self.transition_block(Q_chunk, C_Q_chunk)
    else:
        Q_chunk = Q_chunk + self.transition_block(Q_chunk)

    return Q_chunk


def compact_decoder_parallel(
    self,  # CompactStreamingDecoder instance
    A_I,
    S_I,
    Q_L,
    C_L,
    P_chunk,
    tok_idx,
    indices,
    query_start,
    query_end,
    all_gather_fn,
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
        self: CompactStreamingDecoder instance
        A_I: Token features [B, I, c_token]
        S_I: Single features [I, c_s]
        Q_L: Atom features [B, L, c_atom]
        C_L: Conditioned atom features [B, L, c_atom]
        P_chunk: This GPU's P_LL rows [L_par, L, c_pair]
        tok_idx: Atom to token mapping [L]
        indices: Full attention indices [B, L, k]
        query_start: Start index of this GPU's queries
        query_end: End index of this GPU's queries
        all_gather_fn: Function to gather tensors from all GPUs
        world_size: Number of GPUs

    Returns:
        A_I: Updated token features [B, I, c_token]
        Q_L: Updated atom features [B, L, c_atom]
        o: Empty dict (for compatibility)
    """
    B = A_I.shape[0]
    L = Q_L.shape[1]
    L_par = query_end - query_start

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # Ensure decoder modules are on the correct device
    self.to(device)

    indices_chunk = indices[:, query_start:query_end, :]

    # Run blocks with cross-attention
    for i in range(self.n_blocks):
        # Upcast: token → atom
        Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)

        # Extract this GPU's Q_L chunk for attention
        Q_L_chunk = Q_L[:, query_start:query_end, :]
        C_L_chunk = C_L[:, query_start:query_end, :]

        # Cross-attention: Q_L_chunk queries attend to Q_L keys
        Q_L_chunk = structure_local_atom_block_cross_attn(
            self.atom_transformer[i],
            Q_chunk=Q_L_chunk,
            C_Q_chunk=C_L_chunk,
            K_V_full=Q_L,
            C_KV_full=C_L,
            P_chunk=P_chunk,
            indices_chunk=indices_chunk,
        )

        # All-gather Q_L chunks to reconstruct full Q_L
        Q_L = all_gather_fn(Q_L_chunk, world_size, dim=1)

    # Downcast to sequence
    A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

    o = {}
    return A_I, Q_L, o


def compact_decoder_parallel_sparse(
    self,  # CompactStreamingDecoder instance
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
    all_gather_fn,
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
        self: CompactStreamingDecoder instance
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
        all_gather_fn: Function to gather tensors from all GPUs
        world_size: Number of GPUs

    Returns:
        A_I: Updated token features [B, I, c_token]
        Q_L: Updated atom features [B, L, c_atom]
        o: Empty dict
    """
    B = A_I.shape[0]
    L = Q_L.shape[1]
    L_par = query_end - query_start

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")

    # Ensure decoder modules and embedder are on the correct device
    self.to(device)
    if chunked_pairwise_embedder is not None:
        chunked_pairwise_embedder.to(device)

    # Run blocks
    for i in range(self.n_blocks):
        # Upcast: token → atom
        Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)

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
            _log_tensor_stats(f"decoder_P_sparse_chunk_block{i}", P_sparse_chunk)

        # Sparse cross-attention
        Q_L_chunk = structure_local_atom_block_sparse_cross_attn(
            self.atom_transformer[i],
            Q_chunk=Q_L_chunk,
            C_Q_chunk=C_L_chunk,
            K_V_full=Q_L,
            C_KV_full=C_L,
            P_sparse_chunk=P_sparse_chunk,
            indices_chunk=indices_chunk,
        )

        if i == 0:
            _log_tensor_stats(f"decoder_Q_L_chunk_after_attn_block{i}", Q_L_chunk)

        # All-gather Q_L chunks to reconstruct full Q_L
        Q_L = all_gather_fn(Q_L_chunk, world_size, dim=1)

        if i == 0:
            _log_tensor_stats(f"decoder_Q_L_after_gather_block{i}", Q_L)

    # Downcast to sequence
    A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

    o = {}
    return A_I, Q_L, o
