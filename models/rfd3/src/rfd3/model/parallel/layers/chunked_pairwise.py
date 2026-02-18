"""
Parallel implementation of chunked pairwise embedding for multi-GPU inference.

Free function extracted from chunked_pairwise.py.
"""

import torch
import torch.distributed as dist

from rfd3.model.debug_context import debug_ctx, debug_tensor_all_ranks


def chunked_pairwise_forward_parallel(
    embedder,  # ChunkedPairwiseEmbedder instance
    indices_chunk,
    query_start,
    query_end,
    f,
    initializer_outputs,
):
    """
    Compute P_LL_sparse for a CHUNK of queries only.

    For multi-GPU parallelization:
    - Only computes [L_par, k, c_atompair] instead of [L, k, c_atompair]
    - Memory: O(L_par * k) instead of O(L * k) per GPU
    - Each GPU calls this with its query_start:query_end range

    Args:
        embedder: ChunkedPairwiseEmbedder instance
        indices_chunk: [B, L_par, k] - sparse neighbor indices for this chunk
        query_start: Start index of this GPU's queries
        query_end: End index of this GPU's queries
        f: Feature dict with atom positions, masks, etc.
        initializer_outputs: Dict with tok_idx, Z_init_II, etc.

    Returns:
        P_LL_sparse_chunk: [B, L_par, k, c_atompair]
    """
    B = indices_chunk.shape[0]
    L_par = query_end - query_start
    k = indices_chunk.shape[2]
    device = indices_chunk.device

    # Ensure embedder modules are on the correct device
    embedder.to(device)

    # Get full data from initializer_outputs and move to correct device
    tok_idx = f.get("atom_to_token_map", initializer_outputs.get("tok_idx"))
    if tok_idx is not None:
        tok_idx = tok_idx.to(device)
    Z_init_II = initializer_outputs.get("Z_II", initializer_outputs.get("Z_init_II"))
    if Z_init_II is not None and hasattr(Z_init_II, 'to'):
        Z_init_II = Z_init_II.to(device)
    C_L = initializer_outputs.get("C_L")
    if C_L is not None:
        C_L = C_L.to(device)

    # Get dtype from available tensor
    dtype = C_L.dtype if C_L is not None else torch.bfloat16

    # Initialize output for this chunk
    P_LL_sparse_chunk = torch.zeros(
        B, L_par, k, embedder.c_atompair, device=device, dtype=dtype
    )

    # Ensure indices_chunk is properly batched
    if indices_chunk.dim() == 2:
        indices_chunk = indices_chunk.unsqueeze(0).expand(B, -1, -1)

    # 1. Single embeddings for chunk (VECTORIZED)
    if C_L is not None:
        if C_L.dim() == 2:
            C_L = C_L.unsqueeze(0)
        if C_L.shape[0] != B:
            C_L = C_L.expand(B, -1, -1)

        C_L_chunk = C_L[:, query_start:query_end, :]
        C_L_queries = C_L_chunk.unsqueeze(2).expand(-1, -1, k, -1)

        # Gather key features - VECTORIZED
        indices_for_gather = indices_chunk.unsqueeze(-1).expand(-1, -1, -1, C_L.shape[-1])
        indices_clamped = torch.clamp(indices_for_gather, 0, C_L.shape[1] - 1)
        C_L_keys = torch.gather(C_L.unsqueeze(2).expand(-1, -1, k, -1), 1, indices_clamped)

        single_l = embedder._get_process_single_l()(C_L_queries)
        single_m = embedder._get_process_single_m()(C_L_keys)
        P_LL_sparse_chunk = P_LL_sparse_chunk + single_l + single_m

    # 2. Token pair features Z for chunk (VECTORIZED)
    if tok_idx is not None and Z_init_II is not None:
        if tok_idx.dim() == 1:
            tok_idx_expanded = tok_idx.unsqueeze(0).expand(B, -1)
        else:
            tok_idx_expanded = tok_idx
            if tok_idx_expanded.shape[0] != B:
                tok_idx_expanded = tok_idx_expanded.expand(B, -1)

        # Token indices for this chunk's queries
        tok_queries_chunk = tok_idx_expanded[:, query_start:query_end].unsqueeze(2).expand(-1, -1, k)

        # Get token indices for keys via gather - VECTORIZED
        tok_idx_for_keys = tok_idx_expanded.unsqueeze(2).expand(-1, -1, k)
        indices_clamped = torch.clamp(indices_chunk, 0, tok_idx_expanded.shape[1] - 1)
        tok_keys_chunk = torch.gather(tok_idx_for_keys, 1, indices_clamped)

        # Check for parallel mode
        parallel_mode = initializer_outputs.get("parallel_mode", False)
        z_chunk_range = initializer_outputs.get("z_chunk_range")

        if parallel_mode and z_chunk_range is not None:
            # Streaming mode: Z_init_II is pre-computed [I_par, I, c_z] chunk
            start_i, end_i = z_chunk_range
            I_par_z = Z_init_II.shape[0]
            I_z = Z_init_II.shape[1]

            tq = tok_queries_chunk[0]
            tk = tok_keys_chunk[0]

            # Map to local chunk indices
            local_tq = torch.clamp(tq - start_i, 0, I_par_z - 1)
            tk = torch.clamp(tk, 0, I_z - 1)

            # Index into chunked Z
            Z_pairs_full = Z_init_II[
                local_tq.flatten(),
                tk.flatten()
            ].view(L_par, k, -1)

            Z_pairs_processed = embedder._get_process_z()(Z_pairs_full)
            Z_pairs_processed_chunk = Z_pairs_processed.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            # Standard mode: full Z tensor [I, I, c_z]
            I_z = Z_init_II.shape[0]
            Z_processed = embedder._get_process_z()(Z_init_II)

            # VECTORIZED gather for Z pairs
            tq_flat = torch.clamp(tok_queries_chunk.reshape(-1), 0, I_z - 1)
            tk_flat = torch.clamp(tok_keys_chunk.reshape(-1), 0, I_z - 1)
            Z_pairs_flat = Z_processed[tq_flat, tk_flat, :]
            Z_pairs_processed_chunk = Z_pairs_flat.reshape(B, L_par, k, -1)

        # Add Z_pairs to P_LL
        P_LL_sparse_chunk = P_LL_sparse_chunk + Z_pairs_processed_chunk

    # Final MLP
    P_LL_sparse_chunk = P_LL_sparse_chunk + embedder._get_pair_mlp()(P_LL_sparse_chunk)

    # MULTI-GPU DIAGNOSTIC
    debug_tensor_all_ranks("P_LL_SPARSE", f"forward_chunked_parallel_q{query_start}-{query_end}", P_LL_sparse_chunk)

    return P_LL_sparse_chunk.contiguous()
