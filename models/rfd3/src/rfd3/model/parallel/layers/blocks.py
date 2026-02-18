"""
Parallel block-level cross-attention functions for multi-GPU inference.

Each function wraps a StructureLocalAtomTransformerBlock to perform
cross-attention (chunk queries → all keys) instead of self-attention.
"""


def structure_local_atom_block_cross_attn(
    block,  # StructureLocalAtomTransformerBlock instance
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
        block: StructureLocalAtomTransformerBlock instance
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
        block.attention_pair_bias,
        Q_chunk=Q_chunk,
        C_Q_chunk=C_Q_chunk,
        K_V_full=K_V_full,
        C_KV_full=C_KV_full,
        P_chunk=P_chunk,
        indices_chunk=indices_chunk,
    )

    Q_chunk = Q_chunk + block.dropout(attn_out)

    # Transition block on query chunk
    from foundry.common import exists
    if exists(C_Q_chunk):
        Q_chunk = Q_chunk + block.transition_block(Q_chunk, C_Q_chunk)
    else:
        Q_chunk = Q_chunk + block.transition_block(Q_chunk)

    return Q_chunk


def structure_local_atom_block_sparse_cross_attn(
    block,  # StructureLocalAtomTransformerBlock instance
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
        block: StructureLocalAtomTransformerBlock instance
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
        block.attention_pair_bias,
        Q_chunk=Q_chunk,
        C_Q_chunk=C_Q_chunk,
        K_V_full=K_V_full,
        C_KV_full=C_KV_full,
        P_sparse_chunk=P_sparse_chunk,
        indices_chunk=indices_chunk,
    )

    Q_chunk = Q_chunk + block.dropout(attn_out)

    # Transition block on query chunk
    from foundry.common import exists
    if exists(C_Q_chunk):
        Q_chunk = Q_chunk + block.transition_block(Q_chunk, C_Q_chunk)
    else:
        Q_chunk = Q_chunk + block.transition_block(Q_chunk)

    return Q_chunk
