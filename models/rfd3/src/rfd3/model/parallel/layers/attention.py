"""
Parallel implementations of attention methods for multi-GPU inference.

Free functions extracted from attention.py.
"""

from math import sqrt

import torch
import torch.distributed as dist

from foundry.common import exists
from rfd3.model.debug_context import debug_ctx


def sparse_cross_attention(Q, K, V, B, indices, H, G=None):
    """
    Cross-attention where queries (chunk) attend to all keys via sparse indices.

    Unlike sparse_pairbias_attention, Q has different length than K/V:
    - Q: [D, L_par, c] - queries from a chunk
    - K, V: [D, L, c] - keys/values from all tokens
    - B: [L_par, L, H] or [D, L_par, L, H] - pair bias for chunk×all
    - indices: [D, L_par, k] - which keys each query attends to

    This enables multi-GPU parallel inference where each GPU processes
    a chunk of queries (L_par = L // n_gpus) attending to all keys (L).

    Args:
        Q: Query vectors [D, L_par, c]
        K: Key vectors [D, L, c]
        V: Value vectors [D, L, c]
        B: Attention bias [L_par, L, H] or [D, L_par, L, H]
        indices: Sparse indices [D, L_par, k] specifying which keys to attend
        H: Number of attention heads
        G: Optional gating [D, L_par, c]

    Returns:
        attn_out: [D, L_par, c] - attention output for query chunk
    """
    D, L_par, c = Q.shape
    L = K.shape[1]
    k = indices.shape[-1]

    # Gather K, V using sparse indices
    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)
    K_gathered = K[batch_idx, indices].contiguous()
    V_gathered = V[batch_idx, indices].contiguous()

    # Gather bias
    query_idx = torch.arange(L_par, device=Q.device).view(1, L_par, 1)
    query_idx = query_idx.expand(D, -1, k)

    if B.ndim == 3:  # [L_par, L, H]
        B_gathered = B[query_idx, indices, :]
    elif B.ndim == 4:  # [D, L_par, L, H]
        B_gathered = B[batch_idx, query_idx, indices, :]
    else:
        raise ValueError(f"B must have 3 or 4 dims, got {B.ndim}")
    B_gathered = B_gathered.contiguous()

    # Debug logging (rank 0 only, if stats enabled)
    do_debug = debug_ctx.stats_enabled and ((not dist.is_initialized()) or dist.get_rank() == 0)
    if do_debug:
        with torch.no_grad():
            def _stat(t):
                return {
                    "shape": list(t.shape),
                    "mean": float(t.float().mean()),
                    "std": float(t.float().std()),
                    "min": float(t.float().min()),
                    "max": float(t.float().max()),
                }
            def _pct(t):
                try:
                    qs = torch.tensor([0.01, 0.5, 0.99], device=t.device, dtype=torch.float32)
                    vals = torch.quantile(t.float().flatten(), qs)
                    return [float(v) for v in vals]
                except Exception:
                    return None
            print(
                f"{debug_ctx.prefix('ATTN')} sparse_xattn inputs:",
                {
                    "Q": _stat(Q),
                    "K_gathered": _stat(K_gathered),
                    "V_gathered": _stat(V_gathered),
                    "B_gathered": {
                        **_stat(B_gathered),
                        "pct_1_50_99": _pct(B_gathered),
                    },
                },
                flush=True,
            )

    # Split into heads
    Q = Q.reshape(D, L_par, H, c // H)
    K_gathered = K_gathered.reshape(D, L_par, k, H, c // H)
    V_gathered = V_gathered.reshape(D, L_par, k, H, c // H)
    B_gathered = B_gathered.reshape(D, L_par, k, H)

    # Permute to [D, H, L_par, ...] for attention
    Q = Q.permute(0, 2, 1, 3)
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)
    B_gathered = B_gathered.permute(0, 3, 1, 2)

    # Attention: Q @ K^T
    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)
    attn = attn / sqrt(c // H)
    attn = attn + B_gathered

    if do_debug:
        with torch.no_grad():
            print(
                f"{debug_ctx.prefix('ATTN')} logits:",
                {
                    "mean": float(attn.float().mean()),
                    "std": float(attn.float().std()),
                    "min": float(attn.float().min()),
                    "max": float(attn.float().max()),
                },
                flush=True,
            )

    attn = torch.softmax(attn, dim=-1)

    if do_debug:
        with torch.no_grad():
            print(
                f"{debug_ctx.prefix('ATTN')} softmax:",
                {
                    "mean": float(attn.float().mean()),
                    "std": float(attn.float().std()),
                    "min": float(attn.float().min()),
                    "max": float(attn.float().max()),
                },
                flush=True,
            )

    # Apply attention to values
    attn_out = torch.einsum("...ij,...ijc->...ic", attn, V_gathered)

    # Optional gating
    if G is not None:
        G = G.reshape(D, L_par, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L_par, c).contiguous()

    if do_debug:
        with torch.no_grad():
            print(
                f"{debug_ctx.prefix('ATTN')} attn_out:",
                {
                    "mean": float(attn_out.float().mean()),
                    "std": float(attn_out.float().std()),
                    "min": float(attn_out.float().min()),
                    "max": float(attn_out.float().max()),
                },
                flush=True,
            )

    return attn_out


def sparse_cross_attention_pregathered_bias(Q, K, V, B_sparse, indices, H, G=None):
    """
    Sparse cross-attention with PRE-GATHERED bias (already [L_par, k, H]).

    Unlike sparse_cross_attention where B is [L_par, L, H] and we gather to [L_par, k, H],
    here B_sparse is already [L_par, k, H] - the bias has been computed only for the
    sparse (query, neighbor) pairs.

    This is the most memory-efficient attention:
    - Q: [D, L_par, c] - queries from a GPU's chunk
    - K, V: [D, L, c] - keys/values from all tokens
    - B_sparse: [D, L_par, k, H] - bias ONLY for k neighbors (not dense L)
    - indices: [D, L_par, k] - which keys each query attends to

    Memory: O(L_par * k) instead of O(L_par * L) or O(L²)

    Args:
        Q: Query vectors [D, L_par, c]
        K: Key vectors [D, L, c]
        V: Value vectors [D, L, c]
        B_sparse: Pre-gathered attention bias [D, L_par, k, H]
        indices: Sparse indices [D, L_par, k]
        H: Number of attention heads
        G: Optional gating [D, L_par, c]

    Returns:
        attn_out: [D, L_par, c] - attention output for query chunk
    """
    D, L_par, c = Q.shape
    k = indices.shape[-1]

    # Gather K, V using sparse indices
    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)
    K_gathered = K[batch_idx, indices].contiguous()
    V_gathered = V[batch_idx, indices].contiguous()

    # B_sparse is already [D, L_par, k, H] - no gathering needed!

    # Split into heads
    Q = Q.reshape(D, L_par, H, c // H)
    K_gathered = K_gathered.reshape(D, L_par, k, H, c // H)
    V_gathered = V_gathered.reshape(D, L_par, k, H, c // H)
    B_sparse = B_sparse.reshape(D, L_par, k, H)

    # Permute to [D, H, L_par, ...] for attention
    Q = Q.permute(0, 2, 1, 3)
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)
    B_sparse = B_sparse.permute(0, 3, 1, 2)

    # Attention: Q @ K^T
    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)
    attn = attn / sqrt(c // H)
    attn = attn + B_sparse
    attn = torch.softmax(attn, dim=-1)

    # Apply attention to values
    attn_out = torch.einsum("...ij,...ijc->...ic", attn, V_gathered)

    # Optional gating
    if G is not None:
        G = G.reshape(D, L_par, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L_par, c).contiguous()

    return attn_out


def local_attention_cross_attn(
    attn,  # LocalAttentionPairBias instance
    Q_chunk,
    C_Q_chunk,
    K_V_full,
    C_KV_full,
    P_chunk,
    indices_chunk,
):
    """
    Cross-attention: queries from a chunk attend to all keys.

    This avoids materializing [L, L] tensors by:
    - Q from chunk [L_par] only
    - K, V from all [L]
    - P_chunk is [L_par, L] not [L, L]

    Args:
        attn: LocalAttentionPairBias instance
        Q_chunk: Query features [D, L_par, c_a]
        C_Q_chunk: Query conditioning [D, L_par, c_s]
        K_V_full: Key/Value features [D, L, c_a]
        C_KV_full: K/V conditioning (currently unused)
        P_chunk: Pair bias [L_par, L, c_pair] or [D, L_par, L, c_pair]
        indices_chunk: Sparse indices [D, L_par, k]

    Returns:
        attn_out: [D, L_par, c_a] - attention output for this chunk
    """
    D, L_par, _ = Q_chunk.shape
    L = K_V_full.shape[1]

    # Normalize queries with conditioning
    if exists(C_Q_chunk):
        Q_normed = attn.ada_ln_1(Q_chunk, C_Q_chunk)
    else:
        Q_normed = attn.ln_1(Q_chunk)

    # Normalize K/V - use full input for keys/values
    if exists(C_KV_full):
        KV_normed = attn.ada_ln_1(K_V_full, C_KV_full)
    else:
        KV_normed = attn.ln_1(K_V_full)

    # Project to Q, K, V
    q = attn.to_q(Q_normed)
    k = attn.to_k(KV_normed)
    v = attn.to_v(KV_normed)
    g = attn.to_g(Q_normed)

    # Apply KQ norm if enabled
    if attn.kq_norm:
        q = attn.ln_q(q)
        k = attn.ln_k(k)

    # Project pair bias
    if P_chunk.ndim == 3:
        b = attn.to_b(P_chunk)
    else:  # [D, L_par, L, c_pair]
        b = attn.to_b(P_chunk)

    # Cross-attention with sparse indices
    attn_out = sparse_cross_attention(
        Q=q,
        K=k,
        V=v,
        B=b,
        G=g,
        indices=indices_chunk,
        H=attn.n_head,
    )

    # Output projection
    attn_out = attn.to_o(attn_out)

    # Apply output gating with query conditioning
    if exists(C_Q_chunk):
        attn_out = attn.linear_output_project(C_Q_chunk) * attn_out

    return attn_out


def local_attention_sparse_cross_attn(
    attn,  # LocalAttentionPairBias instance
    Q_chunk,
    C_Q_chunk,
    K_V_full,
    C_KV_full,
    P_sparse_chunk,
    indices_chunk,
):
    """
    Sparse cross-attention: queries attend to k sparse neighbors.

    Combines two optimizations:
    1. Cross-attention: L_par queries (chunked across GPUs)
    2. Sparse attention: k neighbors per query (not full L)

    Memory: O(L_par * k) instead of O(L_par * L) or O(L²)

    Args:
        attn: LocalAttentionPairBias instance
        Q_chunk: Query features [D, L_par, c_a]
        C_Q_chunk: Query conditioning [D, L_par, c_s]
        K_V_full: Key/Value features [D, L, c_a]
        C_KV_full: K/V conditioning [D, L, c_s]
        P_sparse_chunk: Sparse pair bias [D, L_par, k, c_pair]
        indices_chunk: Sparse attention indices [D, L_par, k]

    Returns:
        attn_out: [D, L_par, c_a] - attention output for this chunk
    """
    D, L_par, _ = Q_chunk.shape
    k = indices_chunk.shape[-1]

    # Normalize queries with conditioning
    if exists(C_Q_chunk):
        Q_normed = attn.ada_ln_1(Q_chunk, C_Q_chunk)
    else:
        Q_normed = attn.ln_1(Q_chunk)

    # Normalize K/V - use full input for keys/values
    if exists(C_KV_full):
        KV_normed = attn.ada_ln_1(K_V_full, C_KV_full)
    else:
        KV_normed = attn.ln_1(K_V_full)

    # Project to Q, K, V
    q = attn.to_q(Q_normed)
    k_proj = attn.to_k(KV_normed)
    v = attn.to_v(KV_normed)
    g = attn.to_g(Q_normed)

    # Apply KQ norm if enabled
    if attn.kq_norm:
        q = attn.ln_q(q)
        k_proj = attn.ln_k(k_proj)

    # Project pair bias: [D, L_par, k, c_pair] → [D, L_par, k, H]
    b_sparse = attn.to_b(P_sparse_chunk)

    # Sparse cross-attention: gather K/V using indices, use pre-gathered bias
    attn_out = sparse_cross_attention_pregathered_bias(
        Q=q,
        K=k_proj,
        V=v,
        B_sparse=b_sparse,
        G=g,
        indices=indices_chunk,
        H=attn.n_head,
    )

    # Output projection
    attn_out = attn.to_o(attn_out)

    # Apply output gating with query conditioning
    if exists(C_Q_chunk):
        attn_out = attn.linear_output_project(C_Q_chunk) * attn_out

    return attn_out
