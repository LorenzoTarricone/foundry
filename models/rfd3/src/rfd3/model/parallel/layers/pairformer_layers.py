"""
Parallel implementations of Pairformer attention methods.

Free functions extracted from pairformer_layers.py for multi-GPU parallel inference.
"""

import torch


def attention_pair_bias_forward_chunked(
    self,  # AttentionPairBiasPairformerDeepspeed instance
    A_I_query: torch.Tensor,
    A_I_key: torch.Tensor,
    Z_chunk: torch.Tensor,
    Beta_II: torch.Tensor = None,
) -> torch.Tensor:
    """
    Chunked cross-attention for multi-GPU parallel inference.

    Each GPU computes attention for its query chunk against ALL keys.
    Mimics standard self-attention but with Q from chunk, K/V from all.

    Args:
        self: AttentionPairBiasPairformerDeepspeed instance
        A_I_query: [I_par, C_a] - query tokens for this GPU
        A_I_key: [I, C_a] - all key/value tokens
        Z_chunk: [I_par, I, C_z] - pair bias (query rows only)
        Beta_II: Optional scalar bias

    Returns:
        [I_par, C_a] - updated query tokens
    """
    I_par = A_I_query.shape[0]
    I = A_I_key.shape[0]

    # Shape assertions for debugging
    assert A_I_query.dim() == 2, f"A_I_query must be 2D [I_par, C], got {A_I_query.shape}"
    assert A_I_key.dim() == 2, f"A_I_key must be 2D [I, C], got {A_I_key.shape}"
    assert Z_chunk.dim() == 3, f"Z_chunk must be 3D [I_par, I, C], got {Z_chunk.shape}"
    assert Z_chunk.shape[0] == I_par, f"Z_chunk dim 0 ({Z_chunk.shape[0]}) must match I_par ({I_par})"
    assert Z_chunk.shape[1] == I, f"Z_chunk dim 1 ({Z_chunk.shape[1]}) must match I ({I})"

    # Normalize (same ln_1 for both, like standard self-attn)
    A_I_query_normed = self.ln_1(A_I_query)
    A_I_key_normed = self.ln_1(A_I_key)

    if self.force_bfloat16:
        A_I_query_normed = A_I_query_normed.to(torch.bfloat16)
        A_I_key_normed = A_I_key_normed.to(torch.bfloat16)

    # Q from chunk, K/V from all (cross-attention pattern)
    Q_chunk = self.to_q(A_I_query_normed)
    G_chunk = self.to_g(A_I_query_normed)
    K_all = self.to_k(A_I_key_normed)
    V_all = self.to_v(A_I_key_normed)

    # Pair bias from Z chunk
    B_chunk = self.to_b(self.ln_0(Z_chunk))
    if Beta_II is not None:
        if Beta_II.dim() == 0 or Beta_II.numel() == 1:
            B_chunk = B_chunk + Beta_II
        else:
            B_chunk = B_chunk + Beta_II[..., None]

    # Scale queries
    Q_chunk = Q_chunk / torch.sqrt(
        torch.tensor(self.c, device=Q_chunk.device, dtype=torch.bfloat16)
    )

    # Cross-attention: [I_par] queries attend to [I] keys
    attn_logits = torch.einsum("qhd,khd->qkh", Q_chunk, K_all) + B_chunk
    attn_weights = torch.softmax(attn_logits, dim=1)

    # Weighted sum of values
    out = torch.einsum("qkh,khc->qhc", attn_weights, V_all)

    # Gating and output projection
    out = G_chunk * out
    out = out.flatten(start_dim=-2)
    out = self.to_a(out)

    return out
