import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from opt_einsum import contract as einsum
from rfd3.model.layers.block_utils import (
    create_attention_indices,
    indices_to_mask,
)
from rfd3.model.layers.layer_utils import (
    AdaLN,
    LinearBiasInit,
    RMSNorm,
    linearNoBias,
)

from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.ddp import RankedLogger
from rfd3.model.debug_context import debug_ctx, debug_log, debug_tensor

ranked_logger = RankedLogger(__name__, rank_zero_only=True)

try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias

    # ranked_logger.info("Fused PairBiasAttention enabled!")
    _CUEQ_AVAILABLE = True
except Exception:
    # ranked_logger.warning(
    #     "Using pytorch implementation instead of NVIDIA kernel"
    #     "Ensure you are using the latest apptainer."
    # )
    _CUEQ_AVAILABLE = False


@torch.compiler.disable
def kernel_pairbias_attention(
    *,
    s: torch.Tensor,  # (B, U, D) sequence features used for gating/output inside the kernel
    q: torch.Tensor,  # (B, H, U, DH)
    k: torch.Tensor,  # (B, H, V, DH)
    v: torch.Tensor,  # (B, H, V, DH)
    z: torch.Tensor,  # (B, U, V, z_dim)
    mask: torch.Tensor | None,  # (B, V) or (B*M, V) with 1=keep, 0=mask
    num_heads: int,
    w_proj_z: torch.Tensor,  # (H, z_dim)
    w_proj_g: torch.Tensor,  # (D, D)
    w_proj_o: torch.Tensor,  # (D, D)
    w_ln_z: torch.Tensor,  # (z_dim,)
    b_ln_z: torch.Tensor,  # (z_dim,)
    b_proj_z: torch.Tensor | None = None,  # (H,)
    b_proj_g: torch.Tensor | None = None,  # (D,)
    b_proj_o: torch.Tensor | None = None,  # (D,)
    attn_scale: float | None = None,
    compute_pair_bias: bool = True,
    multiplicity: int = 1,
) -> torch.Tensor:
    """Thin wrapper around cuequivariance_torch.attention_pair_bias."""
    raise NotImplementedError("CUDA Kernel for attention pair bias not implemented")
    out, _proj_z = cueq_attention_pair_bias(
        s=s,
        q=q,
        k=k,
        v=v,
        z=z,
        mask=mask,
        num_heads=num_heads,
        w_proj_z=w_proj_z,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        b_proj_z=b_proj_z,
        b_proj_g=b_proj_g,
        b_proj_o=b_proj_o,
        attn_scale=attn_scale,
        compute_pair_bias=compute_pair_bias,
        multiplicity=multiplicity,
    )
    return out  # (B, U, D)


######################################################################################
##########################        Network Modules           ##########################
######################################################################################


class GatedCrossAttention(nn.Module):
    def __init__(
        self,
        c_query,
        c_kv,
        c_pair=None,
        c_model=128,
        n_head=4,
        kq_norm=True,
        dropout=0.0,
        **_,
    ):
        super().__init__()
        self.n_head = n_head
        self.scale = 1 / math.sqrt(c_model // n_head)
        assert c_model % n_head == 0, "c_model must be divisible by n_heads"

        self.ln_q = RMSNorm(c_query)
        self.ln_kv = RMSNorm(c_kv)

        self.to_q = linearNoBias(c_query, c_model)
        self.to_k = linearNoBias(c_kv, c_model)
        self.to_v = linearNoBias(c_kv, c_model)
        self.to_g = nn.Sequential(
            linearNoBias(c_query, c_model),
            nn.Sigmoid(),
        )
        self.to_out = nn.Sequential(nn.Linear(c_model, c_query), nn.Dropout(dropout))
        self.kq_norm = kq_norm
        if self.kq_norm:
            self.k_norm = RMSNorm(c_model)
            self.q_norm = RMSNorm(c_model)

        self.c_pair = c_pair
        if c_pair is not None:
            self.to_b = nn.Sequential(RMSNorm(c_pair), linearNoBias(c_pair, n_head))
        self.reset_parameter()

    def reset_parameter(self):
        # query/key/value projection: Xavier uniform
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_k.weight)
        nn.init.xavier_uniform_(self.to_v.weight)
        nn.init.xavier_uniform_(self.to_g[0].weight)
        nn.init.xavier_uniform_(self.to_out[0].weight)

    def forward(self, q, kv, attn_mask=None, pair_bias=None):
        """
        Args:
            q: [B, tok, n_q, c_query]
            kv: [B, tok, n_kv, c_kv]
            attn_mask: [n_q, n_kv]
        Returns:
            attn_out: [B, tok, n_q, c_query]
        """

        q = self.ln_q(q)
        kv = self.ln_kv(kv)

        q, k, v, g = self.to_q(q), self.to_k(kv), self.to_v(kv), self.to_g(q)

        if self.kq_norm:
            k = self.k_norm(k)
            q = self.q_norm(q)

        q, k, v, g = map(
            lambda t: rearrange(t, "b t n (h c) -> b h t n c", h=self.n_head),
            (q, k, v, g),
        )  # [B, tok, n, heads, c] ->  [B, heads, tok, n, c]

        attn = einsum("bhtqc,bhtkc->bhtqk", q, k) * self.scale

        if pair_bias is not None:
            b = self.to_b(pair_bias)
            b = rearrange(b, "b t q k (h) -> b (h) t q k", h=self.n_head)
            attn = attn + b

        # Invalid query handling:
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[None, None], float("-inf"))

            # Bugfix: Empty queries need to have a constant value otherwise nans are in the forward graph. I don't
            # know why this causes instabilities because the invalid queries are masked out later. Oh well!
            invalid_queries = torch.logical_not(
                torch.any(attn_mask, dim=-1, keepdim=False)
            )  # [n_q,]
            attn[:, :, invalid_queries, :] = 0.0

        attn = F.softmax(attn, dim=-1)
        attn_out = einsum("bhtqk,bhtkd->bhtqd", attn, v)
        attn_out = attn_out * g

        attn_out = rearrange(attn_out, "b h t n c -> b t n (h c)")
        attn_out = self.to_out(attn_out)  # [B, n_tok, n_k, c]
        return attn_out


class LocalAttentionPairBias(nn.Module):
    def __init__(
        self,
        c_a,
        c_s,
        c_pair,
        n_head,
        kq_norm=True,
        n_attn_seq_neighbours=2,
        n_attn_keys=128,
    ):
        super().__init__()
        self.c = c_a  # d_model dim same as input features
        self.n_head = n_head

        self.to_q = linearNoBias(c_a, self.c)
        self.to_k = linearNoBias(c_a, self.c)
        self.to_v = linearNoBias(c_a, self.c)
        self.to_b = linearNoBias(c_pair, self.n_head)
        self.to_g = nn.Sequential(
            linearNoBias(c_a, self.c, bias=False),
            nn.Sigmoid(),
        )
        self.kq_norm = kq_norm
        if kq_norm:
            self.ln_q = RMSNorm(self.c)
            self.ln_k = RMSNorm(self.c)

        # Output / Input projections
        self.to_o = linearNoBias(self.c, c_a)  # from attn to Q_L

        # Conditioned
        if exists(c_s):
            self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
            self.linear_output_project = nn.Sequential(
                LinearBiasInit(c_s, c_a, biasinit=-2.0),
                nn.Sigmoid(),
            )
        else:
            self.ln_1 = RMSNorm(c_a)

        # Used if no indices are provided
        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_checkpointing = True

    def forward(
        self,
        Q_L,
        C_L,
        P_LL,
        indices=None,
        f=None,
        X_L=None,
        full=False,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
    ):
        """
        Q_L: [D, L, c_a]
        C_L: [D, L, c_s]
        P_LL: [D, L, L, c_pair] or None (if using chunked mode)
        indices: [D, L, k] long
        chunked_pairwise_embedder: ChunkedPairwiseEmbedder for memory efficient computation
        initializer_outputs: Dict containing features for chunked computation
        """

        # If no indices are provided, prepare indices from
        if not exists(indices):
            indices = create_attention_indices(
                f,
                n_attn_keys=self.n_attn_keys,
                n_attn_seq_neighbours=self.n_attn_seq_neighbours,
                X_L=X_L,
            )

        # Handle chunked P_LL computation
        if chunked_pairwise_embedder is not None and P_LL is None:
            # Compute sparse P_LL only for the attention indices
            P_LL_sparse = chunked_pairwise_embedder.forward_chunked(
                f=f,
                indices=indices,
                C_L=initializer_outputs["C_L"],
                Z_init_II=initializer_outputs["Z_II"],
                tok_idx=f["atom_to_token_map"],
            )
            # P_LL_sparse is already in sparse format [D, L, k, c_pair]
            use_sparse_pll = True
        else:
            # Original full P_LL computation
            P_LL_sparse = None
            use_sparse_pll = False

        use_kernel = False

        def do_attention(Q_L, C_L, P_LL):
            if exists(C_L):
                Q_L = self.ada_ln_1(Q_L, C_L)
            else:
                Q_L = self.ln_1(Q_L)

            if use_kernel and not use_sparse_pll:
                # TODO: Update with latest kernel
                q, k, v, g, b = (
                    self.to_q(Q_L),
                    self.to_k(Q_L),
                    self.to_v(Q_L),
                    self.to_g(Q_L),
                    self.to_b(P_LL),
                )
                q, k = (self.ln_q(q), self.ln_k(k)) if self.kq_norm else (q, k)
                attn_out = _fused_full_pairbias_attention(
                    Q_L=q,  # already projected queries (B, L, c)
                    K_L=k,
                    V_L=v,
                    P_LL=P_LL,  # pair features (B, L, L, c_pair)
                    num_heads=self.n_head,
                    to_b=None,  # pair-bias projector (H, c_pair)
                    to_g_linear=None,  # gating linear (D, D)
                    to_o_linear=None,  # output linear (D, D)
                    w_ln_z_identity=None,
                    b_ln_z_identity=None,
                    attn_scale=1.0 / math.sqrt(self.c // self.n_head),
                )
            else:
                # Sparse attention path
                q, k, v, g = (
                    self.to_q(Q_L),
                    self.to_k(Q_L),
                    self.to_v(Q_L),
                    self.to_g(Q_L),
                )
                q, k = (self.ln_q(q), self.ln_k(k)) if self.kq_norm else (q, k)

                if use_sparse_pll:
                    # Use pre-computed sparse P_LL (already gathered)
                    b = self.to_b(P_LL_sparse)  # [D, L, k, H]
                    attn_out = sparse_pairbias_attention(
                        Q=q,
                        K=k,
                        V=v,
                        B=b,
                        G=g,
                        gather_bias=False,  # Already gathered!
                        indices=indices,
                        H=self.n_head,
                        full=full,
                    )  # [D, L, c]
                else:
                    # Original full P_LL path
                    b = self.to_b(P_LL)
                    attn_out = sparse_pairbias_attention(
                        Q=q,
                        K=k,
                        V=v,
                        B=b,
                        G=g,
                        gather_bias=True,
                        indices=indices,
                        H=self.n_head,
                        full=full,
                    )  # [D, L, c]

            # Output projection (from adaLN-Zero)
            Q_L = self.to_o(attn_out)
            if exists(C_L):
                Q_L = self.linear_output_project(C_L) * Q_L

            return Q_L

        do_attention_ = (
            activation_checkpointing(do_attention)
            if self.use_checkpointing
            else do_attention
        )

        # Call attention with appropriate P_LL
        if use_sparse_pll:
            return do_attention_(Q_L, C_L, P_LL_sparse)
        else:
            return do_attention_(Q_L, C_L, P_LL)

    def forward_cross_attn(
        self,
        Q_chunk,           # [D, L_par, c_a] - queries from this GPU's chunk
        C_Q_chunk,         # [D, L_par, c_s] - conditioning for queries
        K_V_full,          # [D, L, c_a] - keys/values from all tokens
        C_KV_full,         # [D, L, c_s] - conditioning for K/V (unused, for signature)
        P_chunk,           # [L_par, L, c_pair] - pair bias for chunk×all
        indices_chunk,     # [D, L_par, k] - sparse attention indices for queries
    ):
        """
        Cross-attention: queries from a chunk attend to all keys.
        
        This avoids materializing [L, L] tensors by:
        - Q from chunk [L_par] only
        - K, V from all [L]
        - P_chunk is [L_par, L] not [L, L]
        
        Args:
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
            Q_normed = self.ada_ln_1(Q_chunk, C_Q_chunk)  # [D, L_par, c_a]
        else:
            Q_normed = self.ln_1(Q_chunk)
        
        # Normalize K/V - use full input for keys/values
        # Note: We apply ada_ln to full K/V but only to get normalized features
        # This doesn't create L×L tensors
        if exists(C_KV_full):
            KV_normed = self.ada_ln_1(K_V_full, C_KV_full)  # [D, L, c_a]
        else:
            KV_normed = self.ln_1(K_V_full)
        
        # Project to Q, K, V
        q = self.to_q(Q_normed)                            # [D, L_par, c]
        k = self.to_k(KV_normed)                           # [D, L, c]
        v = self.to_v(KV_normed)                           # [D, L, c]
        g = self.to_g(Q_normed)                            # [D, L_par, c]
        
        # Apply KQ norm if enabled
        if self.kq_norm:
            q = self.ln_q(q)                               # [D, L_par, c]
            k = self.ln_k(k)                               # [D, L, c]
        
        # Project pair bias: [L_par, L, c_pair] → [L_par, L, H]
        if P_chunk.ndim == 3:
            b = self.to_b(P_chunk)                         # [L_par, L, H]
        else:  # [D, L_par, L, c_pair]
            b = self.to_b(P_chunk)                         # [D, L_par, L, H]
        
        # Cross-attention with sparse indices
        attn_out = sparse_cross_attention(
            Q=q,                                           # [D, L_par, c]
            K=k,                                           # [D, L, c]
            V=v,                                           # [D, L, c]
            B=b,                                           # [L_par, L, H] or [D, L_par, L, H]
            G=g,                                           # [D, L_par, c]
            indices=indices_chunk,                         # [D, L_par, k]
            H=self.n_head,
        )                                                  # [D, L_par, c]
        
        # Output projection
        attn_out = self.to_o(attn_out)                     # [D, L_par, c_a]
        
        # Apply output gating with query conditioning
        if exists(C_Q_chunk):
            attn_out = self.linear_output_project(C_Q_chunk) * attn_out
        
        return attn_out                                    # [D, L_par, c_a]

    def forward_sparse_cross_attn(
        self,
        Q_chunk,             # [D, L_par, c_a] - queries from this GPU's chunk
        C_Q_chunk,           # [D, L_par, c_s] - conditioning for queries
        K_V_full,            # [D, L, c_a] - keys/values (full, for gathering)
        C_KV_full,           # [D, L, c_s] - conditioning for K/V
        P_sparse_chunk,      # [D, L_par, k, c_pair] - SPARSE pair bias
        indices_chunk,       # [D, L_par, k] - which keys each query attends to
    ):
        """
        Sparse cross-attention: queries attend to k sparse neighbors.
        
        Combines two optimizations:
        1. Cross-attention: L_par queries (chunked across GPUs)
        2. Sparse attention: k neighbors per query (not full L)
        
        Memory: O(L_par * k) instead of O(L_par * L) or O(L²)
        
        P_sparse_chunk is [L_par, k] - already indexed to the k neighbors,
        not a dense [L_par, L] matrix.
        
        Args:
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
            Q_normed = self.ada_ln_1(Q_chunk, C_Q_chunk)  # [D, L_par, c_a]
        else:
            Q_normed = self.ln_1(Q_chunk)
        
        # Normalize K/V - use full input for keys/values
        if exists(C_KV_full):
            KV_normed = self.ada_ln_1(K_V_full, C_KV_full)  # [D, L, c_a]
        else:
            KV_normed = self.ln_1(K_V_full)
        
        # Project to Q, K, V
        q = self.to_q(Q_normed)                            # [D, L_par, c]
        k_proj = self.to_k(KV_normed)                      # [D, L, c]
        v = self.to_v(KV_normed)                           # [D, L, c]
        g = self.to_g(Q_normed)                            # [D, L_par, c]
        
        # Apply KQ norm if enabled
        if self.kq_norm:
            q = self.ln_q(q)                               # [D, L_par, c]
            k_proj = self.ln_k(k_proj)                     # [D, L, c]
        
        # Project pair bias: [D, L_par, k, c_pair] → [D, L_par, k, H]
        # NOTE: P is already sparse (k neighbors), not dense [L_par, L]
        b_sparse = self.to_b(P_sparse_chunk)               # [D, L_par, k, H]
        
        # Sparse cross-attention: gather K/V using indices, use pre-gathered bias
        attn_out = sparse_cross_attention_pregathered_bias(
            Q=q,                                           # [D, L_par, c]
            K=k_proj,                                      # [D, L, c]
            V=v,                                           # [D, L, c]
            B_sparse=b_sparse,                             # [D, L_par, k, H]
            G=g,                                           # [D, L_par, c]
            indices=indices_chunk,                         # [D, L_par, k]
            H=self.n_head,
        )                                                  # [D, L_par, c]
        
        # Output projection
        attn_out = self.to_o(attn_out)                     # [D, L_par, c_a]
        
        # Apply output gating with query conditioning
        if exists(C_Q_chunk):
            attn_out = self.linear_output_project(C_Q_chunk) * attn_out
        
        return attn_out                                    # [D, L_par, c_a]


######################################################################################
##########################        Kernel Functions          ##########################
######################################################################################


def sparse_pairbias_attention(
    Q, K, V, B, indices, H, gather_bias=True, G=None, full=False
):
    """
    Computes attention with sparse pairwise bias, where indices specify which
    keys to attend to for each query token.
    Q: (D, L, c)  # query vectors
    K: (D, L, c)  # key vectors
    V: (D, L, c)  # value vectors
    B: (L, L, H)  # attention bias (unbatched or pre-gathered and [D, L, k, H])
    G: (D, L, c)  # Gate (optional)
    B2: (D, L, 14, 14, H)  # attention bias (batched and within token) (optional)
    indices: (D, L, k_neigh) long  # indices of neighbours to attend to
    Returns
    -------
    attn_out: (D, L, c)  # attention output
    """
    D, L, c = Q.shape
    k = indices.shape[-1]  # k_neigh

    if full:
        # During training, compute full attention matrix to create a more optimized torch.tensor graph.
        return pairbias_attention_(
            Q=Q,
            K=K,
            V=V,
            B=B,
            H=H,
            valid_mask=indices_to_mask(indices),
            G=G,
        )

    # Pull vectors from dimension 1 into index torch.tensor according to unique k_neigh axis
    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)  # (D,1,1)
    K_gathered = K[batch_idx, indices].contiguous()  # (D, L, k, c)
    V_gathered = V[batch_idx, indices].contiguous()  # (D, L, k, c)

    # Gather bias or assume pre-gathered
    if gather_bias:
        query_idx = torch.arange(L, device=Q.device).view(1, L, 1)  # (1,L,1)
        query_idx = query_idx.expand(D, -1, k)
        if B.ndim == 3:
            B_gathered = B[query_idx, indices, :]  # (D, L, k, H)
        elif B.ndim == 4:  # (D, L, L, H)
            B_gathered = B[batch_idx, query_idx, indices, :]  # (D, L, k, H)
    else:
        assert B.shape == (D, L, k, H), "B must be batched with shape (D, L, k, H)"
        B_gathered = B
    B_gathered = B_gathered.contiguous()

    # Split into heads
    Q = Q.reshape(D, L, H, c // H)
    K_gathered = K_gathered.reshape(D, L, k, H, c // H)
    V_gathered = V_gathered.reshape(D, L, k, H, c // H)
    B_gathered = B_gathered.reshape(D, L, k, H)
    Q = Q.permute(0, 2, 1, 3)  # [D, H, L, c // H]
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)
    B_gathered = B_gathered.permute(0, 3, 1, 2)

    # Do attention
    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)
    attn = attn / sqrt(c // H)  # scale
    attn = attn + B_gathered  # add bias
    attn = torch.softmax(attn, dim=-1)  # softmax over keys [D, H, L, k]
    attn_out = torch.einsum(
        "...ij,...ijc->...ic", attn, V_gathered
    )  # allocates a max of 4.95 GiB.

    # Optional gating
    if G is not None:
        G = G.reshape(D, L, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L, c).contiguous()

    return attn_out  # [D, L, c]


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
    k = indices.shape[-1]  # number of sparse keys per query
    
    # Gather K, V using sparse indices
    # indices: [D, L_par, k] - for each query, which keys to attend to
    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)  # [D, 1, 1]
    K_gathered = K[batch_idx, indices].contiguous()              # [D, L_par, k, c]
    V_gathered = V[batch_idx, indices].contiguous()              # [D, L_par, k, c]
    
    # Gather bias
    # B is [L_par, L, H] or [D, L_par, L, H]
    # We need B for (query q, key indices[q]) pairs → [D, L_par, k, H]
    query_idx = torch.arange(L_par, device=Q.device).view(1, L_par, 1)  # [1, L_par, 1]
    query_idx = query_idx.expand(D, -1, k)                               # [D, L_par, k]
    
    if B.ndim == 3:  # [L_par, L, H]
        # For each query q (0..L_par-1), gather B[q, indices[q], :]
        B_gathered = B[query_idx, indices, :]                    # [D, L_par, k, H]
    elif B.ndim == 4:  # [D, L_par, L, H]
        B_gathered = B[batch_idx, query_idx, indices, :]         # [D, L_par, k, H]
    else:
        raise ValueError(f"B must have 3 or 4 dims, got {B.ndim}")
    B_gathered = B_gathered.contiguous()
    
    # Debug (rank 0 only, once per call site, only if stats enabled)
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
    Q = Q.reshape(D, L_par, H, c // H)                           # [D, L_par, H, c//H]
    K_gathered = K_gathered.reshape(D, L_par, k, H, c // H)      # [D, L_par, k, H, c//H]
    V_gathered = V_gathered.reshape(D, L_par, k, H, c // H)      # [D, L_par, k, H, c//H]
    B_gathered = B_gathered.reshape(D, L_par, k, H)              # [D, L_par, k, H]
    
    # Permute to [D, H, L_par, ...] for attention
    Q = Q.permute(0, 2, 1, 3)                                    # [D, H, L_par, c//H]
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)               # [D, H, L_par, k, c//H]
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)               # [D, H, L_par, k, c//H]
    B_gathered = B_gathered.permute(0, 3, 1, 2)                  # [D, H, L_par, k]
    
    # Attention: Q @ K^T
    # Q: [D, H, L_par, c//H], K: [D, H, L_par, k, c//H]
    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)    # [D, H, L_par, k]
    attn = attn / sqrt(c // H)                                   # scale
    attn = attn + B_gathered                                     # add bias
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
    attn = torch.softmax(attn, dim=-1)                           # [D, H, L_par, k]
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
    # attn: [D, H, L_par, k], V: [D, H, L_par, k, c//H]
    attn_out = torch.einsum("...ij,...ijc->...ic", attn, V_gathered)  # [D, H, L_par, c//H]
    
    # Optional gating
    if G is not None:
        G = G.reshape(D, L_par, H, c // H).permute(0, 2, 1, 3)   # [D, H, L_par, c//H]
        attn_out = attn_out * G
    
    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)                      # [D, L_par, H, c//H]
    attn_out = attn_out.reshape(D, L_par, c).contiguous()        # [D, L_par, c]
    
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
    batch_idx = torch.arange(D, device=Q.device).view(-1, 1, 1)  # [D, 1, 1]
    K_gathered = K[batch_idx, indices].contiguous()              # [D, L_par, k, c]
    V_gathered = V[batch_idx, indices].contiguous()              # [D, L_par, k, c]
    
    # B_sparse is already [D, L_par, k, H] - no gathering needed!
    
    # Split into heads
    Q = Q.reshape(D, L_par, H, c // H)                           # [D, L_par, H, c//H]
    K_gathered = K_gathered.reshape(D, L_par, k, H, c // H)      # [D, L_par, k, H, c//H]
    V_gathered = V_gathered.reshape(D, L_par, k, H, c // H)      # [D, L_par, k, H, c//H]
    B_sparse = B_sparse.reshape(D, L_par, k, H)                  # [D, L_par, k, H]
    
    # Permute to [D, H, L_par, ...] for attention
    Q = Q.permute(0, 2, 1, 3)                                    # [D, H, L_par, c//H]
    K_gathered = K_gathered.permute(0, 3, 1, 2, 4)               # [D, H, L_par, k, c//H]
    V_gathered = V_gathered.permute(0, 3, 1, 2, 4)               # [D, H, L_par, k, c//H]
    B_sparse = B_sparse.permute(0, 3, 1, 2)                      # [D, H, L_par, k]
    
    # Attention: Q @ K^T
    attn = torch.einsum("...ld,...lkd->...lk", Q, K_gathered)    # [D, H, L_par, k]
    attn = attn / sqrt(c // H)                                   # scale
    attn = attn + B_sparse                                       # add bias
    attn = torch.softmax(attn, dim=-1)                           # [D, H, L_par, k]
    
    # Apply attention to values
    attn_out = torch.einsum("...ij,...ijc->...ic", attn, V_gathered)  # [D, H, L_par, c//H]
    
    # Optional gating
    if G is not None:
        G = G.reshape(D, L_par, H, c // H).permute(0, 2, 1, 3)   # [D, H, L_par, c//H]
        attn_out = attn_out * G
    
    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)                      # [D, L_par, H, c//H]
    attn_out = attn_out.reshape(D, L_par, c).contiguous()        # [D, L_par, c]
    
    return attn_out


def pairbias_attention_(Q, K, V, B, H, valid_mask=None, G=None):
    """
    Fully connected variant of pairbias attention with optional gating and valid mask.
    Equivalent to sparse attention but with all keys

    Attn_out: [batch_size, query_length, H * head_dim]
    """
    D, L, c = Q.shape
    k = L

    # Split into heads
    Q = Q.reshape(D, L, H, c // H)
    K = K.reshape(D, k, H, c // H)
    V = V.reshape(D, k, H, c // H)
    B = B.reshape(D, L, k, H)

    # Flip heads upwards [..., H, d_model] -> [B, H, ..., d_model]
    Q = Q.permute(0, 2, 1, 3)  # [D, H, L, c // H]
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    B = B.permute(0, 3, 1, 2)

    # Do attention
    attn = torch.einsum("...ld,...kd->...lk", Q, K)
    attn = attn / sqrt(c // H)  # scale
    attn = attn + B  # add bias
    if exists(valid_mask):
        # expand valid mask over heads [D, H, L, L]
        attn = attn.masked_fill(~valid_mask.unsqueeze(1), float("-inf"))
    attn = torch.softmax(attn, dim=-1)  # softmax over keys [D, H, L, k]
    attn_out = torch.einsum("...ij,...jc->...ic", attn, V)

    # Optional gating
    if G is not None:
        G = G.reshape(D, L, H, c // H).permute(0, 2, 1, 3)
        attn_out = attn_out * G

    # Merge heads
    attn_out = attn_out.permute(0, 2, 1, 3)
    attn_out = attn_out.reshape(D, L, c).contiguous()

    return attn_out


def _fused_full_pairbias_attention(
    *,
    Q_L,  # (B, L, c)  -- sequence features used to make q,k,v and for gating
    K_L,  # (B, L, c)
    V_L,  # (B, L, c)
    P_LL,  # (B, L, L, c_pair)
    num_heads: int,
    to_b: nn.Linear,  # projects pair features -> heads (H)
    to_g_linear: nn.Linear,  # weight (D, D), bias optional/None (pre-sigmoid, kernel handles gate)
    to_o_linear: nn.Linear,  # weight (D, D), bias optional/None (kernel handles output proj)
    w_ln_z_identity: torch.torch.Tensor,  # (c_pair,)
    b_ln_z_identity: torch.torch.Tensor,  # (c_pair,)
    attn_scale: float | None = None,
):
    """
    Uses cuequivariance_torch.attention_pair_bias for dense (full) attention.
    Expects Q/K/V to be projected *before* calling this function.
    """
    B, L, c = Q_L.shape
    H = num_heads
    assert c % H == 0, "Model dim must be divisible by num_heads"
    DH = c // H

    # q, k, v as (B, H, L, DH)
    q = Q_L.reshape(B, L, H, DH).permute(0, 2, 1, 3).contiguous()
    k = K_L.reshape(B, L, H, DH).permute(0, 2, 1, 3).contiguous()
    v = V_L.reshape(B, L, H, DH).permute(0, 2, 1, 3).contiguous()

    # s is the sequence features for gating/output projections
    s = Q_L.contiguous()  # (B, L, c)

    # mask: None (kernel supports key padding mask shape (B,V) or (B*M,V); we don't need it here)
    mask = None

    # weights/biases for kernel (shapes per doc):
    # w_proj_z: (H, z_dim)
    w_proj_z = to_b.weight  # (H, c_pair)
    b_proj_z = to_b.bias if hasattr(to_b, "bias") else None

    # w_proj_g / o: (D, D)
    w_proj_g = to_g_linear.weight  # (D, D)
    b_proj_g = to_g_linear.bias if hasattr(to_g_linear, "bias") else None

    w_proj_o = to_o_linear.weight  # (D, D)
    b_proj_o = to_o_linear.bias if hasattr(to_o_linear, "bias") else None

    # z-LN params
    w_ln_z = w_ln_z_identity.to(dtype=P_LL.dtype, device=P_LL.device)
    b_ln_z = b_ln_z_identity.to(dtype=P_LL.dtype, device=P_LL.device)

    # optional scaling (match your manual path)
    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(DH)

    # Call the fused kernel (B*M collapses to B here; multiplicity=1)
    out, _proj_z = cueq_attention_pair_bias(
        s=s,
        q=q,
        k=k,
        v=v,
        z=P_LL,
        mask=mask,
        num_heads=H,
        w_proj_z=w_proj_z,
        w_proj_g=w_proj_g,
        w_proj_o=w_proj_o,
        w_ln_z=w_ln_z,
        b_ln_z=b_ln_z,
        b_proj_z=b_proj_z,
        b_proj_g=b_proj_g,
        b_proj_o=b_proj_o,
        attn_scale=attn_scale,
        compute_pair_bias=True,
        multiplicity=1,
    )
    # out: (B, L, c) already gated & projected
    return out