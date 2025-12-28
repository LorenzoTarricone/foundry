import os
import torch
from rfd3.model.layers.layer_utils import (
    MultiDimLinear,
    RMSNorm,
    Transition,
    linearNoBias,
)
from torch import nn

from foundry.training.checkpoint import activation_checkpointing
from foundry.utils.torch import device_of


class AttentionPairBiasPairformerDeepspeed(nn.Module):
    def __init__(self, c_a, c_s, c_pair, n_head, kq_norm=False):
        super().__init__()
        self.n_head = n_head
        self.c_a = c_a
        self.c_pair = c_pair
        self.c = c_a // n_head

        self.to_q = MultiDimLinear(c_a, (n_head, self.c))
        self.to_k = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_v = MultiDimLinear(c_a, (n_head, self.c), bias=False, norm=kq_norm)
        self.to_b = linearNoBias(c_pair, n_head)
        self.to_g = nn.Sequential(
            MultiDimLinear(c_a, (n_head, self.c), bias=False),
            nn.Sigmoid(),
        )
        self.to_a = linearNoBias(c_a, c_a)
        # self.linear_output_project = nn.Sequential(
        # LinearBiasInit(c_s, c_a, biasinit=-2.),
        # nn.Sigmoid(),
        # )
        self.ln_0 = RMSNorm((c_pair,))
        # self.ada_ln_1 = AdaLN(c_a=c_a, c_s=c_s)
        self.ln_1 = RMSNorm((c_a,))
        self.use_deepspeed_evo = False
        self.force_bfloat16 = True
        # Optional streaming mode: split queries into chunks to avoid full I×I materialization
        self.attn_parallel = os.environ.get("RFD3_ATTENTION_PARALLEL", None)

    def forward(
        self,
        A_I,  # [I, C_a]
        S_I,  # [I, C_a] | None
        Z_II,  # [I, I, C_z]
        Beta_II=None,  # [I, I]
    ):
        # Input projections
        assert S_I is None
        A_I = self.ln_1(A_I)

        if self.use_deepspeed_evo or self.force_bfloat16:
            A_I = A_I.to(torch.bfloat16)

        Q_IH = self.to_q(A_I)  # / np.sqrt(self.c)
        K_IH = self.to_k(A_I)
        V_IH = self.to_v(A_I)
        B_IIH = self.to_b(self.ln_0(Z_II)) + Beta_II[..., None]
        G_IH = self.to_g(A_I)

        B, L = B_IIH.shape[:2]

        if not self.use_deepspeed_evo or L <= 24:
            Q_IH = Q_IH / torch.sqrt(
                torch.tensor(self.c).to(Q_IH.device, torch.bfloat16)
            )

            # Streaming attention over queries to avoid full I×I
            if self.attn_parallel is not None:
                n_par = max(int(self.attn_parallel), 1)
                chunk = max(1, (L + n_par - 1) // n_par)
            else:
                chunk = L  # original behavior

            outputs = []
            for i_start in range(0, L, chunk):
                i_end = min(i_start + chunk, L)
                Q_chunk = Q_IH[..., i_start:i_end, :, :]  # [..., Q, H, C]
                B_chunk = B_IIH[..., i_start:i_end, :, :]  # [..., Q, L, H]

                attn = torch.softmax(
                    torch.einsum("...qhd,...jhd->...qjh", Q_chunk, K_IH) + B_chunk,
                    dim=-2,
                )  # [..., Q, L, H]
                out = torch.einsum("...qjh,...jhc->...qhc", attn, V_IH)
                out = G_IH[..., i_start:i_end, :, :] * out
                outputs.append(out)

            A_I = torch.cat(outputs, dim=-3)  # [..., I, H, C]
            A_I = A_I.flatten(start_dim=-2)  # [B, I, Ca]
        else:
            raise NotImplementedError

        A_I = self.to_a(A_I)

        return A_I

    def forward_chunked(
        self,
        A_I_query: torch.Tensor,   # [I_par, C_a] - query tokens (this GPU's chunk)
        A_I_key: torch.Tensor,     # [I, C_a] - key tokens (all)
        Z_chunk: torch.Tensor,     # [I_par, I, C_z] - pair bias for query rows
        Beta_II: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Chunked cross-attention for multi-GPU parallel inference.
        
        Each GPU computes attention for its query chunk against ALL keys.
        Mimics standard self-attention but with Q from chunk, K/V from all.
        
        Args:
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
        A_I_query_normed = self.ln_1(A_I_query)            # [I_par, C_a]
        A_I_key_normed = self.ln_1(A_I_key)                # [I, C_a]
        
        if self.force_bfloat16:
            A_I_query_normed = A_I_query_normed.to(torch.bfloat16)
            A_I_key_normed = A_I_key_normed.to(torch.bfloat16)
        
        # Q from chunk, K/V from all (cross-attention pattern)
        Q_chunk = self.to_q(A_I_query_normed)              # [I_par, H, C]
        G_chunk = self.to_g(A_I_query_normed)              # [I_par, H, C]
        K_all = self.to_k(A_I_key_normed)                  # [I, H, C]
        V_all = self.to_v(A_I_key_normed)                  # [I, H, C]
        
        # Pair bias from Z chunk
        B_chunk = self.to_b(self.ln_0(Z_chunk))            # [I_par, I, H]
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
        # Using same einsum pattern as standard but with different Q/K sources
        attn_logits = torch.einsum("qhd,khd->qkh", Q_chunk, K_all) + B_chunk  # [I_par, I, H]
        attn_weights = torch.softmax(attn_logits, dim=1)   # softmax over keys
        
        # Weighted sum of values
        out = torch.einsum("qkh,khc->qhc", attn_weights, V_all)  # [I_par, H, C]
        
        # Gating and output projection (same as standard)
        out = G_chunk * out
        out = out.flatten(start_dim=-2)                    # [I_par, C_a]
        out = self.to_a(out)
        
        return out


class PairformerBlock(nn.Module):
    """
    Attempt to replicate AF3 architecture from scratch.
    """

    def __init__(
        self,
        c_s,
        c_z,
        attention_pair_bias,
        p_drop=0.1,
        triangle_multiplication=None,
        triangle_attention=None,
        n_transition=4,
        use_deepspeed_evo=True,
        use_triangle_mult=False,
        use_triangle_attn=False,
    ):
        super().__init__()

        # self.drop_row = Dropout(broadcast_dim=-2, p_drop=p_drop)
        # self.drop_col = Dropout(broadcast_dim=-3, p_drop=p_drop)

        self.z_transition = Transition(c=c_z, n=n_transition)

        if c_s > 0:
            self.s_transition = Transition(c=c_s, n=n_transition)

            self.attention_pair_bias = AttentionPairBiasPairformerDeepspeed(
                c_a=c_s, c_s=0, c_pair=c_z, **attention_pair_bias
            )

    @activation_checkpointing
    def forward(self, S_I, Z_II):
        with torch.amp.autocast(
            device_type=device_of(self).type, enabled=True, dtype=torch.bfloat16
        ):
            Z_II = Z_II + self.z_transition(Z_II)
            if S_I is not None:
                S_I = S_I + self.attention_pair_bias(
                    S_I, None, Z_II, Beta_II=torch.tensor([0.0], device=Z_II.device)
                )
                S_I = S_I + self.s_transition(S_I)
        return S_I, Z_II
