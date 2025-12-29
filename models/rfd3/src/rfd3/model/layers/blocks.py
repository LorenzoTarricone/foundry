import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from einops import rearrange
from rfd3.model.layers.attention import (
    GatedCrossAttention,
    LocalAttentionPairBias,
)
from rfd3.model.layers.block_utils import (
    build_valid_mask,
    create_attention_indices,
    group_atoms,
    ungroup_atoms,
)
from rfd3.model.layers.layer_utils import (
    AdaLN,
    EmbeddingLayer,
    LinearBiasInit,
    RMSNorm,
    Transition,
    collapse,
    linearNoBias,
)
from rfd3.model.layers.pairformer_layers import PairformerBlock
from torch.nn.functional import one_hot

from foundry import DISABLE_CHECKPOINTING
from foundry.common import exists

logger = logging.getLogger(__name__)


# SwiGLU transition block with adaptive layernorm
class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_token, c_s, n=2):
        super().__init__()
        self.ada_ln = AdaLN(c_a=c_token, c_s=c_s)
        self.linear_1 = linearNoBias(c_token, c_token * n)
        self.linear_2 = linearNoBias(c_token, c_token * n)
        self.linear_output_project = nn.Sequential(
            LinearBiasInit(c_s, c_token, biasinit=-2.0),
            nn.Sigmoid(),
        )
        self.linear_3 = linearNoBias(c_token * n, c_token)

    def forward(
        self,
        Ai,  # [B, I, C_token]
        Si,  # [B, I, C_token]
    ):
        Ai = self.ada_ln(Ai, Si)
        # BUG: This is not the correct implementation of SwiGLU
        # Bi = torch.sigmoid(self.linear_1(Ai)) * self.linear_2(Ai)
        # FIX: This is the correct implementation of SwiGLU
        Bi = torch.nn.functional.silu(self.linear_1(Ai)) * self.linear_2(Ai)

        # Output projection (from adaLN-Zero)
        return self.linear_output_project(Si) * self.linear_3(Bi)


class PositionPairDistEmbedder(nn.Module):
    """
    Embeds pairwise position differences into pair features.
    
    Supports both full and chunked (streaming) forward passes:
    - forward(): Full [I, I, c] or [L, L, c] output
    - forward_chunk(): Chunked [I_par, I, c] output for streaming mode
    """
    
    def __init__(self, c_atompair, embed_frame=True):
        super().__init__()
        self.embed_frame = embed_frame
        if embed_frame:
            self.process_d = linearNoBias(3, c_atompair)

        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward_af3(self, D_LL, V_LL):
        """Forward the same way reference positions are embeded in AF3"""

        P_LL = self.process_d(D_LL) * V_LL

        # Embed pairwise inverse squared distances, and the valid mask
        if self.training:
            P_LL = (
                P_LL
                + self.process_inverse_dist(
                    1 / (1 + torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2)
                )
                * V_LL
            )
            P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        else:
            P_LL[V_LL[..., 0]] += self.process_inverse_dist(
                1
                / (1 + torch.linalg.norm(D_LL[V_LL[..., 0]], dim=-1, keepdim=True) ** 2)
            )
            P_LL[V_LL[..., 0]] += self.process_valid_mask(
                V_LL[V_LL[..., 0]].to(P_LL.dtype)
            )
        return P_LL

    def forward(self, ref_pos, valid_mask):
        """
        Full forward pass.
        
        Args:
            ref_pos: [I, 3] or [L, 3] reference positions
            valid_mask: [I, I, 1] or [L, L, 1] validity mask
            
        Returns:
            P_LL: [I, I, c] or [L, L, c] pair embeddings
        """
        D_LL = ref_pos.unsqueeze(-2) - ref_pos.unsqueeze(-3)  # [I, I, 3]
        V_LL = valid_mask  # [I, I, 1]

        if self.embed_frame:
            # Embed pairwise distances
            return self.forward_af3(D_LL, V_LL)
        norm = torch.linalg.norm(D_LL, dim=-1, keepdim=True) ** 2
        norm = torch.clamp(norm, min=1e-6)
        inv_dist = 1 / (1 + norm)
        P_LL = self.process_inverse_dist(inv_dist) * V_LL
        P_LL = P_LL + self.process_valid_mask(V_LL.to(P_LL.dtype)) * V_LL
        return P_LL

    def forward_chunk(
        self, 
        ref_pos_query: torch.Tensor,  # [I_par, 3] query positions
        ref_pos_key: torch.Tensor,    # [I, 3] all key positions
        valid_mask: torch.Tensor,     # [I_par, I, 1] validity mask for this chunk
    ) -> torch.Tensor:
        """
        Chunked forward pass for streaming mode.
        
        Computes pair embeddings for a subset of query positions against all key positions.
        Never materializes full [I, I, c] tensor.
        
        Args:
            ref_pos_query: [I_par, 3] query reference positions
            ref_pos_key: [I, 3] key reference positions (all tokens)
            valid_mask: [I_par, I, 1] validity mask
            
        Returns:
            P_chunk: [I_par, I, c] pair embeddings for query chunk
        """
        # Compute pairwise differences: [I_par, I, 3]
        D_chunk = ref_pos_query.unsqueeze(-2) - ref_pos_key.unsqueeze(-3)  # [I_par, I, 3]
        V_chunk = valid_mask  # [I_par, I, 1]
        
        if self.embed_frame:
            # Embed using AF3 style
            P_chunk = self.process_d(D_chunk) * V_chunk  # [I_par, I, c]
            
            norm = torch.linalg.norm(D_chunk, dim=-1, keepdim=True) ** 2  # [I_par, I, 1]
            norm = torch.clamp(norm, min=1e-6)
            inv_dist = 1 / (1 + norm)  # [I_par, I, 1]
            
            P_chunk = P_chunk + self.process_inverse_dist(inv_dist) * V_chunk
            P_chunk = P_chunk + self.process_valid_mask(V_chunk.to(P_chunk.dtype)) * V_chunk
        else:
            # Simpler embedding without frame
            norm = torch.linalg.norm(D_chunk, dim=-1, keepdim=True) ** 2
            norm = torch.clamp(norm, min=1e-6)
            inv_dist = 1 / (1 + norm)
            P_chunk = self.process_inverse_dist(inv_dist) * V_chunk
            P_chunk = P_chunk + self.process_valid_mask(V_chunk.to(P_chunk.dtype)) * V_chunk
        
        return P_chunk  # [I_par, I, c]


class OneDFeatureEmbedder(nn.Module):
    """
    Embeds 1D features into a single vector.

    Args:
        features (dict): Dictionary of feature names and their number of channels.
        output_channels (int): Output dimension of the projected embedding.
    """

    def __init__(self, features, output_channels):
        super().__init__()
        self.features = {k: v for k, v in features.items() if exists(v)}
        total_embedding_input_features = sum(self.features.values())
        self.embedders = nn.ModuleDict(
            {
                feature: EmbeddingLayer(
                    n_channels, total_embedding_input_features, output_channels
                )
                for feature, n_channels in self.features.items()
            }
        )

    def forward(self, f, collapse_length):
        return sum(
            tuple(
                self.embedders[feature](collapse(f[feature].float(), collapse_length))
                for feature, n_channels in self.features.items()
                if exists(n_channels)
            )
        )


class SinusoidalDistEmbed(nn.Module):
    """
    Applies sinusoidal embedding to pairwise distances and projects to c_atompair.

    Supports both full and chunked (streaming) forward passes:
    - forward(): Full [L, L, c] output
    - forward_chunk(): Chunked [L_par, L, c] output for streaming mode

    Args:
        c_atompair (int): Output dimension of the projected embedding (must be even).
    """

    def __init__(self, c_atompair, n_freqs=32):
        super().__init__()
        assert c_atompair % 2 == 0, "Output embedding dim must be even"

        self.n_freqs = (
            n_freqs  # Number of sin/cos pairs → total sinusoidal dim = 2 * n_freqs
        )
        self.c_atompair = c_atompair

        self.output_proj = linearNoBias(2 * n_freqs, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def forward(self, pos, valid_mask):
        """
        Full forward pass.
        
        Args:
            pos: [L, 3] or [B, L, 3] ground truth atom positions
            valid_mask: [L, L, 1] or [B, L, L, 1] boolean mask
        Returns:
            P_LL: [L, L, c_atompair] or [B, L, L, c_atompair]
        """
        # Compute pairwise distances
        D_LL = pos.unsqueeze(-2) - pos.unsqueeze(-3)  # [L, L, 3] or [B, L, L, 3]
        dist_matrix = torch.linalg.norm(D_LL, dim=-1)  # [L, L] or [B, L, L]

        # Sinusoidal embedding
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32)
            / half_dim
        ).to(dist_matrix.device)  # [n_freqs]

        angles = dist_matrix.unsqueeze(-1) * freq  # [..., D/2]
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [..., D]

        # Linear projection
        P_LL = self.output_proj(sincos_embed)  # [..., c_atompair]
        P_LL = P_LL * valid_mask

        # Add linear embedding of valid mask
        P_LL = P_LL + self.process_valid_mask(valid_mask.to(P_LL.dtype)) * valid_mask
        return P_LL

    def forward_chunk(
        self, 
        pos_query: torch.Tensor,   # [B, I_par, 3] query positions
        pos_key: torch.Tensor,     # [B, I, 3] all key positions
        valid_mask: torch.Tensor,  # [I_par, I, 1] validity mask
    ) -> torch.Tensor:
        """
        Chunked forward pass for streaming mode.
        
        Computes sinusoidal distance embeddings for a subset of query positions
        against all key positions. Never materializes full [I, I, c] tensor.
        
        Args:
            pos_query: [B, I_par, 3] query positions
            pos_key: [B, I, 3] key positions (all tokens)
            valid_mask: [I_par, I, 1] validity mask
            
        Returns:
            P_chunk: [B, I_par, I, c_atompair] embeddings for query chunk
        """
        # Compute pairwise distances for chunk: [B, I_par, I, 3]
        D_chunk = pos_query.unsqueeze(-2) - pos_key.unsqueeze(-3)  # [B, I_par, I, 3]
        dist_matrix = torch.linalg.norm(D_chunk, dim=-1)           # [B, I_par, I]
        
        # Sinusoidal embedding
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32)
            / half_dim
        ).to(dist_matrix.device)  # [n_freqs]
        
        angles = dist_matrix.unsqueeze(-1) * freq  # [B, I_par, I, n_freqs]
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [B, I_par, I, 2*n_freqs]
        
        # Linear projection
        P_chunk = self.output_proj(sincos_embed)   # [B, I_par, I, c_atompair]
        
        # Apply mask (broadcast over batch)
        valid_mask_expanded = valid_mask.unsqueeze(0)  # [1, I_par, I, 1]
        P_chunk = P_chunk * valid_mask_expanded
        
        # Add linear embedding of valid mask
        P_chunk = P_chunk + self.process_valid_mask(
            valid_mask_expanded.to(P_chunk.dtype)
        ) * valid_mask_expanded
        
        return P_chunk  # [B, I_par, I, c_atompair]


class LinearEmbedWithPool(nn.Module):
    def __init__(self, c_token):
        super().__init__()
        self.c_token = c_token
        self.linear = linearNoBias(3, c_token)

    def forward(self, R_L, tok_idx):
        B = R_L.shape[0]
        I = int(tok_idx.max().item()) + 1
        A_I_shape = (
            B,
            I,
            self.c_token,
        )
        Q_L = self.linear(R_L)
        A_I = (
            torch.zeros(A_I_shape, device=R_L.device, dtype=Q_L.dtype)
            .index_reduce(
                -2,
                tok_idx.long(),
                Q_L,
                "mean",
                include_self=False,
            )
            .clone()
        )
        return A_I


class SimpleRecycler(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        template_embedder,
        msa_module,
        n_pairformer_blocks,
        pairformer_block,
    ):
        super().__init__()
        self.c_z = c_z
        self.process_zh = nn.Sequential(
            RMSNorm(c_z),
            linearNoBias(c_z, c_z),
        )
        self.process_sh = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )
        # Templates and msa's removed:
        # self.template_embedder = TemplateEmbedder(c_z=c_z, **template_embedder)
        # self.msa_module = MSAModule(**msa_module)

    def forward(
        self,
        f,
        S_inputs_I,
        S_init_I,
        Z_init_II,
        S_I,
        Z_II,
    ):
        Z_II = Z_init_II + self.process_zh(Z_II)

        # Templates and msa's removed:
        # Z_II = Z_II + self.template_embedder(f, Z_II)
        # Z_II = self.msa_module(f, Z_II, S_inputs_I)

        S_I = S_init_I + self.process_sh(S_I)
        for block in self.pairformer_stack:
            S_I, Z_II = block(S_I, Z_II)
        return S_I, Z_II


class RelativePositionEncodingWithIndexRemoval(nn.Module):
    """
    Usual RPE but utilizes `is_motif_atom_3d_unindexed` to ensure within-chain position is spoofed.
    """

    def __init__(self, r_max, s_max, c_z):
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z

        self.num_tok_pos_bins = (
            2 * self.r_max + 2
        ) + 1  # original af3 + 1 for unknown index
        self.linear = linearNoBias(
            2 * self.num_tok_pos_bins + (2 * self.s_max + 2) + 1, c_z
        )
        self.chunk_size = None
        # If attention parallel is enabled, stream RPE computation to avoid full LxL materialization
        attn_parallel = os.environ.get("RFD3_ATTENTION_PARALLEL", None)
        if attn_parallel is not None:
            n_parallel = max(int(attn_parallel), 1)
            # will be set at runtime based on L to keep chunks ~L/n_parallel
            self.chunk_size = n_parallel

    def forward(self, f):
        if self.chunk_size is None:
            return self._forward_full(f)
        else:
            return self._forward_chunked(f)

    def _forward_full(self, f):
        b_samechain_II = f["asym_id"].unsqueeze(-1) == f["asym_id"].unsqueeze(-2)
        b_same_entity_II = f["entity_id"].unsqueeze(-1) == f["entity_id"].unsqueeze(-2)
        d_residue_II = torch.where(
            b_samechain_II,
            torch.clip(
                f["residue_index"].unsqueeze(-1)
                - f["residue_index"].unsqueeze(-2)
                + self.r_max,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )
        b_sameresidue_II = f["residue_index"].unsqueeze(-1) == f[
            "residue_index"
        ].unsqueeze(-2)
        tok_distance = (
            f["token_index"].unsqueeze(-1) - f["token_index"].unsqueeze(-2) + self.r_max
        )
        d_token_II = torch.where(
            b_samechain_II * b_sameresidue_II,
            torch.clip(
                tok_distance,
                0,
                2 * self.r_max,
            ),
            2 * self.r_max + 1,
        )

        # Chain distances are kept
        d_chain_II = torch.where(
            # NOTE: Implementing bugfix from the Protenix Technical report, where we use `same_entity` instead of `not same_chain` (as in the AF-3 pseudocode)
            # Reference: https://github.com/bytedance/Protenix/blob/main/Protenix_Technical_Report.pdf
            b_same_entity_II,
            torch.clip(
                f["sym_id"].unsqueeze(-1) - f["sym_id"].unsqueeze(-2) + self.s_max,
                0,
                2 * self.s_max,
            ),
            2 * self.s_max + 1,
        )
        A_relchain_II = one_hot(d_chain_II.long(), 2 * self.s_max + 2)

        #########################################################
        # Cancel out distances from unidexed motifs
        unindexing_pair_mask = f[
            "unindexing_pair_mask"
        ]  # [L, L] representing the parts which shouldnt' talk to one another

        # Special position case
        d_token_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1
        d_residue_II[unindexing_pair_mask] = self.num_tok_pos_bins - 1

        A_relpos_II = one_hot(d_residue_II.long(), self.num_tok_pos_bins)
        A_reltoken_II = one_hot(d_token_II, self.num_tok_pos_bins)
        #########################################################

        return self.linear(
            torch.cat(
                [
                    A_relpos_II,
                    A_reltoken_II,
                    b_same_entity_II.unsqueeze(-1),
                    A_relchain_II,
                ],
                dim=-1,
            ).to(torch.float)
        )

    def _forward_chunked(self, f):
        # Stream over query dimension to avoid holding full LxL intermediates
        L = f["asym_id"].shape[0]
        n_parallel = max(self.chunk_size, 1)
        chunk = max(1, (L + n_parallel - 1) // n_parallel)
        device = f["asym_id"].device

        # Precompute 1D arrays
        asym = f["asym_id"]
        entity = f["entity_id"]
        residue = f["residue_index"]
        token_idx = f["token_index"]
        sym_id = f["sym_id"]
        unindex_mask = f["unindexing_pair_mask"]  # [L, L]

        outputs = []
        for i_start in range(0, L, chunk):
            i_end = min(i_start + chunk, L)
            qs = slice(i_start, i_end)

            b_samechain = asym[qs, None] == asym[None, :]
            b_same_entity = entity[qs, None] == entity[None, :]

            # residue distances
            res_diff = residue[qs, None] - residue[None, :]
            d_res = torch.where(
                b_samechain,
                torch.clip(res_diff + self.r_max, 0, 2 * self.r_max),
                2 * self.r_max + 1,
            )

            b_sameres = residue[qs, None] == residue[None, :]
            tok_diff = token_idx[qs, None] - token_idx[None, :] + self.r_max
            d_tok = torch.where(
                b_samechain * b_sameres,
                torch.clip(tok_diff, 0, 2 * self.r_max),
                2 * self.r_max + 1,
            )

            # chain distances
            sym_diff = sym_id[qs, None] - sym_id[None, :]
            d_chain = torch.where(
                b_same_entity,
                torch.clip(sym_diff + self.s_max, 0, 2 * self.s_max),
                2 * self.s_max + 1,
            )

            # unindexing mask
            unmask = unindex_mask[qs, :]
            d_tok[unmask] = self.num_tok_pos_bins - 1
            d_res[unmask] = self.num_tok_pos_bins - 1

            A_res = one_hot(d_res.long(), self.num_tok_pos_bins)
            A_tok = one_hot(d_tok.long(), self.num_tok_pos_bins)
            A_chain = one_hot(d_chain.long(), 2 * self.s_max + 2)

            feats = torch.cat(
                [A_res, A_tok, b_same_entity.unsqueeze(-1).float(), A_chain], dim=-1
            )
            outputs.append(self.linear(feats))

        return torch.cat(outputs, dim=0)

    def forward_chunk(self, f: dict, start_i: int, end_i: int) -> torch.Tensor:
        """
        Compute RPE for a specific row range (query chunk) against all columns (keys).
        
        For streaming mode: computes [I_par, I, c_z] instead of full [I, I, c_z].
        
        Args:
            f: Feature dictionary containing asym_id, entity_id, residue_index, etc.
            start_i: Start index for query tokens
            end_i: End index for query tokens
            
        Returns:
            rpe_chunk: [I_par, I, c_z] relative position encoding for query chunk
        """
        I_par = end_i - start_i  # Number of queries
        qs = slice(start_i, end_i)
        
        # Get 1D arrays
        asym = f["asym_id"]           # [I]
        entity = f["entity_id"]       # [I]
        residue = f["residue_index"]  # [I]
        token_idx = f["token_index"]  # [I]
        sym_id = f["sym_id"]          # [I]
        unindex_mask = f["unindexing_pair_mask"]  # [I, I]
        
        # Compute pairwise comparisons: [I_par, I]
        b_samechain = asym[qs, None] == asym[None, :]     # [I_par, I]
        b_same_entity = entity[qs, None] == entity[None, :]  # [I_par, I]
        
        # Residue distances: [I_par, I]
        res_diff = residue[qs, None] - residue[None, :]
        d_res = torch.where(
            b_samechain,
            torch.clip(res_diff + self.r_max, 0, 2 * self.r_max),
            2 * self.r_max + 1,
        )  # [I_par, I]
        
        # Token distances: [I_par, I]
        b_sameres = residue[qs, None] == residue[None, :]
        tok_diff = token_idx[qs, None] - token_idx[None, :] + self.r_max
        d_tok = torch.where(
            b_samechain * b_sameres,
            torch.clip(tok_diff, 0, 2 * self.r_max),
            2 * self.r_max + 1,
        )  # [I_par, I]
        
        # Chain distances: [I_par, I]
        sym_diff = sym_id[qs, None] - sym_id[None, :]
        d_chain = torch.where(
            b_same_entity,
            torch.clip(sym_diff + self.s_max, 0, 2 * self.s_max),
            2 * self.s_max + 1,
        )  # [I_par, I]
        
        # Apply unindexing mask: [I_par, I]
        unmask = unindex_mask[qs, :]
        d_tok[unmask] = self.num_tok_pos_bins - 1
        d_res[unmask] = self.num_tok_pos_bins - 1
        
        # One-hot encodings: [I_par, I, num_bins]
        A_res = one_hot(d_res.long(), self.num_tok_pos_bins)    # [I_par, I, num_tok_pos_bins]
        A_tok = one_hot(d_tok.long(), self.num_tok_pos_bins)    # [I_par, I, num_tok_pos_bins]
        A_chain = one_hot(d_chain.long(), 2 * self.s_max + 2)   # [I_par, I, 2*s_max+2]
        
        # Concatenate features and project: [I_par, I, c_z]
        feats = torch.cat(
            [A_res, A_tok, b_same_entity.unsqueeze(-1).float(), A_chain], 
            dim=-1
        )  # [I_par, I, input_dim]
        
        return self.linear(feats)  # [I_par, I, c_z]


class VirtualPredictor(nn.Module):
    def __init__(self, c_atom):
        super(VirtualPredictor, self).__init__()
        self.process_atom_embeddings = nn.Sequential(
            RMSNorm((c_atom,)), linearNoBias(c_atom, 1)
        )

    def forward(self, Q_L):
        return self.process_atom_embeddings(Q_L)


class SequenceHead(nn.Module):
    def __init__(self, c_token):
        super(SequenceHead, self).__init__()

        # Distogram feature extraction
        self.dist_fc1 = nn.Linear(196, 128)
        self.dist_relu = nn.ReLU()
        self.dist_fc2 = nn.Linear(128, 64)

        # Embedding feature extraction
        self.embed_fc1 = nn.Linear(c_token, 128)
        self.embed_relu = nn.ReLU()
        self.embed_fc2 = nn.Linear(128, 64)

        # Fusion layer
        self.fusion_fc = nn.Linear(128, 32)

        # Sequence encoding
        self.sequence_encoding_ = AF3SequenceEncoding()

    def forward(self, A_I, Q_L, X_L, f):
        B, L, _ = X_L.shape
        max_res_id = f["atom_to_token_map"].max().item() + 1

        # Detach tensors to avoid gradients through main module
        # X_L = X_L.detach()
        # A_I = A_I.detach()
        # Q_L = Q_L.detach()

        # Compute distograms
        residue_distogram = torch.zeros(B, max_res_id, 14, 14, device=X_L.device)
        for i in range(max_res_id):
            residue_mask = f["atom_to_token_map"] == i
            if residue_mask.sum() == 14:
                coords = X_L[:, residue_mask]  # (B, 14, 3)
                residue_distogram[:, i] = torch.cdist(coords, coords)

        # Flatten distogram
        dist_features = residue_distogram.view(B, max_res_id, 196)

        # Pass through separate MLPs
        dist_out = self.dist_fc1(dist_features)
        dist_out = self.dist_relu(dist_out)
        dist_out = self.dist_fc2(dist_out)

        embed_out = self.embed_fc1(A_I)
        embed_out = self.embed_relu(embed_out)
        embed_out = self.embed_fc2(embed_out)

        # Fusion via concatenation
        fused = torch.cat([dist_out, embed_out], dim=-1)
        Seq_I = self.fusion_fc(fused)

        indices = self.decode(Seq_I)

        return Seq_I, indices

    def decode(self, Seq_I):
        indices = Seq_I.argmax(dim=-1)  # [B, L]
        return indices


class LinearSequenceHead(nn.Module):
    def __init__(self, c_token):
        super().__init__()
        n_tok_all = 32
        disallowed_idxs = AF3SequenceEncoding().encode(["UNK", "X", "DX", "<G>"])
        mask = torch.ones(n_tok_all, dtype=torch.bool)
        mask[disallowed_idxs] = False
        self.register_buffer("valid_out_mask", mask)
        self.linear = nn.Linear(c_token, n_tok_all)

    def forward(self, A_I, **_):
        logits = self.linear(A_I)
        indices = self.decode(logits)
        return logits, indices

    def decode(self, logits):
        # logits: [D, L, 28]
        # indices: [D, L] in [0,32-1]
        D, I, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        probs = probs * self.valid_out_mask[None, None, :].to(probs.device)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        indices = probs.argmax(axis=-1)
        return indices


class Upcast(nn.Module):
    def __init__(
        self, c_token, c_atom, method="broadcast", cross_attention_block=None, n_split=6
    ):
        super().__init__()
        self.method = method
        self.n_split = n_split
        if self.method == "broadcast":
            self.project = nn.Sequential(
                RMSNorm((c_token,)), linearNoBias(c_token, c_atom)
            )
        elif self.method == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_atom, c_kv=c_token // self.n_split, **cross_attention_block
            )
        else:
            raise ValueError(f"Unknown upcast method: {self.method}")

    def forward_(self, Q_IA, A_I, valid_mask=None):
        if self.method == "broadcast":
            Q_IA = Q_IA + self.project(A_I)[..., None, :]
        elif self.method == "cross_attention":
            assert exists(A_I) and exists(valid_mask)
            # Split Tokens
            A_I = rearrange(A_I, "b n (s c) -> b n s c", s=self.n_split)
            n_tokens, n_atom_per_tok = Q_IA.shape[1], Q_IA.shape[2]

            # Attention mask: ..., n_atom_per_tok, n_split
            attn_mask = torch.full(
                (n_tokens, 1, n_atom_per_tok), True, device=Q_IA.device
            )
            attn_mask[~valid_mask.view_as(attn_mask)] = False

            attn_mask = torch.ones(
                (n_tokens, n_atom_per_tok, self.n_split), device=A_I.device, dtype=bool
            )
            attn_mask[~valid_mask, :] = False

            Q_IA = Q_IA + self.gca(q=Q_IA, kv=A_I, attn_mask=attn_mask)
        return Q_IA

    def forward(self, Q_L, A_I, tok_idx):
        valid_mask = build_valid_mask(tok_idx)
        Q_IA = ungroup_atoms(Q_L, valid_mask)
        Q_IA = self.forward_(Q_IA, A_I, valid_mask)
        Q_L = group_atoms(Q_IA, valid_mask)
        return Q_L


class Downcast(nn.Module):
    """Downcast modules for when atoms are already reshaped from N_atoms -> (N_tokens, 14)"""

    def __init__(
        self, c_atom, c_token, c_s=None, method="mean", cross_attention_block=None
    ):
        super().__init__()
        self.method = method
        self.c_token = c_token
        self.c_atom = c_atom
        if c_s is not None:
            self.process_s = nn.Sequential(
                RMSNorm((c_s,)),
                linearNoBias(c_s, c_token),
            )
        else:
            self.process_s = None

        if self.method == "mean":
            self.project = linearNoBias(c_atom, c_token)
        elif self.method == "cross_attention":
            self.gca = GatedCrossAttention(
                c_query=c_token,
                c_kv=c_atom,
                **cross_attention_block,
            )
        else:
            raise ValueError(f"Unknown downcast method: {self.method}")

    def forward_(self, Q_IA, A_I, S_I=None, valid_mask=None):
        if self.method == "mean":
            A_I_update = self.project(Q_IA).sum(-2) / valid_mask.sum(-1, keepdim=True)
        elif self.method == "cross_attention":
            assert exists(A_I) and exists(valid_mask)
            # Attention mask: ..., 1, n_atom_per_tok (1 querying token to atoms in token)
            attn_mask = valid_mask[..., None, :]
            A_I_update = self.gca(
                q=A_I[..., None, :], kv=Q_IA, attn_mask=attn_mask
            ).squeeze(-2)

        A_I = A_I + A_I_update if exists(A_I) else A_I_update

        if self.process_s is not None:
            A_I = A_I + self.process_s(S_I)
        return A_I

    def forward(self, Q_L, A_I, S_I=None, tok_idx=None):
        valid_mask = build_valid_mask(tok_idx)
        if Q_L.ndim == 2:
            squeeze = True
            Q_L = Q_L.unsqueeze(0)
        else:
            squeeze = False

        A_I = A_I.unsqueeze(0) if exists(A_I) and A_I.ndim == 2 else A_I
        S_I = S_I.unsqueeze(0) if exists(S_I) and S_I.ndim == 2 else S_I

        Q_IA = ungroup_atoms(Q_L, valid_mask)

        A_I = self.forward_(Q_IA, A_I, S_I, valid_mask=valid_mask)

        if squeeze:
            A_I = A_I.squeeze(0)
        return A_I


######################################################################################
##########################     Local Atom Transformer       ##########################
######################################################################################


class LocalTokenTransformer(nn.Module):
    def __init__(
        self,
        c_token,
        c_tokenpair,
        c_s,
        n_block,
        diffusion_transformer_block,
        n_registers=None,
        n_local_tokens=8,
        n_keys=32,
    ):
        super().__init__()
        self.n_local_tokens = n_local_tokens
        self.n_keys = n_keys
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_token,
                    c_s=c_s,
                    c_atompair=c_tokenpair,
                    **diffusion_transformer_block,
                )
                for _ in range(n_block)
            ]
        )

    def forward(self, A_I, S_I, Z_II, f, X_L, full=False):
        indices = create_attention_indices(
            X_L=X_L,
            f=f,
            tok_idx=torch.arange(A_I.shape[1], device=A_I.device),
            n_attn_keys=self.n_keys,
            n_attn_seq_neighbours=self.n_local_tokens,
        )

        for i, block in enumerate(self.blocks):
            # Set checkpointing
            block.attention_pair_bias.use_checkpointing = not DISABLE_CHECKPOINTING
            # A_I: [B, L, C_token]
            # S_I: [B, L, C_s]
            # Z_II: [B, L, L, C_tokenpair]
            A_I = block(
                A_I,
                S_I,
                Z_II,
                indices=indices,
                full=full,  # (self.training and torch.is_grad_enabled()),  # Does not accelerate inference, but memory *does* scale better
            )

        return A_I

    def forward_cross_attn(
        self,
        A_I_chunk,         # [B, I_par, c_token] - queries from this GPU's chunk
        S_I_chunk,         # [I_par, c_s] or [B, I_par, c_s] - conditioning for queries
        A_I_full,          # [B, I, c_token] - keys/values from all tokens
        S_I_full,          # [I, c_s] or [B, I, c_s] - conditioning for K/V
        Z_chunk,           # [I_par, I, c_z] or [B, I_par, I, c_z] - pair bias
        f,                 # Feature dictionary
        X_L,               # [B, I, 3] - CA positions for index computation
        query_start,       # int - global start index of this GPU's queries
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
            
        Returns:
            A_I_chunk: Updated query features [B, I_par, c_token]
        """
        B, I_par, c_token = A_I_chunk.shape
        I = A_I_full.shape[1]
        device = A_I_chunk.device
        
        # Ensure transformer blocks are on the correct device for this rank
        self.to(device)
        
        # Compute attention indices for the query chunk
        # indices specify which keys (from 0..I-1) each query should attend to
        indices_full = create_attention_indices(
            X_L=X_L,
            f=f,
            tok_idx=torch.arange(I, device=device),
            n_attn_keys=self.n_keys,
            n_attn_seq_neighbours=self.n_local_tokens,
        )  # [B, I, k]
        
        # Slice indices for this GPU's queries
        query_end = query_start + I_par
        indices_chunk = indices_full[:, query_start:query_end, :]  # [B, I_par, k]
        
        # Expand S_I if needed for batch dimension
        if S_I_chunk.ndim == 2:
            S_I_chunk = S_I_chunk.unsqueeze(0).expand(B, -1, -1)  # [B, I_par, c_s]
        if S_I_full.ndim == 2:
            S_I_full = S_I_full.unsqueeze(0).expand(B, -1, -1)    # [B, I, c_s]
        
        # Run cross-attention through blocks
        for block in self.blocks:
            block.attention_pair_bias.use_checkpointing = not DISABLE_CHECKPOINTING
            
            # Cross-attention: chunk queries → all keys
            A_I_chunk = block.forward_cross_attn(
                Q_chunk=A_I_chunk,                  # [B, I_par, c_token]
                C_Q_chunk=S_I_chunk,                # [B, I_par, c_s]
                K_V_full=A_I_full,                  # [B, I, c_token]
                C_KV_full=S_I_full,                 # [B, I, c_s]
                P_chunk=Z_chunk,                    # [I_par, I, c_z] or [B, I_par, I, c_z]
                indices_chunk=indices_chunk,        # [B, I_par, k]
            )  # [B, I_par, c_token]
        
        return A_I_chunk  # [B, I_par, c_token]


class LocalAtomTransformer(nn.Module):
    def __init__(self, c_atom, c_s, c_atompair, atom_transformer_block, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_s,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, Q_L, C_L, P_LL, **kwargs):
        for block in self.blocks:
            Q_L = block(Q_L, C_L, P_LL, **kwargs)
        return Q_L
    
    def forward_parallel(
        self,
        Q_L,                        # [B, L, c_atom] - full atom features
        C_L,                        # [B, L, c_atom] - conditioned features
        indices,                    # [B, L, k] - sparse attention indices
        query_start,                # int - this GPU's query start
        query_end,                  # int - this GPU's query end  
        f,                          # feature dict
        chunked_pairwise_embedder,  # computes P_LL_sparse on-the-fly
        initializer_outputs,        # embedder state
        all_gather_fn,              # gather function
        world_size,                 # number of GPUs
    ):
        """
        Parallel forward: each GPU processes L_par = L/n_par atom queries.
        
        No L×L tensor is materialized:
        - P_LL_sparse computed on-the-fly for L_par queries only
        - Each GPU computes [L_par, k, c] instead of full [L, k, c]
        - Q_L chunks gathered after each block
        
        Args:
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
        
        # Compute max chunk size (for padding to ensure equal sizes for all_gather)
        max_L_par = (L + world_size - 1) // world_size  # Ceiling division
        
        import torch.distributed as dist
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # Ensure all tensors are on the correct device for this rank BEFORE slicing
        Q_L = Q_L.to(device)
        C_L = C_L.to(device)
        indices = indices.to(device)
        
        # This GPU's indices (slice AFTER moving to device)
        indices_chunk = indices[:, query_start:query_end, :]  # [B, L_par, k]
        
        # Ensure encoder blocks and embedder are on the correct device for this rank
        self.to(device)
        if chunked_pairwise_embedder is not None:
            chunked_pairwise_embedder.to(device)
        
        for block_idx, block in enumerate(self.blocks):
            # Extract this GPU's query chunk
            Q_L_chunk = Q_L[:, query_start:query_end, :]      # [B, L_par, c_atom]
            C_L_chunk = C_L[:, query_start:query_end, :]      # [B, L_par, c_atom]
            
            # Compute P_LL_sparse for THIS GPU's queries only
            # This is the key: [L_par, k, c] not [L, k, c]
            P_sparse_chunk = chunked_pairwise_embedder.forward_chunked_parallel(
                indices_chunk=indices_chunk,                   # [B, L_par, k]
                query_start=query_start,
                query_end=query_end,
                f=f,
                initializer_outputs=initializer_outputs,
            )                                                  # [B, L_par, k, c_pair]
            
            # Sparse cross-attention: Q_L_chunk attends to sparse neighbors
            Q_L_chunk = block.forward_sparse_cross_attn(
                Q_chunk=Q_L_chunk,                              # [B, L_par, c_atom]
                C_Q_chunk=C_L_chunk,                            # [B, L_par, c_atom]
                K_V_full=Q_L,                                   # [B, L, c_atom]
                C_KV_full=C_L,                                  # [B, L, c_atom]
                P_sparse_chunk=P_sparse_chunk,                  # [B, L_par, k, c_pair]
                indices_chunk=indices_chunk,                    # [B, L_par, k]
            )                                                  # [B, L_par, c_atom]
            
            # Pad to max_L_par for all_gather (NCCL requires equal sizes)
            if L_par < max_L_par:
                pad_size = max_L_par - L_par
                Q_L_chunk = torch.nn.functional.pad(Q_L_chunk, (0, 0, 0, pad_size))  # [B, max_L_par, c_atom]
            
            # All-gather padded chunks
            Q_L_gathered = all_gather_fn(Q_L_chunk, world_size, dim=1)  # [B, max_L_par*world_size, c_atom]
            
            # Trim back to actual L (remove padding) and ensure on correct device
            Q_L = Q_L_gathered[:, :L, :].to(device)  # [B, L, c_atom]
        
        return Q_L


class StructureLocalAtomTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        c_atom,
        c_s,
        c_atompair,
        dropout,
        no_residual_connection_between_attention_and_transition,
        **transformer_block,
    ):
        super().__init__()
        assert not no_residual_connection_between_attention_and_transition
        self.c_s = c_s
        self.dropout = nn.Dropout(dropout)
        self.attention_pair_bias = LocalAttentionPairBias(
            c_a=c_atom, c_s=c_s, c_pair=c_atompair, **transformer_block
        )
        if exists(c_s):
            self.transition_block = ConditionedTransitionBlock(c_token=c_atom, c_s=c_s)
        else:
            self.transition_block = Transition(c=c_atom, n=4)

    def forward(
        self,
        Q_L,  # [..., I, C_token]
        C_L,  # [..., I, C_s]
        P_LL,  # [..., I, I, C_tokenpair]
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        **kwargs,
    ):
        Q_L = Q_L + self.dropout(
            self.attention_pair_bias(
                Q_L,
                C_L,
                P_LL,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                **kwargs,
            )
        )
        if exists(C_L):
            Q_L = Q_L + self.transition_block(Q_L, C_L)
        else:
            Q_L = Q_L + self.transition_block(Q_L)
        return Q_L

    def forward_cross_attn(
        self,
        Q_chunk,           # [D, L_par, c_a] - queries from this GPU's chunk
        C_Q_chunk,         # [D, L_par, c_s] - conditioning for queries
        K_V_full,          # [D, L, c_a] - keys/values from all tokens
        C_KV_full,         # [D, L, c_s] - conditioning for K/V
        P_chunk,           # [L_par, L, c_pair] - pair bias for chunk×all
        indices_chunk,     # [D, L_par, k] - sparse attention indices
    ):
        """
        Cross-attention forward: queries from chunk attend to all keys.
        
        For multi-GPU parallel inference:
        - Each GPU processes L_par = L // n_gpus queries
        - All GPUs have access to all L keys
        - P_chunk is [L_par, L] not [L, L] - avoids L×L memory
        
        Args:
            Q_chunk: Query features [D, L_par, c_a]
            C_Q_chunk: Query conditioning [D, L_par, c_s]
            K_V_full: Key/Value features [D, L, c_a]
            C_KV_full: K/V conditioning [D, L, c_s]
            P_chunk: Pair bias [L_par, L, c_pair] or [D, L_par, L, c_pair]
            indices_chunk: Attention indices [D, L_par, k]
            
        Returns:
            Q_chunk: Updated query features [D, L_par, c_a]
        """
        # Cross-attention: Q_chunk queries attend to K_V_full keys
        attn_out = self.attention_pair_bias.forward_cross_attn(
            Q_chunk=Q_chunk,                               # [D, L_par, c_a]
            C_Q_chunk=C_Q_chunk,                           # [D, L_par, c_s]
            K_V_full=K_V_full,                             # [D, L, c_a]
            C_KV_full=C_KV_full,                           # [D, L, c_s]
            P_chunk=P_chunk,                               # [L_par, L, c_pair]
            indices_chunk=indices_chunk,                   # [D, L_par, k]
        )                                                  # [D, L_par, c_a]
        
        Q_chunk = Q_chunk + self.dropout(attn_out)         # [D, L_par, c_a]
        
        # Transition block on query chunk
        if exists(C_Q_chunk):
            Q_chunk = Q_chunk + self.transition_block(Q_chunk, C_Q_chunk)
        else:
            Q_chunk = Q_chunk + self.transition_block(Q_chunk)
        
        return Q_chunk                                     # [D, L_par, c_a]

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
        
        Combines parallelism (L_par queries) with sparse attention (k neighbors):
        - P_sparse_chunk is [L_par, k] not [L_par, L] or [L, L]
        - K/V are gathered from indices, not dense
        - Memory: O(L_par * k) instead of O(L_par * L) or O(L²)
        
        Args:
            Q_chunk: Query features [D, L_par, c_a]
            C_Q_chunk: Query conditioning [D, L_par, c_s]
            K_V_full: Key/Value features [D, L, c_a]
            C_KV_full: K/V conditioning [D, L, c_s]
            P_sparse_chunk: Sparse pair bias [D, L_par, k, c_pair]
            indices_chunk: Sparse attention indices [D, L_par, k]
            
        Returns:
            Q_chunk: Updated query features [D, L_par, c_a]
        """
        # Sparse cross-attention: Q_chunk queries attend to sparse K/V
        attn_out = self.attention_pair_bias.forward_sparse_cross_attn(
            Q_chunk=Q_chunk,                               # [D, L_par, c_a]
            C_Q_chunk=C_Q_chunk,                           # [D, L_par, c_s]
            K_V_full=K_V_full,                             # [D, L, c_a]
            C_KV_full=C_KV_full,                           # [D, L, c_s]
            P_sparse_chunk=P_sparse_chunk,                 # [D, L_par, k, c_pair]
            indices_chunk=indices_chunk,                   # [D, L_par, k]
        )                                                  # [D, L_par, c_a]
        
        Q_chunk = Q_chunk + self.dropout(attn_out)         # [D, L_par, c_a]
        
        # Transition block on query chunk
        if exists(C_Q_chunk):
            Q_chunk = Q_chunk + self.transition_block(Q_chunk, C_Q_chunk)
        else:
            Q_chunk = Q_chunk + self.transition_block(Q_chunk)
        
        return Q_chunk                                     # [D, L_par, c_a]


class CompactStreamingDecoder(nn.Module):
    def __init__(
        self,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_tokenpair,
        atom_transformer_block,
        upcast,
        downcast,
        n_blocks,
        diffusion_transformer_block=False,
    ):
        super().__init__()
        self.n_blocks = n_blocks

        self.upcast = nn.ModuleList(
            [Upcast(c_atom=c_atom, c_token=c_token, **upcast) for _ in range(n_blocks)]
        )
        self.atom_transformer = nn.ModuleList(
            [
                StructureLocalAtomTransformerBlock(
                    c_atom=c_atom,
                    c_s=c_atom,
                    c_atompair=c_atompair,
                    **atom_transformer_block,
                )
                for _ in range(n_blocks)
            ]
        )
        self.downcast = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)

    def forward(
        self,
        A_I,
        S_I,
        Z_II,
        Q_L,
        C_L,
        P_LL,
        tok_idx,
        indices,
        f=None,
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
    ):
        for i in range(self.n_blocks):
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)
            Q_L = self.atom_transformer[i](
                Q_L,
                C_L,
                P_LL,
                indices=indices,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )

        # Downcast to sequence
        A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o

    def forward_parallel(
        self,
        A_I,                   # [B, I, c_token] - full token features
        S_I,                   # [I, c_s] - single features
        Q_L,                   # [B, L, c_atom] - atom features
        C_L,                   # [B, L, c_atom] - conditioned atom features
        P_chunk,               # [L_par, L, c_pair] - THIS GPU's P_LL rows
        tok_idx,               # [L] - atom to token mapping
        indices,               # [B, L, k] - full attention indices
        query_start,           # int - start index of this GPU's atom queries
        query_end,             # int - end index of this GPU's atom queries
        all_gather_fn,         # Callable to gather tensors from all GPUs
        world_size,            # int - number of GPUs
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
            all_gather_fn: Function to gather tensors from all GPUs
            world_size: Number of GPUs
            
        Returns:
            A_I: Updated token features [B, I, c_token]
            Q_L: Updated atom features [B, L, c_atom]
            o: Empty dict (for compatibility)
        """
        B = A_I.shape[0]
        L = Q_L.shape[1]
        L_par = query_end - query_start                    # This GPU's chunk size
        
        # Get the correct device for this rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # Ensure decoder modules are on the correct device for this rank
        self.to(device)
        
        # Slice indices for this GPU's queries
        indices_chunk = indices[:, query_start:query_end, :]  # [B, L_par, k]
        
        # Run blocks with cross-attention
        for i in range(self.n_blocks):
            # Upcast: token → atom (operates on full, no L×L)
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)  # [B, L, c_atom]
            
            # Extract this GPU's Q_L chunk for attention
            Q_L_chunk = Q_L[:, query_start:query_end, :]    # [B, L_par, c_atom]
            C_L_chunk = C_L[:, query_start:query_end, :]    # [B, L_par, c_atom]
            
            # Cross-attention: Q_L_chunk queries attend to Q_L keys
            # P_chunk is [L_par, L] - no L×L tensor
            Q_L_chunk = self.atom_transformer[i].forward_cross_attn(
                Q_chunk=Q_L_chunk,                          # [B, L_par, c_atom]
                C_Q_chunk=C_L_chunk,                        # [B, L_par, c_atom]
                K_V_full=Q_L,                               # [B, L, c_atom]
                C_KV_full=C_L,                              # [B, L, c_atom]
                P_chunk=P_chunk,                            # [L_par, L, c_pair]
                indices_chunk=indices_chunk,                # [B, L_par, k]
            )                                              # [B, L_par, c_atom]
            
            # All-gather Q_L chunks to reconstruct full Q_L
            Q_L = all_gather_fn(Q_L_chunk, world_size, dim=1)  # [B, L, c_atom]

        # Downcast to sequence (operates on full Q_L, no L×L)
        A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o

    def forward_parallel_sparse(
        self,
        A_I,                        # [B, I, c_token] - full token features
        S_I,                        # [I, c_s] - single features
        Q_L,                        # [B, L, c_atom] - atom features
        C_L,                        # [B, L, c_atom] - conditioned atom features
        tok_idx,                    # [L] - atom to token mapping
        tok_idx_chunk,              # [L_par] - this GPU's atoms
        indices,                    # [B, L, k] - full attention indices
        indices_chunk,              # [B, L_par, k] - this GPU's attention indices
        query_start,                # int - start index
        query_end,                  # int - end index
        f,                          # Feature dict
        chunked_pairwise_embedder,  # ChunkedPairwiseEmbedder
        initializer_outputs,        # Dict with embedder state
        all_gather_fn,              # Callable to gather tensors
        world_size,                 # int - number of GPUs
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
            all_gather_fn: Function to gather tensors from all GPUs
            world_size: Number of GPUs
            
        Returns:
            A_I: Updated token features [B, I, c_token]
            Q_L: Updated atom features [B, L, c_atom]
            o: Empty dict
        """
        B = A_I.shape[0]
        L = Q_L.shape[1]
        L_par = query_end - query_start                    # This GPU's chunk size
        
        # Get the correct device for this rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        
        # Ensure decoder modules and embedder are on the correct device for this rank
        self.to(device)
        if chunked_pairwise_embedder is not None:
            chunked_pairwise_embedder.to(device)
        
        # Run blocks
        for i in range(self.n_blocks):
            # Upcast: token → atom (operates on full, no L×L)
            Q_L = self.upcast[i](Q_L, A_I, tok_idx=tok_idx)  # [B, L, c_atom]
            
            # Extract this GPU's Q_L chunk for attention
            Q_L_chunk = Q_L[:, query_start:query_end, :]    # [B, L_par, c_atom]
            C_L_chunk = C_L[:, query_start:query_end, :]    # [B, L_par, c_atom]
            
            # Compute SPARSE P_LL for this GPU's chunk only
            # Uses chunked_pairwise_embedder which computes P only for (L_par, k) pairs
            # indices_chunk tells which k neighbors each of the L_par atoms attends to
            # NOTE: Pass FULL C_L (not chunked) because forward_chunked gathers keys from any atom
            P_sparse_chunk = chunked_pairwise_embedder.forward_chunked(
                f=f,
                indices=indices_chunk,                      # [B, L_par, k]
                C_L=initializer_outputs["C_L"],             # [L, c_atom] - FULL, not chunked
                Z_init_II=initializer_outputs["Z_II"],      # [I, I, c_z] or StreamingZContainer
                tok_idx=f["atom_to_token_map"],             # [L]
                query_start=query_start,                    # Offset for parallel mode
            )                                              # [B, L_par, k, c_atompair]
            
            # Sparse cross-attention: Q_L_chunk queries attend to sparse K/V
            # Uses indices_chunk to gather sparse keys/values
            Q_L_chunk = self.atom_transformer[i].forward_sparse_cross_attn(
                Q_chunk=Q_L_chunk,                          # [B, L_par, c_atom]
                C_Q_chunk=C_L_chunk,                        # [B, L_par, c_atom]
                K_V_full=Q_L,                               # [B, L, c_atom]
                C_KV_full=C_L,                              # [B, L, c_atom]
                P_sparse_chunk=P_sparse_chunk,              # [B, L_par, k, c_atompair]
                indices_chunk=indices_chunk,                # [B, L_par, k]
            )                                              # [B, L_par, c_atom]
            
            # All-gather Q_L chunks to reconstruct full Q_L
            Q_L = all_gather_fn(Q_L_chunk, world_size, dim=1)  # [B, L, c_atom]

        # Downcast to sequence (operates on full Q_L, no L×L)
        A_I = self.downcast(Q_L.detach(), A_I.detach(), S_I.detach(), tok_idx=tok_idx)

        o = {}
        return A_I, Q_L, o
