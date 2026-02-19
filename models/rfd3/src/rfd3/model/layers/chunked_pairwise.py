"""
Chunked pairwise embedding implementation for memory-efficient large structure processing.

This module provides memory-optimized versions of pairwise embedders that compute
only the pairs needed for sparse attention, reducing memory usage from O(L²) to O(L×k).
"""

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias
from rfd3.model.debug_context import debug_ctx, debug_tensor, debug_log, debug_elements, debug_tensor_all_ranks, debug_log_all_ranks

logger = logging.getLogger(__name__)

def _log_tensor_stats(name: str, tensor: torch.Tensor, rank: int = 0):
    """Log tensor statistics for debugging. Controlled by verbose_stats config flag."""
    if not debug_ctx.stats_enabled:
        return
    debug_tensor("PAIRWISE", name, tensor, rank)


class ChunkedPositionPairDistEmbedder(nn.Module):
    """
    Memory-efficient version of PositionPairDistEmbedder that computes pairs on-demand.
    """

    def __init__(self, c_atompair, embed_frame=True):
        super().__init__()
        self.c_atompair = c_atompair
        self.embed_frame = embed_frame
        if embed_frame:
            self.process_d = linearNoBias(3, c_atompair)

        self.process_inverse_dist = linearNoBias(1, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,  # [B, 3]
        key_pos: torch.Tensor,  # [B, k, 3]
        valid_mask: torch.Tensor,  # [B, k, 1]
    ) -> torch.Tensor:
        """
        Compute pairwise embeddings for specific query-key pairs.

        Args:
            query_pos: Query positions [B, 3]
            key_pos: Key positions [B, k, 3]
            valid_mask: Valid pair mask [B, k, 1]

        Returns:
            P_sparse: Pairwise embeddings [B, k, c_atompair]
        """
        B, k = key_pos.shape[:2]

        # Compute pairwise distances: [B, k, 3]
        D_pairs = query_pos.unsqueeze(1) - key_pos  # [B, 1, 3] - [B, k, 3] = [B, k, 3]

        if self.embed_frame:
            # Embed pairwise distances
            P_pairs = self.process_d(D_pairs) * valid_mask  # [B, k, c_atompair]

            # Add inverse distance embedding
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2  # [B, k, 1]
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = P_pairs + self.process_inverse_dist(inv_dist) * valid_mask

            # Add valid mask embedding
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )
        else:
            # Simplified version without frame embedding
            norm_sq = torch.linalg.norm(D_pairs, dim=-1, keepdim=True) ** 2
            norm_sq = torch.clamp(norm_sq, min=1e-6)
            inv_dist = 1 / (1 + norm_sq)
            P_pairs = self.process_inverse_dist(inv_dist) * valid_mask
            P_pairs = (
                P_pairs
                + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
            )

        return P_pairs


class ChunkedSinusoidalDistEmbed(nn.Module):
    """
    Memory-efficient version of SinusoidalDistEmbed.
    """

    def __init__(self, c_atompair, n_freqs=32):
        super().__init__()
        assert c_atompair % 2 == 0, "Output embedding dim must be even"

        self.n_freqs = n_freqs
        self.c_atompair = c_atompair

        self.output_proj = linearNoBias(2 * n_freqs, c_atompair)
        self.process_valid_mask = linearNoBias(1, c_atompair)

    def compute_pairs_chunked(
        self,
        query_pos: torch.Tensor,  # [B, 3]
        key_pos: torch.Tensor,  # [B, k, 3]
        valid_mask: torch.Tensor,  # [B, k, 1]
    ) -> torch.Tensor:
        """
        Compute sinusoidal distance embeddings for specific query-key pairs.
        """
        B, k = key_pos.shape[:2]
        device = query_pos.device

        # Compute pairwise distances
        D_pairs = query_pos.unsqueeze(1) - key_pos  # [B, k, 3]
        dist_matrix = torch.linalg.norm(D_pairs, dim=-1)  # [B, k]

        # Sinusoidal embedding
        half_dim = self.n_freqs
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, dtype=torch.float32, device=device)
            / half_dim
        )  # [n_freqs]

        angles = dist_matrix.unsqueeze(-1) * freq  # [B, k, n_freqs]
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        sincos_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [B, k, 2*n_freqs]

        # Linear projection
        P_pairs = self.output_proj(sincos_embed)  # [B, k, c_atompair]
        P_pairs = P_pairs * valid_mask

        # Add linear embedding of valid mask
        P_pairs = (
            P_pairs + self.process_valid_mask(valid_mask.to(P_pairs.dtype)) * valid_mask
        )

        return P_pairs


class ChunkedPairwiseEmbedder(nn.Module):
    """
    Main chunked pairwise embedder that combines all embedding types.
    This replaces the full P_LL computation with sparse computation.
    """

    def __init__(
        self,
        c_atompair: int,
        motif_pos_embedder: Optional[ChunkedPositionPairDistEmbedder] = None,
        ref_pos_embedder: Optional[ChunkedPositionPairDistEmbedder] = None,
        process_single_l: Optional[nn.Module] = None,
        process_single_m: Optional[nn.Module] = None,
        process_z: Optional[nn.Module] = None,
        pair_mlp: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.c_atompair = c_atompair
        self.motif_pos_embedder = motif_pos_embedder
        self.ref_pos_embedder = ref_pos_embedder

        # CRITICAL FIX: Store shared modules in a plain dict to prevent re-registration!
        # When modules are passed from a parent (e.g., TokenInitializer), assigning them
        # to self.xxx re-registers them under a new path (chunked_pairwise_embedder.xxx).
        # The checkpoint only has weights at the original path (token_initializer.xxx),
        # causing the re-registered modules to be reinitialized with random weights.
        # Using a plain dict avoids this issue while still allowing access to the modules.
        self._shared_modules = {}

        # Use shared trained MLPs if provided, otherwise create new ones
        if process_single_l is not None:
            # Store in dict to avoid re-registration
            self._shared_modules['process_single_l'] = process_single_l
        else:
            self.process_single_l = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_single_m is not None:
            self._shared_modules['process_single_m'] = process_single_m
        else:
            self.process_single_m = nn.Sequential(
                nn.ReLU(), linearNoBias(128, c_atompair)
            )

        if process_z is not None:
            self._shared_modules['process_z'] = process_z
        else:
            self.process_z = nn.Sequential(RMSNorm(128), linearNoBias(128, c_atompair))

        if pair_mlp is not None:
            self._shared_modules['pair_mlp'] = pair_mlp
        else:
            self.pair_mlp = nn.Sequential(
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
                nn.ReLU(),
                linearNoBias(c_atompair, c_atompair),
            )

    def _get_process_single_l(self):
        """Get process_single_l module (shared or owned)."""
        return self._shared_modules.get('process_single_l', getattr(self, 'process_single_l', None))

    def _get_process_single_m(self):
        """Get process_single_m module (shared or owned)."""
        return self._shared_modules.get('process_single_m', getattr(self, 'process_single_m', None))

    def _get_process_z(self):
        """Get process_z module (shared or owned)."""
        return self._shared_modules.get('process_z', getattr(self, 'process_z', None))

    def _get_pair_mlp(self):
        """Get pair_mlp module (shared or owned)."""
        return self._shared_modules.get('pair_mlp', getattr(self, 'pair_mlp', None))

    def forward_chunked(
        self,
        f: dict,
        indices: torch.Tensor,  # [B, L, k] or [B, L_par, k] - sparse attention indices
        C_L: torch.Tensor,  # [L, c_token] or [B, L, c_token] - atom features (FULL)
        Z_init_II: torch.Tensor,  # [I, I, c_z] - token pair features
        tok_idx: torch.Tensor,  # [L] - atom to token mapping (FULL)
        query_start: int = 0,  # Offset for parallel mode: which atoms indices correspond to
        **kwargs,  # streaming_mode, z_chunk_range for parallel mode
    ) -> torch.Tensor:
        # Add logging for chunked P_LL computation
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"ChunkedPairwiseEmbedder: Computing sparse P_LL for {indices.shape[1]} atoms with {indices.shape[2]} neighbors each"
        )
        """
        Compute P_LL only for the pairs specified by attention indices.
        
        Args:
            f: Feature dictionary
            indices: Sparse attention indices [B, L_par, k] (may be chunked in parallel mode)
            C_L: Atom-level features [L, c_token] (FULL - needed for key gathering)
            Z_init_II: Token-level pair features [I, I, c_z] (full) or [I_par, I, c_z] (chunked)
            tok_idx: Atom to token mapping [L] (FULL)
            query_start: Offset for query positions (0 for non-parallel, query_start for parallel)
            
        Returns:
            P_LL_sparse: Sparse pairwise features [B, L_par, k, c_atompair]
        """
        B, L_par, k = indices.shape  # L_par may be < L in parallel mode
        device = indices.device
        
        # Ensure embedder modules are on the correct device
        self.to(device)
        
        # Move input tensors to correct device
        C_L = C_L.to(device)
        tok_idx = tok_idx.to(device)
        # Move feature dict tensors to correct device
        for key, val in f.items():
            if isinstance(val, torch.Tensor):
                f[key] = val.to(device)

        # Initialize sparse P_LL for this chunk
        P_LL_sparse = torch.zeros(
            B, L_par, k, self.c_atompair, device=device, dtype=C_L.dtype
        )
        
        # Compute query end for this chunk
        query_end = query_start + L_par

        # Handle both batched and non-batched C_L
        if C_L.dim() == 2:  # [L, c_token] - add batch dimension
            C_L = C_L.unsqueeze(0)  # [1, L, c_token]
        # Add bounds checking to prevent index errors
        L_full = C_L.shape[1]  # Full length (may differ from L_par in parallel mode)
        valid_indices = torch.clamp(
            indices, 0, L_full - 1
        )  # Clamp indices to valid range

        # Ensure indices have the right shape for gathering
        if valid_indices.dim() == 2:  # [L_par, k] - add batch dimension
            valid_indices = valid_indices.unsqueeze(0).expand(
                C_L.shape[0], -1, -1
            )  # [B, L_par, k]

        # 1. Motif position embedding (if exists)
        if self.motif_pos_embedder is not None and "motif_pos" in f:
            motif_pos = f["motif_pos"]  # [L_full, 3]
            is_motif = f["is_motif_atom_with_fixed_coord"]  # [L_full]
            is_motif_idx = torch.where(is_motif)[0]
            # Filter to only query positions in this chunk
            is_motif_idx = is_motif_idx[(is_motif_idx >= query_start) & (is_motif_idx < query_end)]
            # For each query position in this chunk
            for l_global in is_motif_idx:
                l_local = l_global - query_start  # Local index in chunk
                key_indices = valid_indices[:, l_local, :]  # [B, k] - use clamped indices
                key_pos = motif_pos[key_indices]  # [B, k, 3]
                query_pos = motif_pos[l_global].unsqueeze(0).expand(B, -1)  # [B, 3]

                # Valid mask: both query and keys must be motif
                key_is_motif = is_motif[key_indices]  # [B, k]
                valid_mask = key_is_motif.unsqueeze(-1).float()  # [B, k, 1]

                if valid_mask.sum() > 0:
                    motif_pairs = self.motif_pos_embedder.compute_pairs_chunked(
                        query_pos, key_pos, valid_mask
                    )
                    P_LL_sparse[:, l_local, :, :] += motif_pairs

        # 2. Reference position embedding (if exists)
        if self.ref_pos_embedder is not None and "ref_pos" in f:
            ref_pos = f["ref_pos"]  # [L_full, 3]
            ref_space_uid = f["ref_space_uid"]  # [L_full]
            is_motif_seq = f["is_motif_atom_with_fixed_seq"]  # [L_full]
            is_motif_seq_idx = torch.where(is_motif_seq)[0]
            # Filter to only query positions in this chunk
            is_motif_seq_idx = is_motif_seq_idx[(is_motif_seq_idx >= query_start) & (is_motif_seq_idx < query_end)]
            for l_global in is_motif_seq_idx:
                l_local = l_global - query_start  # Local index in chunk
                key_indices = valid_indices[:, l_local, :]  # [B, k] - use clamped indices
                key_pos = ref_pos[key_indices]  # [B, k, 3]
                query_pos = ref_pos[l_global].unsqueeze(0).expand(B, -1)  # [B, 3]

                # Valid mask: same token and both have sequence
                key_space_uid = ref_space_uid[key_indices]  # [B, k]
                key_is_motif_seq = is_motif_seq[key_indices]  # [B, k]

                same_token = key_space_uid == ref_space_uid[l_global]  # [B, k]
                valid_mask = (
                    (same_token & key_is_motif_seq).unsqueeze(-1).float()
                )  # [B, k, 1]

                if valid_mask.sum() > 0:
                    ref_pairs = self.ref_pos_embedder.compute_pairs_chunked(
                        query_pos, key_pos, valid_mask
                    )
                    P_LL_sparse[:, l_local, :, :] += ref_pairs

        # 3. Single embedding terms (broadcasted)
        # Expand C_L to match batch dimension
        if C_L.shape[0] != B:
            C_L = C_L.expand(B, -1, -1)  # [B, L_full, c_token]
        
        # Get C_L for query positions in this chunk
        C_L_query_chunk = C_L[:, query_start:query_end, :]  # [B, L_par, c_token]
        
        # Expand for sparse attention: [B, L_par, k, c_token]
        C_L_queries = C_L_query_chunk.unsqueeze(2).expand(-1, -1, k, -1)  # [B, L_par, k, c_token]
        
        # Gather key features using sparse indices (indices point into full C_L)
        C_L_expanded = C_L.unsqueeze(2).expand(-1, -1, k, -1)  # [B, L_full, k, c_token]
        C_L_keys = torch.gather(
            C_L_expanded,
            1,
            valid_indices.unsqueeze(-1).expand(-1, -1, -1, C_L.shape[-1]),
        )  # [B, L_par, k, c_token]

        # Add single embeddings - match standard implementation structure
        single_l = self._get_process_single_l()(C_L_queries)  # [B, L_par, k, c_atompair]
        single_m = self._get_process_single_m()(C_L_keys)  # [B, L_par, k, c_atompair]
        P_LL_sparse += single_l + single_m

        # 4. Token pair features Z_init_II
        # Map atoms to tokens and gather token pair features
        # Handle tok_idx dimensions properly
        if tok_idx.dim() == 1:  # [L_full] - add batch dimension for consistency
            tok_idx_expanded = tok_idx.unsqueeze(0)  # [1, L_full]
        else:
            tok_idx_expanded = tok_idx

        # Expand tok_idx_expanded to match batch dimension
        if tok_idx_expanded.shape[0] != B:
            tok_idx_expanded = tok_idx_expanded.expand(B, -1)  # [B, L_full]
        
        # Get token indices for query positions in this chunk
        tok_idx_query_chunk = tok_idx_expanded[:, query_start:query_end]  # [B, L_par]
        tok_queries = tok_idx_query_chunk.unsqueeze(2).expand(-1, -1, k)  # [B, L_par, k]
        
        # Get token indices for key positions using sparse indices
        tok_idx_full_expanded = tok_idx_expanded.unsqueeze(2).expand(-1, -1, k)  # [B, L_full, k]
        tok_keys = torch.gather(tok_idx_full_expanded, 1, valid_indices)  # [B, L_par, k]

        # Gather Z_init_II[tok_queries, tok_keys] with safe indexing
        # Z_init_II shape is [I, I, c_z] (3D full) or [I_par, I, c_z] (3D chunked in streaming mode)
        # tok_queries shape: [B, L_par, k] - each value is a token index
        # We want: Z_init_II[tok_queries[d,l,k], tok_keys[d,l,k], :] for all d,l,k

        # Check for streaming mode via kwargs (Z_init_II is a chunked tensor)
        streaming_mode = kwargs.get("streaming_mode", False)
        z_chunk_range = kwargs.get("z_chunk_range")

        # DEBUG: Log streaming mode check
        import torch.distributed as dist
        if debug_ctx.stats_enabled and dist.is_initialized():
            rank = dist.get_rank()
            print(f"[DEBUG-STREAMING] RANK{rank}: streaming_mode={streaming_mode}, z_chunk_range={z_chunk_range}, "
                  f"Z_init_II.shape={list(Z_init_II.shape)}, kwargs.keys()={list(kwargs.keys())}", flush=True)

        if streaming_mode and z_chunk_range is not None:
            debug_log("PAIRWISE", "Z_init_II", f"CHUNKED TENSOR shape={list(Z_init_II.shape)}, range={z_chunk_range}")
            # Streaming mode: Z_init_II is pre-computed [I_par, I, c_z] chunk
            # z_chunk_range tells us which token rows this chunk covers
            start_i, end_i = z_chunk_range
            I_par_z = Z_init_II.shape[0]  # Chunk size
            I_z = Z_init_II.shape[1]      # Full I (all keys)

            Z_pairs_processed = torch.zeros(
                B, L_par, k, self.c_atompair, device=device, dtype=C_L.dtype
            )

            for b in range(B):
                tq = tok_queries[b]  # [L_par, k] - token indices in [0, I)
                tk = tok_keys[b]    # [L_par, k] - token indices in [0, I)

                # Map tok_queries to local chunk indices
                # Assumes tok_queries falls within [start_i, end_i) - alignment assumption
                local_tq = tq - start_i  # [L_par, k] - now in [0, I_par_z)

                # Clamp to valid ranges
                local_tq = torch.clamp(local_tq, 0, I_par_z - 1)
                tk = torch.clamp(tk, 0, I_z - 1)

                # Index into chunked Z tensor
                Z_pairs_full = Z_init_II[
                    local_tq.flatten(),
                    tk.flatten()
                ].view(L_par, k, -1)  # [L_par, k, c_z]

                if b == 0:
                    debug_log("PAIRWISE", "Z_pairs_chunked_tensor",
                              f"tq=[{tq.min().item()},{tq.max().item()}], "
                              f"local_tq=[{local_tq.min().item()},{local_tq.max().item()}], "
                              f"tk=[{tk.min().item()},{tk.max().item()}], "
                              f"mean={Z_pairs_full.float().mean().item():.6f}")
                    debug_elements("PAIRWISE", "Z_pairs_full", Z_pairs_full, [(0, 0), (100, 50)])

                Z_pairs_processed[b] = self._get_process_z()(Z_pairs_full)  # [L_par, k, c_atompair]
        else:
            debug_log("PAIRWISE", "Z_init_II", f"TENSOR shape={list(Z_init_II.shape)}")
            # Standard mode: full Z tensor available
            I_z, I_z2, c_z = Z_init_II.shape

            # DEBUG: Print Z_init_II stats BEFORE process_z
            debug_log("PAIRWISE", "Z_init_II_stats",
                      f"mean={Z_init_II.float().mean().item():.6f}, std={Z_init_II.float().std().item():.6f}")

            # CRITICAL: Match standard implementation exactly!
            # Standard does: self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
            # This means: 1) Process Z_init_II first, 2) Then do double token indexing

            # Step 1: Process Z_init_II to get processed token pair features
            Z_processed = self._get_process_z()(Z_init_II)  # [I, I, c_atompair]

            # Step 2: Do the double indexing like the standard implementation
            # Standard: Z_processed[..., tok_idx, :, :][..., tok_idx, :]
            # This creates Z_processed[tok_idx, :][:, tok_idx] which is [L, L, c_atompair]
            # Then we need to gather the sparse version

            Z_pairs_processed = torch.zeros(
                B, L_par, k, self.c_atompair, device=device, dtype=Z_processed.dtype
            )

            for b in range(B):
                # For this batch, get the token queries and keys
                tq = tok_queries[b]  # [L_par, k]
                tk = tok_keys[b]  # [L_par, k]

                # DIAGNOSTIC: Print pre-clamp tq range to detect silent clamping
                if b == 0:
                    tq_pre_min, tq_pre_max = tq.min().item(), tq.max().item()
                    n_clamped_tq = (tq >= I_z).sum().item()
                    debug_log("PAIRWISE", "Z_clamp_check",
                              f"I_z={I_z}, tq_pre_clamp=[{tq_pre_min},{tq_pre_max}], "
                              f"n_clamped={n_clamped_tq}/{tq.numel()} "
                              f"({'BUG: chunked Z used as full!' if n_clamped_tq > 0 else 'ok'})")

                # Ensure indices are within bounds
                tq = torch.clamp(tq, 0, I_z - 1)
                tk = torch.clamp(tk, 0, I_z2 - 1)

                # DEBUG: Print Z_pairs at same sparse indices BEFORE process_z
                if b == 0:
                    Z_pairs_unprocessed = Z_init_II[tq, tk]  # [L, k, c_z]
                    debug_log("PAIRWISE", "Z_pairs_tensor",
                              f"tq=[{tq.min().item()},{tq.max().item()}], "
                              f"tk=[{tk.min().item()},{tk.max().item()}], "
                              f"mean={Z_pairs_unprocessed.float().mean().item():.6f}")
                    debug_elements("PAIRWISE", "Z_pairs", Z_pairs_unprocessed, [(0, 0), (100, 50)])
                    # DIAGNOSTIC: Print EXACT token indices at positions [0,0] and [100,50]
                    if tq.shape[0] > 100 and tq.shape[1] > 50:
                        debug_log("PAIRWISE", "TENSOR_TOKEN_INDICES",
                                  f"[0,0] tq={tq[0,0].item()} tk={tk[0,0].item()} | "
                                  f"[100,50] tq={tq[100,50].item()} tk={tk[100,50].item()}")

                # Apply the double token indexing like standard implementation
                Z_pairs_processed[b] = Z_processed[tq, tk]  # [L, k, c_atompair]

        P_LL_sparse += Z_pairs_processed

        # 5. Final MLP - ADD the result, don't replace (to match standard implementation)
        P_LL_sparse = P_LL_sparse + self._get_pair_mlp()(P_LL_sparse)

        # DIAGNOSTIC: Log detailed intermediate values
        _log_tensor_stats("P_LL_sparse_forward_chunked", P_LL_sparse)
        _log_tensor_stats("forward_chunked_C_L_queries", C_L_queries)
        _log_tensor_stats("forward_chunked_C_L_keys", C_L_keys)
        _log_tensor_stats("forward_chunked_single_l", single_l)
        _log_tensor_stats("forward_chunked_single_m", single_m)
        _log_tensor_stats("forward_chunked_Z_pairs_processed", Z_pairs_processed)
        debug_log("PAIRWISE", "forward_chunked", f"query_start={query_start}, L_par={L_par}, L_full={L_full}")

        # DIAGNOSTIC: Split P_LL_sparse by token half to compare standard vs parallel chunks
        if debug_ctx.stats_enabled and L_par > 0:
            tok_idx_chunk = tok_idx[:L_par] if tok_idx.dim() == 1 else tok_idx[0, query_start:query_start+L_par]
            I_total = tok_idx_chunk.max().item() + 1
            I_mid = I_total // 2
            first_half_mask = tok_idx_chunk < I_mid   # atoms whose token < I/2
            second_half_mask = ~first_half_mask        # atoms whose token >= I/2
            n_first = first_half_mask.sum().item()
            n_second = second_half_mask.sum().item()
            if n_first > 0 and n_second > 0:
                p_first = P_LL_sparse[0, first_half_mask].float()
                p_second = P_LL_sparse[0, second_half_mask].float()
                debug_log("PAIRWISE", "P_LL_split_by_token_half",
                          f"I_mid={I_mid}, "
                          f"first_half(tok<{I_mid}): n={n_first}, mean={p_first.mean().item():.6f}, "
                          f"second_half(tok>={I_mid}): n={n_second}, mean={p_second.mean().item():.6f}")

        return P_LL_sparse.contiguous()
