import functools
import logging
import os
from contextlib import ExitStack
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from rfd3.model.layers.block_utils import (
    bucketize_scaled_distogram,
    bucketize_scaled_distogram_chunked,
    create_attention_indices,
)
from rfd3.model.layers.blocks import (
    CompactStreamingDecoder,
    Downcast,
    LinearEmbedWithPool,
    LinearSequenceHead,
    LocalAtomTransformer,
    LocalTokenTransformer,
)
from rfd3.model.layers.encoders import (
    DiffusionTokenEncoder,
)
from rfd3.model.layers.layer_utils import RMSNorm, linearNoBias
from rfd3.model.layers.streaming import compute_chunk_ranges
from rfd3.model.debug_context import debug_ctx, debug_tensor, debug_log, debug_tensor_all_ranks, debug_log_all_ranks, verify_tensor_sync

from foundry.model.layers.blocks import (
    FourierEmbedding,
)

logger = logging.getLogger(__name__)

# Diagnostic flag - set to True to enable detailed parallel debugging
PARALLEL_DEBUG = True  # Hardcoded for debugging


def _log_tensor_stats(name: str, tensor: torch.Tensor, rank: int = 0):
    """Log tensor statistics for debugging parallel vs non-parallel differences."""
    if not PARALLEL_DEBUG:
        return
    # Use centralized debug context for consistent formatting
    debug_tensor("MODEL", name, tensor, rank)


def _is_streaming_mode() -> bool:
    """
    Check if streaming/parallel mode is enabled.

    Env var scheme:
      - RFD3_ATTENTION_PARALLEL=0 or unset → standard mode (False)
      - RFD3_ATTENTION_PARALLEL=1 → parallel mode (True)
    """
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    return val == "1"


def _get_gpu_rank_and_world_size() -> tuple[int, int]:
    """
    Get current GPU rank and world size for distributed processing.
    
    Returns:
        (rank, world_size): Tuple of (GPU rank, total GPUs).
                           Returns (0, 1) if not in distributed mode.
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_gather_along_dim(
    tensor: torch.Tensor,
    world_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """
    Gather tensor chunks from all GPUs and concatenate along specified dimension.
    
    Args:
        tensor: Local tensor chunk
        world_size: Total number of GPUs
        dim: Dimension to concatenate along
        
    Returns:
        Concatenated tensor from all GPUs [... sum(chunk_sizes) ...]
    """
    if world_size == 1:
        return tensor
    
    # Ensure tensor is contiguous (required for NCCL)
    tensor = tensor.contiguous()
    
    # Gather from all ranks
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=dim)




def _get_gpu_rank_and_world_size():
    """Get current GPU rank and world size for distributed processing."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _all_gather_concat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Gather tensors from all GPUs and concatenate along specified dimension."""
    import torch.distributed as dist
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    return torch.cat(gathered, dim=dim)


class RFD3DiffusionModule(nn.Module):
    def __init__(
        self,
        *,
        c_atom,
        c_atompair,
        c_token,
        c_s,
        c_z,
        c_t_embed,
        sigma_data,
        f_pred,
        n_attn_seq_neighbours,
        n_attn_keys,
        n_recycle,
        atom_attention_encoder,
        diffusion_token_encoder,
        diffusion_transformer,
        atom_attention_decoder,
        # upcast,
        downcast,
        use_local_token_attention=True,
        **_,
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s = c_s
        self.c_z = c_z
        self.f_pred = f_pred
        self.n_attn_seq_neighbours = n_attn_seq_neighbours
        self.n_attn_keys = n_attn_keys
        self.use_local_token_attention = use_local_token_attention

        # Auxiliary
        self.process_r = linearNoBias(3, c_atom)
        self.to_r_update = nn.Sequential(RMSNorm((c_atom,)), linearNoBias(c_atom, 3))
        self.sequence_head = LinearSequenceHead(c_token=c_token)

        self.n_recycle = n_recycle
        self.n_bins = 65
        self.bucketize_fn = functools.partial(
            bucketize_scaled_distogram,
            min_dist=1,
            max_dist=30,
            sigma_data=1,
            n_bins=self.n_bins,
        )

        # Time processing
        self.fourier_embedding = nn.ModuleList(
            [FourierEmbedding(c_t_embed), FourierEmbedding(c_t_embed)]
        )
        self.process_n = nn.ModuleList(
            [
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_atom)),
                nn.Sequential(RMSNorm(c_t_embed), linearNoBias(c_t_embed, c_s)),
            ]
        )
        self.downcast_c = Downcast(c_atom=c_atom, c_token=c_s, c_s=None, **downcast)
        self.downcast_q = Downcast(c_atom=c_atom, c_token=c_token, c_s=c_s, **downcast)
        self.process_a = LinearEmbedWithPool(c_token)
        self.process_c = nn.Sequential(RMSNorm(c_atom), linearNoBias(c_atom, c_atom))

        # UNet-like architecture for processing across tokens and atoms
        self.encoder = LocalAtomTransformer(
            c_atom=c_atom, c_s=c_atom, c_atompair=c_atompair, **atom_attention_encoder
        )

        self.diffusion_token_encoder = DiffusionTokenEncoder(
            c_s=c_s,
            c_token=c_token,
            c_z=c_z,
            c_atompair=c_atompair,
            **diffusion_token_encoder,
        )

        self.diffusion_transformer = LocalTokenTransformer(
            c_token=c_token,
            c_tokenpair=c_z,
            c_s=c_s,
            **diffusion_transformer,
        )

        self.decoder = CompactStreamingDecoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            c_s=c_s,
            c_tokenpair=c_z,
            **atom_attention_decoder,
        )

    def scale_positions_in(self, X_noisy_L, t):
        if t.ndim == 1:
            t = t[..., None, None]  # [B, (n_atoms), (3)]
        elif t.ndim == 2:
            t = t[..., None]  # [B, n_atoms, (3)]

        if self.f_pred == "edm":
            R_noisy_L = X_noisy_L / torch.sqrt(t**2 + self.sigma_data**2)
        elif self.f_pred == "unconditioned":
            R_noisy_L = torch.zeros_like(X_noisy_L)
        elif self.f_pred == "noise_pred":
            R_noisy_L = X_noisy_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return R_noisy_L

    def scale_positions_out(self, R_update_L, X_noisy_L, t):
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]  # [B, n_atoms, (3)]

        if self.f_pred == "edm":
            X_out_L = (self.sigma_data**2 / (self.sigma_data**2 + t**2)) * X_noisy_L + (
                self.sigma_data * t / (self.sigma_data**2 + t**2) ** 0.5
            ) * R_update_L
        elif self.f_pred == "unconditioned":
            X_out_L = R_update_L
        elif self.f_pred == "noise_pred":
            X_out_L = X_noisy_L + R_update_L
        else:
            raise Exception(f"{self.f_pred=} unrecognized")
        return X_out_L

    def process_time_(self, t_L, i):
        C_L = self.process_n[i](
            self.fourier_embedding[i](
                1 / 4 * torch.log(torch.clamp(t_L, min=1e-20) / self.sigma_data)
            )
        )
        # Mask out zero-time features;
        C_L = C_L * (t_L > 0).float()[..., None]  # [B, L, C_atom]
        return C_L

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
        if gpu_rank == 0:
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
        A_I_chunk = self.diffusion_transformer.forward_cross_attn(
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
        A_I = _all_gather_along_dim(A_I_chunk, world_size, dim=1)  # [B, I, c_token]

        # MULTI-GPU DIAGNOSTIC: Verify A_I is synchronized after all_gather
        debug_tensor_all_ranks("DIFF_XATTN", "A_I_after_gather", A_I)
        verify_tensor_sync("DIFF_XATTN", "A_I_SYNC", A_I)

        return A_I

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

        # Call decoder with cross-attention
        A_I, Q_L, o = self.decoder.forward_parallel(
            A_I=A_I,                                       # [B, I, c_token]
            S_I=S_I,                                       # [I, c_s]
            Q_L=Q_L,                                       # [B, L, c_atom]
            C_L=C_L,                                       # [B, L, c_atom]
            P_chunk=P_LL_chunk,                            # [L_par, L, c_atompair]
            tok_idx=tok_idx,                               # [L]
            indices=f["attn_indices"],                     # [B, L, k]
            query_start=start_l,
            query_end=end_l,
            all_gather_fn=_all_gather_along_dim,
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

        # Call decoder with sparse P_LL computation for this GPU's chunk
        # The chunked_pairwise_embedder computes P_LL only for the (L_par, k) pairs
        A_I, Q_L, o = self.decoder.forward_parallel_sparse(
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
            all_gather_fn=_all_gather_along_dim,
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

            # Check for streaming mode via kwargs
            streaming_mode = kwargs.get("streaming_mode", False)
            z_chunk_range = kwargs.get("z_chunk_range")

            if streaming_mode and z_chunk_range is not None:
                # Streaming mode: Z_II is pre-computed [I_par, I, c_z] chunk
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

    def forward(
        self,
        X_noisy_L,
        t,
        f,
        # Features from initialization
        Q_L_init,
        C_L,
        P_LL,
        S_I,
        Z_II,
        n_recycle=None,
        # Chunked memory optimization parameters
        chunked_pairwise_embedder=None,
        initializer_outputs=None,
        streaming_mode=False,  # Flag from TokenInitializer
        **kwargs,
    ):
        """
        Diffusion forward pass with recycling.
        Computes denoised positions given encoded features and noisy coordinates.
        
        Args:
            X_noisy_L: [B, L, 3] noisy atom positions
            t: [B] noise levels
            f: Feature dictionary
            Q_L_init: [L, c_atom] initial atom features
            C_L: [L, c_atom] conditioned atom features
            P_LL: [L, L, c_atompair] or None (chunked/streaming mode)
            S_I: [I, c_s] token single features
            Z_II: [I, I, c_z] (full) OR [I_par, I, c_z] (chunked in streaming mode)
            n_recycle: Number of recycle iterations
            chunked_pairwise_embedder: ChunkedPairwiseEmbedder or None
            initializer_outputs: Dict with additional outputs
            streaming_mode: If True, Z_II is [I_par, I, c_z] (chunked tensor)
            
        Returns:
            dict with X_L, sequence_indices_I, sequence_logits_I
        """
        # ... Collect inputs
        # Ensure module and all inputs are on the same device for distributed processing
        device = X_noisy_L.device
        self.to(device)
        
        # Move input tensors to correct device
        C_L = C_L.to(device)
        S_I = S_I.to(device)
        Q_L_init = Q_L_init.to(device)
        t = t.to(device)
        
        # Move feature dict tensors to correct device
        for key, val in f.items():
            if isinstance(val, torch.Tensor):
                f[key] = val.to(device)
        
        tok_idx = f["atom_to_token_map"]                   # [L]
        L = len(tok_idx)
        I = tok_idx.max() + 1                              # Number of tokens

        # DIAGNOSTIC: Log input shapes at model entry (auto-detects mode from env)
        debug_ctx.auto_detect_mode()
        z_info = f"shape={list(Z_II.shape)}" if hasattr(Z_II, 'shape') else "tensor"
        p_info = f"shape={list(P_LL.shape)}" if P_LL is not None else "None (chunked)"
        debug_log("MODEL", "forward_START", f"L={L}, I={I}, X={list(X_noisy_L.shape)}, Z={z_info}, P={p_info}")

        # Check if we're in streaming mode (Z_II is [I_par, I, c_z] instead of [I, I, c_z])
        is_streaming = streaming_mode
        
        # Create attention indices
        f["attn_indices"] = create_attention_indices(
            X_L=X_noisy_L,
            f=f,
            n_attn_keys=self.n_attn_keys,
            n_attn_seq_neighbours=self.n_attn_seq_neighbours,
        )                                                  # [B, L, k] 

        # ... Expand t tensors
        t_L = t.unsqueeze(-1).expand(-1, L) * (
            ~f["is_motif_atom_with_fixed_coord"]
        ).float().unsqueeze(0)                             # [B, L]
        t_I = t.unsqueeze(-1).expand(-1, I) * (
            ~f["is_motif_token_with_fully_fixed_coord"]
        ).float().unsqueeze(0)                             # [B, I]

        # ... Create scaled positions
        R_L_uniform = self.scale_positions_in(X_noisy_L, t)  # [B, L, 3]
        R_noisy_L = self.scale_positions_in(X_noisy_L, t_L)  # [B, L, 3]

        # ... Pool initial representation to sequence level
        A_I = self.process_a(R_noisy_L, tok_idx=tok_idx)   # [B, I, c_token]
        S_I = self.downcast_c(C_L, S_I, tok_idx=tok_idx)   # [I, c_s]

        # ... Add batch-wise features to inputs
        Q_L = Q_L_init.unsqueeze(0) + self.process_r(R_noisy_L)  # [B, L, c_atom]
        C_L = C_L.unsqueeze(0) + self.process_time_(t_L, i=0)    # [B, L, c_atom]
        S_I = S_I.unsqueeze(0) + self.process_time_(t_I, i=1)    # [B, I, c_s]
        C_L = C_L + self.process_c(C_L)                          # [B, L, c_atom]

        # ... Run Local-Atom Self Attention and Pool
        # Parallel mode: =0 or unset → standard, =1 → parallel (GPU count auto-detected)
        attn_parallel = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
        attn_parallel_enabled = (
            attn_parallel == "1"
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

        if attn_parallel_enabled:
            # PARALLEL MODE: Split L atoms across GPUs
            # Each GPU processes L_par = L / n_gpus atom queries
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            L = Q_L.shape[1]
            L_par = (L + world_size - 1) // world_size
            query_start = rank * L_par
            query_end = min(query_start + L_par, L)
            
            def all_gather_concat(tensor, ws, dim):
                # =======================================================================
                # MEMORY OPTIMIZATION: Avoid creating ws separate tensors per GPU
                # Old code created O(ws) tensors, causing O(ws²) total memory with GPUs
                # New code uses all_gather_into_tensor for O(1) temporary allocations
                # =======================================================================

                # --- OLD IMPLEMENTATION ---
                # tensor = tensor.contiguous()
                # gathered = [torch.zeros_like(tensor) for _ in range(ws)]
                # dist.all_gather(gathered, tensor)
                # return torch.cat(gathered, dim=dim)
                # --- END OLD IMPLEMENTATION ---

                # --- NEW MEMORY-EFFICIENT IMPLEMENTATION ---
                tensor = tensor.contiguous()

                if hasattr(dist, 'all_gather_into_tensor'):
                    # Use single flat output tensor instead of list of ws tensors
                    flat_input = tensor.view(-1)
                    flat_output = torch.empty(flat_input.numel() * ws, dtype=tensor.dtype, device=tensor.device)
                    dist.all_gather_into_tensor(flat_output, flat_input)

                    # Reshape back to proper dimensions
                    chunk_shape = list(tensor.shape)
                    reshaped = flat_output.view(ws, *chunk_shape)

                    if dim == 0:
                        result = reshaped.view(-1, *chunk_shape[1:])
                    else:
                        perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, len(chunk_shape) + 1))
                        transposed = reshaped.permute(*perm)
                        final_shape = list(tensor.shape)
                        final_shape[dim] = final_shape[dim] * ws
                        result = transposed.contiguous().view(*final_shape)

                    del flat_input, flat_output, reshaped  # Explicit cleanup
                    return result
                else:
                    # Fallback with explicit cleanup
                    gathered = [torch.zeros_like(tensor) for _ in range(ws)]
                    dist.all_gather(gathered, tensor)
                    result = torch.cat(gathered, dim=dim)
                    del gathered  # Free ws tensors immediately
                    return result
                # --- END NEW IMPLEMENTATION ---
            
            # Use parallel encoder - each GPU handles L_par atoms
            Q_L = self.encoder.forward_parallel(
                Q_L=Q_L,
                C_L=C_L,
                indices=f["attn_indices"],
                query_start=query_start,
                query_end=query_end,
                f=f,
                chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
                all_gather_fn=all_gather_concat,
                world_size=world_size,
            )                                              # [B, L, c_atom]
            
            # Move all tensors and modules to this rank's device (Q_L is now on LOCAL_RANK's device)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            local_device = torch.device(f"cuda:{local_rank}")
            self.to(local_device)  # Move all sub-modules (downcast_q, etc.) to local device
            tok_idx = tok_idx.to(local_device)
            A_I = A_I.to(local_device)
            S_I = S_I.to(local_device)
            C_L = C_L.to(local_device)
            X_noisy_L = X_noisy_L.to(local_device)
            R_L_uniform = R_L_uniform.to(local_device)
            t_L = t_L.to(local_device)
        elif chunked_pairwise_embedder is not None:
            # Low-memory mode: sparse P_LL but single GPU
            Q_L = self.encoder(
                Q_L, C_L, P_LL=None, indices=f["attn_indices"],
                f=f, chunked_pairwise_embedder=chunked_pairwise_embedder,
                initializer_outputs=initializer_outputs,
            )                                              # [B, L, c_atom]
        else:
            # Standard mode: use full P_LL
            Q_L = self.encoder(Q_L, C_L, P_LL, indices=f["attn_indices"])  # [B, L, c_atom]
        
        # DIAGNOSTIC: Log Q_L after encoder
        _log_tensor_stats("Q_L_after_encoder", Q_L)
        
        A_I = self.downcast_q(Q_L, A_I=A_I, S_I=S_I, tok_idx=tok_idx)  # [B, I, c_token]
        
        # DIAGNOSTIC: Log A_I after downcast_q
        _log_tensor_stats("A_I_after_downcast_q", A_I)

        # ... Run forward with recycling
        recycled_features = self.forward_with_recycle(
            n_recycle,
            X_noisy_L=X_noisy_L,
            R_L_uniform=R_L_uniform,
            t_L=t_L,
            f=f,
            Q_L=Q_L,
            C_L=C_L,
            P_LL=P_LL,
            A_I=A_I,
            S_I=S_I,
            Z_II=Z_II,
            chunked_pairwise_embedder=chunked_pairwise_embedder,
            initializer_outputs=initializer_outputs,
            streaming_mode=is_streaming,
        )

        # ... Collect outputs
        outputs = {
            "X_L": recycled_features["X_L"],               # [B, L, 3] denoised positions
            "sequence_indices_I": recycled_features["sequence_indices_I"],  # [B, I]
            "sequence_logits_I": recycled_features["sequence_logits_I"],    # [B, I, vocab]
        }
        return outputs

    def forward_with_recycle(
        self,
        n_recycle,
        streaming_mode=False,
        **kwargs,
    ):
        """
        Run forward pass with recycling.
        
        Args:
            n_recycle: Number of recycle iterations (None = use self.n_recycle)
            streaming_mode: If True, Z_II is handled as [I_par, I, c_z] chunk
            **kwargs: Passed to process_()
        """
        if not self.training:
            n_recycle = self.n_recycle
        else:
            assert n_recycle is not None

        recycled_features = {}
        for i in range(n_recycle):
            with ExitStack() as stack:
                last = not (i < n_recycle - 1)
                if not last:
                    stack.enter_context(torch.no_grad())

                # Clear the autocast cache if gradients are enabled (workaround for autocast bug)
                # See: https://github.com/pytorch/pytorch/issues/65766
                if torch.is_grad_enabled():
                    torch.clear_autocast_cache()

                # Run forward
                recycled_features = self.process_(
                    D_II_self=recycled_features.get("D_II_self"),
                    X_L_self=recycled_features.get("X_L"),
                    streaming_mode=streaming_mode,
                    **kwargs,
                )

                # =======================================================================
                # MEMORY OPTIMIZATION: Clear CUDA cache between recycle iterations
                # In multi-GPU streaming mode, memory fragmentation builds up across
                # recycle iterations. empty_cache() helps defragment between iterations.
                # =======================================================================
                if streaming_mode and not last:
                    torch.cuda.empty_cache()

        return recycled_features

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
        streaming_mode=False,
        **kwargs,
    ):
        """
        Single recycling step.
        
        Args:
            D_II_self: [B, I, I, n_bins] self-conditioning distogram or None
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
            streaming_mode: If True, use streaming Z processing
            
        Returns:
            dict with X_L, D_II_self, sequence_logits_I, sequence_indices_I
        """
        # Determine if Z_II is streaming (Z_II is [I_par, I, c_z] instead of [I, I, c_z])
        is_streaming = streaming_mode

        # Get z_chunk_range from kwargs or initializer_outputs
        z_chunk_range = kwargs.get("z_chunk_range")
        if z_chunk_range is None and initializer_outputs is not None:
            z_chunk_range = initializer_outputs.get("z_chunk_range")

        # ... Embed token level features with atom level encodings
        S_I, Z_II = self.diffusion_token_encoder(
            f=f,
            R_L=R_L_uniform,                               # [B, L, 3]
            D_II_self=D_II_self,                           # [B, I, I, n_bins] or None
            S_init_I=S_I,                                  # [B, I, c_s]
            Z_init_II=Z_II,                                # [I, I, c_z] or [I_par, I, c_z]
            C_L=C_L,                                       # [B, L, c_atom]
            P_LL=P_LL,                                     # [L, L, c_atompair] or None
            streaming_mode=streaming_mode,
            z_chunk_range=z_chunk_range,
        )                                                  # Returns: [I, c_s], [I, I, c_z] or [I_par, I, c_z]

        # Determine full mode for transformer
        gpu_rank, world_size = _get_gpu_rank_and_world_size()
        use_full_attention = not (
            os.environ.get("RFD3_LOW_MEMORY_MODE", None) == "1" or _is_streaming_mode()
        )
        
        # Check if Z_II is chunked (from streaming encoder)
        # In multi-GPU mode, Z_II is [I_par, I, c_z] per GPU, not full [I, I, c_z]
        z_is_chunked = is_streaming and world_size > 1

        # ... Diffusion transformer with GPU-parallel attention
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
            # Standard mode
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

        # DIAGNOSTIC: Log A_I after diffusion transformer
        _log_tensor_stats("A_I_after_transformer", A_I)
        
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

        # DIAGNOSTIC: Log Q_L after decoder
        _log_tensor_stats("Q_L_after_decoder", Q_L)
        
        # ... Process outputs to positions update
        R_update_L = self.to_r_update(Q_L)                 # [B, L, 3]
        
        # DIAGNOSTIC: Log R_update_L (the model's prediction)
        _log_tensor_stats("R_update_L", R_update_L)
        
        X_out_L = self.scale_positions_out(R_update_L, X_noisy_L, t_L)  # [B, L, 3]
        
        # DIAGNOSTIC: Log X_out_L (final denoised positions)
        _log_tensor_stats("X_out_L", X_out_L)

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
                # The downstream code (encoders._forward_streaming) immediately slices
                # D_II_self back to [I_par, I], so gathering is wasteful.
                # For I=9000 (length 150), full D_II_self = 21 GB per GPU!
                # Keeping it chunked saves massive memory.
                # =======================================================================

                # --- OLD IMPLEMENTATION (gathered full I×I tensor) ---
                # D_II_self = _all_gather_along_dim(D_II_self_chunk, world_size, dim=1)
                # --- END OLD IMPLEMENTATION ---

                # --- NEW IMPLEMENTATION: Keep D_II_self as chunk ---
                # Each GPU keeps its own [B, I_par, I, n_bins] chunk
                # encoders._forward_streaming will use it directly without slicing
                D_II_self = D_II_self_chunk  # [B, I_par, I, n_bins] - NOT gathered!
                # --- END NEW IMPLEMENTATION ---
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
            "D_II_self": D_II_self,                        # [B, I, I, n_bins] or None
            "sequence_logits_I": sequence_logits_I,        # [B, I, vocab]
            "sequence_indices_I": sequence_indices_I,      # [B, I]
        } | o
