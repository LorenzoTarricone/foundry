import functools
import logging
import os
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
from rfd3.model.layers.block_utils import (
    bucketize_scaled_distogram,
    bucketize_scaled_distogram_chunked,
    pairwise_mean_pool,
)
from rfd3.model.layers.blocks import (
    Downcast,
    LocalAtomTransformer,
    OneDFeatureEmbedder,
    PositionPairDistEmbedder,
    RelativePositionEncodingWithIndexRemoval,
    SinusoidalDistEmbed,
)
from rfd3.model.layers.chunked_pairwise import (
    ChunkedPairwiseEmbedder,
    ChunkedPositionPairDistEmbedder,
    ChunkedSinusoidalDistEmbed,
)
from rfd3.model.layers.layer_utils import (
    RMSNorm,
    Transition,
    linearNoBias,
)
from rfd3.model.layers.pairformer_layers import PairformerBlock
from rfd3.model.parallel.utils import compute_chunk_ranges

from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing
from rfd3.model.debug_context import (
    debug_ctx, debug_tensor, debug_log, debug_tensor_all_ranks, debug_log_all_ranks,
    verify_tensor_sync, debug_memory, debug_gpu_memory_snapshot, debug_chunking, debug_time,
    timed_all_gather
)

logger = logging.getLogger(__name__)

# =============================================================================
# EXTRA_CHUNKING: Control sequential loop behavior in parallel mode
# =============================================================================
# Default False = vectorized operation (fast, O(1) scaling, more memory)
# Set True = chunked operation (slower, O(N) scaling, less memory)
#
# In parallel mode, the chunked functions loop over the full I dimension which
# causes O(N) time scaling. Setting EXTRA_CHUNKING=False (default) bypasses
# these loops and uses vectorized operations instead.
# =============================================================================
EXTRA_CHUNKING = os.environ.get("RFD3_EXTRA_CHUNKING", "0") == "1"


def _is_parallel_mode() -> bool:
    """
    Check if streaming/parallel mode is enabled.

    Env var scheme:
      - RFD3_ATTENTION_PARALLEL=0 or unset → standard mode (False)
      - RFD3_ATTENTION_PARALLEL=1 or any non-zero value → parallel mode (True)

    NOTE: design_parallel.py sets this to world_size (e.g., "4" for 4 GPUs),
    so we check for any truthy non-zero value, not just "1".
    """
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    # Enable streaming if value is any non-zero number (1, 2, 4, etc.)
    return val not in ("0", "", "false", "False")

def _get_world_size() -> int:
    """Get actual GPU count from distributed runtime."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_world_size()
    return 1




def _z_transition_chunked(Z: torch.Tensor, transition_fn, key_chunk: int = 512, extra_chunking: bool = None) -> torch.Tensor:
    """
    Apply z_transition, optionally with chunking for memory efficiency.

    By default (extra_chunking=False), uses a single vectorized operation for O(1) scaling.
    If extra_chunking=True, processes in sub-chunks to reduce peak memory (O(N) scaling).

    Args:
        Z: [I_par, I, c_z] or [B, I_par, I, c_z]
        transition_fn: z_transition module
        key_chunk: chunk size along key dimension (only used if extra_chunking=True)
        extra_chunking: If True, use chunked processing. If None, uses EXTRA_CHUNKING env var.

    Returns:
        Z + transition_fn(Z)
    """
    # Determine if chunking is enabled (default: use global EXTRA_CHUNKING)
    use_chunking = extra_chunking if extra_chunking is not None else EXTRA_CHUNKING

    # DEFAULT: Vectorized operation (fast, O(1) scaling)
    if not use_chunking or Z.dim() not in [3, 4]:
        if debug_ctx.memory_enabled and Z.dim() in [3, 4]:
            I = Z.shape[1] if Z.dim() == 3 else Z.shape[2]
            z_mem_gb = Z.numel() * Z.element_size() / 1e9
            # SwiGLU creates 4x intermediate: c_z -> 4*c_z -> c_z
            swiglu_peak_gb = Z.numel() * 4 * Z.element_size() / 1e9  # 4x expansion
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                free_mem, total_mem = torch.cuda.mem_get_info()
                free_gb = free_mem / 1e9
                debug_chunking(f"_z_transition_chunked: VECTORIZED, I={I}")
                debug_chunking(f"  Z shape={list(Z.shape)}, Z_mem={z_mem_gb:.2f}GB")
                debug_chunking(f"  SwiGLU 4x expansion will need ~{swiglu_peak_gb:.2f}GB")
                debug_chunking(f"  GPU: allocated={allocated:.2f}GB, free={free_gb:.2f}GB")
                if swiglu_peak_gb > free_gb:
                    debug_chunking(f"  WARNING: SwiGLU peak ({swiglu_peak_gb:.2f}GB) > free ({free_gb:.2f}GB) - OOM likely!")
                    # Take a full memory snapshot to see what's consuming memory
                    debug_gpu_memory_snapshot("OOM_DEBUG", f"before_z_transition_I={I}", min_size_mb=50.0)
            else:
                debug_chunking(f"_z_transition_chunked: VECTORIZED, I={I}, Z_mem={z_mem_gb:.2f}GB, SwiGLU_peak={swiglu_peak_gb:.2f}GB")
        return Z + transition_fn(Z)

    # OPTIONAL: Chunked operation (slow, O(N) scaling, less memory)
    if Z.dim() == 3:
        # [I_par, I, c_z]
        I = Z.shape[1]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"_z_transition_chunked(3D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, k_start:k_end, :]
            out_chunks.append(Z_sub + transition_fn(Z_sub))
        result = torch.cat(out_chunks, dim=1)
        del out_chunks
        return result
    else:  # Z.dim() == 4
        # [B, I_par, I, c_z]
        I = Z.shape[2]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"_z_transition_chunked(4D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, :, k_start:k_end, :]
            out_chunks.append(Z_sub + transition_fn(Z_sub))
        result = torch.cat(out_chunks, dim=2)
        del out_chunks
        return result


def _process_z_chunked(Z: torch.Tensor, process_fn, key_chunk: int = 512, extra_chunking: bool = None) -> torch.Tensor:
    """
    Apply process_z (Linear), optionally with chunking for memory efficiency.

    By default (extra_chunking=False), uses a single vectorized operation for O(1) scaling.
    If extra_chunking=True, processes in sub-chunks to reduce peak memory (O(N) scaling).

    Args:
        Z: [I_par, I, c_in] or [B, I_par, I, c_in]
        process_fn: process_z module (typically nn.Sequential with RMSNorm + Linear)
        key_chunk: chunk size along key dimension (only used if extra_chunking=True)
        extra_chunking: If True, use chunked processing. If None, uses EXTRA_CHUNKING env var.

    Returns:
        process_fn(Z)
    """
    # Determine if chunking is enabled (default: use global EXTRA_CHUNKING)
    use_chunking = extra_chunking if extra_chunking is not None else EXTRA_CHUNKING

    # DEFAULT: Vectorized operation (fast, O(1) scaling)
    if not use_chunking or Z.dim() not in [3, 4]:
        if debug_ctx.memory_enabled and Z.dim() in [3, 4]:
            I = Z.shape[1] if Z.dim() == 3 else Z.shape[2]
            z_mem_gb = Z.numel() * Z.element_size() / 1e9
            # process_fn is typically RMSNorm + Linear, output similar size to input
            # But the Linear may have different output dimension - estimate 2x peak
            peak_gb = z_mem_gb * 2
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                free_mem, total_mem = torch.cuda.mem_get_info()
                free_gb = free_mem / 1e9
                debug_chunking(f"_process_z_chunked: VECTORIZED, I={I}")
                debug_chunking(f"  Z shape={list(Z.shape)}, Z_mem={z_mem_gb:.2f}GB")
                debug_chunking(f"  Estimated peak ~{peak_gb:.2f}GB")
                debug_chunking(f"  GPU: allocated={allocated:.2f}GB, free={free_gb:.2f}GB")
            else:
                debug_chunking(f"_process_z_chunked: VECTORIZED, I={I}, Z_mem={z_mem_gb:.2f}GB")
        return process_fn(Z)

    # OPTIONAL: Chunked operation (slow, O(N) scaling, less memory)
    if Z.dim() == 3:
        # [I_par, I, c_in]
        I = Z.shape[1]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"_process_z_chunked(3D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, k_start:k_end, :]
            out_chunks.append(process_fn(Z_sub))
            del Z_sub
        result = torch.cat(out_chunks, dim=1)
        del out_chunks
        return result
    else:  # Z.dim() == 4
        # [B, I_par, I, c_in]
        I = Z.shape[2]
        n_chunks = (I + key_chunk - 1) // key_chunk
        if debug_ctx.memory_enabled:
            debug_chunking(f"_process_z_chunked(4D): CHUNKED, I={I}, chunk={key_chunk}, n_iter={n_chunks} (O(N) scaling!)")
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, :, k_start:k_end, :]
            out_chunks.append(process_fn(Z_sub))
            del Z_sub
        result = torch.cat(out_chunks, dim=2)
        del out_chunks
        return result


def _get_gpu_rank_and_world_size() -> Tuple[int, int]:
    """Get current GPU rank and world size for distributed processing."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _compute_gpu_query_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Compute the query index range for a specific GPU.

    Uses compute_chunk_ranges to ensure consistent chunk distribution
    across all code paths (encoder, transformer, decoder).

    Args:
        total: Total number of queries (I tokens)
        rank: This GPU's rank (0 to world_size-1)
        world_size: Total number of GPUs

    Returns:
        (start_idx, end_idx): Query range [start, end) for this GPU
    """
    chunk_ranges = compute_chunk_ranges(total, world_size)
    if rank >= len(chunk_ranges):
        return total, total  # No tokens for this rank
    return chunk_ranges[rank]


def _all_gather_concat(tensor: torch.Tensor, dim: int = 0, total_size: int = None) -> torch.Tensor:
    """
    Gather tensors from all GPUs and concatenate along specified dimension.

    IMPORTANT: Handles uneven chunk sizes across GPUs. When dividing N elements
    across W GPUs, the last GPU may have fewer elements (N % W != 0). This function
    pads the smaller chunk before gathering and slices to the correct total size.

    Args:
        tensor: Local tensor chunk to gather
        dim: Dimension to concatenate along
        total_size: The expected total size along `dim` after gathering.
                    If provided, the output is sliced to this exact size.
                    This handles uneven chunk sizes (e.g., 13800 / 7 GPUs).

    Returns:
        Full tensor reassembled from all GPUs (size = total_size along dim)
    """
    import torch.distributed as dist
    import os

    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if world_size == 1:
        return tensor

    # Ensure tensor is on the correct device for this rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    expected_device = torch.device(f"cuda:{local_rank}")
    if tensor.device != expected_device:
        tensor = tensor.to(expected_device)

    # ===========================================================================
    # HANDLE UNEVEN CHUNK SIZES: When dividing N by W, the last chunk may be smaller
    # e.g., 13800 / 7 = 1971.4, so ranks 0-5 get 1972, rank 6 gets 1968
    # Solution: Find max chunk size, pad smaller chunks, gather, then slice
    # ===========================================================================

    local_size = tensor.shape[dim]

    # Gather all chunk sizes to find the max
    local_size_tensor = torch.tensor([local_size], dtype=torch.long, device=tensor.device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    timed_all_gather(all_sizes, local_size_tensor,
                     "ALL_GATHER", "encoders._all_gather_concat.sizes",
                     gather_type="all_gather")
    all_sizes = [s.item() for s in all_sizes]
    max_size = max(all_sizes)
    actual_total = sum(all_sizes)

    # Pad tensor if needed (only the last rank typically needs padding)
    if local_size < max_size:
        pad_size = max_size - local_size
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)

    # Ensure tensor is contiguous (required for NCCL)
    tensor = tensor.contiguous()

    # Now all tensors have the same size (max_size), gather them
    if hasattr(dist, 'all_gather_into_tensor'):
        # Flatten for all_gather_into_tensor
        flat_input = tensor.contiguous().view(-1)
        flat_output = torch.empty(flat_input.numel() * world_size, dtype=tensor.dtype, device=tensor.device)
        timed_all_gather(flat_output, flat_input,
                         "ALL_GATHER", "encoders._all_gather_concat.data",
                         gather_type="all_gather_into_tensor")

        # Reshape: split into world_size chunks, then merge along dim
        chunk_shape = list(tensor.shape)  # [max_size, ...]
        reshaped = flat_output.view(world_size, *chunk_shape)

        if dim == 0:
            output_tensor = reshaped.view(-1, *chunk_shape[1:])
        else:
            perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, len(chunk_shape) + 1))
            transposed = reshaped.permute(*perm)
            final_shape = list(tensor.shape)
            final_shape[dim] = final_shape[dim] * world_size
            output_tensor = transposed.contiguous().view(*final_shape)

        del flat_input, flat_output, reshaped
    else:
        # Fallback for older PyTorch
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        timed_all_gather(gathered, tensor,
                         "ALL_GATHER", "encoders._all_gather_concat.data_fallback",
                         gather_type="all_gather")
        output_tensor = torch.cat(gathered, dim=dim)
        del gathered

    # ===========================================================================
    # SLICE TO CORRECT SIZE: Remove padding to get exact total_size
    # If total_size is provided, use it; otherwise use the sum of actual chunk sizes
    # ===========================================================================
    final_size = total_size if total_size is not None else actual_total
    current_size = output_tensor.shape[dim]

    if current_size > final_size:
        # Slice to remove padding (take first `final_size` elements along dim)
        indices = [slice(None)] * output_tensor.dim()
        indices[dim] = slice(0, final_size)
        output_tensor = output_tensor[tuple(indices)].contiguous()

    return output_tensor


# NOTE: StreamingZContainer has been removed. Z chunks are now computed
# directly in TokenInitializer._forward_parallel() and returned as tensors.
# The chunked Z tensor [I_par, I, c_z] is passed through with parallel_mode=True
# and z_chunk_range=(start_i, end_i) kwargs.


class TokenInitializer(nn.Module):
    """
    Token embedding module for RFD3.

    Supports three modes:
    1. Standard mode: Full L×L and I×I tensors materialized
    2. Chunked mode (use_chunked_pll): Sparse P_LL for attention
    3. Streaming mode (RFD3_ATTENTION_PARALLEL): No full I×I/L×L tensors ever
       - Returns Z_II as [I_par, I, c_z] chunk tensor with z_chunk_range
       - All downstream modules must support streaming
    """

    def __init__(
        self,
        c_s,
        c_z,
        c_atom,
        c_atompair,
        relative_position_encoding,
        n_pairformer_blocks,
        pairformer_block,
        downcast,
        token_1d_features,
        atom_1d_features,
        atom_transformer,
        use_chunked_pll=False,  # Memory optimization for P_LL
    ):
        super().__init__()
        
        # Store dimensions
        self.c_s = c_s
        self.c_z = c_z

        # Store mode flags
        self.use_chunked_pll = use_chunked_pll

        # Features
        self.atom_1d_embedder_1 = OneDFeatureEmbedder(atom_1d_features, c_s)
        self.atom_1d_embedder_2 = OneDFeatureEmbedder(atom_1d_features, c_atom)
        self.token_1d_embedder = OneDFeatureEmbedder(token_1d_features, c_s)

        #REMARK: cross attention done in Downcast
        self.downcast_atom = Downcast(c_atom=c_s, c_token=c_s, c_s=None, **downcast)
        self.transition_post_token = Transition(c=c_s, n=2)
        self.transition_post_atom = Transition(c=c_s, n=2)
        self.process_s_init = nn.Sequential(
            RMSNorm(c_s),
            linearNoBias(c_s, c_s),
        )

        # Operations to mix into Z_II and S_I
        self.to_z_init_i = linearNoBias(c_s, c_z)
        self.to_z_init_j = linearNoBias(c_s, c_z)
        #REMARK: Should have dimension I x I x c_z
        self.relative_position_encoding = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.relative_position_encoding2 = RelativePositionEncodingWithIndexRemoval(
            c_z=c_z, **relative_position_encoding
        )
        self.process_token_bonds = linearNoBias(1, c_z)

        # Processing of Z_init
        self.process_z_init = nn.Sequential(
            RMSNorm(c_z * 2),
            linearNoBias(c_z * 2, c_z),
        )
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )
        #REMARK: Should have dimension I x I x c_z. Forward processes atoms distances ( L x L x 3)
        self.ref_pos_embedder_tok = PositionPairDistEmbedder(c_z, embed_frame=False)

        # Pairformer without triangle updates
        self.transformer_stack = nn.ModuleList(
            [
                #REMARK: Should have dimension I x I x c_z
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )

        #############################################################################
        # Token track processing
        self.process_s_trunk = nn.Sequential(RMSNorm(c_s), linearNoBias(c_s, c_atom))
        self.process_single_l = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_single_m = nn.Sequential(
            nn.ReLU(), linearNoBias(c_atom, c_atompair)
        )
        self.process_z = nn.Sequential(RMSNorm(c_z), linearNoBias(c_z, c_atompair))

        # ALWAYS create these MLPs - they will be shared between chunked and standard modes
        #REMARK: Should have dimension L x L x c_atompair
        self.motif_pos_embedder = SinusoidalDistEmbed(c_atompair=c_atompair)
        #REMARK: Should have dimension L x L x c_atompair
        self.ref_pos_embedder = PositionPairDistEmbedder(c_atompair, embed_frame=False)
        #REMARK: Should have dimension L x L x c_atompair
        self.pair_mlp = nn.Sequential(
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            linearNoBias(c_atompair, c_atompair),
        )

        # Atom pair feature processing
        if self.use_chunked_pll:
            # Initialize chunked embedders and share the trained MLPs!
            self.chunked_pairwise_embedder = ChunkedPairwiseEmbedder(
                c_atompair=c_atompair,
                motif_pos_embedder=ChunkedSinusoidalDistEmbed(c_atompair=c_atompair),
                ref_pos_embedder=ChunkedPositionPairDistEmbedder(
                    c_atompair, embed_frame=False
                ),
                process_single_l=self.process_single_l,  # Share trained parameters!
                process_single_m=self.process_single_m,  # Share trained parameters!
                process_z=self.process_z,  # Share trained parameters!
                pair_mlp=self.pair_mlp,  # Share trained parameters!
            )
        self.process_pll = linearNoBias(c_atompair, c_atompair)
        self.project_pll = linearNoBias(c_atompair, c_z)

        if atom_transformer["n_blocks"] > 0:
            self.atom_transformer = LocalAtomTransformer(
                c_atom=c_atom, c_s=None, c_atompair=c_atompair, **atom_transformer
            )
        else:
            self.atom_transformer = None

        # Post-processing
        # self.process_s_post = nn.Sequential(
        #     RMSNorm(c_s),
        #     linearNoBias(c_s, c_s),
        # )
        # self.process_z_post = nn.Sequential(
        #     RMSNorm(c_z),
        #     linearNoBias(c_z, c_z),
        # )

    def sync_chunked_embedder_weights(self):
        """Copy weights from standard embedders to chunked embedders.

        Called after checkpoint loading to ensure chunked embedders use trained
        weights, since the checkpoint doesn't contain chunked embedder parameters
        (they are new modules not present in the original training run).

        The standard and chunked embedder classes have identical layer structures:
          - SinusoidalDistEmbed / ChunkedSinusoidalDistEmbed: output_proj, process_valid_mask
          - PositionPairDistEmbedder / ChunkedPositionPairDistEmbedder: process_inverse_dist, process_valid_mask
        """
        if not self.use_chunked_pll:
            return

        chunked = self.chunked_pairwise_embedder

        # motif_pos_embedder: SinusoidalDistEmbed -> ChunkedSinusoidalDistEmbed
        chunked.motif_pos_embedder.output_proj.weight.data.copy_(
            self.motif_pos_embedder.output_proj.weight.data
        )
        chunked.motif_pos_embedder.process_valid_mask.weight.data.copy_(
            self.motif_pos_embedder.process_valid_mask.weight.data
        )

        # ref_pos_embedder: PositionPairDistEmbedder -> ChunkedPositionPairDistEmbedder
        chunked.ref_pos_embedder.process_inverse_dist.weight.data.copy_(
            self.ref_pos_embedder.process_inverse_dist.weight.data
        )
        chunked.ref_pos_embedder.process_valid_mask.weight.data.copy_(
            self.ref_pos_embedder.process_valid_mask.weight.data
        )

        print("[TokenInitializer] Synced chunked embedder weights from standard embedders.", flush=True)

    def forward(self, f):
        """
        Provides initial representation for atom and token representations.

        Returns:
            dict containing:
                - Q_L_init: [L, c_atom] initial atom features
                - C_L: [L, c_atom] conditioned atom features
                - S_I: [I, c_s] token single features
                - Z_II: [I, I, c_z] (full) or [I_par, I, c_z] (chunked in streaming mode)
                - P_LL: [L, L, c_atompair] (standard mode only)
                - chunked_pairwise_embedder: (chunked/streaming mode only)
        """
        tok_idx = f["atom_to_token_map"]                  # [L]
        L = len(tok_idx)
        f["ref_atom_name_chars"] = f["ref_atom_name_chars"].reshape(L, -1)
        I = len(f["restype"])

        # DIAGNOSTIC: Print streaming mode status and env var
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        env_val = os.environ.get("RFD3_ATTENTION_PARALLEL", "NOT_SET")

        # DIAGNOSTIC: Print feature sizes to debug dimension mismatch
        if rank == 0 and debug_ctx.stats_enabled:
            print(f"[Rank {rank}] TokenInitializer.forward: L={L}, I=len(f['restype'])={I}", flush=True)
            print(f"[Rank {rank}]   f['token_bonds'].shape = {f['token_bonds'].shape}", flush=True)
            print(f"[Rank {rank}]   f['is_ca'].sum() = {f['is_ca'].sum().item()}", flush=True)
            print(f"[Rank {rank}]   f['asym_id'].shape = {f['asym_id'].shape if 'asym_id' in f else 'N/A'}", flush=True)
        debug_memory("TOKEN_INIT", f"forward_start_L{L}_I{I}")

        return self._forward_standard(f, tok_idx, L, I)

    def _forward_standard(self, f, tok_idx, L, I):
        """
        Standard forward: may materialize full I×I and L×L tensors.
        """
        def init_tokens():
            # Embed token features
            S_I = self.token_1d_embedder(f, I)            # [I, c_s]
            S_I = S_I + self.transition_post_token(S_I)   # [I, c_s]

            # Embed atom features and downcast to token features
            S_I = self.downcast_atom(
                Q_L=self.atom_1d_embedder_1(f, L),        # [L, c_s]
                A_I=S_I,                                   # [I, c_s]
                tok_idx=tok_idx                            # [L]
            )                                              # [I, c_s]
            S_I = S_I + self.transition_post_atom(S_I)    # [I, c_s]
            S_I = self.process_s_init(S_I)                # [I, c_s]

            # Embed Z_II - THIS CREATES FULL I×I TENSOR
            Z_init_II = self.to_z_init_i(S_I).unsqueeze(-3) + self.to_z_init_j(
                S_I
            ).unsqueeze(-2)                                # [I, I, c_z]
            
            # DEBUG: Print step 1 Z stats (compare with streaming mode)
            # Also print mean of first half for direct comparison with parallel GPU 0
            I_half = Z_init_II.shape[0] // 2
            debug_log("ENCODER", "Z_init_step1",
                      f"Z_i+Z_j mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            Z_init_II = Z_init_II + self.relative_position_encoding(f)  # [I, I, c_z]
            debug_log("ENCODER", "Z_init_step2_rpe",
                      f"after RPE mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            Z_init_II = Z_init_II + self.process_token_bonds(
                f["token_bonds"].unsqueeze(-1).float()
            )                                              # [I, I, c_z]
            debug_log("ENCODER", "Z_init_step3_bonds",
                      f"after token_bonds mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            # Embed reference coordinates of ligands
            token_id = f["ref_space_uid"][f["is_ca"]]     # [I]
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)).unsqueeze(
                -1
            )                                              # [I, I, 1]
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(
                f["ref_pos"][f["is_ca"]], valid_mask
            )                                              # [I, I, c_z]
            debug_log("ENCODER", "Z_init_step4_refpos",
                      f"after ref_pos mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            # Run a small transformer to provide position encodings to single.
            for block_idx, block in enumerate(self.transformer_stack):
                S_I, Z_init_II = block(S_I, Z_init_II)    # [I, c_s], [I, I, c_z]
                debug_log("ENCODER", f"Z_init_step5_block{block_idx}",
                          f"after block mean={Z_init_II.float().mean():.6f}, "
                          f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            # Also cat the relative position encoding and mix
            Z_init_II = torch.cat(
                [
                    Z_init_II,
                    self.relative_position_encoding2(f),
                ],
                dim=-1,
            )                                              # [I, I, 2*c_z]
            debug_log("ENCODER", "Z_init_step6_rpe2cat",
                      f"after rpe2 concat mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            Z_init_II = self.process_z_init(Z_init_II)    # [I, I, c_z]
            debug_log("ENCODER", "Z_init_step7_processzinit",
                      f"after process_z_init mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)  # [I, I, c_z]
                debug_log("ENCODER", f"Z_init_step8_trans{b}",
                          f"after transition_{b} mean={Z_init_II.float().mean():.6f}, "
                          f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            # DEBUG: Print final Z stats (compare with streaming mode)
            debug_log("ENCODER", "Z_init_FINAL",
                      f"mean={Z_init_II.float().mean():.6f}, "
                      f"first_half_mean={Z_init_II[:I_half].float().mean():.6f}")

            return {"S_init_I": S_I, "Z_init_II": Z_init_II}

        @activation_checkpointing
        def init_atoms(S_init_I, Z_init_II):
            Q_L_init = self.atom_1d_embedder_2(f, L)      # [L, c_atom]
            C_L = Q_L_init + self.process_s_trunk(S_init_I)[..., tok_idx, :]  # [L, c_atom]

            if self.use_chunked_pll:
                # Chunked mode: return embedder for later sparse computation
                return {
                    "Q_L_init": Q_L_init,                  # [L, c_atom]
                    "C_L": C_L,                            # [L, c_atom]
                    "chunked_pairwise_embedder": self.chunked_pairwise_embedder,
                    "S_I": S_init_I,                       # [I, c_s]
                    "Z_II": Z_init_II,                     # [I, I, c_z]
                }
            else:
                # Original full P_LL computation
                ##################################################################################
                # Embed motif coordinates - THIS CREATES FULL L×L TENSOR
                valid_mask = (
                    f["is_motif_atom_with_fixed_coord"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_coord"].unsqueeze(-2)
                ).unsqueeze(-1)                            # [L, L, 1]
                P_LL = self.motif_pos_embedder(
                    f["motif_pos"], valid_mask
                )                                          # [L, L, c_atompair]

                # Embed ref pos
                atoms_in_same_token = (
                    f["ref_space_uid"].unsqueeze(-1) == f["ref_space_uid"].unsqueeze(-2)
                ).unsqueeze(-1)                            # [L, L, 1]
                atoms_has_seq = (
                    f["is_motif_atom_with_fixed_seq"].unsqueeze(-1)
                    & f["is_motif_atom_with_fixed_seq"].unsqueeze(-2)
                ).unsqueeze(-1)                            # [L, L, 1]
                valid_mask = atoms_in_same_token & atoms_has_seq  # [L, L, 1]
                P_LL = P_LL + self.ref_pos_embedder(f["ref_pos"], valid_mask)  # [L, L, c_atompair]

                ##################################################################################

                P_LL = P_LL + (
                    self.process_single_l(C_L).unsqueeze(-2)
                    + self.process_single_m(C_L).unsqueeze(-3)
                )                                          # [L, L, c_atompair]
                P_LL = (
                    P_LL
                    + self.process_z(Z_init_II)[..., tok_idx, :, :][..., tok_idx, :]
                )                                          # [L, L, c_atompair]
                P_LL = P_LL + self.pair_mlp(P_LL)         # [L, L, c_atompair]
                P_LL = P_LL.contiguous()

                # Pool P_LL to token level to provide atom-level resolution for token track
                pooled_atom_level_features = pairwise_mean_pool(
                    pairwise_atom_features=self.process_pll(P_LL).unsqueeze(0),  # [1, L, L, c_atompair]
                    atom_to_token_map=tok_idx,             # [L]
                    I=int(tok_idx.max().item()) + 1,
                    dtype=P_LL.dtype,
                ).squeeze(0)                               # [I, I, c_atompair]
                Z_init_II = Z_init_II + self.project_pll(pooled_atom_level_features)  # [I, I, c_z]

                # Mix atom conditioning features via sequence-local attention
                if exists(self.atom_transformer):
                    C_L = self.atom_transformer(
                        C_L.unsqueeze(0), None, P_LL, indices=None, f=f, X_L=None
                    ).squeeze(0)                           # [L, c_atom]

                return {
                    "Q_L_init": Q_L_init,                  # [L, c_atom]
                    "C_L": C_L,                            # [L, c_atom]
                    "P_LL": P_LL,                          # [L, L, c_atompair]
                    "S_I": S_init_I,                       # [I, c_s]
                    "Z_II": Z_init_II,                     # [I, I, c_z]
                }

        tokens = init_tokens()
        return init_atoms(**tokens)


class DiffusionTokenEncoder(nn.Module):
    """
    Encodes token-level features for diffusion.
    
    Supports streaming mode where Z_init_II is a [I_par, I, c_z] chunk tensor
    instead of a full [I, I, c_z] tensor.
    """
    
    def __init__(
        self,
        c_s,
        c_z,
        c_token,
        c_atompair,
        sigma_data,
        n_pairformer_blocks,
        pairformer_block,
        use_distogram,
        use_self,
        use_sinusoidal_distogram_embedder=True,
        **_,
    ):
        super().__init__()
        
        self.c_z = c_z
        self.c_s = c_s

        # Sequence processing
        self.transition_1 = nn.ModuleList(
            [
                Transition(c=c_s, n=2),
                Transition(c=c_s, n=2),
            ]
        )

        # Post-processing of z
        self.n_bins_distogram = 65  # n bins for both self distogram and distogram
        n_bins_noise = self.n_bins_distogram
        self.use_self = use_self
        self.use_distogram = use_distogram
        self.use_sinusoidal_distogram_embedder = use_sinusoidal_distogram_embedder
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                self.dist_embedder = SinusoidalDistEmbed(c_atompair=c_z)
                n_bins_noise = c_z
            else:
                self.bucketize_fn = functools.partial(
                    bucketize_scaled_distogram,
                    min_dist=1,
                    max_dist=30,
                    sigma_data=sigma_data,
                    n_bins=self.n_bins_distogram,
                )
        cat_c_z = (
            c_z
            + int(self.use_distogram) * n_bins_noise
            + int(self.use_self) * self.n_bins_distogram
        )
        self.process_z = nn.Sequential(
            RMSNorm(cat_c_z),
            linearNoBias(cat_c_z, c_z),
        )

        self.transition_2 = nn.ModuleList(
            [
                Transition(c=c_z, n=2),
                Transition(c=c_z, n=2),
            ]
        )

        # Pairformer without triangle updates
        self.pairformer_stack = nn.ModuleList(
            [
                PairformerBlock(c_s=c_s, c_z=c_z, **pairformer_block)
                for _ in range(n_pairformer_blocks)
            ]
        )
        
        # Check for streaming mode
        # Parallel mode: =0 or unset → standard, any non-zero value → parallel
    def forward(self, f, R_L, S_init_I, Z_init_II, C_L, P_LL, **kwargs):
        """
        Forward pass for token encoding.

        Args:
            f: Feature dictionary
            R_L: [B, L, 3] scaled positions
            S_init_I: [I, c_s] initial single features
            Z_init_II: [I, I, c_z] (full) OR [I_par, I, c_z] (chunked in streaming mode)
            C_L: [L, c_atom] atom conditioning features
            P_LL: [L, L, c_atompair] or None
            **kwargs: D_II_self, parallel_mode, z_chunk_range

        Returns:
            S_I: [I, c_s] updated single features
            Z_II: [I, I, c_z] (full) OR [I_par, I, c_z] (chunked)
        """
        # Check for streaming mode via kwargs (Z_init_II is a chunked tensor)
        return self._forward_standard(f, R_L, S_init_I, Z_init_II, **kwargs)

    def _forward_standard(self, f, R_L, S_init_I, Z_init_II, **kwargs):
        """
        Standard forward: may create full I×I tensors.
        """
        B = R_L.shape[0]

        @activation_checkpointing
        def token_embed(S_init_I, Z_init_II):
            S_I = S_init_I                                # [I, c_s]
            for b in range(2):
                S_I = S_I + self.transition_1[b](S_I)     # [I, c_s]

            Z_II = Z_init_II.unsqueeze(0).expand(B, -1, -1, -1)  # [B, I, I, c_z]

            Z_II_list = [Z_II]
            if self.use_distogram:
                # Noise / self conditioning pair - CREATES I×I
                if self.use_sinusoidal_distogram_embedder:
                    mask = f["is_motif_atom_with_fixed_coord"][f["is_ca"]]  # [I]
                    mask = (mask[None, :] != mask[:, None]).unsqueeze(-1)   # [I, I, 1]
                    D_LL = self.dist_embedder(R_L[..., f["is_ca"], :], ~mask)  # [B, I, I, c_z]
                else:
                    D_LL = self.bucketize_fn(R_L[..., f["is_ca"], :])  # [B, I, I, n_bins]
                Z_II_list.append(D_LL)
            if self.use_self:
                D_II_self = kwargs.get("D_II_self")
                if D_II_self is None:
                    D_II_self = torch.zeros(
                        Z_II.shape[:-1] + (self.n_bins_distogram,),
                        device=Z_II.device,
                        dtype=Z_II.dtype,
                    )                                      # [B, I, I, n_bins]
                Z_II_list.append(D_II_self)
            Z_II = torch.cat(Z_II_list, dim=-1)            # [B, I, I, c_z + ...]

            # Flatten concatenated dims
            Z_II = self.process_z(Z_II)                    # [B, I, I, c_z]

            for b in range(2):
                Z_II = Z_II + self.transition_2[b](Z_II)   # [B, I, I, c_z]

            # Pairformer to mix
            for block in self.pairformer_stack:
                S_I, Z_II = block(S_I, Z_II)               # [I, c_s], [B, I, I, c_z]

            return S_I, Z_II

        return token_embed(S_init_I, Z_init_II)
