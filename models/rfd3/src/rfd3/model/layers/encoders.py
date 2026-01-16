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

from foundry.common import exists
from foundry.training.checkpoint import activation_checkpointing
from rfd3.model.debug_context import debug_ctx, debug_tensor, debug_log, debug_tensor_all_ranks, debug_log_all_ranks, verify_tensor_sync

logger = logging.getLogger(__name__)


def _is_streaming_mode() -> bool:
    """
    Check if streaming/parallel mode is enabled.

    Env var scheme:
      - RFD3_ATTENTION_PARALLEL=0 or unset → standard mode (False)
      - RFD3_ATTENTION_PARALLEL=1 → parallel mode (True)
    """
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    return val == "1"


def _get_world_size() -> int:
    """Get actual GPU count from distributed runtime."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _compute_chunk_ranges(total: int, n_par: int) -> List[Tuple[int, int]]:
    """Compute (start, end) ranges for chunked processing."""
    chunk_size = (total + n_par - 1) // n_par
    ranges = []
    for i in range(n_par):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        if start < total:
            ranges.append((start, end))
    return ranges


def _z_transition_chunked(Z: torch.Tensor, transition_fn, key_chunk: int = 512) -> torch.Tensor:
    """
    Apply z_transition in sub-chunks along key dimension to reduce peak memory.
    
    SwiGLU creates 4x intermediate: [I_par, I, c] -> [I_par, I, 4*c] -> [I_par, I, c]
    By chunking keys: [I_par, chunk, c] -> [I_par, chunk, 4*c] reduces memory 4x.
    
    Args:
        Z: [I_par, I, c_z] or [B, I_par, I, c_z]
        transition_fn: z_transition module
        key_chunk: chunk size along key dimension
    
    Returns:
        Z + transition_fn(Z), computed memory-efficiently
    """
    if Z.dim() == 3:
        # [I_par, I, c_z]
        I = Z.shape[1]
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, k_start:k_end, :]
            out_chunks.append(Z_sub + transition_fn(Z_sub))
        return torch.cat(out_chunks, dim=1)
    elif Z.dim() == 4:
        # [B, I_par, I, c_z]
        I = Z.shape[2]
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, :, k_start:k_end, :]
            out_chunks.append(Z_sub + transition_fn(Z_sub))
        return torch.cat(out_chunks, dim=2)
    else:
        return Z + transition_fn(Z)


def _process_z_chunked(Z: torch.Tensor, process_fn, key_chunk: int = 512) -> torch.Tensor:
    """
    Apply process_z (Linear) in sub-chunks along key dimension to reduce peak memory.

    process_z transforms [I_par, I, c_in] -> [I_par, I, c_out].
    For I=8100, c_in=258, c_out=128, full tensor is 22 GB.
    By chunking keys: [I_par, chunk, c_in] -> [I_par, chunk, c_out] reduces memory.

    Unlike _z_transition_chunked, this has NO residual connection - just applies process_fn.

    Args:
        Z: [I_par, I, c_in] or [B, I_par, I, c_in]
        process_fn: process_z module (typically nn.Sequential with RMSNorm + Linear)
        key_chunk: chunk size along key dimension

    Returns:
        process_fn(Z) computed memory-efficiently
    """
    if Z.dim() == 3:
        # [I_par, I, c_in]
        I = Z.shape[1]
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, k_start:k_end, :]
            out_chunks.append(process_fn(Z_sub))
            del Z_sub  # Free memory immediately
        return torch.cat(out_chunks, dim=1)
    elif Z.dim() == 4:
        # [B, I_par, I, c_in]
        I = Z.shape[2]
        out_chunks = []
        for k_start in range(0, I, key_chunk):
            k_end = min(k_start + key_chunk, I)
            Z_sub = Z[:, :, k_start:k_end, :]
            out_chunks.append(process_fn(Z_sub))
            del Z_sub  # Free memory immediately
        return torch.cat(out_chunks, dim=2)
    else:
        return process_fn(Z)


def _get_gpu_rank_and_world_size() -> Tuple[int, int]:
    """Get current GPU rank and world size for distributed processing."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _compute_gpu_query_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    """
    Compute the query index range for a specific GPU.
    
    Each GPU handles a contiguous chunk of queries (tokens).
    This ensures no GPU ever needs to compute full I×I tensors.
    
    Args:
        total: Total number of queries (I tokens)
        rank: This GPU's rank (0 to world_size-1)
        world_size: Total number of GPUs
        
    Returns:
        (start_idx, end_idx): Query range [start, end) for this GPU
    """
    chunk_size = total // world_size
    remainder = total % world_size
    
    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size
    
    return start, end


def _all_gather_concat(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Gather tensors from all GPUs and concatenate along specified dimension.
    
    Args:
        tensor: Local tensor chunk to gather
        dim: Dimension to concatenate along
        
    Returns:
        Full tensor reassembled from all GPUs
    """
    import torch.distributed as dist
    import os
    
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    # Ensure tensor is on the correct device for this rank
    # This is critical for distributed training - each rank must use its assigned GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    expected_device = torch.device(f"cuda:{local_rank}")
    if tensor.device != expected_device:
        tensor = tensor.to(expected_device)
    
    # Ensure tensor is contiguous (required for NCCL)
    tensor = tensor.contiguous()

    # ===========================================================================
    # MEMORY OPTIMIZATION: Use all_gather_into_tensor for better memory scaling
    # Old implementation created world_size copies on EACH GPU, causing O(n²)
    # memory growth with GPU count. New version uses single pre-allocated output.
    # Commented out old code preserved for rollback if needed.
    # ===========================================================================

    # --- OLD IMPLEMENTATION (creates world_size tensors per GPU) ---
    # gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    # dist.all_gather(gathered, tensor)
    # return torch.cat(gathered, dim=dim)
    # --- END OLD IMPLEMENTATION ---

    # --- NEW MEMORY-EFFICIENT IMPLEMENTATION ---
    # Pre-allocate single output tensor instead of world_size separate tensors
    # This reduces peak memory from O(world_size) to O(1) temporary allocations

    # Calculate output shape: expand the gather dimension by world_size
    output_shape = list(tensor.shape)
    output_shape[dim] = output_shape[dim] * world_size
    output_tensor = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)

    # Use all_gather_into_tensor if available (PyTorch >= 1.13), else fallback
    if hasattr(dist, 'all_gather_into_tensor'):
        # Flatten for all_gather_into_tensor, then reshape
        # all_gather_into_tensor expects flat output tensor
        flat_input = tensor.contiguous().view(-1)
        flat_output = torch.empty(flat_input.numel() * world_size, dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(flat_output, flat_input)

        # Reshape: split into world_size chunks along dim 0, then move to correct dim
        chunk_shape = list(tensor.shape)
        reshaped = flat_output.view(world_size, *chunk_shape)

        # Move the world_size dimension to position `dim` and merge
        # e.g., for dim=0: [W, I_par, c] -> [W*I_par, c]
        # e.g., for dim=1: [W, B, L_par, c] -> [B, W*L_par, c]
        if dim == 0:
            output_tensor = reshaped.view(-1, *chunk_shape[1:])
        else:
            # Transpose world_size dim to target position, then flatten
            perm = list(range(1, dim + 1)) + [0] + list(range(dim + 1, len(chunk_shape) + 1))
            transposed = reshaped.permute(*perm)
            final_shape = list(tensor.shape)
            final_shape[dim] = final_shape[dim] * world_size
            output_tensor = transposed.contiguous().view(*final_shape)

        del flat_input, flat_output, reshaped  # Explicit cleanup
    else:
        # Fallback for older PyTorch: use list-based all_gather but cleanup immediately
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        output_tensor = torch.cat(gathered, dim=dim)
        del gathered  # Explicit cleanup to free world_size tensors

    return output_tensor
    # --- END NEW IMPLEMENTATION ---


# NOTE: StreamingZContainer has been removed. Z chunks are now computed
# directly in TokenInitializer._forward_streaming() and returned as tensors.
# The chunked Z tensor [I_par, I, c_z] is passed through with streaming_mode=True
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
        
        # Check for streaming mode (no full I×I/L×L tensors)
        # Parallel mode: =0 or unset → standard, =1 → parallel (GPU count auto-detected)
        self.use_streaming = _is_streaming_mode()
        if self.use_streaming:
            logger.info(
                f"TokenInitializer: Streaming mode enabled. "
                f"No full I×I tensors will be materialized."
            )

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

    def _process_s_through_transformer_stack(
        self,
        S_I: torch.Tensor,     # [I, c_s]
        f: dict,
        I: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Process S_I through transformer_stack using chunked attention.
        
        In standard mode, transformer_stack.forward does:
            Z_II = Z_II + z_transition(Z_II)
            S_I = S_I + attention_pair_bias(S_I, None, Z_II, ...)
            S_I = S_I + s_transition(S_I)
        
        IMPORTANT: Z_II accumulates z_transition updates across blocks!
        Block 0 applies z_transition_0, Block 1 applies z_transition_1, etc.
        The attention bias at block N uses Z with all previous transitions applied.
        
        In streaming mode, we can't materialize full Z_II, so we:
        1. Compute base Z_chunk for this GPU's query range
        2. Maintain Z_chunk across blocks, applying z_transition at each block
        3. Apply chunked attention to S_I
        4. All-gather S_I chunks to reconstruct full S_I
        
        Args:
            S_I: Single features [I, c_s]
            f: Feature dictionary
            I: Number of tokens
            device: Device for tensors
            dtype: Dtype for tensors
            
        Returns:
            S_I: Updated single features [I, c_s]
        """
        import torch.distributed as dist
        import os
        
        # Ensure S_I is on the correct device for this rank
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            expected_device = torch.device(f"cuda:{local_rank}")
            if S_I.device != expected_device:
                S_I = S_I.to(expected_device)
            # Override device parameter with computed expected_device for consistency
            device = expected_device
        else:
            expected_device = device
        
        # Get GPU range for chunking
        gpu_rank, world_size = _get_gpu_rank_and_world_size()
        chunk_ranges = _compute_chunk_ranges(I, world_size)
        
        if gpu_rank >= len(chunk_ranges):
            return S_I  # Edge case: more GPUs than tokens
        
        start_i, end_i = chunk_ranges[gpu_rank]
        I_par = end_i - start_i
        
        # DEBUG: Helper function for printing tensor stats (define before use)
        def _stat_tensor(t):
            return {
                "shape": list(t.shape),
                "mean": float(t.float().mean().item()),
                "std": float(t.float().std().item()),
                "min": float(t.float().min().item()),
                "max": float(t.float().max().item()),
            }
        
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        
        # Pre-compute Z_j once (used for base Z)
        Z_j = self.to_z_init_j(S_I)                           # [I, c_z]
        
        # Get reference positions for Z computation
        ref_pos = f["ref_pos"][f["is_ca"]]                    # [I, 3]
        ref_space_uid = f["ref_space_uid"][f["is_ca"]]        # [I]
        
        # ================================================================
        # Compute base Z_chunk ONCE (before any z_transition)
        # CRITICAL: This matches standard mode where Z_init_II is computed
        # from INITIAL S_I before the transformer_stack loop
        # ================================================================
        S_I_chunk = S_I[start_i:end_i]                        # [I_par, c_s]
        
        # DEBUG: Print S_I_chunk used for Z_chunk computation
        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} Z_CHUNK_COMPUTATION: S_I_chunk={_stat_tensor(S_I_chunk)}, Z_j={_stat_tensor(Z_j)}", flush=True)
        
        Z_i = self.to_z_init_i(S_I_chunk).unsqueeze(-2)       # [I_par, 1, c_z]
        Z_j_full = Z_j.unsqueeze(0)                           # [1, I, c_z]
        Z_chunk = Z_i + Z_j_full                              # [I_par, I, c_z]
        
        # DEBUG: Print initial Z_chunk
        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} Z_CHUNK_INITIAL: Z_chunk={_stat_tensor(Z_chunk)}", flush=True)
        
        # Add RPE
        Z_chunk = Z_chunk + self.relative_position_encoding.forward_chunk(
            f, start_i, end_i
        )                                                      # [I_par, I, c_z]
        
        # Add token bonds
        token_bonds_chunk = f["token_bonds"][start_i:end_i, :]  # [I_par, I]
        Z_chunk = Z_chunk + self.process_token_bonds(
            token_bonds_chunk.unsqueeze(-1).float()
        )                                                      # [I_par, I, c_z]
        
        # Add reference position embedding
        ref_pos_chunk = ref_pos[start_i:end_i]                # [I_par, 3]
        ref_space_uid_chunk = ref_space_uid[start_i:end_i]    # [I_par]
        valid_mask = (
            ref_space_uid_chunk.unsqueeze(-1) == ref_space_uid.unsqueeze(0)
        ).unsqueeze(-1)                                        # [I_par, I, 1]
        ref_pos_embed = self.ref_pos_embedder_tok.forward_chunk(
            ref_pos_chunk, ref_pos, valid_mask
        )                                                      # [I_par, I, c_z]
        Z_chunk = Z_chunk + ref_pos_embed
        
        # ================================================================
        # Process through transformer_stack (Z_chunk accumulates updates!)
        # ================================================================
        # Note: _stat_tensor and is_rank0 are already defined above
        
        for block_idx, block in enumerate(self.transformer_stack):
            # DEBUG: Print S_I stats at start of each block to verify it's being updated
            if is_rank0:
                print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} START: S_I={_stat_tensor(S_I)}, device={S_I.device}", flush=True)
            
            # Step 1: Apply z_transition (ACCUMULATES across blocks)
            Z_chunk = _z_transition_chunked(Z_chunk, block.z_transition)
            
            # Step 2: Apply attention_pair_bias to S_I using Z_chunk as bias
            if hasattr(block, 'attention_pair_bias'):
                # CRITICAL: Slice S_I at the start of each iteration to get updated values
                # S_I should have been updated from the previous iteration's all_gather
                S_I_chunk = S_I[start_i:end_i].clone()        # [I_par, c_s] - clone to ensure fresh tensor
                
                # DEBUG: Print S_I_chunk stats after slicing
                if is_rank0:
                    print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} AFTER_SLICE: S_I_chunk={_stat_tensor(S_I_chunk)}", flush=True)
                
                S_I_chunk = S_I_chunk + block.attention_pair_bias.forward_chunked(
                    A_I_query=S_I_chunk,                       # [I_par, c_s]
                    A_I_key=S_I,                               # [I, c_s]
                    Z_chunk=Z_chunk,                           # [I_par, I, c_z]
                    Beta_II=torch.tensor([0.0], device=device),
                )                                              # [I_par, c_s]
                S_I_chunk = S_I_chunk + block.s_transition(S_I_chunk)
                
                # DEBUG: Print S_I_chunk stats after processing
                if is_rank0:
                    print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} AFTER_PROCESS: S_I_chunk={_stat_tensor(S_I_chunk)}", flush=True)
            
                # Step 3: All-gather S_I chunks to update full S_I for next block
                # CRITICAL: This updates S_I for the next iteration
                if world_size > 1:
                    S_I_gathered = _all_gather_concat(S_I_chunk, dim=0)  # [I, c_s]
                    # Ensure S_I is on the correct device after all_gather
                    if S_I_gathered.device != expected_device:
                        S_I_gathered = S_I_gathered.to(expected_device)

                    # DEBUG: Print S_I stats from ALL ranks after all_gather to verify sync
                    debug_tensor_all_ranks("ENCODER-S_I", f"block{block_idx}_AFTER_ALLGATHER", S_I_gathered)

                    # MULTI-GPU DIAGNOSTIC: Verify S_I is synchronized across all ranks
                    # This broadcasts from rank 0 and compares on all ranks
                    verify_tensor_sync("ENCODER-S_I", f"block{block_idx}_S_I_SYNC", S_I_gathered)

                    S_I = S_I_gathered
                else:
                    S_I = S_I_chunk
                    # DEBUG: Single GPU case
                    if is_rank0:
                        print(f"{debug_ctx.prefix('ENCODER-S_I')} block{block_idx} SINGLE_GPU: S_I={_stat_tensor(S_I)}", flush=True)

        # MULTI-GPU DIAGNOSTIC: Final S_I sync check
        if world_size > 1:
            debug_tensor_all_ranks("ENCODER-S_I", "FINAL_S_I", S_I)
            verify_tensor_sync("ENCODER-S_I", "FINAL_S_I_SYNC", S_I)

        return S_I

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
        
        # Use streaming mode if enabled (no full I×I tensors)
        if self.use_streaming:
            return self._forward_streaming(f, tok_idx, L, I)
        else:
            return self._forward_standard(f, tok_idx, L, I)
    
    def _forward_streaming(self, f, tok_idx, L, I):
        """
        Streaming forward: never materializes full I×I tensors.
        
        Returns Z_II as [I_par, I, c_z] chunk tensor with z_chunk_range.
        """
        import torch.distributed as dist
        import os
        
        # Get the correct device for this rank (CRITICAL for distributed training)
        # Use LOCAL_RANK to ensure each rank uses its assigned GPU
        if dist.is_initialized():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = tok_idx.device
        
        dtype = self.to_z_init_i.weight.dtype
        
        # Ensure TokenInitializer modules are on the correct device for this rank
        self.to(device)
        
        # Move input tensors to the correct device
        tok_idx = tok_idx.to(device)
        # Move feature dict tensors to correct device
        for key, val in f.items():
            if isinstance(val, torch.Tensor):
                f[key] = val.to(device)
        
        # ============================================================
        # Step 1: Compute S_I (single token features) - no I×I here
        # ============================================================
        S_I = self.token_1d_embedder(f, I)                # [I, c_s]
        S_I = S_I + self.transition_post_token(S_I)       # [I, c_s]

        # Embed atom features and downcast to token features
        S_I = self.downcast_atom(
            Q_L=self.atom_1d_embedder_1(f, L),            # [L, c_s]
            A_I=S_I,                                       # [I, c_s]
            tok_idx=tok_idx                                # [L]
        )                                                  # [I, c_s]
        S_I = S_I + self.transition_post_atom(S_I)        # [I, c_s]
        S_I = self.process_s_init(S_I)                    # [I, c_s]
        
        # DEBUG: Print initial S_I after process_s_init
        is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        def _stat(t):
            return {
                "shape": list(t.shape),
                "mean": float(t.float().mean().item()),
                "std": float(t.float().std().item()),
                "min": float(t.float().min().item()),
                "max": float(t.float().max().item()),
            }
        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} INITIAL_AFTER_PROCESS_S_INIT: S_I={_stat(S_I)}, device={S_I.device}", flush=True)
        
        # ============================================================
        # CRITICAL FIX: Save INITIAL S_I for Z computation
        # ============================================================
        # In non-parallel mode, Z_init_II is computed from S_I BEFORE the
        # transformer_stack processes it. The transformer_stack then updates
        # BOTH S_I and Z_init_II together. We must match this behavior:
        # 1. Compute Z from INITIAL S_I (before transformer_stack)
        # 2. Process S_I through transformer_stack to get UPDATED S_I
        # 3. Use UPDATED S_I for downstream (atom features, etc.)
        S_I_initial = S_I.clone()  # Save INITIAL S_I for Z computation
        
        # ============================================================
        # Step 2: Pre-compute Z_j from INITIAL S_I (before transformer_stack)
        # ============================================================
        # This matches non-parallel mode where Z_init_II = to_z_init_i(S_I) + to_z_init_j(S_I)
        # is computed BEFORE the transformer_stack loop
        Z_j = self.to_z_init_j(S_I_initial)              # [I, c_z] from INITIAL S_I
        
        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} Z_j computed from INITIAL S_I: Z_j={_stat(Z_j)}", flush=True)
        
        # ============================================================
        # Step 3: Process S_I through transformer_stack
        # ============================================================
        # In standard mode, transformer_stack processes BOTH S_I and Z_II.
        # In streaming mode, we must also process S_I through the attention
        # layers, using chunked Z computation to avoid full I×I tensor.
        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} BEFORE_TRANSFORMER_STACK: S_I={_stat(S_I)}, device={S_I.device}", flush=True)

        # CRITICAL FIX: On single GPU, use standard transformer_stack to ensure
        # bit-identical S_I computation with standard mode. The chunked attention
        # path in _process_s_through_transformer_stack has small numerical differences.
        gpu_rank, world_size = _get_gpu_rank_and_world_size()

        if world_size == 1:
            # Single GPU: Use standard transformer_stack loop for identical S_I
            # Compute full Z_init_II like standard mode
            Z_init_II = self.to_z_init_i(S_I).unsqueeze(-3) + self.to_z_init_j(S_I).unsqueeze(-2)  # [I, I, c_z]
            Z_init_II = Z_init_II + self.relative_position_encoding(f)  # [I, I, c_z]
            Z_init_II = Z_init_II + self.process_token_bonds(
                f["token_bonds"].unsqueeze(-1).float()
            )  # [I, I, c_z]

            # Reference position embedding
            token_id = f["ref_space_uid"][f["is_ca"]]  # [I]
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)).unsqueeze(-1)  # [I, I, 1]
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(
                f["ref_pos"][f["is_ca"]], valid_mask
            )  # [I, I, c_z]

            # Standard transformer_stack loop - processes BOTH S_I and Z_init_II
            for block in self.transformer_stack:
                S_I, Z_init_II = block(S_I, Z_init_II)  # [I, c_s], [I, I, c_z]

            # CRITICAL: Continue processing Z_init_II exactly like standard mode!
            # Concatenate with relative_position_encoding2 and apply process_z_init + transition_1
            Z_init_II = torch.cat(
                [
                    Z_init_II,
                    self.relative_position_encoding2(f),
                ],
                dim=-1,
            )  # [I, I, 2*c_z]
            Z_init_II = self.process_z_init(Z_init_II)  # [I, I, c_z]
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)  # [I, I, c_z]

            if is_rank0:
                print(f"{debug_ctx.prefix('ENCODER-S_I')} SINGLE_GPU_STANDARD_PATH: S_I={_stat(S_I)}", flush=True)
                print(f"{debug_ctx.prefix('ENCODER-S_I')} SINGLE_GPU_Z_init_II: mean={Z_init_II.float().mean():.6f}", flush=True)

            # For single GPU, use the full Z_init_II tensor directly
            # This ensures bit-identical results with standard mode
            Q_L_init = self.atom_1d_embedder_2(f, L)  # [L, c_atom]
            C_L = Q_L_init + self.process_s_trunk(S_I)[..., tok_idx, :]  # [L, c_atom]

            return {
                "Q_L_init": Q_L_init,
                "C_L": C_L,
                "chunked_pairwise_embedder": self.chunked_pairwise_embedder if self.use_chunked_pll else None,
                "S_I": S_I,
                "Z_II": Z_init_II,  # Full tensor [I, I, c_z]
                "streaming_mode": False,  # Use standard path since we have full Z tensor
            }
        else:
            # Multi-GPU: Use chunked attention to avoid full I×I tensor
            S_I = self._process_s_through_transformer_stack(
                S_I=S_I,
                f=f,
                I=I,
                device=device,
                dtype=dtype,
            )

        # DEBUG: Print S_I after processing
        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER-S_I')} AFTER_TRANSFORMER_STACK: S_I={_stat(S_I)}, device={S_I.device}", flush=True)

        # CRITICAL: Ensure S_I is synchronized across all ranks before continuing
        if dist.is_initialized():
            dist.barrier()

        # ============================================================
        # Step 4: Compute Z chunk [I_par, I, c_z] for this GPU directly
        # ============================================================
        # Each GPU computes its portion of Z [I_par, I, c_z].
        # This is equivalent to Z_init_II[start_i:end_i, :, :] but never
        # materializes the full [I, I, c_z] tensor.

        # Get this GPU's query range
        start_i, end_i = _compute_gpu_query_range(I, gpu_rank, world_size)
        I_par = end_i - start_i
        qs = slice(start_i, end_i)

        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER')} Computing Z chunk [{start_i}:{end_i}] (I_par={I_par}) for GPU {gpu_rank}/{world_size}", flush=True)

        # Step 4a: Base Z = Z_i + Z_j (cross-attention style)
        # Use S_I_initial (before transformer_stack) to match standard mode
        S_I_chunk = S_I_initial[qs]                        # [I_par, c_s]
        Z_i = self.to_z_init_i(S_I_chunk).unsqueeze(-2)    # [I_par, 1, c_z]
        Z_j_expanded = Z_j.unsqueeze(0)                    # [1, I, c_z]
        Z_chunk = Z_i + Z_j_expanded                       # [I_par, I, c_z]

        debug_log("ENCODER", "Z_chunk_step1",
                  f"Z_i+Z_j mean={Z_chunk.float().mean():.6f}")

        # Step 4b: Add RPE (chunked)
        Z_chunk = Z_chunk + self.relative_position_encoding.forward_chunk(f, start_i, end_i)
        debug_log("ENCODER", "Z_chunk_step2_rpe",
                  f"after RPE mean={Z_chunk.float().mean():.6f}")

        # Step 4c: Add token bonds
        token_bonds_chunk = f["token_bonds"][qs, :]       # [I_par, I]
        Z_chunk = Z_chunk + self.process_token_bonds(
            token_bonds_chunk.unsqueeze(-1).float()
        )
        debug_log("ENCODER", "Z_chunk_step3_bonds",
                  f"after token_bonds mean={Z_chunk.float().mean():.6f}")

        # Step 4d: Add reference position embedding (chunked)
        ref_pos = f["ref_pos"][f["is_ca"]]                # [I, 3]
        ref_space_uid = f["ref_space_uid"][f["is_ca"]]    # [I]
        ref_pos_chunk = ref_pos[qs]                        # [I_par, 3]
        ref_space_uid_chunk = ref_space_uid[qs]            # [I_par]

        valid_mask = (
            ref_space_uid_chunk.unsqueeze(-1) == ref_space_uid.unsqueeze(0)
        ).unsqueeze(-1)                                    # [I_par, I, 1]

        Z_chunk = Z_chunk + self.ref_pos_embedder_tok.forward_chunk(
            ref_pos_chunk, ref_pos, valid_mask
        )
        debug_log("ENCODER", "Z_chunk_step4_refpos",
                  f"after ref_pos mean={Z_chunk.float().mean():.6f}")

        # Step 4e: Pairformer Z transitions
        for block_idx, block in enumerate(self.transformer_stack):
            Z_chunk = _z_transition_chunked(Z_chunk, block.z_transition)
            debug_log("ENCODER", f"Z_chunk_step5_block{block_idx}",
                      f"after block mean={Z_chunk.float().mean():.6f}")

        # Step 4f: Concatenate with second RPE and process
        rpe2_chunk = self.relative_position_encoding2.forward_chunk(f, start_i, end_i)
        Z_chunk = torch.cat([Z_chunk, rpe2_chunk], dim=-1)  # [I_par, I, 2*c_z]
        debug_log("ENCODER", "Z_chunk_step6_rpe2cat",
                  f"after rpe2 concat mean={Z_chunk.float().mean():.6f}")

        Z_chunk = self.process_z_init(Z_chunk)              # [I_par, I, c_z]
        debug_log("ENCODER", "Z_chunk_step7_processzinit",
                  f"after process_z_init mean={Z_chunk.float().mean():.6f}")

        # Step 4g: Apply transitions
        for b, transition in enumerate(self.transition_1):
            Z_chunk = _z_transition_chunked(Z_chunk, transition)
            debug_log("ENCODER", f"Z_chunk_step8_trans{b}",
                      f"after transition_{b} mean={Z_chunk.float().mean():.6f}")

        debug_log("ENCODER", "Z_chunk_FINAL",
                  f"[{start_i}:{end_i}] mean={Z_chunk.float().mean():.6f}")

        if is_rank0:
            print(f"{debug_ctx.prefix('ENCODER')} Z_chunk computed: shape={list(Z_chunk.shape)}, mean={Z_chunk.float().mean():.6f}", flush=True)

        # ============================================================
        # Step 5: Compute atom features (Q_L_init, C_L)
        # ============================================================
        Q_L_init = self.atom_1d_embedder_2(f, L)          # [L, c_atom]
        C_L = Q_L_init + self.process_s_trunk(S_I)[..., tok_idx, :]  # [L, c_atom]

        return {
            "Q_L_init": Q_L_init,                          # [L, c_atom]
            "C_L": C_L,                                    # [L, c_atom]
            "chunked_pairwise_embedder": self.chunked_pairwise_embedder if self.use_chunked_pll else None,
            "S_I": S_I,                                    # [I, c_s]
            "Z_II": Z_chunk,                               # [I_par, I, c_z] - this GPU's chunk
            "streaming_mode": True,                        # Flag: Z_II is chunked [I_par, I] not full [I, I]
            "z_chunk_range": (start_i, end_i),             # This GPU's query range for Z
        }
    
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
        # Parallel mode: =0 or unset → standard, =1 → parallel
        self.use_streaming = _is_streaming_mode()

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
            **kwargs: D_II_self, streaming_mode, z_chunk_range

        Returns:
            S_I: [I, c_s] updated single features
            Z_II: [I, I, c_z] (full) OR [I_par, I, c_z] (chunked)
        """
        # Check for streaming mode via kwargs (Z_init_II is a chunked tensor)
        streaming_mode = kwargs.get("streaming_mode", False)

        if streaming_mode:
            return self._forward_streaming(f, R_L, S_init_I, Z_init_II, **kwargs)
        else:
            return self._forward_standard(f, R_L, S_init_I, Z_init_II, **kwargs)

    def _forward_streaming(self, f, R_L, S_init_I, Z_init_II, **kwargs):
        """
        Multi-GPU streaming forward: each GPU processes only its query chunk.

        Args:
            Z_init_II: Pre-computed Z chunk tensor [I_par, I, c_z]
            **kwargs: Must include z_chunk_range=(start_i, end_i)

        Multi-GPU parallelism:
        - Each GPU processes Z rows for its assigned query range [gpu_start, gpu_end)
        - This produces [I_par, I, c_z] per GPU, never full [I, I, c_z]
        - Single features S_I are computed per-GPU then all_gathered

        Returns:
            S_I: [I, c_s] - updated single features (all_gathered)
            Z_II: [I_par, I, c_z] - THIS GPU's Z rows only (NOT full I×I!)
        """
        B = R_L.shape[0]
        device = R_L.device
        dtype = R_L.dtype

        # Ensure encoder modules are on the correct device for this rank
        self.to(device)

        # Z_init_II is a pre-computed chunk tensor [I_par, I, c_z]
        Z_chunk = Z_init_II
        z_chunk_range = kwargs.get("z_chunk_range")
        if z_chunk_range is None:
            raise ValueError("streaming_mode=True requires z_chunk_range in kwargs")
        gpu_start, gpu_end = z_chunk_range
        I = Z_chunk.shape[1]                            # Second dim is full I
        _, world_size = _get_gpu_rank_and_world_size()
        base_z_dim = Z_chunk.shape[-1]                  # c_z from tensor

        I_par = gpu_end - gpu_start

        # Step 1: Update S_I (operates on full I, no I×I)
        S_I = S_init_I
        has_batch_S = S_I.dim() == 3
        for b in range(2):
            S_I = S_I + self.transition_1[b](S_I)

        # Step 2: Expand Z chunk for batch dimension
        Z_chunk = Z_chunk.unsqueeze(0).expand(B, -1, -1, -1)  # [B, I_par, I, c_z]

        # Step 3: Add distogram for this GPU's query chunk
        if self.use_distogram:
            R_ca = R_L[..., f["is_ca"], :]                 # [B, I, 3]

            if self.use_sinusoidal_distogram_embedder:
                R_ca_query = R_ca[:, gpu_start:gpu_end, :] # [B, I_par, 3]

                # Mask: [I_par, I, 1] - query chunk vs all keys
                motif_mask = f["is_motif_atom_with_fixed_coord"][f["is_ca"]]  # [I]
                motif_query = motif_mask[gpu_start:gpu_end]                    # [I_par]
                mask_chunk = (motif_query[:, None] != motif_mask[None, :]).unsqueeze(-1)

                # Sinusoidal distance embedding for this chunk
                D_chunk = self.dist_embedder.forward_chunk(
                    R_ca_query, R_ca, ~mask_chunk
                )                                          # [B, I_par, I, c_z]
            else:
                # Bucketized distogram - compute for query chunk vs all atoms
                D_chunk = bucketize_scaled_distogram_chunked(
                    R_ca, gpu_start, gpu_end,
                    min_dist=1, max_dist=30, sigma_data=16,  # default sigma_data
                    n_bins=self.n_bins_distogram
                )                                          # [B, I_par, I, n_bins]
            Z_chunk = torch.cat([Z_chunk, D_chunk], dim=-1)
            del D_chunk  # Free memory immediately - no longer needed after cat

        # Step 4: Add self-conditioning for this GPU's chunk
        expected_dim = base_z_dim
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                expected_dim += self.c_z  # sinusoidal uses DiffusionTokenEncoder.c_z
            else:
                expected_dim += self.n_bins_distogram
        if self.use_self:
            expected_dim += self.n_bins_distogram
        
        if self.use_self:
            D_II_self = kwargs.get("D_II_self")
            if D_II_self is not None:
                # =======================================================================
                # MEMORY OPTIMIZATION: D_II_self may already be chunked [B, I_par, I, n_bins]
                # In multi-GPU mode, RFD3_diffusion_module now returns the chunk directly
                # instead of gathering to full [B, I, I, n_bins] to save 12-21 GB per GPU.
                # =======================================================================

                # --- OLD IMPLEMENTATION (assumed full [B, I, I, n_bins]) ---
                # D_self_chunk = D_II_self[:, gpu_start:gpu_end, :]  # [B, I_par, I, n_bins]
                # --- END OLD IMPLEMENTATION ---

                # --- NEW IMPLEMENTATION: Check if already chunked ---
                if D_II_self.shape[1] == I_par:
                    # Already chunked [B, I_par, I, n_bins] - use directly
                    D_self_chunk = D_II_self
                else:
                    # Full tensor [B, I, I, n_bins] (standard mode) - slice it
                    D_self_chunk = D_II_self[:, gpu_start:gpu_end, :]  # [B, I_par, I, n_bins]
                # --- END NEW IMPLEMENTATION ---
            else:
                D_self_chunk = torch.zeros(
                    B, I_par, I, self.n_bins_distogram,
                    device=device, dtype=dtype
                )                                          # [B, I_par, I, n_bins]
            Z_chunk = torch.cat([Z_chunk, D_self_chunk], dim=-1)
            del D_self_chunk  # Free memory immediately - no longer needed after cat

        # Verify dimensions before process_z (diagnostic)
        actual_dim = Z_chunk.shape[-1]
        
        # CRITICAL: process_z expects cat_c_z = self.c_z + distogram + self_cond
        # If TokenInitializer.c_z != DiffusionTokenEncoder.c_z, dimensions won't match
        process_z_expected = self.c_z
        if self.use_distogram:
            if self.use_sinusoidal_distogram_embedder:
                process_z_expected += self.c_z
            else:
                process_z_expected += self.n_bins_distogram
        if self.use_self:
            process_z_expected += self.n_bins_distogram
        
        if actual_dim != process_z_expected:
            raise RuntimeError(
                f"DiffusionTokenEncoder._forward_streaming: dimension mismatch!\n"
                f"  process_z expects: {process_z_expected} (based on DiffusionTokenEncoder.c_z={self.c_z})\n"
                f"  Z_chunk actual: {actual_dim}\n"
                f"  Z_chunk shape: {Z_chunk.shape}\n"
                f"  TokenInitializer.c_z (Z_streaming): {base_z_dim}\n"
                f"  use_distogram={self.use_distogram}, use_sinusoidal={self.use_sinusoidal_distogram_embedder}, "
                f"use_self={self.use_self}, n_bins={self.n_bins_distogram}\n"
                f"  Streaming mode requires TokenInitializer.c_z == DiffusionTokenEncoder.c_z"
            )
        
        # Step 5: Process concatenated Z features
        # Match standard: Z_II = self.process_z(Z_II)
        # =======================================================================
        # MEMORY OPTIMIZATION: Use key-chunking for process_z to avoid 22+ GB tensor
        # Z_chunk is [B, I_par, I, c_in] where c_in = c_z + distogram + self_cond
        # For I=8100, c_in=258, full tensor = 22 GB which causes OOM.
        # Processing in key-chunks of 512: [B, I_par, 512, c_in] = ~1.4 GB
        # =======================================================================
        Z_chunk = _process_z_chunked(Z_chunk, self.process_z)  # [B, I_par, I, c_z]
        
        # Match standard: Z_II = Z_II + self.transition_2[b](Z_II)
        # Use key-chunking to reduce peak memory from SwiGLU 4x expansion
        for b in range(2):
            Z_chunk = _z_transition_chunked(Z_chunk, self.transition_2[b])  # [B, I_par, I, c_z]
        
        # Step 6: Pairformer with chunked attention
        # S_I attention: this GPU's query chunk [I_par] attends to all keys [I]
        # Output is [I_par, c_s] per GPU, then all_gathered to [I, c_s]
        
        # Handle batch dimension in S_I
        if has_batch_S:
            # S_I is [B, I, c_s] - slice along dim 1, squeeze batch for attention
            S_I_chunk = S_I[:, gpu_start:gpu_end, :].squeeze(0)  # [I_par, c_s]
            S_I_unbatched = S_I.squeeze(0)                        # [I, c_s]
        else:
            # S_I is [I, c_s] - slice directly
            S_I_chunk = S_I[gpu_start:gpu_end]                    # [I_par, c_s]
            S_I_unbatched = S_I                                   # [I, c_s]
        
        for block in self.pairformer_stack:
            # Z transition (key-chunked to reduce memory)
            Z_chunk = _z_transition_chunked(Z_chunk, block.z_transition)  # [B, I_par, I, c_z]
            
            # Attention: queries [I_par] attend to all keys [I] using Z_chunk as bias
            if hasattr(block, 'attention_pair_bias'):
                # Chunked attention: S_I_chunk queries, S_I keys, Z_chunk bias
                # forward_chunked expects 2D inputs [I_par, c_s] and [I, c_s]
                S_I_chunk = S_I_chunk + block.attention_pair_bias.forward_chunked(
                    A_I_query=S_I_chunk,                   # [I_par, c_s]
                    A_I_key=S_I_unbatched,                 # [I, c_s]
                    Z_chunk=Z_chunk[0],                    # [I_par, I, c_z]
                    Beta_II=torch.tensor([0.0], device=device),
                )                                          # [I_par, c_s]
                S_I_chunk = S_I_chunk + block.s_transition(S_I_chunk)
            
            # CRITICAL: All-gather S_I_chunk to update keys for next block!
            # In standard mode, S_I is updated in each iteration and used as both
            # queries and keys in the next block. We must do the same here.
            if world_size > 1:
                S_I_unbatched = _all_gather_concat(S_I_chunk, dim=0)  # [I, c_s]
            else:
                S_I_unbatched = S_I_chunk
        
        # Step 7: All-gather S_I chunks from all GPUs (final)
        # Each GPU has [I_par, c_s], gather to get full [I, c_s]
        if world_size > 1:
            S_I = _all_gather_concat(S_I_chunk, dim=0)     # [I, c_s]
        else:
            # Single GPU: just use the chunk (which is full I in this case)
            S_I = S_I_chunk
        
        # Return this GPU's Z chunk (NOT full I×I!)
        # Downstream modules must also support chunked Z
        return S_I, Z_chunk[0] if B == 1 else Z_chunk      # S_I: [I, c_s], Z: [I_par, I, c_z]
    
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
