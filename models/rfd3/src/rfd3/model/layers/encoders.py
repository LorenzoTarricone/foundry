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

logger = logging.getLogger(__name__)


def _get_n_parallel() -> int:
    """Get parallelism factor from environment, or 0 if not set."""
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", None)
    if val is None:
        return 0
    try:
        return int(val)
    except ValueError:
        return 0


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
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    return torch.cat(gathered, dim=dim)


class StreamingZContainer:
    """
    Container for streaming Z_II computation with multi-GPU support.
    
    Instead of materializing full [I, I, c_z] tensor, stores components needed
    to compute row chunks [I_par, I, c_z] on-the-fly.
    
    In multi-GPU mode:
    - Each GPU only computes Z rows for its assigned query range
    - Query range is [gpu_start, gpu_end) where gpu_start/end are determined by rank
    - This ensures no single GPU ever materializes full [I, I, c_z]
    
    Downstream modules call get_gpu_chunk() to get this GPU's Z rows.
    """
    
    def __init__(
        self,
        S_I: torch.Tensor,                    # [I, c_s] single features
        to_z_init_i: nn.Module,               # Projection for queries
        to_z_init_j: torch.Tensor,            # [I, c_z] pre-computed key projection  
        rpe_module: nn.Module,                # RelativePositionEncoding module
        rpe_module2: nn.Module,               # Second RPE module  
        token_bonds: torch.Tensor,            # [I, I] token bond matrix (kept full, small)
        process_token_bonds: nn.Module,       # Linear for token bonds
        ref_pos_embedder: nn.Module,          # Reference position embedder
        ref_pos: torch.Tensor,                # [I, 3] reference positions (CA atoms)
        ref_space_uid: torch.Tensor,          # [I] reference space UIDs
        f: dict,                              # Feature dictionary
        transformer_stack: nn.ModuleList,     # Pairformer blocks
        process_z_init: nn.Module,            # Post-concatenation processor
        transition_modules: nn.ModuleList,    # Transition layers
        c_z: int,                             # Pair embedding dimension
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize streaming Z container with all components for on-demand computation.
        
        Multi-GPU: Each GPU will only compute Z for its assigned query range,
        producing [I_par, I, c_z] instead of [I, I, c_z].
        """
        self.S_I = S_I                        # [I, c_s]
        self.I = S_I.shape[0]
        self.c_z = c_z
        self.device = device
        self.dtype = dtype
        
        # Store modules and pre-computations
        self.to_z_init_i = to_z_init_i
        self.Z_j = to_z_init_j                # [I, c_z] - pre-computed key projection
        self.rpe_module = rpe_module
        self.rpe_module2 = rpe_module2
        self.token_bonds = token_bonds        # [I, I] - keep full (relatively small)
        self.process_token_bonds = process_token_bonds
        self.ref_pos_embedder = ref_pos_embedder
        self.ref_pos = ref_pos                # [I, 3]
        self.ref_space_uid = ref_space_uid    # [I]
        self.f = f
        self.transformer_stack = transformer_stack
        self.process_z_init = process_z_init
        self.transition_modules = transition_modules
        
        # Multi-GPU: Determine this GPU's query range
        self.gpu_rank, self.world_size = _get_gpu_rank_and_world_size()
        self.gpu_start, self.gpu_end = _compute_gpu_query_range(
            self.I, self.gpu_rank, self.world_size
        )
        self.I_par = self.gpu_end - self.gpu_start  # This GPU's chunk size
    
    def get_chunk(self, start_i: int, end_i: int) -> torch.Tensor:
        """
        Compute Z_II[start_i:end_i, :, :] on-the-fly.
        
        Computes [I_par, I, c_z] - queries in range attend to ALL keys.
        This is cross-attention style, avoiding full [I, I, c_z].
        
        Args:
            start_i: Start index for query tokens
            end_i: End index for query tokens
            
        Returns:
            Z_chunk: [I_par, I, c_z] pair features for query chunk
        """
        # Ensure all stored tensors/modules are on this rank's device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        self.S_I = self.S_I.to(device)
        self.Z_j = self.Z_j.to(device)
        self.to_z_init_i.to(device)
        self.token_bonds = self.token_bonds.to(device)
        self.ref_pos = self.ref_pos.to(device)
        self.ref_space_uid = self.ref_space_uid.to(device)
        self.process_token_bonds.to(device)
        self.rpe_module.to(device)
        self.rpe_module2.to(device)
        self.ref_pos_embedder.to(device)
        self.process_z_init.to(device)
        for block in self.transformer_stack:
            block.to(device)
        for mod in self.transition_modules:
            mod.to(device)
        # Move feature dict tensors to correct device
        for key, val in self.f.items():
            if isinstance(val, torch.Tensor):
                self.f[key] = val.to(device)
        
        I_par = end_i - start_i
        I = self.I
        qs = slice(start_i, end_i)
        
        # Step 1: Base Z = Z_i + Z_j (cross-attention style)
        # Z_i: [I_par, 1, c_z] - queries
        # Z_j: [1, I, c_z] - all keys
        S_I_chunk = self.S_I[qs]                          # [I_par, c_s]
        Z_i = self.to_z_init_i(S_I_chunk).unsqueeze(-2)   # [I_par, 1, c_z]
        Z_j = self.Z_j.unsqueeze(0)                       # [1, I, c_z]
        Z_chunk = Z_i + Z_j                               # [I_par, I, c_z]
        
        # Debug: track dimensions
        _dims = {"step1": Z_chunk.shape[-1]}
        
        # Step 2: Add RPE (chunked)
        Z_chunk = Z_chunk + self.rpe_module.forward_chunk(
            self.f, start_i, end_i
        )  # [I_par, I, c_z]
        _dims["step2_rpe"] = Z_chunk.shape[-1]
        
        # Step 3: Add token bonds (slice query rows, all key columns)
        token_bonds_chunk = self.token_bonds[qs, :]       # [I_par, I]
        Z_chunk = Z_chunk + self.process_token_bonds(
            token_bonds_chunk.unsqueeze(-1).float()
        )  # [I_par, I, c_z]
        
        # Step 4: Add reference position embedding (chunked)
        ref_pos_chunk = self.ref_pos[qs]                  # [I_par, 3]
        ref_space_uid_chunk = self.ref_space_uid[qs]      # [I_par]
        
        # Valid mask: [I_par, I, 1]
        valid_mask = (
            ref_space_uid_chunk.unsqueeze(-1) == self.ref_space_uid.unsqueeze(0)
        ).unsqueeze(-1)  # [I_par, I, 1]
        
        ref_pos_embed = self.ref_pos_embedder.forward_chunk(
            ref_pos_chunk, self.ref_pos, valid_mask
        )  # [I_par, I, c_z]
        Z_chunk = Z_chunk + ref_pos_embed
        
        # Step 5: Pairformer Z transitions (chunked along keys to save memory)
        for block in self.transformer_stack:
            Z_chunk = _z_transition_chunked(Z_chunk, block.z_transition)  # [I_par, I, c_z]
        
        # Step 6: Concatenate with second RPE and process
        rpe2_chunk = self.rpe_module2.forward_chunk(
            self.f, start_i, end_i
        )  # [I_par, I, c_z]
        _dims["step5_pre_cat"] = Z_chunk.shape[-1]
        _dims["rpe2_dim"] = rpe2_chunk.shape[-1]
        
        Z_chunk = torch.cat([Z_chunk, rpe2_chunk], dim=-1)  # [I_par, I, 2*c_z]
        _dims["step6_post_cat"] = Z_chunk.shape[-1]
        
        Z_chunk = self.process_z_init(Z_chunk)              # [I_par, I, c_z]
        _dims["step6_post_process"] = Z_chunk.shape[-1]
        
        # Step 7: Apply transitions (chunked along keys to save memory)
        for transition in self.transition_modules:
            Z_chunk = _z_transition_chunked(Z_chunk, transition)  # [I_par, I, c_z]
        
        # Debug check: verify dimensions match expected c_z
        actual_dim = Z_chunk.shape[-1]
        _dims["final"] = actual_dim
        
        if actual_dim != self.c_z:
            # Check where the mismatch originates
            z_i_dim = self.to_z_init_i(self.S_I[:1]).shape[-1]
            z_j_dim = self.Z_j.shape[-1]
            
            raise RuntimeError(
                f"StreamingZContainer.get_chunk dimension mismatch:\n"
                f"  Expected c_z={self.c_z}\n"
                f"  Got: {actual_dim}\n"
                f"  Dimension trace: {_dims}\n"
                f"  Z_i (to_z_init_i output) dim: {z_i_dim}\n"
                f"  Z_j dim: {z_j_dim}\n"
                f"  process_z_init expects: {2*self.c_z} -> {self.c_z}\n"
                f"  Note: process_z_init should output {self.c_z} dims"
            )
        
        return Z_chunk  # [I_par, I, c_z]
    
    def get_gpu_chunk(self) -> torch.Tensor:
        """
        Get Z rows for THIS GPU's assigned query range.
        
        Multi-GPU: Each GPU calls this to get only its portion.
        Returns [I_par, I, c_z] where I_par = I / world_size.
        
        Returns:
            Z_chunk: [I_par, I, c_z] - this GPU's Z rows
        """
        return self.get_chunk(self.gpu_start, self.gpu_end)
    
    def get_gpu_query_range(self) -> Tuple[int, int]:
        """Get this GPU's query range [start, end)."""
        return self.gpu_start, self.gpu_end
    
    def get_all_chunks(self, n_par: int) -> List[torch.Tensor]:
        """
        Get all chunks without assembling them (single-GPU mode).
        
        Args:
            n_par: Number of parallel chunks
            
        Returns:
            List of [I_par, I, c_z] tensors
        """
        ranges = _compute_chunk_ranges(self.I, n_par)
        return [self.get_chunk(start, end) for start, end in ranges]
    
    def assemble(self) -> torch.Tensor:
        """
        Assemble full Z_II tensor (for backward compatibility).
        
        WARNING: This materializes full [I, I, c_z] tensor!
        Only use when streaming is not possible downstream.
        
        Returns:
            Z_II: [I, I, c_z]
        """
        n_par = _get_n_parallel()
        if n_par <= 1:
            n_par = 4  # Default chunking for assembly
        chunks = self.get_all_chunks(n_par)
        return torch.cat(chunks, dim=0)  # [I, I, c_z]
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return virtual shape [I, I, c_z] for compatibility."""
        return (self.I, self.I, self.c_z)
    
    def get_base_z_at_pairs(
        self, 
        tok_queries: torch.Tensor,  # [L, k] or similar token indices for queries
        tok_keys: torch.Tensor,      # [L, k] or similar token indices for keys (same shape)
        chunk_size: int = 512,       # Process in chunks to avoid OOM
    ) -> torch.Tensor:
        """
        Compute BASE Z values at sparse (query, key) token pairs.
        
        This computes Z_i + Z_j WITHOUT the full RPE/token_bonds/etc processing.
        Used by ChunkedPairwiseEmbedder when Z_II is streaming.
        Processes in chunks to avoid OOM on large inputs.
        
        Args:
            tok_queries: Token indices for query positions [L, k]
            tok_keys: Token indices for key positions [L, k]
            chunk_size: Number of L rows to process at once
            
        Returns:
            Z_pairs: [L, k, c_z] Z values at the sparse pairs
        """
        L = tok_queries.shape[0]
        k = tok_queries.shape[1] if tok_queries.dim() > 1 else 1
        device = tok_queries.device
        
        # Ensure stored tensors and modules are on the correct device
        S_I = self.S_I.to(device)
        Z_j = self.Z_j.to(device)
        self.to_z_init_i.to(device)
        
        # Process in chunks to avoid OOM
        Z_pairs_list = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            tq_chunk = tok_queries[start:end]            # [chunk, k]
            tk_chunk = tok_keys[start:end]               # [chunk, k]
            
            # Z_i: query projection at tok_queries
            S_queries = S_I[tq_chunk]                    # [chunk, k, c_s]
            Z_i = self.to_z_init_i(S_queries)            # [chunk, k, c_z]
            
            # Z_j: key projection at tok_keys  
            Z_j_chunk = Z_j[tk_chunk]                    # [chunk, k, c_z]
            
            # Base Z = Z_i + Z_j
            Z_chunk = Z_i + Z_j_chunk                    # [chunk, k, c_z]
            Z_pairs_list.append(Z_chunk)
        
        return torch.cat(Z_pairs_list, dim=0)            # [L, k, c_z]


class TokenInitializer(nn.Module):
    """
    Token embedding module for RFD3.
    
    Supports three modes:
    1. Standard mode: Full L×L and I×I tensors materialized
    2. Chunked mode (use_chunked_pll): Sparse P_LL for attention
    3. Streaming mode (RFD3_ATTENTION_PARALLEL): No full I×I/L×L tensors ever
       - Returns StreamingZContainer instead of Z_II tensor
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
        self.n_parallel = _get_n_parallel()
        self.use_streaming = self.n_parallel > 1
        if self.use_streaming:
            logger.info(
                f"TokenInitializer: Streaming mode enabled with n_parallel={self.n_parallel}. "
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

    def forward(self, f):
        """
        Provides initial representation for atom and token representations.
        
        Returns:
            dict containing:
                - Q_L_init: [L, c_atom] initial atom features
                - C_L: [L, c_atom] conditioned atom features
                - S_I: [I, c_s] token single features
                - Z_II: [I, I, c_z] OR StreamingZContainer (streaming mode)
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
        
        Returns StreamingZContainer instead of Z_II tensor.
        """
        device = tok_idx.device
        dtype = self.to_z_init_i.weight.dtype
        
        # Ensure TokenInitializer modules are on the correct device for this rank
        self.to(device)
        
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
        
        # ============================================================
        # Step 2: Pre-compute Z_j (key projections) - [I, c_z], NOT I×I
        # ============================================================
        Z_j = self.to_z_init_j(S_I)                       # [I, c_z]
        
        # ============================================================
        # Step 3: Create StreamingZContainer for on-demand Z computation
        # ============================================================
        # Get reference positions for CA atoms
        ref_pos = f["ref_pos"][f["is_ca"]]                # [I, 3]
        ref_space_uid = f["ref_space_uid"][f["is_ca"]]    # [I]
        
        streaming_z = StreamingZContainer(
            S_I=S_I,                                       # [I, c_s]
            to_z_init_i=self.to_z_init_i,
            to_z_init_j=Z_j,                               # [I, c_z] pre-computed
            rpe_module=self.relative_position_encoding,
            rpe_module2=self.relative_position_encoding2,
            token_bonds=f["token_bonds"],                  # [I, I] (small, keep full)
            process_token_bonds=self.process_token_bonds,
            ref_pos_embedder=self.ref_pos_embedder_tok,
            ref_pos=ref_pos,                               # [I, 3]
            ref_space_uid=ref_space_uid,                   # [I]
            f=f,
            transformer_stack=self.transformer_stack,
            process_z_init=self.process_z_init,
            transition_modules=self.transition_1,
            c_z=self.c_z,
            device=device,
            dtype=dtype,
        )
        
        # ============================================================
        # Step 4: Compute atom features (Q_L_init, C_L)
        # ============================================================
        Q_L_init = self.atom_1d_embedder_2(f, L)          # [L, c_atom]
        C_L = Q_L_init + self.process_s_trunk(S_I)[..., tok_idx, :]  # [L, c_atom]
        
        return {
            "Q_L_init": Q_L_init,                          # [L, c_atom]
            "C_L": C_L,                                    # [L, c_atom]
            "chunked_pairwise_embedder": self.chunked_pairwise_embedder if self.use_chunked_pll else None,
            "S_I": S_I,                                    # [I, c_s]
            "Z_II": streaming_z,                           # StreamingZContainer (NOT tensor!)
            "streaming_mode": True,                        # Flag for downstream
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
            Z_init_II = Z_init_II + self.relative_position_encoding(f)  # [I, I, c_z]
            Z_init_II = Z_init_II + self.process_token_bonds(
                f["token_bonds"].unsqueeze(-1).float()
            )                                              # [I, I, c_z]

            # Embed reference coordinates of ligands
            token_id = f["ref_space_uid"][f["is_ca"]]     # [I]
            valid_mask = (token_id.unsqueeze(-1) == token_id.unsqueeze(-2)).unsqueeze(
                -1
            )                                              # [I, I, 1]
            Z_init_II = Z_init_II + self.ref_pos_embedder_tok(
                f["ref_pos"][f["is_ca"]], valid_mask
            )                                              # [I, I, c_z]

            # Run a small transformer to provide position encodings to single.
            for block in self.transformer_stack:
                S_I, Z_init_II = block(S_I, Z_init_II)    # [I, c_s], [I, I, c_z]

            # Also cat the relative position encoding and mix
            Z_init_II = torch.cat(
                [
                    Z_init_II,
                    self.relative_position_encoding2(f),
                ],
                dim=-1,
            )                                              # [I, I, 2*c_z]
            Z_init_II = self.process_z_init(Z_init_II)    # [I, I, c_z]
            for b in range(2):
                Z_init_II = Z_init_II + self.transition_1[b](Z_init_II)  # [I, I, c_z]

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
    
    Supports streaming mode where Z_init_II is a StreamingZContainer
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
        self.n_parallel = _get_n_parallel()
        self.use_streaming = self.n_parallel > 1

    def forward(self, f, R_L, S_init_I, Z_init_II, C_L, P_LL, **kwargs):
        """
        Forward pass for token encoding.
        
        Args:
            f: Feature dictionary
            R_L: [B, L, 3] scaled positions
            S_init_I: [I, c_s] initial single features
            Z_init_II: [I, I, c_z] OR StreamingZContainer
            C_L: [L, c_atom] atom conditioning features
            P_LL: [L, L, c_atompair] or None
            **kwargs: D_II_self for self-conditioning
            
        Returns:
            S_I: [I, c_s] updated single features  
            Z_II: [I, I, c_z] OR StreamingZContainer
        """
        # Check if Z_init_II is a StreamingZContainer
        is_streaming = isinstance(Z_init_II, StreamingZContainer)
        
        if is_streaming:
            return self._forward_streaming(f, R_L, S_init_I, Z_init_II, **kwargs)
        else:
            return self._forward_standard(f, R_L, S_init_I, Z_init_II, **kwargs)
    
    def _forward_streaming(self, f, R_L, S_init_I, Z_streaming, **kwargs):
        """
        True multi-GPU streaming forward: each GPU processes only its query chunk.
        
        Multi-GPU parallelism:
        - Each GPU computes Z rows for its assigned query range [gpu_start, gpu_end)
        - This produces [I_par, I, c_z] per GPU, never full [I, I, c_z]
        - Single features S_I are computed per-GPU then all_gathered
        - Distogram/self features are also computed in chunks
        
        Args:
            Z_streaming: StreamingZContainer (GPU-aware)
            
        Returns:
            S_I: [I, c_s] - updated single features (all_gathered)
            Z_II: [I_par, I, c_z] - THIS GPU's Z rows only (NOT full I×I!)
        """
        B = R_L.shape[0]
        I = Z_streaming.I
        device = R_L.device
        dtype = R_L.dtype
        
        # Ensure encoder modules are on the correct device for this rank
        self.to(device)
        
        # Get this GPU's query range
        gpu_start, gpu_end = Z_streaming.get_gpu_query_range()
        I_par = gpu_end - gpu_start                        # This GPU's chunk size
        world_size = Z_streaming.world_size
        
        # Step 1: Update S_I (operates on full I, no I×I)
        # S_init_I may have batch dim [B, I, c_s] or not [I, c_s]
        S_I = S_init_I
        has_batch_S = S_I.dim() == 3
        for b in range(2):
            S_I = S_I + self.transition_1[b](S_I)
        
        # Step 2: Get THIS GPU's Z chunk (never materializes full I×I)
        Z_chunk = Z_streaming.get_gpu_chunk()              # [I_par, I, c_z]
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
        
        # Step 4: Add self-conditioning for this GPU's chunk
        #
        # IMPORTANT: Base Z comes from TokenInitializer (Z_streaming.c_z), NOT self.c_z!
        # The expected dimension for process_z is calculated at init time using self.c_z,
        # so we MUST ensure TokenInitializer.c_z == DiffusionTokenEncoder.c_z.
        #
        base_z_dim = Z_streaming.c_z  # TokenInitializer's c_z
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
                # Slice self-conditioning to this GPU's query rows
                D_self_chunk = D_II_self[:, gpu_start:gpu_end, :]  # [B, I_par, I, n_bins]
            else:
                D_self_chunk = torch.zeros(
                    B, I_par, I, self.n_bins_distogram,
                    device=device, dtype=dtype
                )                                          # [B, I_par, I, n_bins]
            Z_chunk = torch.cat([Z_chunk, D_self_chunk], dim=-1)
        
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
        Z_chunk = self.process_z(Z_chunk)                # [B, I_par, I, c_z]
        
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
        
        # Step 7: All-gather S_I chunks from all GPUs
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
