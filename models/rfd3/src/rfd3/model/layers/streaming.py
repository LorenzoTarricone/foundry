"""
Streaming utilities for memory-efficient inference.

This module provides utilities for processing token and atom features in a 
streaming/chunked manner to avoid materializing full L×L or I×I tensors.

Key concepts:
- n_par: parallelism factor (typically = number of GPUs)
- I_par = I // n_par: number of token queries per chunk
- L_par = L // n_par: number of atom queries per chunk

In streaming mode:
- Instead of self-attention (queries attend to all keys including themselves),
  we use cross-attention where each chunk of queries attends to ALL keys.
- This produces tensors of shape [I_par, I, c] instead of [I, I, c]
- Outputs are concatenated along the query dimension to produce full outputs.
"""

import logging
import os
from typing import Optional, Tuple, List, Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def is_streaming_mode() -> bool:
    """
    Check if streaming/parallel mode is enabled.

    Env var scheme:
      - RFD3_ATTENTION_PARALLEL=0 or unset → standard mode (False)
      - RFD3_ATTENTION_PARALLEL=1 or any non-zero value → parallel mode (True)
    """
    val = os.environ.get("RFD3_ATTENTION_PARALLEL", "0")
    return val not in ("0", "", "false", "False")


def get_world_size() -> int:
    """Get actual GPU count from distributed runtime."""
    import torch.distributed as dist
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def compute_chunk_ranges(total: int, n_par: int) -> List[Tuple[int, int]]:
    """
    Compute start/end indices for chunked processing.

    Uses floor division with remainder distributed to early ranks.
    For 13800 tokens / 7 GPUs: ranks 0-2 get 1972, ranks 3-6 get 1971.

    Args:
        total: Total number of elements (I or L)
        n_par: Number of parallel chunks

    Returns:
        List of (start, end) tuples for each chunk
    """
    chunk_size = total // n_par
    remainder = total % n_par

    ranges = []
    for rank in range(n_par):
        if rank < remainder:
            start = rank * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = rank * chunk_size + remainder
            end = start + chunk_size
        if start < total:
            ranges.append((start, end))
    return ranges


class StreamingPairFeatures:
    """
    Container for streaming pair features (Z_II or P_LL).
    
    Instead of storing a full [I, I, c] tensor, this stores:
    - S_i: [I, c_z] - single features for computing Z[i, :] = f(S_i) + g(S_j)
    - Additional per-token features needed for reconstruction
    
    When a row chunk is needed, it can be computed on-the-fly.
    """
    
    def __init__(
        self,
        S_I: torch.Tensor,           # [I, c_s] - single token features
        to_z_i: nn.Module,           # Linear layer for query projection
        to_z_j: nn.Module,           # Linear layer for key projection
        rpe_module: nn.Module,       # Relative position encoding module
        token_bonds: torch.Tensor,   # [I, I] - token bond matrix (kept full, small)
        process_token_bonds: nn.Module,  # Linear layer for token bonds
        ref_pos_embedder: nn.Module,     # Reference position embedder
        ref_pos: torch.Tensor,           # [I, 3] - reference positions for CA atoms
        ref_space_uid: torch.Tensor,     # [I] - reference space UIDs
        f: dict,                         # Feature dictionary for RPE
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize streaming pair features.
        
        Args:
            S_I: Single token features [I, c_s]
            to_z_i: Projection for query tokens
            to_z_j: Projection for key tokens
            rpe_module: Relative position encoding module
            token_bonds: Token bond matrix [I, I]
            process_token_bonds: Linear for token bonds
            ref_pos_embedder: Reference position embedder
            ref_pos: Reference positions [I, 3]
            ref_space_uid: Reference space UIDs [I]
            f: Feature dictionary
            device: Device for tensors
            dtype: Data type for tensors
        """
        self.S_I = S_I                    # [I, c_s]
        self.to_z_i = to_z_i
        self.to_z_j = to_z_j
        self.rpe_module = rpe_module
        self.token_bonds = token_bonds    # [I, I] - small enough to keep full
        self.process_token_bonds = process_token_bonds
        self.ref_pos_embedder = ref_pos_embedder
        self.ref_pos = ref_pos            # [I, 3]
        self.ref_space_uid = ref_space_uid  # [I]
        self.f = f
        self.device = device
        self.dtype = dtype
        
        self.I = S_I.shape[0]
        self.c_z = to_z_i.out_features if hasattr(to_z_i, 'out_features') else None
        
        # Pre-compute key projections (used for all query chunks)
        # Z_j: [I, c_z] - contribution from keys
        self.Z_j = to_z_j(S_I)  # [I, c_z]
    
    def get_row_chunk(
        self,
        start_i: int,
        end_i: int,
    ) -> torch.Tensor:
        """
        Compute Z_II[start_i:end_i, :, :] on-the-fly.
        
        Args:
            start_i: Start index for query tokens
            end_i: End index for query tokens
            
        Returns:
            Z_chunk: [I_par, I, c_z] - pair features for query chunk
        """
        I_par = end_i - start_i  # Number of queries in this chunk
        I = self.I
        
        # Step 1: Compute Z = Z_i + Z_j for this chunk
        # Z_i: [I_par, 1, c_z] (queries)
        # Z_j: [1, I, c_z] (all keys)
        S_I_chunk = self.S_I[start_i:end_i]  # [I_par, c_s]
        Z_i = self.to_z_i(S_I_chunk).unsqueeze(-2)  # [I_par, 1, c_z]
        Z_j = self.Z_j.unsqueeze(0)  # [1, I, c_z]
        Z_chunk = Z_i + Z_j  # [I_par, I, c_z]
        
        # Step 2: Add relative position encoding (chunked)
        # Only compute RPE for the query chunk
        Z_chunk = Z_chunk + self.rpe_module.forward_chunk(
            self.f, start_i, end_i
        )  # [I_par, I, c_z]
        
        # Step 3: Add token bonds
        token_bonds_chunk = self.token_bonds[start_i:end_i, :]  # [I_par, I]
        Z_chunk = Z_chunk + self.process_token_bonds(
            token_bonds_chunk.unsqueeze(-1).float()
        )  # [I_par, I, c_z]
        
        # Step 4: Add reference position embedding
        ref_pos_chunk = self.ref_pos[start_i:end_i]  # [I_par, 3]
        ref_space_uid_chunk = self.ref_space_uid[start_i:end_i]  # [I_par]
        
        # Valid mask: [I_par, I, 1]
        valid_mask = (
            ref_space_uid_chunk.unsqueeze(-1) == self.ref_space_uid.unsqueeze(0)
        ).unsqueeze(-1)  # [I_par, I, 1]
        
        # Compute pairwise distances for ref_pos embedding
        ref_pos_embed = self.ref_pos_embedder.forward_chunk(
            ref_pos_chunk, self.ref_pos, valid_mask
        )  # [I_par, I, c_z]
        Z_chunk = Z_chunk + ref_pos_embed
        
        return Z_chunk  # [I_par, I, c_z]


def streaming_pairformer_attention(
    A_I: torch.Tensor,           # [I, c_a] - single features
    Z_II_streaming: StreamingPairFeatures,  # Streaming pair features
    to_q: nn.Module,
    to_k: nn.Module,
    to_v: nn.Module,
    to_b: nn.Module,
    to_g: nn.Module,
    to_a: nn.Module,
    ln_0: nn.Module,
    ln_1: nn.Module,
    n_head: int,
    c_head: int,
    n_par: int,
) -> torch.Tensor:
    """
    Streaming version of Pairformer attention that never materializes full I×I.
    
    Processes queries in chunks, each chunk attending to all keys.
    
    Args:
        A_I: Single features [I, c_a]
        Z_II_streaming: Streaming pair features container
        to_q, to_k, to_v, to_b, to_g, to_a: Projection layers
        ln_0, ln_1: Layer norms
        n_head: Number of attention heads
        c_head: Dimension per head
        n_par: Parallelism factor
        
    Returns:
        A_I_out: Updated single features [I, c_a]
    """
    I = A_I.shape[0]
    device = A_I.device
    dtype = A_I.dtype
    c_a = A_I.shape[-1]
    
    # Pre-compute key and value projections (used for all query chunks)
    A_I_normed = ln_1(A_I)  # [I, c_a]
    if A_I_normed.dtype != torch.bfloat16:
        A_I_normed = A_I_normed.to(torch.bfloat16)
    
    K_IH = to_k(A_I_normed)  # [I, n_head, c_head]
    V_IH = to_v(A_I_normed)  # [I, n_head, c_head]
    
    # Output tensor
    A_out = torch.zeros(I, c_a, device=device, dtype=dtype)
    
    # Process queries in chunks
    chunk_ranges = compute_chunk_ranges(I, n_par)
    
    for start_i, end_i in chunk_ranges:
        I_par = end_i - start_i  # Number of queries in this chunk
        
        # Get query projections for this chunk
        A_chunk_normed = A_I_normed[start_i:end_i]  # [I_par, c_a]
        Q_chunk = to_q(A_chunk_normed)  # [I_par, n_head, c_head]
        G_chunk = to_g(A_chunk_normed)  # [I_par, n_head, c_head]
        
        # Scale queries
        Q_chunk = Q_chunk / torch.sqrt(
            torch.tensor(c_head, device=device, dtype=torch.bfloat16)
        )
        
        # Get pair bias for this chunk: [I_par, I, n_head]
        Z_chunk = Z_II_streaming.get_row_chunk(start_i, end_i)  # [I_par, I, c_z]
        B_chunk = to_b(ln_0(Z_chunk))  # [I_par, I, n_head]
        
        # Attention: Q_chunk @ K^T + B
        # Q: [I_par, n_head, c_head]
        # K: [I, n_head, c_head]
        # attn: [I_par, n_head, I] after einsum, need [I_par, I, n_head]
        attn_logits = torch.einsum(
            "ihd,jhd->ijh", Q_chunk, K_IH
        ) + B_chunk  # [I_par, I, n_head]
        
        attn_weights = torch.softmax(attn_logits, dim=1)  # softmax over keys (dim=1)
        
        # Apply attention to values
        # attn_weights: [I_par, I, n_head]
        # V_IH: [I, n_head, c_head]
        attn_out = torch.einsum(
            "ijh,jhc->ihc", attn_weights, V_IH
        )  # [I_par, n_head, c_head]
        
        # Gating
        attn_out = G_chunk * attn_out  # [I_par, n_head, c_head]
        
        # Flatten heads
        attn_out = attn_out.flatten(start_dim=-2)  # [I_par, c_a]
        
        # Output projection
        A_out[start_i:end_i] = to_a(attn_out)
    
    return A_out  # [I, c_a]


def streaming_transition(
    Z_streaming: StreamingPairFeatures,
    transition: nn.Module,
    n_par: int,
) -> List[torch.Tensor]:
    """
    Apply transition layer to streaming pair features.
    
    Returns list of Z chunks rather than assembling full tensor.
    
    Args:
        Z_streaming: Streaming pair features
        transition: Transition layer
        n_par: Parallelism factor
        
    Returns:
        List of Z chunks after transition, each [I_par, I, c_z]
    """
    I = Z_streaming.I
    chunk_ranges = compute_chunk_ranges(I, n_par)
    
    Z_chunks = []
    for start_i, end_i in chunk_ranges:
        Z_chunk = Z_streaming.get_row_chunk(start_i, end_i)  # [I_par, I, c_z]
        Z_chunk = Z_chunk + transition(Z_chunk)  # [I_par, I, c_z]
        Z_chunks.append(Z_chunk)
    
    return Z_chunks


def assemble_chunks(chunks: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    Assemble chunks along specified dimension.
    
    Args:
        chunks: List of tensor chunks
        dim: Dimension to concatenate along
        
    Returns:
        Assembled tensor
    """
    return torch.cat(chunks, dim=dim)


