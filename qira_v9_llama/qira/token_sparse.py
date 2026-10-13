"""
Token-Sparse Attention Module (Jo et al., ICML 2026).
Provides dynamic per-head token sparsification for Transformer prefill.
Fully decoupled from transformers package for maximum portability and speed.
"""

import math
from typing import Optional, List, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSparseAttentionWrapper:
    """
    Modular implementation of Token Sparse Attention (Jo et al., ICML 2026).
    Applies per-head dynamic token selection during prefill on layers >= L_dec.
    Supports native Triton kernel or vectorized PyTorch fallback.
    """
    def __init__(
        self,
        num_layers: int,
        decision_layer_idx: int,
        sparse_layers: Optional[List[int]] = None,
        coverage: float = 0.01,
        window_size: int = 128,
        kernel_size: int = 7,
        min_tokens: int = 512,
    ):
        self.num_layers = num_layers
        self.decision_layer_idx = decision_layer_idx
        self.coverage = coverage
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.min_tokens = min_tokens

        # Default sparsification starts at midpoint (matching QIRA L_dec)
        if sparse_layers is None:
            self.sparse_layers = set(range(decision_layer_idx, num_layers))
        else:
            self.sparse_layers = set(sparse_layers)

        # Check for triton availability
        self.use_triton = False
        try:
            import triton
            from qira.tsa_kernel import triton_get_attn_cache
            self.triton_fn = triton_get_attn_cache
            self.use_triton = True
        except Exception:
            self.use_triton = False

    def compute_prefix_importance_pytorch(
        self,
        q: torch.Tensor,  # (bsz, num_heads, seq_len, head_dim)
        k: torch.Tensor,  # (bsz, num_heads, seq_len, head_dim)
    ) -> torch.Tensor:
        """
        Pure PyTorch exact vectorized implementation of Token-Sparse Attention's
        windowed query to prefix key importance scoring.
        """
        bsz, num_heads, seq_len, head_dim = q.shape
        w = self.window_size
        prefix_len = seq_len - w
        if prefix_len <= 0:
            return torch.zeros((bsz, num_heads, 0), device=q.device, dtype=q.dtype)

        # Trailing window queries [seq_len - w, seq_len)
        q_win = q[:, :, prefix_len:, :]  # (bsz, num_heads, w, head_dim)

        # Attention scores from window queries to all keys
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q_win, k.transpose(-1, -2)) * scale

        # Causal mask for the trailing window
        q_pos = torch.arange(prefix_len, seq_len, device=q.device)[:, None]
        k_pos = torch.arange(seq_len, device=q.device)[None, :]
        causal_mask = k_pos > q_pos
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax over all valid keys for each window query
        probs = F.softmax(scores.float(), dim=-1).to(q.dtype)  # (bsz, num_heads, w, seq_len)

        # Average attention received by each prefix key across all window queries
        prefix_probs = probs[:, :, :, :prefix_len].mean(dim=2)  # (bsz, num_heads, prefix_len)
        return prefix_probs

    def get_sparse_indices(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Computes the retained token indices for Token-Sparse Attention.
        Returns tensor of shape (bsz, num_heads, compressed_len) or None if dense.
        """
        bsz, num_heads, seq_len, head_dim = q.shape
        if seq_len <= (self.min_tokens + self.window_size) or self.coverage <= 0.0:
            return None

        # 1. Compute prefix attention cache
        if self.use_triton and q.is_cuda:
            try:
                attn_cache = self.triton_fn(
                    query_states=q,
                    key_states=k,
                    window_size=self.window_size,
                    out_dtype=q.dtype
                )
            except Exception:
                attn_cache = self.compute_prefix_importance_pytorch(q, k)
        else:
            attn_cache = self.compute_prefix_importance_pytorch(q, k)

        if attn_cache.shape[-1] == 0:
            return None

        # 2. 1D Average Pooling smoothing across sequence
        pad = self.kernel_size // 2
        attn_cache_smooth = F.avg_pool1d(attn_cache, kernel_size=self.kernel_size, padding=pad, stride=1)
        if attn_cache_smooth.shape[-1] > attn_cache.shape[-1]:
            attn_cache_smooth = attn_cache_smooth[:, :, :attn_cache.shape[-1]]

        # 3. Global normalization & cumulative coverage cutoff
        attn_cache_global = attn_cache_smooth.mean(dim=1)  # (bsz, prefix_len)
        norm_global = attn_cache_global / (attn_cache_global.sum(dim=-1, keepdim=True) + 1e-8)
        norm_per_head = attn_cache_smooth / (attn_cache_smooth.sum(dim=-1, keepdim=True) + 1e-8)

        sorted_global, _ = torch.sort(norm_global, dim=-1, descending=False)
        cum_coverage = torch.cumsum(sorted_global.float(), dim=-1)

        # Number of tokens to sparse out (prune)
        num_sparse_tokens = (cum_coverage < self.coverage).sum(dim=-1).item()
        if num_sparse_tokens == 0:
            return None

        prefix_len = seq_len - self.window_size
        max_capacity = max(self.min_tokens, prefix_len - num_sparse_tokens)
        max_capacity = min(prefix_len, max_capacity)

        # Top-k prefix tokens per head
        topk_indices = norm_per_head.topk(max_capacity, dim=-1).indices

        # Window tokens are unconditionally preserved
        win_indices = torch.arange(prefix_len, seq_len, device=q.device)
        win_indices = win_indices.unsqueeze(0).unsqueeze(0).expand(bsz, num_heads, -1)

        # Concatenate and sort
        indices = torch.cat([topk_indices, win_indices], dim=-1).sort(dim=-1, descending=False)[0]
        return indices
