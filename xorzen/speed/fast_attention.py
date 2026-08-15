"""
fast_attention.py — C++-backed LocalAttentionPathway drop-in
=============================================================
Uses the AVX2 window attention kernel + cached masks.
Falls back to F.scaled_dot_product_attention (PyTorch ≥ 2.0) or manual
path if C++ extension is not compiled.
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False

_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

# Global window-mask cache keyed by (seq_len, window_size, device_str)
_MASK_CACHE: Dict[Tuple, torch.Tensor] = {}
_MASK_CACHE_MAX = 64


def _get_window_mask(S: int, window: int, device: torch.device) -> torch.Tensor:
    key = (S, window, str(device))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    pos  = torch.arange(S, device=device)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
    mask = dist <= window
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        del _MASK_CACHE[next(iter(_MASK_CACHE))]
    _MASK_CACHE[key] = mask
    return mask


class FlashLocalAttention(nn.Module):
    """
    Drop-in for LocalAttentionPathway.
    Priority: C++ kernel → SDPA → manual.
    """

    def __init__(self, hidden_dim, num_heads, window_size,
                 dropout=0.0, causal=True):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        self.window_size = window_size
        self.causal      = causal
        self.dropout_p   = dropout

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    @classmethod
    def from_slow(cls, slow: nn.Module) -> "FlashLocalAttention":
        dp = slow.attn_dropout.p if hasattr(slow.attn_dropout, "p") else 0.0
        fast = cls(slow.hidden_dim, slow.num_heads, slow.window_size,
                   dropout=dp, causal=slow.causal)
        for attr in ["q_proj", "k_proj", "v_proj", "out_proj", "ln_q", "ln_k"]:
            getattr(fast, attr).load_state_dict(getattr(slow, attr).state_dict())
        return fast

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim

        def _proj(proj, inp):
            return proj(inp).view(B, S, H, D).transpose(1, 2)

        q = _proj(self.q_proj, x)
        k = _proj(self.k_proj, x)
        v = _proj(self.v_proj, x)

        # QK-norm
        q = self.ln_q(q.transpose(1, 2)).transpose(1, 2)
        k = self.ln_k(k.transpose(1, 2)).transpose(1, 2)

        # ── C++ kernel path (CPU only) ────────────────────────────
        if _CPP_AVAILABLE and not x.is_cuda and position_bias is None and attention_mask is None:
            # cpp_window_attention expects [B, H, S, D] float32 contiguous
            q_c = q.contiguous().float()
            k_c = k.contiguous().float()
            v_c = v.contiguous().float()
            out = _ext.cpp_window_attention(q_c, k_c, v_c, self.window_size)
            out = out.to(x.dtype).view(B, H, S, D)
            out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)
            out = self.out_proj(out)
            return self.resid_dropout(out)

        # ── Build additive bias for SDPA / manual path ────────────
        bias: Optional[torch.Tensor] = None
        need_bias = (self.window_size > 0 or position_bias is not None
                     or attention_mask is not None)

        if need_bias:
            bias = torch.zeros(1, 1, S, S, device=x.device, dtype=x.dtype)
            if self.window_size > 0:
                win = _get_window_mask(S, self.window_size, x.device)
                bias = bias.masked_fill(~win[None, None], float("-inf"))
            if self.causal:
                cm = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
                bias = bias.masked_fill(~cm[None, None], float("-inf"))
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, None, :]
                pad = attention_mask.to(x.dtype)
                bias = bias + pad.masked_fill(pad == 0, float("-inf")).masked_fill(pad != 0, 0.0)
            if position_bias is not None:
                bias = bias + position_bias

        # ── SDPA path ─────────────────────────────────────────────
        if _SDPA_AVAILABLE:
            dp = self.dropout_p if self.training else 0.0
            is_causal = self.causal and bias is None
            out = F.scaled_dot_product_attention(q, k, v,
                      attn_mask=bias, dropout_p=dp, is_causal=is_causal)
        else:
            # Manual path
            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if bias is not None:
                scores = scores + bias
            elif self.causal:
                cm = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
                scores = scores.masked_fill(~cm[None, None], float("-inf"))
            probs = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                probs = F.dropout(probs, p=self.dropout_p)
            out = torch.matmul(probs, v)

        out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)
        out = self.out_proj(out)
        return self.resid_dropout(out)

    def get_compute_stats(self, seq_len, batch_size=1):
        qkv = 3 * batch_size * seq_len * self.hidden_dim ** 2
        attn = batch_size * self.num_heads * seq_len * seq_len * self.head_dim * 2
        out  = batch_size * seq_len * self.hidden_dim ** 2
        return {
            "flops_total"             : qkv + attn + out,
            "flops_per_token"         : (qkv + attn + out) / (batch_size * seq_len),
            "param_count"             : sum(p.numel() for p in self.parameters()),
            "param_memory_bytes"      : sum(p.numel() for p in self.parameters()) * 4,
            "activation_memory_bytes" : batch_size * seq_len * self.hidden_dim * 4 * 10,
            "window_size"             : self.window_size,
            "cpp_kernel_active"       : _CPP_AVAILABLE,
            "sdpa_active"             : _SDPA_AVAILABLE,
        }
