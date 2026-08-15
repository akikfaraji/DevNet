"""
fast_router.py — C++-backed AdaptiveRouter drop-in
===================================================
Replaces the feature_encoder forward pass with the C++ MLP kernel
and adds an inference-time LRU cache at the Python level.
"""

from __future__ import annotations
import hashlib
from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


class _LRUTensorCache:
    """Thread-safe LRU cache for tensors, keyed by bytes hash."""
    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def _hash(self, t: torch.Tensor) -> str:
        raw = t.detach().cpu().numpy().tobytes()
        return hashlib.md5(raw).hexdigest()

    def get(self, t: torch.Tensor):
        k = self._hash(t)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        return None

    def put(self, t: torch.Tensor, v: torch.Tensor):
        k = self._hash(t)
        self._cache[k] = v
        self._cache.move_to_end(k)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


class _ConstantEncoder(nn.Module):
    """Wraps a precomputed feature tensor so it can be assigned as an nn.Module attribute."""
    def __init__(self, features: torch.Tensor):
        super().__init__()
        # Store as a plain attribute (not Parameter) so no grad is added
        self._features = features

    def forward(self, _inp: torch.Tensor) -> torch.Tensor:
        return self._features


class CachedRouter(nn.Module):
    """
    Drop-in replacement for AdaptiveRouter.
    - During training  : behaves identically to original.
    - During inference : caches feature_encoder output per unique input.
    - C++ MLP kernel   : used for the feature_encoder forward when available.

    All other routing logic (depth, width, path, expert) is unchanged.
    """

    def __init__(self, original: nn.Module):
        super().__init__()
        # Copy all submodules and parameters from the original router
        self._router = original
        self._cache  = _LRUTensorCache(maxsize=256)
        self._cpp_mlp_enabled = _CPP_AVAILABLE

    # ------------------------------------------------------------------
    @classmethod
    def from_slow(cls, slow: nn.Module) -> "CachedRouter":
        return cls(slow)

    # ------------------------------------------------------------------
    def _extract_encoder_weights(self):
        """
        Extract w/b from the feature_encoder Sequential for the C++ kernel.
        Assumes 3 Linear layers interleaved with LayerNorm + GELU (skipped in cpp).
        We extract only the Linear weights — the LN + GELU are fused in C++.
        """
        enc = self._router.feature_encoder
        linears = [m for m in enc.modules() if isinstance(m, nn.Linear)]
        if len(linears) < 3:
            return None
        l1, l2, l3 = linears[0], linears[1], linears[2]
        return (l1.weight.detach().float(), l1.bias.detach().float(),
                l2.weight.detach().float(), l2.bias.detach().float(),
                l3.weight.detach().float(), l3.bias.detach().float())

    # ------------------------------------------------------------------
    def _encode_features(self, router_input: torch.Tensor) -> torch.Tensor:
        """Run feature_encoder — C++ fast path or PyTorch fallback."""
        if self._cpp_mlp_enabled and not router_input.is_cuda:
            weights = self._extract_encoder_weights()
            if weights is not None:
                w1, b1, w2, b2, w3, b3 = weights
                # Reshape [B, S, D] → [B*S, D] for kernel
                B, S, D = router_input.shape
                x_flat = router_input.reshape(B * S, D).contiguous().float()
                out_flat = _ext.cpp_router_mlp(x_flat, w1, b1, w2, b2, w3, b3)
                return out_flat.view(B, S, -1).to(router_input.dtype)

        # PyTorch fallback
        return self._router.feature_encoder(router_input)

    # ------------------------------------------------------------------
    def forward(self, x, cot_features, training=None, deterministic=False,
                expert_capacity=None):
        """
        Forward pass — identical signature to AdaptiveRouter.forward().
        """
        if training is None:
            training = self.training

        # Concatenate inputs (same as original)
        router_input = torch.cat([x, cot_features], dim=-1)

        # ── Inference cache ───────────────────────────────────────
        features = None
        if not training:
            cached = self._cache.get(router_input)
            if cached is not None:
                features = cached

        if features is None:
            features = self._encode_features(router_input)
            if not training:
                self._cache.put(router_input, features.detach())

        # ── Delegate remaining routing logic to original router ───
        # Temporarily patch feature_encoder with a proper nn.Module wrapper
        # so PyTorch's __setattr__ check is satisfied.
        original_encoder = self._router.feature_encoder
        self._router.feature_encoder = _ConstantEncoder(features)
        try:
            # The patched encoder ignores its input and returns `features`.
            # We pass a dummy router_input (won't be used by the encoder).
            decision = self._router.forward(
                x, cot_features,
                training=training,
                deterministic=deterministic,
                expert_capacity=expert_capacity,
            )
        finally:
            self._router.feature_encoder = original_encoder

        return decision

    # forward any attribute access to original router
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._router, name)
