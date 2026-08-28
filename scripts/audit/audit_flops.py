"""
Xorzen Active-Compute Audit — Part 2: FLOPs measurement
=========================================================
Independently measure the actual FLOPs of a forward pass for each
variant (small ones only — 7B is too big for CPU here).

We use a manual counter via forward hooks on Linear/Conv/Embedding
layers.  This is the only way to get accurate per-layer FLOPs for a
model with Python-level routing loops that thop can't introspect.

Output: /home/z/my-project/download/audit/flops.json
"""
from __future__ import annotations
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("XORZENX_VERBOSE", "0")

import torch
import torch.nn as nn

from xorzen.utils.logger import get_logger
_ul = get_logger()
_underlying = getattr(_ul, "logger", None) or _ul
if hasattr(_underlying, "setLevel"):
    import logging
    _underlying.setLevel(logging.WARNING)

import xorzen

OUT = Path("/home/z/my-project/download/audit/flops.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# Manual FLOP counter using forward hooks
# ---------------------------------------------------------------
# Convention: count MACs (multiply-accumulate) as 2 FLOPs each.
# Linear: 2 * in_features * out_features * (batch...)
# Conv1d (depthwise): 2 * kernel * out_channels * L
# Conv1d (full): 2 * kernel * in_channels * out_channels * L
# Embedding: 0 (lookup, not MAC)
# LayerNorm: ~2 * D per element
# Softmax: ~3 * D per element
# We count only the heavy ops (Linear + Conv1d) for simplicity, and
# separately count attention scores and SSM scan via custom logic.

class FlopCounter:
    """Lightweight hook-based FLOP counter."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[Any] = []
        self.linear_flops: Dict[str, int] = {}
        self.conv_flops: Dict[str, int] = {}
        self.attn_flops: Dict[str, int] = {}
        self.ssm_flops: Dict[str, int] = {}

    def _hook_linear(self, module: nn.Linear, inputs, output, name: str):
        # inputs[0]: [..., in_features]
        x = inputs[0]
        n = x.numel() // x.shape[-1]
        # MACs = n * in * out ; FLOPs = 2 * MACs
        flops = 2 * n * module.in_features * module.out_features
        self.linear_flops[name] = self.linear_flops.get(name, 0) + flops

    def _hook_conv1d(self, module: nn.Conv1d, inputs, output, name: str):
        x = inputs[0]  # [B, C_in, L]
        out = output   # [B, C_out, L_out]
        L_out = out.shape[-1]
        if module.groups == module.in_channels:
            # depthwise
            macs = module.kernel_size[0] * module.out_channels * L_out * x.shape[0]
        else:
            macs = module.kernel_size[0] * module.in_channels * module.out_channels * L_out * x.shape[0]
        self.conv_flops[name] = self.conv_flops.get(name, 0) + 2 * macs

    def attach(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                h = mod.register_forward_hook(lambda m, i, o, n=name: self._hook_linear(m, i, o, n))
                self.handles.append(h)
            elif isinstance(mod, nn.Conv1d):
                h = mod.register_forward_hook(lambda m, i, o, n=name: self._hook_conv1d(m, i, o, n))
                self.handles.append(h)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def total(self) -> int:
        return sum(self.linear_flops.values()) + sum(self.conv_flops.values())

    def by_component(self) -> Dict[str, int]:
        """Bucket FLOPs by top-level component."""
        buckets: Dict[str, int] = {}
        for name, flops in self.linear_flops.items():
            bucket = self._bucket(name)
            buckets[bucket] = buckets.get(bucket, 0) + flops
        for name, flops in self.conv_flops.items():
            bucket = self._bucket(name)
            buckets[bucket] = buckets.get(bucket, 0) + flops
        return buckets

    @staticmethod
    def _bucket(name: str) -> str:
        if name.startswith("token_embedding") or name.startswith("position_embedding"):
            return "embeddings"
        if name.startswith("blocks."):
            parts = name.split(".")
            sub = parts[2]
            if sub == "pathways":
                return f"hass_{parts[3]}"
            elif sub == "pathway_gate":
                return "hass_pathway_gate"
            elif sub == "ffn":
                return "hass_ffn"
            else:
                return "hass_other"
        if name.startswith("router."):
            return "router"
        if name.startswith("moe."):
            return "moe_fabric"
        if name.startswith("merger."):
            return "merger"
        if name.startswith("cot."):
            return "cot"
        if name.startswith("lm_head"):
            return "lm_head"
        return "other"

    def attention_flops_manual(self, seq_len: int, batch: int, hidden: int, heads: int, window: int, layer_count: int) -> int:
        """Estimate local-attention FLOPs separately (not captured by Linear hooks)."""
        head_dim = hidden // heads
        # QK^T: 2 * B * H * T * T * head_dim  (but windowed)
        # We use the full causal triangle estimate; window is half-width
        # so effective pairs = B * H * T * (2*window+1) approx.
        # For simplicity compute the actual triangular count
        # pairs = T*(T+1)/2 for full causal, but windowed = T*(2*W+1) - W*(W+1)
        T = seq_len
        W = window
        # Per-head, per-batch attention pairs:
        if W >= T:
            pairs = T * (T + 1) // 2  # full causal
        else:
            # For each position t, attends to max(0, t-W)..t  => (t+1) up to W+1, then W+1
            pairs = sum(min(t + 1, W + 1) for t in range(T))
        qk_flops = 2 * batch * heads * pairs * head_dim
        sv_flops = 2 * batch * heads * pairs * head_dim
        per_layer = qk_flops + sv_flops
        return per_layer * layer_count

    def ssm_flops_manual(self, seq_len: int, batch: int, hidden: int, state_dim: int, layer_count: int) -> int:
        """Estimate SSM scan FLOPs separately (sequential loop, not captured by hooks)."""
        # Per layer per token: state update = 2 * state_dim (Ab * state + Bv)
        #                          output = 2 * state_dim (C * state)
        per_token_per_layer = 4 * state_dim
        total = 2 * batch * seq_len * layer_count * per_token_per_layer
        return total


def measure_variant(variant_name: str, seq_lens: List[int] = (16, 32, 64, 128)) -> Dict[str, Any]:
    print(f"\n=== FLOPs: {variant_name} ===")
    info: Dict[str, Any] = {"variant": variant_name, "by_seq_len": {}}
    try:
        if variant_name == "zero_tiny_23k":
            m = xorzen.zero_tiny_23k(test_mode=True)
        elif variant_name == "zero_1M":
            m = xorzen.zero_1M(test_mode=True)
        elif variant_name == "zero_10M":
            m = xorzen.zero_10M(test_mode=True)
        elif variant_name == "zero_50M":
            m = xorzen.zero_50M(test_mode=True)
        else:
            return {"variant": variant_name, "error": "skipped (too big for CPU)"}
    except Exception as e:
        return {"variant": variant_name, "error": f"{type(e).__name__}: {e}"}

    cfg = m.config
    info["config"] = {
        "hidden_size": cfg.hidden_size,
        "num_layers": cfg.num_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "vocab_size": cfg.vocab_size,
        "expert_count": cfg.expert_count,
        "top_k_experts": cfg.top_k_experts,
        "expert_hidden_multiplier": cfg.expert_hidden_multiplier,
        "local_window_size": cfg.local_window_size,
        "low_rank_dim": cfg.low_rank_dim,
        "ssm_state_dim": cfg.ssm_state_dim,
        "context_length": cfg.context_length,
        "tie_word_embeddings": cfg.tie_word_embeddings,
    }

    m.eval()
    fc = FlopCounter(m)

    for L in seq_lens:
        try:
            fc.attach()
            fc.linear_flops.clear()
            fc.conv_flops.clear()
            with torch.no_grad():
                x = torch.randint(0, cfg.vocab_size, (1, L))
                t0 = time.perf_counter()
                out = m(x)
                t1 = time.perf_counter()
            fc.detach()

            linear_total = sum(fc.linear_flops.values())
            conv_total = sum(fc.conv_flops.values())
            attn_est = fc.attention_flops_manual(L, 1, cfg.hidden_size,
                                                 cfg.num_attention_heads // 2,  # HASS uses H/2 for local
                                                 cfg.local_window_size, cfg.num_layers)
            ssm_est = fc.ssm_flops_manual(L, 1, cfg.hidden_size, cfg.ssm_state_dim, cfg.num_layers)
            total_est = linear_total + conv_total + attn_est + ssm_est

            # Framework's own estimate
            cfg_flops = cfg.estimate_flops_per_token() * L  # per-token * seq_len

            info["by_seq_len"][L] = {
                "linear_flops": linear_total,
                "conv_flops": conv_total,
                "attn_flops_est": attn_est,
                "ssm_flops_est": ssm_est,
                "total_flops_est": total_est,
                "flops_per_token_est": total_est // max(1, L),
                "framework_estimate_flops_per_token": cfg.estimate_flops_per_token(),
                "framework_estimate_total": cfg_flops,
                "framework_per_token_x_actual_per_token": cfg.estimate_flops_per_token() / max(1, total_est / L),
                "wall_time_sec": t1 - t0,
                "tokens_per_sec": L / (t1 - t0) if t1 > t0 else 0,
                "by_component": fc.by_component(),
            }
            print(f"  L={L:>4}  total={total_est:>15,}  per_tok={total_est//L:>12,}  cfg/actual={cfg.estimate_flops_per_token()/(total_est/L):.3f}  t={t1-t0:.3f}s")
        except Exception as e:
            info["by_seq_len"][L] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  L={L}: FAILED — {e}")

    del m
    gc.collect()
    return info


def main():
    variants = ["zero_tiny_23k", "zero_1M", "zero_10M", "zero_50M"]
    results = {}
    for v in variants:
        results[v] = measure_variant(v)

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
