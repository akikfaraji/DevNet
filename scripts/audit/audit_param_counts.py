"""
Xorzen Active-Compute Audit — Part 1
=====================================
Independently instantiate every zero variant (test_mode=True to skip disk
sharding for large variants), count the actual parameters by component,
and compare against:
  - config.estimate_parameters()          (the framework's own formula)
  - config.estimate_active_parameters()   (the framework's own formula)
  - model._estimate_active_params()       (the model's own runtime estimator)
  - variant.PARAM_COUNT                   (the hardcoded constant)

Output: /home/z/my-project/download/audit/param_counts.json
"""
from __future__ import annotations
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("XORZENX_VERBOSE", "0")

import torch
import torch.nn as nn

# Silence the xorzen logger
from xorzen.utils.logger import get_logger
_ul = get_logger()
_underlying = getattr(_ul, "logger", None) or _ul
if hasattr(_underlying, "setLevel"):
    import logging
    _underlying.setLevel(logging.WARNING)

import xorzen
from xorzen.config import ConfigFactory, ModelSize

OUT = Path("/home/z/my-project/download/audit/param_counts.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def component_param_breakdown(model: nn.Module) -> Dict[str, int]:
    """Walk the module tree and bucket parameters by component."""
    buckets: Dict[str, int] = {}
    for name, param in model.named_parameters():
        # Choose a top-level bucket from the parameter name prefix.
        if name.startswith("token_embedding") or name.startswith("position_embedding"):
            bucket = "embeddings"
        elif name.startswith("blocks."):
            # bucket by sub-component inside HASSBlock
            parts = name.split(".")
            # parts[0] = "blocks", parts[1] = layer idx, parts[2] = sub
            sub = parts[2]
            if sub == "pathways":
                pathway = parts[3]
                bucket = f"hass_{pathway}"
            elif sub == "pathway_gate":
                bucket = "hass_pathway_gate"
            elif sub == "ffn":
                bucket = "hass_ffn"
            elif sub.startswith("ln"):
                bucket = "hass_layernorm"
            else:
                bucket = f"hass_other:{sub}"
        elif name.startswith("router."):
            bucket = "router"
        elif name.startswith("moe."):
            bucket = "moe_fabric"
        elif name.startswith("merger."):
            bucket = "merger"
        elif name.startswith("cot."):
            bucket = "cot"
        elif name.startswith("final_norm") or name.startswith("norm_f"):
            bucket = "final_norm"
        elif name.startswith("lm_head"):
            bucket = "lm_head"
        else:
            bucket = f"other:{name.split('.')[0]}"
        buckets[bucket] = buckets.get(bucket, 0) + param.numel()
    return buckets


def count_module_params(m: nn.Module, only_trainable: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if (not only_trainable or p.requires_grad))


def instantiate(variant_name: str) -> Tuple[Optional[nn.Module], Dict[str, Any]]:
    """Try to instantiate a variant; return (model, info)."""
    info: Dict[str, Any] = {"variant": variant_name}
    try:
        # Use test_mode=True for all variants to avoid disk-sharding explosions
        # for the big ones. test_mode uses a single dummy expert.
        if variant_name == "zero_tiny_23k":
            m = xorzen.zero_tiny_23k(test_mode=True)
        elif variant_name == "zero_1M":
            m = xorzen.zero_1M(test_mode=True)
        elif variant_name == "zero_10M":
            m = xorzen.zero_10M(test_mode=True)
        elif variant_name == "zero_50M":
            m = xorzen.zero_50M(test_mode=True)
        elif variant_name == "zero_277M":
            m = xorzen.zero_277M(test_mode=True)
        elif variant_name == "zero_500M":
            m = xorzen.zero_500M(test_mode=True)
        elif variant_name == "zero_1_3B":
            m = xorzen.zero_1_3B(test_mode=True)
        elif variant_name == "zero_7B":
            m = xorzen.zero_7B(test_mode=True)
        elif variant_name == "IGRIS_Nano":
            from xorzen.models.igris.variants import IGRIS_Nano
            m = IGRIS_Nano()
        elif variant_name == "IGRIS_Micro":
            from xorzen.models.igris.variants import IGRIS_Micro
            m = IGRIS_Micro()
        else:
            return None, {"variant": variant_name, "error": "unknown"}
        return m, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()[-2000:]
        return None, info


def audit_variant(variant_name: str) -> Dict[str, Any]:
    print(f"\n=== {variant_name} ===")
    m, info = instantiate(variant_name)
    if m is None:
        print(f"  FAILED: {info.get('error')}")
        return info

    # 1. Actual counts
    total = count_module_params(m, only_trainable=False)
    trainable = count_module_params(m, only_trainable=True)
    info["actual_total_params"] = total
    info["actual_trainable_params"] = trainable

    # 2. Component breakdown
    buckets = component_param_breakdown(m)
    info["component_breakdown"] = buckets

    # 3. Config-derived counts
    cfg = m.config
    info["config_estimate_parameters"] = cfg.estimate_parameters()
    info["config_estimate_active_parameters"] = cfg.estimate_active_parameters()
    info["config_target_active_ratio"] = cfg.target_active_ratio
    info["config_hidden_size"] = cfg.hidden_size
    info["config_num_layers"] = cfg.num_layers
    info["config_vocab_size"] = cfg.vocab_size
    info["config_expert_count"] = cfg.expert_count
    info["config_top_k_experts"] = cfg.top_k_experts
    info["config_expert_hidden_multiplier"] = cfg.expert_hidden_multiplier
    info["config_context_length"] = cfg.context_length
    info["config_local_window_size"] = cfg.local_window_size
    info["config_low_rank_dim"] = cfg.low_rank_dim
    info["config_ssm_state_dim"] = cfg.ssm_state_dim
    info["config_cot_dim"] = cfg.cot_dim
    info["config_cot_components"] = cfg.cot_components
    info["config_width_choices"] = list(cfg.width_choices) if hasattr(cfg, "width_choices") else None
    info["config_tie_word_embeddings"] = cfg.tie_word_embeddings

    # 4. Model's runtime estimator (only for zero family)
    if hasattr(m, "_estimate_active_params"):
        try:
            info["model_runtime_active_estimate"] = int(m._estimate_active_params())
        except Exception as e:
            info["model_runtime_active_estimate_error"] = str(e)

    # 5. PARAM_COUNT constant (zero family only)
    if hasattr(m, "PARAM_COUNT"):
        info["variant_PARAM_COUNT"] = int(m.PARAM_COUNT)

    # 6. Active % as logged at init time vs computed
    if total > 0:
        info["active_pct_of_total"] = 100.0 * info["config_estimate_active_parameters"] / total
        info["active_pct_of_trainable"] = 100.0 * info["config_estimate_active_parameters"] / max(1, trainable)

    # 7. Forward pass smoke test (small batch)
    try:
        m.eval()
        with torch.no_grad():
            x = torch.randint(0, cfg.vocab_size, (1, 16))
            out = m(x)
        info["forward_smoke_test"] = "pass"
        if hasattr(out, "logits"):
            info["forward_logits_shape"] = list(out.logits.shape)
        elif isinstance(out, tuple) and out[0] is not None:
            info["forward_logits_shape"] = list(out[0].shape)
    except Exception as e:
        info["forward_smoke_test"] = f"fail: {type(e).__name__}: {e}"
        info["forward_traceback"] = traceback.format_exc()[-1500:]

    # 8. Discrepancy summary
    # Note: in test_mode=True the expert fabric uses a single dummy expert
    # instead of cfg.expert_count real experts.  So `total` excludes the
    # full expert pool.  Compute what the count WOULD be with all experts.
    expert_hidden = int(cfg.hidden_size * cfg.expert_hidden_multiplier)
    # SwiGLU expert has gate, up, down -> 3 projections of H x D_int
    params_per_expert_swiglu = 3 * cfg.hidden_size * expert_hidden
    params_per_expert_swiglu += 3 * expert_hidden  # biases
    full_expert_pool_params = cfg.expert_count * params_per_expert_swiglu
    info["computed_expert_params_per_expert_swiglu"] = params_per_expert_swiglu
    info["computed_full_expert_pool_params_swiglu"] = full_expert_pool_params
    info["computed_total_with_full_experts_swiglu"] = total + full_expert_pool_params
    # Alternative: 2-projection FFN (the formula used by config.estimate_active_parameters)
    params_per_expert_2 = 2 * cfg.hidden_size * expert_hidden
    full_pool_2 = cfg.expert_count * params_per_expert_2
    info["computed_full_expert_pool_params_2proj"] = full_pool_2
    info["computed_total_with_full_experts_2proj"] = total + full_pool_2

    discrepancies = []
    if "variant_PARAM_COUNT" in info:
        # Compare against both test-mode total and full-expert total
        for label, val in [("test_mode_total", total),
                           ("full_expert_swiglu", total + full_expert_pool_params),
                           ("full_expert_2proj", total + full_pool_2)]:
            delta = info["variant_PARAM_COUNT"] - val
            discrepancies.append({
                "claim": f"variant.PARAM_COUNT == {label}",
                "claimed": info["variant_PARAM_COUNT"],
                "actual": val,
                "delta": delta,
                "pct_off": 100.0 * delta / max(1, info["variant_PARAM_COUNT"]),
            })
    if "config_estimate_parameters" in info:
        delta = info["config_estimate_parameters"] - total
        discrepancies.append({
            "claim": "config.estimate_parameters() == test_mode_total",
            "claimed": info["config_estimate_parameters"],
            "actual": total,
            "delta": delta,
            "pct_off": 100.0 * delta / max(1, info["config_estimate_parameters"]),
        })
    info["discrepancies"] = discrepancies

    print(f"  total={total:,}  trainable={trainable:,}")
    pc = info.get('variant_PARAM_COUNT','n/a')
    pc_str = f"{pc:,}" if isinstance(pc,int) else str(pc)
    ce = info.get('config_estimate_parameters','n/a')
    ce_str = f"{ce:,}" if isinstance(ce,int) else str(ce)
    ca = info.get('config_estimate_active_parameters','n/a')
    ca_str = f"{ca:,}" if isinstance(ca,int) else str(ca)
    print(f"  PARAM_COUNT={pc_str}  cfg.estimate={ce_str}")
    print(f"  cfg.active={ca_str}  runtime_active={info.get('model_runtime_active_estimate','n/a')}")
    if discrepancies:
        print(f"  DISCREPANCIES:")
        for d in discrepancies:
            print(f"    - {d['claim']}: delta={d['delta']:,} ({d['pct_off']:+.2f}%)")

    del m
    gc.collect()
    return info


def main():
    # zero_7B takes too long on this CPU-only 2.4GB-disk box; cover it
    # by config-only inspection below.
    variants = [
        "zero_tiny_23k",
        "zero_1M",
        "zero_10M",
        "zero_50M",
        "zero_277M",
        "zero_500M",
        "zero_1_3B",
        # "zero_7B",
        "IGRIS_Nano",
        "IGRIS_Micro",
    ]
    results = {}
    for v in variants:
        try:
            results[v] = audit_variant(v)
        except Exception as e:
            results[v] = {"variant": v, "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-1500:]}

    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT}")
    print(f"\n=== SUMMARY ===")
    for v, r in results.items():
        if "error" in r and "actual_total_params" not in r:
            print(f"  {v}: FAILED — {r['error']}")
        else:
            print(f"  {v}: total={r.get('actual_total_params','?'):>15,}  PARAM_COUNT={r.get('variant_PARAM_COUNT','n/a')}  cfg.estimate={r.get('config_estimate_parameters','n/a'):>15,}  active={r.get('config_estimate_active_parameters','n/a'):>10,}")


if __name__ == "__main__":
    main()
