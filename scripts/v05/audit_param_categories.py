"""
Independent audit of the Xorzen scaling-law table.

For every variant, independently compute from the actual code:
  - P_full     : all declared capacity (E experts materialized + full pathways + full depth)
  - P_resident : parameters physically in state_dict / RAM at init
                 - in test_mode=True   : structural + 1 dummy expert
                 - in production mode  : structural only (experts live on disk,
                                         LRU cache starts empty)
  - P_active   : parameters actually executed per token at sparse inference
                 (pathway_top_k * L_avg layers * K experts * width-avg)

Then compare against the current "P_total (actual)" column in
/home/z/my-project/xorzen_dev/reports/scaling/scaling_law.json
and determine which quantity that column actually reports.

DO NOT change configs, rename labels, or fix anything — this is an audit only.
"""
from __future__ import annotations
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

PROJ = "/home/z/my-project/xorzen_dev"
sys.path.insert(0, PROJ)

from xorzen.config import ConfigFactory, ModelConfig, ModelSize
from xorzen.models.zero.variants import (
    zero_tiny_23k, zero_1M, zero_10M, zero_50M, zero_277M,
    zero_500M, zero_1_3B,
)
from xorzen.model.zmoe import ShardedExpertFabric, ExpertFFN

SCALING_LAW_PATH = os.path.join(PROJ, "reports", "scaling", "scaling_law.json")
OUT_PATH = "/home/z/my-project/scripts/param_category_audit.json"


# -------------------- helpers --------------------

def count_state_dict_params(model: torch.nn.Module) -> int:
    """Total params physically present in state_dict (what .parameters() yields)."""
    return int(sum(p.numel() for p in model.parameters()))


def count_dummy_expert_params(model: torch.nn.Module) -> int:
    """Count params of the dummy_expert submodule (only exists in test_mode=True).

    The zeroModel exposes the ShardedExpertFabric as `model.moe` (not `moe_fabric`).
    In test_mode, `moe.dummy_expert` is a single ExpertFFN submodule.
    """
    moe = getattr(model, "moe", None) or getattr(model, "moe_fabric", None)
    if moe is not None and hasattr(moe, "dummy_expert"):
        return int(sum(p.numel() for p in moe.dummy_expert.parameters()))
    return 0


def count_structural_params_excluding_experts(model: torch.nn.Module) -> int:
    """Structural params = state_dict params MINUS the dummy expert (== production
    mode state_dict, where experts live on disk and LRU cache starts empty)."""
    total = count_state_dict_params(model)
    dummy = count_dummy_expert_params(model)
    return total - dummy


def expert_per_params(H: int, M: float, bias: bool = False) -> int:
    """One ExpertFFN (SwiGLU) parameter count.

    From zmoe.ExpertFFN:
        gate_proj = Linear(H, H*M, bias=bias)  -> H*(H*M) + (H*M if bias else 0)
        up_proj   = Linear(H, H*M, bias=bias)  -> same
        down_proj = Linear(H*M, H, bias=bias)  -> (H*M)*H + (H if bias else 0)
    ExpertFFN.__init__ defaults bias=False.
    """
    intermediate = int(H * M)
    gate = H * intermediate + (intermediate if bias else 0)
    up   = H * intermediate + (intermediate if bias else 0)
    down = intermediate * H + (H if bias else 0)
    return gate + up + down


def compute_P_full(structural_no_experts: int, E: int, expert_per: int) -> int:
    """P_full = structural (with no experts in state_dict) + ALL E experts in RAM."""
    return structural_no_experts + E * expert_per


def compute_P_resident_test_mode(structural_no_experts: int, expert_per: int) -> int:
    """In test_mode=True: structural + 1 dummy expert."""
    return structural_no_experts + 1 * expert_per


def compute_P_resident_production_initial(structural_no_experts: int) -> int:
    """In production mode at init: structural only (LRU cache empty)."""
    return structural_no_experts


def compute_P_resident_production_max_cache(
    structural_no_experts: int, max_cache: int, expert_per: int
) -> int:
    """In production mode at steady-state with full LRU cache: structural + max_cache experts."""
    return structural_no_experts + max_cache * expert_per


def compute_P_active_per_token(arch: Dict[str, Any]) -> int:
    """Re-implement compute_active_per_token from phase8 to independently verify.

    Active routing decisions (defaults):
      - pathway_top_k = 2 of 3 pathways active
      - L_avg = (max_depth + min_depth) / 2 layers active
      - width: average of width_choices / Wmax (genuine slicing)
      - MoE: top-K of E experts active
    """
    H = arch["hidden_size"]
    L = arch["num_layers"]
    V = arch["vocab_size"]
    E = arch["expert_count"]
    K = arch["top_k_experts"]
    M = arch["expert_hidden_multiplier"]
    D_lr = arch["low_rank_dim"]
    D_ssm = arch["ssm_state_dim"]
    P_cot = arch["total_cot_dim"]
    widths = arch["width_choices"]
    Wmax = int(H * M)
    W_avg = sum(widths) / len(widths)
    width_factor = W_avg / Wmax
    L_avg = (arch["max_depth"] + arch["min_depth"]) / 2.0
    k_path = 2  # default pathway_top_k

    head_dim = H // max(1, (H // 64))
    local_per = 4 * (H * H + H) + 2 * head_dim
    Dlr4 = D_lr * 4
    lowrank_per = (H * Dlr4 + Dlr4) + (Dlr4 * H + H) + Dlr4 + (H + H) + (Dlr4 + Dlr4)
    S = D_ssm
    ssm_per = S + (H * S + S) + (H * S + S) + (H * S + S) + (S * H + H) + (H * 3 + H) + (H * 2 * H + 2 * H) + (H * 2) + (S * 2)
    ffn_per_full = (H * Wmax + Wmax) + (Wmax * H + H) + (H * 2) + (Wmax * 2)
    ffn_per_active = int(ffn_per_full * width_factor)
    ln_per = 2 * H * 2

    pathway_total_per_block = local_per + lowrank_per + ssm_per
    pathway_active_per_block = int(pathway_total_per_block * (k_path / 3.0))
    block_active = pathway_active_per_block + ffn_per_active + ln_per
    layer_active = int(L_avg * block_active)

    # Router / CoT / Merger / final_norm — always full (small)
    _h = max(1, H // 4)
    _enc1 = max(128, _h * 4); _enc2 = max(64, _h * 2); _enc3 = max(32, _h); _head = max(32, _h // 2)
    in_dim = H + P_cot
    router_active = (
        (in_dim * _enc1 + _enc1) + (_enc1 * 2) +
        (_enc1 * _enc2 + _enc2) + (_enc2 * 2) +
        (_enc2 * _enc3 + _enc3) + (_enc3 * 2) +
        (_enc3 * _head + _head) + (_head * 2) + (_head * L) +
        (_enc3 * _head + _head) + (_head * 2) + (_head * len(widths)) +
        (_enc3 * _head + _head) + (_head * 2) + (_head * 3) +
        (_enc3 * _enc3 + _enc3) + (_enc3 * 2) + (_enc3 * max(1, E)) +
        (_enc3 * _head + _head) + (_head * 2) + (_head * 1) +
        (_enc3 * _head + _head) + (_head * 2) + (_head * 1)
    )
    D_comp = P_cot // 6
    cot_components = 6 * ((H * 2 * D_comp + 2 * D_comp) + (D_comp * 2))
    cot_gru = 3 * (P_cot * P_cot + P_cot) + 3 * (P_cot * P_cot + P_cot) + 3 * P_cot + 3 * P_cot
    cot_extra = (D_comp * 2) + (P_cot * 2) + (P_cot * H + H) + (H * H + H) + ((H + P_cot) * 128 + 128) + (128 + 1)
    cot_active = cot_components + cot_gru + cot_extra

    expert_per = 3 * H * (H * M)  # no bias
    moe_active = K * expert_per

    MH = H
    merger_input = 2 * H + P_cot
    merger_active = (merger_input * MH + MH) + (MH * 3 + 3) + (P_cot * H + H) + (H * 2)
    final_active = 2 * H
    emb_active = V * H  # always full (lookup)

    return int(emb_active + layer_active + router_active + cot_active + moe_active + merger_active + final_active)


# -------------------- main audit --------------------

def audit_variant(variant_cls, label: str) -> Dict[str, Any]:
    """Audit one variant: independently measure P_full, P_resident, P_active."""
    print(f"\n[{label}] instantiating test_mode=True ...")
    t0 = time.time()
    cfg = ConfigFactory.get_config(variant_cls.MODEL_SIZE)
    try:
        model = variant_cls(config=cfg, test_mode=True)
    except Exception as e:
        return {"label": label, "error": f"{type(e).__name__}: {e}"}

    # ---- independently measured quantities ----
    state_dict_total_test = count_state_dict_params(model)               # what current audit reports
    dummy_expert_params   = count_dummy_expert_params(model)             # 1 ExpertFFN
    structural_no_experts = count_structural_params_excluding_experts(model)  # production state_dict (cache empty)
    expert_per            = expert_per_params(cfg.hidden_size, cfg.expert_hidden_multiplier, bias=False)

    # Verify our formula matches the actual dummy expert
    formula_check_delta = expert_per - dummy_expert_params

    # ---- compute the three category values ----
    P_resident_test_mode     = compute_P_resident_test_mode(structural_no_experts, expert_per)
    P_resident_prod_initial  = compute_P_resident_production_initial(structural_no_experts)
    P_resident_prod_maxcache = compute_P_resident_production_max_cache(
        structural_no_experts, cfg.max_expert_cache, expert_per
    )
    P_full                   = compute_P_full(structural_no_experts, cfg.expert_count, expert_per)

    arch = {
        "hidden_size": cfg.hidden_size, "num_layers": cfg.num_layers, "vocab_size": cfg.vocab_size,
        "expert_count": cfg.expert_count, "top_k_experts": cfg.top_k_experts,
        "expert_hidden_multiplier": cfg.expert_hidden_multiplier,
        "low_rank_dim": cfg.low_rank_dim, "ssm_state_dim": cfg.ssm_state_dim,
        "total_cot_dim": cfg.cot_dim * cfg.cot_components,
        "width_choices": list(cfg.width_choices),
        "max_depth": cfg.max_depth, "min_depth": cfg.min_depth,
        "target_active_ratio": cfg.target_active_ratio,
        "max_expert_cache": cfg.max_expert_cache,
        "pathway_top_k": cfg.pathway_top_k,
    }
    P_active = compute_P_active_per_token(arch)

    # cross-check: state_dict_total_test should equal P_resident_test_mode
    state_dict_check_delta = state_dict_total_test - P_resident_test_mode

    rec = {
        "label": label,
        "PARAM_COUNT_label": int(getattr(variant_cls, "PARAM_COUNT", -1)),
        "arch": arch,
        "independent_measurements": {
            "state_dict_total_test_mode": state_dict_total_test,
            "dummy_expert_params_actual": dummy_expert_params,
            "dummy_expert_params_formula": expert_per,
            "formula_check_delta": formula_check_delta,  # should be 0
            "structural_no_experts": structural_no_experts,
            "expert_per_params": expert_per,
            "max_expert_cache_config": cfg.max_expert_cache,
        },
        "computed_categories": {
            "P_full": P_full,
            "P_resident_test_mode": P_resident_test_mode,
            "P_resident_production_initial": P_resident_prod_initial,
            "P_resident_production_max_cache": P_resident_prod_maxcache,
            "P_active_per_token": P_active,
        },
        "state_dict_vs_formula_check_delta": state_dict_check_delta,  # should be 0
        "instantiate_time_sec": round(time.time() - t0, 3),
    }
    print(f"  state_dict_total_test = {state_dict_total_test:,}")
    print(f"  dummy_expert (actual) = {dummy_expert_params:,}  (formula={expert_per:,}, delta={formula_check_delta})")
    print(f"  structural_no_experts = {structural_no_experts:,}")
    print(f"  P_full                = {P_full:,}  (= structural + {cfg.expert_count} experts)")
    print(f"  P_resident_test_mode  = {P_resident_test_mode:,}")
    print(f"  P_resident_prod_init  = {P_resident_prod_initial:,}")
    print(f"  P_resident_prod_max   = {P_resident_prod_maxcache:,}  (cache={cfg.max_expert_cache})")
    print(f"  P_active_per_token    = {P_active:,}")

    del model, cfg
    gc.collect()
    return rec


def main():
    print("=" * 78)
    print("  Xorzen Parameter-Category Audit")
    print("  Goal: independently compute P_full / P_resident / P_active per variant")
    print("=" * 78)

    variants = [
        (zero_tiny_23k, "zero_tiny_23k"),
        (zero_1M,       "zero_1M"),
        (zero_10M,      "zero_10M"),
        (zero_50M,      "zero_50M"),
        (zero_277M,     "zero_277M"),
        (zero_500M,     "zero_500M"),
        (zero_1_3B,     "zero_1_3B"),
    ]

    records = [audit_variant(v, lbl) for v, lbl in variants]

    # Load current scaling-law table for comparison
    with open(SCALING_LAW_PATH) as f:
        sl = json.load(f)
    current_table = {r["variant"]: r for r in sl["scaling_law_table"]}

    # Build final comparison table
    comparison = []
    for rec in records:
        if "error" in rec:
            comparison.append({"label": rec["label"], "error": rec["error"]})
            continue
        lbl = rec["label"]
        cur = current_table.get(lbl, {})
        comp = rec["computed_categories"]
        cur_P_actual = cur.get("P_total_actual")
        # Determine which category the current "P_total (actual)" matches
        match = []
        for cat_name, cat_val in comp.items():
            if cur_P_actual is not None and abs(cat_val - cur_P_actual) <= max(1, 0.005 * cur_P_actual):
                match.append(cat_name)
        comparison.append({
            "label": lbl,
            "PARAM_COUNT_label": rec["PARAM_COUNT_label"],
            "P_full": comp["P_full"],
            "P_resident_test_mode": comp["P_resident_test_mode"],
            "P_resident_production_initial": comp["P_resident_production_initial"],
            "P_resident_production_max_cache": comp["P_resident_production_max_cache"],
            "P_active_per_token": comp["P_active_per_token"],
            "current_P_actual_in_scaling_law_table": cur_P_actual,
            "category_matched_by_current_P_actual": match,
        })

    # Print final clear table
    print("\n" + "=" * 78)
    print("  FINAL AUDIT TABLE — independent measurements")
    print("=" * 78)
    print(f"{'label':<18}{'P_full':>16}{'P_res_test':>16}{'P_res_prod_init':>18}{'P_res_prod_max':>17}{'P_active/tok':>16}")
    for c in comparison:
        if "error" in c:
            print(f"{c['label']:<18}  ERROR: {c['error']}")
            continue
        print(f"{c['label']:<18}{c['P_full']:>16,}{c['P_resident_test_mode']:>16,}"
              f"{c['P_resident_production_initial']:>18,}{c['P_resident_production_max_cache']:>17,}"
              f"{c['P_active_per_token']:>16,}")

    print("\n" + "=" * 78)
    print("  COMPARISON: which category does the current 'P_total (actual)' match?")
    print("=" * 78)
    print(f"{'label':<18}{'current_P_actual':>18}  | matched category")
    print("-" * 78)
    for c in comparison:
        if "error" in c:
            continue
        matched = c["category_matched_by_current_P_actual"]
        if not matched:
            matched_str = "??? (no exact match — see deltas below)"
        else:
            matched_str = " | ".join(matched)
        print(f"{c['label']:<18}{c['current_P_actual_in_scaling_law_table']:>18,}  | {matched_str}")

    # Also print deltas to identify which category is closest when no exact match
    print("\n  Deltas (current_P_actual minus each category):")
    print(f"{'label':<18}{'Δ P_full':>14}{'Δ P_res_test':>16}{'Δ P_res_prod_init':>22}{'Δ P_res_prod_max':>20}{'Δ P_active':>14}")
    for c in comparison:
        if "error" in c:
            continue
        cur = c["current_P_actual_in_scaling_law_table"]
        d_full = cur - c["P_full"]
        d_test = cur - c["P_resident_test_mode"]
        d_prod = cur - c["P_resident_production_initial"]
        d_max  = cur - c["P_resident_production_max_cache"]
        d_act  = cur - c["P_active_per_token"]
        print(f"{c['label']:<18}{d_full:>+14,}{d_test:>+16,}{d_prod:>+22,}{d_max:>+20,}{d_act:>+14,}")

    out = {
        "audit_description": "Independent computation of P_full / P_resident / P_active per variant. No configs changed.",
        "records": records,
        "comparison_table": comparison,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved audit JSON: {OUT_PATH}")


if __name__ == "__main__":
    main()
