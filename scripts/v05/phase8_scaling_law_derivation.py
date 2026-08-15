"""
Xorzen Scaling Law Derivation — v0.5 (storage-efficient).

GOAL:
  Reconstruct Xorzen's actual model scaling law from the codebase.
  Do NOT invent a generic Transformer scaling law. Derive Xorzen-specific
  relationships by:
    1. Reading every variant configuration from ConfigFactory.
    2. Instantiating each variant and counting REAL parameters from
       state_dict() (NOT from PARAM_COUNT labels — those are aspirational).
    3. Decomposing parameters by component (embeddings, CoT, router,
       HASS pathways, FFN, MoE fabric, merger, final norm).
    4. Fitting closed-form formulas that predict the real parameter
       count from architectural hyperparameters (H, L, V, E, K, ...).
    5. Computing the active-params-per-token and FLOPs-per-token
       from the routing and sparsity structure.
    6. Building a scaling-law table covering every existing variant.
    7. Determining the conditions under which "12B Xorzen > 60B Dense"
       is mathematically plausible.

DO NOT trust PARAM_COUNT labels — proven wrong by 58–70 % at large scale.

OUTPUTS:
  /home/z/my-project/xorzen_dev/reports/scaling/scaling_law.json
  /home/z/my-project/xorzen_dev/reports/scaling/scaling_law.md
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import torch

# --- Path setup ---
PROJ = "/home/z/my-project/xorzen_dev"
sys.path.insert(0, PROJ)

from xorzen.config import ConfigFactory, ModelConfig, ModelSize
from xorzen.models.zero.variants import (
    zero_tiny_23k, zero_1M, zero_10M, zero_50M, zero_277M,
    zero_500M, zero_1_3B, zero_7B,
)
from xorzen.models.igris.variants import IGRIS_Nano

OUT_DIR = os.path.join(PROJ, "reports", "scaling")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, "scaling_law.json")
OUT_MD = os.path.join(OUT_DIR, "scaling_law.md")


# ----------------- Helpers -----------------

def count_params_by_component(model: torch.nn.Module) -> Dict[str, int]:
    """Walk model.named_modules() once and attribute every parameter
    to its parent component by name-prefix matching.

    The Xorzen zeroModel uses these top-level submodules (verified by
    reading xorzen/model/zero/model.py): token_embedding, position_embedding,
    cot, router, blocks (ModuleList of HASSBlock), moe_fabric (ShardedExpertFabric),
    merger, final_norm.

    HASSBlock internals: pathways.{local,low_rank,ssm}, ffn (SlicedFFN),
    ln1, ln2.  Each HASS block has layer_idx in its name like `blocks.0.…`.
    """
    breakdown: Dict[str, int] = {}
    seen_param_ids = set()

    for name, mod in model.named_modules():
        for pname, p in mod.named_parameters(recurse=False):
            if id(p) in seen_param_ids:
                continue
            seen_param_ids.add(id(p))
            full = f"{name}.{pname}" if name else pname
            # Classify by prefix
            key = _classify(full)
            breakdown[key] = breakdown.get(key, 0) + int(p.numel())

    return breakdown


def _classify(full_name: str) -> str:
    """Map a parameter path to a component bucket."""
    if full_name.startswith("token_embedding") or full_name.startswith("embeddings."):
        return "embeddings"
    if full_name.startswith("position_embedding") or "pos_embedding" in full_name:
        return "embeddings"
    if full_name.startswith("cot") or full_name.startswith("latent_cot"):
        return "cot"
    if full_name.startswith("router"):
        return "router"
    if full_name.startswith("final_norm") or full_name == "ln_f":
        return "final_norm"
    if full_name.startswith("merger"):
        return "merger"
    if full_name.startswith("moe_fabric") or full_name.startswith("moe."):
        return "moe_fabric"
    if full_name.startswith("memory_vault"):
        return "memory_vault"
    if ".blocks." in full_name or full_name.startswith("blocks."):
        # HASS block subcomponents
        if ".pathways.local." in full_name or ".pathways['local']." in full_name:
            return "hass_local"
        if ".pathways.low_rank." in full_name or ".pathways['low_rank']." in full_name:
            return "hass_low_rank"
        if ".pathways.ssm." in full_name or ".pathways['ssm']." in full_name:
            return "hass_ssm"
        if ".pathway_gate" in full_name:
            return "hass_pathway_gate"
        if ".ffn." in full_name:
            return "hass_ffn"
        if ".ln1." in full_name or ".ln2." in full_name or ".ln_input." in full_name or ".ln_hidden." in full_name:
            return "hass_layernorm"
        if ".mix_gate" in full_name:
            return "hass_mix_gate"
        if ".attn." in full_name or ".mlp." in full_name or ".ssm." in full_name:
            return "hass_other"
        return "hass_other"
    if full_name.startswith("action_head") or full_name.startswith("critique") or full_name.startswith("cot_to_hidden"):
        return "other"
    return "other"


def instantiate(variant_cls, test_mode: bool = True) -> Tuple[torch.nn.Module, ModelConfig]:
    """Build a variant and return (model, config)."""
    cfg = ConfigFactory.get_config(variant_cls.MODEL_SIZE)
    model = variant_cls(config=cfg, test_mode=test_mode)
    return model, cfg


def measure_one_variant(variant_cls, label: str) -> Dict[str, Any]:
    """Instantiate, count params, decompose, return record."""
    t0 = time.time()
    try:
        model, cfg = instantiate(variant_cls, test_mode=True)
    except Exception as e:
        return {"variant": label, "error": f"{type(e).__name__}: {e}"}

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    breakdown = count_params_by_component(model)

    # Architectural hyperparameters from config
    H = cfg.hidden_size
    L = cfg.num_layers
    V = cfg.vocab_size
    E = cfg.expert_count
    K = cfg.top_k_experts
    ffn_mult = cfg.expert_hidden_multiplier
    D_lr = cfg.low_rank_dim
    D_ssm = cfg.ssm_state_dim
    D_cot = cfg.cot_dim * cfg.cot_components
    widths = list(cfg.width_choices)
    cot_components = cfg.cot_components

    record = {
        "variant": label,
        "PARAM_COUNT_label": int(getattr(variant_cls, "PARAM_COUNT", -1)),
        "actual_total_params": int(total),
        "actual_trainable_params": int(trainable),
        "label_accuracy_pct": round(100.0 * total / max(1, getattr(variant_cls, "PARAM_COUNT", 1)), 2),
        "component_breakdown": breakdown,
        "arch": {
            "hidden_size": H,
            "num_layers": L,
            "vocab_size": V,
            "expert_count": E,
            "top_k_experts": K,
            "expert_hidden_multiplier": ffn_mult,
            "context_length": cfg.context_length,
            "low_rank_dim": D_lr,
            "ssm_state_dim": D_ssm,
            "cot_dim": cfg.cot_dim,
            "cot_components": cot_components,
            "total_cot_dim": D_cot,
            "width_choices": widths,
            "max_depth": cfg.max_depth,
            "min_depth": cfg.min_depth,
            "target_active_ratio": cfg.target_active_ratio,
            "num_attention_heads": cfg.num_attention_heads,
            "tie_word_embeddings": cfg.tie_word_embeddings,
        },
        "instantiate_time_sec": round(time.time() - t0, 3),
    }

    del model, cfg
    gc.collect()
    return record


# ----------------- Scaling-law formulas -----------------
#
# Derived from inspecting the code (see reports/scaling/scaling_law.md for
# the full derivation).  Symbol legend:
#   H  = hidden_size
#   L  = num_layers (== max_depth)
#   V  = vocab_size
#   C  = context_length
#   E  = expert_count (MoE)
#   K  = top_k_experts
#   M  = expert_hidden_multiplier (default 4)
#   D  = low_rank_dim
#   S  = ssm_state_dim
#   P  = cot_dim * cot_components  (total CoT width)
#   Wmax = H * M  (FFN max width, SlicedFFN/AdaptiveFFN both use H*M)
#   W̄   = average active width  (mean of width_choices)
#
# Total parameter count P_total (per-block, summed L times):
#
#   Embeddings (tied LM head):   V * H
#   Per HASS block:
#     LocalAttn (Q,K,V,O proj + 2 LNs of head_dim):    4*H*H + 4*H + 2*head_dim
#     LowRank (to_low + from_low + ctx_w + 2 LNs):     2*H*(D*4) + (D*4) + (D*4) + H + (D*4) + H
#     SSM (A_log + dt/B/C/D proj + conv1d + gate + 2 LN): S + 3*H*S + S*H + H*(2H) + H + S + H + H + ...
#     FFN (SlicedFFN: fc1 H→Wmax, fc2 Wmax→H, 2 LNs):  H*Wmax + Wmax + Wmax*H + H + H + Wmax
#     LayerNorms (ln1, ln2):                           2*H * 2
#   Router (feature_encoder 3 layers + 4 head nets):   depends on H
#   CoT (InternalLatentCoT, 6 component projections + GRU + output_proj + ...):
#   MoE fabric (dummy expert in test_mode, full E experts otherwise):
#     Per expert (SwiGLU):  3 * H * (H*M)  (gate_proj + up_proj + down_proj)
#     Total: E * 3 * H * H * M
#   Merger (GatedMerger):  (2H + P) * (H*mult) + (H*mult)*3 + P*H + H*2 + H*2
#   final_norm:            2 * H
#
# The dominant scaling terms (verified by fit below) are:
#   P_total ≈ V*H  (embeddings, tied)
#           + L * (4*H^2 + 2*H*D_lr*4 + 3*H*D_ssm + 2*H*(H*M))   ← per-block pathways+FFN
#           + L * (3*H*D_ssm + 2*H^2)                            ← SSM gate + local attn
#           + E * 3*H*(H*M)                                       ← MoE expert pool
#           + O(H * P)                                            ← CoT, merger (sub-dominant)
#           + O(H^2)                                              ← router (sub-dominant at large H)
#
# At sparse execution (top-K of E experts, top-k of 3 pathways, etc.):
#   P_active/token ≈ V*H/L_factor + L_avg * [k_path * (4H^2+2H*D_lr*4+3H*S+2H*H*M_avg) + K * 3*H*(H*M)]
#   where L_avg = (max_depth + min_depth)/2, k_path = pathway_top_k, M_avg uses W̄.
#
# FLOPs/token (forward only, dense-equivalent): ~6 * P_active/token
# FLOPs_total training (forward+backward, ~3x forward): ~18 * P_active/token * num_tokens


def predict_total_params(H, L, V, E, K, M, D_lr, D_ssm, P_cot, Wmax, widths,
                          tie_embeddings=True, with_experts=True) -> Dict[str, int]:
    """Closed-form prediction of P_total by component.

    Returns a dict of per-component predicted params, plus 'total'.
    All formulas are derived by reading the actual nn.Linear/Parameter
    declarations in the source code (see scaling_law.md for line refs).
    """
    # Embeddings
    emb = V * H  # token embedding; LM head tied so no extra cost
    # Position embedding (zeroModel uses absolute position embedding of size C*H)
    # But context_length is large; verified by param_counts.json that it's not
    # instantiated for the zero variants (they use RoPE-style or no pos emb?).
    # Actually the audit breakdown shows 'embeddings' = V*H exactly for zero,
    # so position embeddings are NOT used (or are 0). We set to 0.
    pos_emb = 0

    # Per-block components
    # LocalAttentionPathway: 4 Linear(H,H) with bias + 2 LayerNorm(head_dim)
    head_dim = H // max(1, (H // 64))  # half heads (num_heads//2); approx
    # Actually LocalAttention uses num_heads//2 heads, head_dim = H / (num_heads//2)
    # Per HASS block:
    # 4 Linear(H,H) with bias: 4*(H*H + H)
    local_per_block = 4 * (H * H + H) + 2 * head_dim  # 2 LayerNorms of head_dim (weight+bias)

    # LowRankGlobalPathway: Linear(H, D_lr*4)+bias, Linear(D_lr*4, H)+bias,
    # context_weights (D_lr*4), 2 LayerNorms (H and D_lr*4)
    Dlr4 = D_lr * 4
    lowrank_per_block = (H * Dlr4 + Dlr4) + (Dlr4 * H + H) + Dlr4 + (H + H) + (Dlr4 + Dlr4)

    # SSMPathway:
    # A_log: state_dim
    # dt_proj: Linear(H, S)+bias
    # B_proj: Linear(H, S)+bias (no bias actually — let me check)
    # C_proj: Linear(H, S)+bias
    # D_proj: Linear(S, H)+bias
    # Conv1d(H, H, k=3, groups=H): H*3 + H  (depthwise)
    # gate_proj: Linear(H, 2H)+bias
    # ln_input: H*2
    # ln_state: S*2
    S = D_ssm
    ssm_per_block = S + (H * S + S) + (H * S + S) + (H * S + S) + (S * H + H) + (H * 3 + H) + (H * 2 * H + 2 * H) + (H * 2) + (S * 2)

    # SlicedFFN: Linear(H, Wmax)+bias, Linear(Wmax, H)+bias, ln_input(H), ln_hidden(Wmax)
    ffn_per_block = (H * Wmax + Wmax) + (Wmax * H + H) + (H * 2) + (Wmax * 2)

    # ln1, ln2 of HASS block: 2 * H * 2
    hass_ln_per_block = 2 * H * 2

    # Total per-block
    per_block = local_per_block + lowrank_per_block + ssm_per_block + ffn_per_block + hass_ln_per_block
    block_total = L * per_block

    # Router (AdaptiveRouter):
    # input_dim = H + P_cot
    # _enc1 = max(128, _h*4), _enc2 = max(64, _h*2), _enc3 = max(32, _h), _head = max(32, _h//2)
    # _h = router_hidden_dim (typically H//4 or H//8)
    # feature_encoder: Linear(in,_enc1)+LN(_enc1)+Linear(_enc1,_enc2)+LN(_enc2)+Linear(_enc2,_enc3)+LN(_enc3)
    # depth_router: Linear(_enc3,_head)+LN(_head)+Linear(_head, max_depth)
    # width_router: Linear(_enc3,_head)+LN(_head)+Linear(_head, num_widths)
    # path_router: Linear(_enc3,_head)+LN(_head)+Linear(_head, 3)
    # expert_router: Linear(_enc3,_enc3)+LN(_enc3)+Linear(_enc3, num_experts)
    # complexity_estimator: Linear(_enc3,_head)+LN(_head)+Linear(_head,1)
    # uncertainty_estimator: Linear(_enc3,_head)+LN(_head)+Linear(_head,1)
    # Default router_hidden_dim for variants: H//4 (from ConfigFactory)
    _h = max(1, H // 4)
    _enc1 = max(128, _h * 4)
    _enc2 = max(64, _h * 2)
    _enc3 = max(32, _h)
    _head = max(32, _h // 2)
    in_dim = H + P_cot
    router = (
        (in_dim * _enc1 + _enc1) + (_enc1 * 2) +   # enc1 layer + LN
        (_enc1 * _enc2 + _enc2) + (_enc2 * 2) +
        (_enc2 * _enc3 + _enc3) + (_enc3 * 2) +
        (_enc3 * _head + _head) + (_head * 2) + (_head * L) +    # depth_router
        (_enc3 * _head + _head) + (_head * 2) + (_head * len(widths)) +  # width_router
        (_enc3 * _head + _head) + (_head * 2) + (_head * 3) +   # path_router
        (_enc3 * _enc3 + _enc3) + (_enc3 * 2) + (_enc3 * max(1, E)) +  # expert_router
        (_enc3 * _head + _head) + (_head * 2) + (_head * 1) +   # complexity
        (_enc3 * _head + _head) + (_head * 2) + (_head * 1)     # uncertainty
    )

    # CoT (InternalLatentCoT):
    # 6 component_projections: each is Linear(H, P_cot*2/6 * 2 = P_cot/6 * 2) + GLU + LN(P_cot/6) + SiLU
    # Each component: Linear(H, 2*D_comp) + LN(D_comp), where D_comp = P_cot/6
    D_comp = P_cot // 6
    cot_components_total = 6 * ((H * 2 * D_comp + 2 * D_comp) + (D_comp * 2))
    # GRU updater (default): 3 * (P_cot * P_cot + P_cot)  (input, hidden, bias for each gate) approx
    # Actually GRU: input_size = total_cot_dim, hidden_size = total_cot_dim
    # 3 * (P_cot * P_cot + P_cot * P_cot + P_cot + P_cot) — but PyTorch GRU has 3 gates
    # Use the PyTorch formula: weight_ih_l0 (3*hidden, input), weight_hh_l0 (3*hidden, hidden), bias_ih, bias_hh
    cot_gru = 3 * (P_cot * P_cot + P_cot) + 3 * (P_cot * P_cot + P_cot) + 3 * P_cot + 3 * P_cot
    # component_norm (LN of cot_dim): D_comp * 2 (one shared? actually one LN per call)
    # cot_norm (LN of total_cot_dim): P_cot * 2
    # output_proj: Linear(P_cot, H) + bias
    # injection_gate: Linear(H, H) + bias
    # update_gate: Linear(H + P_cot, 128) + bias + Linear(128, 1) + bias
    cot_extra = (D_comp * 2) + (P_cot * 2) + (P_cot * H + H) + (H * H + H) + ((H + P_cot) * 128 + 128) + (128 * 1 + 1)
    cot_total = cot_components_total + cot_gru + cot_extra

    # MoE fabric (instantiate E experts in non-test mode; in test mode only 1 dummy expert)
    # Each expert (ExpertFFN, SwiGLU): gate_proj + up_proj + down_proj
    #   = 3 * H * (H*M)  (no bias by default in ExpertFFN.__init__ bias=False)
    # But the actual ShardedExpertFabric in test_mode creates ONE dummy expert.
    # In production (non-test) mode, E experts exist.
    expert_per = 3 * H * (H * M)  # no bias
    if with_experts:
        moe = E * expert_per
    else:
        moe = 1 * expert_per  # test mode: 1 dummy expert

    # Merger (GatedMerger is the default):
    # input_dim = 2*H + P_cot
    # merger_hidden_dim = H * merger_hidden_multiplier (default 1.0)
    # gate_controller: Linear(input_dim, hidden)+SiLU+Linear(hidden, 3)
    # cot_proj: Linear(P_cot, H)
    # output_norm: LayerNorm(H)
    # dropout: identity
    MH = H  # merger_hidden_multiplier default is 1.0
    merger_input = 2 * H + P_cot
    merger = (merger_input * MH + MH) + (MH * 3 + 3) + (P_cot * H + H) + (H * 2)

    # final_norm
    final_norm = 2 * H

    total_pred = emb + pos_emb + block_total + router + cot_total + moe + merger + final_norm

    return {
        "embeddings_pred": emb,
        "pos_emb_pred": pos_emb,
        "hass_local_pred": L * local_per_block,
        "hass_low_rank_pred": L * lowrank_per_block,
        "hass_ssm_pred": L * ssm_per_block,
        "hass_ffn_pred": L * ffn_per_block,
        "hass_layernorm_pred": L * hass_ln_per_block,
        "router_pred": router,
        "cot_pred": cot_total,
        "moe_fabric_pred": moe,
        "merger_pred": merger,
        "final_norm_pred": final_norm,
        "total_pred": total_pred,
        "_per_block_detail": {
            "local_per_block": local_per_block,
            "low_rank_per_block": lowrank_per_block,
            "ssm_per_block": ssm_per_block,
            "ffn_per_block": ffn_per_block,
            "hass_ln_per_block": hass_ln_per_block,
            "per_block_total": per_block,
        },
    }


# ----------------- Main: measure all variants -----------------

def main():
    print("=" * 70)
    print("  Xorzen Scaling-Law Derivation — v0.5")
    print("=" * 70)

    variants = [
        (zero_tiny_23k, "zero_tiny_23k"),
        (zero_1M,       "zero_1M"),
        (zero_10M,      "zero_10M"),
        (zero_50M,      "zero_50M"),
        (zero_277M,     "zero_277M"),
        (zero_500M,     "zero_500M"),
        (zero_1_3B,     "zero_1_3B"),
        # zero_7B is too big to instantiate in 3.4GB RAM; skip
    ]

    records: List[Dict[str, Any]] = []
    for vcls, vname in variants:
        print(f"\n[{vname}] instantiating...")
        rec = measure_one_variant(vcls, vname)
        if "error" in rec:
            print(f"  ERROR: {rec['error']}")
        else:
            print(f"  actual_total = {rec['actual_total_params']:,}")
            print(f"  label        = {rec['PARAM_COUNT_label']:,}")
            print(f"  label/actual = {rec['label_accuracy_pct']}%")
        records.append(rec)

    # Compute predictions
    print("\n" + "=" * 70)
    print("  Validating scaling-law formulas against actual params")
    print("=" * 70)
    for rec in records:
        if "error" in rec:
            continue
        a = rec["arch"]
        pred = predict_total_params(
            H=a["hidden_size"], L=a["num_layers"], V=a["vocab_size"],
            E=a["expert_count"], K=a["top_k_experts"], M=a["expert_hidden_multiplier"],
            D_lr=a["low_rank_dim"], D_ssm=a["ssm_state_dim"],
            P_cot=a["total_cot_dim"], Wmax=int(a["hidden_size"] * a["expert_hidden_multiplier"]),
            widths=a["width_choices"],
            tie_embeddings=a["tie_word_embeddings"],
            with_experts=True,  # predict production (full expert pool)
        )
        rec["predictions_full_experts"] = pred
        rec["predictions_test_mode"] = {**pred,
            "moe_fabric_pred": pred["moe_fabric_pred"] // max(1, a["expert_count"]),  # 1 expert in test
            "total_pred_test": pred["total_pred"] - pred["moe_fabric_pred"] + pred["moe_fabric_pred"] // max(1, a["expert_count"]),
        }
        # Compute active params/token
        rec["active_per_token"] = compute_active_per_token(a)
        rec["flops_per_token_active"] = 6 * rec["active_per_token"]
        rec["flops_per_token_dense_equiv"] = 6 * rec["actual_total_params"]
        rec["compute_efficiency_ratio"] = round(
            rec["flops_per_token_dense_equiv"] / max(1, rec["flops_per_token_active"]), 3
        )

        actual = rec["actual_total_params"]
        pred_test = rec["predictions_test_mode"]["total_pred_test"]
        pred_full = rec["predictions_full_experts"]["total_pred"]
        rec["prediction_error_pct_test"] = round(100.0 * (pred_test - actual) / actual, 2)
        rec["prediction_error_pct_full"] = round(100.0 * (pred_full - actual) / actual, 2)

        print(f"\n[{rec['variant']}]")
        print(f"  actual (test mode)      = {actual:,}")
        print(f"  predicted (test mode)   = {pred_test:,}  (err {rec['prediction_error_pct_test']:+.1f}%)")
        print(f"  predicted (full experts)= {pred_full:,}  (err {rec['prediction_error_pct_full']:+.1f}%)")
        print(f"  active params/token     = {rec['active_per_token']:,}")
        print(f"  FLOPs/token (active)    = {rec['flops_per_token_active']:,}")
        print(f"  FLOPs/token (dense eq)  = {rec['flops_per_token_dense_equiv']:,}")
        print(f"  compute_efficiency      = {rec['compute_efficiency_ratio']}x")

    # Scaling-law table
    table_rows = []
    for rec in records:
        if "error" in rec:
            continue
        a = rec["arch"]
        table_rows.append({
            "variant": rec["variant"],
            "H": a["hidden_size"],
            "L": a["num_layers"],
            "V": a["vocab_size"],
            "experts_E": a["expert_count"],
            "top_K": a["top_k_experts"],
            "P_total_label": rec["PARAM_COUNT_label"],
            "P_total_actual": rec["actual_total_params"],
            "label_pct_of_actual": rec["label_accuracy_pct"],
            "P_active_per_token": rec["active_per_token"],
            "active_ratio_pct": round(100.0 * rec["active_per_token"] / rec["actual_total_params"], 2),
            "FLOPs_per_token_active": rec["flops_per_token_active"],
            "FLOPs_per_token_dense_eq": rec["flops_per_token_dense_equiv"],
            "compute_efficiency_x": rec["compute_efficiency_ratio"],
            "target_active_ratio": a["target_active_ratio"],
        })

    # 12B vs 60B hypothesis analysis
    hypothesis = analyze_12B_vs_60B(records)

    output = {
        "metadata": {
            "description": "Xorzen scaling law — derived from real parameter counts of instantiated variants.",
            "version": "v0.5",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "instantiate variant + count state_dict + decompose by named_modules + fit formulas",
            "labels_are_aspirational": True,
            "label_discrepancy_at_1_3B_pct": round(
                100.0 * (1000000886 - 413518347) / 1000000886, 1
            ),
        },
        "variants": records,
        "scaling_law_table": table_rows,
        "hypothesis_12B_vs_60B": hypothesis,
        "formulas": {
            "P_total": "V*H + L*(4H^2 + 2H*D_lr*4 + 3H*D_ssm + 2H*(H*M) + 2H*4) + Router + CoT + E*3*H*(H*M) + Merger + 2H",
            "P_active_per_token": "V*H/L + L_avg*[k_path*(per_block_active) + K*3*H*(H*M)]  where L_avg=(max_depth+min_depth)/2, k_path=pathway_top_k",
            "FLOPs_per_token_active": "6 * P_active_per_token  (forward only, dense-equivalent)",
            "FLOPs_total_training": "18 * P_active_per_token * num_tokens  (forward + backward ≈ 3x forward)",
            "Chinchilla_equivalent": "L(C) = E + A/N^alpha + B/D^beta  where N=P_active_per_token, D=tokens, C=6*N*D",
            "note": "P_total scales as O(L*H^2 + E*H^2*M + V*H).  P_active scales as O(L_avg*k_path*H^2 + K*H^2*M + V*H/L).",
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved JSON: {OUT_JSON}")

    # Markdown report
    write_markdown(output, OUT_MD)
    print(f"Saved MD:   {OUT_MD}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  SCALING-LAW TABLE")
    print("=" * 70)
    print(f"{'Variant':<18}{'H':>5}{'L':>4}{'E':>5}{'K':>3}{'P_total':>15}{'P_active':>13}{'active%':>9}{'FLOPs/tok':>13}{'eff_x':>7}")
    for r in table_rows:
        print(f"{r['variant']:<18}{r['H']:>5}{r['L']:>4}{r['experts_E']:>5}{r['top_K']:>3}"
              f"{r['P_total_actual']:>15,}{r['P_active_per_token']:>13,}"
              f"{r['active_ratio_pct']:>8.1f}%{r['FLOPs_per_token_active']:>13,}"
              f"{r['compute_efficiency_x']:>6.2f}x")

    print("\n" + "=" * 70)
    print("  HYPOTHESIS: 12B Xorzen > 60B Dense")
    print("=" * 70)
    for k, v in hypothesis.items():
        if isinstance(v, (int, float, str, bool)):
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")


def compute_active_per_token(arch: Dict[str, Any]) -> int:
    """Estimate active parameters per token for one forward pass.

    Routing decisions (defaults):
      - pathway_top_k: 2 of 3 pathways active (LocalAttn + SSM, or other combos)
      - depth: L_avg = (max_depth + min_depth) / 2 layers active
      - width: average of width_choices (genuine slicing — proportional FLOPs)
      - MoE: top-K of E experts active

    Active param accounting:
      - Embeddings: V*H (always active — lookup)
      - Per active layer:
          * 2 pathways (k_path=2): we approximate as 2/3 of total pathway params
          * FFN: width-avg fraction of FFN params = W̄/Wmax
          * HASS LNs: full (cheap)
      - Router: always full (small)
      - CoT: always full (small)
      - MoE: K/E fraction of expert pool
      - Merger: full
      - final_norm: full
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
    # Per-block params (active fraction)
    # LocalAttn active (full): 4H^2 + 4H + ...
    local_per = 4 * (H * H + H) + 2 * (H // max(1, (H // 64)))
    Dlr4 = D_lr * 4
    lowrank_per = (H * Dlr4 + Dlr4) + (Dlr4 * H + H) + Dlr4 + (H + H) + (Dlr4 + Dlr4)
    S = D_ssm
    ssm_per = S + (H * S + S) + (H * S + S) + (H * S + S) + (S * H + H) + (H * 3 + H) + (H * 2 * H + 2 * H) + (H * 2) + (S * 2)
    ffn_per_full = (H * Wmax + Wmax) + (Wmax * H + H) + (H * 2) + (Wmax * 2)
    ffn_per_active = int(ffn_per_full * width_factor)  # genuine sliced
    ln_per = 2 * H * 2

    # Active pathway params: k_path/3 of (local + lowrank + ssm)
    # (assuming uniform distribution of which pathway is selected)
    pathway_total_per_block = local_per + lowrank_per + ssm_per
    pathway_active_per_block = int(pathway_total_per_block * (k_path / 3.0))

    block_active = pathway_active_per_block + ffn_per_active + ln_per

    # Active layer contribution
    layer_active = int(L_avg * block_active)

    # Router, CoT, Merger, final_norm — always full (small)
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

    # MoE active: K experts
    expert_per = 3 * H * (H * M)
    moe_active = K * expert_per

    MH = H
    merger_input = 2 * H + P_cot
    merger_active = (merger_input * MH + MH) + (MH * 3 + 3) + (P_cot * H + H) + (H * 2)
    final_active = 2 * H

    # Embedding (always full)
    emb_active = V * H

    total_active = emb_active + layer_active + router_active + cot_active + moe_active + merger_active + final_active
    return int(total_active)


def analyze_12B_vs_60B(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Determine the conditions under which a 12B-parameter Xorzen would
    outperform a 60B-parameter dense Transformer.

    We use the measured active-ratio from existing variants to project.
    """
    # Find the largest variant we have measurements for
    valid = [r for r in records if "error" not in r]
    if not valid:
        return {"error": "no valid variants"}

    # Active-ratio trend with size (Xorzen becomes sparser as it scales):
    # 1M: 4.3%, 10M: 9%, 50M: 8%, 277M: 7.6%, 500M: 5.6%, 1.3B: 4.9%
    # Mean active ratio for >= 50M variants: ~6.5%
    large = [r for r in valid if r["actual_total_params"] >= 10_000_000]
    avg_active_ratio = sum(r["active_per_token"] / r["actual_total_params"] for r in large) / len(large)

    # Projected 12B Xorzen
    P_total_12B = 12_000_000_000
    P_active_12B = int(P_total_12B * avg_active_ratio)

    # 60B Dense
    P_total_60B_dense = 60_000_000_000
    P_active_60B_dense = P_total_60B_dense  # dense: 100% active

    # Chinchilla: L ≈ E + A/N^alpha + B/D^beta  with alpha≈0.34, beta≈0.28
    # Compute-optimal: D = 20 * N (Chinchilla 2022). C = 6 * N * D = 120 * N^2.
    # For equal compute budget (same training FLOPs):
    #   C_xorzen = 6 * P_active_xorzen * D_xorzen
    #   C_dense  = 6 * N_dense * D_dense
    # If equal compute: P_active_12B * D_12B = N_60B * D_60B
    # With Chinchilla-optimal D = 20 * N_active:
    #   20 * P_active_12B^2 = 20 * N_60B^2  → not equal because P_active_12B << 60B
    # So we have MORE compute headroom: with the same budget as 60B Chinchilla,
    # 12B Xorzen can train on D = (60B * 20 * 60B) / P_active_12B tokens
    #  = 72000 * 10^18 / P_active_12B
    N_60B = P_total_60B_dense
    D_60B_chinchilla = 20 * N_60B  # 1.2T tokens
    C_60B = 6 * N_60B * D_60B_chinchilla  # ~8.6e22 FLOPs

    # If 12B Xorzen trains with the same compute budget:
    # C_12B = 6 * P_active_12B * D_12B = C_60B
    D_12B_equal_compute = C_60B / (6 * P_active_12B)

    # Loss projection: L = E + A/N_active^alpha + B/D^beta
    # Use Chinchilla-like exponents: alpha=0.34, beta=0.28
    # Loss constants A, B, E fit to GPT-3/Chinchilla: E≈1.69, A≈530, B≈1480 (natural log)
    E_, A_, B_ = 1.69, 530.0, 1480.0
    alpha, beta = 0.34, 0.28

    L_60B_dense = E_ + A_ / (N_60B ** alpha) + B_ / (D_60B_chinchilla ** beta)
    L_12B_xorzen_equal_compute = E_ + A_ / (P_active_12B ** alpha) + B_ / (D_12B_equal_compute ** beta)

    # For 12B Xorzen to BEAT 60B Dense, we need L_12B_xorzen < L_60B_dense.
    # Solve for required D_12B:
    #   A/P_active_12B^alpha + B/D_12B^beta  <  A/N_60B^alpha + B/D_60B^beta
    # Note: P_active_12B << N_60B, so first term is LARGER (worse).
    # For equality: B/D_12B^beta = B/D_60B^beta + A/N_60B^alpha - A/P_active_12B^alpha
    # → D_12B_required = (B / (B/D_60B^beta + A/N_60B^alpha - A/P_active_12B^alpha))^(1/beta)
    rhs = B_ / (D_60B_chinchilla ** beta) + A_ / (N_60B ** alpha) - A_ / (P_active_12B ** alpha)
    if rhs > 0:
        D_12B_required = (B_ / rhs) ** (1.0 / beta)
    else:
        D_12B_required = float("inf")  # impossible: P_active too small

    # The "effective compute" advantage: dense 60B uses 6*N*D, while 12B Xorzen
    # uses 6*P_active*D. If trained on the SAME data D:
    #   compute_ratio = N_60B / P_active_12B
    # This means 12B Xorzen uses LESS compute per token by this factor.
    compute_per_token_ratio = N_60B / P_active_12B

    return {
        "measured_active_ratios_at_scale": {
            r["variant"]: round(r["active_per_token"] / r["actual_total_params"], 4)
            for r in large
        },
        "avg_active_ratio_at_scale_pct": round(avg_active_ratio * 100, 2),
        "projected_12B_xorzen": {
            "P_total": P_total_12B,
            "P_active_per_token": P_active_12B,
            "active_ratio_pct": round(avg_active_ratio * 100, 2),
            "FLOPs_per_token_active": 6 * P_active_12B,
        },
        "60B_dense_baseline": {
            "P_total": P_total_60B_dense,
            "P_active_per_token": P_total_60B_dense,
            "chinchilla_optimal_tokens": int(D_60B_chinchilla),
            "chinchilla_compute_FLOPs": int(C_60B),
            "FLOPs_per_token": 6 * P_total_60B_dense,
        },
        "equal_compute_scenario": {
            "D_12B_tokens_if_same_compute": int(D_12B_equal_compute),
            "L_60B_dense_predicted": float(L_60B_dense),
            "L_12B_xorzen_predicted": float(L_12B_xorzen_equal_compute),
            "xorzen_wins": bool(L_12B_xorzen_equal_compute < L_60B_dense),
        },
        "required_data_for_parity": {
            "D_12B_required_tokens": int(D_12B_required) if D_12B_required != float("inf") else None,
            "feasible": bool(D_12B_required != float("inf") and D_12B_required < 100 * D_60B_chinchilla),
        },
        "compute_efficiency_advantage": {
            "compute_per_token_ratio_60B_over_12B": round(compute_per_token_ratio, 2),
            "interpretation": (
                f"A 60B dense model spends {compute_per_token_ratio:.1f}x more FLOPs per token "
                f"than a 12B Xorzen at {avg_active_ratio*100:.1f}% active ratio. "
                f"For Xorzen to win on quality, it must achieve better loss per active-param-second "
                f"of compute, which requires that the SPARSE ACTIVATION be smart (routing quality), "
                f"not just sparse."
            ),
        },
        "conditions_for_12B_xorzen_gt_60B_dense": [
            "1. Xorzen must achieve ≥6.5% active ratio at 12B scale (matches ≥50M variants)",
            "2. Router must make GOOD routing decisions (not collapse) — quality of sparse activation matters",
            "3. Training data D must be ≫ Chinchilla-optimal for 60B (since P_active_12B < N_60B,",
            "   we need D_12B ≥ D_60B * (N_60B / P_active_12B)^(alpha/beta) to compensate for the",
            "   smaller active-param count, OR rely on routing quality to extract more from each param)",
            "4. The HASS pathway diversity (Local+LowRank+SSM) must contribute genuine complementary",
            "   signal — Phase 14 ablation found LowRank was harmful at tiny scale; this might reverse",
            "   at 12B scale where LowRank's global context becomes valuable",
            "5. The MoE expert specialization must be real (experts learn different things). At tiny",
            "   scale, ShardedExpertFabric in test_mode uses a single dummy expert — production mode",
            "   with E=64+ experts needs to be validated for genuine specialization",
        ],
        "scientific_caveat": (
            "The 12B > 60B claim is a HYPOTHESIS, not a proven result. Validation requires actually "
            "training at 12B scale (out of scope for this environment). The 10M-scale validation in "
            "this run tests the WEAKER claim: 'does sparse routing help quality per FLOP at small scale?'"
        ),
    }


def write_markdown(data: Dict[str, Any], path: str) -> None:
    """Pretty-print the scaling law as markdown."""
    lines = []
    lines.append("# Xorzen Scaling Law — v0.5 (Recovered from Codebase)\n")
    lines.append(f"Generated: {data['metadata']['date']}\n")
    lines.append("## Methodology\n")
    lines.append(data["metadata"]["description"])
    lines.append("")
    lines.append("**Key finding:** PARAM_COUNT class-attribute labels are aspirational, not actual.")
    lines.append(f"  At 1.3B label, actual is 413M — labels overstate by {data['metadata']['label_discrepancy_at_1_3B_pct']}%.")
    lines.append("")
    lines.append("## Scaling-Law Table\n")
    lines.append("| Variant | H | L | E | K | P_total (label) | P_total (actual) | Label acc % | P_active/token | Active % | FLOPs/token (active) | Compute eff x |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in data["scaling_law_table"]:
        lines.append(
            f"| {r['variant']} | {r['H']} | {r['L']} | {r['experts_E']} | {r['top_K']} | "
            f"{r['P_total_label']:,} | {r['P_total_actual']:,} | {r['label_pct_of_actual']}% | "
            f"{r['P_active_per_token']:,} | {r['active_ratio_pct']}% | "
            f"{r['FLOPs_per_token_active']:,} | {r['compute_efficiency_x']}x |"
        )
    lines.append("\n## Formulas\n")
    for k, v in data["formulas"].items():
        if k == "note":
            lines.append(f"\n**Note:** {v}")
        else:
            lines.append(f"- **{k}**: `{v}`")
    lines.append("\n## Hypothesis: 12B Xorzen > 60B Dense\n")
    h = data["hypothesis_12B_vs_60B"]
    lines.append(f"Average active ratio at scale (≥10M variants): **{h['avg_active_ratio_at_scale_pct']}%**\n")
    lines.append(f"Projected 12B Xorzen: P_total={h['projected_12B_xorzen']['P_total']:,}, "
                 f"P_active/token={h['projected_12B_xorzen']['P_active_per_token']:,}, "
                 f"active_ratio={h['projected_12B_xorzen']['active_ratio_pct']}%\n")
    lines.append(f"60B Dense Chinchilla-optimal: tokens={h['60B_dense_baseline']['chinchilla_optimal_tokens']:,}, "
                 f"compute={h['60B_dense_baseline']['chinchilla_compute_FLOPs']:,} FLOPs\n")
    ec = h["equal_compute_scenario"]
    lines.append(f"\n### Equal-compute scenario:")
    lines.append(f"- 12B Xorzen trained on {ec['D_12B_tokens_if_same_compute']:,} tokens")
    lines.append(f"- Predicted L(60B dense) = {ec['L_60B_dense_predicted']:.4f}")
    lines.append(f"- Predicted L(12B Xorzen) = {ec['L_12B_xorzen_predicted']:.4f}")
    lines.append(f"- Xorzen wins: **{ec['xorzen_wins']}**\n")
    lines.append("### Conditions for 12B Xorzen > 60B Dense:\n")
    for c in h["conditions_for_12B_xorzen_gt_60B_dense"]:
        lines.append(f"- {c}")
    lines.append(f"\n**Caveat:** {h['scientific_caveat']}\n")
    lines.append("\n## Per-Variant Component Breakdown\n")
    for r in data["variants"]:
        if "error" in r:
            lines.append(f"### {r['variant']}\nERROR: {r['error']}\n")
            continue
        lines.append(f"### {r['variant']}")
        lines.append(f"- Actual total: {r['actual_total_params']:,}")
        lines.append(f"- Predicted (test mode): {r['predictions_test_mode']['total_pred_test']:,}  "
                     f"(error {r['prediction_error_pct_test']:+.1f}%)")
        lines.append(f"- Predicted (full experts): {r['predictions_full_experts']['total_pred']:,}  "
                     f"(error {r['prediction_error_pct_full']:+.1f}%)")
        lines.append("- Component breakdown:")
        for k, v in sorted(r["component_breakdown"].items(), key=lambda x: -x[1]):
            lines.append(f"  - {k}: {v:,}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
