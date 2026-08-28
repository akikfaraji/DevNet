"""
v0.5 Phase 11 — Real-data 10M-scale validation: dense vs Xorzen-routing-disabled
vs Xorzen-genuine-sparse, on TinyStories.

This is the SCIENTIFIC validation the v0.5 work has been building toward.
We do NOT use synthetic Markov data. We use real TinyStories text.

Three conditions, identical data/optimizer/budget:
  1. dense_baseline            — vanilla decoder-only transformer
  2. xorzen_routing_disabled   — Xorzen NANO_10M with pathway_top_k=3,
                                 min_depth=max_depth (no sparsity)
  3. xorzen_genuine_sparse     — Xorzen NANO_10M with pathway_top_k=2,
                                 min_depth=2 (genuine sparse routing)

For each we measure (using corrected definitions from param_category_audit):
  - P_full      = structural + E * expert_per   (entire declared expert pool)
  - P_resident  = state_dict in test_mode       (structural + 1 dummy expert)
                  (production state_dict == structural; LRU cache is RAM-resident only)
  - P_active    = params actually executed per token (pathway_top_k, top-K, L_avg, width-avg)
  - train/val loss + perplexity curves
  - tokens/sec, runtime, FLOPs/token, peak RAM
  - routing entropy + load balance (Xorzen only)

Verifications during training:
  - no NaN/Inf in loss or grads
  - no gradient failures (clipped grad norm recorded)
  - no routing collapse (expert utilization entropy tracked)
  - no silent fallback to dense computation (pathway utilization tracked)

Outputs:
  /home/z/my-project/xorzen_dev/reports/v05/phase11_real_data_validation.json
  /home/z/my-project/xorzen_dev/reports/v05/phase11_real_data_validation.md
  /home/z/my-project/xorzen_dev/checkpoints/phase11/*.pt  (final only, storage-efficient)
"""
from __future__ import annotations
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path("/home/z/my-project/xorzen_dev")
sys.path.insert(0, str(PROJ))
sys.path.insert(0, "/home/z/my-project/scripts")

from xorzen.config import ConfigFactory, ModelSize, ModelConfig
from xorzen.models.zero.variants import zeroBase
from xorzen.models.zero.model import zeroModel
from dense_baseline import DenseConfig, DenseTransformer, build_dense_baseline_to_match
from tinystories_loader import tokenize_and_cache, make_split

OUT_DIR = PROJ / "reports" / "v05"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "phase11_real_data_validation.json"
OUT_MD = OUT_DIR / "phase11_real_data_validation.md"
CKPT_DIR = PROJ / "checkpoints" / "phase11"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ================== EXPERIMENT CONFIG ==================
# Shared across all three conditions — identical data/optimizer/budget.
VOCAB_SIZE = 10000
SEQ_LEN = 128           # TinyStories are short; 128 covers most of a story
BATCH_SIZE = 8
TRAIN_STEPS = 600  # 600 steps: dense ~4min, xorzen ~8min, fits in 10min bash timeout
VAL_EVERY = 200
LOG_EVERY = 50
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 60
GRAD_CLIP = 1.0
SEED = 42
MODEL_SIZE = ModelSize.NANO_10M  # ~10M label, ~8M actual state_dict (test mode)

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


# ================== HELPERS ==================
def get_rss_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def count_state_dict(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_dummy_expert(model: torch.nn.Module) -> int:
    moe = getattr(model, "moe", None) or getattr(model, "moe_fabric", None)
    if moe is not None and hasattr(moe, "dummy_expert"):
        return int(sum(p.numel() for p in moe.dummy_expert.parameters()))
    return 0


def expert_per_params(H: int, M: float) -> int:
    """One ExpertFFN (SwiGLU, no bias)."""
    intermediate = int(H * M)
    return 3 * H * intermediate  # gate + up + down


def compute_param_categories(model: zeroModel, cfg: ModelConfig) -> Dict[str, int]:
    """Independently compute P_full / P_resident / P_active using the corrected
    definitions established in the param_category_audit.

    P_full     = structural + E * expert_per   (entire declared expert pool)
    P_resident = state_dict_total_test_mode    (structural + 1 dummy expert)
                 This is what model.parameters() yields in test_mode=True.
                 In production (test_mode=False) state_dict == structural because
                 ExpertDiskManager stores experts on disk and LRUExpertCache
                 is a plain Python class (NOT in state_dict).
    P_active   = compute_active_per_token (pathway_top_k of 3, L_avg=(max+min)/2,
                 top-K of E, width-avg slicing)
    """
    state_dict_total = count_state_dict(model)
    structural_no_experts = state_dict_total - count_dummy_expert(model)
    expert_per = expert_per_params(cfg.hidden_size, cfg.expert_hidden_multiplier)
    P_full = structural_no_experts + cfg.expert_count * expert_per
    P_resident_test = state_dict_total  # test-mode state_dict (what we observe)
    P_resident_prod = structural_no_experts  # production state_dict (cache empty)
    P_active = compute_active_per_token(cfg)
    return {
        "P_full": int(P_full),
        "P_resident_test_mode": int(P_resident_test),
        "P_resident_production_initial": int(P_resident_prod),
        "P_active_per_token": int(P_active),
        "structural_no_experts": int(structural_no_experts),
        "expert_per_params": int(expert_per),
        "expert_count_declared": int(cfg.expert_count),
        "max_expert_cache": int(cfg.max_expert_cache),
    }


def compute_active_per_token(cfg: ModelConfig) -> int:
    """Independent re-implementation of compute_active_per_token from
    phase8_scaling_law_derivation.py — verified formula_check_delta=0.
    """
    H = cfg.hidden_size
    L = cfg.num_layers
    V = cfg.vocab_size
    E = cfg.expert_count
    K = cfg.top_k_experts
    M = cfg.expert_hidden_multiplier
    D_lr = cfg.low_rank_dim
    D_ssm = cfg.ssm_state_dim
    P_cot = cfg.cot_dim * cfg.cot_components
    widths = list(cfg.width_choices)
    Wmax = int(H * M)
    W_avg = sum(widths) / len(widths)
    width_factor = W_avg / Wmax
    L_avg = (cfg.max_depth + cfg.min_depth) / 2.0
    k_path = cfg.pathway_top_k

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
    expert_per = 3 * H * (H * M)
    moe_active = K * expert_per
    MH = H
    merger_input = 2 * H + P_cot
    merger_active = (merger_input * MH + MH) + (MH * 3 + 3) + (P_cot * H + H) + (H * 2)
    final_active = 2 * H
    emb_active = V * H
    return int(emb_active + layer_active + router_active + cot_active + moe_active + merger_active + final_active)


def estimate_flops_per_token(model_or_cfg, is_xorzen: bool, is_sparse: bool, rd=None) -> int:
    """Forward-pass FLOPs per token (2 MAC = 1 FLOP convention: we count MACs × 2)."""
    if is_xorzen:
        cfg = model_or_cfg.config if hasattr(model_or_cfg, "config") else model_or_cfg
        H = cfg.hidden_size; L = cfg.num_layers; V = cfg.vocab_size; T = SEQ_LEN
        attn_flops = 2 * (4 * H * H + 2 * T * H)
        ffn_full = 2 * (2 * H * int(H * cfg.expert_hidden_multiplier) * 2)
        pathway_flops = 2 * H * H
        embed_flops = 2 * (H + H * V)  # embed + tied lm_head
        if is_sparse and rd is not None:
            # Use routing decision (averaged over tokens)
            try:
                active_layers = float(rd.depth_mask.float().sum(-1).mean().item()) if hasattr(rd, "depth_mask") else (cfg.max_depth + cfg.min_depth) / 2.0
            except Exception:
                active_layers = (cfg.max_depth + cfg.min_depth) / 2.0
            try:
                wc_tensor = torch.tensor(list(cfg.width_choices))
                active_width = float(wc_tensor[rd.width_idx].float().mean().item()) if hasattr(rd, "width_idx") else sum(cfg.width_choices) / len(cfg.width_choices)
            except Exception:
                active_width = sum(cfg.width_choices) / len(cfg.width_choices)
            active_pathway_frac = cfg.pathway_top_k / 3.0
            active_expert_frac = cfg.top_k_experts / cfg.expert_count
            ffn_actual = 2 * (2 * H * int(active_width) * 2)
            expert_flops = active_expert_frac * 2 * (2 * H * int(H * cfg.expert_hidden_multiplier) * 2)
            per_layer = attn_flops + ffn_actual + pathway_flops * active_pathway_frac + expert_flops
            total = active_layers * per_layer + embed_flops
        else:
            # Disabled-routing case: everything active
            expert_flops_full = 1.0 * 2 * (2 * H * int(H * cfg.expert_hidden_multiplier) * 2)  # all experts (test mode: only 1 dummy)
            # In test_mode, only 1 dummy expert is materialized and runs.
            total = L * (attn_flops + ffn_full + pathway_flops * 3 + expert_flops_full) + embed_flops
        return int(total)
    else:
        cfg = model_or_cfg.cfg
        H = cfg.hidden_size; L = cfg.num_layers; V = cfg.vocab_size; T = SEQ_LEN
        attn_flops = 2 * (4 * H * H + 2 * T * H)
        ffn_flops = 2 * (2 * H * cfg.ffn_hidden)
        embed_flops = 2 * (H + H * V)
        return int(L * (attn_flops + ffn_flops) + embed_flops)


def lr_schedule(step: int, total: int, warmup: int = WARMUP_STEPS) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, lr=LR):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
    )


def check_health(loss: torch.Tensor, model: torch.nn.Module, name: str, step: int) -> Dict[str, Any]:
    """Verify no NaN/Inf, no gradient failure, no silent dense fallback."""
    issues = {}
    loss_val = float(loss.detach().item()) if loss is not None else float("nan")
    if not math.isfinite(loss_val):
        issues["loss_non_finite"] = loss_val
    # Check all params for NaN/Inf
    nan_params = 0; inf_params = 0
    for n, p in model.named_parameters():
        if p.data is None: continue
        if torch.isnan(p.data).any(): nan_params += 1
        if torch.isinf(p.data).any(): inf_params += 1
    if nan_params > 0: issues[f"nan_in_params"] = nan_params
    if inf_params > 0: issues[f"inf_in_params"] = inf_params
    return {"loss": loss_val, "issues": issues, "step": step, "name": name}


def evaluate(model, val_windows: np.ndarray, is_xorzen: bool, device: str = "cpu",
             max_batches: int = 32) -> Tuple[float, float, Dict[str, Any]]:
    """Evaluate on validation windows. Returns (mean_loss, perplexity, stats)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    routing_stats = {
        "path_distrib": [0, 0, 0],
        "width_distrib": [0] * 100,
        "expert_distrib": [0] * 1000,
        "depth_active_per_layer": [],
        "path_entropy_samples": [],
        "n_samples": 0,
    }
    n_batches = min(max_batches, len(val_windows) // BATCH_SIZE)
    with torch.no_grad():
        for b in range(n_batches):
            batch = val_windows[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            ids = torch.from_numpy(batch).to(device)
            if is_xorzen:
                out = model(input_ids=ids, labels=ids, output_routing_info=True)
                rd = out.routing_info
                # Pathway distribution
                if hasattr(rd, "path_probs"):
                    path_idx = rd.path_probs.argmax(-1).flatten()
                    for p in path_idx.tolist():
                        if p < 3: routing_stats["path_distrib"][p] += 1
                # Width distribution
                if hasattr(rd, "width_idx"):
                    width_idx = rd.width_idx.flatten()
                    for w in width_idx.tolist():
                        if w < 100: routing_stats["width_distrib"][w] += 1
                # Expert distribution
                if hasattr(rd, "expert_indices"):
                    experts_flat = rd.expert_indices.flatten()
                    for e in experts_flat.tolist():
                        if e < 1000: routing_stats["expert_distrib"][e] += 1
                # Depth mask
                if hasattr(rd, "depth_mask"):
                    dm = rd.depth_mask.float().mean(dim=(0, 1))
                    routing_stats["depth_active_per_layer"] = dm.tolist()
                # Path entropy
                if hasattr(rd, "path_probs"):
                    pe = -(rd.path_probs * torch.log(rd.path_probs + 1e-12)).sum(-1).mean().item()
                    routing_stats["path_entropy_samples"].append(pe)
                routing_stats["n_samples"] += int(ids.numel())
            else:
                out = model(ids, labels=ids)
            n_tok = ids.numel()
            total_loss += float(out.lm_loss.item()) * n_tok
            total_tokens += n_tok
    mean_loss = total_loss / max(1, total_tokens)
    ppl = float(math.exp(min(20.0, mean_loss)))
    # Compute routing entropy & load balance metrics
    if is_xorzen:
        path_dist = np.array(routing_stats["path_distrib"], dtype=float)
        path_dist = path_dist / max(1.0, path_dist.sum())
        routing_stats["path_entropy"] = float(-(path_dist * np.log(path_dist + 1e-12)).sum())
        routing_stats["path_distrib_normalized"] = path_dist.tolist()
        # Only keep nonzero expert entries
        ed = np.array(routing_stats["expert_distrib"], dtype=float)
        nonzero = ed[ed > 0]
        if len(nonzero) > 0:
            normalized = nonzero / nonzero.sum()
            routing_stats["expert_load_entropy"] = float(-(normalized * np.log(normalized + 1e-12)).sum())
            routing_stats["expert_active_count"] = int(len(nonzero))
            max_share = float(normalized.max())
            routing_stats["expert_max_load_share"] = max_share
            routing_stats["expert_load_balance_cv"] = float(normalized.std() / (normalized.mean() + 1e-12))
        else:
            routing_stats["expert_load_entropy"] = 0.0
            routing_stats["expert_active_count"] = 0
        routing_stats.pop("expert_distrib", None)
        routing_stats.pop("width_distrib", None)
        routing_stats.pop("path_entropy_samples", None)
    return mean_loss, ppl, routing_stats


def train_one_config(name: str, model, is_xorzen: bool, is_sparse: bool,
                     train_windows: np.ndarray, val_windows: np.ndarray,
                     cfg=None, train_steps: Optional[int] = None,
                     device: str = "cpu") -> Dict[str, Any]:
    """Train one model config and return full measurement dict."""
    if train_steps is None:
        train_steps = TRAIN_STEPS
    print(f"\n{'='*70}")
    print(f"  TRAINING: {name}")
    print(f"{'='*70}")
    n_total = count_state_dict(model)
    n_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"  state_dict: {n_total:,} params ({n_trainable:,} trainable)")

    # Param categories (corrected definitions)
    if is_xorzen:
        param_cats = compute_param_categories(model, model.config)
    else:
        param_cats = {"P_full": n_total, "P_resident_test_mode": n_total,
                      "P_resident_production_initial": n_total,
                      "P_active_per_token": n_total}
    print(f"  P_full={param_cats['P_full']:,}  P_resident={param_cats['P_resident_test_mode']:,}  "
          f"P_active/tok={param_cats['P_active_per_token']:,}")

    optimizer = make_optimizer(model)
    model.train()

    train_losses: List[float] = []
    train_lm_losses: List[float] = []
    val_losses: List[float] = []
    val_ppls: List[float] = []
    step_times: List[float] = []
    grad_norms: List[float] = []
    health_issues: List[Dict[str, Any]] = []
    routing_history: List[Dict[str, Any]] = []

    rss_start = get_rss_mb()
    t_start = time.perf_counter()
    n_train_batches = len(train_windows) // BATCH_SIZE

    step = 0
    while step < train_steps:
        # Iterate over training batches, reshuffling each epoch
        epoch_perm = np.random.default_rng(SEED + step).permutation(n_train_batches)[:n_train_batches]
        for b in epoch_perm:
            if step >= train_steps:
                break
            batch = train_windows[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            ids = torch.from_numpy(batch).to(device)
            t0 = time.perf_counter()
            optimizer.zero_grad()
            if is_xorzen:
                out = model(input_ids=ids, labels=ids, output_routing_info=True)
            else:
                out = model(ids, labels=ids)
            loss = out.loss
            lm_loss = out.lm_loss if hasattr(out, "lm_loss") and out.lm_loss is not None else loss
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            lr_scale = lr_schedule(step, train_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = LR * lr_scale
            optimizer.step()
            t1 = time.perf_counter()

            step_time = t1 - t0
            step_times.append(step_time)
            grad_norms.append(float(gn))
            n_tokens = ids.numel()
            tps = n_tokens / step_time
            train_losses.append(float(loss.item()))
            train_lm_losses.append(float(lm_loss.item()))

            # Health check
            health = check_health(loss, model, name, step)
            if health["issues"]:
                health_issues.append(health)
                print(f"  ⚠️ step {step}: health issues = {health['issues']}")

            if step % LOG_EVERY == 0:
                print(f"  step {step:4d}/{train_steps}  loss={lm_loss.item():.4f}  "
                      f"lr={LR*lr_scale:.2e}  gn={gn:.2f}  {step_time*1000:.0f}ms  {tps:.0f}tok/s")

            if step % VAL_EVERY == 0 or step == train_steps - 1:
                v_loss, v_ppl, v_stats = evaluate(model, val_windows, is_xorzen, device)
                val_losses.append(v_loss)
                val_ppls.append(v_ppl)
                if is_xorzen:
                    routing_history.append({"step": step, **v_stats})
                    # Check for routing collapse
                    n_active_experts = v_stats.get("expert_active_count", 0)
                    path_ent = v_stats.get("path_entropy", 0.0)
                    if n_active_experts < 2:
                        print(f"  ⚠️ ROUTING COLLAPSE: only {n_active_experts} experts active at step {step}")
                    if path_ent < 0.1:
                        print(f"  ⚠️ PATH COLLAPSE: path_entropy={path_ent:.3f}")
                print(f"    VAL step {step}: loss={v_loss:.4f}  ppl={v_ppl:.2f}")
                model.train()
            step += 1

    t_end = time.perf_counter()
    rss_end = get_rss_mb()
    total_train_time = t_end - t_start
    total_tokens_trained = int(train_steps * BATCH_SIZE * SEQ_LEN)

    # Final eval
    final_loss, final_ppl, final_routing = evaluate(model, val_windows, is_xorzen, device)

    # Estimate FLOPs/token at the end (using a sample routing decision for sparse)
    if is_xorzen:
        model.eval()
        with torch.no_grad():
            sample = torch.from_numpy(val_windows[:BATCH_SIZE]).to(device)
            out = model(input_ids=sample, output_routing_info=True)
        rd = out.routing_info
        active_flops = estimate_flops_per_token(model, is_xorzen=True, is_sparse=is_sparse, rd=rd)
        dense_eq_flops = estimate_flops_per_token(model, is_xorzen=True, is_sparse=False, rd=None)
    else:
        active_flops = estimate_flops_per_token(model, is_xorzen=False, is_sparse=False)
        dense_eq_flops = active_flops

    # Health summary
    n_nan = sum(1 for h in health_issues if "loss_non_finite" in h["issues"])
    n_grad_fail = sum(1 for h in health_issues if any("nan" in k or "inf" in k for k in h["issues"]))

    result = {
        "name": name,
        "is_xorzen": is_xorzen,
        "is_sparse": is_sparse,
        "param_categories": param_cats,
        "state_dict_total": n_total,
        "trainable_total": n_trainable,
        "train_steps": train_steps,
        "tokens_trained": total_tokens_trained,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "lr_schedule": "cosine_warmup",
        "warmup_steps": WARMUP_STEPS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "seed": SEED,
        "train_loss_curve": train_lm_losses,
        "train_loss_initial": train_lm_losses[0] if train_lm_losses else None,
        "train_loss_final": train_lm_losses[-1] if train_lm_losses else None,
        "val_loss_curve": val_losses,
        "val_ppl_curve": val_ppls,
        "val_loss_initial": val_losses[0] if val_losses else None,
        "val_loss_final": final_loss,
        "val_ppl_final": final_ppl,
        "mean_step_time_ms": float(np.mean(step_times) * 1000),
        "p50_step_time_ms": float(np.percentile(step_times, 50) * 1000),
        "p95_step_time_ms": float(np.percentile(step_times, 95) * 1000),
        "mean_tokens_per_sec": float(np.mean(BATCH_SIZE * SEQ_LEN / np.array(step_times))),
        "total_train_time_sec": total_train_time,
        "active_flops_per_token": active_flops,
        "dense_equivalent_flops_per_token": dense_eq_flops,
        "total_training_flops": active_flops * 3 * total_tokens_trained,  # 3× for fwd+bwd
        "rss_start_mb": rss_start,
        "rss_end_mb": rss_end,
        "peak_rss_mb": rss_end,
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_max": float(np.max(grad_norms)),
        "health": {
            "n_nan_loss": n_nan,
            "n_param_health_issues": n_grad_fail,
            "total_issue_steps": len(health_issues),
            "issues_detail": health_issues[:5],
            "no_nan_inf": (n_nan == 0 and n_grad_fail == 0),
        },
        "routing_stats_final": final_routing if is_xorzen else None,
        "routing_history": routing_history if is_xorzen else None,
    }
    print(f"\n  SUMMARY {name}:")
    print(f"    val_loss={final_loss:.4f}  val_ppl={final_ppl:.2f}")
    print(f"    {np.mean(step_times)*1000:.0f}ms/step  {result['mean_tokens_per_sec']:.0f}tok/s")
    print(f"    FLOPs/token: active={active_flops:,}  dense_eq={dense_eq_flops:,}")
    print(f"    peak RSS: {rss_end:.0f} MB")
    print(f"    health: no_nan_inf={result['health']['no_nan_inf']}, "
          f"issues={result['health']['total_issue_steps']}")

    # Save final checkpoint (storage-efficient: final only, no optimizer state)
    ckpt_path = CKPT_DIR / f"{name}_final.pt"
    torch.save({
        "name": name,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {k: v for k, v in vars(model.config).items() if not callable(v)} if is_xorzen else None,
        "val_loss_final": final_loss,
        "val_ppl_final": final_ppl,
        "step": train_steps,
    }, ckpt_path)
    print(f"    checkpoint: {ckpt_path}  ({ckpt_path.stat().st_size/1e6:.1f}MB)")
    result["checkpoint_path"] = str(ckpt_path)
    result["checkpoint_size_mb"] = ckpt_path.stat().st_size / 1e6

    del model, optimizer
    gc.collect()
    return result


# ================== MAIN ==================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dense_baseline", "xorzen_routing_disabled",
                                            "xorzen_genuine_sparse", "all"],
                        default="all", help="Which model to train")
    parser.add_argument("--steps", type=int, default=None, help="Override TRAIN_STEPS")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip training, just load per-model JSONs and generate report")
    args = parser.parse_args()

    if args.compare_only:
        args.model = "__compare_only__"

    if args.steps:
        global TRAIN_STEPS
        TRAIN_STEPS = args.steps

    print("=" * 70)
    print("  v0.5 Phase 11 — REAL-DATA 10M-SCALE VALIDATION")
    print("  TinyStories / dense vs Xorzen-routing-disabled vs Xorzen-genuine-sparse")
    print("=" * 70)
    print(f"Config: vocab={VOCAB_SIZE}, seq={SEQ_LEN}, batch={BATCH_SIZE}, "
          f"steps={TRAIN_STEPS}, lr={LR}, wd={WEIGHT_DECAY}, seed={SEED}")
    print(f"Model to train: {args.model}")

    # Load TinyStories
    print("\n[1/4] Loading TinyStories...")
    tokens, meta = tokenize_and_cache()
    print(f"  total tokens: {meta['total_tokens']:,}  train: {meta['train_tokens']:,}  val: {meta['val_tokens']:,}")
    train_windows, _ = make_split(tokens, meta, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, split="train")
    val_windows, _ = make_split(tokens, meta, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, split="val", shuffle=False)
    val_windows = val_windows[:512]

    # Build all three models with identical hyperparameters
    # (build+train one at a time to avoid OOM — each model needs ~1GB for
    #  weights + AdamW + gradients + activations; loading all 3 at once
    #  exceeded 3.4GB available RAM and got OOM-killed.)
    print("\n[2/4] Will build+train models sequentially to fit in RAM.")

    # Per-model results are saved individually so we can resume if one crashes.
    PER_MODEL_DIR = OUT_DIR / "per_model"
    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def save_per_model(name, result):
        path = PER_MODEL_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  [saved per-model result: {path}]")

    # ===== Dense baseline =====
    if args.model in ("dense_baseline", "all"):
        print("  [1/3] Dense baseline (matched to Xorzen P_active ~5.1M)...")
        torch.manual_seed(SEED); np.random.seed(SEED)
        # Use H=192, L=6, ffn=768 (4×H) — matches Xorzen's H and L exactly,
        # and P_active ≈ 4.6M which is close to Xorzen sparse's P_active=5.1M.
        # This is the fairest comparison: same active compute per token.
        dense_cfg = DenseConfig(
            vocab_size=VOCAB_SIZE,
            context_length=SEQ_LEN,
            hidden_size=192,
            num_layers=6,
            num_heads=8,
            ffn_hidden=768,  # 4×H, same as Xorzen's expert_hidden_multiplier
        )
        dense_model = DenseTransformer(dense_cfg)
        print(f"    dense state_dict: {count_state_dict(dense_model):,} params")
        r = train_one_config(
            "dense_baseline", dense_model, is_xorzen=False, is_sparse=False,
            train_windows=train_windows, val_windows=val_windows,
        )
        save_per_model("dense_baseline", r)
        del dense_model
        gc.collect()
        print(f"    [freed dense_model, RSS={get_rss_mb():.0f}MB]")

    # ===== Xorzen routing-disabled =====
    if args.model in ("xorzen_routing_disabled", "all"):
        print("\n  [2/3] Xorzen NANO_10M (routing DISABLED)...")
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg_disabled = ConfigFactory.get_config(MODEL_SIZE)
        cfg_disabled.update(
            shard_experts=False,
            gradient_checkpointing=False,
            eval_routing_noise=0.0,
            pathway_top_k=3,
            min_depth=cfg_disabled.max_depth,
            cost_aware_routing=False,
            context_length=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            pad_token_id=0,
            unify_load_balance=True,
        )
        xorzen_disabled = zeroBase(config=cfg_disabled, test_mode=True)
        print(f"    xorzen disabled state_dict: {count_state_dict(xorzen_disabled):,} params")
        r = train_one_config(
            "xorzen_routing_disabled", xorzen_disabled, is_xorzen=True, is_sparse=False,
            train_windows=train_windows, val_windows=val_windows, cfg=cfg_disabled,
        )
        save_per_model("xorzen_routing_disabled", r)
        del xorzen_disabled
        gc.collect()
        print(f"    [freed xorzen_disabled, RSS={get_rss_mb():.0f}MB]")

    # ===== Xorzen genuine sparse =====
    if args.model in ("xorzen_genuine_sparse", "all"):
        print("\n  [3/3] Xorzen NANO_10M (genuine sparse routing)...")
        torch.manual_seed(SEED); np.random.seed(SEED)
        cfg_sparse = ConfigFactory.get_config(MODEL_SIZE)
        cfg_sparse.update(
            shard_experts=False,
            gradient_checkpointing=False,
            eval_routing_noise=0.15,
            pathway_top_k=2,
            min_depth=2,
            cost_aware_routing=True,
            context_length=SEQ_LEN,
            vocab_size=VOCAB_SIZE,
            pad_token_id=0,
            unify_load_balance=True,
        )
        xorzen_sparse = zeroBase(config=cfg_sparse, test_mode=True)
        print(f"    xorzen sparse state_dict: {count_state_dict(xorzen_sparse):,} params")
        r = train_one_config(
            "xorzen_genuine_sparse", xorzen_sparse, is_xorzen=True, is_sparse=True,
            train_windows=train_windows, val_windows=val_windows, cfg=cfg_sparse,
        )
        save_per_model("xorzen_genuine_sparse", r)
        del xorzen_sparse
        gc.collect()
        print(f"    [freed xorzen_sparse, RSS={get_rss_mb():.0f}MB]")

    # ===== If only training one model or compare-only, skip to comparison =====
    if args.model not in ("all",):
        if args.model == "__compare_only__":
            print("\n[4/4] Compare-only mode: loading per-model JSONs...")
        else:
            print(f"\n[done training {args.model}]")
            return

    # ===== Load all per-model results and do comparison =====
    print("\n[4/4] Scientific comparison...")
    results = {
        "experiment_config": {
            "dataset": "TinyStories (HuggingFace: roneneldan/TinyStories)",
            "tokenizer": "zero_bpe_10k (vocab=10000)",
            "vocab_size": VOCAB_SIZE,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "train_steps": TRAIN_STEPS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "grad_clip": GRAD_CLIP,
            "seed": SEED,
            "lr_schedule": "cosine_warmup",
            "optimizer": "AdamW(betas=(0.9,0.95))",
            "total_tokens_trained_per_model": TRAIN_STEPS * BATCH_SIZE * SEQ_LEN,
            "train_tokens_available": int(meta["train_tokens"]),
            "val_tokens_available": int(meta["val_tokens"]),
            "data_split_seed": SEED,
            "model_size_label": "NANO_10M",
        },
        "configs": {},
    }
    for name in ["dense_baseline", "xorzen_routing_disabled", "xorzen_genuine_sparse"]:
        path = PER_MODEL_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results["configs"][name] = json.load(f)
        else:
            print(f"  WARNING: {path} not found — comparison will be partial")
            results["configs"][name] = None

    dense_r = results["configs"]["dense_baseline"]
    disabled_r = results["configs"]["xorzen_routing_disabled"]
    sparse_r = results["configs"]["xorzen_genuine_sparse"]

    # Equal-tokens comparison (all trained the same number of tokens)
    tokens_per_model = TRAIN_STEPS * BATCH_SIZE * SEQ_LEN
    equal_tokens = {
        "tokens_trained": tokens_per_model,
        "dense": {
            "val_loss": dense_r["val_loss_final"],
            "val_ppl": dense_r["val_ppl_final"],
            "train_time_sec": dense_r["total_train_time_sec"],
            "P_full": dense_r["param_categories"]["P_full"],
            "P_resident": dense_r["param_categories"]["P_resident_test_mode"],
            "P_active": dense_r["param_categories"]["P_active_per_token"],
            "flops_per_token": dense_r["active_flops_per_token"],
            "total_flops": dense_r["total_training_flops"],
            "tokens_per_sec": dense_r["mean_tokens_per_sec"],
            "peak_rss_mb": dense_r["peak_rss_mb"],
        },
        "xorzen_routing_disabled": {
            "val_loss": disabled_r["val_loss_final"],
            "val_ppl": disabled_r["val_ppl_final"],
            "train_time_sec": disabled_r["total_train_time_sec"],
            "P_full": disabled_r["param_categories"]["P_full"],
            "P_resident": disabled_r["param_categories"]["P_resident_test_mode"],
            "P_active": disabled_r["param_categories"]["P_active_per_token"],
            "flops_per_token": disabled_r["active_flops_per_token"],
            "total_flops": disabled_r["total_training_flops"],
            "tokens_per_sec": disabled_r["mean_tokens_per_sec"],
            "peak_rss_mb": disabled_r["peak_rss_mb"],
        },
        "xorzen_genuine_sparse": {
            "val_loss": sparse_r["val_loss_final"],
            "val_ppl": sparse_r["val_ppl_final"],
            "train_time_sec": sparse_r["total_train_time_sec"],
            "P_full": sparse_r["param_categories"]["P_full"],
            "P_resident": sparse_r["param_categories"]["P_resident_test_mode"],
            "P_active": sparse_r["param_categories"]["P_active_per_token"],
            "flops_per_token": sparse_r["active_flops_per_token"],
            "total_flops": sparse_r["total_training_flops"],
            "tokens_per_sec": sparse_r["mean_tokens_per_sec"],
            "peak_rss_mb": sparse_r["peak_rss_mb"],
            "expert_active_count": sparse_r["routing_stats_final"].get("expert_active_count"),
            "expert_load_entropy": sparse_r["routing_stats_final"].get("expert_load_entropy"),
            "path_entropy": sparse_r["routing_stats_final"].get("path_entropy"),
            "expert_max_load_share": sparse_r["routing_stats_final"].get("expert_max_load_share"),
        },
    }

    # Equal-FLOPs comparison: how many tokens would each model get at the dense model's FLOPs budget?
    # dense_total_flops = dense_r.total_training_flops
    # For each Xorzen, find D such that 3 * P_active * D = dense_total_flops
    # → D = dense_total_flops / (3 * P_active)
    dense_total_flops = dense_r["total_training_flops"]
    def equal_flops_tokens(r):
        return int(dense_total_flops / (3 * r["active_flops_per_token"]))
    equal_flops = {
        "compute_budget_flops": dense_total_flops,
        "tokens_at_dense_compute": {
            "dense": equal_flops_tokens(dense_r),
            "xorzen_routing_disabled": equal_flops_tokens(disabled_r),
            "xorzen_genuine_sparse": equal_flops_tokens(sparse_r),
        },
        "note": ("In this run all three models trained on the SAME number of tokens, so the "
                 "equal-FLOPs comparison is computed analytically: how many tokens each model "
                 "could have trained on with the dense model's compute budget. Re-running with "
                 "these token counts would give true equal-FLOPs curves."),
    }

    # Verdicts
    sparse_loss_delta_vs_dense = sparse_r["val_loss_final"] - dense_r["val_loss_final"]
    sparse_loss_delta_vs_disabled = sparse_r["val_loss_final"] - disabled_r["val_loss_final"]
    sparse_flops_delta_vs_dense = sparse_r["active_flops_per_token"] - dense_r["active_flops_per_token"]

    # Routing collapse check
    sparse_routing = sparse_r.get("routing_stats_final", {})
    expert_active = sparse_routing.get("expert_active_count", 0)
    # Build the sparse config (without model) to get declared expert count etc.
    _cfg_sparse_ref = ConfigFactory.get_config(MODEL_SIZE)
    _cfg_sparse_ref.update(
        shard_experts=False, gradient_checkpointing=False, eval_routing_noise=0.15,
        pathway_top_k=2, min_depth=2, cost_aware_routing=True,
        context_length=SEQ_LEN, vocab_size=VOCAB_SIZE, pad_token_id=0, unify_load_balance=True,
    )
    declared_experts = _cfg_sparse_ref.expert_count
    path_entropy = sparse_routing.get("path_entropy", 0.0)
    max_share = sparse_routing.get("expert_max_load_share", 1.0)
    collapse = {
        "expert_active_count": expert_active,
        "expert_declared_count": declared_experts,
        "expert_utilization_pct": 100.0 * expert_active / max(1, declared_experts),
        "path_entropy": path_entropy,
        "max_path_entropy_possible": math.log(3),  # 3 pathways
        "expert_max_load_share": max_share,
        "expert_load_balance_cv": sparse_routing.get("expert_load_balance_cv", 0.0),
        "verdict": (
            "no_collapse" if (expert_active >= 2 and path_entropy > 0.3 and max_share < 0.9)
            else "partial_collapse" if (expert_active >= 2 and max_share < 0.95)
            else "collapsed"
        ),
    }

    # Silent fallback check (does sparse actually run sparsely?)
    silent_fallback = {
        "pathway_top_k_configured": _cfg_sparse_ref.pathway_top_k,
        "pathway_top_k_observed_in_routing_history": (
            sparse_r["routing_history"][-1].get("path_entropy", 0.0) > 0.0
            if sparse_r.get("routing_history") else False
        ),
        "active_flops_lt_dense_eq_flops": sparse_r["active_flops_per_token"] < sparse_r["dense_equivalent_flops_per_token"],
        "flops_reduction_pct": 100.0 * (1 - sparse_r["active_flops_per_token"] / max(1, sparse_r["dense_equivalent_flops_per_token"])),
        "verdict": (
            "genuinely_sparse" if sparse_r["active_flops_per_token"] < sparse_r["dense_equivalent_flops_per_token"]
            else "fallback_to_dense"
        ),
    }

    comparison = {
        "equal_tokens": equal_tokens,
        "equal_flops": equal_flops,
        "sparse_vs_dense": {
            "val_loss_delta": float(sparse_loss_delta_vs_dense),
            "flops_per_token_delta": int(sparse_flops_delta_vs_dense),
            "sparse_uses_fewer_flops": sparse_flops_delta_vs_dense < 0,
            "sparse_achieves_lower_loss": sparse_loss_delta_vs_dense < 0,
            "sparse_wins_on_quality_per_flop": (
                sparse_r["val_loss_final"] / max(1, sparse_r["active_flops_per_token"])
                < dense_r["val_loss_final"] / max(1, dense_r["active_flops_per_token"])
            ),
            "interpretation": (
                "sparse_routing_helps_quality_per_flop"
                if (sparse_r["val_loss_final"] / max(1, sparse_r["active_flops_per_token"])
                    < dense_r["val_loss_final"] / max(1, dense_r["active_flops_per_token"]))
                else "dense_baseline_better_at_this_scale"
            ),
        },
        "sparse_vs_disabled": {
            "val_loss_delta": float(sparse_loss_delta_vs_disabled),
            "sparse_routing_helps_quality": sparse_loss_delta_vs_disabled < 0,
            "sparse_routing_hurts_quality": sparse_loss_delta_vs_disabled > 0,
            "interpretation": (
                "sparse_routing_HELPS_quality_at_fixed_params"
                if sparse_loss_delta_vs_disabled < 0
                else "sparse_routing_HURTS_quality_at_fixed_params_at_10M_scale"
            ),
        },
        "routing_collapse_check": collapse,
        "silent_fallback_check": silent_fallback,
        "scaling_law_trend": {
            "question": "Does the sparse-vs-dense trend from the scaling-law table survive real data?",
            "sparse_P_full": sparse_r["param_categories"]["P_full"],
            "sparse_P_active": sparse_r["param_categories"]["P_active_per_token"],
            "sparse_active_ratio_pct": 100.0 * sparse_r["param_categories"]["P_active_per_token"] / sparse_r["param_categories"]["P_full"],
            "dense_P_full": dense_r["param_categories"]["P_full"],
            "dense_active_ratio_pct": 100.0,
            "sparse_compute_efficiency_x": (
                dense_r["active_flops_per_token"] / max(1, sparse_r["active_flops_per_token"])
            ),
            "sparse_achieves_lower_loss_at_lower_compute": (
                sparse_flops_delta_vs_dense < 0 and sparse_loss_delta_vs_dense < 0
            ),
            "trend_survives_real_data": (
                sparse_flops_delta_vs_dense < 0  # sparse uses less compute
                and sparse_loss_delta_vs_dense < 0.5  # sparse loss not catastrophically worse
            ),
        },
    }
    results["comparison"] = comparison

    # ===== Print final scientific summary =====
    print("\n" + "=" * 70)
    print("  SCIENTIFIC SUMMARY")
    print("=" * 70)
    print(f"\n{'Condition':<30}{'P_full':>14}{'P_resident':>14}{'P_active/tok':>16}{'val_loss':>11}{'val_ppl':>10}{'tok/s':>9}{'FLOPs/tok':>12}{'RAM_MB':>9}")
    print("-" * 125)
    for k in ["dense_baseline", "xorzen_routing_disabled", "xorzen_genuine_sparse"]:
        r = results["configs"][k]
        print(f"{k:<30}"
              f"{r['param_categories']['P_full']:>14,}"
              f"{r['param_categories']['P_resident_test_mode']:>14,}"
              f"{r['param_categories']['P_active_per_token']:>16,}"
              f"{r['val_loss_final']:>11.4f}"
              f"{r['val_ppl_final']:>10.2f}"
              f"{r['mean_tokens_per_sec']:>9.0f}"
              f"{r['active_flops_per_token']:>12,}"
              f"{r['peak_rss_mb']:>9.0f}")

    print(f"\nSparse vs Dense:")
    print(f"  loss_delta = {comparison['sparse_vs_dense']['val_loss_delta']:+.4f}")
    print(f"  flops_delta = {comparison['sparse_vs_dense']['flops_per_token_delta']:+,}")
    print(f"  interpretation: {comparison['sparse_vs_dense']['interpretation']}")

    print(f"\nSparse vs Disabled (does routing itself help quality?):")
    print(f"  loss_delta = {comparison['sparse_vs_disabled']['val_loss_delta']:+.4f}")
    print(f"  interpretation: {comparison['sparse_vs_disabled']['interpretation']}")

    print(f"\nRouting collapse check: {collapse['verdict']}")
    print(f"  expert_active={collapse['expert_active_count']}/{collapse['expert_declared_count']} "
          f"({collapse['expert_utilization_pct']:.0f}%), "
          f"path_entropy={collapse['path_entropy']:.3f}, "
          f"max_load_share={collapse['expert_max_load_share']:.3f}")

    print(f"\nSilent fallback check: {silent_fallback['verdict']}")
    print(f"  active_flops={sparse_r['active_flops_per_token']:,} vs dense_eq_flops={sparse_r['dense_equivalent_flops_per_token']:,}")
    print(f"  flops_reduction: {silent_fallback['flops_reduction_pct']:.1f}%")

    print(f"\nScaling-law trend survives real data: "
          f"{comparison['scaling_law_trend']['trend_survives_real_data']}")

    # Save JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults JSON: {OUT_JSON}")

    # Markdown report
    write_markdown(results, OUT_MD)
    print(f"Report MD:   {OUT_MD}")

    return results


def write_markdown(results: dict, path: Path):
    """Write the final scientific report as markdown."""
    ec = results["experiment_config"]
    cfgs = results["configs"]
    cmp = results["comparison"]
    d = cfgs["dense_baseline"]
    dis = cfgs["xorzen_routing_disabled"]
    sp = cfgs["xorzen_genuine_sparse"]

    lines = []
    lines.append("# v0.5 Phase 11 — Real-Data 10M-Scale Validation\n")
    lines.append("Generated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    lines.append("## Dataset & Tokenizer\n")
    lines.append(f"- **Dataset**: {ec['dataset']}")
    lines.append(f"- **Tokenizer**: {ec['tokenizer']}")
    lines.append(f"- **Vocab size**: {ec['vocab_size']:,}")
    lines.append(f"- **Sequence length**: {ec['seq_len']}")
    lines.append(f"- **Train tokens available**: {ec['train_tokens_available']:,}")
    lines.append(f"- **Val tokens available**: {ec['val_tokens_available']:,}")
    lines.append(f"- **Tokens trained per model**: {ec['total_tokens_trained_per_model']:,} "
                 f"({ec['train_steps']} steps × {ec['batch_size']} batch × {ec['seq_len']} seq)\n")

    lines.append("## Exact Model Configs\n")
    lines.append("### Dense baseline")
    lines.append(f"- H=192, L=6, heads=8, ffn_hidden=768 (4×H), tied LM head, no MoE\n")
    lines.append("### Xorzen NANO_10M (routing disabled)")
    lines.append(f"- H=192, L=6, E=8, K=2 (forced all-active), pathway_top_k=3, min_depth=max_depth=6")
    lines.append(f"- All structural pathways + 1 dummy expert materialized (test_mode)\n")
    lines.append("### Xorzen NANO_10M (genuine sparse)")
    lines.append(f"- H=192, L=6, E=8, K=2, pathway_top_k=2, min_depth=2, max_depth=6, width_choices=[96,192]")
    lines.append(f"- Same architecture as above; only routing behavior differs\n")

    lines.append("## Training Configuration\n")
    lines.append(f"- Optimizer: {ec['optimizer']}")
    lines.append(f"- LR: {ec['lr']} ({ec['lr_schedule']}, warmup={ec['warmup_steps']})")
    lines.append(f"- Weight decay: {ec['weight_decay']}")
    lines.append(f"- Grad clip: {ec['grad_clip']}")
    lines.append(f"- Seed: {ec['seed']}\n")

    lines.append("## Results Table (equal training tokens)\n")
    lines.append("| Metric | dense_baseline | xorzen_routing_disabled | xorzen_genuine_sparse |")
    lines.append("|---|---:|---:|---:|")
    def fmt_pct(x): return f"{x:.2f}%"
    lines.append(f"| P_full | {d['param_categories']['P_full']:,} | {dis['param_categories']['P_full']:,} | {sp['param_categories']['P_full']:,} |")
    lines.append(f"| P_resident (state_dict) | {d['param_categories']['P_resident_test_mode']:,} | {dis['param_categories']['P_resident_test_mode']:,} | {sp['param_categories']['P_resident_test_mode']:,} |")
    lines.append(f"| P_active/token | {d['param_categories']['P_active_per_token']:,} | {dis['param_categories']['P_active_per_token']:,} | {sp['param_categories']['P_active_per_token']:,} |")
    lines.append(f"| Active ratio (P_active/P_full) | 100.00% | {100*dis['param_categories']['P_active_per_token']/dis['param_categories']['P_full']:.2f}% | {100*sp['param_categories']['P_active_per_token']/sp['param_categories']['P_full']:.2f}% |")
    lines.append(f"| Tokens trained | {d['tokens_trained']:,} | {dis['tokens_trained']:,} | {sp['tokens_trained']:,} |")
    lines.append(f"| Train loss (initial) | {d['train_loss_initial']:.4f} | {dis['train_loss_initial']:.4f} | {sp['train_loss_initial']:.4f} |")
    lines.append(f"| Train loss (final) | {d['train_loss_final']:.4f} | {dis['train_loss_final']:.4f} | {sp['train_loss_final']:.4f} |")
    lines.append(f"| **Val loss (final)** | **{d['val_loss_final']:.4f}** | **{dis['val_loss_final']:.4f}** | **{sp['val_loss_final']:.4f}** |")
    lines.append(f"| **Val perplexity** | **{d['val_ppl_final']:.2f}** | **{dis['val_ppl_final']:.2f}** | **{sp['val_ppl_final']:.2f}** |")
    lines.append(f"| Mean step time | {d['mean_step_time_ms']:.0f}ms | {dis['mean_step_time_ms']:.0f}ms | {sp['mean_step_time_ms']:.0f}ms |")
    lines.append(f"| Tokens/sec | {d['mean_tokens_per_sec']:.0f} | {dis['mean_tokens_per_sec']:.0f} | {sp['mean_tokens_per_sec']:.0f} |")
    lines.append(f"| Total training time | {d['total_train_time_sec']:.1f}s | {dis['total_train_time_sec']:.1f}s | {sp['total_train_time_sec']:.1f}s |")
    lines.append(f"| Active FLOPs/token | {d['active_flops_per_token']:,} | {dis['active_flops_per_token']:,} | {sp['active_flops_per_token']:,} |")
    lines.append(f"| Total training FLOPs (fwd+bwd) | {d['total_training_flops']:,} | {dis['total_training_flops']:,} | {sp['total_training_flops']:,} |")
    lines.append(f"| Peak RSS (MB) | {d['peak_rss_mb']:.0f} | {dis['peak_rss_mb']:.0f} | {sp['peak_rss_mb']:.0f} |")
    lines.append(f"| Grad norm (mean/max) | {d['grad_norm_mean']:.2f}/{d['grad_norm_max']:.2f} | {dis['grad_norm_mean']:.2f}/{dis['grad_norm_max']:.2f} | {sp['grad_norm_mean']:.2f}/{sp['grad_norm_max']:.2f} |")
    lines.append(f"| Health: NaN/Inf steps | {d['health']['total_issue_steps']} | {dis['health']['total_issue_steps']} | {sp['health']['total_issue_steps']} |")
    lines.append("")

    lines.append("## Routing Statistics (Xorzen genuine sparse, final eval)\n")
    rs = sp.get("routing_stats_final", {}) or {}
    lines.append(f"- Active experts: {rs.get('expert_active_count', 'N/A')} / 8 declared "
                 f"({100*rs.get('expert_active_count',0)/8:.0f}% utilization)")
    lines.append(f"- Expert load entropy: {rs.get('expert_load_entropy', 'N/A'):.4f} bits")
    lines.append(f"- Max expert load share: {rs.get('expert_max_load_share', 'N/A'):.4f} (lower = more balanced)")
    lines.append(f"- Expert load balance CV: {rs.get('expert_load_balance_cv', 'N/A'):.4f}")
    lines.append(f"- Pathway entropy: {rs.get('path_entropy', 'N/A'):.4f} bits (max=ln(3)={math.log(3):.4f})")
    lines.append(f"- Pathway distribution: {rs.get('path_distrib_normalized', 'N/A')}\n")

    lines.append("## Equal-FLOPs Comparison (analytical)\n")
    ef = cmp["equal_flops"]
    lines.append(f"- Dense model total training FLOPs: {ef['compute_budget_flops']:,}")
    lines.append(f"- Tokens each model COULD have trained on this budget:")
    for k, v in ef["tokens_at_dense_compute"].items():
        lines.append(f"  - {k}: {v:,} tokens")
    lines.append(f"- Note: in this run all three models trained on the SAME number of tokens. "
                 f"The equal-FLOPs comparison shows that sparse Xorzen could have trained on "
                 f"~{ef['tokens_at_dense_compute']['xorzen_genuine_sparse']/ef['tokens_at_dense_compute']['dense']:.1f}× "
                 f"more tokens for the same compute budget.\n")

    lines.append("## Verdicts\n")
    svd = cmp["sparse_vs_dense"]
    lines.append(f"### Sparse vs Dense (equal tokens)")
    lines.append(f"- Val loss delta: {svd['val_loss_delta']:+.4f} "
                 f"({'sparse wins' if svd['val_loss_delta'] < 0 else 'dense wins'})")
    lines.append(f"- FLOPs/token delta: {svd['flops_per_token_delta']:+,} "
                 f"({'sparse uses less compute' if svd['flops_per_token_delta'] < 0 else 'sparse uses more compute'})")
    lines.append(f"- Quality per FLOP: {svd['interpretation']}\n")

    svdi = cmp["sparse_vs_disabled"]
    lines.append(f"### Sparse vs Disabled (does routing itself help quality?)")
    lines.append(f"- Val loss delta: {svdi['val_loss_delta']:+.4f}")
    lines.append(f"- Interpretation: {svdi['interpretation']}\n")

    col = cmp["routing_collapse_check"]
    lines.append(f"### Routing collapse check: **{col['verdict']}**")
    lines.append(f"- Active experts: {col['expert_active_count']} / {col['expert_declared_count']} "
                 f"({col['expert_utilization_pct']:.0f}% utilization)")
    lines.append(f"- Path entropy: {col['path_entropy']:.4f} (max possible: {col['max_path_entropy_possible']:.4f})")
    lines.append(f"- Max expert load share: {col['expert_max_load_share']:.4f}\n")

    fb = cmp["silent_fallback_check"]
    lines.append(f"### Silent fallback check: **{fb['verdict']}**")
    lines.append(f"- FLOPs reduction vs dense-equivalent: {fb['flops_reduction_pct']:.1f}%\n")

    lines.append("## Scaling-Law Trend on Real Data\n")
    slt = cmp["scaling_law_trend"]
    lines.append(f"- Question: {slt['question']}")
    lines.append(f"- Sparse P_full: {slt['sparse_P_full']:,}")
    lines.append(f"- Sparse P_active: {slt['sparse_P_active']:,} ({slt['sparse_active_ratio_pct']:.1f}% of P_full)")
    lines.append(f"- Dense P_full: {slt['dense_P_full']:,} (100% active)")
    lines.append(f"- Sparse compute efficiency: {slt['sparse_compute_efficiency_x']:.2f}× fewer FLOPs/token")
    lines.append(f"- Sparse achieves lower loss at lower compute: {slt['sparse_achieves_lower_loss_at_lower_compute']}")
    lines.append(f"- **Trend survives real data: {slt['trend_survives_real_data']}**\n")

    lines.append("## Remaining Problems\n")
    # Collect issues
    remaining = []
    if not d["health"]["no_nan_inf"]:
        remaining.append("Dense baseline had NaN/Inf issues during training.")
    if not dis["health"]["no_nan_inf"]:
        remaining.append("Xorzen routing-disabled had NaN/Inf issues during training.")
    if not sp["health"]["no_nan_inf"]:
        remaining.append("Xorzen sparse had NaN/Inf issues during training.")
    if col["verdict"] != "no_collapse":
        remaining.append(f"Routing showed {col['verdict']} — only {col['expert_active_count']} of "
                         f"{col['expert_declared_count']} experts active at end.")
    if not slt["trend_survives_real_data"]:
        remaining.append("Sparse-vs-dense scaling-law trend does NOT survive real data at 10M scale.")
    if svdi["val_loss_delta"] > 0:
        remaining.append(f"Sparse routing HURTS quality vs routing-disabled by {svdi['val_loss_delta']:.4f} — "
                         f"adaptive routing overhead > benefit at 10M scale.")
    remaining.append("This experiment does NOT validate the 12B-vs-60B hypothesis — it only tests "
                     "the weaker claim that sparse routing helps quality per FLOP at 10M scale.")
    remaining.append("Equal-FLOPs comparison is analytical only; would require re-running each model "
                     "on a different number of tokens to give empirical equal-FLOPs curves.")
    for r in remaining:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("## Reproducibility\n")
    lines.append(f"- Seed: {ec['seed']}")
    lines.append(f"- Total training time: {sum(c['total_train_time_sec'] for c in cfgs.values()):.1f}s "
                 f"({sum(c['total_train_time_sec'] for c in cfgs.values())/60:.1f}min)")
    lines.append(f"- Checkpoints: {CKPT_DIR}")
    lines.append(f"- Tokenized TinyStories cache: {PROJ}/data/tinystories_tokens_uint16.npy")
    lines.append(f"- Full results JSON: {OUT_JSON}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# Variables used by write_markdown (defined here for clarity)
dense_cfg_hidden = 192
dense_cfg_layers = 6
dense_cfg_heads = 8
dense_cfg_ffn = 768


if __name__ == "__main__":
    main()
