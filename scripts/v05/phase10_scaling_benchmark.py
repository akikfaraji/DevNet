"""
Phase 10 (v0.5) — Quick scaling sweep + benchmark.

Sweep across NANO_1M and NANO_10M Xorzen variants to show the scaling trend
at fixed compute budget. Also benchmarks the routing diversity, FLOPs/token,
and memory across variants.

This complements phase9 (which did detailed training at 10M scale).

OUTPUTS:
  /home/z/my-project/xorzen_dev/reports/v05/phase10_scaling_benchmark.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

PROJ = "/home/z/my-project/xorzen_dev"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts", "v05"))

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase
from markov_data import MarkovCorpus

OUT = os.path.join(PROJ, "reports", "v05", "phase10_scaling_benchmark.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

VOCAB = 1000
SEQ = 64
BATCH = 4
SEED = 42


def measure_variant(model_size: ModelSize, train_steps: int) -> Dict[str, Any]:
    """Build variant, do a short training run, measure everything."""
    print(f"\n=== {model_size} ===")
    torch.manual_seed(SEED); np.random.seed(SEED)

    cfg = ConfigFactory.get_config(model_size)
    cfg.update(
        shard_experts=False, gradient_checkpointing=True,
        eval_routing_noise=0.15, pathway_top_k=2, min_depth=2,
        cost_aware_routing=True,
        context_length=SEQ, vocab_size=VOCAB, pad_token_id=0,
    )
    t0 = time.time()
    model = zeroBase(config=cfg, test_mode=True)
    build_time = time.time() - t0

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_total:,} total, {n_train:,} trainable")

    # Memory breakdown
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    # Build streaming corpus
    corpus = MarkovCorpus(vocab_size=VOCAB, num_clusters=50, seed=SEED)
    rng = np.random.RandomState(SEED)

    def make_batch():
        seqs = np.zeros((BATCH, SEQ + 1), dtype=np.int64)
        for i in range(BATCH):
            cur = rng.randint(0, VOCAB)
            seqs[i, 0] = cur
            for t in range(1, SEQ + 1):
                idx = rng.choice(corpus.top_k, p=corpus.next_probs[cur])
                cur = int(corpus.next_indices[cur, idx])
                seqs[i, t] = cur
        return torch.from_numpy(seqs)

    # Train
    losses = []
    step_times = []
    for step in range(train_steps):
        batch = make_batch()
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=inputs)
        logits = out.logits
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1), ignore_index=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step_times.append(time.time() - t0)
        losses.append(loss.item())
        if (step + 1) % 20 == 0:
            print(f"    step {step+1:3d}/{train_steps}  loss={loss.item():.4f}  "
                  f"{step_times[-1]*1000:.0f}ms/step")

    # Optimizer state memory
    opt_bytes = 0
    for s in opt.state.values():
        if isinstance(s, dict):
            for v in s.values():
                if torch.is_tensor(v):
                    opt_bytes += v.numel() * v.element_size()
        elif torch.is_tensor(s):
            opt_bytes += s.numel() * s.element_size()

    # Routing diversity (eval mode)
    model.eval()
    with torch.no_grad():
        sample = make_batch()
        out = model(input_ids=sample[:, :-1], output_routing_info=True)

    routing_summary = {}
    if hasattr(out, "routing_info") and out.routing_info is not None:
        rd = out.routing_info
        for attr in ["path_probs", "width_probs", "expert_probs", "depth_probs"]:
            if hasattr(rd, attr) and getattr(rd, attr) is not None:
                t = getattr(rd, attr)
                if torch.is_tensor(t) and t.numel() > 0:
                    # Path entropy
                    p = t.float()
                    p_norm = p / (p.sum(dim=-1, keepdim=True) + 1e-9)
                    entropy = -(p_norm * (p_norm + 1e-9).log()).sum(dim=-1).mean().item()
                    max_entropy = np.log(p.shape[-1])
                    routing_summary[attr] = {
                        "shape": list(p.shape),
                        "mean_entropy": float(entropy),
                        "max_entropy": float(max_entropy),
                        "entropy_ratio": float(entropy / max(max_entropy, 1e-9)),
                        "active_choices": int((p > 1e-3).sum(dim=-1).float().mean().item()),
                    }

    # Active params/token (from scaling law script computation)
    # We re-derive it here for self-containment
    active_per_token = compute_active_per_token(cfg)

    result = {
        "variant": str(model_size),
        "params_total": n_total,
        "params_trainable": n_train,
        "build_time_sec": round(build_time, 3),
        "memory": {
            "weights_mb": round(weights_bytes / 1024**2, 3),
            "buffers_mb": round(buffers_bytes / 1024**2, 3),
            "optimizer_state_mb": round(opt_bytes / 1024**2, 3),
            "total_static_mb": round((weights_bytes + buffers_bytes + opt_bytes) / 1024**2, 3),
        },
        "training": {
            "steps": train_steps,
            "loss_initial": float(losses[0]),
            "loss_final": float(losses[-1]),
            "loss_reduction_pct": round(100.0 * (1 - losses[-1] / max(losses[0], 1e-9)), 2),
            "mean_step_ms": round(float(np.mean(step_times) * 1000), 1),
            "mean_tokens_per_sec": round(BATCH * SEQ / float(np.mean(step_times)), 0),
            "total_train_time_sec": round(float(sum(step_times)), 2),
        },
        "active_per_token": int(active_per_token),
        "active_ratio_pct": round(100.0 * active_per_token / n_total, 2),
        "flops_per_token_active": int(6 * active_per_token),
        "flops_per_token_dense_equiv": int(6 * n_total),
        "compute_efficiency_x": round(6 * n_total / max(1, 6 * active_per_token), 3),
        "routing_summary": routing_summary,
        "arch": {
            "H": cfg.hidden_size,
            "L": cfg.num_layers,
            "V": cfg.vocab_size,
            "E": cfg.expert_count,
            "K": cfg.top_k_experts,
            "max_depth": cfg.max_depth,
            "min_depth": cfg.min_depth,
            "width_choices": list(cfg.width_choices),
            "expert_hidden_mult": cfg.expert_hidden_multiplier,
            "low_rank_dim": cfg.low_rank_dim,
            "ssm_state_dim": cfg.ssm_state_dim,
            "total_cot_dim": cfg.cot_dim * cfg.cot_components,
        },
    }
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}  ({result['training']['loss_reduction_pct']}%)")
    print(f"  {result['training']['mean_step_ms']}ms/step  {result['training']['mean_tokens_per_sec']:.0f} tok/s")
    print(f"  active_ratio: {result['active_ratio_pct']}%  compute_eff: {result['compute_efficiency_x']}x")

    del model, opt
    gc.collect()
    return result


def compute_active_per_token(cfg) -> int:
    """Same formula as scaling_law_derivation.py — see that file for derivation."""
    H = cfg.hidden_size; L = cfg.num_layers; V = cfg.vocab_size
    E = cfg.expert_count; K = cfg.top_k_experts; M = cfg.expert_hidden_multiplier
    D_lr = cfg.low_rank_dim; D_ssm = cfg.ssm_state_dim
    P_cot = cfg.cot_dim * cfg.cot_components
    widths = list(cfg.width_choices)
    Wmax = int(H * M)
    W_avg = sum(widths) / len(widths)
    width_factor = W_avg / Wmax
    L_avg = (cfg.max_depth + cfg.min_depth) / 2.0
    k_path = 2

    local_per = 4 * (H * H + H) + 2 * (H // max(1, (H // 64)))
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
    total_active = emb_active + layer_active + router_active + cot_active + moe_active + merger_active + final_active
    return int(total_active)


def main():
    print("=" * 70)
    print("  Phase 10 — Scaling sweep + benchmark")
    print("=" * 70)

    # Sweep across two scales (NANO_1M and NANO_10M) at same compute budget
    sweep_configs = [
        (ModelSize.NANO_1M,  100),   # 1M params, 100 steps
        (ModelSize.NANO_10M, 100),   # 10M params, 100 steps (same compute budget for optimizer)
    ]

    results = {
        "experiment_config": {
            "vocab_size": VOCAB, "seq_length": SEQ, "batch_size": BATCH,
            "seed": SEED, "variants": [str(s) for s, _ in sweep_configs],
        },
        "variants": {},
    }

    for ms, steps in sweep_configs:
        try:
            r = measure_variant(ms, steps)
            results["variants"][str(ms)] = r
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results["variants"][str(ms)] = {"error": f"{type(e).__name__}: {e}"}

    # Scaling-law cross-variant comparison
    print("\n" + "=" * 70)
    print("  SCALING SWEEP — COMPARISON")
    print("=" * 70)
    print(f"{'Variant':<22}{'P_total':>12}{'P_active':>12}{'active%':>9}{'eff_x':>7}{'loss_i':>9}{'loss_f':>9}{'ms/step':>9}")
    for k, r in results["variants"].items():
        if "error" in r: continue
        t = r["training"]
        print(f"{k:<22}{r['params_total']:>12,}{r['active_per_token']:>12,}"
              f"{r['active_ratio_pct']:>8.1f}%{r['compute_efficiency_x']:>6.2f}x"
              f"{t['loss_initial']:>9.4f}{t['loss_final']:>9.4f}"
              f"{t['mean_step_ms']:>8.0f}ms")

    # Save
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
