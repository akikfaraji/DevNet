"""
Phase 6 — Router Stability Experiments.

Phase 4 discovered pathway collapse: all 128 tokens pick SSM (pathway 2).
Phase 6 systematically tests whether the balancing losses prevent collapse
across 4 regimes:

  1. no_balancing     — all balancing/diversity losses = 0
  2. weak_balancing   — current weights * 0.1
  3. current_balancing — current weights (default)
  4. strong_balancing — current weights * 10

For each regime, train for 100 steps on the deterministic dataset and track:
  - pathway top-1 distribution (collapse = single value)
  - expert utilization (Gini, CV, dead experts)
  - depth utilization (avg depth, depth collapse)
  - width utilization (distribution over width choices)
  - routing entropy
  - loss curve

The goal is NOT to force perfect uniformity. The goal is to find the best
tradeoff between specialization and collapse prevention.
"""

import os
import sys
import json
import math
import copy
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/z/my-project/xorzen_dev")

import torch
import torch.nn as nn
import numpy as np

SEED = 1337

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase


def make_tiny_config():
    H = 32
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name="xorzen_v04_phase6",
        vocab_size=32, context_length=16, hidden_size=H,
        num_layers=3, num_attention_heads=2, max_depth=3, min_depth=1,
        width_choices=(H // 2, H), cot_dim=8, cot_components=6,
        expert_count=4, top_k_experts=2,
        router_hidden_dim=16, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        gradient_checkpointing=False,
        pathway_top_k=2,  # explicit
    )
    return cfg


def make_data(cfg, seed=SEED):
    rng = np.random.RandomState(seed)
    sequences = []
    for i in range(8):
        offset = rng.randint(1, cfg.vocab_size)
        start = rng.randint(0, cfg.vocab_size)
        seq = [(start + offset * t) % cfg.vocab_size for t in range(cfg.context_length)]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)


def gini(coeffs):
    """Gini coefficient of a 1-D numpy array (0=equal, 1=monopoly)."""
    a = np.sort(np.asarray(coeffs, dtype=np.float64))
    if a.sum() == 0:
        return 0.0
    n = len(a)
    cum = np.cumsum(a)
    return (n + 1 - 2 * (cum.sum() / a.sum())) / n


def cv(coeffs):
    """Coefficient of variation."""
    a = np.asarray(coeffs, dtype=np.float64)
    m = a.mean()
    if m == 0:
        return 0.0
    return a.std() / m


def set_balancing_regime(model, regime):
    """
    Patch the router's loss weights to implement the balancing regime.
    """
    router = model.router
    if regime == 'no_balancing':
        router.lb_loss_weight = 0.0
        router.z_loss_weight = 0.0
        router.path_div_weight = 0.0
        model.config.load_balancing_weight = 0.0
    elif regime == 'weak_balancing':
        router.lb_loss_weight = 0.00001
        router.z_loss_weight = 0.00001
        router.path_div_weight = 0.002
        model.config.load_balancing_weight = 0.0001
    elif regime == 'current_balancing':
        router.lb_loss_weight = 0.0001
        router.z_loss_weight = 0.0001
        router.path_div_weight = 0.02
        model.config.load_balancing_weight = 0.001
    elif regime == 'strong_balancing':
        router.lb_loss_weight = 0.001
        router.z_loss_weight = 0.001
        router.path_div_weight = 0.2
        model.config.load_balancing_weight = 0.01
    else:
        raise ValueError(f"unknown regime {regime}")


def routing_stats(model, routing_decision):
    """Compute routing stability metrics."""
    stats = {}

    # Depth
    depth_mask = routing_decision.depth_mask.float()
    stats['depth_avg'] = float(depth_mask.sum(dim=-1).mean().item())
    stats['depth_max'] = model.config.num_layers

    # Pathway
    path_probs = routing_decision.path_probs  # [B, T, 3]
    path_entropy = -(path_probs * (path_probs + 1e-12).log()).sum(dim=-1).mean()
    stats['path_entropy'] = float(path_entropy.item())
    stats['path_max_entropy'] = float(math.log(path_probs.shape[-1]))
    top1 = path_probs.argmax(dim=-1)
    unique, counts = torch.unique(top1, return_counts=True)
    path_dist = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
    stats['pathway_top1_dist'] = path_dist
    # Pathway collapse: Gini of counts
    if len(path_dist) > 0:
        path_counts_arr = np.array(list(path_dist.values()), dtype=np.float64)
        stats['path_gini'] = gini(path_counts_arr)
        stats['path_n_active_pathways'] = len(path_dist)
    else:
        stats['path_gini'] = 1.0
        stats['path_n_active_pathways'] = 0

    # Expert
    expert_indices = routing_decision.expert_indices  # [B, T, K]
    flat = expert_indices.reshape(-1).tolist()
    expert_dist = defaultdict(int)
    for e in flat:
        expert_dist[int(e)] += 1
    stats['expert_call_count'] = dict(expert_dist)
    stats['expert_n_called'] = len(expert_dist)
    stats['expert_n_total'] = model.config.expert_count
    stats['expert_dead_count'] = stats['expert_n_total'] - stats['expert_n_called']
    # Gini of expert usage
    if len(expert_dist) > 0:
        exp_counts = np.zeros(stats['expert_n_total'])
        for k, v in expert_dist.items():
            exp_counts[k] = v
        stats['expert_gini'] = gini(exp_counts)
        stats['expert_cv'] = cv(exp_counts)
    else:
        stats['expert_gini'] = 1.0
        stats['expert_cv'] = 0.0

    # Width
    width_idx = routing_decision.width_idx
    unique, counts = torch.unique(width_idx, return_counts=True)
    width_dist = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
    stats['width_idx_dist'] = width_dist
    if len(width_dist) > 0:
        w_counts = np.zeros(model.config.num_widths if hasattr(model.config, 'num_widths') else len(model.config.width_choices))
        for k, v in width_dist.items():
            w_counts[k] = v
        stats['width_gini'] = gini(w_counts)
    else:
        stats['width_gini'] = 1.0

    return stats


def train_one_regime(regime, num_steps=100, seed=SEED):
    """Train one regime for num_steps and return routing stats over time."""
    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = make_tiny_config()
    # The regime determines the load_balancing_weight config; we'll patch the
    # model router directly via set_balancing_regime after construction.
    model = zeroBase(config=cfg, test_mode=True)
    model.train()
    set_balancing_regime(model, regime)

    sequences = make_data(cfg, seed=seed)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95)
    )

    history = []
    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 10 == 0 or step == num_steps - 1:
            with torch.no_grad():
                lm_loss = float(out.lm_loss.item() if out.lm_loss is not None else loss.item())
                stats = routing_stats(model, out.routing_info)
                stats['step'] = step
                stats['lm_loss'] = lm_loss
                history.append(stats)

    return {
        'regime': regime,
        'history': history,
        'final': history[-1] if history else {},
        'initial': history[0] if history else {},
    }


def main():
    out_dir = Path("/home/z/my-project/xorzen_dev/reports/v04")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("PHASE 6 — ROUTER STABILITY EXPERIMENTS")
    print("="*72)

    regimes = ['no_balancing', 'weak_balancing', 'current_balancing', 'strong_balancing']
    results = {}

    for regime in regimes:
        print(f"\n[REGIME] {regime}")
        try:
            r = train_one_regime(regime, num_steps=100)
            results[regime] = r
            f = r['final']
            print(f"  initial loss={r['initial'].get('lm_loss', 0):.4f} -> final loss={f.get('lm_loss', 0):.4f}")
            print(f"  path: entropy={f.get('path_entropy', 0):.3f}/{f.get('path_max_entropy', 0):.3f}  "
                  f"n_active={f.get('path_n_active_pathways', 0)}  gini={f.get('path_gini', 0):.3f}")
            print(f"  expert: n_called={f.get('expert_n_called', 0)}/{f.get('expert_n_total', 0)}  "
                  f"dead={f.get('expert_dead_count', 0)}  gini={f.get('expert_gini', 0):.3f}  cv={f.get('expert_cv', 0):.3f}")
            print(f"  depth: avg={f.get('depth_avg', 0):.2f}/{f.get('depth_max', 0)}")
            print(f"  width dist: {f.get('width_idx_dist', {})}  gini={f.get('width_gini', 0):.3f}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results[regime] = {'error': str(e)}

    # Summary comparison table
    print("\n" + "="*72)
    print("PHASE 6 SUMMARY — routing stability comparison")
    print("="*72)
    print(f"{'Regime':<22} {'Loss':>7}  {'PathGini':>9} {'PathN':>6} {'ExpGini':>9} {'ExpDead':>8} {'Depth':>6} {'WidthGini':>10}")
    print("-" * 95)
    for regime in regimes:
        r = results.get(regime, {})
        if 'final' not in r:
            print(f"{regime:<22}  ERROR")
            continue
        f = r['final']
        print(f"{regime:<22} {f.get('lm_loss', 0):>7.4f}  "
              f"{f.get('path_gini', 0):>9.3f} {f.get('path_n_active_pathways', 0):>6d} "
              f"{f.get('expert_gini', 0):>9.3f} {f.get('expert_dead_count', 0):>8d} "
              f"{f.get('depth_avg', 0):>6.2f} {f.get('width_gini', 0):>10.3f}")

    # Verdicts
    print("\n" + "="*72)
    print("VERDICTS")
    print("="*72)
    verdicts = {}
    for regime in regimes:
        r = results.get(regime, {})
        if 'final' not in r:
            verdicts[regime] = {'error': True}
            continue
        f = r['final']
        i = r['initial']
        verdicts[regime] = {
            'loss_decreased': f.get('lm_loss', 1e9) < i.get('lm_loss', 1e9),
            'pathway_collapse': f.get('path_n_active_pathways', 0) == 1,
            'expert_collapse': f.get('expert_dead_count', 0) > 0,
            'depth_collapse': f.get('depth_avg', 0) < f.get('depth_max', 1) * 0.5,
            'path_gini': f.get('path_gini', 1.0),
            'expert_gini': f.get('expert_gini', 1.0),
        }

    for regime, v in verdicts.items():
        collapse = v.get('pathway_collapse', False) or v.get('expert_collapse', False) or v.get('depth_collapse', False)
        marker = "COLLAPSE" if collapse else ("OK" if v.get('loss_decreased', False) else "NO LEARN")
        print(f"  [{marker:9s}] {regime:<22}  path_gini={v.get('path_gini', 0):.3f}  "
              f"expert_gini={v.get('expert_gini', 0):.3f}  "
              f"path_collapse={v.get('pathway_collapse', False)}  "
              f"expert_collapse={v.get('expert_collapse', False)}")

    # Best regime recommendation
    print("\n[RECOMMENDATION]")
    candidates = []
    for regime, v in verdicts.items():
        if v.get('loss_decreased', False) and not v.get('pathway_collapse', False) and not v.get('expert_collapse', False):
            # Lower gini = better balance
            score = v.get('path_gini', 1.0) + v.get('expert_gini', 1.0)
            candidates.append((regime, score))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        best = candidates[0][0]
        print(f"  Best regime (no collapse + lowest gini): {best}")
    else:
        print(f"  No regime fully avoids collapse — need stronger diversity loss or different router design.")

    # Save
    with open(out_dir / "phase6_router_stability.json", "w") as f:
        # Make all values JSON-serializable
        def clean(o):
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            elif isinstance(o, list):
                return [clean(x) for x in o]
            elif isinstance(o, (np.integer,)):
                return int(o)
            elif isinstance(o, (np.floating,)):
                return float(o)
            elif isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
            return o
        json.dump({'results': clean(results), 'verdicts': clean(verdicts)}, f, indent=2)

    print(f"\n[SAVED] {out_dir/'phase6_router_stability.json'}")

    # Return overall success: at least one regime avoids collapse
    return any(
        v.get('loss_decreased', False) and not v.get('pathway_collapse', False) and not v.get('expert_collapse', False)
        for v in verdicts.values()
    )


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
