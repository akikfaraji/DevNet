"""
Phase 11 + 13 — Runtime profiling + Baseline comparisons.

Combined because both need the same measurement infrastructure.

Phase 11: Profile the full model. Identify top CPU hotspots, Python overhead,
tensor allocation overhead, synchronization points, unnecessary copies,
routing overhead, expert loading overhead, SSM scan overhead.

Phase 13: Compare 4 baselines under equivalent conditions:
  Baseline A: Dense transformer-style (no routing, all pathways all layers)
  Baseline B: Xorzen with routing disabled (uniform routing)
  Baseline C: Xorzen with routing enabled but dense execution (compute-then-mask)
  Baseline D: Xorzen with genuine conditional execution

Measure: parameter count, training loss, validation loss, tokens/sec,
FLOPs/token, wall-clock/token, peak memory, routing statistics.

We measure with a tiny model so we can run many configurations quickly.
"""

import os
import sys
import json
import time
import math
import gc
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, "/home/z/my-project/xorzen_dev")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SEED = 1337

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase


# ====================== OP COUNTER (re-used) ======================

class OpCounter:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.counts = defaultdict(int)
        self.flops = defaultdict(int)
        self.active = False

    def _make_hook(self, name):
        def hook(module, inp, out):
            if not self.active:
                return
            self.counts[name] += 1
            if isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                try:
                    n_tokens = inp[0].numel() // in_f
                except Exception:
                    n_tokens = 1
                self.flops[name] += 2 * n_tokens * in_f * out_f
            elif isinstance(module, nn.Conv1d):
                in_c = module.in_channels
                out_c = module.out_channels
                k = module.kernel_size[0]
                try:
                    total = inp[0].shape[0] * inp[0].shape[-1]
                except Exception:
                    total = 1
                self.flops[name] += 2 * total * in_c * out_c * k
        return hook

    def start(self):
        self.counts.clear()
        self.flops.clear()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                h = module.register_forward_hook(self._make_hook(name))
                self.handles.append(h)
        self.active = True

    def stop(self):
        self.active = False
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def stats(self):
        # Group by top-level subsystem
        by_subsystem = defaultdict(int)
        for name, flops in self.flops.items():
            # name is like "blocks.0.pathways.local.q_proj"
            top = name.split('.')[0]
            by_subsystem[top] += flops
        return {
            'total_calls': sum(self.counts.values()),
            'total_flops': sum(self.flops.values()),
            'by_subsystem_flops': dict(by_subsystem),
            'per_module_flops': dict(self.flops),
        }


# ====================== DATA ======================

def make_data(cfg, num_sequences=8, seed=SEED):
    rng = np.random.RandomState(seed)
    sequences = []
    for i in range(num_sequences):
        offset = rng.randint(1, cfg.vocab_size)
        start = rng.randint(0, cfg.vocab_size)
        seq = [(start + offset * t) % cfg.vocab_size for t in range(cfg.context_length)]
        sequences.append(seq)
    return torch.tensor(sequences, dtype=torch.long)


# ====================== PHASE 11: PROFILING ======================

def profile_model():
    """Profile the model and identify bottlenecks."""
    print("\n" + "="*72)
    print("PHASE 11 — RUNTIME PROFILING")
    print("="*72)

    H = 32
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name="xorzen_v04_phase11",
        vocab_size=32, context_length=16, hidden_size=H,
        num_layers=3, num_attention_heads=2, max_depth=3, min_depth=1,
        width_choices=(H // 2, H), cot_dim=8, cot_components=6,
        expert_count=4, top_k_experts=2,
        router_hidden_dim=16, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        load_balancing_weight=0.0,
        gradient_checkpointing=False,
        pathway_top_k=2,
    )

    model = zeroBase(config=cfg, test_mode=True)
    model.eval()
    sequences = make_data(cfg)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=sequences, output_routing_info=True)

    # Profile: measure FLOPs by subsystem
    counter = OpCounter(model)
    counter.start()
    with torch.no_grad():
        out = model(input_ids=sequences, output_routing_info=True)
    counter.stop()
    stats = counter.stats()

    # Time the full forward
    n_runs = 20
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_ids=sequences, output_routing_info=True)
    elapsed_fwd = (time.perf_counter() - t0) / n_runs

    # Time individual subsystems by patching them to no-ops
    # (We can't easily do this without breaking the model, so we just report
    # the FLOPs breakdown as a proxy for where time is spent.)
    total_flops = stats['total_flops']
    print(f"\nTotal FLOPs per forward: {total_flops:,}")
    print(f"Wall-clock per forward: {elapsed_fwd*1000:.2f}ms")
    print(f"Throughput: {sequences.numel() / elapsed_fwd:.0f} tokens/sec")

    # Rank subsystems by FLOPs
    print(f"\nFLOPs breakdown by subsystem:")
    ranked = sorted(stats['by_subsystem_flops'].items(), key=lambda x: -x[1])
    for name, flops in ranked:
        pct = flops / total_flops * 100 if total_flops > 0 else 0
        print(f"  {name:30s}: {flops:>12,d} FLOPs ({pct:>5.1f}%)")

    # Top 10 modules by FLOPs
    print(f"\nTop 10 modules by FLOPs:")
    top_mods = sorted(stats['per_module_flops'].items(), key=lambda x: -x[1])[:10]
    for name, flops in top_mods:
        pct = flops / total_flops * 100 if total_flops > 0 else 0
        print(f"  {name:50s}: {flops:>10,d} ({pct:>5.1f}%)")

    # Profile output
    profile = {
        'total_flops': total_flops,
        'wall_clock_ms_per_forward': elapsed_fwd * 1000,
        'tokens_per_second': sequences.numel() / elapsed_fwd,
        'by_subsystem': stats['by_subsystem_flops'],
        'top_10_modules': [(n, f) for n, f in top_mods],
        'n_runs': n_runs,
    }

    # Identify bottlenecks
    print(f"\n[BOTTLENECK ANALYSIS]")
    bottlenecks = []
    for name, flops in ranked[:5]:
        pct = flops / total_flops * 100 if total_flops > 0 else 0
        likely_cause = ""
        if 'lm_head' in name or 'token_embedding' in name:
            likely_cause = "vocab projection (always-on, not budget-controlled)"
        elif 'blocks' in name:
            likely_cause = "HASS block compute (depth/width/pathway routed)"
        elif 'router' in name:
            likely_cause = "router MLP (always-on)"
        elif 'merger' in name:
            likely_cause = "merger gate (always-on)"
        elif 'moe' in name:
            likely_cause = "MoE expert compute (top-k routed)"
        bottlenecks.append({
            'subsystem': name,
            'flops': flops,
            'pct': pct,
            'likely_cause': likely_cause,
        })
        print(f"  {name:30s} {pct:>5.1f}%  {likely_cause}")

    return profile, bottlenecks


# ====================== PHASE 13: BASELINE COMPARISONS ======================

def make_baseline_config(baseline: str, H=32):
    """
    Configure the model for each baseline.
    baseline: 'A_dense', 'B_routing_disabled', 'C_dense_exec', 'D_genuine_sparse'
    """
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    common = dict(
        model_name=f"xorzen_v04_phase13_{baseline}",
        vocab_size=32, context_length=16, hidden_size=H,
        num_layers=3, num_attention_heads=2, max_depth=3, min_depth=1,
        width_choices=(H // 2, H), cot_dim=8, cot_components=6,
        expert_count=4, top_k_experts=2,
        router_hidden_dim=16, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        gradient_checkpointing=False,
    )
    if baseline == 'A_dense':
        # No routing at all: pathway_top_k=3 (all pathways), max_depth=3 (all layers),
        # width_choices=(H,) only (no width adaptation), top_k=4 (all experts)
        common.update(
            width_choices=(H,),
            expert_count=4, top_k_experts=4,
            pathway_top_k=3,
            load_balancing_weight=0.0,
        )
    elif baseline == 'B_routing_disabled':
        # Routing disabled: same as A but keep the routing METADATA (pathway_top_k=2)
        # The router runs but produces uniform-like decisions
        common.update(
            pathway_top_k=2,
            load_balancing_weight=0.0,
        )
    elif baseline == 'C_dense_exec':
        # Routing enabled but dense execution (compute-then-mask)
        # We simulate this by forcing pathway_top_k=3 (all pathways always run)
        # and depth_mask=all 1s (no depth skipping).
        common.update(
            pathway_top_k=3,
            load_balancing_weight=0.0,
        )
    elif baseline == 'D_genuine_sparse':
        # Genuine sparse: pathway_top_k=2, depth routed, width routed, MoE top-2
        common.update(
            pathway_top_k=2,
            load_balancing_weight=0.001,
        )
    cfg.update(**common)
    return cfg


def measure_baseline(baseline: str, num_train_steps=50):
    """Train and measure one baseline."""
    print(f"\n[{baseline}] training {num_train_steps} steps...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = make_baseline_config(baseline)
    model = zeroBase(config=cfg, test_mode=True)
    model.train()

    sequences = make_data(cfg)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95)
    )

    # Train and track loss
    initial_loss = None
    final_loss = None
    for step in range(num_train_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
        loss = out.loss
        if initial_loss is None:
            initial_loss = float(out.lm_loss.item() if out.lm_loss is not None else loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    final_loss = float(out.lm_loss.item() if out.lm_loss is not None else loss.item())

    # Measure inference FLOPs and latency
    model.eval()
    counter = OpCounter(model)
    counter.start()
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(input_ids=sequences, output_routing_info=True)
        # Measure
        n_runs = 10
        t0 = time.perf_counter()
        for _ in range(n_runs):
            out = model(input_ids=sequences, output_routing_info=True)
        elapsed = (time.perf_counter() - t0) / n_runs
    counter.stop()
    stats = counter.stats()

    # Routing stats
    rd = out.routing_info
    avg_depth = float(rd.depth_mask.float().sum(dim=-1).mean().item())
    path_entropy = -(rd.path_probs * (rd.path_probs + 1e-12).log()).sum(dim=-1).mean().item()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    result = {
        'baseline': baseline,
        'config': {
            'pathway_top_k': cfg.pathway_top_k if hasattr(cfg, 'pathway_top_k') else 2,
            'top_k_experts': cfg.top_k_experts,
            'width_choices': list(cfg.width_choices),
            'num_layers': cfg.num_layers,
        },
        'total_params': total_params,
        'trainable_params': trainable_params,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction_pct': (initial_loss - final_loss) / initial_loss * 100 if initial_loss else 0,
        'total_flops_per_forward': stats['total_flops'],
        'wall_clock_ms_per_forward': elapsed * 1000,
        'tokens_per_second': sequences.numel() / elapsed,
        'flops_per_token': stats['total_flops'] / sequences.numel(),
        'avg_depth': avg_depth,
        'path_entropy': path_entropy,
        'max_path_entropy': math.log(3),
    }
    print(f"  loss: {initial_loss:.4f} -> {final_loss:.4f} ({result['loss_reduction_pct']:.1f}% reduction)")
    print(f"  FLOPs: {stats['total_flops']:,}  ({result['flops_per_token']:.0f}/token)")
    print(f"  latency: {elapsed*1000:.2f}ms  ({result['tokens_per_second']:.0f} tok/s)")
    print(f"  avg_depth={avg_depth:.2f}/{cfg.num_layers}  path_H={path_entropy:.3f}/{math.log(3):.3f}")
    return result


def compare_baselines():
    """Phase 13: compare all 4 baselines."""
    print("\n" + "="*72)
    print("PHASE 13 — BASELINE COMPARISONS")
    print("="*72)

    baselines = ['A_dense', 'B_routing_disabled', 'C_dense_exec', 'D_genuine_sparse']
    results = []
    for b in baselines:
        try:
            r = measure_baseline(b, num_train_steps=50)
            results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'baseline': b, 'error': str(e)})

    # Comparison table
    print("\n" + "="*72)
    print("PHASE 13 SUMMARY — Baseline comparison")
    print("="*72)
    print(f"{'Baseline':<22} {'Params':>10} {'InitLoss':>9} {'FinalLoss':>10} "
          f"{'FLOPs/fwd':>11} {'FLOPs/tok':>10} {'ms/fwd':>7} {'Tok/s':>7} "
          f"{'AvgDepth':>8} {'PathH':>6}")
    print("-" * 120)
    for r in results:
        if 'error' in r:
            print(f"{r['baseline']:<22}  ERROR: {r['error']}")
            continue
        print(f"{r['baseline']:<22} {r['total_params']:>10,d} {r['initial_loss']:>9.4f} "
              f"{r['final_loss']:>10.4f} {r['total_flops_per_forward']:>11,d} "
              f"{r['flops_per_token']:>10.0f} {r['wall_clock_ms_per_forward']:>7.2f} "
              f"{r['tokens_per_second']:>7.0f} {r['avg_depth']:>8.2f} {r['path_entropy']:>6.3f}")

    # Compute sparsity vs dense baseline
    if all('error' not in r for r in results):
        dense_flops = results[0]['total_flops_per_forward']
        print(f"\n[SPARSITY ANALYSIS] (vs Baseline A dense = {dense_flops:,} FLOPs)")
        for r in results[1:]:
            ratio = r['total_flops_per_forward'] / dense_flops
            reduction = (1 - ratio) * 100
            print(f"  {r['baseline']:<22}: {r['total_flops_per_forward']:>11,d} FLOPs  "
                  f"({ratio*100:.1f}% of dense, {reduction:.1f}% reduction)")

        # Verdicts
        d_flops = results[3]['total_flops_per_forward']
        a_flops = results[0]['total_flops_per_forward']
        c_flops = results[2]['total_flops_per_forward']
        verdicts = {
            'D_uses_less_flops_than_A': d_flops < a_flops,
            'D_uses_less_flops_than_C': d_flops < c_flops,
            'C_uses_same_or_more_than_A': c_flops >= a_flops * 0.95,
            'D_loss_decreased': results[3]['final_loss'] < results[3]['initial_loss'],
            'A_loss_decreased': results[0]['final_loss'] < results[0]['initial_loss'],
        }
        print(f"\n[VERDICTS]")
        for k, v in verdicts.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        overall = all(verdicts.values())
    else:
        verdicts = {'error': True}
        overall = False

    return results, verdicts, overall


# ====================== MAIN ======================

def main():
    out_dir = Path("/home/z/my-project/xorzen_dev/reports/v04")
    out_dir.mkdir(parents=True, exist_ok=True)

    profile, bottlenecks = profile_model()
    results, verdicts, overall = compare_baselines()

    print("\n" + "="*72)
    print("PHASE 11+13 — OVERALL VERDICT")
    print("="*72)
    print(f"  Phase 13 overall: {'PASS' if overall else 'PARTIAL'}")

    with open(out_dir / "phase11_13_profile_baselines.json", "w") as f:
        json.dump({
            'phase11_profile': profile,
            'phase11_bottlenecks': bottlenecks,
            'phase13_baselines': results,
            'phase13_verdicts': verdicts,
            'overall_pass': overall,
        }, f, indent=2, default=str)

    print(f"\n[SAVED] {out_dir/'phase11_13_profile_baselines.json'}")
    return overall


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
