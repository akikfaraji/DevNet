"""
Phase 7 — Compute Budget Validation.

The ComputeController (xorzen.model.components.compute_controller) is supposed
to allocate compute across depth/width/pathway/experts based on a global
budget in [0, 1]. But it's not currently wired into zeroModel — the model
uses AdaptiveRouter directly.

This Phase 7 test wraps the ComputeController as a drop-in router replacement
to measure whether the budget actually controls compute. For each budget in
[0.10, 0.25, 0.50, 0.75, 1.00] we measure:

  - predicted_compute (from controller.actual_compute)
  - actual_compute (OpCounter FLOPs)
  - avg_depth
  - avg_width
  - avg_pathway_count (top-k pathway selection)
  - avg_expert_count (always = top_k, fixed)
  - wall_clock_runtime
  - peak_memory

We then build the table:
  requested_budget -> predicted_compute -> actual_compute -> runtime

And investigate mismatches.
"""

import os
import sys
import json
import time
import math
import gc
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/home/z/my-project/xorzen_dev")

import torch
import torch.nn as nn
import numpy as np

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase
from xorzen.model.components.compute_controller import ComputeController, ComputeAllocation
from xorzen.model.components.routing import RoutingDecision


# ====================== OP COUNTER (from Phase 5) ======================

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
                    n_positions = inp[0].shape[-1]
                    n_batch = inp[0].shape[0]
                    total = n_batch * n_positions
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
        return {
            'total_calls': sum(self.counts.values()),
            'total_flops': sum(self.flops.values()),
        }


# ====================== UTIL ======================

def make_config():
    H = 32
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name="xorzen_v04_phase7",
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
    return cfg


def allocation_to_routing_decision(alloc: ComputeAllocation, cfg, B, T) -> RoutingDecision:
    """Convert ComputeAllocation to RoutingDecision for model consumption."""
    H = cfg.hidden_size
    D = cfg.max_depth
    W = len(cfg.width_choices)
    P = 3
    E = cfg.expert_count
    K = cfg.top_k_experts

    return RoutingDecision(
        depth_logits=torch.zeros(B, T, D),
        depth_probs=alloc.depth_mask.float(),
        depth_mask=alloc.depth_mask,
        width_logits=torch.zeros(B, T, W),
        width_probs=alloc.width_probs,
        width_idx=alloc.width_idx,
        width_multiplier=torch.ones(B, T, 1),  # constant; width_idx drives SlicedFFN
        path_logits=torch.zeros(B, T, P),
        path_probs=alloc.path_probs,
        expert_logits=torch.zeros(B, T, E),
        expert_probs=torch.zeros(B, T, E),
        expert_indices=alloc.expert_indices,
        expert_weights=alloc.expert_weights,
        complexity=torch.ones(B, T, 1),
        uncertainty=torch.zeros(B, T, 1),
        auxiliary={},
    )


def measure_budget(model, cfg, compute_controller, budget, B=2, T=16):
    """
    Run a forward pass with a specific compute_budget and measure everything.
    """
    torch.manual_seed(SEED)
    x = torch.randint(0, cfg.vocab_size, (B, T))

    # Patch the model's router to use the compute controller
    orig_router_forward = model.router.forward

    def make_fake_forward(b, holder):
        def fake_forward(*args, **kwargs):
            x_inner = kwargs.get('x', args[0] if len(args) > 0 else None)
            if x_inner is None:
                return None
            alloc = compute_controller(x_inner, compute_budget=b, training=False, deterministic=True)
            holder['alloc'] = alloc  # save for later inspection
            rd = allocation_to_routing_decision(alloc, cfg, B, T)
            return rd
        return fake_forward

    alloc_holder = {}
    model.router.forward = make_fake_forward(budget, alloc_holder)
    model.eval()

    # Warmup
    try:
        with torch.no_grad():
            _ = model(input_ids=x, output_routing_info=True)
    except Exception as e:
        model.router.forward = orig_router_forward
        return {'budget': budget, 'error': f'warmup failed: {e}'}

    # Measure
    counter = OpCounter(model)
    counter.start()
    try:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        # Track peak memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=x, output_routing_info=True)
        elapsed = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    finally:
        counter.stop()
        model.router.forward = orig_router_forward

    stats = counter.stats()
    rd = out.routing_info

    # Compute actual pathway count per token (top-k pathways selected)
    # For top_k=2, each token selects exactly 2 pathways. So avg_pathway_count = 2.
    # But if budget is low, top_k might effectively reduce. For now, pathway_top_k
    # is fixed at config.pathway_top_k.
    avg_pathway_count = float(cfg.pathway_top_k if hasattr(cfg, 'pathway_top_k') else 2)

    # Avg depth
    avg_depth = float(rd.depth_mask.float().sum(dim=-1).mean().item())

    # Avg width multiplier
    width_values = list(cfg.width_choices)
    width_idx = rd.width_idx
    avg_width_idx = float(width_idx.float().mean().item())
    avg_width_value = float(sum(width_values[int(i)] for i in width_idx.reshape(-1).tolist()) / (B * T))
    max_width = max(width_values)
    avg_width_fraction = avg_width_value / max_width

    # Predicted compute from the controller
    alloc = alloc_holder.get('alloc')
    predicted_compute = float(alloc.actual_compute.mean().item()) if alloc is not None else 0.0

    return {
        'budget': budget,
        'predicted_compute': predicted_compute,
        'actual_flops': stats['total_flops'],
        'actual_linear_calls': stats['total_calls'],
        'avg_depth': avg_depth,
        'avg_depth_fraction': avg_depth / cfg.max_depth,
        'avg_width_value': avg_width_value,
        'avg_width_fraction': avg_width_fraction,
        'avg_width_idx': avg_width_idx,
        'avg_pathway_count': avg_pathway_count,
        'avg_expert_count': float(cfg.top_k_experts),
        'wall_clock_ms': elapsed * 1000,
        'peak_memory_bytes': peak_mem,
        'output_finite': bool(torch.isfinite(out.logits).all().item()),
    }


def main():
    out_dir = Path("/home/z/my-project/xorzen_dev/reports/v04")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("PHASE 7 — COMPUTE BUDGET VALIDATION")
    print("="*72)

    cfg = make_config()
    model = zeroBase(config=cfg, test_mode=True)
    model.eval()

    # Build a ComputeController that matches the model's dimensions
    cc = ComputeController(
        hidden_dim=cfg.hidden_size,
        max_depth=cfg.max_depth,
        width_choices=cfg.width_choices,
        num_paths=3,
        num_experts=cfg.expert_count,
        top_k=cfg.top_k_experts,
        router_hidden_dim=16,
    )
    cc.eval()

    # Set pathway_top_k on the controller so it knows how many pathways to use
    cc._pathway_top_k = cfg.pathway_top_k if hasattr(cfg, 'pathway_top_k') else 2

    budgets = [0.10, 0.25, 0.50, 0.75, 1.00]
    results = []

    for budget in budgets:
        print(f"\n[MEASURE] budget={budget:.2f}")
        try:
            r = measure_budget(model, cfg, cc, budget)
            results.append(r)
            if 'error' in r:
                print(f"  [ERROR] {r['error']}")
            else:
                print(f"  predicted_compute = {r['predicted_compute']:.4f}")
                print(f"  actual_flops      = {r['actual_flops']:,}")
                print(f"  avg_depth         = {r['avg_depth']:.2f}/{cfg.max_depth}  ({r['avg_depth_fraction']*100:.0f}%)")
                print(f"  avg_width         = {r['avg_width_value']:.1f}  ({r['avg_width_fraction']*100:.0f}% of max)")
                print(f"  avg_pathway_count = {r['avg_pathway_count']}")
                print(f"  avg_expert_count  = {r['avg_expert_count']}")
                print(f"  wall_clock_ms     = {r['wall_clock_ms']:.2f}")
                print(f"  output_finite     = {r['output_finite']}")
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results.append({'budget': budget, 'error': str(e)})

    # Build the requested_budget -> predicted_compute -> actual_compute -> runtime table
    print("\n" + "="*72)
    print("PHASE 7 TABLE — requested_budget vs predicted vs actual vs runtime")
    print("="*72)
    print(f"{'Budget':>7}  {'PredComp':>9}  {'ActFLOPs':>11}  {'ActCalls':>9}  "
          f"{'AvgDepth':>8}  {'AvgWidth':>8}  {'RuntimeMs':>10}  {'Finite':>6}")
    print("-" * 95)
    for r in results:
        if 'error' in r:
            print(f"{r['budget']:>7.2f}  ERROR: {r['error']}")
            continue
        print(f"{r['budget']:>7.2f}  {r['predicted_compute']:>9.4f}  {r['actual_flops']:>11,d}  "
              f"{r['actual_linear_calls']:>9d}  {r['avg_depth']:>8.2f}  "
              f"{r['avg_width_fraction']*100:>7.1f}%  {r['wall_clock_ms']:>10.2f}  "
              f"{str(r['output_finite']):>6}")

    # Investigate mismatches
    print("\n[ANALYSIS]")
    if len(results) >= 2 and all('error' not in r for r in results):
        flops_low = results[0]['actual_flops']
        flops_high = results[-1]['actual_flops']
        if flops_low == flops_high:
            print(f"  WARNING: actual FLOPs identical for budget {results[0]['budget']} and {results[-1]['budget']}")
            print(f"           ComputeController is NOT actually controlling compute.")
        else:
            ratio = flops_low / flops_high
            print(f"  FLOPs at budget={results[0]['budget']}: {flops_low:,}")
            print(f"  FLOPs at budget={results[-1]['budget']}: {flops_high:,}")
            print(f"  Ratio: {ratio:.3f} (expected ~{results[0]['budget']/results[-1]['budget']:.3f})")

        # Predicted vs actual compute correlation
        preds = [r['predicted_compute'] for r in results]
        acts = [r['actual_flops'] for r in results]
        # Simple check: monotonic increase
        is_monotonic = all(acts[i] <= acts[i+1] + 1e-6 for i in range(len(acts)-1))
        print(f"  Actual compute is monotonically increasing in budget: {is_monotonic}")

        # Verdicts
        verdicts = {
            'budget_affects_compute': flops_low != flops_high,
            'actual_monotonic_in_budget': is_monotonic,
            'predicted_correlates_with_actual': all(
                (preds[i] - preds[i+1]) * (acts[i] - acts[i+1]) >= -1e-6
                for i in range(len(preds)-1)
            ),
            'output_finite_at_all_budgets': all(r['output_finite'] for r in results),
        }
        print("\n[VERDICTS]")
        for k, v in verdicts.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")

        overall_pass = all(verdicts.values())
    else:
        verdicts = {'error': True}
        overall_pass = False

    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL/PARTIAL'}")

    # Save
    with open(out_dir / "phase7_compute_budget.json", "w") as f:
        json.dump({
            'results': results,
            'verdicts': verdicts,
            'overall_pass': overall_pass,
        }, f, indent=2, default=str)

    print(f"\n[SAVED] {out_dir/'phase7_compute_budget.json'}")
    return overall_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
