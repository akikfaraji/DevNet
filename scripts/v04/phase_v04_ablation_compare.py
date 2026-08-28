"""
v0.4 architecture comparison: OLD (AdaptiveFFN, no width_div, path_div=0.1,
double LB losses, no cost-aware) vs NEW (SlicedFFN, width_div, path_div=0.2,
unified LB loss, cost-aware routing).

For each variant, run the Phase 4 overfit test and the Phase 14 ablation
suite, then produce a comparison table.

This script answers: did the v0.4 architecture changes actually improve
anything measurable?
"""

import os
import sys
import json
import math
import time
from pathlib import Path
from collections import defaultdict

# Determinism FIRST
SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, "/home/z/my-project/xorzen_dev")

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase


def make_base_config(variant: str) -> "ModelConfig":
    """variant in {'old', 'new'}."""
    H = 32
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name=f"xorzen_v04_compare_{variant}",
        vocab_size=32, context_length=16, hidden_size=H,
        num_layers=3, num_attention_heads=2, max_depth=3, min_depth=1,
        width_choices=(H // 2, H), cot_dim=8, cot_components=6,
        expert_count=4, top_k_experts=2,
        router_hidden_dim=16, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        load_balancing_weight=0.001,
        gradient_checkpointing=False,
        pathway_top_k=2,
    )
    if variant == 'old':
        # v0.3/v0.4-old behavior
        cfg.update(
            use_sliced_ffn=False,
            width_div_weight=0.0,
            path_div_weight=0.1,
            unify_load_balance=False,
            cost_aware_routing=False,
        )
    else:
        # v0.4-new behavior (defaults from config.py)
        cfg.update(
            use_sliced_ffn=True,
            width_div_weight=0.1,
            path_div_weight=0.2,
            unify_load_balance=True,
            cost_aware_routing=True,
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


def token_accuracy(logits, labels):
    pred = logits[:, :-1, :].argmax(dim=-1)
    target = labels[:, 1:]
    return float((pred == target).float().mean().item())


class OpCounter:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.flops = defaultdict(int)
        self.active = False

    def _make_hook(self, name):
        def hook(module, inp, out):
            if not self.active:
                return
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

    def total_flops(self):
        return sum(self.flops.values())


def train_and_measure(model, cfg, sequences, num_steps=100):
    """Train and measure loss + FLOPs + routing diversity."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95)
    )

    initial_loss = None
    final_loss = None
    final_acc = 0.0
    final_path_entropy = 0.0
    final_width_dist = {}
    final_pathway_top1 = {}
    final_expert_calls = {}
    final_avg_depth = 0.0

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
        if initial_loss is None:
            initial_loss = float(out.lm_loss.item() if out.lm_loss is not None else out.loss.item())
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Final measurements (eval mode)
    model.eval()
    with torch.no_grad():
        out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
        final_loss = float(out.lm_loss.item() if out.lm_loss is not None else out.loss.item())
        final_acc = token_accuracy(out.logits, sequences)
        rd = out.routing_info
        # Path diversity
        path_probs = rd.path_probs
        path_entropy = -(path_probs * (path_probs + 1e-12).log()).sum(dim=-1).mean()
        final_path_entropy = float(path_entropy.item())
        max_path_entropy = float(math.log(path_probs.shape[-1]))
        top1 = path_probs.argmax(dim=-1)
        unique, counts = torch.unique(top1, return_counts=True)
        final_pathway_top1 = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
        # Width diversity
        unique, counts = torch.unique(rd.width_idx, return_counts=True)
        final_width_dist = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
        # Expert utilization
        flat = rd.expert_indices.reshape(-1).tolist()
        for e in flat:
            final_expert_calls[int(e)] = final_expert_calls.get(int(e), 0) + 1
        # Avg depth
        depth_mask = rd.depth_mask.float()
        final_avg_depth = float(depth_mask.sum(dim=-1).mean().item())

    # Measure FLOPs
    counter = OpCounter(model)
    counter.start()
    with torch.no_grad():
        _ = model(input_ids=sequences, output_routing_info=True)
    counter.stop()
    total_flops = counter.total_flops()

    # Measure runtime (10 forward passes, take median)
    model.eval()
    times = []
    with torch.no_grad():
        # warmup
        for _ in range(3):
            _ = model(input_ids=sequences, output_routing_info=True)
        for _ in range(10):
            t0 = time.perf_counter()
            _ = model(input_ids=sequences, output_routing_info=True)
            times.append(time.perf_counter() - t0)
    median_runtime_ms = (sorted(times)[len(times) // 2] * 1000)

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction_pct': (initial_loss - final_loss) / initial_loss * 100 if initial_loss else 0,
        'final_accuracy': final_acc,
        'total_flops': total_flops,
        'total_params': sum(p.numel() for p in model.parameters()),
        'median_runtime_ms': median_runtime_ms,
        'avg_depth': final_avg_depth,
        'max_depth': cfg.max_depth,
        'path_entropy': final_path_entropy,
        'max_path_entropy': max_path_entropy,
        'pathway_top1_dist': final_pathway_top1,
        'num_active_pathways': len(final_pathway_top1),
        'width_dist': final_width_dist,
        'num_active_widths': len(final_width_dist),
        'expert_call_count': final_expert_calls,
        'num_active_experts': len(final_expert_calls),
    }


def main():
    out_dir = Path("/home/z/my-project/xorzen_dev/reports/v04")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("v0.4 ARCHITECTURE COMPARISON: OLD vs NEW")
    print("=" * 78)

    results = {}
    for variant in ['old', 'new']:
        print(f"\n[VARIANT] {variant}")
        cfg = make_base_config(variant)
        print(f"  use_sliced_ffn={cfg.use_sliced_ffn}")
        print(f"  width_div_weight={cfg.width_div_weight}")
        print(f"  path_div_weight={cfg.path_div_weight}")
        print(f"  unify_load_balance={cfg.unify_load_balance}")
        print(f"  cost_aware_routing={cfg.cost_aware_routing}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = zeroBase(config=cfg, test_mode=True)
        sequences = make_data(cfg)
        r = train_and_measure(model, cfg, sequences, num_steps=200)
        r['variant'] = variant
        results[variant] = r
        print(f"  loss: {r['initial_loss']:.4f} -> {r['final_loss']:.4f} ({r['loss_reduction_pct']:.1f}%)")
        print(f"  accuracy: {r['final_accuracy']*100:.2f}%")
        print(f"  FLOPs: {r['total_flops']:,}")
        print(f"  params: {r['total_params']:,}")
        print(f"  runtime: {r['median_runtime_ms']:.2f}ms")
        print(f"  avg_depth: {r['avg_depth']:.2f}/{r['max_depth']}")
        print(f"  path_entropy: {r['path_entropy']:.3f}/{r['max_path_entropy']:.3f}")
        print(f"  num_active_pathways: {r['num_active_pathways']}/3")
        print(f"  num_active_widths: {r['num_active_widths']}/{len(cfg.width_choices)}")
        print(f"  num_active_experts: {r['num_active_experts']}/{cfg.expert_count}")
        del model

    # Compute deltas
    old = results['old']
    new = results['new']
    deltas = {
        'loss_reduction_delta_pct': new['loss_reduction_pct'] - old['loss_reduction_pct'],
        'final_loss_delta': new['final_loss'] - old['final_loss'],
        'accuracy_delta': new['final_accuracy'] - old['final_accuracy'],
        'flops_delta_pct': (new['total_flops'] - old['total_flops']) / old['total_flops'] * 100,
        'runtime_delta_pct': (new['median_runtime_ms'] - old['median_runtime_ms']) / old['median_runtime_ms'] * 100,
        'path_entropy_delta': new['path_entropy'] - old['path_entropy'],
        'num_active_pathways_delta': new['num_active_pathways'] - old['num_active_pathways'],
        'num_active_widths_delta': new['num_active_widths'] - old['num_active_widths'],
    }

    print("\n" + "=" * 78)
    print("COMPARISON SUMMARY")
    print("=" * 78)
    print(f"{'Metric':<35} {'OLD':>15} {'NEW':>15} {'DELTA':>15}")
    print("-" * 80)
    print(f"{'Final loss':<35} {old['final_loss']:>15.4f} {new['final_loss']:>15.4f} {deltas['final_loss_delta']:>+15.4f}")
    print(f"{'Loss reduction %':<35} {old['loss_reduction_pct']:>15.1f} {new['loss_reduction_pct']:>15.1f} {deltas['loss_reduction_delta_pct']:>+15.1f}")
    print(f"{'Token accuracy':<35} {old['final_accuracy']*100:>14.2f} {new['final_accuracy']*100:>14.2f} {deltas['accuracy_delta']*100:>+14.2f}")
    print(f"{'FLOPs':<35} {old['total_flops']:>15,d} {new['total_flops']:>15,d} {deltas['flops_delta_pct']:>+14.1f}%")
    print(f"{'Runtime (ms)':<35} {old['median_runtime_ms']:>15.2f} {new['median_runtime_ms']:>15.2f} {deltas['runtime_delta_pct']:>+14.1f}%")
    print(f"{'Path entropy':<35} {old['path_entropy']:>15.3f} {new['path_entropy']:>15.3f} {deltas['path_entropy_delta']:>+15.3f}")
    print(f"{'Active pathways (of 3)':<35} {old['num_active_pathways']:>15d} {new['num_active_pathways']:>15d} {deltas['num_active_pathways_delta']:>+15d}")
    print(f"{'Active widths':<35} {old['num_active_widths']:>15d} {new['num_active_widths']:>15d} {deltas['num_active_widths_delta']:>+15d}")
    print(f"{'Params':<35} {old['total_params']:>15,d} {new['total_params']:>15,d} {new['total_params']-old['total_params']:>+15,d}")

    print(f"\n[OLD] pathway top-1 dist: {old['pathway_top1_dist']}")
    print(f"[NEW] pathway top-1 dist: {new['pathway_top1_dist']}")
    print(f"[OLD] width dist: {old['width_dist']}")
    print(f"[NEW] width dist: {new['width_dist']}")
    print(f"[OLD] expert calls: {old['expert_call_count']}")
    print(f"[NEW] expert calls: {new['expert_call_count']}")

    # Save
    output = {
        'description': 'v0.4 architecture comparison: old vs new (200 training steps)',
        'config_old': {
            'use_sliced_ffn': False, 'width_div_weight': 0.0,
            'path_div_weight': 0.1, 'unify_load_balance': False,
            'cost_aware_routing': False,
        },
        'config_new': {
            'use_sliced_ffn': True, 'width_div_weight': 0.1,
            'path_div_weight': 0.2, 'unify_load_balance': True,
            'cost_aware_routing': True,
        },
        'results': results,
        'deltas': deltas,
    }
    out_path = out_dir / "phase_v04_old_vs_new.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
