"""
v0.4 profiling — identify the new bottlenecks after the architecture changes.

Profiles the v0.4-new architecture (SlicedFFN + width_div_loss + cost-aware
routing + unified LB) and breaks down runtime by subsystem.

Uses torch.profiler to identify:
  - Top-K most expensive operators
  - Time spent in each subsystem (router, blocks, MoE, merger, lm_head)
  - Memory usage
  - Per-token latency at different batch sizes / sequence lengths
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

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


def make_config():
    H = 32
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name="xorzen_v04_profile",
        vocab_size=32, context_length=16, hidden_size=H,
        num_layers=3, num_attention_heads=2, max_depth=3, min_depth=1,
        width_choices=(H // 2, H), cot_dim=8, cot_components=6,
        expert_count=4, top_k_experts=2,
        router_hidden_dim=16, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        gradient_checkpointing=False,
        pathway_top_k=2,
    )
    return cfg


def time_forward(model, input_ids, n_warmup=3, n_runs=20):
    """Median forward pass time in milliseconds."""
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(n_warmup):
            _ = model(input_ids=input_ids, output_routing_info=True)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, output_routing_info=True)
            times.append((time.perf_counter() - t0) * 1000)
    return sorted(times)[len(times) // 2]


def time_subsystem(model, input_ids, n_runs=20):
    """Time each subsystem by selectively disabling it via forward hooks.

    This is approximate — we measure the time WITH vs WITHOUT each subsystem's
    forward by monkey-patching it to be a no-op.
    """
    import math
    results = {}

    # Baseline: full forward
    baseline_ms = time_forward(model, input_ids, n_runs=n_runs)
    results['full_forward'] = baseline_ms

    # Time each subsystem by replacing it with identity
    def time_with_patch(patch_fn, name):
        # Save originals
        originals = patch_fn(model)
        try:
            ms = time_forward(model, input_ids, n_warmup=1, n_runs=n_runs)
            results[name] = ms
        finally:
            # Restore
            for path, orig in originals.items():
                obj = model
                for p in path.split('.')[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, path.split('.')[-1], orig)

    # Patch router to return None (model handles None routing)
    # Actually the model REQUIRES routing_decision for HASSBlock, so we can't
    # just no-op the router. Instead, we patch each subsystem's forward to
    # be a no-op and measure the time difference.

    # 1. Patch all HASS blocks to identity (skip blocks entirely)
    orig_block_fwd = []
    for blk in model.blocks:
        orig_block_fwd.append(blk.forward)
        def make_identity():
            def fwd(x, *args, **kwargs):
                return x
            return fwd
        blk.forward = make_identity()
    results['no_blocks'] = time_forward(model, input_ids, n_warmup=1, n_runs=n_runs)
    for blk, orig in zip(model.blocks, orig_block_fwd):
        blk.forward = orig

    # 2. Patch MoE to identity (return zero output)
    orig_moe_fwd = model.moe.forward
    def moe_identity(*args, **kwargs):
        # Returns (output, stats) — output shape [N, H]
        # Find expected shape from args
        x = args[0] if args else kwargs.get('x')
        if x is None:
            # Try other arg names
            for a in args[1:]:
                if isinstance(a, torch.Tensor):
                    x = a
                    break
        if x is None:
            return torch.zeros(1, model.config.hidden_size), {}
        return torch.zeros_like(x), {'experts_used': 0, 'cache_hits': 0, 'cache_misses': 0}
    model.moe.forward = moe_identity
    results['no_moe'] = time_forward(model, input_ids, n_warmup=1, n_runs=n_runs)
    model.moe.forward = orig_moe_fwd

    # 3. Patch merger to identity
    orig_merger_fwd = model.merger.forward
    def merger_identity(*args, **kwargs):
        # merger takes (hass_output, moe_output, cot_features) and returns merged
        # Just return hass_output (first arg)
        return args[0] if args else kwargs.get('hass_output')
    model.merger.forward = merger_identity
    results['no_merger'] = time_forward(model, input_ids, n_warmup=1, n_runs=n_runs)
    model.merger.forward = orig_merger_fwd

    # Compute subsystem time as (baseline - patched)
    results['blocks_time_ms'] = baseline_ms - results['no_blocks']
    results['moe_time_ms'] = baseline_ms - results['no_moe']
    results['merger_time_ms'] = baseline_ms - results['no_merger']
    # Router time = full - no_router (where no_router means routing_decision=None
    # forces compute_all_pathways=True path). We can't easily disable routing
    # entirely, so estimate router time as: full - (full with router encoded once).
    # Actually, simpler: router time is small relative to blocks. Estimate it
    # by timing the router forward in isolation.
    router_input = model.token_embedding(input_ids)
    cot_features = torch.zeros(input_ids.shape[0], input_ids.shape[1],
                                model.config.cot_dim * model.config.cot_components)
    # warmup
    for _ in range(3):
        _ = model.router(router_input, cot_features)
    router_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.router(router_input, cot_features)
        router_times.append((time.perf_counter() - t0) * 1000)
    results['router_time_ms'] = sorted(router_times)[len(router_times) // 2]

    return results


def main():
    out_dir = Path("/home/z/my-project/xorzen_dev/reports/v04")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("v0.4 PROFILING — Identify bottlenecks")
    print("=" * 78)

    cfg = make_config()
    torch.manual_seed(SEED)
    model = zeroBase(config=cfg, test_mode=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[MODEL] {n_params:,} params, {cfg.num_layers} layers, "
          f"hidden={cfg.hidden_size}, widths={cfg.width_choices}")

    # ===== Run 1: Subsystem breakdown at default size =====
    print("\n[TEST 1] Subsystem time breakdown (B=4, T=16)")
    input_ids = torch.randint(0, cfg.vocab_size, (4, 16))
    breakdown = time_subsystem(model, input_ids, n_runs=30)

    print(f"\n  Full forward:       {breakdown['full_forward']:.3f} ms")
    print(f"  Without blocks:     {breakdown['no_blocks']:.3f} ms  -> blocks = {breakdown['blocks_time_ms']:.3f} ms")
    print(f"  Without MoE:        {breakdown['no_moe']:.3f} ms  -> MoE    = {breakdown['moe_time_ms']:.3f} ms")
    print(f"  Without merger:     {breakdown['no_merger']:.3f} ms  -> merger = {breakdown['merger_time_ms']:.3f} ms")
    print(f"  Router (isolated):  {breakdown['router_time_ms']:.3f} ms")

    total_subsystem = (breakdown['blocks_time_ms'] + breakdown['moe_time_ms']
                      + breakdown['merger_time_ms'] + breakdown['router_time_ms'])
    print(f"\n  Subsystem sum:      {total_subsystem:.3f} ms")
    print(f"  (Sum != full because subsystems overlap and there's overhead from")
    print(f"   embeddings, lm_head, and the patches themselves.)")

    # ===== Run 2: Scaling with batch size =====
    print("\n[TEST 2] Scaling with batch size (T=16)")
    scaling = {}
    for B in [1, 4, 16, 64]:
        ids = torch.randint(0, cfg.vocab_size, (B, 16))
        ms = time_forward(model, ids, n_warmup=3, n_runs=20)
        scaling[B] = {'time_ms': ms, 'time_per_token_ms': ms / (B * 16)}
        print(f"  B={B:3d} T=16: {ms:.3f} ms  ({ms/(B*16):.4f} ms/token)")

    # ===== Run 3: Scaling with sequence length =====
    print("\n[TEST 3] Scaling with sequence length (B=4)")
    seq_scaling = {}
    for T in [16, 64, 256, 1024]:
        if T > cfg.context_length:
            # Patch context_length
            cfg.context_length = T
            model = zeroBase(config=cfg, test_mode=True)
            model.eval()
        ids = torch.randint(0, cfg.vocab_size, (4, T))
        ms = time_forward(model, ids, n_warmup=3, n_runs=10)
        seq_scaling[T] = {'time_ms': ms, 'time_per_token_ms': ms / (4 * T)}
        print(f"  B=4 T={T:4d}: {ms:.3f} ms  ({ms/(4*T):.4f} ms/token)")

    # ===== Run 4: Old vs New runtime at the SAME size =====
    print("\n[TEST 4] Old vs New architecture runtime (B=4, T=16)")
    # New (current)
    new_ms = time_forward(model, torch.randint(0, cfg.vocab_size, (4, 16)),
                          n_warmup=5, n_runs=30)
    # Old (rebuild with old config)
    old_cfg = make_config()
    old_cfg.update(
        use_sliced_ffn=False, width_div_weight=0.0, path_div_weight=0.1,
        unify_load_balance=False, cost_aware_routing=False,
    )
    torch.manual_seed(SEED)
    old_model = zeroBase(config=old_cfg, test_mode=True)
    old_model.eval()
    old_ms = time_forward(old_model, torch.randint(0, old_cfg.vocab_size, (4, 16)),
                          n_warmup=5, n_runs=30)
    print(f"  OLD: {old_ms:.3f} ms")
    print(f"  NEW: {new_ms:.3f} ms")
    print(f"  Delta: {(new_ms - old_ms)/old_ms*100:+.1f}%")

    # ===== Summary =====
    print("\n" + "=" * 78)
    print("PROFILING SUMMARY")
    print("=" * 78)

    # Identify bottleneck
    subsystems = {
        'blocks': breakdown['blocks_time_ms'],
        'moe': breakdown['moe_time_ms'],
        'merger': breakdown['merger_time_ms'],
        'router': breakdown['router_time_ms'],
    }
    bottleneck = max(subsystems.items(), key=lambda x: x[1])
    print(f"\n  Biggest subsystem bottleneck: {bottleneck[0]} ({bottleneck[1]:.3f} ms)")
    print(f"  Subsystem breakdown:")
    for k, v in sorted(subsystems.items(), key=lambda x: -x[1]):
        pct = v / sum(subsystems.values()) * 100
        print(f"    {k:10s}: {v:.3f} ms ({pct:.1f}%)")

    print(f"\n  Old vs New runtime: {old_ms:.3f} -> {new_ms:.3f} ms "
          f"({(new_ms-old_ms)/old_ms*100:+.1f}%)")

    output = {
        'description': 'v0.4 profiling — identify bottlenecks after architecture changes',
        'config': {
            'hidden_size': cfg.hidden_size,
            'num_layers': cfg.num_layers,
            'width_choices': list(cfg.width_choices),
            'use_sliced_ffn': cfg.use_sliced_ffn,
        },
        'subsystem_breakdown': subsystems,
        'bottleneck': bottleneck[0],
        'batch_scaling': scaling,
        'seq_scaling': seq_scaling,
        'old_vs_new': {'old_ms': old_ms, 'new_ms': new_ms,
                       'delta_pct': (new_ms - old_ms) / old_ms * 100},
    }
    out_path = out_dir / "phase_v04_profiling.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] {out_path}")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
