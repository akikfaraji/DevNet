"""
Phase 10 — Disk-Sharded MoE Stress Test.

Tests the full lifecycle of the disk-sharded MoE system:
  1. Create experts
  2. Save experts
  3. Destroy manager
  4. Create a new manager
  5. Reload experts
  6. Verify metadata
  7. Verify expert weights
  8. Execute inference
  9. Execute training (gradient flow)
  10. Evict experts repeatedly
  11. Reload evicted experts

Verifies:
  - restart correctness (the int-key/string-key JSON bug)
  - cache correctness (LRU ordering, hit/miss tracking)
  - memory bounds
  - dtype correctness
  - training/eval mode correctness
  - gradient correctness
"""

import os
import sys
import json
import shutil
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/xorzen_dev")

import torch
import torch.nn as nn
import numpy as np

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)

from xorzen.config import ConfigFactory, ModelSize
from xorzen.model.zmoe import (
    ShardedExpertFabric, ExpertDiskManager, LRUExpertCache, ExpertFFN,
)


def make_test_config(shard_dir, num_experts=4, hidden=16, mult=2.0, cache_size=3):
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        model_name="xorzen_v04_phase10",
        vocab_size=32, context_length=16, hidden_size=hidden,
        num_layers=1, num_attention_heads=2, max_depth=1, min_depth=1,
        width_choices=(hidden,), cot_dim=4, cot_components=6,
        expert_count=num_experts, top_k_experts=2,
        expert_hidden_multiplier=mult,
        expert_shard_dir=shard_dir,
        max_expert_cache=cache_size,
        router_hidden_dim=8, router_num_layers=1, merger_num_layers=1,
        shard_experts=False, pad_token_id=0, dropout=0.0,
        load_balancing_weight=0.0,
        gradient_checkpointing=False,
    )
    return cfg


def main():
    out_dir = Path("/home/z/my-project/xorzen_dev/reports/v04")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*72)
    print("PHASE 10 — DISK-SHARDED MoE LIFECYCLE STRESS TEST")
    print("="*72)

    tmp_dir = Path(tempfile.mkdtemp(prefix="phase10_moe_"))
    print(f"\n[SETUP] tmp_dir = {tmp_dir}")

    verdicts = {}
    metrics = {}

    # ============ STEP 1-2: Create + Save ============
    print("\n[STEP 1-2] Create and save experts")
    shard_dir_1 = tmp_dir / "shards_v1"
    cfg = make_test_config(shard_dir=shard_dir_1, num_experts=4, hidden=16, mult=2.0, cache_size=3)
    fabric1 = ShardedExpertFabric(config=cfg, test_mode=False)
    fabric1.train()

    # Initialize experts on disk
    fabric1.disk_manager.initialize_all_experts(force=True)
    n_saved = len(fabric1.disk_manager.manifest['experts'])
    print(f"  Saved {n_saved} experts to {shard_dir_1}")
    verdicts['step1_save_experts'] = n_saved == 4

    # Capture original expert weights for later comparison
    original_weights = {}
    for eid in range(4):
        exp = fabric1.disk_manager.load_expert(eid)
        original_weights[eid] = {k: v.clone() for k, v in exp.state_dict().items()}
    print(f"  Captured original weights for {len(original_weights)} experts")

    # ============ STEP 3: Destroy manager ============
    print("\n[STEP 3] Destroy manager (simulate restart)")
    del fabric1
    import gc
    gc.collect()

    # ============ STEP 4-5: Create new manager + reload ============
    print("\n[STEP 4-5] Create new manager and reload experts")
    fabric2 = ShardedExpertFabric(config=cfg, test_mode=False)

    # Verify metadata loaded
    n_loaded = len(fabric2.disk_manager.manifest['experts'])
    print(f"  Loaded manifest with {n_loaded} experts")
    verdicts['step5_reload_manifest'] = n_loaded == 4

    # The known bug was: JSON converts int keys to strings, so expert_exists(eid)
    # was checking `str(eid) in self.manifest['experts']`. Verify this works.
    for eid in range(4):
        exists = fabric2.disk_manager.expert_exists(eid)
        if not exists:
            print(f"  [FAIL] expert_exists({eid}) = False after restart")
            verdicts[f'step5_expert_{eid}_exists'] = False
        else:
            verdicts[f'step5_expert_{eid}_exists'] = True

    # ============ STEP 6-7: Verify metadata + weights ============
    print("\n[STEP 6-7] Verify metadata and weights match originals")
    weight_matches = 0
    for eid in range(4):
        loaded = fabric2.disk_manager.load_expert(eid)
        match = True
        for k, v_orig in original_weights[eid].items():
            v_loaded = loaded.state_dict()[k]
            if not torch.equal(v_orig, v_loaded):
                print(f"  [FAIL] expert {eid} weight {k} mismatch")
                match = False
                break
        if match:
            weight_matches += 1
    print(f"  Weight match: {weight_matches}/4 experts")
    verdicts['step7_weights_match'] = weight_matches == 4

    # ============ STEP 8: Execute inference ============
    print("\n[STEP 8] Execute inference")
    fabric2.eval()
    N, H = 8, 16
    K = 2
    x = torch.randn(N, H)
    # Route tokens to experts (round-robin)
    idx = torch.zeros(N, K, dtype=torch.long)
    for n in range(N):
        idx[n, 0] = n % 4
        idx[n, 1] = (n + 1) % 4
    w = torch.ones(N, K) / K

    with torch.no_grad():
        out, stats = fabric2(x, idx, w)
    print(f"  Output shape: {out.shape}")
    print(f"  Output finite: {torch.isfinite(out).all().item()}")
    print(f"  Stats: {stats}")
    verdicts['step8_inference'] = bool(torch.isfinite(out).all().item() and out.shape == (N, H))

    # ============ STEP 9: Execute training (gradient flow) ============
    print("\n[STEP 9] Execute training (gradient flow)")
    fabric2.train()
    x = torch.randn(N, H, requires_grad=True)
    out, stats = fabric2(x, idx, w)
    loss = out.sum()
    loss.backward()
    # Verify x has gradient
    x_has_grad = x.grad is not None and x.grad.abs().sum() > 0
    # Verify cached experts have gradients
    expert_grads = 0
    with fabric2.cache.lock:
        for eid, exp in fabric2.cache.cache.items():
            for p in exp.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    expert_grads += 1
                    break
    print(f"  x has gradient: {x_has_grad}")
    print(f"  Cached experts with gradient: {expert_grads}")
    verdicts['step9_training_gradients'] = bool(x_has_grad and expert_grads > 0)

    # ============ STEP 10: Evict experts repeatedly ============
    print("\n[STEP 10] Evict experts repeatedly (LRU stress)")
    # Force cache eviction by loading more experts than cache capacity (3)
    fabric2.clear_cache()
    # Load 4 experts in order 0, 1, 2, 3 — this should evict 0 (LRU)
    for eid in [0, 1, 2, 3]:
        exp = fabric2.disk_manager.load_expert(eid)
        fabric2.cache.put(eid, exp)
    cache_stats_after = fabric2.get_cache_statistics()
    print(f"  Cache stats after 4 loads (capacity 3): {cache_stats_after}")
    # The cache should hold experts 1, 2, 3 (0 was evicted)
    cache_keys = list(fabric2.cache.cache.keys())
    print(f"  Cache contents: {cache_keys}")
    verdicts['step10_lru_eviction'] = (
        cache_stats_after['evictions'] >= 1 and  # at least one eviction occurred
        0 not in cache_keys and  # expert 0 (oldest) was evicted
        set(cache_keys) == {1, 2, 3}  # experts 1, 2, 3 remain (LRU order)
    )

    # ============ STEP 11: Reload evicted expert ============
    print("\n[STEP 11] Reload evicted expert 0")
    # Access expert 0 — should be a cache miss and reload from disk
    cache_misses_before = fabric2.cache.misses
    exp0 = fabric2.cache.get(0)
    if exp0 is None:
        # Need to load from disk
        exp0 = fabric2.disk_manager.load_expert(0)
        fabric2.cache.put(0, exp0)
        # Now expert 1 should be evicted (LRU)
    cache_stats_after_reload = fabric2.get_cache_statistics()
    cache_keys_after = list(fabric2.cache.cache.keys())
    print(f"  Cache stats after reload: {cache_stats_after_reload}")
    print(f"  Cache contents: {cache_keys_after}")
    # Verify expert 0 weights still match original
    exp0_weights = exp0.state_dict()
    exp0_match = all(torch.equal(v_orig, exp0_weights[k]) for k, v_orig in original_weights[0].items())
    print(f"  Reloaded expert 0 weights match original: {exp0_match}")
    verdicts['step11_reload_evicted'] = bool(exp0_match and 0 in cache_keys_after)

    # ============ STEP 12: Cache hit/miss tracking ============
    print("\n[STEP 12] Cache hit/miss tracking")
    fabric2.clear_cache()
    # First access — miss
    e = fabric2.disk_manager.load_expert(0)
    fabric2.cache.put(0, e)
    misses_1 = fabric2.cache.misses
    hits_1 = fabric2.cache.hits
    # Second access — hit
    e = fabric2.cache.get(0)
    hits_2 = fabric2.cache.hits
    print(f"  After 1st put: misses={misses_1}, hits={hits_1}")
    print(f"  After 2nd get: hits={hits_2}")
    verdicts['step12_cache_tracking'] = bool(hits_2 == hits_1 + 1)

    # ============ STEP 13: Training/eval mode correctness ============
    print("\n[STEP 13] Training/eval mode correctness")
    fabric2.train()
    train_mode_ok = all(e.training for e in fabric2.cache.cache.values())
    fabric2.eval()
    eval_mode_ok = all(not e.training for e in fabric2.cache.cache.values())
    print(f"  train() propagates to cached experts: {train_mode_ok}")
    print(f"  eval() propagates to cached experts: {eval_mode_ok}")
    verdicts['step13_train_eval_mode'] = bool(train_mode_ok and eval_mode_ok)

    # ============ STEP 14: dtype correctness ============
    print("\n[STEP 14] dtype correctness")
    # Experts should be float32 by default
    dtypes_ok = True
    for eid in range(4):
        exp = fabric2.disk_manager.load_expert(eid)
        for p in exp.parameters():
            if p.dtype != torch.float32:
                dtypes_ok = False
                break
    print(f"  All expert params float32: {dtypes_ok}")
    verdicts['step14_dtype'] = bool(dtypes_ok)

    # ============ Cleanup ============
    print(f"\n[CLEANUP] removing {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ============ Summary ============
    print("\n" + "="*72)
    print("PHASE 10 — VERDICTS")
    print("="*72)
    total_pass = 0
    total_check = 0
    for k, v in verdicts.items():
        total_check += 1
        if v:
            total_pass += 1
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  Total: {total_pass}/{total_check} PASS")
    overall_pass = total_pass == total_check
    print(f"  OVERALL: {'PASS' if overall_pass else 'PARTIAL/FAIL'}")

    metrics = {
        'n_experts': 4,
        'cache_capacity': 3,
        'experts_reloaded_after_restart': n_loaded,
        'weights_matched_after_restart': weight_matches,
        'cache_evictions_observed': cache_stats_after.get('evictions', 0),
        'cache_hits_observed': hits_2 if 'hits_2' in dir() else 0,
    }

    with open(out_dir / "phase10_moe_stress.json", "w") as f:
        json.dump({
            'verdicts': verdicts,
            'metrics': metrics,
            'overall_pass': overall_pass,
        }, f, indent=2)

    print(f"\n[SAVED] {out_dir/'phase10_moe_stress.json'}")
    return overall_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
