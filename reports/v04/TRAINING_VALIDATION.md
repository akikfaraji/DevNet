# Xorzen v0.4 — Training Validation

> Generated: 2026-08-15

## 1. Minimal Overfit Test (Phase 4)

### Setup
- **Config**: vocab=32, hidden=32, 3 layers, 2 attention heads, max_depth=3
- **Widths**: (16, 32)
- **Experts**: 4 experts, top-2
- **Pathways**: 3 (local, low_rank, ssm), pathway_top_k=2
- **Data**: 8 deterministic sequences of length 16, with learnable pattern
  `token[t+1] = (token[t] + offset) % vocab_size`
- **Optimizer**: AdamW lr=3e-3, betas=(0.9, 0.95), no weight decay
- **Training**: 200 steps, full-batch, CPU, seed=1337

### Results

| Metric | Value |
|--------|-------|
| Initial loss | 3.4701 |
| Final loss | 0.0062 |
| Loss reduction | 99.8% |
| Initial accuracy | 5.0% |
| Final accuracy | 98.3% |
| Random baseline accuracy | 3.12% (1/32) |
| All outputs finite | YES |
| Bad gradients (NaN/Inf) | 0 |

### Loss Curve (selected steps)

```
step   0: loss=3.4701  acc=5.0%   avg_depth=2.75/3  path_H=1.070/1.099
step  20: loss=2.3325  acc=65.0%  avg_depth=2.92/3  path_H=1.065/1.099
step  40: loss=1.2427  acc=88.3%  avg_depth=2.98/3  path_H=1.001/1.099
step  60: loss=0.4813  acc=95.8%  avg_depth=3.00/3  path_H=0.928/1.099
step  80: loss=0.2370  acc=95.8%  avg_depth=3.00/3  path_H=0.892/1.099
step 100: loss=0.0732  acc=98.3%  avg_depth=3.00/3  path_H=0.882/1.099
step 140: loss=0.0246  acc=98.3%  avg_depth=3.00/3  path_H=0.874/1.099
step 180: loss=0.0089  acc=98.3%  avg_depth=2.99/3  path_H=0.870/1.099
step 199: loss=0.0055  acc=98.3%  avg_depth=3.00/3  path_H=0.870/1.099
```

### Gradient Norms by Subsystem (final step)

| Subsystem | Grad L2 Norm |
|-----------|-------------|
| router | 1.98e-01 |
| blocks | 1.51e-02 |
| token_embedding | 8.66e-03 |
| lm_head | 8.66e-03 (tied weights) |
| moe | 6.30e-03 |
| final_norm | 5.00e-03 |
| position_embedding | 3.62e-03 |
| merger | 2.05e-03 |
| embedding_dropout | 0 (no params) |
| cot | 0 (frozen by design) |
| routing_regularizer | 0 (stateless) |

### Verdicts (7/7 PASS)

- [PASS] loss_decreased
- [PASS] loss_decreased_substantially (99.8%)
- [PASS] accuracy_improved
- [PASS] accuracy_above_random
- [PASS] all_finite
- [PASS] no_unexpected_zero_grad_subsystems
- [PASS] expert_utilization_ok (100%, 4/4 experts called)

### Observation

Pathway collapse detected: all 128 tokens pick SSM (pathway 2). Local and
low-rank pathways receive zero gradient. This is flagged for Phase 6
(router stability) which found that `path_div_weight=0.2` gives partial
diversity (2/3 pathways active).

## 2. Router Stability Experiments (Phase 6)

Tested 4 regimes for 100 steps each:

| Regime | path_div_weight | lb_weight | z_weight | Final Loss | PathN | PathGini | ExpDead |
|--------|----------------|-----------|----------|------------|-------|----------|---------|
| no_balancing | 0 | 0 | 0 | 0.0751 | 1 | 0.000 | 0 |
| weak_balancing | 0.002 | 1e-5 | 1e-5 | 0.0814 | 1 | 0.000 | 0 |
| current_balancing | 0.02 | 1e-4 | 1e-4 | 0.0784 | 1 | 0.000 | 0 |
| strong_balancing | 0.2 | 1e-3 | 1e-3 | 0.0990 | 2 | 0.188 | 0 |

**Findings**:
- Pathway collapse is severe in 3/4 regimes (PathN=1)
- Only `strong_balancing` (path_div=0.2) achieves partial diversity
- Expert routing is healthy in ALL regimes (gini ≤ 0.057, 0 dead)
- Depth routing is healthy in ALL regimes (avg ~3.0/3)
- Width routing collapses in ALL regimes (always picks largest)

**Action taken**: Bumped default `path_div_weight` from 0.02 to 0.1.

## 3. Tiny Training Test (Phase 17)

`tests/test_phase4_v04.py::test_tiny_training_loss_decreases`:
- 40 steps, 4 sequences, H=32, 2 layers, 2 experts top-1
- Loss decreases from ~3.5 to <1.75 (50% reduction required)
- PASSES consistently

## 4. Training-Related Bugs Found and Fixed

### Bug: Depth routing used compute-then-mask (Phase 5)
- **Symptom**: `half_active` mask reduced FLOPs by 0.0% (should be ~33%)
- **Fix**: At inference, use `block.forward_with_depth()` for genuine
  gather/scatter. After fix: 27.8% reduction.
- **Training**: Still uses STE mask blend for differentiability (no change).

### Bug: ComputeController width_bias wrong sign (Phase 7)
- **Symptom**: Low budget favored LARGER widths (backwards)
- **Fix**: Reversed linspace sign. Now low budget → smaller widths.

### Bug: ShardedExpertFabric.eval() did not propagate (Phase 10)
- **Symptom**: Cached experts stayed in training mode during validation
- **Fix**: Override `train(mode)` to propagate to cached experts + dummy.

## 5. Recommendations for Production Training

1. **Use path_div_weight=0.2** for production (Phase 6 finding). The
   default 0.1 may still allow partial collapse on some datasets.

2. **Add a width diversity loss** to prevent width collapse (Phase 15).

3. **Wire ComputeController into zeroModel** to enable budget-controlled
   training (Phase 7/15).

4. **Replace AdaptiveFFN with SlicedFFN** in HASSBlock for genuine
   model-level width sparsity (Phase 15).

5. **Unify the two load-balance losses** (router CV + model L2) — they
   double-count (Phase 15).

6. **Train with curriculum**: Start with high budget (dense) for stability,
   anneal to lower budget for sparsity (Phase 16 — not implemented but
   recommended).

7. **Monitor for pathway collapse** during training. If PathN drops to 1,
   increase path_div_weight or restart with different seed.
