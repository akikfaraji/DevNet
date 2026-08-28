# Xorzen v0.4 — Performance

> Generated: 2026-08-15
> All measurements on CPU (single-threaded torch). Hardware: cloud dev container.

## 1. SSM Scan Performance

### Theoretical Complexities

| Scan | Work | Parallel Depth | Memory |
|------|------|----------------|--------|
| sequential | O(T·N) | O(T) | O(T·N) |
| parallel (Hillis-Steele) | O(T·log(T)·N) | O(log T) | O(T·log(T)·N) |
| chunked | O(T·N) | O(T/chunk + chunk) | O(chunk·N) |

### Empirical Timing (B=1, N=32, fp32, mean of 3 runs)

| T | sequential | parallel | chunked |
|------|-----------|----------|---------|
| 64 | 0.40 ms | 0.16 ms | 0.39 ms |
| 128 | 0.77 ms | 0.21 ms | 0.76 ms |
| 256 | 1.52 ms | 0.31 ms | 1.43 ms |
| 512 | 2.91 ms | 0.52 ms | 3.05 ms |
| 1024 | 6.41 ms | 1.06 ms | 6.03 ms |
| 2048 | 36.37 ms | 1.52 ms | 12.29 ms |

**Finding**: On CPU, parallel is fastest despite O(T log T) work, because it
has no Python loop. Chunked is 2nd — its T/chunk Python iterations add
overhead. Sequential is slowest — T Python iterations.

On GPU, the order would reverse for large T: chunked (O(T) work) would
beat parallel (O(T log T) work).

## 2. Long-Sequence Stability (Phase 9.3)

| T | state_max | state_mean | grad_A_max | grad_B_max | finite | time |
|------|-----------|------------|------------|------------|--------|------|
| 1024 | 0.66 | 0.096 | 1.52 | 5.56 | YES | 6 ms |
| 4096 | 0.73 | 0.096 | 2.03 | 5.61 | YES | 24 ms |
| 16384 | 0.63 | 0.096 | 2.49 | 5.89 | YES | 111 ms |

State magnitudes stay bounded (≤0.73) across 16× sequence length increase.
Gradient magnitudes grow slowly (1.5 → 2.5 for A, 5.5 → 5.9 for B).
No NaN/Inf at any length.

## 3. Conditional Compute FLOPs Reduction (Phase 5)

### Depth Routing

| Scenario | depth_fraction | FLOPs | Reduction |
|----------|---------------|-------|-----------|
| all_active | 100% | 8,499,200 | 0% (baseline) |
| half_active | 66.7% | 6,139,904 | 27.8% |
| none_active (min_depth=1) | 33.3% | 3,780,608 | 55.5% |

**Verdict**: Genuine sparse depth. Half the tokens skipping a layer reduces
that layer's compute by ~50% (some overhead from always-on components).

### Width Routing (SlicedFFN standalone)

| Width | % of max | FLOPs | Ratio |
|-------|----------|-------|-------|
| 16 | 25% | 32,768 | 0.25 |
| 32 | 50% | 65,536 | 0.50 |
| 64 | 100% | 131,072 | 1.00 |

**Verdict**: Genuine linear scaling with width. Verified via Linear hook
shapes — fc1 output dim matches selected width exactly.

### Pathway Routing

| K | Pathways called | Calls |
|---|----------------|-------|
| 1 | local only | {local:1, low_rank:0, ssm:0} |
| 2 | local + low_rank | {local:1, low_rank:1, ssm:0} |
| 3 | all | {local:1, low_rank:1, ssm:1} |
| adversarial (all→SSM) | ssm only | {local:0, low_rank:0, ssm:1} |

**Verdict**: Genuine top-k sparse dispatch. Unselected pathways receive
zero forward calls.

### MoE Routing

| K | Experts called | Uncalled experts | Uncalled grad |
|---|---------------|-----------------|---------------|
| 1 | [0] | [1,2,3] | 0.0 |
| 2 | [0,1] | [2,3] | 0.0 |
| 3 | [0,1,2] | [3] | 0.0 |

**Verdict**: Genuine top-k expert dispatch. Uncalled experts receive
exactly zero gradient.

## 4. Compute Budget Control (Phase 7)

| Budget | Predicted | Actual FLOPs | AvgDepth | AvgWidth | Runtime |
|--------|-----------|--------------|----------|----------|---------|
| 0.10 | 0.0071 | 1,693,696 | 0.12/3 | 50% | 4.4 ms |
| 0.25 | 0.0238 | 2,306,048 | 0.41/3 | 50% | 6.4 ms |
| 0.50 | 0.0844 | 4,478,976 | 1.31/3 | 50% | 6.0 ms |
| 0.75 | 0.1451 | 6,090,752 | 1.97/3 | 55% | 5.6 ms |
| 1.00 | 0.2033 | 6,936,576 | 2.34/3 | 78% | 7.6 ms |

**Verdict**: Budget genuinely controls compute (4× FLOPs range). Predicted
and actual correlate. Width direction correct after Phase 7 sign fix.

## 5. Baseline Comparison (Phase 13)

50 training steps, tiny config (H=32, 3 layers, 4 experts top-2).

| Baseline | Params | Init Loss | Final Loss | FLOPs/fwd | ms/fwd | Tok/s |
|----------|--------|-----------|------------|-----------|--------|-------|
| A_dense | 227,946 | 3.46 | 0.68 | 603M | 13.9 | 9228 |
| B_routing_disabled | 231,195 | 3.47 | 0.85 | 358M | 11.3 | 11364 |
| C_dense_exec | 231,195 | 3.47 | 0.65 | 604M | 10.9 | 11698 |
| D_genuine_sparse | 231,195 | 3.47 | 0.85 | 358M | 14.3 | 8972 |

**Key findings**:
- D uses 40.6% fewer FLOPs than A (genuine sparsity)
- C uses same FLOPs as A (confirms compute-then-mask wastes compute)
- D loss decreased 75.6% (3.47 → 0.85)
- D is slower than A in wall-clock at tiny scale (Python dispatch overhead)

## 6. Training Performance (Phase 4)

200 steps full-batch AdamW lr=3e-3 on 8 deterministic sequences:

| Metric | Value |
|--------|-------|
| Initial loss | 3.4701 |
| Final loss | 0.0062 |
| Loss reduction | 99.8% |
| Initial accuracy | 5.0% |
| Final accuracy | 98.3% |
| All outputs finite | YES |
| Gradient reach | All expected subsystems |

## 7. Ablation Impact (Phase 14)

50 training steps each. Base final loss ≈ 0.92.

| Ablation | Final Loss | vs Base | Verdict |
|----------|-----------|---------|---------|
| none (base) | ~0.92 | — | baseline |
| no_ssm | 0.92 | same | neutral |
| no_local | 0.94 | same | neutral |
| no_low_rank | 0.80 | BETTER | HELPS (!) |
| no_adaptive_depth | 0.94 | same | neutral |
| no_adaptive_width | 0.83 | BETTER | HELPS (!) |
| no_moe | 0.88 | BETTER | HELPS (!) |
| no_balancing | 0.93 | same | neutral |

**Key finding**: `no_low_rank` significantly HELPS (16% loss reduction). The
low-rank pathway is hurting quality at tiny scale because the router
collapses to SSM (Phase 6), leaving low-rank params with random gradients.

## 8. Profiling Breakdown (Phase 11)

Top subsystems by FLOPs (tiny config, 8 sequences × 16 tokens):

| Subsystem | % of FLOPs | Notes |
|-----------|-----------|-------|
| lm_head | ~30% | Always-on, vocab projection |
| token_embedding | ~25% | Always-on |
| blocks (HASS) | ~20% | Depth/width/pathway routed |
| router | ~10% | Always-on |
| moe | ~10% | Top-k routed |
| merger | ~5% | Always-on |

**Bottleneck**: vocab projection (lm_head + token_embedding) dominates at
~55% of FLOPs. This is NOT budget-controlled — even at budget=0.10, the
vocab projections run at full cost.

## 9. Disk-Sharded MoE Performance (Phase 10)

| Metric | Value |
|--------|-------|
| Cache capacity | 3 experts |
| Cache hits observed | 1 (after 1 put + 1 get) |
| Cache misses observed | 17 (across full lifecycle) |
| Cache evictions | 1 (when 4th expert loaded) |
| Weight match after restart | 4/4 exact |
| Train/eval propagation | YES (after Phase 10 fix) |

## 10. Limitations and Caveats

1. All measurements on CPU at tiny scale (H=32). GPU performance at
   production scale (H=1024+) would differ significantly.

2. Sparse dispatch has Python overhead that dominates at tiny scale on CPU.
   At production scale on GPU, sparse would win.

3. The FLOPs counter only measures Linear and Conv1d operations. Other
   operations (LayerNorm, softmax, etc.) are not counted.

4. Wall-clock measurements have ~1ms resolution; tiny-config runs are
   near the resolution limit.

5. The compute budget ratio mismatch (0.244 vs 0.10) is because
   embeddings/lm_head are always-on regardless of budget. A cost-weighted
   budget controller would fix this (Phase 8 followup).
