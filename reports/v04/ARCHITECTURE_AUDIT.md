# Xorzen v0.4 — Architecture Audit

> Generated: 2026-08-15
> 23 claims audited. Each classified as PROVEN / EMPIRICALLY VERIFIED / CONDITIONALLY TRUE / PARTIALLY IMPLEMENTED / NOT VERIFIED / FALSE.

## Summary

| Status | Count |
|--------|-------|
| PROVEN | 12 |
| EMPIRICALLY VERIFIED | 5 |
| PARTIALLY IMPLEMENTED | 3 |
| FALSE | 2 |
| FALSE (by design) | 1 |
| **Total** | **23** |

The 2 FALSE claims are dead code (ComputeController and adaptive_halting
exist as standalone modules but are not wired into zeroModel). They are
documented as architectural gaps, not as failures of verified components.

The 1 FALSE (by design) is CoT being frozen for pre-training — this is
intentional and documented.

## Claim Status Definitions

- **PROVEN**: Mathematically derived or verified by comprehensive testing with no known counterexamples.
- **EMPIRICALLY VERIFIED**: Verified by measurement under specific conditions; may not hold in all configurations.
- **CONDITIONALLY TRUE**: True under specific conditions (hyperparameters, scales, configurations).
- **PARTIALLY IMPLEMENTED**: Code exists but is not fully wired into the model or has known gaps.
- **NOT VERIFIED**: Not tested or measured.
- **FALSE**: Claimed but not true (either was never true, or regressed).

## SSM Claims

### [PROVEN] SSM-1: ZOH discretization of both A and B
- **Implementation**: `discretize_zoh()` computes `A_bar = exp(dt*A)` and `B_bar = ((exp(dt*A)-1)/A)*B` with Taylor fallback for `|dt*A| < eps`.
- **Test**: `test_ssm_zoh_discretization_stable`; Phase 9 numerical equivalence (8 configs, fp32+bf16).
- **Evidence**: All 8 configs pass tolerance (fp32 <1e-4, bf16 <1e-2). Gradient equivalence verified (err <1e-6).
- **Limitations**: Only diagonal A supported.

### [PROVEN] SSM-2: Sequential scan O(L) complexity
- **Implementation**: Python loop over T, vectorized over B and N.
- **Test**: Phase 9.4 with empirical timing at T=64..2048.
- **Evidence**: Theoretical O(T*N) work, O(T) depth. Empirical linear scaling confirmed.

### [PROVEN] SSM-3: Parallel scan O(L log L) work, O(log L) depth (Hillis-Steele)
- **Implementation**: `log2(T)` iterations, each touches all T elements. Pure PyTorch.
- **Test**: Phase 9.4.
- **Evidence**: Theoretical O(T*log(T)*N) work, O(log T) depth, O(T*log(T)*N) memory. Empirical: 1.5ms at T=2048 on CPU (faster than chunked due to no Python loop).
- **Limitations**: On GPU, O(T log T) work makes it slower than chunked for very large T.

### [PROVEN] SSM-4: Chunked scan O(L) work, O(L/chunk + chunk) depth
- **Implementation**: T/chunk_size Python iterations, each runs sequential_scan on a chunk.
- **Test**: Phase 9.4.
- **Evidence**: Theoretical O(T*N) work, O(T/chunk + chunk) depth, O(chunk*N) memory. Empirical: 12ms at T=2048.

### [EMPIRICALLY VERIFIED] SSM-5: Long-sequence stability (16K+)
- **Implementation**: chunked_scan with LayerNorm on states.
- **Test**: Phase 9.3: T=1024, 4096, 16384.
- **Evidence**: T=16384: state_max=0.63, grad_max=5.89, all finite. No NaN/Inf.
- **Limitations**: Not tested at 32K (memory). Stability depends on A_bar in (0,1).

## Sparsity Claims

### [PROVEN] SPARSE-1: Depth routing genuinely skips computation
- **Implementation**: At inference, `block.forward_with_depth()` gathers active tokens, runs block on subset, scatters back.
- **Test**: Phase 5.1: half_active reduces FLOPs 27.8%, none_active 55.5%.
- **Evidence**: OpCounter measured: all=8.5M, half=6.1M, none=3.8M FLOPs.
- **Limitations**: Training uses STE mask blend for differentiability. 27.8% vs theoretical 33% because embeddings/MoE/lm_head always-on.

### [PROVEN] SPARSE-2: Width routing genuinely changes matmul dimensions
- **Implementation**: `SlicedFFN`: fc1 H→W, fc2 W→H, only selected W executes.
- **Test**: Phase 5.2: width=16:32:64 FLOPs = 0.25:0.50:1.00.
- **Limitations**: SlicedFFN is NOT wired into HASSBlock (uses AdaptiveFFN). Phase 15 recommendation.

### [PROVEN] SPARSE-3: Pathway routing genuinely skips unselected pathways
- **Implementation**: `sparse_pathway_dispatch`: for each pathway, find tokens that selected it, only call forward on those. Pathways with no tokens are NOT called.
- **Test**: Phase 5.3: K=1 → 1 pathway, K=2 → 2, K=3 → all 3. Adversarial: all tokens to SSM → only SSM called.
- **Evidence**: Pathway call counter confirms 0 calls to unselected pathways.

### [PROVEN] SPARSE-4: MoE top-k genuinely executes only K experts
- **Implementation**: `ShardedExpertFabric.forward`: loops over top_k slots, groups by expert id. Unselected experts NOT loaded.
- **Test**: Phase 5.4: K=1/2/3 verified. Uncalled experts get zero gradient.
- **Evidence**: Expert call counts + gradient norms confirm.

### [EMPIRICALLY VERIFIED] SPARSE-5: Genuine sparse achieves FLOPs reduction
- **Test**: Phase 13: D (sparse) 358M FLOPs vs A (dense) 603M (40.6% reduction).
- **Limitations**: At tiny scale on CPU, D is slower in wall-clock due to Python dispatch overhead.

## Router Stability Claims

### [PARTIALLY IMPLEMENTED] ROUTER-1: Path diversity loss prevents collapse
- **Implementation**: `path_diversity_loss = -entropy(path_probs)`, default weight 0.1 (raised from 0.02).
- **Test**: Phase 6: 4 regimes tested.
- **Evidence**: no_balancing → collapse. strong (0.2) → 2/3 pathways active.
- **Limitations**: Even at 0.2, only 2/3 pathways. Necessary but not sufficient.

### [EMPIRICALLY VERIFIED] ROUTER-2: Expert load balancing prevents collapse
- **Implementation**: Switch formula `E * sum(f*p)` + L2 loss (double-counts, Phase 15 noted).
- **Test**: Phase 6: all regimes had 0 dead experts, gini ≤ 0.057.

### [EMPIRICALLY VERIFIED] ROUTER-3: Depth routing does not collapse
- **Test**: Phase 4: avg_depth=3.00/3. Phase 6: all regimes ~3.0.
- **Limitations**: min_depth=1 forces layer 0 active.

## Compute Budget Claims

### [PARTIALLY IMPLEMENTED] BUDGET-1: ComputeController allocates by budget
- **Implementation**: `compute_controller.py` takes `compute_budget ∈ [0,1]`.
- **Test**: Phase 7: FLOPs scale 1.7M → 6.9M across budgets 0.10 → 1.00.
- **Limitations**: NOT wired into zeroModel (Phase 7 used drop-in replacement). Width bias had wrong sign (FIXED).

### [EMPIRICALLY VERIFIED] BUDGET-2: Budget controls actual compute
- **Test**: Phase 7: predicted and actual both monotonic in budget.
- **Limitations**: FLOPs ratio (0.244) ≠ budget ratio (0.10) due to always-on components.

## MoE Disk Sharding Claims

### [PROVEN] MOE-1: Save → restart → reload preserves weights
- **Test**: Phase 10 steps 1-7. `test_moe_restart_preserves_weights`.
- **Evidence**: 4/4 expert weights match exactly after restart.

### [PROVEN] MOE-2: LRU cache evicts least-recently-used
- **Test**: Phase 10 step 10: 4 experts loaded with cap=3, expert 0 evicted.

### [PROVEN] MOE-3: Train/eval mode propagates to cached experts
- **Test**: Phase 10 step 13. `test_moe_eval_mode_propagates_to_cached_experts`.
- **Limitations**: Was a BUG in v0.3 (LRUExpertCache not nn.Module). Fixed in Phase 10.

## Training Claims

### [PROVEN] TRAIN-1: Architecture can train
- **Test**: Phase 4: 200 steps, loss 3.47 → 0.006. `test_tiny_training_loss_decreases`.
- **Evidence**: 99.8% loss reduction, 98.3% accuracy, all finite.
- **Limitations**: CoT frozen by design. Pathway collapse means local + low_rank get zero grad.

## Dead Code / False Claims

### [FALSE] DEAD-1: ComputeController is part of the model
- **Evidence**: grep shows no import in zero/model.py or zmoe.py.
- **Recommendation**: Wire in or remove (Phase 15).

### [FALSE] DEAD-2: Adaptive halting is part of the model
- **Evidence**: `adaptive_halting.py` exists but is not imported.
- **Recommendation**: Remove or wire in.

### [PARTIALLY IMPLEMENTED] DEAD-3: SlicedFFN provides model-level width sparsity
- **Evidence**: Works standalone (Phase 5.2) but not used by HASSBlock.
- **Recommendation**: Replace AdaptiveFFN with SlicedFFN (Phase 15).

### [FALSE (by design)] DEAD-4: CoT contributes to training
- **Evidence**: Frozen (requires_grad=False) for pre-training.
- **Recommendation**: Document clearly. `model.enable_cot()` unfreezes for fine-tuning.
