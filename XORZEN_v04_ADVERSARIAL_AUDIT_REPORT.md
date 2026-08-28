# XORZEN v0.4 — Adversarial Architecture Audit Report

**Date**: 2026-08-28  
**Auditor**: Adversarial ML Architecture Researcher  
**Scope**: Complete forensic verification of XORZEN v0.4 implementation vs. claims  
**Method**: Claim-by-claim re-derivation from source code; no trust in prior documentation or test labels

---

## Executive Summary

XORZEN v0.4 makes **meaningful architectural progress** (SlicedFFN, sparse pathway dispatch, cost-aware routing, unified load-balance design) but **several flagship claims are false in the current implementation**:

| Claim | Status | Reality |
|-------|--------|---------|
| Depth routing skips computation | ❌ **FALSE** | Training: blend (no FLOP savings); Inference: batch-level only |
| SSM ZOH discretization (A & B) | ❌ **FALSE** | A discretized; B uses raw projection (no Δ factor) |
| Load-balance loss prevents collapse | ⚠️ **PARTIAL** | Correct Switch formula exists in `load_balance.py` but **unused** |
| SPPQ progressive quantization | ❌ **FALSE** | Fake quantization — dequantizes to float32, 0% memory savings |
| CoT maintains reasoning state | ❌ **FALSE** | Frozen, zero signal, dead code during pre-training |
| Active params ≈ target_active_ratio | ⚠️ **UNVERIFIED** | Heuristic formula, never measured via backward hooks |

**The v0.4 benchmarks (2× lower loss, -22.8% FLOPs) are real improvements** from changes that DO work. But they don't validate the broken claims.

---

## Phase 1: Architecture Map (Reconstructed from Implementation)

| Component | File | Key Findings |
|-----------|------|--------------|
| **Core Config** | `config.py` | 800+ lines, `ModelConfig` with all routing params, `estimate_active_parameters()` heuristic |
| **AdaptiveRouter** | `model/components/routing.py` | 1200+ lines, 4-axis routing (depth/width/path/expert), STE at training, hard top-k at inference |
| **SlicedFFN** | `model/components/sliced_ffn.py` | **Genuine nested slicing**: `fc1[:, :W_i]`, `fc2[:W_i, :]` — FLOPs ∝ W_i |
| **AdaptiveFFN (legacy)** | `model/components/hass_block.py:569` | **Compute-then-blend**: ALWAYS runs full base FFN + adapters, **INCREASES compute** |
| **HASSBlock** | `model/components/hass_block.py` | 3 pathways (Local/LowRank/SSM), `pathway_top_k` sparse dispatch at inference only |
| **SSMPathway** | `model/components/hass_block.py:323` | ZOH for A only; chunked scan (Python loop over T/chunk_size) |
| **ShardedExpertFabric** | `model/zmoe.py` | Disk-sharded MoE, LRU cache, top-k routing, **test_mode uses 1 dummy expert** |
| **zeroModel** | `models/zero/model.py` | Main model: embeddings → router → HASS blocks → MoE → merger → LM head |
| **SPPQ** | `utils/sppq.py` | **Fake quantization**: dequantizes back to float32, `_update_target_bits` is `pass` |
| **Tokenizer** | `tokenizer/loader.py` | HF tokenizer wrapper, 65k BPE variant |
| **Load Balance** | `model/components/load_balance.py` | New canonical Switch Transformer formula; old CV/L2 formulas still exist in 3 places |

---

## Phase 2: Claim-by-Claim Verification

### SPARSE-1: Depth routing genuinely skips computation
**Status: ❌ INCORRECT**

| Aspect | Finding |
|--------|---------|
| **Theory** | Per-token gather-scatter at inference |
| **Training** | ALL layers computed; output blended via STE: `block_out * mask + x * (1-mask)` |
| **Inference** | Skips layer ONLY if ENTIRE BATCH has hard-0 mask (`if n_active == 0: continue`) |
| **Dead Code** | `HASSBlock.forward_with_depth` exists (line 942) but NEVER called by `zeroModel.forward` |
| **Evidence** | `model.py:482-503` (inference skip), `model.py:534` (training blend), `hass_block.py:942` (dead method) |

### SPARSE-2: Width routing genuinely changes matmul dimensions
**Status: ✅ PROVEN (model-level)**

| Aspect | Finding |
|--------|---------|
| **Theory** | Nested slicing: smaller width → proportionally fewer FLOPs |
| **Implementation** | `SlicedFFN._forward_single_width`: slices `fc1.weight[:w,:]`, `fc2.weight[:,:w]` |
| **Verified** | `phase_v04_model_level_sparse.py` Test 1: width=16 vs 32 → FFN FLOPs ratio 0.500 (exact 16/32) |
| **Caveat** | Only works when `use_sliced_ffn=True` (v0.4 default). Legacy `AdaptiveFFN` is compute-then-blend |

### SPARSE-3: Pathway routing genuinely skips unselected pathways
**Status: ✅ PROVEN (inference only)**

| Aspect | Finding |
|--------|---------|
| **Theory** | Top-k sparse dispatch |
| **Implementation** | `sparse_pathway_dispatch.py`: groups tokens by selected pathway, only calls those pathways |
| **Training** | STE — forward uses hard top-k, backward sees soft probs |
| **Verified** | Test 3: forcing all tokens → SSM → local=0, low_rank=0, ssm=3 |

### SPARSE-4: MoE top-k genuinely executes only K experts per token
**Status: ✅ PROVEN (with test_mode=False)**

| Aspect | Finding |
|--------|---------|
| **Theory** | Only K experts per token receive compute |
| **Implementation** | `ShardedExpertFabric.forward`: loops over `top_k` slots, groups tokens by expert |
| **test_mode** | **Breaks this claim**: uses single dummy expert for ALL tokens |
| **Verified** | `phase10_moe_stress.py` 14/14 PASS (with test_mode=False, disk-sharded experts) |

---

### P1: Active-parameter sparsity ≤ 10%
**Status: ⚠️ PARTIALLY PROVEN**

- **Formula**: `active = embeddings + avg_layers × layer_params × width_factor × target_active_ratio + router + cot + top_k × expert_params`
- **Issues**: (1) Ignores HASS pathways, merger, LM head in "always-on"; (2) Expert count uses `*2` instead of `*3` for SwiGLU (undercounts); (3) Not measured — just a formula
- **Test**: `verify_architecture.py` checks 0% < active% ≤ 100% (trivially passes)

### P2: SSM linear-time recurrence
**Status: 📊 EMPIRICALLY VERIFIED ONLY**

- **Implementation**: `chunked_scan` — Python loop over T/chunk_size iterations (e.g., 8 for T=2048)
- **Docstring Lie**: `ssm.py:13` claims "O(L log L) parallel scan" but `SSMPathway.forward` uses chunked sequential
- **Verified**: T=128→1024 gives 6.6× time (expected 8× for linear)

### P3: SSM ZOH discretization for A and B
**Status: ❌ INCORRECT**

| Theory | Implementation |
|--------|----------------|
| `A_bar = exp(dt*A)` | ✅ `A_bar = exp(dt*a)` |
| `B_bar = (A_bar-1)/A * B` (exact ZOH) | ❌ `Bv = B_proj(x)` — **NO Δ factor** |
| `B_bar = dt * B` (first-order) | ❌ Not used |

**Evidence**: `hass_block.py:460-466` — `Bv` computed without `dt` multiplication; `ssm_scan.py:62-106` has correct `discretize_zoh` but it's NOT used by `SSMPathway`

**Corrected**: `Bv = dt * B_proj(...)` (first-order) or `Bv = ((A_bar-1)/a) * B_proj(...)` (exact ZOH)

### P4: Top-k MoE routing validity
**Status: ⚠️ PARTIALLY PROVEN**

- Verified: simplex constraint, indices in range
- Gap: Does not verify only K experts actually RUN (test_mode uses dummy expert)

### P5: Load-balance loss L_lb ≥ 0, min at uniform
**Status: ❌ INCORRECT (old formulas) / ⚠️ PARTIALLY PROVEN (new formula)**

| Location | Formula | Issue |
|----------|---------|-------|
| `routing.py:26` | `f = dispatch.mean()` sums to **top_k**, not 1 | Balanced gives L=K, not 1.0 |
| `routing.py:1090` | CV formula `importance_cv + load_cv` | Different from Switch |
| `zmoe.py:661` | L2 `(f_e - 1/E)²` | Double-counts with router loss |
| `load_balance.py:131` | **Correct Switch formula** | **EXISTS BUT UNUSED** |

**P5 Test Failure**: Uses OLD `routing.load_balance_loss` — expects L(uniform)≈0 but gets L=K

### P6: SPPQ MSE ≤ Δ²/4
**Status: ❌ INCORRECT**

- `SPPQEngine._quantize_parameter`: dequantizes back to float32, stores float32
- **Memory**: No savings (4 bytes/param before and after)
- **Progressive**: `_update_target_bits` is `pass` — schedule never applied

### P7: SPPQ compression ratio
**Status: ❌ FALSE** — Reports 1.0× because `QuantizationMetrics.compute_final` uses `self.average_bits=32`

### P8: state_dict round-trip exact
**Status: ✅ PROVEN** — max\|Δlogits\| = 0.0

### P9: Path simplex (sum=1, all≥0)
**Status: ✅ PROVEN** — softmax with uniform prior blend in prob space

### P10: Width monotonicity (complexity → wider)
**Status: 📊 EMPIRICALLY VERIFIED**

- Implementation: `logits + complexity * linspace(-1,1) * 3`
- Verified: bins tokens by complexity, checks `E[w|high_c] ≥ E[w|low_c]`

### P11: Causal mask strictness
**Status: ✅ PROVEN** — `tril` mask cached per seq_len

### P12: Tokenizer round-trip
**Status: ✅ PROVEN** — 65k vocab, decode(encode(x)).startswith(x)

### P13: Expert-shard storage savings
**Status: ⚠️ PARTIALLY PROVEN**

- Theory: RAM = K × shard, Disk = E × shard, saving = (E-K)/E
- test_mode breaks verification (1 dummy expert)
- Analytical: For E=64, K=2: saving 96.9% (verified in test)

### P14: Scaling-law adherence
**Status: 📊 EMPIRICALLY VERIFIED**

- Active% non-increasing: 15.69% → 13.57% → 10.61% → 7.68% → 6.72% → 4.47%
- **But**: `target_active_ratio` is hardcoded per model size, not emergent

---

## Phase 3: Previously Disputed Areas — Deep Dive

### P5 Load-Balance Loss — First-Principles Derivation

**Switch Transformer Formula** (Fedus et al., 2021):
```
f_e = (1/N) Σ_i I(expert_i selected for token i)  -- sums to K
P_e = (1/N) Σ_i p_i(e)                            -- sums to 1
L_lb = E * Σ_e f_e * P_e
```

| Routing | f_e | P_e | L_lb |
|---------|-----|-----|------|
| Uniform top-K | K/E | 1/E | **K** |
| Uniform top-1 | 1/E | 1/E | **1.0** |
| Collapse to expert 0 | K (for e=0) | 1 (for e=0) | **E×K** |

**The new `load_balance.py` computes this correctly. But it's NOT USED:**
- `AdaptiveRouter.forward` (line 642) → old `routing.py:26`
- `AdaptiveRouter._compute_balancing_loss` (line 1090) → CV formula
- `ShardedExpertFabric._compute_load_balance_loss` (zmoe.py:661) → L2 formula
- `zeroModel.forward` (line 614) zeros model-level LB when `unify_load_balance=True`, but router-internal LB remains

---

### Eval-Mode Pathway Collapse

**Problem**: At inference (`deterministic=True`), all axes use `argmax` → every token picks same pathway/width/expert.

**v0.5 Hack** (`eval_routing_noise=0.15`): Adds deterministic Gumbel noise to logits.

**Critique**: This is a **hack**, not a solution:
- Same input → same routing (fixed seed)
- No guarantee of diversity across batch
- Root cause: STE training doesn't incentivize diverse hard selections

**Proper Fix**: 
- Train with entropy regularization (already have `path_div_weight`, `width_div_weight`)
- At inference: use **soft routing with temperature** + top-k execution, not hard argmax
- Or implement **eval-time temperature** (`router_temperature > 0` at eval)

---

### Conditional FLOPs Accounting

**Current**: `OpCounter` hooks on `nn.Linear` + analytical SlicedFFN FLOPs.

**Missing**: attention matmuls (q*k^T, attn*v), SSM state updates, convs, embeddings, LN, CoT, merger, LM head.

**Verdict**: FLOPs numbers are **lower bounds**, useful only for relative comparison.

---

### Scaling-Law Assumptions

**Claim**: "active% non-increasing with scale" — verified empirically.

**Reality**: `target_active_ratio` is **hardcoded** per model size (0.1 for 10M-500M, 0.05 for 7B). Not emergent.

---

### CoT Behavior

**Status: DEAD CODE during pre-training**

```python
# model.py:129-131
self.cot = InternalLatentCoT(config)
self._freeze_cot()  # requires_grad=False

# model.py:442-446
cot_vector_seq = torch.zeros(batch_size, seq_length, total_cot_dim)  # ZEROS
# Never calls self.cot(...)
```

All CoT parameters (projections, GRU, gates) are dead weight.

---

### SPPQ Guarantees

| Claim | Reality |
|-------|---------|
| Progressive quantization | `_update_target_bits = pass` — never applied |
| Memory savings | Dequantizes to float32 — **0% savings** |
| Int8/int4 storage | Stores float32 — fake quantization |
| MSE bound | Tested on math utils, not actual quantized weights |

---

### Expert Sharding

| Claim | Reality |
|-------|---------|
| Disk sharding works | ✅ `ExpertDiskManager` saves/loads experts |
| LRU cache works | ✅ `LRUExpertCache` with capacity |
| test_mode=False verified | ✅ `phase10_moe_stress.py` 14/14 PASS |
| **Restart bug** | ❌ JSON keys are strings, `load_expert` checks `if int not in dict` — **ValueError** |
| **Training bug** | ❌ `load_expert` calls `expert.eval()` unconditionally — disables dropout in experts |

---

## Phase 4: P5 Load-Balance — Correct Formulation & Tests

### Correct Switch Loss (in `load_balance.py:131-177`)

```python
def load_balance_loss_switch(router_probs, expert_indices, num_experts):
    f = compute_load_fractions(expert_indices, num_experts, top_k)  # sums to 1
    p = router_probs.mean(dim=0)  # sums to 1
    return num_experts * (f * p).sum()
```

### Required Test Matrix

| Test Case | Expected L_lb |
|-----------|---------------|
| Uniform top-1 (f_e=1/E, p_e=1/E) | **1.0** |
| Uniform top-2 (f_e=2/E, p_e=1/E) | **2.0** ( = K, not 1.0!) |
| Complete collapse (top-1) | **E** |
| Complete collapse (top-K) | **E×K** |
| Partial concentration (2 experts) | Between K and E×K |
| Vary E ∈ {2,4,8,16,64} | Scales correctly |
| Vary batch size | Invariant to B |
| Vary token count | Invariant to T (LLN) |
| Hard top-1 (deterministic) | Same bounds |
| Soft routing (τ > 0) | Same bounds |

---

## Phase 5: Architecture vs. Implementation

| What XORZEN **Mathematically Guarantees** | What v0.4 **Implementation Demonstrates** |
|-------------------------------------------|------------------------------------------|
| SlicedFFN nested slicing → FLOPs ∝ W | Model-level width sparsity: 0.500× FLOPs ratio ✅ |
| Top-k expert routing → only K experts/token | Disk-sharded MoE executes only K experts (test_mode=False) ✅ |
| Top-k pathway routing → only K pathways/token | Sparse dispatch at inference; STE at training ✅ |
| Switch load-balance loss → uniform routing | Correct formula exists, **not integrated** ❌ |
| ZOH discretization for SSM | **Broken** — B not discretized ❌ |
| Depth sparsity → per-token layer skipping | **Only batch-level skip at inference** ❌ |
| Active params ≈ target_active_ratio | Heuristic formula, not measured ❌ |
| Progressive quantization → memory savings | **Fake** — dequantizes to float32 ❌ |
| CoT → cross-layer reasoning | **Frozen, zero signal** ❌ |

---

## Phase 6: Remediation Plan

### A. Critical Blockers (Must fix before v0.4 is scientifically defensible)

| # | Issue | Fix |
|---|-------|-----|
| 1 | **P5 load-balance not integrated** | Replace all 3 old LB losses with `load_balance_loss_switch` from `load_balance.py` |
| 2 | **SSM B not discretized (ZOH violation)** | Fix `SSMPathway.forward`: `Bv = dt * B_proj(...)` or use `discretize_zoh` |
| 3 | **Depth sparsity fake at training** | Either document "no FLOP savings at training" or implement `forward_with_depth` |
| 4 | **CoT dead code** | Either wire it or delete it |
| 5 | **SPPQ fake quantization** | Either implement real int8 packing or remove claims |

### B. Important Weaknesses

| # | Issue | Fix |
|---|-------|-----|
| 6 | **Eval-mode collapse** | Replace Gumbel-noise hack with eval-time temperature + entropy reg |
| 7 | **Active params not measured** | Add backward-hook instrumentation to count params-with-grad |
| 8 | **Expert restart bug** | Fix int/string key mismatch in `ExpertShardManager._load_metadata` |
| 9 | **Expert eval() bug** | Remove unconditional `expert.eval()` in `ExpertDiskManager.load_expert` |
| 10 | **FLOPs counter incomplete** | Add hooks for attention matmuls, SSM, conv, LN |

### C. Cosmetic/Documentation

| # | Issue | Fix |
|---|-------|-----|
| 11 | Docstring "O(L log L) parallel scan" | Update to "chunked sequential scan, O(T/C) Python iterations" |
| 12 | "Theorem 1/2" in math_utils | Remove or actually state/prove theorems |
| 13 | "Born rule" comment | Remove or fix to `|ψ|²` |
| 14 | `target_active_ratio` as achieved | Label as "configured target" not "achieved" |

### D. New Tests Required

```python
# 1. Load balance loss correctness
def test_switch_lb_uniform_top1():
    E=8; probs=torch.full((100,E),1/E); idx=torch.arange(E).repeat(100//E,1)[:,:1]
    L = load_balance_loss_switch(probs, idx, E)
    assert abs(L.item() - 1.0) < 0.01

def test_switch_lb_uniform_top2():
    E=8; K=2; probs=torch.full((100,E),1/E)
    idx=torch.tensor([[i%8, (i+1)%8] for i in range(100)])
    L = load_balance_loss_switch(probs, idx, E)
    assert abs(L.item() - 2.0) < 0.01  # = K, not 1.0!

def test_switch_lb_collapse():
    E=8; probs=torch.zeros(100,E); probs[:,0]=1.0
    idx=torch.zeros(100,1,dtype=torch.long)
    L = load_balance_loss_switch(probs, idx, E)
    assert abs(L.item() - 8.0) < 0.01

# 2. SSM ZOH B discretization
def test_ssm_zoh_b_discretized():
    # Compare SSMPathway recurrence against discretize_zoh reference
    pass

# 3. Depth sparsity FLOP measurement
def test_depth_sparsity_flops():
    # With test_mode=False, measure FLOPs with all-active vs min-depth masks
    # Assert reduction ≈ (max_depth - min_depth) / max_depth
    pass

# 4. Active parameter counting
def test_active_params_measured():
    # Backward hooks to count params receiving grad
    # Compare to config.estimate_active_parameters()
    pass

# 5. Eval-mode diversity
def test_eval_routing_diversity():
    # Run model on 100 tokens, check pathway/width/expert entropy > threshold
    pass

# 6. SPPQ memory
def test_sppq_memory_savings():
    # After quantize, check param.element_size() == 1 (int8)
    pass
```

### E. Claims to Rewrite

| Current Claim | Scientifically Defensible Replacement |
|---------------|--------------------------------------|
| "Depth routing genuinely skips computation" | "Depth routing at inference skips layers when the entire batch has mask=0; at training it uses STE blend (no FLOP savings). Per-token layer skipping not yet implemented." |
| "Active parameter percentage matches target_active_ratio" | "Active parameter percentage is estimated via a heuristic formula (`estimate_active_parameters()`). The configured `target_active_ratio` is an architectural target, not a measured quantity." |
| "SPPQ provides progressive quantization with memory savings" | "SPPQ currently implements fake quantization (simulates quantization noise on float32 tensors). Real int8/int4 packing and progressive schedule are not implemented." |
| "CoT maintains reasoning state across layers" | "Internal Latent CoT module exists but is frozen during pre-training and contributes zero signal to the forward pass. Enable with `model.enable_cot()` for fine-tuning." |
| "SSM uses proper ZOH discretization for both A and B" | "SSM uses ZOH discretization for A (`A_bar = exp(dt*A)`). B is currently NOT discretized (uses raw projection). Fix pending." |
| "Load balancing loss prevents expert collapse" | "A correct Switch Transformer load-balance loss is implemented in `load_balance.py` but not yet integrated into the training loop. The active losses (router CV, model L2) have known scaling issues with top-K > 1." |
| "Xorzen decouples capacity from compute" | "Mathematically, if MoE executes only K of E experts per token, capacity/compute ≈ (always_on + E×p_e) / (always_on + K×p_e). This is analytically true but only empirically verified for E > K with test_mode=False." |

### F. v0.5 Roadmap (Prioritized)

| Priority | Task | Category |
|----------|------|----------|
| 1 | Integrate `load_balance_loss_switch` everywhere, delete 3 old LB implementations | **Correctness** |
| 2 | Fix SSM ZOH B discretization | **Correctness** |
| 3 | Implement real depth sparsity (call `forward_with_depth`) or document limitation | **Scientific Validity** |
| 4 | Wire CoT or remove it | **Scientific Validity** |
| 5 | Fix expert restart bug (int/string keys) | **Correctness** |
| 6 | Fix expert eval() bug in training | **Correctness** |
| 7 | Replace eval Gumbel-noise with temperature-based soft routing | **Architectural Importance** |
| 8 | Add active-param measurement via backward hooks | **Measurable Performance** |
| 9 | Complete FLOPs counter (attention, SSM, conv) | **Measurable Performance** |
| 10 | Real SPPQ int8 packing or remove claims | **Correctness** |

---

## Conclusion

XORZEN v0.4 demonstrates **genuine improvements in conditional compute** (SlicedFFN, pathway sparsity, cost-aware routing) with **measured 2× loss reduction and 22.8% FLOPs reduction**. However, **four of seven core sparsity claims are either false or unverified** in the implementation. The codebase contains a correct Switch load-balance loss (`load_balance.py`) that is completely disconnected from the training loop, and the SSM discretization violates the claimed ZOH mathematics.

**Recommendation**: Fix Critical Blockers A1–A5 before any release claiming "verified architecture" or "genuine conditional compute." The benchmarks are real; the claims need to match the implementation.