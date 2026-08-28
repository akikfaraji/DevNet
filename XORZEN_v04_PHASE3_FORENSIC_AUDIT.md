# XORZEN v0.4 — Phase 3 Forensic Audit Report

**Objective**: Determine whether XORZEN is architecturally correct and ready for staged training.
**Method**: Systematic code audit with minimal reproduction tests for every suspected issue.
**Classification scheme**: BUG / DESIGN LIMITATION / EXPECTED BEHAVIOR / UNTESTED / NO ISSUE

---

## Audit Scope

Every component was audited for: model initialization, forward-pass tensor shapes, gradient flow, STE implementations, auxiliary losses, loss scaling, depth/width/pathway/expert routing, MoE dispatch, SSM scans, causal attention, tokenizer→model→loss data flow, CoT lifecycle, checkpoint save/load, mixed precision, numerical stability, edge cases, and reproducibility.

### Files Audited

| File | Lines | Component |
|------|------:|------------|
| `xorzen/models/zero/model.py` | 1,582 | Main model forward pass, loss computation |
| `xorzen/model/components/routing.py` | 1,693 | AdaptiveRouter, all 4 routing axes, STE |
| `xorzen/model/components/hass_block.py` | 1,411 | HASS block, 3 pathways, sparse dispatch |
| `xorzen/model/components/sliced_ffn.py` | 255 | Per-token width selection |
| `xorzen/model/components/sparse_dispatch.py` | 291 | Sparse pathway dispatch, STE |
| `xorzen/model/components/merger.py` | 302 | HASS + MoE + CoT fusion |
| `xorzen/model/components/cot_vector.py` | 467 | Chain-of-Thought vector |
| `xorzen/model/components/load_balance.py` | 177 | Switch Transformer load balance loss |
| `xorzen/model/components/ssm_scan.py` | 307 | SSM scan implementations |
| `xorzen/model/ssm.py` | 288 | S4D kernel, SSM block |
| `xorzen/model/zmoe.py` | 894 | ShardedExpertFabric (disk MoE) |
| `xorzen/config.py` | 2,272 | ModelConfig, TrainingConfig, ConfigFactory |
| `xorzen/training/trainer.py` | 678 | XORZENXTrainer training loop |
| `xorzen/training/checkpoint.py` | 751 | CheckpointManager save/load |
| `xorzen/models/zero/variants.py` | 255 | Model size presets |
| `xorzen/models/zero/model_fixed.py` | 39 | Previous fix patch (verified merged) |

---

## Reproduced Bugs (7)

### 🔴 BUG-CRITICAL-1: `features` tensor injected as loss term
**File**: `model.py` lines 600–603
**Claim**: The router stores a `features` tensor `[B, T, enc_dim]` in `routing_decision.auxiliary`. The model's auxiliary loss loop blindly adds all `requires_grad=True` tensors to `routing_loss`, including this non-scalar tensor.
**Root cause**: `model.py` iterates `routing_decision.auxiliary.items()` and adds any `requires_grad` tensor without checking `numel() == 1` or filtering by known loss keys.
**Evidence**: `features shape=(2, 8, 128), requires_grad=True, is_scalar=False`
**Impact**: The `.mean()` guard at line 605 converts to scalar, but `d(mean(features))/d(router_params)` injects a spurious gradient that pushes the feature encoder output toward zero. This actively harms routing quality during pre-training.
**Fix**: Filter auxiliary dict to known loss keys or require `numel() == 1`:
```python
# Option A: Known keys
_LOSS_KEYS = {'load_balance_loss', 'router_z_loss', 'path_div_loss', 'width_div_loss'}
for aux_key, aux_loss_val in routing_decision.auxiliary.items():
    if aux_key in _LOSS_KEYS and isinstance(aux_loss_val, torch.Tensor) and aux_loss_val.requires_grad:
        routing_loss = routing_loss + aux_loss_val

# Option B: Scalar-only
for aux_key, aux_loss_val in routing_decision.auxiliary.items():
    if isinstance(aux_loss_val, torch.Tensor) and aux_loss_val.requires_grad and aux_loss_val.numel() == 1:
        routing_loss = routing_loss + aux_loss_val
```
**Regression test**: Verify that `routing_loss.numel() == 1` BEFORE the `.mean()` guard, and that `features` is excluded.

---

### 🔴 BUG-CRITICAL-2: Disk-sharded experts never receive optimizer updates
**File**: `zmoe.py` lines 590–614
**Claim**: When `shard_experts=True` and `test_mode=False`, experts are loaded from disk into plain Python objects, not registered as `nn.Module` submodules. They never appear in `model.parameters()` and thus never receive optimizer updates.
**Root cause**: `ExpertDiskManager.load_expert()` creates a fresh `ExpertFFN` object that is stored in the LRU cache but never registered via `nn.Module.__setattr__`.
**Evidence**: `param_names=[], has_expert_params=False` (sharded mode, test_mode=False)
**Impact**: In production mode, expert weights are effectively **frozen**. Only the router trains. The model cannot learn expert specialization.
**Fix options**:
1. **Register cached experts dynamically**: Override `__setattr__` in `ShardedExpertFabric` to register loaded experts.
2. **Maintain per-expert optimizer states**: Before evicting an expert from cache, apply accumulated gradients and save optimizer state.
3. **For pre-training smoke test**: Use `test_mode=True` (all experts in memory, properly registered). This is the recommended approach for the initial training stage.
**Regression test**: Verify `any('expert' in n or 'ffn' in n for n, _ in model.named_parameters())` is True when `test_mode=True`.

---

### 🔴 BUG-CRITICAL-3: `forward_with_depth` corrupts attention/SSM across batch boundaries
**File**: `hass_block.py` lines 1108–1112
**Claim**: `forward_with_depth` concatenates all active tokens (across batch elements and sequence positions) into a flat 1D tensor, then processes them as a single contiguous sequence.
**Root cause**: `active_x = x[active_mask]` flattens `[B, T, H]` to `[N_active, H]`, then `active_x.unsqueeze(0)` treats them as one sequence.
**Evidence**: `uses_flatten=True` — confirmed by source code analysis.
**Impact**: 
- **LocalAttentionPathway**: Tokens from batch element B attend to tokens from batch element A.
- **SSMPathway**: The scan propagates hidden state across batch boundaries.
- **Mitigation**: This code path only runs at **inference** (`if not self.training`), so training correctness is NOT affected.
**Fix**: Process each batch element independently:
```python
for b in range(B):
    batch_mask = active_mask[b]  # [T]
    if batch_mask.any():
        batch_x = x[b:b+1]  # [1, T, H]
        # ... process with proper sequence structure
```
**Regression test**: Verify that two identical-input batch elements produce identical outputs in `forward_with_depth`.

---

### 🔴 BUG-CRITICAL-4: Sharded experts invisible to checkpoint save/load
**File**: `zmoe.py` (state_dict), `checkpoint.py` (save/load)
**Claim**: Disk-sharded experts are not part of `model.state_dict()`. `CheckpointManager.save()` uses `model.state_dict()`, so expert weights are never saved. On restore, experts reinitialize from scratch.
**Root cause**: Sharded experts are stored in `ExpertDiskManager`'s LRU cache as plain objects, not as `nn.Module` submodules.
**Evidence**: `state_dict keys=[], has_expert_keys=False` (sharded mode)
**Impact**: Any checkpoint saved in production mode loses all expert weights permanently.
**Fix**: `ShardedExpertFabric.state_dict()` should include cached expert weights. `load_state_dict()` should restore them. Alternatively, `CheckpointManager` should explicitly serialize the expert shard directory.
**Regression test**: `model.state_dict()` should contain expert weight keys when experts are loaded.

---

### 🟠 BUG-MEDIUM-1: SlicedFFN has no STE — width router disconnected from LM loss
**File**: `sliced_ffn.py` lines 150–169
**Claim**: SlicedFFN's training path uses hard argmax on `width_probs` and returns `y_hard` with no STE connection to `width_probs`.
**Root cause**: Line 169: `return y_hard` — the hard path output has no dependency on `width_probs`.
**Evidence**: `width_probs.grad is None: True, grad norm: 0.000000`
**Impact**: The width router receives **zero** gradient from the LM loss through the forward computation graph. It is trained ONLY by auxiliary diversity/entropy losses. For the width routing feature (a flagship of the architecture) to work, auxiliary losses must be enabled and properly weighted.
**Fix options**:
1. Add a STE: `y = y_hard.detach() + (width_probs * y_soft_expected).sum() - (width_probs * y_soft_expected).sum().detach()` — but this requires computing all widths, defeating sparsity.
2. Accept the limitation and ensure auxiliary width diversity/entropy losses are strong enough (current approach).
3. Use the `width_multiplier` path (legacy AdaptiveFFN) which does have gradient.
**Regression test**: With auxiliary losses disabled, verify `width_router` params have zero gradient from LM loss.

---

### 🟠 BUG-MEDIUM-2: `_init_weights` clobbers padding row of tied embeddings
**File**: `model.py` lines 195, 199, 289–293
**Claim**: When `tie_word_embeddings=True`, `self.lm_head.weight` is set to `self.token_embedding.weight` (line 195). Then `self.apply(self._init_weights)` (line 199) runs `_init_weights` on the `lm_head` (nn.Linear), which re-initializes ALL rows including the padding row.
**Root cause**: `nn.Embedding._init_weights` zeros the padding row first, but `nn.Linear._init_weights` then re-initializes it with N(0, 0.02).
**Evidence**: `pad_row is_zero: False, same_tensor: True, pad_row norm: 0.154035`
**Impact**: The padding row of the shared weight has random values instead of zeros. At generation time, `pad_token_id` produces non-zero logits. The LM loss correctly ignores pad positions via `ignore_index`, so training signal is unaffected.
**Fix**: After `self.apply(self._init_weights)`, re-zero the padding row:
```python
if config.tie_word_embeddings and getattr(config, 'pad_token_id', None) is not None:
    self.lm_head.weight.data[config.pad_token_id].zero_()
```
**Regression test**: After init, `model.token_embedding.weight[pad_token_id]` should be all zeros.

---

### 🟠 BUG-MEDIUM-4 (end-to-end confirmation): Width router zero gradient from LM loss
**File**: `sliced_ffn.py` + `model.py` + `routing.py`
**Claim**: End-to-end: with `use_sliced_ffn=True` and all auxiliary losses disabled, the width router receives zero gradient from the LM loss.
**Evidence**: All 6 width_router parameters show `grad_norm=0.00000000`.
**Impact**: Same as BUG-MEDIUM-1, confirmed at the full model level.

---

## Design Limitations (3)

### 🟡 DESIGN-MEDIUM-1: LowRankGlobalPathway is non-causal
**File**: `hass_block.py` lines 282–288
**Claim**: The LowRankGlobalPathway computes `softmax(QK^T)` without a causal mask.
**Evidence**: `has_causal_mask: False, has_softmax: True`
**Impact**: During teacher-forced training, future target tokens are visible through this pathway. At inference, future tokens don't exist, creating a train/inference mismatch. When `pathway_top_k >= 2`, this pathway may be selected and contaminate the causal training signal.
**Assessment**: The name 'global' suggests this may be intentional for capturing bidirectional context. However, for autoregressive language modeling, this is problematic.
**Recommendation**: Add a causal mask (`torch.tril`) or document that this pathway is intended for bidirectional tasks only.

---

### 🟡 DESIGN-LOW-1: Pad tokens flow through router/MoE
**File**: `model.py` lines 426, 450–454
**Claim**: Padding tokens receive non-zero hidden states from position embeddings and flow through the entire pipeline (router, HASS blocks, MoE, merger).
**Evidence**: `path_entropy: 1.0942, expert_entropy: 1.3777` for all-pad input.
**Impact**: Routing/load-balance losses receive noise from pad token expert assignments, diluting the routing signal. Compute is wasted on pad tokens.
**Recommendation**: Zero out hidden states for pad positions after embedding: `hidden_states = hidden_states * attention_mask.unsqueeze(-1).float()`.

---

### 🟡 DESIGN-LOW-2: Taylor expansion inconsistency
**File**: `ssm.py` line 165 vs `ssm_scan.py` line 102
**Claim**: Two SSM ZOH implementations use different Taylor expansion orders: `1 + z/2` (first-order) vs `1 + z/2 + z²/6` (second-order).
**Evidence**: Confirmed by source code.
**Impact**: Minor numerical difference. Both are convergent. The `ssm_scan.py` version is more accurate.
**Recommendation**: Align `ssm.py` to use second-order Taylor to match `ssm_scan.py`.

---

## Verified Correct (No Issue)

### ✅ Depth routing STE gradient flow
**Evidence**: Depth router parameters receive non-zero gradients from LM loss (grad norms 0.000035 – 0.002580). The STE at `routing.py` line 705 correctly routes gradient through `probs` → `depth_router` params.

### ✅ Pathway routing STE (sparse_dispatch.py)
The `topk_pathway_mask` STE at lines 76–84 is textbook-correct. Verified by code analysis.

### ✅ Expert routing gradient flow
Expert weights are soft (continuous), providing full gradient through `expert_router` params. No STE needed.

### ✅ Loss computation and right-shift
Standard causal LM shift: `logits[..., :-1, :]` vs `labels[..., 1:]`. `ignore_index` correctly set to `pad_token_id`.

### ✅ CoT lifecycle
CoT is frozen at init (`_freeze_cot()`), zeroed during forward, consistency loss zeroed. Matches spec.

### ✅ SSM parallel scan (Hillis-Steele)
Correct inclusive prefix scan. Verified against sequential reference.

### ✅ Causal convolution cropping
`Conv1d(kernel=3, padding=2)` → crop `[:, :L, :]` is the standard Mamba left-pad-and-crop.

### ✅ Merger gate gradient flow
No `.detach()` calls in the merger. Gradients flow through gate weights to all three pathways.

### ✅ model_fixed.py fixes already merged
All auxiliary loss handling fixes from `model_fixed.py` are incorporated into `model.py`.

### ✅ Load balance loss (canonical)
`load_balance.py` correctly implements the Switch Transformer formula `L = N * Σ f_e * P_e`.

---

## Additional Audited Areas (No Bugs Found)

| Area | Finding |
|------|----------|
| Gradient checkpointing | Correctly wraps block forward with `torch.utils.checkpoint.checkpoint` |
| Gradient clipping | Applied via `torch.nn.utils.clip_grad_norm_` before optimizer step |
| Mixed precision setup | Correctly uses `torch.autocast` with appropriate dtype |
| Expert dispatch | Top-k loop correctly groups tokens by expert, sums weighted outputs |
| Residual connections | Pre-norm pattern: `x = x + dropout(sublayer(ln(x)))` |
| S4D complex A matrix | Correct for S4 — complex diagonal captures oscillatory modes |
| cfloat conversion | `torch.cfloat` = `complex64`, works correctly on CPU |
| Temperature annealing | `0.99^(step/1000)` — slow decay, may be intentional |
| Complexity bias | Correctly zeros out when `compute_budget=1.0` (default) |
| Router z-loss | Correctly prevents unbounded logits |
| MoE load balance loss weight | Properly scaled at 0.0001 for small models |

---

## PRE-TRAINING BLOCKERS

Issues that **must** be fixed before serious training:

| # | ID | Severity | Description | Fix Complexity |
|---|-----|----------|-------------|-----------------|
| 1 | **BUG-CRITICAL-1** | CRITICAL | `features` tensor added as loss term, injecting spurious gradient | **Trivial** — 2-line filter in model.py |
| 2 | **BUG-CRITICAL-2** | CRITICAL | Disk-sharded experts never receive optimizer updates | **N/A for pre-training** — use `test_mode=True` |
| 3 | **BUG-MEDIUM-2** | MEDIUM | Tied embedding padding row clobbered | **Trivial** — 2-line fix after `_init_weights` |
| 4 | **BUG-MEDIUM-1** | MEDIUM | SlicedFFN no STE — width router disconnected from LM loss | **Acceptable** if auxiliary losses enabled (current default) |
| 5 | **BUG-CRITICAL-3** | CRITICAL | `forward_with_depth` corrupts attention/SSM across batches | **N/A for training** — inference-only code path |
| 6 | **BUG-CRITICAL-4** | CRITICAL | Sharded experts invisible to checkpoints | **N/A for pre-training** — use `test_mode=True` |

**Actual pre-training blockers**: Only #1 and #3 require code changes before pre-training can proceed. Issues #2, #5, #6 affect production/inference mode (sharded experts) but NOT the pre-training smoke test (which uses `test_mode=True`). Issue #4 is mitigated by auxiliary losses being enabled by default.

**Recommended fix order for pre-training**:
1. Fix BUG-CRITICAL-1 (2 minutes) — filter auxiliary dict
2. Fix BUG-MEDIUM-2 (2 minutes) — re-zero padding row
3. Verify existing tests still pass
4. Proceed with training smoke test

---

## POST-TRAINING INVESTIGATIONS

Issues that can only be meaningfully evaluated after a trained model exists:

| # | ID | Severity | Question | How to Evaluate |
|---|-----|----------|----------|-----------------|
| 1 | **BUG-MEDIUM-1** | MEDIUM | Is the width diversity loss sufficient to train good width routing without LM loss gradient? | After training: check width distribution. If collapsed to max width, STE or stronger loss is needed. |
| 2 | **DESIGN-MEDIUM-1** | MEDIUM | Does the non-causal LowRankGlobalPathway degrade autoregressive quality? | Compare training with pathway_top_k=1 (local+SSM only) vs top_k=2 (includes global). |
| 3 | **DESIGN-LOW-1** | LOW | How much do pad tokens pollute routing statistics? | Measure routing entropy with/without pad masking on real data. |
| 4 | Width routing quality | — | Does the width router learn meaningful per-token width choices? | Compare per-token width distribution before/after training. |
| 5 | Depth routing quality | — | Does the depth router learn to skip easy layers? | Measure layer activation frequency before/after training. |
| 6 | Expert specialization | — | Do different experts learn different functions? | Analyze expert weight divergence after training. |
| 7 | CoT gate collapse | — | Does the merger's CoT gate collapse to near-zero during pre-training? | Check `g_cot` gate weight magnitude. |
| 8 | Routing entropy progression | — | Do routing entropies decrease (specialize) or stay uniform? | Track entropy over training steps. |
| 9 | Numerical stability | — | Do any components produce NaN/Inf during training? | Monitor loss, gradients, hidden state norms. |
| 10 | Sharded expert training | — | Can experts actually learn in sharded mode? | Requires fixing BUG-CRITICAL-2, then training with sharding enabled. |

---

## Recommended Exact Order for Remaining Work

```
1. Fix BUG-CRITICAL-1 (filter auxiliary dict in model.py)        [2 min]
2. Fix BUG-MEDIUM-2  (re-zero padding row after _init_weights) [2 min]
3. Run existing test suite to verify no regressions              [5 min]
4. Write training smoke test (test_mode=True, NANO_1M config)    [30 min]
5. Run smoke test: verify loss decreases, gradients flow,     
   no NaN/Inf, checkpoints save/restore, frozen params stay  [10 min]
6. Analyze routing statistics from smoke test                   [10 min]
7. Update audit documentation with smoke test results           [10 min]
8. (Later) Fix BUG-CRITICAL-3 (forward_with_depth) for inference [30 min]
9. (Later) Fix BUG-CRITICAL-2 & 4 (sharded expert training)    [2-4 hours]
10. (Later) Investigate DESIGN-MEDIUM-1 (causal global pathway)  [1 hour]
11. (Later) Address DESIGN-LOW-1 (pad token masking)            [15 min]
12. (Later) Address DESIGN-LOW-2 (Taylor alignment)             [5 min]
```

---

## Additional Findings (Not Bugs, Noteworthy)

### CoTAuxiliaryLoss is dead code
`CoTAuxiliaryLoss` (cot_vector.py:327-394) is defined but never instantiated anywhere in the codebase. Its `token_complexity_head` and `next_token_head` would add significant parameters if registered. Can be removed or ignored.

### NANO_1M lacks width routing
`ConfigFactory` sets `width_choices=(64,)` (single entry) for NANO_1M. Width routing requires 2+ choices. The first config to exercise width routing is NANO_10M. For the pre-training smoke test, use a custom config with `width_choices=(h//2, h)`.

### SSM D initialization
`S4DKernel.D` is initialized as `randn` instead of `ones` (Mamba standard). May slow early training convergence but the model should eventually learn correct D values.

### SSM state not cached across forward calls
The parallel scan always starts from h_{-1}=0. No stateful inference mode. This makes autoregressive generation O(T²) per new token instead of O(T) with KV caching.

### Redundant double layer normalization
HASSBlock applies LN before pathways, then each pathway applies its own LN again. Redundant but not harmful.

### SSMPathway normalizes scan state
`ln_state` is applied to the SSM hidden state after the scan. This discards magnitude information from the state, reducing SSM expressiveness. Not present in standard Mamba/S4.

---

## Reproduction

All bugs were reproduced with minimal tests in:
`scripts/audit_phase3/reproduce_bugs.py`

Results saved to:
`scripts/audit_phase3/bug_reproduction_results.json`

Run with:
```bash
python scripts/audit_phase3/reproduce_bugs.py
```
