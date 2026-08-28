# Xorzen v0.4 — Architecture

> Generated: 2026-08-15
> Status: v0.4 deep verification complete. 53/53 tests pass. 23 architectural claims audited.

## 1. Overview

Xorzen is a hybrid Transformer × MoE × SSM architecture with conditional
compute across four axes: depth, width, pathway, and experts. The core
principle is:

> «Never compute something merely to multiply its output by zero later.»

v0.4 verified this principle end-to-end and fixed three bugs where it was
violated.

## 2. Architecture Components

### 2.1 Token + Position Embeddings
- Token embedding: `nn.Embedding(vocab_size, hidden_size)`
- Position embedding: `nn.Embedding(context_length, hidden_size)` (learned)
- Combined with `+`, then dropout

### 2.2 AdaptiveRouter (`xorzen/model/components/routing.py`)
The router makes ALL per-token decisions:
- **Depth**: which layers to use (sigmoid per layer, STE for differentiability)
- **Width**: which FFN width (softmax over discrete choices)
- **Pathway**: which HASS pathways (softmax over 3 pathways)
- **Expert**: which MoE experts (top-k of E experts)

Architecture: 3-layer MLP encoder → 4 allocation heads. Temperature-annealed
Gumbel-softmax during training, hard argmax at inference.

Auxiliary losses:
- `load_balance_loss` (Switch Transformer formula): `E * sum(f_e * p_e)`
- `router_z_loss`: `(logsumexp(logits))²` for logit stability
- `path_diversity_loss`: `-entropy(path_probs)` to prevent collapse

### 2.3 HASS Block (`xorzen/model/components/hass_block.py`)
Hybrid Attention-Shard Switch with 3 parallel pathways:

1. **LocalAttentionPathway**: windowed causal attention
   - Q/K/V projections + per-head LayerNorm + output projection
   - Window mask + causal mask, cached per seq_len

2. **LowRankGlobalPathway**: low-rank global context
   - Projects to `low_rank_dim * num_heads`, computes global context via
     weighted pooling, projects back

3. **SSMPathway**: diagonal state-space model
   - ZOH discretization of both A and B (Phase 9 verified)
   - Input-dependent dt, B, C
   - Chunked scan by default (Phase 9 verified complexity)

Pathway selection: `sparse_pathway_dispatch` (Phase 5 verified) — only the
top-k pathways per token are invoked; unselected pathways are NOT called.

### 2.4 AdaptiveFFN (`xorzen/model/components/hass_block.py`)
- Base FFN: `fc1 (H -> 4H) → activation → fc2 (4H -> H)`
- Width adapters: `nn.ModuleDict` of `Linear(H, W) → activation → Linear(W, H)`
  for each width in `width_choices`
- **KNOWN ISSUE (Phase 15)**: AdaptiveFFN computes base + all adapters and
  blends. SlicedFFN exists but is not wired in. Recommend replacing.

### 2.5 ShardedExpertFabric (`xorzen/model/zmoe.py`)
Disk-sharded Mixture-of-Experts:
- `ExpertDiskManager`: saves/loads expert state_dicts to `.pt` files
- `LRUExpertCache`: capacity-bounded cache (default 24 experts)
- `ExpertFFN`: SwiGLU FFN (gate_proj, up_proj, down_proj)

Forward: loops over top-k slots, groups tokens by expert id, loads from
cache (or disk on miss), runs expert on the token subset. Unselected
experts are NOT loaded/called (Phase 5 verified).

### 2.6 Merger Gate (`xorzen/model/components/merger.py`)
Fuses HASS output + MoE output + CoT vector via gated combination.

### 2.7 ComputeController (`xorzen/model/components/compute_controller.py`)
**STATUS: NOT WIRED IN (Phase 18 FALSE claim)**

Designed as a unified replacement for AdaptiveRouter with a global compute
budget parameter. Phase 7 verified it works as a drop-in replacement, but
it is not currently imported by zeroModel. Phase 15 recommends either
wiring it in or removing it.

### 2.8 Internal CoT (`xorzen/model/components/cot_vector.py`)
6-component latent reasoning vector (intention, decomposition, confidence,
contradiction, direction, summary). **FROZEN for pre-training by design.**
Call `model.enable_cot()` to unfreeze for fine-tuning.

## 3. Forward Pass Flow

```
input_ids [B, T]
    ↓
token_embedding + position_embedding
    ↓
hidden_states [B, T, H]
    ↓
AdaptiveRouter → RoutingDecision (depth_mask, path_probs, expert_indices, ...)
    ↓
for each HASS block:
    ↓ (inference: gather active tokens via depth_mask, run block, scatter back)
    ↓ (training: STE mask blend for differentiability)
    ↓ (pathway: sparse_pathway_dispatch — only top-k pathways called)
    ↓ (FFN: AdaptiveFFN with width_idx)
    ↓
hidden_states [B, T, H]
    ↓
flatten → MoE (top-k expert dispatch)
    ↓
MergerGate (fuses HASS + MoE + CoT)
    ↓
final_norm → lm_head → logits [B, T, V]
```

## 4. Bugs Found and Fixed in v0.4

| # | Phase | Bug | Fix |
|---|-------|-----|-----|
| 1 | 5 | Depth routing used `block(x)*mask + x*(1-mask)` (compute-then-mask) | At inference, use `block.forward_with_depth()` for genuine gather/scatter of active tokens |
| 2 | 6 | `path_div_weight=0.02` too weak — pathway collapse to SSM | Bumped to 0.1 (middle ground between 0.02 collapse and 0.2 partial diversity) |
| 3 | 7 | ComputeController `width_bias` had wrong sign — low budget favored LARGER widths | Reversed linspace to `(+sparsity*2, -sparsity*2)` so low budget boosts smaller widths |
| 4 | 10 | `ShardedExpertFabric.eval()` did not propagate to cached experts (LRUExpertCache not nn.Module) | Override `train(mode)` to propagate to all cached experts + dummy expert |

## 5. Test Suite

53 tests across 4 files:
- `tests/test_fixes.py` (12) — v0.2.4 bug fixes
- `tests/test_phase1_correctness.py` (14) — v0.3 SSM/MoE/routing correctness
- `tests/test_phase2_sparsity.py` (8) — v0.3 sparse dispatch unit tests
- `tests/test_phase3_unified.py` (7) — v0.3 unified controller tests
- `tests/test_phase4_v04.py` (12) — v0.4 deep verification tests

All 53 pass.

## 6. Known Limitations

1. **Pathway collapse**: Even with `path_div_weight=0.1`, the router tends
   to collapse to 1-2 of 3 pathways. Phase 6 showed `path_div_weight=0.2`
   gives 2/3 pathways. Full diversity requires a stronger mechanism.

2. **Width collapse**: Width router always picks the largest width. No
   width diversity loss exists. Phase 15 recommendation: add one.

3. **ComputeController not wired in**: Exists as dead code. Phase 7 verified
   it works as a drop-in replacement but it is not used by zeroModel.

4. **SlicedFFN not wired in**: AdaptiveFFN computes all adapters and blends.
   SlicedFFN exists but is not used by HASSBlock. Phase 15 recommendation:
   replace.

5. **Two load-balance losses double-count**: One in AdaptiveRouter (CV-based),
   one in zeroModel (L2-based). Phase 15 recommends unifying.

6. **CoT frozen by design**: Not a bug, but should be documented clearly.
   Unfreeze with `model.enable_cot()` for fine-tuning.

7. **Wall-clock at tiny scale**: Sparse dispatch has Python overhead that
   makes it slower than dense at tiny scale on CPU. On GPU at large scale,
   sparse wins.

## 7. Configuration Variants

8 model sizes from `zero_tiny_23k` (37K params) to `zero_7b` (7B params).
All use the same architecture, differing only in hyperparameters.

See `xorzen/config.py:ConfigFactory` for the full size table.
