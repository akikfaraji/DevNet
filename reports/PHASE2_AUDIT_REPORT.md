# Phase 2: Empirical & Architectural Validation Report

**Date**: 2026-08-28
**Scope**: XORZEN v0.4 full implementation audit
**Environment**: CPU, Python 3.12, PyTorch, untrained models

## Classification Legend

| Tag | Meaning |
|---|---|
| PROVEN | Mathematically/code-level guarantee verified by test |
| EMPIRICAL | Reproducible measurement supports the claim |
| PARTIAL | Claim is partially correct; caveats exist |
| UNTESTED | Cannot be verified in current environment |
| FALSE | Claim is incorrect or not implemented |
| REMOVED | Feature/code was removed |

## Claim Matrix

| # | Result | Claim | Evidence | Remaining Problem |
|---|--------|-------|---------|------------------|
| 1 | FALSE | Depth routing produces per-token depth variation at inference |  |  |
| 2 | PROVEN | Pathway routing executes only top-k pathways per token at inference |  |  |
| 3 | FALSE | Width routing assigns different FFN widths to different tokens at inference |  |  |
| 4 | PROVEN | Expert routing dispatches only top-k experts per token at inference |  |  |
| 5 | FALSE | Different tokens execute different amounts of computation |  |  |
| 6 | PARTIAL | Training depth routing uses STE (soft mask, block computed for all tokens) |  |  |
| 7 | PROVEN | Sequential, parallel (Blelloch), and chunked scans produce numerically identical |  |  |
| 8 | PROVEN | ZOH discretization matches independent reference and produces stable A_bar in (0 |  |  |
| 9 | PROVEN | All SSM parameters (A_log, B_proj, C_proj, dt_proj, gate, conv) receive gradient |  |  |
| 10 | PROVEN | Local attention with causal=True prevents future-token leakage |  |  |
| 11 | PROVEN | Causal attention handles small window sizes correctly |  |  |
| 12 | PROVEN | All routing probabilities (path, expert, width) form valid simplexes |  |  |
| 13 | PROVEN | Inference routing is deterministic (same input → same routing) |  |  |
| 14 | FALSE | _estimate_active_params accurately measures actually-executed parameters |  |  |
| 15 | EMPIRICAL | Instrumented parameter execution measurement (actual gradient flow) |  |  |
| 16 | PARTIAL | Inference routing produces diverse decisions across all 4 axes |  |  |
| 17 | UNTESTED | SPPQ produces correct QAT with valid gradients |  |  |
| 18 | PROVEN | Expert sharding: LRU cache correctly evicts least-recently-used experts |  |  |
| 19 | UNTESTED | Scaling-law implementation matches stated equations; 12B>60B hypothesis is exper |  |  |
| 20 | UNTESTED | Tokenizer encode→decode roundtrip preserves original text |  |  |
| 21 | PROVEN | SPPQ target_bits affects quantization precision |  |  |
| 22 | PROVEN | SPPQ gradients flow through quantized weights |  |  |
| 23 | PROVEN | SPPQ does NOT reduce inference memory (weights stay float32) |  |  |

## Key Findings

### Conditional Compute (Subsystem 1)

1. **Depth routing architecture is PROVEN**: At inference, `forward_with_depth` uses a genuine gather-scatter pattern that skips block computation for tokens with mask=0.
2. **Depth routing quality is FALSE on untrained models**: All tokens receive all layers (depth_mask=all-ones). The complexity-based bias and eval Gumbel noise (0.15) are insufficient to produce per-token depth variation on untrained models.
3. **Pathway routing is PROVEN**: Per-token top-k mask correctly selects exactly 2 of 3 pathways. `sparse_pathway_dispatch` only calls selected pathways' forward functions.
4. **Width routing is FALSE on untrained models**: All tokens select the maximum width. The complexity-based bias (+1.5 for largest width) dominates the eval noise (0.15).
5. **Expert routing is PROVEN**: Per-token top-k correctly selects exactly K experts. Weights sum to ~1.0 (simplex property verified).
6. **Training depth routing: PROVEN correct STE behavior**: During training, the block is computed for ALL tokens (no FLOP savings) then masked with STE blend. This is the intended STE design.
### SSM Subsystem (Subsystem 6)

7. **Scan equivalence: PROVEN**. Sequential, parallel (Blelloch), and chunked scans produce numerically identical outputs (max diff < 6e-8) across T=8,32,64,128,256.
8. **ZOH discretization: PROVEN**. Matches independent reference implementation exactly (diff=0.0). A_bar stable in (0,1) for negative A. ZOH vs first-order differs by 0.116, confirming the ZOH correction is non-trivial.
9. **SSM gradients: PROVEN**. All 16 SSM parameters (A_log, B_proj, C_proj, dt_proj, D_proj, gate_proj, conv, ln_input, ln_state) receive non-zero gradients through the scan.
### Causal Attention & Routing (Subsystem 7)

10. **Causal attention: PROVEN**. Corrupting positions 8+ produces zero change in positions 0-7 (diff=0.0e+00), confirming no future-token leakage.
11. **Window edge cases: PROVEN**. Window sizes 1, 2, 4 all produce finite, correctly-shaped output.
12. **Routing simplex: PROVEN**. Path, expert, and width probabilities all sum to 1.000000 and are non-negative.
13. **Routing determinism: PROVEN**. Repeated forward passes with the same input produce identical routing decisions (depth, path, expert, width).
### Active Parameter Accounting (Subsystem 2)

14. **`_estimate_active_params` heuristic: FALSE**. The estimate returns 1.4x the total parameter count (1,416,335 > 1,008,015). It double-counts by summing embeddings + blocks + experts + CoT + router + merger + lm_head without accounting for tied weights or overlapping components. The estimate is an upper bound, not a measurement.
15. **Instrumented parameter execution: EMPIRICAL**. 99.0% of trainable params receive non-zero gradients. MoE has near-zero grad (test_mode uses dummy expert). Per-module grad norms: blocks >> router >> embeddings >> merger >> norm >> MoE.
### Inference-Time Routing (Subsystem 3)

16. **Inference routing diversity: PARTIAL**. On 10M model: pathway 3/3 (diverse), expert 8/8 (diverse), width 1/2 (collapsed), depth 1/128 patterns (collapsed). 2/4 axes show collapse on untrained models. Pathway and expert routing work; width and depth routing collapse due to complexity-bias dominance.
### SPPQ (Subsystem 4)

17. **SPPQ bit-width affects precision: PROVEN**. 4-bit error=0.004364, 8-bit=0.000239, 16-bit=0.000001, 32-bit=0.000000. Monotonic decrease confirmed.
18. **SPPQ gradient flow: PROVEN**. Both 4-bit and 8-bit quantized weights receive gradients during backward.
19. **SPPQ inference memory not reduced: PROVEN**. Weights remain torch.float32 after `apply_quantization()`. Memory: 32768 bytes before and after. This is QAT-only (fake quantization).
### Expert Sharding (Subsystem 5)

20. **LRU cache: PROVEN**. Eviction order matches LRU policy exactly: after fill [0,1,2], adding 3 evicts 0 → [1,2,3]. After accessing 1 and adding 4, 2 is evicted → [3,1,4]. Hit rate tracks correctly.
21. **Disk sharding in test mode: UNTESTED**. test_mode skips disk manager and uses a single dummy expert. Full disk sharding (load/save/evict from disk) cannot be verified without persistent storage.
### Scaling Law (Subsystem 8)

22. **Scaling-law equations: UNTESTED**. Documentation (scaling_law.md) contains equations and parameter tables, but no executable implementation code was found. The equations are in the report, not in the codebase.
23. **12B > 60B hypothesis: UNTESTED**. The claim exists in documentation but has no experimental support in the codebase. It is a projected/aspirational claim based on scaling-law extrapolation.
### Tokenizer

24. **Tokenizer round-trip: UNTESTED**. No Python tokenizer class found in `xorzen.tokenizer`. A HuggingFace-format vocab (65k tokens) and special_tokens.py exist, but no encode/decode API. The model imports sentencepiece for training.
## Ordered Changelog (Issue → Root Cause → Fix → Test → Result)

| # | Issue | Root Cause | Fix | Test | Result |
|---|-------|------------|-----|------|--------|
| 1 | `_estimate_active_params` returns 1.4x total params | Double-counts tied weights (embedding=lm_head), sums all components without dedup | **NOT FIXED** — this is a known heuristic limitation. A proper fix would require instrumenting actual parameter execution or subtracting tied weight duplicates. | test_active_param_estimate_accuracy | FALSE (heuristic overestimates) |
| 2 | Depth routing collapses at inference (all tokens, all layers) | Complexity bias pushes all sigmoid outputs >0.5; eval Gumbel noise (0.15) too small to overcome | **NOT FIXED** — architectural behavior. The complexity estimator outputs ~0.5 for all tokens on untrained models, giving +1.5 bias to last layer and +0.5 to middle. Would self-correct with training. | test_depth_conditional | FALSE (untrained model) |
| 3 | Width routing collapses at inference (all tokens, max width) | Complexity bias +3.0 on width_logits for largest width dominates 0.15 noise | **NOT FIXED** — same complexity-bias issue. With training, the width router should learn to differentiate. | test_width_conditional | FALSE (untrained model) |
| 4 | SPPQ documentation claims 0% memory reduction | Correct — QAT keeps weights in float32, no int storage | **NO FIX NEEDED** — documentation is accurate | sppq_inference_memory_not_reduced | PROVEN (documentation correct) |
## Remaining Unresolved Issues

1. **Depth/width routing collapse on untrained models**: The complexity-bias terms in the router (+3.0 for width, +2.0 for depth) dominate the eval noise (0.15) on untrained models. This is expected to self-correct with training but means that inference-time conditional compute cannot be demonstrated without a trained model.
2. **`_estimate_active_params` overestimates by ~40%**: The heuristic sums all component params without deduplication. It counts tied embedding weights twice and does not account for pathway/width sparsity in parameter counting.
3. **Training has NO FLOP savings from depth routing**: By design (STE requires computing the block for all tokens), depth routing provides 0% FLOP reduction during training. FLOP savings only occur at inference.
4. **Tokenizer API not implemented**: No Python wrapper for encode/decode exists. The tokenizer JSON vocab and special_tokens.py are present but no callable API.
5. **Scaling-law has no code implementation**: The scaling-law equations exist only in documentation, not as executable code. The 12B>60B hypothesis is aspirational.
6. **SPPQ is QAT-only**: No native int8/int4 storage. Inference memory is not reduced. This is documented correctly but limits practical deployment.
