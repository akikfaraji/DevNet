# XORZEN Post-Training Inference Forensic Audit

**Date**: 2026-08-29
**Checkpoint**: `experiments/staged_training_001/stage2_final.pt`
**Config**: NANO_1M (hidden=64, layers=3, heads=4, experts=2, top-1)

---

## Overall Verdict

**PARTIALLY WORKING - critical bugs found**

| Metric | Value |
|--------|-------|
| Checkpoint loaded | True |
| Config match | True |
| Reproducibility diff | 0.00e+00 |
| Inference works | True |
| Avg loss | 5.5383 |
| Logits finite | True |

## Routing Forensics

### Depth
- Active layers/token: 0.869 +/- 0.338
- Unique patterns: 2
- Entropy: 0.8353 bits

### Width
- Choices: (64,)
- Distribution: {0: 640}
- Entropy: 0.0000 bits

### Pathway
- Distribution: {'local': 0.1712, 'low_rank': 0.3055, 'ssm': 0.392}
- Entropy: 0.8998 bits (max=1.58)

### Expert
- Used: 2/2
- Entropy: 0.5839 bits

## Execution Trace

| Axis | Sparse | Evidence |
|------|--------|----------|
| Depth | True | 2 layers skipped |
| Pathway | True | {'local': 0, 'low_rank': 1, 'ssm': 1} |
| Width | False | 1 choices |
| Expert | False | 2/2 |

## Active Parameters

| Metric | Value |
|--------|-------|
| Total | 1,210,959 |
| Heuristic | 44,329 |
| Runtime | 1,601,231 |
| Active % | 132.2% |

## Compute

| Metric | XORZEN | Dense | Ratio |
|--------|---------|-------|-------|
| Latency | 7.82ms | 1.48ms | 0.19x |
| Throughput | 4474.9 tok/s | - | - |
| FLOPs | 0.0231 GFLOPs | - | - |
| RAM | 0.0 MB | - | - |

## Claim Matrix

| Claim | Description | Status |
|-------|-------------|--------|
| A | Per-token routing | **PROVEN** |
| B | Input-dependent routing | **FALSE** |
| C | Different paths execute | **PROVEN** |
| D | Skipped saves FLOPs | **PARTIAL** |
| E | Real inference savings | **FALSE** |
| F | Quality useful | **PROVEN** |

## Inference Correctness

| Check | Result |
|-------|--------|
| Causal masking | FAIL |
| SSM varies | PASS |
| Path weights sum=1 | PASS |
| Expert valid | PASS |
| Logits finite | PASS |

## Findings

### P1-1: Checkpoint loads successfully
- **PROVEN** / INFO
- Loaded experiments/staged_training_001/stage2_final.pt with 254 parameter tensors, 0 with NaN
- Verdict: Stage-2 checkpoint is intact and loadable

### P1-2: Deterministic inference
- **PROVEN** / INFO
- Max logit diff between identical forward passes: 0.00e+00
- Verdict: Model is deterministic in eval mode

### P2-1: Model produces valid inference outputs
- **PROVEN** / INFO
- Ran inference on 20 texts, avg loss=5.5383
- Verdict: Trained model works for forward pass

### P3-W1: Width routing degenerate at NANO_1M (single width choice)
- **DESIGN LIMITATION** / MEDIUM
- width_choices=(64,)
- Verdict: Width routing cannot demonstrate conditional compute at this config

### P3-R1: Routing is input-dependent
- **PROVEN** / INFO
- Depth: 0.0000, Width: 0.0000, Path: 0.0000, Expert: 0.0000
- Verdict: Different inputs produce different routing decisions

### P4-1: Pathway sparsity is genuine
- **PROVEN** / INFO
- pathway_top_k=2, calls={'local': 0, 'low_rank': 1, 'ssm': 1}
- Verdict: Unselected pathways are NOT executed

### P4-2: Depth sparsity is genuine
- **PROVEN** / INFO
- 2/3 layers fully skipped
- Verdict: Inference depth routing correctly skips computation

### P5-1: Active parameter estimation consistent
- **PROVEN** / INFO
- Heuristic=44,329, Runtime=1,601,231
- Verdict: ~132.2% of parameters active per forward

### P7-1: XORZEN is SLOWER than dense baseline
- **EMPIRICAL** / MEDIUM
- Dense=1.48ms, XORZEN=7.82ms
- Verdict: Routing overhead exceeds compute savings at this scale

### P10-1: Causal masking broken
- **BUG** / CRITICAL
- Early diff=7.99e-02
- Verdict: Information leaks from future to past

### P11-1: SPPQ not applied
- **UNTESTED** / LOW
- No quantized parameters found
- Verdict: SPPQ exists as utility but not in pipeline
