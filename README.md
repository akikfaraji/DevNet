# Xorzen v0.4 — Genuine Conditional Compute + Cost-Aware Routing

Advanced Hybrid Transformer × MoE × SSM framework with **genuine** per-token
conditional compute (depth / width / pathway / expert), cost-aware routing,
and a unified load-balance loss.

## v0.4 architecture improvements (vs v0.3)

The v0.3 architecture had several "compute-then-mask" patterns and dead code.
v0.4 acts on the Phase 14/15 ablation findings and Phase 18 audit:

| # | Change | Evidence | Effect |
|---|--------|----------|--------|
| A | Replace `AdaptiveFFN` with `SlicedFFN` in `HASSBlock` (config.`use_sliced_ffn=True`) | Phase 5 proved standalone SlicedFFN gives proportional FLOPs scaling. Phase 18: AdaptiveFFN was compute-then-blend. | Genuine **model-level** width sparsity (FFN FLOPs ratio 0.500 = 16/32). |
| B | Add `width_diversity_loss` (config.`width_div_weight=0.1`) | Phase 6: width router always picks largest (collapse). | Width router now picks smaller widths for easy tokens. |
| C | Unify double-counting load-balance losses (config.`unify_load_balance=True`) | Phase 15: v0.3 had TWO LB losses (router Switch-formula + model L2). | Only the standard Switch formula is used. |
| D | Cost-aware routing in `AdaptiveRouter` (config.`cost_aware_routing=True`) | Phase 18: ComputeController was dead code. | Per-axis compute cost estimates bias routing toward `compute_budget`. |
| E | Bump `path_div_weight` from 0.1 to 0.2 | Phase 6: 0.2 gives 2/3 pathways active (vs 1/3 at 0.1). | Pathway collapse broken in training mode. |
| F | Remove `adaptive_halting.py` (dead code, 158 lines) | Phase 18: not wired into `zeroModel`. | Cleaner codebase. |

### Measured impact (200 training steps, H=32, 3 layers, 4 experts/top-2)

| Metric | v0.3 (OLD) | v0.4 (NEW) | Delta |
|---|---|---|---|
| Final loss | 0.0132 | 0.0064 | **2× lower** |
| Loss reduction % | 99.6% | 99.8% | +0.2pp |
| Token accuracy | 98.33% | 98.33% | same |
| FLOPs | 27,557,888 | 21,266,432 | **-22.8%** |
| Runtime (ms) | 7.93 | 7.58 | **-4.5%** |
| Path entropy | 0.405 | 0.672 | +0.267 |
| Params | 231,195 | 227,979 | -3,216 |

### Model-level conditional compute verification (4/4 PASS)

| Test | What it verifies | Result |
|---|---|---|
| Width sparsity | Forcing width=16 vs 32 changes model-level FFN FLOPs by 0.500× | **PASS** |
| Depth sparsity | Gather-scatter at inference reduces FLOPs by 31.8% (all vs none active) | **PASS** |
| Pathway sparsity | Forcing all tokens → SSM: local=0, low_rank=0, ssm=3 | **PASS** |
| MoE top-k | Forcing top-1 expert: experts_used=1 | **PASS** |

## Claims audit (v0.4 final)

24 architectural claims audited. Status distribution:

| Status | v0.3 | v0.4 |
|---|---|---|
| PROVEN | 12 | **13** |
| EMPIRICALLY VERIFIED | 5 | **7** |
| PARTIALLY TRUE / IMPLEMENTED | 3 | 1 |
| FALSE | 2 | 1 |
| FALSE (by design) | 1 | 1 |
| REMOVED | 0 | 1 |

Key improvements: SPARSE-2 and DEAD-3 went from "standalone only" to "PROVEN
(model-level)"; ROUTER-1 went from PARTIALLY IMPLEMENTED to EMPIRICALLY
VERIFIED; adaptive_halting.py was REMOVED (dead code); a new claim ROUTER-4
(width diversity loss) was added and verified.

See `reports/v04/phase_v04_final_audit.json` for the full audit.

## Reports & audits

All verification, audit, and benchmark reports live under [`reports/`](reports/).
v0.4 reports are under [`reports/v04/`](reports/v04/):

| File | Description |
|---|---|
| `reports/v04/phase4_overfit.json` | Minimal overfit training test (loss 3.48 → 0.006, 99.8% reduction). |
| `reports/v04/phase_v04_old_vs_new.json` | Head-to-head: v0.3 vs v0.4 architecture. |
| `reports/v04/phase_v04_model_level_sparse.json` | Model-level conditional compute (4/4 PASS). |
| `reports/v04/phase_v04_profiling.json` | Subsystem breakdown + old vs new runtime. |
| `reports/v04/phase_v04_final_audit.json` | Final 24-claim audit with v0.3→v0.4 status deltas. |
| `reports/v04/phase9_ssm_deep_validation.json` | SSM numerical equivalence (8 configs, fp32+bf16). |
| `reports/v04/phase10_moe_stress.json` | Disk-sharded MoE lifecycle (14/14 PASS). |
| `reports/v04/phase6_router_stability.json` | Router collapse experiments. |
| `reports/v04/phase7_compute_budget.json` | Compute budget validation. |

## Installation

```bash
# From the repository root:
pip install -e .

# Required runtime dependencies are declared in pyproject.toml and include:
#   torch, transformers, tokenizers, sentencepiece, einops, tqdm, numpy,
#   pydantic, psutil
```

The package installs in editable mode so changes to `xorzen/` are picked up
immediately.

## Quick smoke test

```python
import xorzen

# 1. Models
m = xorzen.zero_1M(test_mode=True)
print(f"models: {xorzen.list_models()}")

# 2. Tokenizer (both 65k and 10k variants load from the installed package)
tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
print(f"vocab_size: {tk.get_vocab_size()}")  # -> 65000

# 3. SPPQ quantization
from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig, QuantizationType
import torch
cfg = QuantizationConfig(bits=8, quantization_type=QuantizationType.SYMMETRIC, observe_iterations=2)
q = SPPQQuantizer(m, cfg)
x = torch.randint(0, 1000, (2, 32))
with torch.no_grad():
    for _ in range(2): _ = m(x)
q.calibrate()
q.apply_quantization()
print("quantized OK")
```

## Running the regression tests

```bash
# Full suite (59 tests, ~18s on CPU)
pytest tests/ -v

# v0.4-specific tests only
pytest tests/test_phase4_v04.py -v
```

All 59 tests must pass. Key test groups:

- `tests/test_fixes.py` — v0.2.4 bug fixes (12 tests)
- `tests/test_phase1_*.py` — architecture property tests
- `tests/test_phase2_sparsity.py` — standalone sparsity tests
- `tests/test_phase3_unified.py` — ComputeController + adaptive halting (halting tests removed in v0.4)
- `tests/test_phase4_v04.py` — v0.4 architecture + conditional compute (20 tests, including 8 new v0.4 tests)

## v0.4 config flags (new)

```python
from xorzen.config import ConfigFactory, ModelSize

cfg = ConfigFactory.get_config(ModelSize.NANO_10M)

# v0.4 architecture (all default True except width_div_weight which auto-disables
# when num_widths == 1)
cfg.use_sliced_ffn       = True   # genuine per-token width sparsity in HASSBlock
cfg.width_div_weight     = 0.1    # entropy reg for width router
cfg.path_div_weight      = 0.2    # entropy reg for path router (was 0.1)
cfg.unify_load_balance   = True   # drop the double-counting model-level L2 loss
cfg.cost_aware_routing   = True   # bias routing toward compute_budget
cfg.compute_budget       = 1.0    # 1.0 = full compute, 0.25 = quarter compute
```

To restore v0.3 behavior for ablation comparison:

```python
cfg.update(
    use_sliced_ffn=False, width_div_weight=0.0, path_div_weight=0.1,
    unify_load_balance=False, cost_aware_routing=False,
)
```

## Known limitations

* At tiny scale (H=32), wall-clock speedup (4.5%) is much smaller than FLOPs
  reduction (22.8%) due to Python dispatch overhead. On GPU at production
  scale, sparse execution would win more.
* In eval mode (deterministic=True), the hard top-1 pathway/width selection
  still collapses to 1 pathway/1 width per token. The soft probs are diverse
  (path entropy 0.672 vs 0.405 in v0.3), but argmax is winner-take-all. To
  fully break eval-mode collapse, would need KL-to-uniform loss or eval-mode
  temperature scheduling.
* `ComputeController.py` is still present as a standalone module (not wired
  into `zeroModel`). The cost-aware functionality it was supposed to provide
  is now in `AdaptiveRouter.forward()` (config.`cost_aware_routing=True`).
  Future cleanup could remove `ComputeController.py` entirely.
* CoT (Chain-of-Thought) is frozen by design during pre-training. Call
  `model.enable_cot()` to unfreeze for fine-tuning.
* The v0.2.4 smoke benchmark used `test_mode=True` "dummy" experts for the
  small CPU models. This is convenient for verifying wiring but is **not**
  equivalent to a production-scale GPU benchmark.
* `bench_sppq_shards` in `benchmarks/03_extra_benchmark.py` constructs a
  `QuantizationState` with kwargs that do not exist on the actual
  dataclass. This is a benchmark-script API mismatch, not a library bug.

## License

XORZENX Proprietary License. See `LICENSE` for full terms.
