# Xorzen v0.2.4 — Patched

Advanced Hybrid Transformer Framework with MoE, SSM, and Adaptive Routing.

This branch contains targeted fixes for three bugs identified during the
v0.2.4 rigorous benchmark:

| # | Bug | Symptom | Root cause | Fix |
|---|-----|---------|------------|-----|
| 1 | SPPQ quantization | `'SPPQEngine' object has no attribute 'engine'` and `cannot import name 'SPPQQuantizer'` | `SPPQEngine.apply_fake_quantization()` delegated to a non-existent `self.engine`; the public API names `SPPQQuantizer` / `QuantizationConfig` were missing; `SPPQEngine.step()` called `_update_parameter_state(name, None)` with an extra arg | Fix `apply_fake_quantization` to walk parameters directly; add `QuantizationConfig` and `SPPQQuantizer` as the real public API; fix the `step()` call signature |
| 2 | 65k tokenizer registry path | `TokenizerLoadError: ...zarx_agi_tokenizer_65k.json` (file not found) | Typo in `metadata.json` and `xorzen_agi_tokenizer_65k.meta.json`: registry name was `zarx_agi_tokenizer_65k` but the actual file is `xorzen_agi_tokenizer_65k.json`; meta.json also contained machine-specific Windows training paths | Rename the registry entry to `xorzen_agi_tokenizer_65k`; strip machine-specific paths; harden `_register_pretrained_tokenizers` with a fuzzy resolver that falls back to suffix matching so historical typos don't silently break loading |
| 3 | Active-parameter % logger | Init log printed `~0.0% of N params` | Init-time formula was `top_k_experts * hidden_size * expert_hidden_multiplier`, omitting the `* 2` for input+output projections of each expert and every always-on component (embeddings, HASS blocks, router, merger, CoT, LM head) | Replace with `config.estimate_active_parameters()`, the same accounting used by the runtime estimator and the smoke test |

See `tests/test_fixes.py` for regression coverage of all three fixes.

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
pytest tests/test_fixes.py -v
```

All 12 tests must pass:

```
tests/test_fixes.py::test_bug1_sppq_public_api_imports PASSED
tests/test_fixes.py::test_bug1_sppq_quantizer_round_trip PASSED
tests/test_fixes.py::test_bug1_sppq_engine_apply_fake_quantization_no_attribute_error PASSED
tests/test_fixes.py::test_bug1_sppq_engine_step_no_argument_mismatch PASSED
tests/test_fixes.py::test_bug2_tokenizer_registered_under_correct_name PASSED
tests/test_fixes.py::test_bug2_65k_tokenizer_loads_from_package_layout PASSED
tests/test_fixes.py::test_bug2_tokenizer_file_path_exists PASSED
tests/test_fixes.py::test_bug2_no_machine_specific_paths_in_meta PASSED
tests/test_fixes.py::test_bug2_loader_fuzzy_resolution_for_legacy_typo PASSED
tests/test_fixes.py::test_bug3_init_logger_reports_nonzero_pct PASSED
tests/test_fixes.py::test_bug3_init_logger_matches_config_estimate PASSED
tests/test_fixes.py::test_bug3_active_vs_total_vs_expert_distinction PASSED
```

## Known limitations

* The v0.2.4 smoke benchmark used `test_mode=True` "dummy" experts for the
  small CPU models. This is convenient for verifying wiring but is **not**
  equivalent to a production-scale GPU benchmark. Treat the small-CPU
  speedup numbers (1.9–2.4× slower than dense baselines) as indicative of
  routing/SSM overhead on CPU, not as a production throughput claim.
* `bench_sppq_shards` in `benchmarks/03_extra_benchmark.py` constructs a
  `QuantizationState` with kwargs that do not exist on the actual
  dataclass (`quantization_type`, `original_shape`, `original_dtype`,
  `quantized_data`). This is a benchmark-script API mismatch, not a
  library bug — the library's SPPQ quantization itself works correctly
  (proven by `tests/test_fixes.py::test_bug1_sppq_quantizer_round_trip`).
  The benchmark script was not modified per the project policy of not
  manipulating benchmark numbers.
* The runtime `_estimate_active_params` over-counts because it does not
  apply `target_active_ratio` or `width_factor` to the HASS block
  contribution. The analytical `config.estimate_active_parameters()`
  (used by the now-fixed init logger and by the smoke test's `active_pct`
  field) does apply these factors and produces the ~7.8–13% range that
  matches the claimed ~9.4% figure. Both estimators are left in place
  unchanged.

## License

XORZENX Proprietary License. See `LICENSE` for full terms.
