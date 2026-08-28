# Xorzen v0.2.4 — Scripts

Verification, audit, and benchmark scripts for Xorzen v0.2.4. All scripts are runnable from the repository root after `pip install -e .`.

## Top-level scripts

| File | Purpose |
|---|---|
| `reproduce_bugs.py`         | Reproduce the three fixed bugs (SPPQ `engine.engine`, 65k tokenizer path, active-% = 0.0%) and confirm they are fixed. |
| `serialization_check.py`    | Component integrity checks: model builds, forward pass works, state_dict round-trip is bit-exact. |
| `verify_architecture.py`    | 26-check verification suite for the 14 architectural properties (P1–P14). Outputs `verification_results.json`. |
| `verify_load_balance.py`    | Targeted verification of the Switch-Transformer load-balance loss bounds (P5). |
| `clean_tokenizer_meta.py`   | One-shot cleanup script that strips machine-specific Windows training paths from tokenizer metadata. |
| `verification_results.json` | Output of `verify_architecture.py` — 26 PASS/FAIL records with observed vs. expected values. |

## `audit/` subdirectory — adversarial audit scripts

| File | Purpose |
|---|---|
| `audit/audit_p1_p14.py`        | Re-derive each of P1–P14 independently from the implementation. Locates the exact source lines, re-derives the math, and classifies the property. Writes `../../reports/audit/p1_p14.{md,json}`. |
| `audit/audit_flops.py`         | Independent FLOPs and throughput audit per `zero` variant. Writes `../../reports/audit/flops.json`. |
| `audit/audit_param_counts.py`  | Independent parameter-count audit per `zero` variant. Writes `../../reports/audit/param_counts.json`. |
| `audit/01_smoke_test.py`       | Smoke benchmark: tiny/1M/10M/50M model build + forward + tokenizer + SPPQ. |
| `audit/02_full_benchmark.py`   | Full benchmark: all model sizes × (build, forward, SPPQ, dense baseline, storage). |
| `audit/03_extra_benchmark.py`  | Extra benchmark: expert-shard storage, SPPQ shard benchmark, compute extrapolation to GPU scales. |
| `audit/04_generate_report.py`  | Generate the PDF report (`../../reports/xorzen_v0.2.4_benchmark_report.pdf`) from benchmark JSON output. |

## Execution order (full reproduction)

```bash
pip install -e .

# 1. Confirm bugs are fixed
python scripts/reproduce_bugs.py

# 2. Confirm component integrity
python scripts/serialization_check.py

# 3. Run sympathetic verification (26 checks)
python scripts/verify_architecture.py
python scripts/verify_load_balance.py

# 4. Run benchmarks (smoke → full → extra)
python scripts/audit/01_smoke_test.py
python scripts/audit/02_full_benchmark.py
python scripts/audit/03_extra_benchmark.py

# 5. Generate PDF report from benchmark output
python scripts/audit/04_generate_report.py

# 6. Run adversarial audit (re-derives P1–P14 from implementation)
python scripts/audit/audit_param_counts.py
python scripts/audit/audit_flops.py
python scripts/audit/audit_p1_p14.py
```

All outputs land in `reports/` and `reports/audit/`.
