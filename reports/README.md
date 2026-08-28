# Xorzen v0.2.4 — Reports & Audits

This directory contains all verification, audit, and benchmark reports produced for Xorzen v0.2.4.

## Top-level reports

| File | Description |
|---|---|
| `xorzen_v0.2.4_fix_report.md`         | End-to-end report for the three bug fixes (SPPQ quantization, 65k tokenizer path, active-% logger). Includes root-cause analysis, fix descriptions, regression tests, and benchmark before/after. |
| `xorzen_verified_proofs.md`           | 14 architectural properties (P1–P14) with empirical observation, theorem statement, and rigorous proof for each. Verification suite: 26 checks, 26 PASS. |
| `xorzen_v0.2.4_benchmark_report.pdf`  | Generated PDF benchmark report covering model catalog, compute, memory, quantization, and tokenizer storage. |

## `audit/` subdirectory — adversarial audit artifacts

These files were produced by the **adversarial audit** (Part 1 + Part 2 of the audit programme). The audit re-derives every claim independently from the implementation and classifies each as PROVEN / EMPIRICALLY VERIFIED ONLY / PARTIALLY PROVEN / INCORRECT / UNTESTED.

| File | Description |
|---|---|
| `audit/p1_p14.md`           | Human-readable adversarial audit of properties P1–P14. Classifications: 1 PROVEN, 1 EMPIRICALLY VERIFIED ONLY, 5 INCORRECT, 1 PARTIALLY PROVEN, 6 UNTESTED. Includes corrected theorems and concrete failure evidence. |
| `audit/p1_p14.json`          | Machine-readable JSON evidence for each property classification (includes source snippets, runtime measurements, and theorem-violation analysis). |
| `audit/flops.json`           | Independent FLOPs/throughput audit per `zero` variant. Compares analytical estimate vs. framework-reported estimate vs. measured tokens/sec. |
| `audit/param_counts.json`    | Independent parameter-count audit per `zero` variant. Breaks down total / trainable / active params and verifies against `config.estimate_active_parameters()`. |

## Reading order

1. Start with `xorzen_v0.2.4_fix_report.md` for the bug-fix context.
2. Then `xorzen_verified_proofs.md` for the original (sympathetic) verification of P1–P14.
3. Then `audit/p1_p14.md` for the **adversarial** re-audit — several properties that were marked PASS in the sympathetic verification are re-classified as INCORRECT or UNTESTED once the implementation is re-derived independently.
4. `audit/flops.json` and `audit/param_counts.json` provide the raw numbers behind the audit conclusions.

## Reproducing the audit

All scripts that generated these reports live under `../scripts/` and `../scripts/audit/`. See `../scripts/README.md` for the execution order.
