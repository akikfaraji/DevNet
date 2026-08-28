"""
Regression tests for the three bugs fixed in Xorzen v0.2.4:

  Bug 1 — SPPQ quantization (`SPPQEngine.apply_fake_quantization` referenced
          `self.engine` which does not exist; the public API names
          `SPPQQuantizer` / `QuantizationConfig` were missing).
  Bug 2 — 65k tokenizer registry path mismatch (metadata.json pointed at
          `zarx_agi_tokenizer_65k.json` but the file on disk is
          `xorzen_agi_tokenizer_65k.json`).
  Bug 3 — Active-parameter percentage logger reported `~0.0%` because the
          init-time formula only counted
          `top_k_experts * hidden_size * expert_hidden_multiplier` and
          omitted the *2 for input/output projection plus every always-on
          component (embeddings, HASS blocks, router, merger, CoT, LM head).

Run with:

    pytest tests/test_fixes.py -v
"""
import os
import sys
import json
import io
import re
import logging
from pathlib import Path

import pytest

os.environ.setdefault("XORZENX_VERBOSE", "0")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _silence_logger():
    """Quiet the xorzen logger so test output stays readable."""
    try:
        from xorzen.utils.logger import get_logger
        ul = get_logger()
        underlying = getattr(ul, "logger", None) or ul
        if hasattr(underlying, "setLevel"):
            underlying.setLevel(logging.WARNING)
    except Exception:
        pass


_silence_logger()


# ---------------------------------------------------------------------------
# Bug 1: SPPQ quantization
# ---------------------------------------------------------------------------

def test_bug1_sppq_public_api_imports():
    """`SPPQQuantizer`, `QuantizationConfig`, `QuantizationType` must be importable."""
    from xorzen.utils.sppq import (
        SPPQQuantizer, QuantizationConfig, QuantizationType,
        SPPQEngine, SPPQ,
    )
    assert QuantizationType.SYMMETRIC.value == "symmetric"
    assert SPPQQuantizer is not None
    assert QuantizationConfig is not None


def test_bug1_sppq_quantizer_round_trip():
    """`SPPQQuantizer.calibrate()` + `apply_quantization()` must actually quantize."""
    import torch
    import xorzen
    from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig, QuantizationType

    model = xorzen.zero_1M(test_mode=True)
    n_pre = sum(p.numel() for p in model.parameters())

    cfg = QuantizationConfig(
        bits=8,
        quantization_type=QuantizationType.SYMMETRIC,
        observe_iterations=2,
    )
    quantizer = SPPQQuantizer(model, cfg)

    # Run a few forward passes to observe
    x = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)

    # Calibrate must not raise
    calib_stats = quantizer.calibrate()
    assert calib_stats, "calibrate() returned empty stats"
    assert all(s["bits"] == 8 for s in calib_stats.values())

    # apply_quantization must not raise and must modify parameters in place
    pre_w = next(model.parameters()).detach().clone()
    result = quantizer.apply_quantization()
    post_w = next(model.parameters()).detach()
    assert result["n_quantized"] > 0
    assert not torch.allclose(pre_w, post_w), "apply_quantization did not modify parameters"

    # Model must still be runnable
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape[0] == x.shape[0]

    # n_params is unchanged (we fake-quantize in place, no removal)
    n_post = sum(p.numel() for p in model.parameters())
    assert n_pre == n_post


def test_bug1_sppq_engine_apply_fake_quantization_no_attribute_error():
    """`SPPQEngine.apply_fake_quantization()` must not reference a missing `self.engine`."""
    import torch
    import xorzen
    from xorzen.utils.sppq import SPPQEngine, SPPQ

    model = xorzen.zero_1M(test_mode=True)

    # SPPQEngine: must not raise AttributeError
    engine = SPPQEngine(model, {
        "enabled": True, "bits": 4, "quant_type": "symmetric",
        "per_channel": True, "min_updates": 0, "quant_levels": [32, 16, 8, 4],
    })
    engine.apply_fake_quantization()

    # SPPQ wrapper: must not raise AttributeError when delegating to engine
    sppq = SPPQ(model, {
        "enabled": True, "bits": 4, "quant_type": "symmetric",
        "min_updates": 0, "quant_levels": [32, 16, 8, 4],
    })
    sppq.apply_fake_quantization()


def test_bug1_sppq_engine_step_no_argument_mismatch():
    """`SPPQEngine.step()` must not call `_update_parameter_state(name, None)`."""
    import xorzen
    from xorzen.utils.sppq import SPPQEngine

    model = xorzen.zero_1M(test_mode=True)
    engine = SPPQEngine(model, {
        "enabled": True, "bits": 4, "quant_type": "symmetric",
        "min_updates": 0, "quant_levels": [32, 16, 8, 4],
    })
    # Must not raise TypeError
    engine.step()


# ---------------------------------------------------------------------------
# Bug 2: 65k tokenizer registry path
# ---------------------------------------------------------------------------

def test_bug2_tokenizer_registered_under_correct_name():
    """The 65k tokenizer must be registered under `xorzen_agi_tokenizer_65k`."""
    import xorzen
    available = xorzen.list_pretrained()
    assert "xorzen_agi_tokenizer_65k" in available, (
        f"xorzen_agi_tokenizer_65k missing from registry; got {available}"
    )


def test_bug2_65k_tokenizer_loads_from_package_layout():
    """Loading the 65k tokenizer from the installed package must succeed."""
    import xorzen
    tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
    assert tk.get_vocab_size() == 65000
    enc = tk.encode("Hello world, the XORZENX framework is here.")
    assert isinstance(enc, list) and len(enc) > 0
    dec = tk.decode(enc)
    assert dec.startswith("Hello"), f"round-trip failed: {dec!r}"


def test_bug2_tokenizer_file_path_exists():
    """The path returned by `get_pretrained_path` must actually exist on disk."""
    import xorzen
    path = xorzen.get_pretrained_path("xorzen_agi_tokenizer_65k")
    assert Path(path).exists(), f"tokenizer file does not exist: {path}"


def test_bug2_no_machine_specific_paths_in_meta():
    """The 65k meta.json must not contain machine-specific Windows paths."""
    import xorzen.tokenizer.loader as L
    meta_path = Path(L.__file__).parent / "pretrained" / "xorzen_agi_tokenizer_65k.meta.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    # training_files (the array of absolute paths) must be removed
    assert "training_files" not in data, "meta.json still contains training_files"
    # No Windows-style absolute paths anywhere in the JSON
    text = json.dumps(data)
    assert "C:\\\\" not in text, "machine-specific Windows path leaked into meta.json"


def test_bug2_loader_fuzzy_resolution_for_legacy_typo():
    """Old typo name `zarx_agi_tokenizer_65k` must still resolve via fuzzy match."""
    from xorzen.tokenizer.loader import (
        _resolve_tokenizer_path, _get_pretrained_tokenizers_dir,
    )
    pretrained_dir = _get_pretrained_tokenizers_dir()
    path = _resolve_tokenizer_path(pretrained_dir, "zarx_agi_tokenizer_65k")
    assert path is not None, "fuzzy resolution failed for legacy typo"
    assert path.exists()
    # The resolved file must be the actual xorzen_agi_tokenizer_65k.json
    assert path.name == "xorzen_agi_tokenizer_65k.json"


# ---------------------------------------------------------------------------
# Bug 3: Active-parameter percentage logger
# ---------------------------------------------------------------------------

def _capture_init_log(model_factory):
    """Run `model_factory()` and capture the xorzen logger's `info()` calls.

    The xorzen logger is a custom XORZENXLogger (not stdlib logging), so we
    monkey-patch its `info` method to record messages during model init.
    """
    from xorzen.utils.logger import get_logger
    ul = get_logger()
    records = []
    original_info = ul.info

    def _capture_info(module, message, data=None):
        records.append(f"[{module}] {message}")
        original_info(module, message, data)

    ul.info = _capture_info
    try:
        model = model_factory()
    finally:
        ul.info = original_info
    return model, records


def _extract_active_pct(log_lines):
    """Find the `Top-K/N experts active per token (~X.X% ...)` line and parse the %."""
    for line in log_lines:
        if "experts active per token" in line:
            m = re.search(r"~(\d+(?:\.\d+)?)%", line)
            if m:
                return float(m.group(1)), line.strip()
    return None, None


def test_bug3_init_logger_reports_nonzero_pct():
    """The init logger must report > 1% active params, not 0.0%."""
    import xorzen
    model, logs = _capture_init_log(lambda: xorzen.zero_1M(test_mode=True))
    pct, line = _extract_active_pct(logs)
    assert pct is not None, f"active-pct line not found in init log:\n{chr(10).join(logs)}"
    assert pct > 1.0, f"active pct still reports ~0.0%: {pct}% (line: {line})"


def test_bug3_init_logger_matches_config_estimate():
    """The init logger percentage must match `config.estimate_active_parameters()`."""
    import xorzen
    model, logs = _capture_init_log(lambda: xorzen.zero_1M(test_mode=True))
    pct, _ = _extract_active_pct(logs)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = 100.0 * int(model.config.estimate_active_parameters()) / max(1, trainable)
    assert abs(pct - expected) < 0.1, f"init pct {pct} != expected {expected:.2f}"


def test_bug3_active_vs_total_vs_expert_distinction():
    """The init log must preserve the distinction between total, trainable, and active."""
    import xorzen
    model, logs = _capture_init_log(lambda: xorzen.zero_1M(test_mode=True))
    line = next((l for l in logs if "experts active per token" in l), None)
    assert line is not None
    # The new log line mentions trainable, active, and total explicitly
    assert "trainable" in line
    assert "active" in line
    assert "total" in line
    # Sanity: trainable <= total; active <= total
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    active = int(model.config.estimate_active_parameters())
    assert trainable <= total
    assert active <= total


if __name__ == "__main__":
    # Allow `python tests/test_fixes.py` for systems without pytest
    sys.exit(pytest.main([__file__, "-v"]))
