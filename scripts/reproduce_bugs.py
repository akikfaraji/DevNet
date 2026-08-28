"""
Reproduce all three bugs identified in the Xorzen v0.2.4 benchmark.

Bug 1: SPPQ quantization — `self.engine.engine` does not exist
Bug 2: 65k tokenizer registry path — registry points to a non-existent file
Bug 3: Active-parameter percentage logger — reports ~0.0%
"""
import os
os.environ["XORZENX_VERBOSE"] = "0"

import sys
import traceback
import json
from pathlib import Path

print("=" * 72)
print("XORZEN v0.2.4 — BUG REPRODUCTION")
print("=" * 72)

# ============================================================
# Bug 1: SPPQ quantization
# ============================================================
print("\n[BUG 1] SPPQ quantization — 'SPPQEngine' object has no attribute 'engine'")
try:
    from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig, QuantizationType
    print("  Import of SPPQQuantizer/QuantizationConfig: OK")
    import torch
    import xorzen
    model = xorzen.zero_1M(test_mode=True)
    cfg = QuantizationConfig(
        bits=8,
        quantization_type=QuantizationType.SYMMETRIC,
        observe_iterations=2,
    )
    quantizer = SPPQQuantizer(model, cfg)
    x = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        for _ in range(2):
            _ = model(x)
    quantizer.calibrate()
    quantizer.apply_quantization()
    print("  SPPQ quantization: OK")
except ImportError as e:
    print(f"  REPRODUCED: ImportError — {e}")
except AttributeError as e:
    print(f"  REPRODUCED: AttributeError — {e}")
except Exception as e:
    print(f"  REPRODUCED: {type(e).__name__} — {e}")
    traceback.print_exc()

# ============================================================
# Bug 2: 65k tokenizer registry path
# ============================================================
print("\n[BUG 2] 65k tokenizer registry path mismatch")
try:
    import xorzen
    available = xorzen.list_pretrained()
    print(f"  Available tokenizers in registry: {available}")
    pretrained_dir = Path(xorzen.tokenizer.loader.__file__).parent / "pretrained"
    print(f"  Pretrained dir: {pretrained_dir}")
    print(f"  Files on disk:")
    for f in sorted(pretrained_dir.glob("*.json")):
        if f.name == "metadata.json":
            continue
        print(f"    - {f.name}")
    print(f"  Metadata entries:")
    meta = json.loads((pretrained_dir / "metadata.json").read_text())
    for name, info in meta.items():
        expected_path = pretrained_dir / f"{name}.json"
        exists = expected_path.exists()
        print(f"    - {name} -> {expected_path.name} (exists={exists})")
    # Try to actually load the 65k tokenizer
    candidates = [n for n in available if "65k" in n.lower()]
    if candidates:
        name = candidates[0]
        print(f"  Attempting load_pretrained('{name}')...")
        try:
            tk = xorzen.load_pretrained(name)
            print(f"  Loaded: vocab_size={tk.get_vocab_size()}")
        except Exception as e:
            print(f"  REPRODUCED: Load failed — {type(e).__name__}: {e}")
    else:
        print("  REPRODUCED: No 65k tokenizer in registry")
except Exception as e:
    print(f"  REPRODUCED: {type(e).__name__} — {e}")
    traceback.print_exc()

# ============================================================
# Bug 3: Active-parameter percentage logger (~0.0%)
# ============================================================
print("\n[BUG 3] Active-parameter percentage logger reports ~0.0%")
try:
    import io
    import contextlib
    import xorzen
    # Capture the logger output during model init
    # The init logger uses logger.info("core", ...)
    # We'll capture stderr/stdout where the xorzen logger writes
    buf = io.StringIO()
    # Patch the logger to also write to our buffer
    from xorzen.utils.logger import get_logger
    xorzen_logger = get_logger()
    # Save original handlers
    import logging
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    # Find the underlying logger
    underlying = getattr(xorzen_logger, "logger", None) or xorzen_logger
    if hasattr(underlying, "addHandler"):
        underlying.addHandler(handler)
    # Now create a model and watch the log
    import torch.nn as nn
    model = xorzen.zero_1M(test_mode=True)
    log_output = buf.getvalue()
    print(f"  Captured init log:\n{log_output}")
    # Look for the percentage line
    pct_line = None
    for line in log_output.splitlines():
        if "experts active per token" in line.lower() or "active" in line.lower() and "%" in line:
            pct_line = line.strip()
            break
    if pct_line:
        print(f"  Init logger line: {pct_line}")
        # Extract percentage value
        import re
        m = re.search(r"~(\d+\.\d+)%", pct_line)
        if m:
            pct = float(m.group(1))
            print(f"  Active % reported by init logger: {pct}%")
            if pct < 1.0:
                print(f"  REPRODUCED: Init logger reports {pct}% (<1.0%, roughly 0.0%)")
            else:
                print(f"  Init logger reports {pct}% (NOT 0.0%)")
    else:
        print("  Could not find active-percentage line in init log")
    # Also compute what the buggy formula gives vs the correct one
    cfg = model.config
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buggy = cfg.top_k_experts * int(cfg.hidden_size * cfg.expert_hidden_multiplier)
    buggy_pct = 100.0 * buggy / max(1, trainable)
    correct = cfg.estimate_active_parameters()
    correct_pct = 100.0 * correct / max(1, trainable)
    print(f"  Trainable params: {trainable:,}")
    print(f"  Buggy formula: active_est={buggy:,}, pct={buggy_pct:.4f}%")
    print(f"  Correct (estimate_active_parameters): active_est={correct:,}, pct={correct_pct:.4f}%")
except Exception as e:
    print(f"  REPRODUCED: {type(e).__name__} — {e}")
    traceback.print_exc()

print("\n" + "=" * 72)
print("Bug reproduction complete.")
print("=" * 72)
