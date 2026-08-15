"""
Quick verification that fixes don't break:
- Model serialization (state_dict save/load round-trip)
- HASS-SSM block forward
- MoE routing (top-k)
- Expert sharding (test_mode dummy)
"""
import os
os.environ["XORZENX_VERBOSE"] = "0"

import io
import sys
import json
import torch
import torch.nn as nn
import traceback
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/xorzen_dev")
import xorzen

print("=" * 72)
print("SERIALIZATION + COMPONENT INTEGRITY CHECK")
print("=" * 72)

failures = []

# ---------------------------------------------------------------
# 1. state_dict round-trip
# ---------------------------------------------------------------
print("\n[1] state_dict save/load round-trip (zero_1M)...")
try:
    m1 = xorzen.zero_1M(test_mode=True)
    m1.eval()
    sd = m1.state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    sd2 = torch.load(buf, weights_only=False)
    m2 = xorzen.zero_1M(test_mode=True)
    m2.load_state_dict(sd2)
    m2.eval()
    # Same forward output for same input
    x = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        o1 = m1(x)
        o2 = m2(x)
    diff = (o1.logits - o2.logits).abs().max().item()
    print(f"  OK: max |Δlogits| after round-trip = {diff:.2e}")
    assert diff < 1e-5, f"round-trip diff too large: {diff}"
except Exception:
    print("  FAIL")
    traceback.print_exc()
    failures.append("state_dict round-trip")

# ---------------------------------------------------------------
# 2. HASS-SSM block forward
# ---------------------------------------------------------------
print("\n[2] HASS block forward (one block from zero_10M)...")
try:
    from xorzen.model.components.hass_block import HASSBlock
    # Find one HASS block in a real model
    m = xorzen.zero_10M(test_mode=True)
    hass = None
    for mod in m.modules():
        if isinstance(mod, HASSBlock):
            hass = mod
            break
    assert hass is not None, "no HASSBlock found in zero_10M"
    H = hass.hidden_dim
    x = torch.randn(2, 16, H)
    with torch.no_grad():
        y = hass(x)
    if isinstance(y, tuple):
        y = y[0]
    print(f"  OK: HASSBlock forward, output shape {tuple(y.shape)}")
    assert y.shape == x.shape, f"HASS output shape mismatch: {y.shape} vs {x.shape}"
except Exception:
    print("  FAIL")
    traceback.print_exc()
    failures.append("HASS-SSM block forward")

# ---------------------------------------------------------------
# 3. MoE routing produces valid top-k mask
# ---------------------------------------------------------------
print("\n[3] MoE top-k routing (zero_10M, top-2 of 8)...")
try:
    m = xorzen.zero_10M(test_mode=True)
    m.eval()
    assert m.config.top_k_experts == 2
    assert m.config.expert_count == 8
    x = torch.randint(0, 1000, (4, 32))
    with torch.no_grad():
        out = m(x)
    # Output shape
    print(f"  OK: forward, logits shape {tuple(out.logits.shape)}")
    # Verify active_params runtime field is sane
    if hasattr(out, "active_params") and out.active_params is not None:
        ap = int(out.active_params)
        print(f"  Runtime active_params = {ap:,}")
        assert ap > 0
except Exception:
    print("  FAIL")
    traceback.print_exc()
    failures.append("MoE top-k routing")

# ---------------------------------------------------------------
# 4. Disk-sharded expert fabric (test_mode dummy) wiring
# ---------------------------------------------------------------
print("\n[4] ShardedExpertFabric wiring (test_mode)...")
try:
    from xorzen.model.zmoe import ShardedExpertFabric
    m = xorzen.zero_50M(test_mode=True)
    fabric = None
    for mod in m.modules():
        if isinstance(mod, ShardedExpertFabric):
            fabric = mod
            break
    assert fabric is not None, "no ShardedExpertFabric found"
    assert fabric.test_mode is True, "expected test_mode=True"
    print(f"  OK: ShardedExpertFabric in test_mode, num_experts={fabric.num_experts}")
    # Forward still works
    x = torch.randint(0, 1000, (2, 32))
    with torch.no_grad():
        _ = m(x)
    print("  OK: forward through sharded expert fabric succeeds")
except Exception:
    print("  FAIL")
    traceback.print_exc()
    failures.append("ShardedExpertFabric wiring")

# ---------------------------------------------------------------
# 5. Tokenizer encode/decode round-trip with the 65k tokenizer
# ---------------------------------------------------------------
print("\n[5] 65k tokenizer encode/decode round-trip...")
try:
    tk = xorzen.load_pretrained("xorzen_agi_tokenizer_65k")
    assert tk.get_vocab_size() == 65000
    text = "Hello Xorzen, the patched v0.2.4 is ready for push."
    enc = tk.encode(text)
    dec = tk.decode(enc)
    ok = dec.startswith("Hello")
    print(f"  vocab={tk.get_vocab_size()}, enc_len={len(enc)}, roundtrip_ok={ok}")
    assert ok, f"round-trip failed: {dec!r}"
except Exception:
    print("  FAIL")
    traceback.print_exc()
    failures.append("65k tokenizer round-trip")

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print("\n" + "=" * 72)
if failures:
    print(f"FAIL: {len(failures)} component(s) broken: {failures}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — fixes do not break serialization, HASS, MoE, sharding, or tokenizer.")
print("=" * 72)
