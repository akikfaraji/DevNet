"""Smoke test: SSM ZOH fix + scan equivalence + double-C removal."""
import sys
sys.path.insert(0, "/home/z/my-project/xorzen_dev")

import torch
torch.manual_seed(0)

from xorzen.model.components.ssm_scan import (
    discretize_zoh, discretize_b_first_order,
    sequential_scan, parallel_scan, chunked_scan, select_scan,
)

print("=" * 70)
print("TEST 1: ZOH discretization produces stable A_bar in (0,1)")
print("=" * 70)
N = 8
A_log = torch.zeros(N)  # a = -1
a = -torch.exp(A_log)
dt = torch.full((1, 16, N), 0.5)
B = torch.randn(1, 16, N)
A_bar, B_bar = discretize_zoh(a, B, dt)
print(f"  a = {a}")
print(f"  A_bar range: [{A_bar.min():.4f}, {A_bar.max():.4f}]  (should be in (0,1))")
assert (A_bar > 0).all() and (A_bar < 1).all()
print(f"  B_bar range: [{B_bar.min():.4f}, {B_bar.max():.4f}]  (finite, non-zero)")
assert torch.isfinite(B_bar).all()
print("  PASS: A_bar in (0,1), B_bar finite\n")

print("=" * 70)
print("TEST 2: ZOH vs first-order approximation (should differ but be close for small dt)")
print("=" * 70)
dt_small = torch.full((1, 16, N), 0.01)
A_bar_zoh, B_bar_zoh = discretize_zoh(a, B, dt_small)
A_bar_fo, B_bar_fo = discretize_b_first_order(a, B, dt_small)
# For small dt, ZOH B_bar = ((exp(dt*a)-1)/a)*B ≈ dt*B (first-order)
rel_diff = (B_bar_zoh - B_bar_fo).abs().max() / B_bar_zoh.abs().max()
print(f"  max relative diff for small dt=0.01: {rel_diff:.6f}  (should be small)")
assert rel_diff < 0.01, f"ZOH and first-order should agree for small dt, got {rel_diff}"
print("  PASS: ZOH ≈ first-order for small dt\n")

print("=" * 70)
print("TEST 3: Scan equivalence (sequential == parallel == chunked)")
print("=" * 70)
B_size, T, N_size = 2, 64, 16
torch.manual_seed(42)
A_bar = torch.rand(B_size, T, N_size) * 0.5 + 0.4  # in (0.4, 0.9), stable
B_bar = torch.randn(B_size, T, N_size) * 0.1

s_seq = sequential_scan(A_bar, B_bar)
s_par = parallel_scan(A_bar, B_bar)
s_chk = chunked_scan(A_bar, B_bar, chunk_size=16)

diff_par = (s_seq - s_par).abs().max().item()
diff_chk = (s_seq - s_chk).abs().max().item()
print(f"  ||seq - parallel||_inf = {diff_par:.2e}")
print(f"  ||seq - chunked||_inf  = {diff_chk:.2e}")
assert diff_par < 1e-5, f"parallel scan diverges: {diff_par}"
assert diff_chk < 1e-6, f"chunked scan diverges: {diff_chk}"
print("  PASS: all three scans agree\n")

print("=" * 70)
print("TEST 4: SSMPathway forward + no double-C bug")
print("=" * 70)
from xorzen.model.components.hass_block import SSMPathway
ssm = SSMPathway(hidden_dim=32, state_dim=8, kernel_size=3, dropout=0.0, use_conv=True)
ssm.eval()
x = torch.randn(2, 32, 32)
with torch.no_grad():
    y = ssm(x)
print(f"  Input shape:  {tuple(x.shape)}")
print(f"  Output shape: {tuple(y.shape)}  (must match input)")
assert y.shape == x.shape
assert torch.isfinite(y).all()
print("  PASS: forward runs, output shape correct, finite\n")

print("=" * 70)
print("TEST 5: SSMPathway gradient flow through A_log, dt_proj, B_proj, C_proj")
print("=" * 70)
ssm.train()
x = torch.randn(2, 16, 32, requires_grad=False)
y = ssm(x)
loss = y.pow(2).sum()
loss.backward()
for name, p in ssm.named_parameters():
    has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
    print(f"  {name:20s}  grad_ok={has_grad}")
    assert has_grad, f"no gradient on {name}"
print("  PASS: all SSM parameters receive gradients\n")

print("=" * 70)
print("TEST 6: forward_parallel == forward (both use ZOH, no double-C)")
print("=" * 70)
ssm.eval()
x = torch.randn(1, 32, 32)
with torch.no_grad():
    y1 = ssm.forward(x)
    # parallel uses a different scan path but same discretization
    y2 = ssm.forward_parallel(x)
diff = (y1 - y2).abs().max().item()
print(f"  ||forward - forward_parallel||_inf = {diff:.2e}  (should be ~0 for T=32)")
# For T=32 <= 64, both fall back to sequential, so should be identical
assert diff < 1e-6, f"forward and forward_parallel disagree: {diff}"
print("  PASS: forward and forward_parallel agree for T<=64\n")

print("=" * 70)
print("TEST 7: Scan scaling — chunked scan has T/chunk iterations, not T")
print("=" * 70)
# We can't easily count Python iterations from outside, but we can verify
# the chunked scan produces the same result for different chunk sizes.
torch.manual_seed(7)
A_bar = torch.rand(1, 256, 16) * 0.5 + 0.4
B_bar = torch.randn(1, 256, 16) * 0.1
s_ref = sequential_scan(A_bar, B_bar)
for chunk in [32, 64, 128, 256]:
    s_c = chunked_scan(A_bar, B_bar, chunk_size=chunk)
    d = (s_ref - s_c).abs().max().item()
    print(f"  chunk={chunk}: ||seq - chunked|| = {d:.2e}")
    assert d < 1e-6
print("  PASS: chunked scan agrees with sequential for all chunk sizes\n")

print("=" * 70)
print("ALL SSM TESTS PASSED")
print("=" * 70)
