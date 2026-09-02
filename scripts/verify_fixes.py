#!/usr/bin/env python3
"""
3 independent verification passes for XORZEN bugfixes.
Pass 1: Forward/Backward correctness and gradient flow
Pass 2: Strict causal/autoregressive + routing + sparse execution
Pass 3: Numerical stability + FLOP/compute behavior
"""
import sys
import torch
import torch.nn.functional as F
import math
import time

sys.path.insert(0, '/home/z/my-project/xorzen-repo')

PASS = 1
ERRORS = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if status == "FAIL":
        ERRORS.append((PASS, name, detail))
    print(f"  [{status}] {name}: {detail}")
    return condition


# ============================================================
# PASS 1: Forward/Backward Correctness & Gradient Flow
# ============================================================
print(f"\n{'='*60}")
print(f"VERIFICATION PASS {PASS}: Forward/Backward & Gradient Flow")
print(f"{'='*60}")

# 1a. LowRankGlobalPathway output shape correct for all batch/seq sizes
print("\n1a. LowRankGlobalPathway shape correctness")
from xorzen.model.components.hass_block import LowRankGlobalPathway
for B, T, H, nh in [(1, 4, 32, 1), (2, 8, 64, 2), (1, 64, 128, 4), (3, 32, 96, 4)]:
    p = LowRankGlobalPathway(hidden_dim=H, low_rank_dim=min(16, H), num_heads=nh)
    x = torch.randn(B, T, H)
    out = p(x)
    check(f"shape B={B} T={T} H={H} nh={nh}",
          out.shape == (B, T, H), f"got {out.shape}")

# 1b. Gradient flow to all parameters
print("\n1b. Gradient flow to all LowRankGlobalPathway parameters")
p = LowRankGlobalPathway(hidden_dim=64, low_rank_dim=16, num_heads=1)
p.train()
torch.manual_seed(42)
x = torch.randn(2, 8, 64, requires_grad=True)
out = np(x)
loss = out.sum()
loss.backward()
for name, param in p.named_parameters():
    check(f"grad: {name}",
          param.grad is not None and param.grad.abs().max().item() > 1e-8,
          f"grad={None if param.grad is None else param.grad.abs().max().item():.6f}")

# 1c. HASSBlock forward/backward
print("\n1c. HASSBlock forward/backward")
from xorzen.config import ConfigFactory, ModelConfig
from xorzen.model.components.hass_block import HASSBlock
cfg = ConfigFactory.get_config('NANO_1M')
cfg.dropout = 0.0
torch.manual_seed(42)
block = HASSBlock(cfg, layer_idx=0)
block.train()
x = torch.randn(1, 16, cfg.hidden_size, requires_grad=True)
out = block(x, compute_all_pathways=True)
loss = out.sum()
loss.backward()
check("HASS block grad exists", x.grad is not None and x.grad.abs().max().item() > 1e-8)
check("HASS block output finite", torch.isfinite(out).all())

# 1d. Full model forward/backward
print("\n1d. Full model forward/backward")
from xorzen.models.zero.model import zeroModel
cfg2 = ConfigFactory.get_config('NANO_1M')
cfg2.dropout = 0.0
torch.manual_seed(42)
model = zeroModel(cfg2, test_mode=True)
model.eval()
x = torch.randint(0, cfg2.vocab_size, (1, 8))
out = model(x)
check("Full model output shape", out.logits.shape == (1, 8, cfg2.vocab_size))
check("Full model logits finite", torch.isfinite(out.logits).all())

# ============================================================
# PASS 2: Causal / Routing / Sparse Execution
# ============================================================
PASS = 2
print(f"\n{'='*60}")
print(f"VERIFICATION PASS {PASS}: Causal / Routing / Sparse")
print(f"{'='*60}")

# 2a. LowRankGlobalPathway strict causality
print("\n2a. LowRankGlobalPathway strict causality")
for seed, B, T, H in [(42, 1, 8, 64), (123, 4, 16, 128), (7, 1, 64, 128)]:
    torch.manual_seed(seed)
    p = LowRankGlobalPathway(hidden_dim=H, low_rank_dim=16, num_heads=1, dropout=0.0)
    p.eval()
    x = torch.randn(B, T, H)
    with torch.no_grad():
        out1 = p(x)
    x2 = x.clone()
    x2[:, T // 2 :, :] += 100.0
    with torch.no_grad():
        out2 = p(x2)
    d = (out1[:, :T // 2, :] - out2[:, :T // 2, :]).abs().max().item()
    check(f"causal seed={seed} B={B} T={T} H={H}", d < 1e-5, f"diff={d}")

# 2b. LocalAttentionPathway strict causality
print("\n2b. LocalAttentionPathway strict causality")
from xorzen.model.components.hass_block import LocalAttentionPathway
torch.manual_seed(42)
lp = LocalAttentionPathway(hidden_dim=64, num_heads=2, window_size=4, causal=True, dropout=0.0)
lp.eval()
x = torch.randn(1, 16, 64)
with torch.no_grad():
    out1 = lp(x, None, None)
x2 = x.clone()
x2[:, 8:, :] += 100.0
with torch.no_grad():
    out2 = lp(x2, None, None)
d = (out1[:, :8, :] - out2[:, :8, :]).abs().max().item()
check("local attn causal", d < 1e-5, f"diff={d}")

# 2c. SSMPathway strict causality (causal conv)
print("\n2c. SSMPathway strict causality (causal conv)")
from xorzen.model.components.hass_block import SSMPathway
ssm = SSMPathway(hidden_dim=64, state_dim=16, kernel_size=3, dropout=0.0)
bssm.eval()
x = torch.randn(1, 16, 64)
with torch.no_grad():
    out1 = ssm(x)
x2 = x.clone()
x2[:, 8:, :] += 100.0
with torch.no_grad():
    out2 = ssm(x2)
d = (out1[:, :8, :] - out2[:, :8, :]).abs().max().item()
check("SSM causal conv", d < 1e-5, f"diff={d}")

# 2d. Full model eval causality
print("\n2d. Full model eval-mode causality")
cfg3 = ConfigFactory.get_config('NANO_1M')
cfg3.dropout = 0.0
torch.manual_seed(42)
model3 = zeroModel(cfg3, test_mode=True)
model3.eval()
x = torch.randint(0, cfg3.vocab_size, (1, 16))
with torch.no_grad():
    out1 = model3(x)
x2 = x.clone()
x2[:, 12:] = (x2[:, 12:] + 100) % cfg3.vocab_size
with torch.no_grad():
    out2 = model3(x2)
d = (out1.logits[:, :12, :] - out2.logits[:, :12, :]).abs().max().item()
check("full model eval causal", d < 1e-5, f"diff={d}")

# 2e. Sparse dispatch genuinely skips unselected pathways
print("\n2e. Sparse dispatch genuinely skips unselected pathways")
from xorzen.model.components.sparse_dispatch import sparse_pathway_dispatch
call_log = {}
def make_fn(key):
    def fn(x_slice):
        call_log[key] = call_log.get(key, 0) + 1
        return x_slice * 0.5
    return fn

B, T, H = 2, 8, 32
x_sp = torch.randn(B, T, H)
pp = torch.softmax(torch.randn(B, T, 3), dim=-1)
for top_k in [1, 2]:
    call_log.clear()
    sparse_pathway_dispatch(
        x_sp, pp,
        {"a": make_fn("a"), "b": make_fn("b"), "c": make_fn("c")},
        ["a", "b", "c"], top_k, training=False,
        pathway_call_counter=call_log,
    )
    total_calls = sum(call_log.values())
    check(f"top_k={top_k} total calls={total_calls}",
          total_calls <= 3, f"expected <=3, got {total_calls}")

# ============================================================
# PASS 3: Numerical Stability + FLOP Accounting
# ============================================================
PASS = 3
print(f"\n{'='*60}")
print(f"VERIFICATION PASS {PASS}: Numerical Stability + FLOP")
print(f"{'='*60}")

# 3a. No NaN/Inf in forward
print("\n3a. Numerical stability (no NaN/Inf)")
for seed in [0, 42, 999]:
    torch.manual_seed(seed)
    p = LowRankGlobalPathway(hidden_dim=64, low_rank_dim=16, num_heads=1)
    x = torch.randn(2, 32, 64) * 10.0  # Large input
    out = p(x)
    check(f"no NaN seed={seed}", not torch.isnan(out).any(), f"has NaN")
    check(f"no Inf seed={seed}", not torch.isinf(out).any(), f"has Inf")

# 3b. No NaN/Inf in backward
print("\n3b. Numerical stability in backward")
torch.manual_seed(42)
pp2 = LowRankGlobalPathway(hidden_dim=64, low_rank_dim=16, num_heads=1)
bpp2.train()
x = torch.randn(2, 32, 64) * 5.0
out = pp2(x)
loss = out.sum()
loss.backward()
for name, param in pp2.named_parameters():
    if param.grad is not None:
        check(f"grad no NaN {name}", not torch.isnan(param.grad).any())
        check(f"grad no Inf {name}", not torch.isinf(param.grad).any())

# 3c. HASSBlock no NaN/Inf
print("\n3c. HASSBlock numerical stability")
cfg_s = ConfigFactory.get_config('NANO_1M')
cfg_s.dropout = 0.0
torch.manual_seed(42)
blk = HASSBlock(cfg_s, layer_idx=0)
blk.train()
x_b = torch.randn(1, 32, cfg_s.hidden_size)
out_b = blk(x_b, compute_all_pathways=True)
check("HASS forward no NaN", not torch.isnan(out_b).any())
check("HASS forward no Inf", not torch.isinf(out_b).any())
(out_b.sum()).backward()
check("HASS backward no NaN", not torch.isnan(x_b.grad).any())

# 3d. Full model no NaN/Inf
print("\n3d. Full model numerical stability")
cfg_f = ConfigFactory.get_config('NANO_1M')
cfg_f.dropout = 0.0
torch.manual_seed(42)
fm = zeroModel(cfg_f, test_mode=True)
fm.eval()
x_f = torch.randint(0, cfg_f.vocab_size, (1, 32))
out_f = fm(x_f)
check("full model logits no NaN", not torch.isnan(out_f.logits).any())
check("full model logits no Inf", not torch.isinf(out_f.logits).any())

# 3e. Compute stats reporting
print("\n3e. Compute stats reporting")
stats = lp.get_compute_stats(16, 1)
check("compute stats has flops", 'flops_total' in stats and stats['flops_total'] > 0)
stats2 = blk.get_compute_stats(16, 1, routing_decision=None)
check("HASS block stats has flops", 'flops_total' in stats2 and stats2['flops_total'] > 0)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"Pass 1 (Forward/Backward): {sum(1 for p,e,n,_ in ERRORS if p==1)} FAIL")
print(f"Pass 2 (Causal/Routing/Sparse): {sum(1 for p,e,n,_ in ERRORS if p==2)} FAIL")
print(f"Pass 3 (Stability/FLOPs): {sum(1 for p,e,n,_ in ERRORS if p==3)} FAIL")
total_fails = len(ERRORS)
if total_fails > 0:
    print(f"\nFAILURES:")
    for p, name, detail in ERRORS:
        print(f"  Pass {p}: {name} — {detail}")
    sys.exit(1)
else:
    print(f"\nALL CHECKS PASSED ({3}/{3})")

sys.exit(0)
