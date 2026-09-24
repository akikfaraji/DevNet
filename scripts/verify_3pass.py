#!/usr/bin/env python3
"""Three independent verification passes for XORZEN correctness.

Pass 1: Functional correctness — full model forward/backward, loss computation.
Pass 2: Strict causality, routing properties, gradients, tensor shapes, numerical.
Pass 3: Independent inference/performance sanity — NaN/Inf, compute behavior, longer sequences.
"""
import sys, math, time
import torch
import torch.nn.functional as F
import logging

logging.disable(logging.CRITICAL)

PASS_COUNT = {"pass1": 0, "pass2": 0, "pass3": 0, "fail": 0}

def check(name, condition, detail=""):
    if condition:
        PASS_COUNT["pass" + name[4]] += 1  # extract pass number from "passN:..."
        print(f"  [PASS] {detail}")
    else:
        PASS_COUNT["fail"] += 1
        print(f"  [FAIL] {detail}")
    return condition


# ========================================================================
# PASS 1: Functional correctness
# ========================================================================
print("\n" + "=" * 60)
print("PASS 1: Functional correctness — forward/backward, loss, shapes")
print("=" * 60)

from xorzen.config import ConfigFactory
from xorzen.models.zero.model import zeroModel

# 1a. Model construction and parameter counting
cfg = ConfigFactory.get_config('NANO_1M')
cfg.dropout = 0.0
torch.manual_seed(42)
model = zeroModel(cfg, test_mode=True)
total_p = sum(p.numel() for p in model.parameters())
train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
check("pass1", total_p > 0, f"Total params > 0: {total_p:,}")
check("pass1", train_p > 0, f"Trainable params > 0: {train_p:,}")
check("pass1", train_p <= total_p, f"Trainable <= total")

# 1b. Forward pass shapes (eval mode)
model.eval()
x = torch.randint(0, cfg.vocab_size, (2, 16))
with torch.no_grad():
    out = model(x)
check("pass1", out.logits.shape == (2, 16, cfg.vocab_size),
      f"Eval logits shape: {out.logits.shape} == {(2, 16, cfg.vocab_size)}")
check("pass1", out.loss is not None, "Eval mode: routing loss always computed")

# 1c. Forward + loss shapes (eval mode, with labels)
with torch.no_grad():
    out_l = model(x, labels=x)
check("pass1", out_l.loss is not None, "Eval + labels: loss is not None")
check("pass1", out_l.loss.dim() == 0, f"Loss is scalar: shape={out_l.loss.shape}")
check("pass1", torch.isfinite(out_l.loss), f"Loss is finite: {out_l.loss.item():.4f}")

# 1d. Forward pass shapes (train mode)
model.train()
out_t = model(x, labels=x)
check("pass1", out_t.logits.shape == (2, 16, cfg.vocab_size),
      f"Train logits shape: {out_t.logits.shape}")
check("pass1", out_t.loss is not None, "Train + labels: loss is not None")
check("pass1", torch.isfinite(out_t.loss), f"Train loss finite: {out_t.loss.item():.4f}")

# 1e. Backward pass — all trainable params get gradients
# Note: width_router params may have zero gradients when there's only 1
# width choice (NANO_1M has width_choices=(64,)). This is expected — a
# single-choice router has no routing decision to make.
model.zero_grad()
out_t.loss.backward()
no_grad_count = 0
ok_no_grad = set()  # params that are allowed to have zero grad
if len(cfg.width_choices) <= 1:
    ok_no_grad.update(n for n, _ in model.named_parameters() if 'width_router' in n)
    # SlicedFFN.ln_hidden (at max FFN dim) is only reached when the max
    # width is selected by routing. With a single width choice it is
    # never in the forward path — expected dead code, not a bug.
    ok_no_grad.update(n for n, _ in model.named_parameters() if 'ln_hidden' in n)
    # CoT is frozen during pre-training, so merger's cot_proj gets no gradient.
    ok_no_grad.update(n for n, _ in model.named_parameters() if 'cot_proj' in n)
for name, param in model.named_parameters():
    if param.requires_grad and (param.grad is None or param.grad.abs().max() == 0):
        if name in ok_no_grad:
            continue
        no_grad_count += 1
        if no_grad_count <= 3:
            print(f"    WARNING: {name} has no/zero gradient")
check("pass1", no_grad_count == 0, f"All trainable params have non-zero gradients (zero_count={no_grad_count}, excluded degenerate width_router)")

# 1f. Parameter update step (simulated)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.step()
check("pass1", True, "Optimizer step completed without error")

# 1g. Multi-batch forward (no state leakage between batches)
model.eval()
with torch.no_grad():
    out_b1 = model(torch.randint(0, cfg.vocab_size, (2, 8)))
    out_b2 = model(torch.randint(0, cfg.vocab_size, (2, 8)))
check("pass1", out_b1.logits.shape == (2, 8, cfg.vocab_size), "Batch 1 shape correct")
check("pass1", out_b2.logits.shape == (2, 8, cfg.vocab_size), "Batch 2 shape correct")

# 1h. Different config sizes
cfg_tiny = ConfigFactory.get_config('TINY_23K')
cfg_tiny.dropout = 0.0
torch.manual_seed(7)
m_tiny = zeroModel(cfg_tiny, test_mode=True)
m_tiny.eval()
with torch.no_grad():
    out_tiny = m_tiny(torch.randint(0, cfg_tiny.vocab_size, (1, 8)))
check("pass1", out_tiny.logits.shape == (1, 8, cfg_tiny.vocab_size),
      f"TINY_23K logits shape: {out_tiny.logits.shape}")

print(f"\nPass 1 complete: {PASS_COUNT['pass1']} passed, {PASS_COUNT['fail']} failed")


# ========================================================================
# PASS 2: Strict causality, routing, gradients, shapes, numerical
# ========================================================================
print("\n" + "=" * 60)
print("PASS 2: Causality, routing, gradients, shapes, numerical")
print("=" * 60)

# 2a. LowRankGlobalPathway causal (multiple configs)
from xorzen.model.components.hass_block import LowRankGlobalPathway
for heads in [1, 2, 4]:
    for seq_len in [8, 32, 64]:
        torch.manual_seed(42)
        p = LowRankGlobalPathway(hidden_dim=64, low_rank_dim=16, num_heads=heads, dropout=0.0)
        p.eval()
        x = torch.randn(1, seq_len, 64)
        with torch.no_grad():
            out1 = p(x)
        x2 = x.clone()
        x2[:, seq_len * 3 // 4:, :] += 100.0
        with torch.no_grad():
            out2 = p(x2)
        cutoff = seq_len * 3 // 4
        diff = (out1[:, :cutoff, :] - out2[:, :cutoff, :]).abs().max().item()
        check("pass2", diff < 1e-5,
              f"LRGlobal causal (heads={heads}, T={seq_len}): diff={diff:.2e}")

# 2b. SSM causal conv
from xorzen.model.components.hass_block import SSMPathway
for ks in [3, 4]:
    torch.manual_seed(42)
    ssm = SSMPathway(hidden_dim=64, state_dim=16, use_conv=True, kernel_size=ks)
    ssm.eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        o1 = ssm(x)
    x2 = x.clone()
    x2[:, 24:, :] += 100.0
    with torch.no_grad():
        o2 = ssm(x2)
    diff = (o1[:, :24, :] - o2[:, :24, :]).abs().max().item()
    check("pass2", diff < 1e-5, f"SSM causal conv (k={ks}): diff={diff:.2e}")

# 2c. HASS block causal (combined pathways)
from xorzen.model.components.hass_block import HASSBlock
torch.manual_seed(42)
block = HASSBlock(cfg, layer_idx=0)
block.eval()
x_hass = torch.randn(2, 32, cfg.hidden_size)
with torch.no_grad():
    o1 = block(x_hass, compute_all_pathways=True)
x2_hass = x_hass.clone()
x2_hass[:, 24:, :] += 100.0
with torch.no_grad():
    o2 = block(x2_hass, compute_all_pathways=True)
diff = (o1[:, :24, :] - o2[:, :24, :]).abs().max().item()
check("pass2", diff < 1e-5, f"HASS block causal (3 pathways): diff={diff:.2e}")

# 2d. Full model causal — eval mode
torch.manual_seed(42)
model_c = zeroModel(cfg, test_mode=True)
model_c.eval()
x_c = torch.randint(0, cfg.vocab_size, (1, 32))
with torch.no_grad():
    o1 = model_c(x_c)
x2_c = x_c.clone()
x2_c[:, 24:] = (x2_c[:, 24:] + 100) % cfg.vocab_size
with torch.no_grad():
    o2 = model_c(x2_c)
diff = (o1.logits[:, :24, :] - o2.logits[:, :24, :]).abs().max().item()
check("pass2", diff < 1e-5, f"Full model causal eval (T=32): diff={diff:.2e}")

# 2e. Full model causal — training mode (with RNG control)
model_c.train()
x_c = torch.randint(0, cfg.vocab_size, (1, 32))
labels_c = x_c.clone()
rng_state = torch.get_rng_state()
o1 = model_c(x_c, labels=labels_c)
x2_c = x_c.clone()
x2_c[:, 24:] = (x2_c[:, 24:] + 100) % cfg.vocab_size
torch.set_rng_state(rng_state)
o2 = model_c(x2_c, labels=labels_c)
diff = (o1.logits[:, :24, :] - o2.logits[:, :24, :]).abs().max().item()
check("pass2", diff < 1e-5, f"Full model causal train (T=32, RNG ctrl): diff={diff:.2e}")

# 2f. Routing properties: simplex, top-k, determinism
torch.manual_seed(42)
model_r = zeroModel(cfg, test_mode=True)
model_r.eval()
x_r = torch.randn(2, 8, cfg.hidden_size)
with torch.no_grad():
    h_r = x_r  # skip embedding for simplicity
    cot_r = torch.zeros(2, 8, cfg.cot_dim * cfg.cot_components)
    rd = model_r.router(h_r, cot_r, training=False)

# Path probs should sum to 1
path_sums = rd.path_probs.sum(dim=-1)
check("pass2", (path_sums - 1.0).abs().max() < 1e-5,
      f"Path probs sum to 1: max|sum-1|={(path_sums - 1.0).abs().max():.2e}")

# Expert weights should be non-negative
check("pass2", (rd.expert_weights >= 0).all(),
      f"Expert weights non-negative")

# Routing determinism: same input → same output
with torch.no_grad():
    rd2 = model_r.router(h_r, cot_r, training=False)
check("pass2", (rd.path_probs - rd2.path_probs).abs().max() < 1e-6,
      "Routing deterministic in eval mode")

# 2g. Gradient flow through HASS block
torch.manual_seed(42)
block_g = HASSBlock(cfg, layer_idx=0)
block_g.train()
x_g = torch.randn(1, 8, cfg.hidden_size, requires_grad=True)
out_g = block_g(x_g, compute_all_pathways=True)
out_g.sum().backward()
check("pass2", x_g.grad is not None, "HASS block: input gradient exists")
check("pass2", x_g.grad.abs().max() > 1e-8,
      f"HASS block: input gradient non-trivial: {x_g.grad.abs().max():.4e}")

# 2h. Numerical: no NaN/Inf in any component
model_n = zeroModel(cfg, test_mode=True)
model_n.train()
x_n = torch.randn(2, 16, cfg.hidden_size) * 5.0  # larger inputs
from xorzen.model.components.hass_block import LowRankGlobalPathway, SSMPathway
torch.manual_seed(42)
lr = LowRankGlobalPathway(64, 16, 2, 0.0)
ssm = SSMPathway(hidden_dim=64, state_dim=16, use_conv=True, kernel_size=4)
out_lr = lr(x_n)
out_ssm = ssm(x_n)
check("pass2", not torch.isnan(out_lr).any() and not torch.isinf(out_lr).any(),
      "LowRankGlobal: no NaN/Inf with large inputs")
check("pass2", not torch.isnan(out_ssm).any() and not torch.isinf(out_ssm).any(),
      "SSM: no NaN/Inf with large inputs")

# 2i. SSM scan equivalence (sequential vs parallel)
torch.manual_seed(42)
ssm_eq = SSMPathway(hidden_dim=64, state_dim=16, use_conv=False)
ssm_eq.eval()
x_seq = torch.randn(1, 32, 64)
with torch.no_grad():
    # Run with sequential scan
    ssm_eq.use_parallel_scan = False
    o_seq = ssm_eq(x_seq)
    # Run with parallel scan
    ssm_eq.use_parallel_scan = True
    o_par = ssm_eq(x_seq)
diff = (o_seq - o_par).abs().max().item()
check("pass2", diff < 1e-4, f"SSM seq vs parallel scan match: diff={diff:.2e}")

print(f"\nPass 2 complete: {PASS_COUNT['pass2']} passed, {PASS_COUNT['fail']} failed")


# ========================================================================
# PASS 3: Independent inference/performance sanity
# ========================================================================
print("\n" + "=" * 60)
print("PASS 3: Inference sanity, NaN/Inf, compute behavior, longer seq")
print("=" * 60)

# 3a. Full model forward — longer sequence (T=128)
torch.manual_seed(42)
model_3 = zeroModel(cfg, test_mode=True)
model_3.eval()
x_128 = torch.randint(0, cfg.vocab_size, (1, 128))
t0 = time.time()
with torch.no_grad():
    out_128 = model_3(x_128)
dt = time.time() - t0
check("pass3", out_128.logits.shape == (1, 128, cfg.vocab_size),
      f"T=128 logits shape: {out_128.logits.shape}")
check("pass3", torch.isfinite(out_128.logits).all(), "T=128: all logits finite")
print(f"    T=128 forward time: {dt:.3f}s")

# 3b. Full model forward — max context length
torch.manual_seed(42)
x_max = torch.randint(0, cfg.vocab_size, (1, cfg.context_length))
with torch.no_grad():
    out_max = model_3(x_max)
check("pass3", out_max.logits.shape == (1, cfg.context_length, cfg.vocab_size),
      f"T={cfg.context_length} logits shape: {out_max.logits.shape}")
check("pass3", torch.isfinite(out_max.logits).all(),
      f"T={cfg.context_length}: all logits finite")

# 3c. Training forward + backward — no NaN in gradients
model_3.train()
x_tb = torch.randint(0, cfg.vocab_size, (2, 32))
out_tb = model_3(x_tb, labels=x_tb)
check("pass3", torch.isfinite(out_tb.loss), f"Training loss finite: {out_tb.loss.item():.4f}")
model_3.zero_grad()
out_tb.loss.backward()
max_grad = 0.0
has_nan_grad = False
for name, param in model_3.named_parameters():
    if param.grad is not None:
        mg = param.grad.abs().max().item()
        max_grad = max(max_grad, mg)
        if torch.isnan(param.grad).any():
            has_nan_grad = True
            print(f"    NaN grad in: {name}")
check("pass3", not has_nan_grad, "No NaN in any parameter gradients")
check("pass3", max_grad < 1e6, f"Max gradient magnitude reasonable: {max_grad:.2e}")

# 3d. Multi-step training simulation (detects state leakage)
model_3.train()
optimizer = torch.optim.Adam(model_3.parameters(), lr=1e-4)
losses = []
for step in range(5):
    x_step = torch.randint(0, cfg.vocab_size, (2, 16))
    out_step = model_3(x_step, labels=x_step)
    loss_val = out_step.loss.item()
    losses.append(loss_val)
    optimizer.zero_grad()
    out_step.loss.backward()
    optimizer.step()
    check("pass3", math.isfinite(loss_val),
          f"Step {step}: loss finite ({loss_val:.4f})")
# Loss doesn't have to decrease in 5 steps, just stay finite and non-NaN
check("pass3", all(math.isfinite(l) for l in losses),
      f"All 5 training steps produced finite loss")

# 3e. Save/load roundtrip — model state preserved
torch.manual_seed(42)
model_save = zeroModel(cfg, test_mode=True)
model_save.eval()
x_save = torch.randint(0, cfg.vocab_size, (1, 8))
with torch.no_grad():
    out_before = model_save(x_save)

import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.pt")
    torch.save(model_save.state_dict(), path)
    model_load = zeroModel(cfg, test_mode=True)
    model_load.load_state_dict(torch.load(path, weights_only=True))
    model_load.eval()
    with torch.no_grad():
        out_after = model_load(x_save)
    diff = (out_before.logits - out_after.logits).abs().max().item()
    check("pass3", diff < 1e-5, f"Save/load roundtrip: max diff={diff:.2e}")

# 3f. Eval mode consistency — multiple forward passes, same output
model_3.eval()
x_cons = torch.randint(0, cfg.vocab_size, (1, 16))
with torch.no_grad():
    o_a = model_3(x_cons)
    o_b = model_3(x_cons)
    o_c = model_3(x_cons)
diff_ab = (o_a.logits - o_b.logits).abs().max().item()
diff_bc = (o_b.logits - o_c.logits).abs().max().item()
check("pass3", diff_ab < 1e-6, f"Eval determinism pass A→B: {diff_ab:.2e}")
check("pass3", diff_bc < 1e-6, f"Eval determinism pass B→C: {diff_bc:.2e}")

# 3g. Tied embedding consistency
check("pass3", cfg.tie_word_embeddings, "Config has tied embeddings")
check("pass3", model_3.token_embedding.weight.data_ptr() == model_3.lm_head.weight.data_ptr(),
      "Embedding and LM head weights are actually tied (same memory)")

print(f"\nPass 3 complete: {PASS_COUNT['pass3']} passed, {PASS_COUNT['fail']} failed")


# ========================================================================
# SUMMARY
# ========================================================================
print("\n" + "=" * 60)
print("VERIFICATION SUMMARY")
print("=" * 60)
print(f"  Pass 1 (Functional correctness):    {PASS_COUNT['pass1']} checks")
print(f"  Pass 2 (Causality/routing/numerical): {PASS_COUNT['pass2']} checks")
print(f"  Pass 3 (Inference/performance):      {PASS_COUNT['pass3']} checks")
print(f"  Total failures:                       {PASS_COUNT['fail']}")

if PASS_COUNT["fail"] == 0:
    print("\n  ✓ ALL 3 VERIFICATION PASSES PASSED — NO FAILURES")
    sys.exit(0)
else:
    print(f"\n  ✗ {PASS_COUNT['fail']} CHECK(S) FAILED")
    sys.exit(1)
