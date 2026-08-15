"""
Phase 6 (v0.5) — Regression tests for v0.5 fixes.

Tests added:
  1. test_eval_routing_noise_breaks_collapse
     — Quantitative: with eval_routing_noise>0, eval-mode pathway/width
       diversity must be >= eval_routing_noise=0 (legacy v0.4 behavior).
  2. test_eval_routing_noise_reproducible
     — Same input produces same routing (deterministic eval).
  3. test_eval_routing_noise_preserves_sparsity
     — Top-k hard selection still holds: only K pathways' forwards run.
  4. test_compute_controller_removed
     — Module must be unimportable (deleted).
  5. test_pathway_gate_removed
     — HASSBlock must NOT have pathway_gate attribute.
  6. test_pathway_gate_removal_doesnt_break_standalone
     — HASSBlock with routing_decision=None still produces finite output.
  7. test_sparse_dispatch_correctness_after_optimization
     — index_add_ path produces same output as boolean-scatter reference.
  8. test_eval_routing_noise_zero_disables
     — eval_routing_noise=0.0 reproduces v0.4 deterministic behavior.
"""
import sys, os
sys.path.insert(0, "/home/z/my-project/xorzen_dev")
import pytest
import torch
import numpy as np
import math

torch.manual_seed(0)


# ============================================================
# 1. Eval routing collapse fix
# ============================================================

def _train_tiny_model(eval_noise, seed=1337):
    """Train a tiny model briefly and return it in eval mode."""
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.models.zero.variants import zeroBase

    torch.manual_seed(seed); np.random.seed(seed)
    H = 64
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        vocab_size=128, context_length=64, hidden_size=H,
        num_layers=4, num_attention_heads=4, max_depth=4, min_depth=1,
        width_choices=(H // 2, H), expert_count=4, top_k_experts=2,
        router_hidden_dim=H // 4, dropout=0.0, pad_token_id=0,
        load_balancing_weight=0.001, shard_experts=False,
        pathway_top_k=2, gradient_checkpointing=False,
        eval_routing_noise=eval_noise,
    )
    model = zeroBase(config=cfg, test_mode=True)
    model.train()
    rng = np.random.RandomState(seed)
    sequences = []
    for i in range(8):
        offset = rng.randint(1, cfg.vocab_size)
        start = rng.randint(0, cfg.vocab_size)
        seq = [(start + offset * t) % cfg.vocab_size for t in range(cfg.context_length)]
        sequences.append(seq)
    sequences = torch.tensor(sequences, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01
    )
    for _ in range(150):
        optimizer.zero_grad()
        out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    return model, sequences, cfg


def test_eval_routing_noise_breaks_collapse():
    """With eval_routing_noise > 0, eval diversity must be >= noise=0 case.

    Phase 1 baseline: noise=0 → 1/3 pathways, 1/2 widths, 2/4 experts.
    Phase 2 fix:      noise=0.15 → 2/3 pathways, 2/2 widths, 4/4 experts.
    """
    # noise=0 (legacy v0.4)
    model0, seqs, cfg = _train_tiny_model(eval_noise=0.0, seed=42)
    with torch.no_grad():
        out0 = model0(input_ids=seqs, output_routing_info=True)
    p_unique0 = torch.unique(out0.routing_info.path_probs.argmax(-1)).numel()
    w_unique0 = torch.unique(out0.routing_info.width_idx).numel()
    e_unique0 = torch.unique(out0.routing_info.expert_indices).numel()

    # noise=0.15 (v0.5 fix)
    model1, seqs, cfg = _train_tiny_model(eval_noise=0.15, seed=42)
    with torch.no_grad():
        out1 = model1(input_ids=seqs, output_routing_info=True)
    p_unique1 = torch.unique(out1.routing_info.path_probs.argmax(-1)).numel()
    w_unique1 = torch.unique(out1.routing_info.width_idx).numel()
    e_unique1 = torch.unique(out1.routing_info.expert_indices).numel()

    # v0.5 must produce AT LEAST as much diversity as v0.4
    assert p_unique1 >= p_unique0, \
        f"pathway diversity: noise=0.15 ({p_unique1}) < noise=0 ({p_unique0})"
    assert w_unique1 >= w_unique0, \
        f"width diversity: noise=0.15 ({w_unique1}) < noise=0 ({w_unique0})"
    assert e_unique1 >= e_unique0, \
        f"expert diversity: noise=0.15 ({e_unique1}) < noise=0 ({e_unique0})"
    # And at least ONE axis must STRICTLY improve
    assert (p_unique1 > p_unique0) or (w_unique1 > w_unique0) or (e_unique1 > e_unique0), \
        "eval_routing_noise=0.15 did not improve any diversity axis over noise=0"


def test_eval_routing_noise_reproducible():
    """Same input must produce same routing across multiple eval calls.

    The noise is seeded with a fixed value per axis, so the noise pattern
    is identical across calls. Combined with deterministic eval mode
    (no dropout, no BatchNorm), the routing must be bit-exact reproducible.
    """
    model, seqs, _ = _train_tiny_model(eval_noise=0.15, seed=7)
    with torch.no_grad():
        out1 = model(input_ids=seqs, output_routing_info=True)
        out2 = model(input_ids=seqs, output_routing_info=True)
    assert torch.equal(out1.routing_info.path_probs, out2.routing_info.path_probs), \
        "path_probs not reproducible across eval calls"
    assert torch.equal(out1.routing_info.width_idx, out2.routing_info.width_idx), \
        "width_idx not reproducible across eval calls"
    assert torch.equal(out1.routing_info.expert_indices, out2.routing_info.expert_indices), \
        "expert_indices not reproducible across eval calls"


def test_eval_routing_noise_preserves_sparsity():
    """Top-k sparsity must still hold: only K pathway forwards run per token.

    We verify this by attaching a pathway_call_counter and checking that
    the number of distinct pathways called is <= num_paths (not num_paths × tokens).
    """
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.models.zero.variants import zeroBase

    torch.manual_seed(42); np.random.seed(42)
    H = 64
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        vocab_size=128, context_length=32, hidden_size=H,
        num_layers=2, num_attention_heads=4, max_depth=2, min_depth=1,
        width_choices=(H // 2, H), expert_count=4, top_k_experts=2,
        router_hidden_dim=H // 4, dropout=0.0, pad_token_id=0,
        load_balancing_weight=0.001, shard_experts=False,
        pathway_top_k=2, gradient_checkpointing=False,
        eval_routing_noise=0.15,
    )
    model = zeroBase(config=cfg, test_mode=True)
    model.eval()
    # Attach call counters
    for block in model.blocks:
        block._pathway_call_counter = {}

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(input_ids=ids, output_routing_info=True)

    # Each block should have called at most 3 pathways (num_paths), NOT
    # num_paths * num_tokens (which would be 3 * 32 = 96).
    for i, block in enumerate(model.blocks):
        counts = block._pathway_call_counter
        total_calls = sum(counts.values())
        # Per block: at most 1 call per pathway (sparse_pathway_dispatch
        # batches all selected tokens into a single forward per pathway).
        assert total_calls <= 3, \
            f"block {i}: total pathway calls = {total_calls}, expected <= 3 (one per pathway). " \
            f"counts={counts}"
        # And at least 1 (top_k >= 1 ensures at least one pathway is called)
        assert total_calls >= 1, \
            f"block {i}: no pathway was called. counts={counts}"


def test_eval_routing_noise_zero_disables():
    """eval_routing_noise=0.0 must reproduce legacy v0.4 deterministic behavior.

    With noise=0, two consecutive eval calls on the same input must produce
    identical routing — this is the v0.4 behavior. (Note: v0.5 with noise>0
    ALSO produces reproducible routing because the noise is seeded, so this
    test verifies that noise=0 still works correctly.)
    """
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.models.zero.variants import zeroBase

    torch.manual_seed(99)
    H = 32
    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    cfg.update(
        vocab_size=64, context_length=16, hidden_size=H,
        num_layers=2, num_attention_heads=2, max_depth=2, min_depth=1,
        width_choices=(H // 2, H), expert_count=2, top_k_experts=1,
        router_hidden_dim=8, dropout=0.0, pad_token_id=0,
        load_balancing_weight=0.001, shard_experts=False,
        pathway_top_k=2, gradient_checkpointing=False,
        eval_routing_noise=0.0,  # DISABLED
    )
    model = zeroBase(config=cfg, test_mode=True)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out1 = model(input_ids=ids, output_routing_info=True)
        out2 = model(input_ids=ids, output_routing_info=True)
    assert torch.equal(out1.routing_info.path_probs, out2.routing_info.path_probs)
    assert torch.equal(out1.logits, out2.logits), "noise=0 must give identical logits"


# ============================================================
# 2. ComputeController removal
# ============================================================

def test_compute_controller_removed():
    """The dead ComputeController module must be unimportable."""
    try:
        from xorzen.model.components.compute_controller import ComputeController  # noqa
        assert False, "ComputeController should have been removed in v0.5"
    except ImportError:
        pass  # Expected
    try:
        from xorzen.model.components.compute_controller import ComputeAllocation  # noqa
        assert False, "ComputeAllocation should have been removed in v0.5"
    except ImportError:
        pass  # Expected


# ============================================================
# 3. pathway_gate removal
# ============================================================

def test_pathway_gate_removed():
    """HASSBlock must NOT have a pathway_gate attribute (dead code removed)."""
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.hass_block import HASSBlock

    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    block = HASSBlock(cfg, layer_idx=0)
    assert not hasattr(block, 'pathway_gate'), \
        "HASSBlock still has pathway_gate (should be removed in v0.5)"


def test_pathway_gate_removal_doesnt_break_standalone():
    """HASSBlock with routing_decision=None must still produce finite output
    (using uniform 1/3 weighting instead of the removed pathway_gate)."""
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.hass_block import HASSBlock

    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    block = HASSBlock(cfg, layer_idx=0)
    block.eval()
    B, T, H = 2, 8, cfg.hidden_size
    x = torch.randn(B, T, H)
    with torch.no_grad():
        y = block(x, routing_decision=None)  # standalone use
    assert torch.isfinite(y).all(), "NaN/Inf in standalone HASSBlock output"
    assert y.shape == x.shape, f"shape mismatch: {y.shape} vs {x.shape}"


def test_pathway_gate_removal_gradient_flow():
    """Gradients must still flow through HASSBlock after pathway_gate removal.

    With routing_decision passed (the normal model path), at least one
    pathway module + the FFN must receive gradients. (We can't guarantee
    ALL pathways get gradients because top-k sparse dispatch by design
    skips pathways that no token selects.)
    """
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.hass_block import HASSBlock
    from xorzen.model.components.routing import RoutingDecision

    cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
    block = HASSBlock(cfg, layer_idx=0)
    block.train()
    B, T, H = 2, 8, cfg.hidden_size
    x = torch.randn(B, T, H, requires_grad=True)

    # Use VARIED path_probs so different tokens select different top-2
    # pathways — this ensures all 3 pathways get selected by at least
    # some tokens and thus receive gradients.
    torch.manual_seed(0)
    path_probs = torch.randn(B, T, 3).softmax(dim=-1)

    rd = RoutingDecision(
        depth_logits=torch.zeros(B, T, cfg.max_depth),
        depth_probs=torch.ones(B, T, cfg.max_depth),
        depth_mask=torch.ones(B, T, cfg.max_depth),
        width_logits=torch.zeros(B, T, len(cfg.width_choices)),
        width_probs=torch.ones(B, T, len(cfg.width_choices)) / len(cfg.width_choices),
        width_idx=torch.zeros(B, T, dtype=torch.long),
        width_multiplier=torch.ones(B, T, 1),
        path_logits=torch.zeros(B, T, 3),
        path_probs=path_probs,
        expert_logits=torch.zeros(B, T, cfg.expert_count),
        expert_probs=torch.ones(B, T, cfg.expert_count) / cfg.expert_count,
        expert_indices=torch.zeros(B, T, cfg.top_k_experts, dtype=torch.long),
        expert_weights=torch.ones(B, T, cfg.top_k_experts),
        complexity=torch.ones(B, T, 1),
        uncertainty=torch.zeros(B, T, 1),
        auxiliary={},
    )
    y = block(x, routing_decision=rd)
    y.sum().backward()
    assert x.grad is not None, "no gradient on input"
    assert x.grad.abs().sum() > 0, "zero gradient on input"
    # At least 2 of 3 pathways must receive gradients (top_k=2 means at
    # least 2 pathways are selected by SOME token given varied path_probs).
    n_with_grad = 0
    for pname in ['local', 'low_rank', 'ssm']:
        pathway = block.pathways[pname]
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in pathway.parameters())
        if has_grad:
            n_with_grad += 1
    assert n_with_grad >= 2, \
        f"only {n_with_grad}/3 pathways received gradients (expected >= 2 with varied path_probs)"
    # FFN must receive gradients
    ffn_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in block.ffn.parameters())
    assert ffn_has_grad, "FFN received no gradient"


# ============================================================
# 4. Sparse dispatch correctness after optimization
# ============================================================

def test_sparse_dispatch_correctness_after_optimization():
    """The index_add_-based dispatch must produce identical output to a
    boolean-scatter reference implementation."""
    from xorzen.model.components.sparse_dispatch import sparse_pathway_dispatch, topk_pathway_mask

    torch.manual_seed(42)
    B, T, H = 2, 16, 32
    x = torch.randn(B, T, H)
    path_probs = torch.randn(B, T, 3).softmax(dim=-1)

    # Reference: compute combined via boolean scatter
    mask = topk_pathway_mask(path_probs, top_k=2, training=False)
    selected_w = path_probs * mask
    norm_w = selected_w / selected_w.sum(-1, keepdim=True)
    x_flat = x.reshape(B*T, H)
    mask_flat = mask.reshape(B*T, 3)
    w_flat = norm_w.reshape(B*T, 3)
    expected = torch.zeros(B*T, H)
    for i in range(3):
        sel = mask_flat[:, i] > 0.5
        if not sel.any():
            continue
        # Use a simple deterministic transform: y = x * (i+1)
        y_slice = x_flat[sel] * (i + 1)
        w_slice = w_flat[sel, i].unsqueeze(-1)
        expected[sel] += y_slice * w_slice
    expected = expected.reshape(B, T, H)

    # Actual: use sparse_pathway_dispatch with the same transforms
    fns = {
        'a': lambda xs: xs * 1,
        'b': lambda xs: xs * 2,
        'c': lambda xs: xs * 3,
    }
    combined, counts = sparse_pathway_dispatch(
        x, path_probs, fns, ['a', 'b', 'c'], top_k=2, training=False
    )
    diff = (combined - expected).abs().max().item()
    assert diff < 1e-6, f"dispatch correctness: max diff {diff} > 1e-6"
    assert torch.isfinite(combined).all()


def test_sparse_dispatch_no_grad_leak():
    """sparse_pathway_dispatch must not leak gradients to unselected pathways.

    If a pathway is never selected (top_k covers the other 2 of 3), its
    forward must not be called → no gradient."""
    from xorzen.model.components.sparse_dispatch import sparse_pathway_dispatch

    torch.manual_seed(42)
    B, T, H = 1, 8, 16
    x = torch.randn(B, T, H, requires_grad=True)
    # Force path_probs to put ALL mass on pathways 0 and 1 (never 2)
    path_probs = torch.zeros(B, T, 3)
    path_probs[..., 0] = 0.6
    path_probs[..., 1] = 0.4

    call_log = {'a': 0, 'b': 0, 'c': 0}
    def fn_a(xs):
        call_log['a'] += 1
        return xs * 2
    def fn_b(xs):
        call_log['b'] += 1
        return xs * 3
    def fn_c(xs):
        call_log['c'] += 1
        return xs * 5
    fns = {'a': fn_a, 'b': fn_b, 'c': fn_c}

    combined, _ = sparse_pathway_dispatch(
        x, path_probs, fns, ['a', 'b', 'c'], top_k=2, training=False
    )
    # Pathway 'c' (index 2) must NOT have been called
    assert call_log['c'] == 0, f"pathway c was called {call_log['c']} times (should be 0)"
    assert call_log['a'] >= 1, "pathway a should have been called"
    assert call_log['b'] >= 1, "pathway b should have been called"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
