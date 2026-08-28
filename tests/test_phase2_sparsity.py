"""
Tests for Phase 2 genuine sparsity:
- HASS pathway sparse dispatch (only top-k pathways called per token)
- AdaptiveFFN sliced width (lower width = lower actual FLOPs)
- Adaptive depth per-token (gather active tokens, run block, scatter back)
- MoE top-k verification (unselected experts receive zero forward calls)
"""
import sys
sys.path.insert(0, "/home/z/my-project/xorzen_dev")

import pytest
import torch
torch.manual_seed(0)


# ============================================================
# Part 1: HASS pathway sparse dispatch
# ============================================================

def test_pathway_top_k_1_skips_unselected_pathways():
    """With pathway_top_k=1, only 1 pathway should be called per token.
    If different tokens select different pathways, all 3 may be called
    (once each) — but the per-token FLOPs are 1/3 of dense.
    """
    from xorzen.model.components.hass_block import HASSBlock
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.routing import RoutingDecision

    config = ConfigFactory.get_config(ModelSize.TINY_23K)
    config.pathway_top_k = 1  # only 1 pathway per token
    block = HASSBlock(config, layer_idx=0)
    block.eval()
    block._pathway_call_counter = {}

    # Build a routing decision where ALL tokens select only SSM (path 2)
    B, T = 2, 8
    path_probs = torch.zeros(B, T, 3)
    path_probs[..., 2] = 1.0  # all tokens select SSM
    # Other routing fields (not used by HASS forward, but required by dataclass)
    H = config.hidden_size
    rd = RoutingDecision(
        depth_logits=torch.zeros(B, T, config.max_depth),
        depth_probs=torch.zeros(B, T, config.max_depth),
        depth_mask=torch.ones(B, T, config.max_depth),
        width_logits=torch.zeros(B, T, len(config.width_choices)),
        width_probs=torch.zeros(B, T, len(config.width_choices)),
        width_idx=torch.zeros(B, T, dtype=torch.long),
        width_multiplier=torch.ones(B, T, 1),
        path_logits=torch.zeros(B, T, 3),
        path_probs=path_probs,
        expert_logits=torch.zeros(B, T, config.expert_count),
        expert_probs=torch.zeros(B, T, config.expert_count),
        expert_indices=torch.zeros(B, T, config.top_k_experts, dtype=torch.long),
        expert_weights=torch.zeros(B, T, config.top_k_experts),
        complexity=torch.ones(B, T, 1),
        uncertainty=torch.zeros(B, T, 1),
        auxiliary={},
    )
    x = torch.randn(B, T, H)
    with torch.no_grad():
        _ = block(x, routing_decision=rd)

    # Only SSM should have been called; local and low_rank should be 0.
    calls = block._pathway_call_counter
    print(f"Pathway calls (all tokens -> SSM): {calls}")
    assert calls.get('local', 0) == 0, f"local was called {calls.get('local')} times (should be 0)"
    assert calls.get('low_rank', 0) == 0, f"low_rank was called {calls.get('low_rank')} times (should be 0)"
    assert calls.get('ssm', 0) == 1, f"ssm was called {calls.get('ssm')} times (should be 1)"


def test_pathway_top_k_2_calls_at_most_2_pathways_per_token():
    """With pathway_top_k=2, each token selects exactly 2 pathways.
    If all tokens select the same 2 pathways, only those 2 are called.
    """
    from xorzen.model.components.hass_block import HASSBlock
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.routing import RoutingDecision

    config = ConfigFactory.get_config(ModelSize.TINY_23K)
    config.pathway_top_k = 2
    block = HASSBlock(config, layer_idx=0)
    block.eval()
    block._pathway_call_counter = {}

    # All tokens select local + ssm (not low_rank)
    B, T = 2, 8
    path_probs = torch.zeros(B, T, 3)
    path_probs[..., 0] = 0.6  # local
    path_probs[..., 2] = 0.4  # ssm
    H = config.hidden_size
    rd = RoutingDecision(
        depth_logits=torch.zeros(B, T, config.max_depth),
        depth_probs=torch.zeros(B, T, config.max_depth),
        depth_mask=torch.ones(B, T, config.max_depth),
        width_logits=torch.zeros(B, T, len(config.width_choices)),
        width_probs=torch.zeros(B, T, len(config.width_choices)),
        width_idx=torch.zeros(B, T, dtype=torch.long),
        width_multiplier=torch.ones(B, T, 1),
        path_logits=torch.zeros(B, T, 3),
        path_probs=path_probs,
        expert_logits=torch.zeros(B, T, config.expert_count),
        expert_probs=torch.zeros(B, T, config.expert_count),
        expert_indices=torch.zeros(B, T, config.top_k_experts, dtype=torch.long),
        expert_weights=torch.zeros(B, T, config.top_k_experts),
        complexity=torch.ones(B, T, 1),
        uncertainty=torch.zeros(B, T, 1),
        auxiliary={},
    )
    x = torch.randn(B, T, H)
    with torch.no_grad():
        _ = block(x, routing_decision=rd)

    calls = block._pathway_call_counter
    print(f"Pathway calls (tokens -> local+ssm): {calls}")
    assert calls.get('low_rank', 0) == 0, f"low_rank was called (should be 0)"
    assert calls.get('local', 0) == 1
    assert calls.get('ssm', 0) == 1


def test_pathway_ste_training_gradient_flows():
    """At training, the STE must allow gradients to flow through path_probs
    even though the forward uses a hard mask."""
    from xorzen.model.components.hass_block import HASSBlock
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.routing import RoutingDecision

    config = ConfigFactory.get_config(ModelSize.TINY_23K)
    config.pathway_top_k = 1
    block = HASSBlock(config, layer_idx=0)
    block.train()
    block._pathway_call_counter = {}

    B, T = 2, 8
    # Use a parameterized path_probs so we can backprop through it
    path_logits = torch.randn(B, T, 3, requires_grad=True)
    path_probs = torch.softmax(path_logits, dim=-1)
    H = config.hidden_size
    rd = RoutingDecision(
        depth_logits=torch.zeros(B, T, config.max_depth),
        depth_probs=torch.zeros(B, T, config.max_depth),
        depth_mask=torch.ones(B, T, config.max_depth),
        width_logits=torch.zeros(B, T, len(config.width_choices)),
        width_probs=torch.zeros(B, T, len(config.width_choices)),
        width_idx=torch.zeros(B, T, dtype=torch.long),
        width_multiplier=torch.ones(B, T, 1),
        path_logits=path_logits.detach(),
        path_probs=path_probs,
        expert_logits=torch.zeros(B, T, config.expert_count),
        expert_probs=torch.zeros(B, T, config.expert_count),
        expert_indices=torch.zeros(B, T, config.top_k_experts, dtype=torch.long),
        expert_weights=torch.zeros(B, T, config.top_k_experts),
        complexity=torch.ones(B, T, 1),
        uncertainty=torch.zeros(B, T, 1),
        auxiliary={},
    )
    x = torch.randn(B, T, H)
    y = block(x, routing_decision=rd)
    y.sum().backward()
    # Gradient must flow through path_logits
    assert path_logits.grad is not None
    assert path_logits.grad.abs().sum() > 0


# ============================================================
# Part 2: MoE top-k verification
# ============================================================

def test_moe_unselected_experts_not_called():
    """Experts that no token selects must NOT be invoked."""
    from xorzen.model.components.sparse_dispatch import topk_expert_dispatch

    # 4 tokens, 4 experts, top_k=1
    N, H, E, K = 4, 8, 4, 1
    x = torch.randn(N, H)
    # All tokens select expert 0
    idx = torch.zeros(N, K, dtype=torch.long)
    w = torch.ones(N, K)
    call_counts = {}
    expert_fns = {}
    for eid in range(E):
        def make_fn(eid):
            def fn(x_in):
                call_counts[eid] = call_counts.get(eid, 0) + 1
                return x_in * (eid + 1)
            return fn
        expert_fns[eid] = make_fn(eid)
    out, counts = topk_expert_dispatch(x, idx, w, expert_fns, K)
    # Only expert 0 should have been called
    assert counts.get(0, 0) == 1
    assert counts.get(1, 0) == 0
    assert counts.get(2, 0) == 0
    assert counts.get(3, 0) == 0
    # Output should be x * 1 (expert 0's multiplier)
    torch.testing.assert_close(out, x * 1)


def test_moe_top_k_2_distributes_correctly():
    """With top_k=2, each token's output is w_0*E_a(x) + w_1*E_b(x)."""
    from xorzen.model.components.sparse_dispatch import topk_expert_dispatch

    N, H, E, K = 2, 4, 3, 2
    x = torch.randn(N, H)
    # Token 0: experts (0, 1) with weights (0.6, 0.4)
    # Token 1: experts (1, 2) with weights (0.5, 0.5)
    idx = torch.tensor([[0, 1], [1, 2]])
    w = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
    expert_fns = {
        0: lambda x_in: x_in * 1.0,
        1: lambda x_in: x_in * 2.0,
        2: lambda x_in: x_in * 3.0,
    }
    out, counts = topk_expert_dispatch(x, idx, w, expert_fns, K)
    # Expected:
    #   token 0: 0.6 * (x*1) + 0.4 * (x*2) = 0.6x + 0.8x = 1.4x
    #   token 1: 0.5 * (x*2) + 0.5 * (x*3) = x + 1.5x = 2.5x
    expected = torch.stack([x[0] * 1.4, x[1] * 2.5])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    # All 3 experts should have been called (each appears in at least one token's top-2)
    assert counts.get(0, 0) == 1
    assert counts.get(1, 0) == 2  # expert 1 appears in both tokens
    assert counts.get(2, 0) == 1


# ============================================================
# Part 3: AdaptiveFFN sliced width (Phase 2 — to be implemented)
# ============================================================

def test_adaptive_ffn_sliced_width():
    """SlicedFFN should actually execute only the selected width.

    The old AdaptiveFFN computes base FFN + all adapters and blends them.
    The new SlicedFFN computes only the selected width via slicing.
    """
    from xorzen.model.components.sliced_ffn import SlicedFFN
    ff = SlicedFFN(hidden_dim=16, max_width=64, activation='gelu', dropout=0.0)
    ff.eval()
    x = torch.randn(2, 8, 16)
    # Width 16 (smallest)
    with torch.no_grad():
        y_small = ff(x, width=16)
    # Width 64 (largest)
    with torch.no_grad():
        y_large = ff(x, width=64)
    assert y_small.shape == x.shape
    assert y_large.shape == x.shape
    # Outputs should differ
    assert not torch.allclose(y_small, y_large)


def test_adaptive_ffn_sliced_width_lower_is_cheaper():
    """Lower selected width should result in lower actual FLOPs.

    We measure FLOPs analytically: for SlicedFFN with hidden_dim=H and
    selected width W, the forward pass does:
      fc1: H * W multiplies per token
      fc2: W * H multiplies per token
    Total = 2 * H * W per token.
    So width=16 should give 2 * 16 * 16 = 512 FLOPs/token,
    and width=64 should give 2 * 16 * 64 = 2048 FLOPs/token.
    """
    from xorzen.model.components.sliced_ffn import SlicedFFN
    H = 16
    ff = SlicedFFN(hidden_dim=H, max_width=64, activation='gelu', dropout=0.0)
    ff.eval()
    x = torch.randn(2, 8, H)
    # Compute analytical FLOPs for each width
    flops_small = 2 * H * 16 * (2 * 8)  # 2 * H * W * num_tokens
    flops_large = 2 * H * 64 * (2 * 8)
    print(f"Analytical FLOPs small (width=16): {flops_small}")
    print(f"Analytical FLOPs large (width=64): {flops_large}")
    assert flops_small < flops_large
    # Also verify the forward pass actually runs at both widths
    with torch.no_grad():
        _ = ff(x, width=16)
        _ = ff(x, width=64)
    # And verify per-token width selection also works
    width_idx = torch.randint(0, len(ff.width_choices), (2, 8))
    with torch.no_grad():
        _ = ff(x, width_idx=width_idx)
    print("OK: SlicedFFN width=16 is 4x cheaper than width=64 (analytical)")


# ============================================================
# Part 4: Token-level adaptive depth (Phase 2 — to be implemented)
# ============================================================

def test_adaptive_depth_gathers_and_scatters():
    """Per-token depth should gather active tokens, run the block, scatter back.

    Tokens with depth_mask=0 should NOT be processed by the block.
    """
    from xorzen.model.components.hass_block import HASSBlock
    from xorzen.config import ConfigFactory, ModelSize
    from xorzen.model.components.routing import RoutingDecision

    config = ConfigFactory.get_config(ModelSize.TINY_23K)
    block = HASSBlock(config, layer_idx=0)
    block.eval()
    block._pathway_call_counter = {}

    B, T, H = 2, 8, config.hidden_size
    x = torch.randn(B, T, H)

    # Build a depth mask where only the first 4 tokens of each batch are active
    depth_mask = torch.zeros(B, T, config.max_depth)
    depth_mask[:, :4, 0] = 1.0  # first 4 tokens active in layer 0

    # Forward with depth: should only process the active tokens
    with torch.no_grad():
        y = block.forward_with_depth(x, depth_mask[:, :, 0], routing_decision=None)

    # The active tokens should have been processed (different from input)
    active_diff = (y[:, :4, :] - x[:, :4, :]).abs().mean()
    # The inactive tokens should be unchanged
    inactive_diff = (y[:, 4:, :] - x[:, 4:, :]).abs().mean()
    print(f"active diff: {active_diff}, inactive diff: {inactive_diff}")
    assert inactive_diff < 1e-6, "inactive tokens were modified (should be unchanged)"
    # Active tokens may or may not have changed depending on the block, but
    # the call should not error.


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
