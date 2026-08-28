"""
Regression test for P5: Switch Transformer load-balance loss normalization.

Bug: The old load_balance_loss in routing.py computed f = dispatch.mean(dim=[0,1])
which sums to top_k (not 1), giving L=K for balanced routing instead of L=1.

Fix: Delegated to load_balance_loss_switch from load_balance.py which correctly
normalizes f by N*K.

This test would FAIL on the old implementation and PASS on the fixed one.
"""

import sys
sys.path.insert(0, "/home/z/my-project/DevNet")

import pytest
import torch
import torch.nn.functional as F


def test_lb_loss_balanced_equals_one():
    """For uniform routing, Switch LB loss must equal exactly 1.0 (not K)."""
    from xorzen.model.components.routing import load_balance_loss

    E = 4  # experts
    K = 2  # top_k
    B = 2
    S = 8

    # Perfectly uniform routing: every expert gets equal probability
    # and every token routes uniformly.
    # P_e = 1/E for all e.
    # f_e = 1/E for all e (with correct normalization by N*K).
    # L = E * sum(f*p) = E * E * (1/E)*(1/E) = 1.

    probs = torch.full((B, S, E), 1.0 / E)
    # Distribute expert indices evenly so each expert gets K * B * S / E tokens
    indices = torch.zeros(B, S, K, dtype=torch.long)
    idx = 0
    for b in range(B):
        for s in range(S):
            for k_idx in range(K):
                indices[b, s, k_idx] = idx % E
                idx += 1

    loss = load_balance_loss(probs, indices, E)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert abs(loss.item() - 1.0) < 1e-4, (
        f"Balanced routing should give L=1.0, got {loss.item():.6f}. "
        f"The old buggy implementation would give L={K}."
    )


def test_lb_loss_collapse_equals_E():
    """For complete collapse to one expert, Switch LB loss must equal E."""
    from xorzen.model.components.routing import load_balance_loss

    E = 4
    K = 2
    B = 2
    S = 8

    # Collapse: all tokens route to expert 0 with probability 1.
    probs = torch.zeros(B, S, E)
    probs[..., 0] = 1.0
    indices = torch.zeros(B, S, K, dtype=torch.long)  # all -> expert 0

    loss = load_balance_loss(probs, indices, E)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    # For collapse: f_0 = 1, p_0 = 1, all others = 0.
    # L = E * (1*1 + 0 + ... + 0) = E
    assert abs(loss.item() - E) < 1e-3, (
        f"Collapsed routing should give L={E}, got {loss.item():.6f}"
    )


def test_lb_loss_lb_ge_1():
    """L_lb >= 1 for consistent routing (f == p)."""
    from xorzen.model.components.routing import load_balance_loss

    E = 8
    K = 2
    B = 4
    S = 16

    torch.manual_seed(42)
    # Random but consistent routing: indices come from the same distribution as probs
    logits = torch.randn(B, S, E)
    probs = F.softmax(logits, dim=-1)
    _, indices = torch.topk(probs, K, dim=-1)

    loss = load_balance_loss(probs, indices, E)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    # For consistent routing (f≈p), L_lb >= 1 by Cauchy-Schwarz
    assert loss.item() >= 0.99, (
        f"Consistent routing should give L_lb >= 1, got {loss.item():.6f}"
    )


def test_lb_loss_gradient_flows_through_probs():
    """Gradient must flow through router_probs (not through f/dispatch)."""
    from xorzen.model.components.routing import load_balance_loss

    E = 4
    K = 2
    B = 2
    S = 4

    probs = torch.randn(B, S, E, requires_grad=True)
    probs_norm = F.softmax(probs, dim=-1)
    probs_norm.retain_grad()
    _, indices = torch.topk(probs_norm.detach(), K, dim=-1)

    loss = load_balance_loss(probs_norm, indices, E)
    loss.backward()

    assert probs_norm.grad is not None, "No gradient on router_probs"
    assert probs_norm.grad.abs().sum() > 0, "Gradient is all zeros"


def test_lb_loss_matches_load_balance_py():
    """routing.load_balance_loss must match load_balance.load_balance_loss_switch."""
    from xorzen.model.components.routing import load_balance_loss
    from xorzen.model.components.load_balance import load_balance_loss_switch

    E = 4
    K = 2
    B = 3
    S = 12

    torch.manual_seed(99)
    logits = torch.randn(B, S, E)
    probs = F.softmax(logits, dim=-1)
    _, indices = torch.topk(probs, K, dim=-1)

    loss_routing = load_balance_loss(probs, indices, E)
    loss_canonical = load_balance_loss_switch(probs, indices, E)

    assert torch.allclose(loss_routing, loss_canonical, atol=1e-6), (
        f"routing.load_balance_loss ({loss_routing.item():.6f}) != "
        f"load_balance.load_balance_loss_switch ({loss_canonical.item():.6f})"
    )
