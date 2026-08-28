"""
Load-balancing auxiliary loss for Mixture-of-Experts.

This module provides a single canonical implementation of the Switch
Transformer load-balancing loss:

    L_lb = N * Sum_e  f_e * P_e

where:
    N      = number of experts
    f_e    = fraction of tokens dispatched to expert e   (in [0, 1], sums to 1)
    P_e    = mean router probability assigned to expert e (in [0, 1], sums to 1)

Properties (proven in reports/xorzen_verified_proofs.md §P5):
    L_lb >= 0
    L_lb == 1   iff routing is perfectly balanced (f_e = P_e = 1/N for all e)
    L_lb == N   iff all tokens collapse to a single expert
    When the router is consistent (f_e == P_e), L_lb >= 1 by Cauchy-Schwarz.

API
---
The loss accepts the inputs in EITHER of two shapes:

[A] Token-wise probabilities + selected indices (preferred, from AdaptiveRouter):
    ``load_balance_loss_switch(router_probs, expert_indices, num_experts)``
        router_probs:   [B, S, E]  or  [N, E]   softmax probabilities per token
        expert_indices: [B, S, K]  or  [N, K]   top-k selected expert ids per token

[B] Pre-computed (f, P) vectors (for tests and external callers):
    ``load_balance_loss_from_fp(f, p)``
        f: [E]  fraction of tokens per expert (sums to 1)
        p: [E]  mean router prob per expert (sums to 1)

Both entry points return a scalar tensor. Gradients flow through ``p`` (and
through ``router_probs``); ``f`` is a one-hot scatter so it carries no
gradient, which matches the Switch Transformer paper (the dispatch is
treated as a stop-gradient sample).

The old CV-based and L2-based formulas in this codebase
(``RoutingMathematics.load_balancing_loss`` in math_utils.py,
``AdaptiveRouter._compute_balancing_loss`` in routing.py, and
``ShardedExpertFabric._compute_load_balance_loss`` in zmoe.py) are kept for
backward compatibility but should be migrated to this module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "load_balance_loss_switch",
    "load_balance_loss_from_fp",
    "compute_load_fractions",
    "compute_mean_router_probs",
]


def _normalize_probs_and_indices(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
):
    """
    Flatten [B, S, E] -> [N, E] and [B, S, K] -> [N, K] so the rest of the
    function can be shape-agnostic. Also validates shapes.
    """
    if router_probs.dim() == 3:
        B, S, E = router_probs.shape
        probs_flat = router_probs.reshape(B * S, E)
    elif router_probs.dim() == 2:
        probs_flat = router_probs
    else:
        raise ValueError(
            f"router_probs must be [B,S,E] or [N,E], got shape {tuple(router_probs.shape)}"
        )

    if expert_indices.dim() == 3:
        idx_flat = expert_indices.reshape(-1, expert_indices.shape[-1])
    elif expert_indices.dim() == 2:
        idx_flat = expert_indices
    else:
        raise ValueError(
            f"expert_indices must be [B,S,K] or [N,K], got shape {tuple(expert_indices.shape)}"
        )

    N_tokens = probs_flat.shape[0]
    if idx_flat.shape[0] != N_tokens:
        raise ValueError(
            f"router_probs has {N_tokens} tokens but expert_indices has {idx_flat.shape[0]}"
        )
    return probs_flat, idx_flat


def compute_load_fractions(
    expert_indices: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """
    Compute f_e = fraction of (token, slot) pairs dispatched to expert e.

    Args:
        expert_indices: [N, K] long tensor of selected expert ids.
        num_experts: E
        top_k: K (number of slots per token)

    Returns:
        f: [E] float tensor, sums to 1.
    """
    # One-hot the indices and sum over tokens+slots, then normalize by N*K.
    one_hot = F.one_hot(expert_indices, num_classes=num_experts).float()  # [N, K, E]
    counts = one_hot.sum(dim=[0, 1])  # [E]
    return counts / (expert_indices.numel() + 1e-12)


def compute_mean_router_probs(
    router_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute P_e = mean router probability assigned to expert e.

    Args:
        router_probs: [N, E] softmax probabilities per token.

    Returns:
        p: [E] float tensor, sums to ~1.
    """
    return router_probs.mean(dim=0)


def load_balance_loss_from_fp(
    f: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """
    Switch Transformer load-balance loss from pre-computed (f, p).

        L_lb = N * Sum_e f_e * p_e

    Args:
        f: [E] fraction of tokens per expert (sums to 1)
        p: [E] mean router prob per expert (sums to 1)

    Returns:
        scalar tensor.
    """
    num_experts = f.shape[-1]
    return num_experts * (f * p).sum()


def load_balance_loss_switch(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Switch Transformer load-balance loss.

    Args:
        router_probs:   [B, S, E] or [N, E] — softmax probabilities per token
        expert_indices: [B, S, K] or [N, K] — top-k selected expert ids
        num_experts: E

    Returns:
        scalar tensor. Gradients flow through router_probs.

    Bounds (see reports/xorzen_verified_proofs.md §P5):
        L_lb in [1, N] when the router is consistent (f == p)
        L_lb = 1  for perfectly balanced routing
        L_lb = N  for complete collapse to one expert
    """
    probs_flat, idx_flat = _normalize_probs_and_indices(
        router_probs, expert_indices
    )
    f = compute_load_fractions(idx_flat, num_experts, idx_flat.shape[-1])
    p = compute_mean_router_probs(probs_flat)
    return load_balance_loss_from_fp(f, p)
