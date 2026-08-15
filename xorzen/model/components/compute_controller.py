"""
Hierarchical Compute Controller for Xorzen.

This module implements a single compute allocator that coordinates ALL
routing decisions (depth, width, pathway, experts) into a globally
coherent allocation. The previous architecture had four independent
routers (depth_router, width_router, path_router, expert_router) that
could conflict — e.g. depth says "skip this layer" but width says "use
max width", wasting the width compute on a layer that will be blended
out.

The ComputeController answers ONE question:

    «How much computation does this token deserve, and where should
       that computation be spent?»

It takes the token's hidden state and a global ``compute_budget`` in
[0, 1] and produces a unified ``ComputeAllocation`` that the HASS
blocks, FFN, and MoE fabric all consume.

The budget is a single scalar that trades off quality vs compute:
    budget = 1.00 → full compute (all pathways, max width, all layers, top-k=K)
    budget = 0.50 → medium compute (top-2 pathways, 75% width, 80% depth, top-k=K-1)
    budget = 0.25 → cheap compute (top-1 pathway, 50% width, 60% depth, top-k=1)
    budget = 0.10 → minimum compute (top-1 pathway, 25% width, 40% depth, top-k=1)

The controller learns to allocate the budget across the four axes
based on per-token difficulty, so easy tokens get a cheap route and
hard tokens get an expensive route, while respecting the global budget.

Auxiliary losses (see ``compute_stability_losses``):
    L_budget  = |actual_compute - target_compute|     (budget adherence)
    L_entropy = -mean(entropy(routing_distributions)) (prevent collapse)
    L_smooth  = mean(|routing_t - routing_{t-1}|)     (temporal smoothness)

These are added to the task loss with weights from config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xorzen.utils.logger import get_logger

logger = get_logger()

__all__ = [
    "ComputeAllocation",
    "ComputeController",
    "compute_stability_losses",
]


@dataclass
class ComputeAllocation:
    """
    Output of the ComputeController. All fields are per-token tensors of
    shape [B, T, ...] unless noted.

    This is consumed by:
      - HASSBlock: uses path_probs (top-k selected via pathway_top_k),
        depth_active (per-layer skip), width_idx (sliced FFN width).
      - ShardedExpertFabric: uses expert_indices, expert_weights.
    """
    # Pathway allocation: [B, T, num_paths] softmax probabilities.
    path_probs: torch.Tensor
    # Per-layer depth mask: [B, T, max_depth] in {0, 1} (hard) or (0, 1) (soft).
    depth_mask: torch.Tensor
    # Width selection: [B, T] index into width_choices.
    width_idx: torch.Tensor
    # Width probabilities (for training): [B, T, num_widths]
    width_probs: torch.Tensor
    # Expert routing: [B, T, top_k] indices, [B, T, top_k] weights.
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    # Per-token difficulty estimate (for logging / halting): [B, T, 1]
    difficulty: torch.Tensor
    # Per-token compute fraction actually allocated: [B, T, 1] in [0, 1].
    # This is the sum of (pathway_fraction * depth_fraction * width_fraction
    # * expert_fraction), normalized. Useful for the budget adherence loss.
    actual_compute: torch.Tensor
    # Auxiliary losses (populated by compute_stability_losses).
    auxiliary: Dict[str, torch.Tensor] = field(default_factory=dict)


class ComputeController(nn.Module):
    """
    Hierarchical compute allocator.

    The controller has:
      1. A feature encoder (3-layer MLP) that produces a per-token embedding.
      2. A difficulty estimator (sigmoid scalar) that estimates how hard
         the token is.
      3. Four allocation heads:
           - path_head:    [B, T, num_paths]
           - depth_head:   [B, T, max_depth]   (sigmoid, one per layer)
           - width_head:   [B, T, num_widths]  (softmax over discrete widths)
           - expert_head:  [B, T, num_experts] (softmax, then top-k)

    The global ``compute_budget`` (scalar in [0, 1]) modulates the
    temperature of all four heads: low budget → sharper distributions
    (more sparse), high budget → flatter distributions (more dense).

    The controller is differentiable end-to-end. At inference, hard
    top-k / argmax is used; at training, Gumbel-softmax + STE is used.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_depth: int,
        width_choices,
        num_paths: int,
        num_experts: int,
        top_k: int,
        router_hidden_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.width_choices = list(width_choices)
        self.num_paths = num_paths
        self.num_experts = num_experts
        self.top_k = top_k

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden_dim, router_hidden_dim),
            nn.LayerNorm(router_hidden_dim),
            nn.GELU(),
        )

        # Difficulty estimator
        self.difficulty_head = nn.Sequential(
            nn.Linear(router_hidden_dim, router_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(router_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Allocation heads
        self.path_head = nn.Linear(router_hidden_dim, num_paths)
        self.depth_head = nn.Linear(router_hidden_dim, max_depth)
        self.width_head = nn.Linear(router_hidden_dim, len(self.width_choices))
        self.expert_head = nn.Linear(router_hidden_dim, num_experts)

        # Budget modulation: learned scale and bias per head
        # (so the model can learn how to trade off budget across axes)
        self.budget_modulation = nn.Parameter(torch.zeros(4))  # 4 axes

        self._init_weights()
        logger.info("compute", f"ComputeController: hidden={hidden_dim}, "
                              f"max_depth={max_depth}, widths={self.width_choices}, "
                              f"paths={num_paths}, experts={num_experts}, top_k={top_k}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0 / math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        compute_budget: float = 1.0,
        training: bool = False,
        deterministic: bool = False,
    ) -> ComputeAllocation:
        """
        Args:
            x: [B, T, H] token hidden states.
            compute_budget: scalar in [0, 1]. 1.0 = full compute, 0.25 = quarter.
            training: if True, use Gumbel-softmax + STE for differentiability.
            deterministic: if True, hard argmax (inference).

        Returns:
            ComputeAllocation with all routing decisions.
        """
        B, T, H = x.shape
        features = self.encoder(x)  # [B, T, router_hidden_dim]
        difficulty = self.difficulty_head(features)  # [B, T, 1]

        # Budget modulation: lower budget → lower temperature (sharper)
        # The modulation parameter controls how strongly budget affects
        # each axis. Positive modulation = that axis shrinks more under
        # low budget.
        budget = float(max(0.0, min(1.0, compute_budget)))
        # Convert budget to a "sparsity pressure": high budget → low pressure.
        sparsity_pressure = 1.0 - budget

        # ===== PATH ALLOCATION =====
        path_logits = self.path_head(features)  # [B, T, num_paths]
        # Modulate by difficulty: hard tokens get more pathways (flatter dist)
        # Easy tokens get fewer pathways (sharper dist).
        path_temperature = 1.0 + sparsity_pressure * 2.0 * (1.0 - difficulty)  # [B, T, 1]
        if training:
            path_probs = F.gumbel_softmax(path_logits / path_temperature, tau=1.0, hard=False, dim=-1)
        else:
            path_probs = F.softmax(path_logits / path_temperature, dim=-1)

        # ===== DEPTH ALLOCATION =====
        depth_logits = self.depth_head(features)  # [B, T, max_depth]
        # Lower budget → fewer layers active. We shift the logits down by
        # an amount proportional to sparsity_pressure.
        depth_shift = -sparsity_pressure * 2.0 * (1.0 - self.budget_modulation[1].tanh())
        depth_logits = depth_logits + depth_shift
        # Modulate by difficulty: hard tokens get more layers.
        depth_bias = difficulty * 2.0  # [B, T, 1] broadcast to [B, T, max_depth]
        depth_logits = depth_logits + depth_bias
        if training:
            depth_probs = torch.sigmoid(depth_logits)
            # STE: hard mask in forward, soft in backward
            depth_mask_hard = (depth_probs > 0.5).float()
            depth_mask = depth_mask_hard - depth_probs.detach() + depth_probs
        else:
            depth_probs = torch.sigmoid(depth_logits)
            depth_mask = (depth_probs > 0.5).float()

        # ===== WIDTH ALLOCATION =====
        width_logits = self.width_head(features)  # [B, T, num_widths]
        # Lower budget → bias toward smaller widths.
        # width_choices are sorted ascending, so smaller widths have lower index.
        width_bias = torch.linspace(
            -sparsity_pressure * 2.0, sparsity_pressure * 2.0,
            len(self.width_choices), device=width_logits.device
        )
        width_logits = width_logits + width_bias.unsqueeze(0).unsqueeze(0)
        if training:
            width_probs = F.gumbel_softmax(width_logits, tau=1.0, hard=False, dim=-1)
        else:
            width_probs = F.softmax(width_logits, dim=-1)
        width_idx = width_probs.argmax(dim=-1)  # [B, T]

        # ===== EXPERT ALLOCATION =====
        expert_logits = self.expert_head(features)  # [B, T, num_experts]
        if training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(expert_logits) + 1e-10) + 1e-10)
            noisy_logits = expert_logits + gumbel_noise * 0.1
            expert_probs = F.softmax(noisy_logits, dim=-1)
        else:
            expert_probs = F.softmax(expert_logits, dim=-1)
        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(expert_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)

        # ===== ACTUAL COMPUTE FRACTION (for budget adherence) =====
        # Per-token compute fraction = avg_pathway_usage * avg_depth * avg_width_fraction * expert_fraction
        # pathway_usage: top-k / num_paths (fraction of pathways active)
        pathway_top_k = getattr(self, '_pathway_top_k', min(2, self.num_paths))
        pathway_fraction = torch.full((B, T, 1), pathway_top_k / self.num_paths, device=x.device)
        depth_fraction = depth_mask.float().mean(dim=-1, keepdim=True)  # [B, T, 1]
        width_values = torch.tensor(self.width_choices, device=x.device, dtype=torch.float32)
        width_fraction = (width_probs * width_values.unsqueeze(0).unsqueeze(0)).sum(dim=-1, keepdim=True) / max(self.width_choices)
        width_fraction = width_fraction.clamp(0.0, 1.0)
        expert_fraction = torch.full((B, T, 1), self.top_k / self.num_experts, device=x.device)
        actual_compute = (pathway_fraction * depth_fraction * width_fraction * expert_fraction).clamp(0.0, 1.0)

        return ComputeAllocation(
            path_probs=path_probs,
            depth_mask=depth_mask,
            width_idx=width_idx,
            width_probs=width_probs,
            expert_indices=top_k_indices,
            expert_weights=top_k_weights,
            difficulty=difficulty,
            actual_compute=actual_compute,
        )


def compute_stability_losses(
    allocation: ComputeAllocation,
    target_budget: float,
    prev_allocation: Optional[ComputeAllocation] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute auxiliary losses for routing stability and budget adherence.

    Args:
        allocation: current ComputeAllocation.
        target_budget: target compute fraction in [0, 1].
        prev_allocation: previous step's allocation (for temporal smoothness).
        weights: optional dict of loss weights. Defaults:
            budget=1.0, entropy=0.01, smooth=0.1, balance=0.01.

    Returns:
        Dict of scalar tensors: 'budget', 'entropy', 'smooth', 'balance', 'total'.
    """
    if weights is None:
        weights = {'budget': 1.0, 'entropy': 0.01, 'smooth': 0.1, 'balance': 0.01}

    losses: Dict[str, torch.Tensor] = {}

    # Budget adherence: |actual - target|
    losses['budget'] = weights['budget'] * (allocation.actual_compute - target_budget).abs().mean()

    # Entropy: prevent collapse (encourage diversity) but don't maximize blindly.
    # We target an entropy of ~log(num_choices) * 0.5 (half of max entropy).
    path_entropy = -(allocation.path_probs * (allocation.path_probs + 1e-12).log()).sum(dim=-1).mean()
    target_path_entropy = math.log(allocation.path_probs.shape[-1]) * 0.5
    losses['entropy'] = weights['entropy'] * ((path_entropy - target_path_entropy) ** 2)

    # Temporal smoothness: routing shouldn't oscillate wildly across steps.
    if prev_allocation is not None:
        path_diff = (allocation.path_probs - prev_allocation.path_probs).abs().mean()
        depth_diff = (allocation.depth_mask.float() - prev_allocation.depth_mask.float()).abs().mean()
        width_diff = (allocation.width_probs - prev_allocation.width_probs).abs().mean()
        losses['smooth'] = weights['smooth'] * (path_diff + depth_diff + width_diff) / 3.0
    else:
        losses['smooth'] = torch.tensor(0.0, device=allocation.path_probs.device)

    # Expert load balance: use the Switch Transformer formula.
    # f_e = fraction of (token, slot) pairs dispatched to expert e
    # p_e = mean router prob for expert e
    if allocation.expert_indices.numel() > 0 and allocation.expert_indices.dim() == 3:
        flat_idx = allocation.expert_indices.reshape(-1, allocation.expert_indices.shape[-1])  # [N, K]
        flat_w = allocation.expert_weights.reshape(-1, allocation.expert_weights.shape[-1])  # [N, K]
        # Infer num_experts from the max index + 1
        num_experts = int(flat_idx.max().item()) + 1
        # f_e: fraction of (token, slot) pairs dispatched to expert e
        one_hot = F.one_hot(flat_idx, num_classes=num_experts).float()  # [N, K, E]
        f = one_hot.sum(dim=[0, 1]) / (flat_idx.numel() + 1e-12)  # [E]
        # p_e: sum of weights per expert (proxy for mean router prob)
        p = torch.zeros(num_experts, device=flat_idx.device, dtype=flat_w.dtype)
        for k in range(flat_w.shape[-1]):
            p.scatter_add_(0, flat_idx[:, k], flat_w[:, k])
        p = p / (flat_w.shape[0] * flat_w.shape[1] + 1e-12)
        losses['balance'] = weights['balance'] * num_experts * (f * p).sum()
    else:
        losses['balance'] = torch.tensor(0.0, device=allocation.path_probs.device)

    # Total
    losses['total'] = sum(v for v in losses.values() if isinstance(v, torch.Tensor) and v.requires_grad)
    return losses
