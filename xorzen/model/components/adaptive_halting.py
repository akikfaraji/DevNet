"""
Adaptive halting for Xorzen.

Allows per-token early exit from the depth loop. After each HASS block,
the model estimates a per-token "confidence" (probability that the token
is sufficiently processed). Tokens with confidence > halting_threshold
exit the depth loop early; the rest continue to the next layer.

This is genuine adaptive inference: easy tokens consume fewer layers,
hard tokens consume more. The total compute scales with the average
difficulty of the batch, not with the worst-case difficulty.

Mathematical formulation:

    halt_logit_t = W_h · h_t         (scalar per token)
    halt_prob_t  = sigmoid(halt_logit_t)

    halt_decision_t = (halt_prob_t > halting_threshold)

    # During training (STE):
    halt_mask_t = halt_decision_t.float().detach() + (halt_prob_t - halt_prob_t.detach())

    # Token exits at layer L_t = min { l : halt_prob_{t,l} > threshold }
    # or max_depth if never halted.

Safeguards:
    - min_depth: tokens cannot exit before layer min_depth.
    - max_depth: tokens always exit at layer max_depth.
    - The halt probability is clamped to [0, 1] via sigmoid.
    - At training, the STE ensures differentiability.

Ponder cost (optional auxiliary loss):
    L_ponder = mean(1 - halt_prob)  # penalize NOT halting (encourage early exit)
    # Note: this is the OPPOSITE sign of the IGRIS bug. We penalize
    # continued thinking, not halting.

This module is OPTIONAL — it is enabled by ``config.adaptive_halting=True``.
When disabled, all tokens go through all layers (standard transformer).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AdaptiveHalting"]


class AdaptiveHalting(nn.Module):
    """
    Per-token adaptive halting estimator.

    Args:
        hidden_dim: H
        max_depth: maximum number of layers (tokens halt at or before this).
        halting_threshold: confidence threshold in (0, 1). Higher = harder
            to halt (more layers used). Lower = easier to halt (fewer layers).
        min_depth: tokens cannot halt before this layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_depth: int,
        halting_threshold: float = 0.9,
        min_depth: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depth = max_depth
        self.halting_threshold = float(halting_threshold)
        self.min_depth = max(0, int(min_depth))

        # Per-layer halt estimator (one head per layer so the model can
        # learn layer-specific halting behavior).
        self.halt_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            for _ in range(max_depth)
        ])

        self._init_weights()

    def _init_weights(self):
        for head in self.halt_heads:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)  # small init → start near 0.5
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, H] hidden states after layer layer_idx.
            layer_idx: current layer index (0-based).
            training: if True, use STE.

        Returns:
            halt_probs: [B, T, 1] — probability that each token should halt.
            halt_decisions: [B, T, 1] bool — True if token should halt.
                For layers < min_depth, always False.
                For layer >= max_depth - 1, always True (forced halt at last layer).
        """
        B, T, H = x.shape

        # If we're below min_depth, never halt.
        if layer_idx < self.min_depth:
            halt_probs = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)
            halt_decisions = torch.zeros(B, T, 1, dtype=torch.bool, device=x.device)
            return halt_probs, halt_decisions

        # If we're at the last layer, always halt.
        if layer_idx >= self.max_depth - 1:
            halt_probs = torch.ones(B, T, 1, device=x.device, dtype=x.dtype)
            halt_decisions = torch.ones(B, T, 1, dtype=torch.bool, device=x.device)
            return halt_probs, halt_decisions

        # Normal case: compute halt probability.
        halt_logits = self.halt_heads[layer_idx](x)  # [B, T, 1]
        halt_probs = torch.sigmoid(halt_logits)

        if training:
            # STE: hard decision in forward, soft prob in backward
            halt_hard = (halt_probs > self.halting_threshold).float()
            halt_mask = halt_hard.detach() + (halt_probs - halt_probs.detach())
            halt_decisions = halt_mask > 0.5
        else:
            halt_decisions = halt_probs > self.halting_threshold

        return halt_probs, halt_decisions

    def ponder_loss(self, halt_probs: torch.Tensor) -> torch.Tensor:
        """
        Ponder cost: penalize NOT halting (encourage early exit).

        L_ponder = mean(1 - halt_prob)

        This is the CORRECT sign (opposite of the IGRIS bug). High halt_prob
        = early exit = low ponder cost. We want to MINIMIZE this loss, which
        encourages the model to halt early when possible.

        Args:
            halt_probs: [B, T, 1] from forward().

        Returns:
            scalar tensor.
        """
        return (1.0 - halt_probs).mean()
