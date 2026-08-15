"""
Production-grade Adaptive Router for xorzen-zero.
Determines efficiency gains - the MOST critical component.
MUST converge stably, MUST make smart decisions, MUST train efficiently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import numpy as np
from dataclasses import dataclass, field
import warnings

from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import TensorStability, InformationTheory, RoutingMathematics

logger = get_logger()


# ==================== AUXILIARY LOSS FUNCTIONS ====================

def load_balance_loss(router_probs: torch.Tensor, expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Load balancing auxiliary loss from Switch Transformer.
    Prevents router from collapsing (sending all tokens to same expert).
    """
    B, S, E = router_probs.shape
    dispatch = torch.zeros(B, S, E, device=router_probs.device)
    dispatch.scatter_(-1, expert_indices, 1.0)
    f = dispatch.mean(dim=[0, 1])  # fraction per expert
    p = router_probs.mean(dim=[0, 1])  # mean probability per expert
    return num_experts * (f * p).sum()

def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Router z-loss from ST-MoE.
    Prevents logits from growing unbounded.
    """
    log_z = torch.logsumexp(router_logits, dim=-1)
    return (log_z ** 2).mean()


def path_diversity_loss(path_probs: torch.Tensor) -> torch.Tensor:
    """
    Entropy-based diversity loss for path routing.
    Encourages the router to spread probability across ALL pathways,
    preventing SSM (or any single path) from monopolising routing.
    Loss = -mean_entropy → minimised by maximising entropy.
    """
    # path_probs: [B, T, num_paths]
    # Entropy per token
    entropy = -torch.sum(path_probs * torch.log(path_probs + 1e-12), dim=-1)  # [B, T]
    # Negative entropy as loss (we want HIGH entropy → uniform paths)
    return -entropy.mean()


def width_diversity_loss(width_probs: torch.Tensor) -> torch.Tensor:
    """
    Entropy-based diversity loss for width routing.
    Same shape as path_diversity_loss but applied to the width router.
    Without this, the width router collapses onto the largest width
    (Phase 6 finding). Encourages the router to use the full range
    of available widths (small for easy tokens, large for hard tokens).

    Args:
        width_probs: [B, T, num_widths] softmax probabilities.

    Returns:
        Scalar loss = -mean_entropy(width_probs).
    """
    entropy = -torch.sum(width_probs * torch.log(width_probs + 1e-12), dim=-1)  # [B, T]
    return -entropy.mean()


# ==================== DATA STRUCTURES ====================

@dataclass
class RoutingDecision:
    """Complete routing decision for a token."""
    # Depth routing (which layers to use)
    depth_logits: torch.Tensor  # [batch, seq, max_depth]
    depth_probs: torch.Tensor   # [batch, seq, max_depth]
    depth_mask: torch.Tensor    # [batch, seq, max_depth] binary
    
    # Width routing (how much compute per layer)
    width_logits: torch.Tensor  # [batch, seq, num_widths]
    width_probs: torch.Tensor   # [batch, seq, num_widths]
    width_idx: torch.Tensor     # [batch, seq] index
    width_multiplier: torch.Tensor  # [batch, seq, 1] scaled
    
    # Path routing (which HASS pathways to use)
    path_logits: torch.Tensor   # [batch, seq, num_paths]
    path_probs: torch.Tensor    # [batch, seq, num_paths]
    
    # Expert routing (which MoE experts to use)
    expert_logits: torch.Tensor  # [batch, seq, num_experts]
    expert_probs: torch.Tensor   # [batch, seq, num_experts]
    expert_indices: torch.Tensor  # [batch, seq, top_k] indices
    expert_weights: torch.Tensor  # [batch, seq, top_k] weights
    
    # Token complexity
    complexity: torch.Tensor  # [batch, seq, 1]
    
    # Uncertainty
    uncertainty: torch.Tensor  # [batch, seq, 1]
    
    # Auxiliary outputs
    auxiliary: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'depth_probs': self.depth_probs.detach().cpu(),
            'width_idx': self.width_idx.detach().cpu(),
            'path_probs': self.path_probs.detach().cpu(),
            'expert_indices': self.expert_indices.detach().cpu(),
            'expert_weights': self.expert_weights.detach().cpu(),
            'complexity': self.complexity.detach().cpu(),
            'uncertainty': self.uncertainty.detach().cpu(),
        }
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute routing statistics."""
        stats = {}
        
        # Depth statistics
        depth_active = self.depth_mask.float().mean().item()
        stats['depth_active_mean'] = depth_active
        stats['depth_active_std'] = self.depth_mask.float().std().item()
        
        # Average active layers per token
        active_layers = self.depth_mask.sum(dim=-1).float().mean().item()
        stats['active_layers_per_token'] = active_layers
        
        # Width statistics
        stats['width_mean'] = self.width_multiplier.mean().item()
        stats['width_std'] = self.width_multiplier.std().item()
        
        # Path statistics
        path_entropy = -torch.sum(self.path_probs * torch.log(self.path_probs + 1e-12), dim=-1).mean().item()
        stats['path_entropy'] = path_entropy
        
        # Expert statistics
        expert_utilization = (self.expert_weights > 0.01).float().sum(dim=-1).mean().item()
        stats['expert_utilization'] = expert_utilization
        
        # Complexity statistics
        stats['complexity_mean'] = self.complexity.mean().item()
        stats['complexity_std'] = self.complexity.std().item()
        
        # Uncertainty statistics
        stats['uncertainty_mean'] = self.uncertainty.mean().item()
        
        return stats
    
    def compute_efficiency(self) -> Dict[str, float]:
        """Compute routing efficiency metrics."""
        batch, seq_len, max_depth = self.depth_mask.shape
        
        # Compute cost metrics
        total_possible_compute = batch * seq_len * max_depth
        actual_compute = self.depth_mask.sum().item()
        
        compute_efficiency = 1.0 - (actual_compute / total_possible_compute)
        
        # Width-adjusted compute
        width_adjusted_compute = (self.depth_mask.float() * self.width_multiplier).sum().item()
        max_width_compute = total_possible_compute * self.width_multiplier.max().item()
        
        width_efficiency = 1.0 - (width_adjusted_compute / max_width_compute)
        
        # Overall efficiency
        overall_efficiency = (compute_efficiency + width_efficiency) / 2
        
        return {
            'compute_efficiency': compute_efficiency,
            'width_efficiency': width_efficiency,
            'overall_efficiency': overall_efficiency,
            'active_ratio': actual_compute / total_possible_compute,
            'width_adjusted_ratio': width_adjusted_compute / max_width_compute,
        }


@dataclass 
class RoutingMetrics:
    """Training metrics for router."""
    step: int = 0
    loss_total: float = 0.0
    loss_routing: float = 0.0
    loss_balancing: float = 0.0
    loss_consistency: float = 0.0
    
    # Statistics
    depth_active_mean: float = 0.0
    width_mean: float = 0.0
    complexity_mean: float = 0.0
    uncertainty_mean: float = 0.0
    
    # Efficiency
    compute_efficiency: float = 0.0
    overall_efficiency: float = 0.0
    
    # Gradients
    grad_norm: float = 0.0
    grad_mean: float = 0.0
    
    def update(self, step: int, loss_dict: Dict[str, torch.Tensor], 
               decision: RoutingDecision, grad_stats: Dict[str, float]):
        """Update metrics."""
        self.step = step
        
        # Losses
        self.loss_total = loss_dict.get('total', 0.0)
        self.loss_routing = loss_dict.get('routing', 0.0)
        self.loss_balancing = loss_dict.get('balancing', 0.0)
        self.loss_consistency = loss_dict.get('consistency', 0.0)
        
        # Statistics from decision
        stats = decision.compute_statistics()
        self.depth_active_mean = stats.get('depth_active_mean', 0.0)
        self.width_mean = stats.get('width_mean', 0.0)
        self.complexity_mean = stats.get('complexity_mean', 0.0)
        self.uncertainty_mean = stats.get('uncertainty_mean', 0.0)
        
        # Efficiency
        efficiency = decision.compute_efficiency()
        self.compute_efficiency = efficiency.get('compute_efficiency', 0.0)
        self.overall_efficiency = efficiency.get('overall_efficiency', 0.0)
        
        # Gradients
        self.grad_norm = grad_stats.get('grad_norm', 0.0)
        self.grad_mean = grad_stats.get('grad_mean', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step': self.step,
            'loss_total': self.loss_total,
            'loss_routing': self.loss_routing,
            'loss_balancing': self.loss_balancing,
            'loss_consistency': self.loss_consistency,
            'depth_active_mean': self.depth_active_mean,
            'width_mean': self.width_mean,
            'complexity_mean': self.complexity_mean,
            'uncertainty_mean': self.uncertainty_mean,
            'compute_efficiency': self.compute_efficiency,
            'overall_efficiency': self.overall_efficiency,
            'grad_norm': self.grad_norm,
            'grad_mean': self.grad_mean,
        }


# ==================== ROUTER CORE ====================

class AdaptiveRouter(nn.Module):
    """
    Core router for xorzen-zero.
    Takes token embeddings + CoT features -> routing decisions.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.hidden_dim = config.hidden_size
        self.cot_dim = config.cot_dim * config.cot_components
        self.input_dim = self.hidden_dim + self.cot_dim
        
        # Output dimensions
        self.max_depth = config.max_depth
        self.num_widths = len(config.width_choices)
        self.num_paths = 3  # Local, Low-rank, SSM
        self.num_experts = config.expert_count
        self.top_k = config.top_k_experts
        
        # Network architecture
        self._build_network()
        
        # Temperature for gumbel softmax
        self.temperature = config.router_temperature
        self.temperature_annealing = config.router_temperature_annealing
        
        # Load balancing
        self.load_balancing_weight = config.load_balancing_weight
        
        # Training state
        self.training_step = 0
        
        # Auxiliary loss weights — kept tiny; real load-balance comes from
        # model.py _compute_load_balance_loss via config.load_balancing_weight.
        # These router-internal losses are only for logit stability (z-loss)
        # and soft load signal — not the primary training objective.
        self.lb_loss_weight   = 0.0001   # was 0.01 — double-counted with model.py
        self.z_loss_weight    = 0.0001   # was 0.001 — z-loss just for logit stability
        # Path diversity loss weight — prevents SSM (or any single path) from
        # collapsing to 100% usage.
        # Phase 6 experiment found:
        #   path_div_weight=0.002  -> severe collapse (1 of 3 pathways active)
        #   path_div_weight=0.02   -> still collapses (1 of 3 pathways active)
        #   path_div_weight=0.2    -> partial diversity (2 of 3 pathways active)
        # v0.4: bumped default from 0.1 to 0.2 (config.path_div_weight).
        self.path_div_weight  = float(getattr(config, 'path_div_weight', 0.2))
        # Width diversity loss — NEW in v0.4. Prevents the width router from
        # collapsing onto the largest width. Only active when num_widths >= 2.
        self.width_div_weight = float(getattr(config, 'width_div_weight', 0.1))
        self.metrics_history: List[RoutingMetrics] = []
        
        # Expert usage tracking
        self.expert_usage = torch.zeros(self.num_experts, dtype=torch.float)
        self.expert_load = torch.zeros(self.num_experts, dtype=torch.float)
        
        # Initialize weights
        self._init_weights()
        
        logger.info("router", 
                   f"Initialized AdaptiveRouter: depth={self.max_depth}, "
                   f"widths={self.num_widths}, experts={self.num_experts}, "
                   f"top_k={self.top_k}")
    
    def _build_network(self):
        """Build router network architecture."""
        # Scale router dims proportionally to model hidden size.
        # Hardcoded 1024/512/256 was enormous for small models (nano-1M has
        # hidden=128, so the router would dwarf the model itself).
        _h = self.config.router_hidden_dim   # set per model-size in ConfigFactory
        _enc1 = max(128, _h * 4)
        _enc2 = max(64,  _h * 2)
        _enc3 = max(32,  _h)
        _head = max(32,  _h // 2)

        # Feature encoder (shared backbone)
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, _enc1),
            nn.LayerNorm(_enc1),
            nn.GELU(),
            nn.Dropout(self.config.router_dropout),
            nn.Linear(_enc1, _enc2),
            nn.LayerNorm(_enc2),
            nn.GELU(),
            nn.Dropout(self.config.router_dropout),
            nn.Linear(_enc2, _enc3),
            nn.LayerNorm(_enc3),
            nn.GELU(),
        )

        # Depth router
        self.depth_router = nn.Sequential(
            nn.Linear(_enc3, _head),
            nn.LayerNorm(_head),
            nn.GELU(),
            nn.Linear(_head, self.max_depth),
        )

        # Width router
        self.width_router = nn.Sequential(
            nn.Linear(_enc3, _head),
            nn.LayerNorm(_head),
            nn.GELU(),
            nn.Linear(_head, self.num_widths),
        )

        # Path router (HASS pathways)
        self.path_router = nn.Sequential(
            nn.Linear(_enc3, _head),
            nn.LayerNorm(_head),
            nn.GELU(),
            nn.Linear(_head, self.num_paths),
        )

        # Expert router (MoE) — slightly larger head for the harder routing task
        self.expert_router = nn.Sequential(
            nn.Linear(_enc3, _enc3),
            nn.LayerNorm(_enc3),
            nn.GELU(),
            nn.Linear(_enc3, self.num_experts),
        )

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(_enc3, _head),
            nn.LayerNorm(_head),
            nn.GELU(),
            nn.Linear(_head, 1),
            nn.Sigmoid(),
        )

        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(_enc3, _head),
            nn.LayerNorm(_head),
            nn.GELU(),
            nn.Linear(_head, 1),
            nn.Sigmoid(),
        )

        # Store for use in init_weights / elsewhere
        self._enc3 = _enc3
        
        # Width value network (maps index to multiplier)
        width_tensor = torch.tensor(self.config.width_choices, dtype=torch.float32)
        self.register_buffer('width_values', width_tensor)
        self.width_normalizer = self.config.hidden_size  # Normalize to [0, 1]
    
    def _init_weights(self):
        """Initialize router weights for stable training."""
        # Feature encoder
        for layer in self.feature_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        
        # Routers
        for router in [self.depth_router, self.width_router, 
                      self.path_router, self.expert_router]:
            for layer in router:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
        
        # Estimators
        for estimator in [self.complexity_estimator, self.uncertainty_estimator]:
            for layer in estimator:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,  # Token embeddings [batch, seq, hidden]
        cot_features: torch.Tensor,  # CoT features [batch, seq, cot_dim]
        training: Optional[bool] = None,
        deterministic: bool = False,
        expert_capacity: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Main forward pass.
        
        Args:
            x: Token embeddings
            cot_features: CoT features
            training: Whether in training mode
            deterministic: Whether to use deterministic routing
            expert_capacity: Capacity for expert routing
            
        Returns:
            RoutingDecision with all routing information
        """
        batch_size, seq_len, _ = x.shape
        
        # Use PyTorch's self.training flag as source of truth for determinism
        if training is None:
            training = self.training
        # In eval mode, always use deterministic routing for reproducibility
        if not training:
            deterministic = True
        
        # Concatenate inputs
        router_input = torch.cat([x, cot_features], dim=-1)  # [batch, seq, input_dim]
        
        # Encode features — fuse B and T dims so Linear ops run as a single
        # [B*T, D] matmul instead of looping; reshape back after.
        B, T, _ = router_input.shape
        features = self.feature_encoder(router_input.view(B * T, -1)).view(B, T, -1)
        
        # Get all logits
        depth_logits = self.depth_router(features)  # [batch, seq, max_depth]
        width_logits = self.width_router(features)  # [batch, seq, num_widths]
        path_logits = self.path_router(features)  # [batch, seq, num_paths]
        expert_logits = self.expert_router(features)  # [batch, seq, num_experts]
        
        # Estimate complexity and uncertainty
        complexity = self.complexity_estimator(features)  # [batch, seq, 1]
        uncertainty = self.uncertainty_estimator(features)  # [batch, seq, 1]
        
        # Apply temperature
        current_temp = self.temperature
        if self.temperature_annealing and training:
            # Anneal temperature over training
            current_temp = max(0.1, self.temperature * (0.99 ** (self.training_step / 1000)))

        # ===== v0.4 COST-AWARE ROUTING MODULATION =====
        # The router estimates the actual per-axis compute cost (in arbitrary
        # FLOPs-equivalent units) and biases routing decisions toward the
        # global compute_budget. This is the "cost-aware ComputeController"
        # idea — but integrated into AdaptiveRouter (no separate dead module).
        #
        # Cost model (in relative FLOPs units, per token):
        #   depth_cost_per_layer ≈ H * (3*H + 2*H_ffn + ...)  ≈ 4*H^2 per layer
        #   width_cost: linear in selected width (~H * W)
        #   path_cost:  local ≈ 2*H*window; low_rank ≈ 2*H*r; ssm ≈ 2*H*N
        #   expert_cost: top_k * (H * expert_hidden)
        #
        # We use a SIMPLE cost model: each axis contributes a "fraction of
        # total compute" and the budget multiplies the logits so that low
        # budget → sharper / sparser distributions.
        cost_aware = bool(getattr(self.config, 'cost_aware_routing', True))
        if cost_aware:
            budget = float(getattr(self.config, 'compute_budget', 1.0))
            budget = max(0.05, min(1.0, budget))
            # sparsity_pressure in [0, 1]: 0 = full compute, 1 = minimal
            sparsity_pressure = 1.0 - budget
            # Per-axis cost weights (relative): depth dominates, then MoE,
            # then width, then pathway (cheapest).
            # These are not learned — they are a fixed prior based on
            # typical architecture FLOPs breakdown.
            # depth_cost_weight = 4.0 (per-layer cost is high)
            # expert_cost_weight = 2.0 (top-k experts)
            # width_cost_weight = 1.0 (linear in width)
            # path_cost_weight = 0.5 (cheapest axis)
            depth_shift   = -sparsity_pressure * 4.0 * (1.0 - complexity.squeeze(-1))  # [B, T]
            # Width: lower budget → bias toward SMALLER widths.
            # width_choices are sorted ascending (smallest first).
            # Use a linspace that boosts early (small) indices under low budget.
            width_bias_axis = torch.linspace(
                sparsity_pressure * 2.0, -sparsity_pressure * 2.0,
                self.num_widths, device=width_logits.device,
            )  # [num_widths]
            width_logits = width_logits + width_bias_axis.view(1, 1, -1)
            # Path: lower budget → bias toward SSM (cheapest of the three for
            # long sequences, and it has the strongest compression). The
            # ordering is [local, low_rank, ssm] by pathway index.
            path_bias_axis = torch.linspace(
                sparsity_pressure * 1.5, -sparsity_pressure * 0.5,
                self.num_paths, device=path_logits.device,
            )  # [num_paths]
            path_logits = path_logits + path_bias_axis.view(1, 1, -1)
            # Depth: apply the depth_shift (per-token, per-layer).
            # Earlier layers are kept (min_depth enforces this); later layers
            # are pruned under low budget for easy tokens.
            depth_layer_bias = torch.linspace(
                0.0, -sparsity_pressure * 3.0,
                self.max_depth, device=depth_logits.device,
            )  # [max_depth]
            depth_logits = depth_logits + depth_layer_bias.view(1, 1, -1) + depth_shift.unsqueeze(-1)
            # Expert: lower budget → sharper top-k (already normalized);
            # we add a small entropy-encouraging bias so the model doesn't
            # collapse onto one expert. Slight bias toward uniform.
            # (Switch-formula load_balance_loss already handles this.)
        else:
            budget = 1.0
        
        # Depth routing
        depth_probs, depth_mask = self._route_depth(
            depth_logits, complexity, current_temp, training, deterministic
        )
        
        # Width routing
        width_probs, width_idx, width_multiplier = self._route_width(
            width_logits, complexity, current_temp, training, deterministic
        )
        
        # Path routing
        path_probs = self._route_path(
            path_logits, current_temp, training, deterministic
        )
        
        # Expert routing (most complex)
        expert_probs, expert_indices, expert_weights = self._route_experts(
            expert_logits, current_temp, training, deterministic, expert_capacity
        )
        
        # Update expert usage statistics
        if training:
            self._update_expert_usage(expert_indices, expert_weights)
        
        # Create routing decision
        decision = RoutingDecision(
            depth_logits=depth_logits,
            depth_probs=depth_probs,
            depth_mask=depth_mask,
            width_logits=width_logits,
            width_probs=width_probs,
            width_idx=width_idx,
            width_multiplier=width_multiplier,
            path_logits=path_logits,
            path_probs=path_probs,
            expert_logits=expert_logits,
            expert_probs=expert_probs,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            complexity=complexity,
            uncertainty=uncertainty,
            auxiliary={
                'features': features,
                'temperature': torch.tensor(current_temp),
            }
        )
        
        
        # Compute auxiliary losses during training
        if training and not deterministic:
            lb_loss = load_balance_loss(expert_probs, expert_indices, self.num_experts)
            z_loss_val = router_z_loss(expert_logits)
            # Path diversity: penalise routing collapse onto a single pathway
            path_div = path_diversity_loss(path_probs)
            decision.auxiliary.update({
                'load_balance_loss': self.lb_loss_weight * lb_loss,
                'router_z_loss':     self.z_loss_weight * z_loss_val,
                'path_div_loss':     self.path_div_weight * path_div,
            })
            # Width diversity: only meaningful when num_widths >= 2.
            # At NANO_1M (1 width) this would be a no-op anyway, but we
            # skip it entirely to avoid logging noise.
            if self.num_widths >= 2 and self.width_div_weight > 0:
                width_div = width_diversity_loss(width_probs)
                decision.auxiliary['width_div_loss'] = self.width_div_weight * width_div
        
        return decision
    
    def _route_depth(
        self,
        logits: torch.Tensor,
        complexity: torch.Tensor,
        temperature: float,
        training: bool,
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route depth (which layers to use).
        
        Complexity influences depth: complex tokens use more layers.
        """
        batch_size, seq_len, max_depth = logits.shape
        
        # Adjust logits with complexity
        # Complex tokens get boosted for deeper layers
        depth_bias = torch.linspace(0, 1, max_depth, device=logits.device).unsqueeze(0).unsqueeze(0)
        adjusted_logits = logits + complexity * depth_bias * 2.0
        
        # Apply temperature
        scaled_logits = adjusted_logits / max(temperature, 1e-8)
        
        if deterministic:
            # Hard routing for inference
            probs = torch.sigmoid(scaled_logits)
            mask = (probs > 0.5).float()
        else:
            # Gumbel-sigmoid for training (differentiable binary)
            if training:
                # Gumbel noise
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(scaled_logits) + 1e-10) + 1e-10)
                noisy_logits = (scaled_logits + gumbel_noise) / temperature
                probs = torch.sigmoid(noisy_logits)
                
                # Straight-through estimator — keep probs in graph, no detach()
                # mask_hard gives the hard 0/1, but the gradient flows through probs
                mask_hard = (probs > 0.5).float()
                mask = mask_hard - probs.detach() + probs  # STE: forward=hard, backward=soft
                # NOTE: probs.detach() only detaches the copy used for the offset;
                # the LAST +probs keeps the gradient path alive through probs itself.
            else:
                # Soft for evaluation
                probs = torch.sigmoid(scaled_logits)
                mask = (probs > 0.5).float()
        
        # Ensure minimum depth (differentiable: use additive mask, no in-place ops)
        min_depth = self.config.min_depth
        if min_depth > 0:
            forced_mask = torch.zeros_like(mask)
            forced_mask[..., :min_depth] = 1.0
            # Override first min_depth layers without breaking gradient graph
            mask = torch.where(
                forced_mask.bool(),
                torch.ones_like(mask),
                mask
            )
        
        return probs, mask
    
    def _route_width(
        self,
        logits: torch.Tensor,
        complexity: torch.Tensor,
        temperature: float,
        training: bool,
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route width (how much compute per layer).
        
        Complexity influences width: complex tokens get more compute.
        """
        # Adjust logits with complexity
        # Complex tokens pushed toward higher widths
        width_bias = torch.linspace(-1, 1, self.num_widths, device=logits.device).unsqueeze(0).unsqueeze(0)
        adjusted_logits = logits + complexity * width_bias * 3.0
        
        # Apply temperature
        scaled_logits = adjusted_logits / max(temperature, 1e-8)
        
        if deterministic:
            # Hard selection for inference
            probs = F.softmax(scaled_logits, dim=-1)
            width_idx = torch.argmax(probs, dim=-1)
        else:
            # Gumbel-softmax for training
            if training:
                probs = F.gumbel_softmax(scaled_logits, tau=temperature, hard=False, dim=-1)
                width_idx = torch.argmax(probs, dim=-1)
            else:
                probs = F.softmax(scaled_logits, dim=-1)
                width_idx = torch.argmax(probs, dim=-1)
        
        # Convert soft probs to a continuous width multiplier (gradient flows through probs)
        # width_values: [num_widths] (buffer), probs: [B, T, num_widths]
        # Soft expectation: sum(prob_i * width_i) — fully differentiable
        width_multiplier = (probs * self.width_values.unsqueeze(0).unsqueeze(0)) \
            .sum(dim=-1) / self.width_normalizer  # [B, T]
        
        # width_idx for adapter selection (hard, inference-only path; no grad needed)
        with torch.no_grad():
            width_idx_hard = torch.argmax(probs.detach(), dim=-1)
        # Use hard index for discrete adapter lookup, soft multiplier for gradient
        
        return probs, width_idx_hard, width_multiplier.unsqueeze(-1)
    
    def _route_path(
        self,
        logits: torch.Tensor,
        temperature: float,
        training: bool,
        deterministic: bool
    ) -> torch.Tensor:
        """
        Route HASS pathways (Local, Low-rank, SSM).

        Fix v2:
        - Uniform prior applied in PROBABILITY space (not logit space).
          Blending in logit space was a bug: adding uniform_prob ≈ 0.33 to
          logits of magnitude ~1-3 barely shifts the softmax output.
        - Prior weight decays from 0.5 → 0.05 over training steps so the
          router is strongly constrained early but can specialise later.
        - Exploration temperature raised to 2× base temperature early in
          training to prevent premature path collapse.
        """
        num_paths = logits.shape[-1]

        scaled_logits = logits / max(temperature, 1e-8)

        if deterministic:
            probs = F.softmax(scaled_logits, dim=-1)
        else:
            if training:
                # Higher exploration temperature early in training
                explore_tau = max(temperature * 2.0, 1.0)
                raw_probs = F.gumbel_softmax(scaled_logits, tau=explore_tau,
                                             hard=False, dim=-1)
            else:
                raw_probs = F.softmax(scaled_logits, dim=-1)

            if training:
                # Decay prior weight: starts at 0.5, decays toward 0.05
                # Use training_step if available, else stay at 0.3
                step = getattr(self, 'training_step', 0)
                prior_weight = max(0.05, 0.5 * (0.995 ** step))
                uniform = torch.full_like(raw_probs, 1.0 / num_paths)
                # Blend in probability space — correct and differentiable
                probs = (1.0 - prior_weight) * raw_probs + prior_weight * uniform
            else:
                probs = raw_probs

        return probs
    
    def _route_experts(
        self,
        logits: torch.Tensor,
        temperature: float,
        training: bool,
        deterministic: bool,
        expert_capacity: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route to MoE experts.
        
        Implements capacity-aware routing to prevent overload.
        """
        batch_size, seq_len, num_experts = logits.shape
        num_tokens = batch_size * seq_len
        
        # Default capacity: tokens per expert
        if expert_capacity is None:
            expert_capacity = max(1, int(num_tokens * 1.25 / num_experts))
        
        # Reshape for routing
        logits_flat = logits.view(-1, num_experts)  # [num_tokens, num_experts]
        
        if deterministic:
            # Top-k routing for inference
            top_k_weights, top_k_indices = torch.topk(
                F.softmax(logits_flat, dim=-1), 
                self.top_k, 
                dim=-1
            )
            
            # Normalize weights
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)
            
        else:
            # Gumbel-softmax top-k for training
            if training:
                # Add gumbel noise
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits_flat) + 1e-10) + 1e-10)
                noisy_logits = (logits_flat + gumbel_noise) / max(temperature, 1e-8)
                
                # Top-k with softmax
                top_k_weights = F.softmax(noisy_logits, dim=-1)
                top_k_weights, top_k_indices = torch.topk(top_k_weights, self.top_k, dim=-1)
                
                # Normalize
                top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)
                
                # Apply capacity constraint
                if expert_capacity > 0:
                    top_k_weights = self._apply_capacity_constraint(
                        top_k_weights, top_k_indices, expert_capacity
                    )
            else:
                # Soft top-k for evaluation
                probs = F.softmax(logits_flat / temperature, dim=-1)
                top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
                top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-12)
        
        # Reshape back
        expert_probs = F.softmax(logits / temperature, dim=-1)
        expert_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
        expert_weights = top_k_weights.view(batch_size, seq_len, self.top_k)
        
        return expert_probs, expert_indices, expert_weights
    
    def _apply_capacity_constraint(
        self,
        weights: torch.Tensor,
        indices: torch.Tensor,
        capacity: int
    ) -> torch.Tensor:
        """
        Apply capacity constraint to expert routing.
        Uses pure tensor ops (no in-place index assignment) to preserve autograd.
        Vectorised implementation — no Python token loop.
        """
        num_tokens, top_k = weights.shape

        with torch.no_grad():
            # Build a boolean keep-mask entirely with tensor ops.
            # Process slots in descending weight order so high-confidence
            # assignments win, but use cumulative counting per expert.
            device = weights.device

            # Flatten to [N*top_k] and sort by descending weight
            flat_w   = weights.detach().flatten()            # [N*top_k]
            order    = flat_w.argsort(descending=True)       # sorted indices
            t_idx    = order // top_k                        # token index
            e_idx    = indices.detach().flatten()[order]     # expert index

            # For each expert, keep only the first `capacity` arrivals
            # Use cumulative count per expert via scatter
            keep_flat = torch.zeros(num_tokens * top_k, dtype=torch.bool, device=device)
            expert_count = torch.zeros(self.num_experts, dtype=torch.long, device=device)

            # Vectorised per-expert capacity enforcement.
            # For each expert, keep only the first `capacity` arrivals
            # (arrivals are already sorted by descending weight via `order`).
            # Strategy: assign a per-expert arrival index, then keep where index < capacity.
            for e in range(self.num_experts):
                # Positions (in `order`) that route to this expert
                is_e = (e_idx == e)           # [num_tokens*top_k] bool
                if not is_e.any():
                    continue
                # The positions are already in descending weight order (from argsort above).
                # Cumcount within expert: 0-based arrival index.
                positions = is_e.nonzero(as_tuple=False).view(-1)  # indices in sorted order
                allowed   = positions[:capacity]                    # keep first `capacity`
                keep_flat[order[allowed]] = True

            keep_mask = keep_flat.view(num_tokens, top_k).float()

        # Mask weights without in-place ops so gradient graph stays intact
        masked_weights = weights * keep_mask
        masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-12)
        return masked_weights
    
    def _update_expert_usage(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor):
        """Update expert usage statistics — fully vectorised, no Python loops."""
        # expert_indices: [B, T, top_k], expert_weights: [B, T, top_k]
        indices_flat = expert_indices.detach().view(-1)       # [B*T*top_k]
        weights_flat = expert_weights.detach().view(-1)       # [B*T*top_k]

        # Only count tokens with significant weight (threshold 0.01)
        keep = weights_flat > 0.01
        if not keep.any():
            return

        idx  = indices_flat[keep]
        w    = weights_flat[keep]

        # Accumulate onto CPU buffers with scatter_add (no Python loop)
        usage_delta = torch.zeros(self.num_experts, dtype=torch.float)
        load_delta  = torch.zeros(self.num_experts, dtype=torch.float)
        usage_delta.scatter_add_(0, idx.cpu(), torch.ones_like(idx, dtype=torch.float))
        load_delta.scatter_add_(0, idx.cpu(), w.cpu())

        self.expert_usage += usage_delta
        self.expert_load  += load_delta
    
    def compute_loss(
        self,
        decision: RoutingDecision,
        target_complexity: Optional[torch.Tensor] = None,
        balance_experts: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute routing losses.
        
        Args:
            decision: Routing decision
            target_complexity: Target complexity if available
            balance_experts: Whether to add expert balancing loss
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # 1. Routing loss: encourage confident decisions
        routing_loss = self._compute_routing_loss(decision)
        losses['routing'] = routing_loss
        
        # 2. Consistency loss: similar tokens should have similar routing
        consistency_loss = self._compute_consistency_loss(decision)
        losses['consistency'] = consistency_loss
        
        # 3. Complexity loss: if targets available
        if target_complexity is not None:
            complexity_loss = self._compute_complexity_loss(decision.complexity, target_complexity)
            losses['complexity'] = complexity_loss
        
        # 4. Expert balancing loss
        if balance_experts:
            balancing_loss = self._compute_balancing_loss(decision)
            losses['balancing'] = balancing_loss * self.load_balancing_weight
        
        # 5. Efficiency loss: encourage efficiency
        efficiency_loss = self._compute_efficiency_loss(decision)
        losses['efficiency'] = efficiency_loss
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_routing_loss(self, decision: RoutingDecision) -> torch.Tensor:
        """
        Loss to encourage confident routing decisions.

        IMPORTANT: path_entropy is deliberately EXCLUDED here.
        path_diversity_loss (in the forward pass) already pushes path routing
        toward uniform distribution (high entropy). Including path_entropy in
        this low-entropy loss would create a direct contradiction and cause
        SSM-collapse — exactly the bug observed at ~25% training.

        Only depth and width get the sharpness pressure; path stays diverse.
        """
        # Encourage low uncertainty
        uncertainty_loss = decision.uncertainty.mean()

        # Encourage confident depth decisions (sharp sigmoid outputs)
        depth_entropy = -torch.sum(
            decision.depth_probs * torch.log(decision.depth_probs + 1e-12),
            dim=-1
        ).mean()

        # Encourage confident width decisions
        width_entropy = -torch.sum(
            decision.width_probs * torch.log(decision.width_probs + 1e-12),
            dim=-1
        ).mean()

        # NOTE: path_entropy intentionally omitted — diversity loss handles it.

        # Expert: mild sharpness to prevent all-uniform routing
        expert_entropy = -torch.sum(
            decision.expert_probs * torch.log(decision.expert_probs + 1e-12),
            dim=-1
        ).mean()

        total_entropy = (depth_entropy + width_entropy + expert_entropy) / 3

        # Combined loss — keep tiny so LM signal dominates
        routing_loss = uncertainty_loss + total_entropy * 0.05

        return routing_loss
    
    def _compute_consistency_loss(self, decision: RoutingDecision) -> torch.Tensor:
        """Loss to encourage consistent routing for similar tokens."""
        batch_size, seq_len, _ = decision.depth_probs.shape
        
        if seq_len < 2:
            return torch.tensor(0.0, device=decision.depth_probs.device)
        
        # Compare consecutive tokens
        depth_diff = torch.mean((decision.depth_probs[:, 1:] - decision.depth_probs[:, :-1]) ** 2)
        width_diff = torch.mean((decision.width_multiplier[:, 1:] - decision.width_multiplier[:, :-1]) ** 2)
        complexity_diff = torch.mean((decision.complexity[:, 1:] - decision.complexity[:, :-1]) ** 2)
        
        consistency_loss = (depth_diff + width_diff + complexity_diff) / 3
        
        return consistency_loss
    
    def _compute_complexity_loss(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Loss for complexity prediction."""
        return F.mse_loss(predicted, target)
    
    def _compute_balancing_loss(self, decision: RoutingDecision) -> torch.Tensor:
        """Load balancing loss for experts."""
        batch_size, seq_len, num_experts = decision.expert_probs.shape
        
        # Reshape
        probs_flat = decision.expert_probs.view(-1, num_experts)
        
        # Importance (sum of squares)
        importance = torch.sum(probs_flat ** 2, dim=0)
        
        # Load (sum of probabilities)
        load = torch.sum(probs_flat, dim=0)
        
        # Coefficient of variation loss
        importance_mean = torch.mean(importance)
        importance_std = torch.std(importance)
        importance_cv = importance_std / (importance_mean + 1e-12)
        
        load_mean = torch.mean(load)
        load_std = torch.std(load)
        load_cv = load_std / (load_mean + 1e-12)
        
        balancing_loss = importance_cv + load_cv
        
        return balancing_loss
    
    def _compute_efficiency_loss(self, decision: RoutingDecision) -> torch.Tensor:
        """Loss to encourage compute efficiency."""
        efficiency = decision.compute_efficiency()
        
        # Target efficiency (configurable)
        target_efficiency = 0.9  # 90% efficiency target
        
        # Loss: encourage high efficiency
        compute_efficiency = efficiency['compute_efficiency']
        efficiency_loss = F.relu(torch.tensor(target_efficiency - compute_efficiency))
        
        return efficiency_loss
    
    def get_expert_statistics(self) -> Dict[str, Any]:
        """Get expert usage statistics."""
        total_usage = self.expert_usage.sum().item()
        
        if total_usage > 0:
            usage_normalized = self.expert_usage / total_usage
            load_normalized = self.expert_load / (self.expert_load.sum() + 1e-12)
        else:
            usage_normalized = torch.zeros_like(self.expert_usage)
            load_normalized = torch.zeros_like(self.expert_load)
        
        # Statistics
        usage_mean = usage_normalized.mean().item()
        usage_std = usage_normalized.std().item()
        usage_min = usage_normalized.min().item()
        usage_max = usage_normalized.max().item()
        
        load_mean = load_normalized.mean().item()
        load_std = load_normalized.std().item()
        
        # Count experts with significant usage
        active_experts = (usage_normalized > 0.001).sum().item()
        
        return {
            'total_usage': total_usage,
            'usage_mean': usage_mean,
            'usage_std': usage_std,
            'usage_min': usage_min,
            'usage_max': usage_max,
            'load_mean': load_mean,
            'load_std': load_std,
            'active_experts': active_experts,
            'total_experts': self.num_experts,
            'activation_rate': active_experts / self.num_experts,
        }
    
    def reset_expert_statistics(self):
        """Reset expert usage statistics."""
        self.expert_usage.zero_()
        self.expert_load.zero_()
    
    def get_gradient_statistics(self) -> Dict[str, float]:
        """Get gradient statistics for debugging."""
        grad_stats = {}
        
        total_norm = 0.0
        total_mean = 0.0
        param_count = 0
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_stats[f'{name}_norm'] = grad.norm().item()
                grad_stats[f'{name}_mean'] = grad.mean().item()
                if grad.numel() > 1:
                    grad_stats[f'{name}_std'] = grad.std().item()
                else:
                    grad_stats[f'{name}_std'] = 0.0
                
                total_norm += grad.norm().item() ** 2
                total_mean += grad.mean().item()
                param_count += 1
        
        if param_count > 0:
            grad_stats['grad_norm'] = math.sqrt(total_norm)
            grad_stats['grad_mean'] = total_mean / param_count
        
        return grad_stats
    
    def update_training_step(self):
        """Update training step counter for annealing."""
        self.training_step += 1
    
    def get_metrics(self) -> RoutingMetrics:
        """Get current metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return RoutingMetrics()
    
    def log_metrics(self, step: int, loss_dict: Dict[str, torch.Tensor], 
                   decision: RoutingDecision):
        """Log routing metrics."""
        grad_stats = self.get_gradient_statistics()
        
        metrics = RoutingMetrics()
        metrics.update(step, loss_dict, decision, grad_stats)
        
        self.metrics_history.append(metrics)
        
        # Log periodically
        if step % 100 == 0:
            logger.info("router", 
                       f"Step {step}: Loss={metrics.loss_total:.4f}, "
                       f"Depth={metrics.depth_active_mean:.3f}, "
                       f"Width={metrics.width_mean:.3f}, "
                       f"Eff={metrics.overall_efficiency:.3f}, "
                       f"Grad={metrics.grad_norm:.4f}")


# ==================== ROUTER TRAINER ====================

class RouterTrainer:
    """
    Specialized trainer for router to ensure stable convergence.
    Router is the hardest part to train - needs careful handling.
    """
    
    def __init__(
        self,
        router: AdaptiveRouter,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        complexity_targets: bool = False,
    ):
        """
        Initialize router trainer.
        
        Args:
            router: AdaptiveRouter instance
            optimizer: Optimizer for router
            scheduler: Learning rate scheduler
            grad_clip: Gradient clipping value
            warmup_steps: Warmup steps for router training
            complexity_targets: Whether complexity targets are available
        """
        self.router = router
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.complexity_targets = complexity_targets
        
        # Training state
        self.step = 0
        self.best_loss = float('inf')
        self.patience = 0
        self.max_patience = 1000
        
        # Gradient accumulation
        self.grad_accum_steps = 1
        self.grad_accum_count = 0
        
        # Monitoring
        self.loss_history = []
        self.efficiency_history = []
        
        logger.info("router_trainer", f"Initialized RouterTrainer (warmup={warmup_steps})")
    
    def train_step(
        self,
        x: torch.Tensor,
        cot_features: torch.Tensor,
        target_complexity: Optional[torch.Tensor] = None,
        balance_experts: bool = True,
    ) -> Tuple[RoutingDecision, Dict[str, torch.Tensor]]:
        """
        Perform one training step.
        
        Returns:
            (decision, losses)
        """
        self.router.train()
        self.step += 1
        
        # Forward pass
        decision = self.router(
            x, cot_features, 
            training=True, 
            deterministic=False
        )
        
        # Compute losses
        losses = self.router.compute_loss(
            decision, target_complexity, balance_experts
        )
        
        # Scale loss for gradient accumulation
        loss = losses['total'] / self.grad_accum_steps
        
        # Backward
        loss.backward()
        self.grad_accum_count += 1
        
        # Update if accumulation steps reached
        if self.grad_accum_count >= self.grad_accum_steps:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.router.parameters(), 
                self.grad_clip
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            self.grad_accum_count = 0
            
            # Update router training step
            self.router.update_training_step()
        
        # Log metrics
        if self.step % 10 == 0:
            self.router.log_metrics(self.step, losses, decision)
        
        # Update best loss
        current_loss = losses['total'].item()
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience = 0
        else:
            self.patience += 1
        
        # Track efficiency
        efficiency = decision.compute_efficiency()['overall_efficiency']
        self.efficiency_history.append(efficiency)
        
        return decision, losses
    
    def evaluate(
        self,
        x: torch.Tensor,
        cot_features: torch.Tensor,
        target_complexity: Optional[torch.Tensor] = None,
    ) -> Tuple[RoutingDecision, Dict[str, float]]:
        """
        Evaluate router.
        
        Returns:
            (decision, metrics)
        """
        self.router.eval()
        
        with torch.no_grad():
            # Forward pass (deterministic)
            decision = self.router(
                x, cot_features,
                training=False,
                deterministic=True
            )
            
            # Compute losses
            losses = self.router.compute_loss(
                decision, target_complexity, balance_experts=False
            )
            
            # Convert to float
            loss_metrics = {k: v.item() for k, v in losses.items()}
            
            # Add statistics
            stats = decision.compute_statistics()
            loss_metrics.update(stats)
            
            # Add efficiency
            efficiency = decision.compute_efficiency()
            loss_metrics.update(efficiency)
            
            # Add expert statistics
            expert_stats = self.router.get_expert_statistics()
            loss_metrics.update(expert_stats)
        
        return decision, loss_metrics
    
    def should_stop(self) -> bool:
        """Check if training should stop."""
        # Early stopping
        if self.patience >= self.max_patience:
            logger.warning("router_trainer", f"Early stopping at step {self.step}")
            return True
        
        # Check for NaN in losses
        if self.loss_history and np.isnan(self.loss_history[-1]):
            logger.error("router_trainer", "Loss became NaN")
            return True
        
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.loss_history:
            return {}
        
        # Compute statistics
        recent_losses = self.loss_history[-100:] if len(self.loss_history) >= 100 else self.loss_history
        recent_efficiency = self.efficiency_history[-100:] if self.efficiency_history else []
        
        summary = {
            'step': self.step,
            'best_loss': self.best_loss,
            'current_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'loss_mean': np.mean(recent_losses),
            'loss_std': np.std(recent_losses),
            'loss_trend': np.polyfit(range(len(recent_losses)), recent_losses, 1)[0] if len(recent_losses) > 1 else 0.0,
            'efficiency_mean': np.mean(recent_efficiency) if recent_efficiency else 0.0,
            'efficiency_std': np.std(recent_efficiency) if recent_efficiency else 0.0,
            'patience': self.patience,
            'grad_accum_steps': self.grad_accum_steps,
            'grad_clip': self.grad_clip,
        }
        
        # Add router metrics
        router_metrics = self.router.get_metrics()
        summary.update(router_metrics.to_dict())
        
        # Add expert statistics
        expert_stats = self.router.get_expert_statistics()
        summary.update(expert_stats)
        
        return summary
    
    def save_checkpoint(self, path: str):
        """Save trainer checkpoint."""
        checkpoint = {
            'step': self.step,
            'router_state': self.router.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'patience': self.patience,
            'loss_history': self.loss_history,
            'efficiency_history': self.efficiency_history,
        }
        
        torch.save(checkpoint, path)
        logger.info("router_trainer", f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load trainer checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.step = checkpoint['step']
        self.router.load_state_dict(checkpoint['router_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.best_loss = checkpoint['best_loss']
        self.patience = checkpoint['patience']
        self.loss_history = checkpoint['loss_history']
        self.efficiency_history = checkpoint['efficiency_history']
        
        logger.info("router_trainer", f"Loaded checkpoint from {path} (step={self.step})")


# ==================== ROUTER VALIDATION ====================

class RouterValidator:
    """
    Validates router decisions against ground truth if available.
    For debugging and analysis.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Ground truth complexity if available
        self.has_ground_truth = False
        self.complexity_correlation = []
        
        # Decision consistency
        self.decision_consistency = []
        
        # Efficiency tracking
        self.efficiency_history = []
    
    def validate_complexity(
        self,
        predicted: torch.Tensor,
        ground_truth: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Validate complexity predictions."""
        if mask is not None:
            predicted = predicted[mask]
            ground_truth = ground_truth[mask]
        
        # Convert to numpy
        pred_np = predicted.cpu().detach().numpy().flatten()
        gt_np = ground_truth.cpu().detach().numpy().flatten()
        
        # Compute metrics
        mse = np.mean((pred_np - gt_np) ** 2)
        mae = np.mean(np.abs(pred_np - gt_np))
        
        # Correlation
        if len(pred_np) > 1:
            correlation = np.corrcoef(pred_np, gt_np)[0, 1]
        else:
            correlation = 0.0
        
        self.complexity_correlation.append(correlation)
        
        return {
            'complexity_mse': float(mse),
            'complexity_mae': float(mae),
            'complexity_correlation': float(correlation),
        }
    
    def validate_consistency(
        self,
        decision1: RoutingDecision,
        decision2: RoutingDecision,
        similarity_threshold: float = 0.8
    ) -> Dict[str, float]:
        """Validate consistency between two routing decisions."""
        # Compare depth decisions
        depth_similarity = torch.mean(
            (decision1.depth_mask == decision2.depth_mask).float()
        ).item()
        
        # Compare width decisions
        width_agreement = torch.mean(
            (decision1.width_idx == decision2.width_idx).float()
        ).item()
        
        # Compare complexity
        complexity_diff = torch.mean(
            torch.abs(decision1.complexity - decision2.complexity)
        ).item()
        
        consistency_score = (depth_similarity + width_agreement + (1 - complexity_diff)) / 3
        
        self.decision_consistency.append(consistency_score)
        
        return {
            'depth_similarity': depth_similarity,
            'width_agreement': width_agreement,
            'complexity_diff': complexity_diff,
            'consistency_score': consistency_score,
            'is_consistent': consistency_score > similarity_threshold,
        }
    
    def validate_efficiency(
        self,
        decision: RoutingDecision,
        baseline_efficiency: float = 0.1  # Dense model baseline
    ) -> Dict[str, float]:
        """Validate routing efficiency."""
        efficiency = decision.compute_efficiency()
        
        # Gain over baseline
        compute_gain = efficiency['compute_efficiency'] / baseline_efficiency
        overall_gain = efficiency['overall_efficiency'] / baseline_efficiency
        
        self.efficiency_history.append(efficiency['overall_efficiency'])
        
        return {
            **efficiency,
            'compute_gain': compute_gain,
            'overall_gain': overall_gain,
            'is_efficient': efficiency['overall_efficiency'] > 0.5,  # 50% threshold
        }
    
    def validate_expert_usage(
        self,
        decision: RoutingDecision,
        ideal_utilization: float = 0.8  # 80% of experts should be used
    ) -> Dict[str, float]:
        """Validate expert usage."""
        batch_size, seq_len, num_experts = decision.expert_probs.shape
        
        # Count active experts (used by at least one token)
        expert_used = (decision.expert_probs.sum(dim=(0, 1)) > 0.01).float()
        active_experts = expert_used.sum().item()
        
        # Utilization rate
        utilization_rate = active_experts / num_experts
        
        # Load balancing
        expert_load = decision.expert_probs.sum(dim=(0, 1))
        load_std = expert_load.std().item()
        load_mean = expert_load.mean().item()
        load_cv = load_std / (load_mean + 1e-12)
        
        return {
            'total_experts': num_experts,
            'active_experts': active_experts,
            'utilization_rate': utilization_rate,
            'load_mean': load_mean,
            'load_std': load_std,
            'load_cv': load_cv,
            'is_balanced': load_cv < 1.0,  # Coefficient of variation < 1
            'is_utilized': utilization_rate > ideal_utilization,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        summary = {}
        
        if self.complexity_correlation:
            summary['avg_complexity_correlation'] = np.mean(self.complexity_correlation)
            summary['std_complexity_correlation'] = np.std(self.complexity_correlation)
        
        if self.decision_consistency:
            summary['avg_decision_consistency'] = np.mean(self.decision_consistency)
            summary['std_decision_consistency'] = np.std(self.decision_consistency)
        
        if self.efficiency_history:
            summary['avg_efficiency'] = np.mean(self.efficiency_history)
            summary['std_efficiency'] = np.std(self.efficiency_history)
            summary['min_efficiency'] = np.min(self.efficiency_history)
            summary['max_efficiency'] = np.max(self.efficiency_history)
        
        return summary
    
    def reset(self):
        """Reset validation statistics."""
        self.complexity_correlation.clear()
        self.decision_consistency.clear()
        self.efficiency_history.clear()


class RoutingRegularizer(nn.Module):
    """
    Regularizer for routing decisions.
    Adds a small loss on uncertainty so uncertainty_estimator receives gradients.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.uncertainty_weight = getattr(config, 'routing_loss_weight', 0.01)

    def forward(self, decision: RoutingDecision) -> torch.Tensor:
        # Penalise high uncertainty — this keeps uncertainty_estimator in the graph.
        uncertainty_loss = decision.uncertainty.mean() * self.uncertainty_weight
        return uncertainty_loss


# ==================== TESTING ====================

__all__ = [
    'RoutingDecision',
    'RoutingMetrics',
    'AdaptiveRouter',
    'RouterTrainer',
    'RouterValidator',
    'RoutingRegularizer',
]

