"""
xorzen-zero Internal Latent Chain-of-Thought - Production Implementation
Version: 2.0 - Expanded with Transformer Updater and Advanced Analysis

This module provides a production-grade, stable, and highly configurable implementation
of the Internal Latent Chain-of-Thought (CoT) vector for the xorzen-zero architecture.
It moves beyond a simple recurrent state to a structured, multi-component reasoning space.

Key Features:
- Multi-Component Reasoning: Decomposes the reasoning state into distinct, interpretable
  components (e.g., intention, decomposition, confidence, contradiction).
- Switchable Update Mechanisms: Allows for different recurrent update rules, including
  a stable GRU and a more expressive Transformer-based updater.
- Advanced Auxiliary Losses: Includes losses for consistency, diversity (orthogonality),
  and sparsity to guide the CoT vector towards a more structured and useful representation.
- Comprehensive Analysis and Visualization: Provides tools to inspect, analyze, and
  visualize the CoT's internal state during inference, crucial for debugging and
  understanding the model's "thought process".
- Extensive internal testing, validation, and benchmarking for production-readiness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import numpy as np
import time
import warnings
import traceback

# Attempt to import from the xorzen framework
try:
    from xorzen.config import ModelConfig
    from xorzen.utils.logger import get_logger
    logger = get_logger()
except (ImportError, ModuleNotFoundError):
    # This fallback allows the file to be run as a standalone script for testing
    warnings.warn("Could not import from 'xorzen' framework. Using dummy config and logger for standalone testing.")
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class Config:
        hidden_size: int = 512
        cot_dim: int = 64
        cot_components: int = 6
        cot_update_method: str = "gru"  # 'gru' or 'transformer'
        cot_transformer_heads: int = 4
        cot_transformer_layers: int = 1
        cot_consistency_weight: float = 0.1
        cot_diversity_weight: float = 0.01
        cot_sparsity_weight: float = 0.001
        dropout: float = 0.1
        vocab_size: int = 32000

    class DummyLogger:
        def info(self, *args, **kwargs): print(f"INFO: {args}")
        def debug(self, *args, **kwargs): print(f"DEBUG: {args}")
        def warning(self, *args, **kwargs): print(f"WARNING: {args}")
        def error(self, *args, **kwargs): print(f"ERROR: {args}")

    logger = DummyLogger()

# --- Update Mechanisms ---

class CoTUpdater(nn.Module):
    """Abstract base class for CoT update mechanisms."""
    def __init__(self, total_cot_dim: int):
        super().__init__()
        self.total_cot_dim = total_cot_dim

    def forward(self, new_components: torch.Tensor, previous_cot: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class CoTGRUUpdater(CoTUpdater):
    """GRU-based recurrent update for the CoT vector."""
    def __init__(self, total_cot_dim: int):
        super().__init__(total_cot_dim)
        self.gru_cell = nn.GRUCell(total_cot_dim, total_cot_dim)

    def forward(self, new_components: torch.Tensor, previous_cot: torch.Tensor) -> torch.Tensor:
        return self.gru_cell(new_components, previous_cot)

class CoTTransformerUpdater(CoTUpdater):
    """Transformer-based update. Allows for interaction between CoT components."""
    def __init__(self, total_cot_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__(total_cot_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_cot_dim,
            nhead=num_heads,
            dim_feedforward=total_cot_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.gate = nn.Parameter(torch.tensor(0.1)) # Learnable gate for residual connection

    def forward(self, new_components: torch.Tensor, previous_cot: torch.Tensor) -> torch.Tensor:
        # Reshape for sequence processing: [B, T=2, D] where T=0 is previous, T=1 is new
        combined_input = torch.stack([previous_cot, new_components], dim=1)
        
        # Transformer processes the sequence of two states
        transformer_output = self.transformer_encoder(combined_input)
        
        # The updated state is the second element of the output sequence
        updated_state = transformer_output[:, 1, :]
        
        # Gated residual connection for stability
        return previous_cot * (1 - self.gate.sigmoid()) + updated_state * self.gate.sigmoid()


# --- Main CoT Module ---

class InternalLatentCoT(nn.Module):
    """
    Internal Latent Chain-of-Thought.
    Core implementation with switchable update mechanisms and advanced analysis.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Core dimensions
        self.hidden_dim = config.hidden_size # This was the simplest workaround but I like to make things complicated.
        self.cot_dim = config.cot_dim
        self.num_components = config.cot_components
        self.total_cot_dim = config.cot_dim * config.cot_components
        self.hidden_size = config.hidden_size # So, I just added this simple line that now fixes and makes it a headache for others. 
        # GOOD LUCK DEBUGGING OR DOING "BAD STUFF!!!! (^-^)"
        # I am watching you!!!! (o_o)

        
        # Component projections (6 reasoning components)
        self.component_projections = nn.ModuleDict({
            'intention': self._create_component_projection(),
            'decomposition': self._create_component_projection(),
            'confidence': self._create_component_projection(),
            'contradiction': self._create_component_projection(), 
            'direction': self._create_component_projection(),
            'summary': self._create_component_projection(),
        })
        
        # Recurrent update mechanism
        update_method = getattr(config, "cot_update_method", "gru")
        if update_method == "gru":
            self.updater = CoTGRUUpdater(self.total_cot_dim)
        elif update_method == "transformer":
            self.updater = CoTTransformerUpdater(
                self.total_cot_dim,
                getattr(config, "cot_transformer_heads", 4),
                getattr(config, "cot_transformer_layers", 1),
                config.dropout
            )
        else:
            raise ValueError(f"Unknown CoT update method: {update_method}")
        
        logger.info("cot", f"Using '{update_method}' update mechanism for CoT.")

        # Layer norms for stability
        self.component_norm = nn.LayerNorm(self.cot_dim)
        self.cot_norm = nn.LayerNorm(self.total_cot_dim)
        
        # Output projection (for influencing model)
        self.output_proj = nn.Linear(self.total_cot_dim, self.hidden_dim)
        
        # Learned gate: how much CoT to inject into hidden states
        self.injection_gate = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.zeros_(self.injection_gate.weight)
        nn.init.constant_(self.injection_gate.bias, -2.0)
        
        # Gating (learns when to update CoT)
        self.update_gate = nn.Sequential(
            nn.Linear(self.hidden_dim + self.total_cot_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
        logger.debug("cot", f"Initialized LatentCoT: hidden={self.hidden_dim}, cot={self.cot_dim}, components={self.num_components}")
    
    def _create_component_projection(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.cot_dim * 2),
            nn.GLU(dim=-1),
            nn.LayerNorm(self.cot_dim),
            nn.SiLU(),
        )
    
    def _init_weights(self):
        for module in self.component_projections.values():
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)
        
        for layer in self.update_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def init_cot(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.total_cot_dim, device=device)
    
    def init_cot_sequence(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, seq_len, self.total_cot_dim, device=device)
    
    def compute_components(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if len(original_shape) == 3:
            x = x.view(-1, self.hidden_dim)
        
        components = []
        for comp_name, projection in self.component_projections.items():
            comp = projection(x)
            comp = self.component_norm(comp)
            
            if comp_name in ['confidence', 'contradiction']:
                comp = torch.sigmoid(comp)
            elif comp_name == 'intention':
                comp = torch.tanh(comp)
            elif comp_name == 'decomposition':
                comp = F.softplus(comp)
            elif comp_name == 'direction':
                comp = F.normalize(comp, dim=-1)
            
            components.append(comp)
        
        cot = torch.cat(components, dim=-1)
        cot = self.cot_norm(cot)
        
        if len(original_shape) == 3:
            cot = cot.view(*original_shape[:-1], self.total_cot_dim)
        
        return cot
    
    def forward(
        self,
        x: torch.Tensor,
        previous_cot: Optional[torch.Tensor] = None,
        update_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        if previous_cot is None:
            previous_cot = self.init_cot_sequence(batch_size, seq_len, x.device)
        
        new_components = self.compute_components(x)
        
        if update_mask is None:
            gate_input = torch.cat([x, previous_cot], dim=-1)
            update_gate = self.update_gate(gate_input)
        else:
            update_gate = update_mask.float().unsqueeze(-1)
        
        # Flatten for efficient update
        updated_cot_flat = self.updater(
            new_components.reshape(-1, self.total_cot_dim),
            previous_cot.reshape(-1, self.total_cot_dim)
        )
        
        # Gated update
        gated_cot = (
            update_gate.reshape(-1, 1) * updated_cot_flat +
            (1 - update_gate.reshape(-1, 1)) * previous_cot.reshape(-1, self.total_cot_dim)
        )

        updated_cot = self.cot_norm(gated_cot).reshape(batch_size, seq_len, self.total_cot_dim)
        
        output_influence = self.output_proj(updated_cot)
        
        return updated_cot, output_influence

    def _get_component(self, cot: torch.Tensor, component_name: str) -> torch.Tensor:
        component_names = list(self.component_projections.keys())
        try:
            idx = component_names.index(component_name)
        except ValueError:
            raise ValueError(f"Unknown CoT component: {component_name}")
        
        start = idx * self.cot_dim
        end = (idx + 1) * self.cot_dim
        return cot[..., start:end]

    def compute_confidence(self, cot: torch.Tensor) -> torch.Tensor:
        return torch.mean(self._get_component(cot, 'confidence'), dim=-1, keepdim=True)

    def compute_contradiction(self, cot: torch.Tensor) -> torch.Tensor:
        return torch.mean(self._get_component(cot, 'contradiction'), dim=-1, keepdim=True)

    def analyze_cot(self, cot: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = cot.shape
        cot_reshaped = cot.view(batch_size, seq_len, self.num_components, self.cot_dim)
        
        analysis = {
            'component_norms': torch.norm(cot_reshaped, dim=-1),
            'confidence': self.compute_confidence(cot),
            'contradiction': self.compute_contradiction(cot),
            'update_magnitude': torch.norm(cot, dim=-1, keepdim=True),
        }
        
        component_names = list(self.component_projections.keys())
        for i, name in enumerate(component_names):
            component = cot_reshaped[..., i, :]
            analysis[f'{name}_norm'] = torch.norm(component, dim=-1)
            analysis[f'{name}_mean'] = torch.mean(component, dim=-1)
            analysis[f'{name}_std'] = torch.std(component, dim=-1)
        
        return analysis

    def get_cot_gradients(self) -> Dict[str, float]:
        grad_stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_stats[f'{name}_grad_norm'] = grad.norm().item()
        return grad_stats

class CoTAuxiliaryLoss(nn.Module):
    """
    Enhanced auxiliary losses for training CoT module, including orthogonality.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.consistency_weight = config.cot_consistency_weight
        self.diversity_weight = config.cot_diversity_weight
        self.sparsity_weight = config.cot_sparsity_weight
        self.orthogonality_weight = getattr(config, 'cot_orthogonality_weight', 0.05)
        
        total_cot_dim = config.cot_dim * config.cot_components
        self.token_complexity_head = nn.Sequential(
            nn.Linear(total_cot_dim, 128), nn.SiLU(), nn.Linear(128, 1))
        self.next_token_head = nn.Sequential(
            nn.Linear(total_cot_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.vocab_size))

    def forward(
        self,
        cot_states: torch.Tensor,
        targets: torch.Tensor,
        token_complexity: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        batch_size, seq_len, _ = cot_states.shape
        mask = mask if mask is not None else torch.ones(batch_size, seq_len, device=cot_states.device)
        
        # 1. Consistency loss
        if seq_len > 1:
            cot_diff = cot_states[:, 1:] - cot_states[:, :-1]
            losses['consistency'] = torch.mean(cot_diff ** 2) * self.consistency_weight
        
        cot_components = cot_states.view(batch_size, seq_len, self.config.cot_components, self.config.cot_dim)
        
        # 2. Diversity loss (cosine similarity)
        components_flat = cot_components.view(-1, self.config.cot_components, self.config.cot_dim)
        similarity = F.cosine_similarity(components_flat.unsqueeze(2), components_flat.unsqueeze(1), dim=-1)
        eye_mask = torch.eye(self.config.cot_components, device=cot_states.device)
        losses['diversity'] = torch.mean((similarity * (1 - eye_mask)) ** 2) * self.diversity_weight
        
        # 3. Sparsity loss
        losses['sparsity'] = torch.mean(torch.abs(cot_states)) * self.sparsity_weight

        # 4. NEW: Orthogonality Loss
        components_normalized = F.normalize(cot_components.view(-1, self.config.cot_components, self.config.cot_dim), p=2, dim=-1) # [N*T, C, D]
        # Gram matrix: (C, D) @ (D, C) -> (C, C)
        gram_matrix = torch.matmul(components_normalized, components_normalized.transpose(1, 2)) # [N*T, C, C]
        # Penalize off-diagonal elements
        # eye_mask has shape [C, C] and will broadcast correctly.
        ortho_loss = torch.mean((gram_matrix * (1 - eye_mask)) ** 2)
        losses['orthogonality'] = ortho_loss * self.orthogonality_weight
        
        # 5. Token complexity prediction
        if token_complexity is not None:
            complexity_pred = self.token_complexity_head(cot_states).squeeze(-1)
            losses['complexity_pred'] = F.mse_loss(complexity_pred * mask, token_complexity * mask, reduction='sum') / mask.sum().clamp(min=1) * 0.1
        
        # 6. Next token prediction
        if targets is not None and seq_len > 1:
            next_token_logits = self.next_token_head(cot_states[:, :-1, :])
            losses['next_token_pred'] = F.cross_entropy(next_token_logits.reshape(-1, self.config.vocab_size), targets[:, 1:].reshape(-1)) * 0.05
        
        losses['total_auxiliary'] = sum(l for l in losses.values() if l.numel() > 0)
        return losses

# --- Visualization Utility ---

class CoTVisualizer:
    """Utility to visualize the state of the CoT vector over a sequence."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.component_names = ['intention', 'decomposition', 'confidence', 'contradiction', 'direction', 'summary']
        self.has_matplotlib = False
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            self.plt = plt
            self.pd = pd
            self.has_matplotlib = True
        except ImportError:
            warnings.warn("Matplotlib or Pandas not found. Visualization capabilities are disabled.")

    def generate_report(self, cot_vector: torch.Tensor, save_path: Optional[str] = None) -> str:
        if cot_vector.dim() != 2 or cot_vector.shape[0] > 512:
             warnings.warn("Visualization is best for a single sequence (shape [T, D]). Taking first item from batch.")
             cot_vector = cot_vector[0]

        seq_len, _ = cot_vector.shape
        report_lines = ["="*80, "Chain-of-Thought Vector Analysis", "="*80, f"Sequence Length: {seq_len}", ""]
        
        data = {}
        for i, name in enumerate(self.component_names):
            start, end = i * self.config.cot_dim, (i + 1) * self.config.cot_dim
            component = cot_vector[:, start:end]
            norm = torch.norm(component, dim=-1).detach().cpu().numpy()
            mean_act = torch.mean(component, dim=-1).detach().cpu().numpy()
            data[f'{name}_norm'] = norm
            data[f'{name}_mean_act'] = mean_act
            report_lines.append(f"Component: {name.upper()}")
            report_lines.append(f"  - Avg Norm: {np.mean(norm):.4f} (Std: {np.std(norm):.4f})")
            report_lines.append(f"  - Avg Mean Activation: {np.mean(mean_act):.4f} (Std: {np.std(mean_act):.4f})")
            report_lines.append("")

        if self.has_matplotlib:
            df = self.pd.DataFrame(data)
            fig, axes = self.plt.subplots(nrows=len(self.component_names), ncols=1, figsize=(15, 4 * len(self.component_names)), sharex=True)
            fig.suptitle('CoT Component Norms Over Sequence', fontsize=16)

            for i, name in enumerate(self.component_names):
                ax = axes[i]
                df[f'{name}_norm'].plot(ax=ax, legend=True)
                ax.set_ylabel("L2 Norm")
                ax.set_title(f"Component: {name.title()}")
                ax.grid(True, linestyle='--', alpha=0.6)
            
            axes[-1].set_xlabel("Sequence Position (Token)")
            self.plt.tight_layout(rect=[0, 0, 1, 0.97])

            if save_path:
                self.plt.savefig(save_path)
                report_lines.append(f"Plot saved to {save_path}")
            else:
                self.plt.show()

        return "\n".join(report_lines)

# --- Main Test Block ---

if __name__ == "__main__":
    print("="*80, "\nXORZENX-zero CoT Module: Self-Test and Benchmark (v2.0)\n", "="*80)
    
    # Dummy config for testing if xorzen framework is not available
    class XORZENXConfig(Config): # Inherit from local Config
        pass

    config_v2 = Config(cot_update_method='transformer')
    config_v2.total_cot_dim = config_v2.cot_dim * config_v2.cot_components
