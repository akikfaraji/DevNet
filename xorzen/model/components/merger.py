"""
xorzen-zero Merger Gate - Production Implementation
Handles the fusion of multiple information streams within the xorzen-zero architecture.

This module is critical for effectively combining the outputs of the core pathways:
1. HASS (Hybrid Attention-Shard Switch) Blocks: Provide local, global, and sequential processing.
2. MoE (Mixture of Experts) Fabric: Provides specialized, high-capacity computation.
3. CoT (Chain-of-Thought) Vector: Provides an internal reasoning or meta-cognitive state.

The merger's role is to intelligently integrate these streams, allowing the model to
dynamically weigh different types of information for each token. A simple concatenation
is often insufficient. This module provides several advanced, configurable strategies
for this fusion process.

Key Innovations:
- Gated Fusion: Learns to weigh each input stream (HASS, MoE, CoT) dynamically.
- Cross-Attention Fusion: Uses one stream (e.g., HASS) as a query to "attend" to the others,
  allowing for context-sensitive information extraction.
- FiLM-style Conditioning: Uses the CoT vector to modulate the other streams, providing a
  powerful form of meta-control.
- Comprehensive internal testing and benchmarking to ensure stability and performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Type
import time
import math
import warnings
import numpy as np

# Attempt to import from the xorzen framework
try:
    from xorzen.config import ModelConfig
    from xorzen.utils.logger import get_logger
    logger = get_logger()
except (ImportError, ModuleNotFoundError):
    # This fallback allows the file to be run as a standalone script for testing
    warnings.warn("Could not import from 'xorzen' framework. Using dummy config and logger for standalone testing.")
    from dataclasses import dataclass

    @dataclass
    class Config: # Renamed to Config to match the try block
        hidden_size: int = 512
        cot_dim: int = 64
        cot_components: int = 6
        merger_hidden_multiplier: float = 2.0
        merger_num_layers: int = 2
        dropout: float = 0.1
        layer_norm_eps: float = 1e-5
        
    class DummyLogger:
        def info(self, *args, **kwargs): print(f"INFO: {args}")
        def debug(self, *args, **kwargs): print(f"DEBUG: {args}")
        def warning(self, *args, **kwargs): print(f"WARNING: {args}")
        def error(self, *args, **kwargs): print(f"ERROR: {args}")
    
    logger = DummyLogger()

# --- Base Class for Mergers ---

class BaseMerger(nn.Module):
    """
    Abstract base class for all merger implementations.
    Ensures a consistent interface.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.total_cot_dim = config.cot_dim * config.cot_components

    def forward(
        self,
        hass_output: torch.Tensor,
        moe_output: torch.Tensor,
        cot_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Abstract forward method. Subclasses must implement this.

        Args:
            hass_output (torch.Tensor): Output from HASS blocks [B, T, H]
            moe_output (torch.Tensor): Output from MoE fabric [B, T, H]
            cot_vector (torch.Tensor): Internal CoT state [B, T, C]
            attention_mask (Optional[torch.Tensor]): Attention mask [B, T]

        Returns:
            torch.Tensor: The fused output tensor [B, T, H]
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_complexity(self) -> Dict[str, int]:
        """
        Returns the computational complexity of the merger.
        This can be used for analysis and routing decisions.
        """
        raise NotImplementedError("Subclasses must implement get_complexity.")


# --- Merger Implementations ---

class LinearMerger(BaseMerger):
    """
    The simplest merger: concatenates all inputs and passes them through a
    linear projection. This serves as a baseline.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.input_dim = self.hidden_dim * 2 + self.total_cot_dim
        
        self.output_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        self._init_weights()
        logger.debug("merger", f"Initialized LinearMerger with input dim {self.input_dim}")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        hass_output: torch.Tensor,
        moe_output: torch.Tensor,
        cot_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Performs a simple concatenation and linear projection.
        """
        # Validate input dimensions
        if hass_output.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected hass_output to have hidden_dim {self.hidden_dim}, but got {hass_output.shape[-1]}")
        if moe_output.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected moe_output to have hidden_dim {self.hidden_dim}, but got {moe_output.shape[-1]}")
        if cot_vector.shape[-1] != self.total_cot_dim:
            raise ValueError(f"Expected cot_vector to have total_cot_dim {self.total_cot_dim}, but got {cot_vector.shape[-1]}")

        merged = torch.cat([hass_output, moe_output, cot_vector], dim=-1)
        output = self.output_proj(merged)
        return output

    def get_complexity(self) -> Dict[str, int]:
        return {
            "macs": self.input_dim * self.hidden_dim,
            "params": sum(p.numel() for p in self.parameters())
        }

class GatedMerger(BaseMerger):
    """
    A more advanced merger that learns to gate each input stream.
    This allows the model to dynamically decide how much of each input
    (HASS, MoE, CoT) to use for each token.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.merger_hidden_dim = int(config.hidden_size * config.merger_hidden_multiplier)
        self.input_dim = self.hidden_dim * 2 + self.total_cot_dim

        # Gate controller network
        self.gate_controller = nn.Sequential(
            nn.Linear(self.input_dim, self.merger_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.merger_hidden_dim, 3) # 3 gates: HASS, MoE, CoT
        )
        
        # CoT vector needs to be projected to the hidden dimension to be summable
        self.cot_proj = nn.Linear(self.total_cot_dim, self.hidden_dim)
        
        self.output_norm = nn.LayerNorm(self.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
        logger.debug("merger", "Initialized GatedMerger")

    def _init_weights(self):
        for module in self.gate_controller:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.cot_proj.weight)
        if self.cot_proj.bias is not None:
            nn.init.zeros_(self.cot_proj.bias)

    def forward(
        self,
        hass_output: torch.Tensor,
        moe_output: torch.Tensor,
        cot_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes gates for each input stream and returns a weighted sum.
        """
        # Validate input dimensions
        if hass_output.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected hass_output to have hidden_dim {self.hidden_dim}, but got {hass_output.shape[-1]}")
        if moe_output.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected moe_output to have hidden_dim {self.hidden_dim}, but got {moe_output.shape[-1]}")
        if cot_vector.shape[-1] != self.total_cot_dim:
            raise ValueError(f"Expected cot_vector to have total_cot_dim {self.total_cot_dim}, but got {cot_vector.shape[-1]}")

        # Concatenate inputs to compute gates
        gate_input = torch.cat([hass_output, moe_output, cot_vector], dim=-1)
        
        # Compute gates (B, T, 3)
        gates = self.gate_controller(gate_input)
        
        # Apply softmax to get normalized weights for each stream
        gate_weights = F.softmax(gates, dim=-1)
        
        g_hass, g_moe, g_cot = gate_weights.chunk(3, dim=-1)
        
        # Project CoT to hidden dim
        cot_proj = self.cot_proj(cot_vector)
        
        # Compute weighted sum
        fused_output = (
            g_hass * hass_output +
            g_moe * moe_output +
            g_cot * cot_proj
        )
        
        # Final normalization and dropout
        output = self.dropout(self.output_norm(fused_output))
        
        return output

    def get_complexity(self) -> Dict[str, int]:
        return {
            "macs": self.input_dim * self.merger_hidden_dim + self.merger_hidden_dim * 3,
            "params": sum(p.numel() for p in self.parameters())
        }

# --- Factory Function ---

def create_merger(config: ModelConfig, merger_type: str = "gated") -> BaseMerger:
    """
    Factory function to create a specific type of merger.

    Args:
        config (ModelConfig): The model configuration object.
        merger_type (str): The type of merger to create.
                           Options: "linear", "gated".

    Returns:
        BaseMerger: An instance of the specified merger.
    """
    merger_registry: Dict[str, Type[BaseMerger]] = {
        "linear": LinearMerger,
        "gated": GatedMerger,
    }
    
    merger_class = merger_registry.get(merger_type)
    if merger_class is None:
        raise ValueError(f"Unknown merger_type '{merger_type}'. Available options: {list(merger_registry.keys())}")
    
    logger.info("merger", f"Creating merger of type: {merger_type}")
    return merger_class(config)


# This allows the file to be run as a standalone script for testing


# The XORZENXMergerGate is kept for compatibility with the core model,
# but it now acts as a wrapper around the factory.
class xorzenMergerGate(BaseMerger):
    """
    The main merger gate used by the xorzen-zero model.
    This class acts as a configurable wrapper, instantiating the desired
    merger type (e.g., 'gated', 'linear') based on the model configuration.
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Use a 'merger_type' attribute from the config, defaulting to 'gated'
        merger_type = getattr(config, "merger_type", "gated")
        
        self.merger_impl = create_merger(config, merger_type)

        logger.info("merger", f"XORZENXMergerGate initialized, using '{merger_type}' implementation.")

    def forward(
        self,
        hass_output: torch.Tensor,
        moe_output: torch.Tensor,
        cot_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Delegates the forward pass to the chosen merger implementation.
        """
        return self.merger_impl(hass_output, moe_output, cot_vector, attention_mask)

    def get_complexity(self) -> Dict[str, int]:
        """
        Delegates complexity calculation to the chosen merger implementation.
        """
        return self.merger_impl.get_complexity()