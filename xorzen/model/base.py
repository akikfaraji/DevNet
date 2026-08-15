"""
Base Model for xorzen
Version: 1.0
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class ModelOutput:
    """Complete model output with all intermediate states."""
    # Primary outputs
    logits: torch.Tensor  # [batch, seq, vocab] - next token predictions
    loss: Optional[torch.Tensor] = None  # scalar - training loss
    
    # Internal states (for analysis/debugging)
    cot_vector: Optional[torch.Tensor] = None  # [batch, seq, cot_dim] - reasoning trace
    routing_info: Optional[Any] = None  # routing decisions
    layer_outputs: Optional[List[torch.Tensor]] = None  # intermediate layer outputs
    expert_stats: Optional[Dict[str, Any]] = None  # expert usage statistics
    
    # Auxiliary losses
    routing_loss: Optional[torch.Tensor] = None  # routing regularization
    load_balance_loss: Optional[torch.Tensor] = None  # expert load balancing
    cot_consistency_loss: Optional[torch.Tensor] = None  # CoT stability
    lm_loss: Optional[torch.Tensor] = None  # pure cross-entropy (before aux losses added)
    
    # Performance metrics
    active_params: Optional[int] = None  # number of active parameters
    compute_cost: Optional[float] = None  # FLOPs estimate
    
    def total_loss(self) -> Optional[torch.Tensor]:
        """Compute total weighted loss."""
        if self.loss is None:
            return None
        
        total = self.loss
        if self.routing_loss is not None:
            total = total + self.routing_loss
        if self.load_balance_loss is not None:
            total = total + self.load_balance_loss
        if self.cot_consistency_loss is not None:
            total = total + self.cot_consistency_loss
        
        return total

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    
    # Special tokens
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # Early stopping
    early_stopping: bool = True
    min_length: int = 10
    
    # Advanced
    no_repeat_ngram_size: int = 0
    length_penalty: float = 1.0

class BaseModel(nn.Module):
    """
    Base class for all xorzen models.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def save_checkpoint(self, path: str, **kwargs):
        raise NotImplementedError

    def load_checkpoint(self, path: str, **kwargs):
        raise NotImplementedError

    def count_parameters(self, only_trainable: bool = False) -> int:
        """Count total or trainable parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_memory_footprint(self) -> Dict[str, Any]:
        """Get model memory footprint."""
        param_size = 0
        param_count = 0
        buffer_size = 0
        buffer_count = 0
        
        for param in self.parameters():
            param_count += param.numel()
            param_size += param.numel() * param.element_size()
        
        for buffer in self.buffers():
            buffer_count += buffer.numel()
            buffer_size += buffer.numel() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_count': param_count,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_count': buffer_count,
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'total_size_mb': total_size / (1024 ** 2),
            'total_size_gb': total_size / (1024 ** 3)
        }

