"""
Production-grade HASS Block for xorzen-zero.
Hybrid Attention-Shard Switch with 3 pathways: Local Attention, Low-Rank Global, SSM.
This is where tokens actually get processed based on router decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import numpy as np

from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import TensorStability
from xorzen.model.components.routing import RoutingDecision

logger = get_logger()


# ==================== PATHWAY IMPLEMENTATIONS ====================

class LocalAttentionPathway(nn.Module):
    """
    Local Causal Attention Pathway.
    Efficient attention with windowed context.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        causal: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size = window_size
        self.causal = causal
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Layer norms for stability
        self.ln_q = nn.LayerNorm(self.head_dim)
        self.ln_k = nn.LayerNorm(self.head_dim)
        
        # Cached masks — built once, reused every forward call
        self._cached_window_mask_len: int = -1
        self._cached_causal_mask_len: int = -1
        self.register_buffer("_window_mask_cache", None, persistent=False)
        self.register_buffer("_causal_mask_cache", None, persistent=False)
        
        # Initialize
        self._init_weights()
        
        logger.debug("hass", 
                    f"LocalAttention: hidden={hidden_dim}, heads={num_heads}, "
                    f"window={window_size}, causal={causal}")
    
    def _init_weights(self):
        """Initialize attention weights."""
        # QKV projections
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(proj.bias)
        
        # Output projection
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with local attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            attention_mask: Attention mask [batch, seq_len] or [batch, seq_len, seq_len]
            position_bias: Position bias for relative positions
            
        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x)  # [batch, seq, hidden]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply layer norms per head
        q = self.ln_q(q.transpose(1, 2)).transpose(1, 2)
        k = self.ln_k(k.transpose(1, 2)).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply window mask for local attention
        if self.window_size > 0:
            attn_scores = self._apply_window_mask(attn_scores, seq_len)
        
        # Apply causal mask if needed
        if self.causal:
            attn_scores = self._apply_causal_mask(attn_scores)
        
        # Apply external attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply position bias if provided
        if position_bias is not None:
            attn_scores = attn_scores + position_bias
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output
    
    def _apply_window_mask(self, attn_scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply window mask for local attention — cached per seq_len."""
        device = attn_scores.device
        if self._cached_window_mask_len != seq_len or self._window_mask_cache is None:
            window_mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end   = min(seq_len, i + self.window_size + 1)
                window_mask[i, :start] = False
                window_mask[i, end:]   = False
            self._window_mask_cache  = window_mask
            self._cached_window_mask_len = seq_len
        window_mask = self._window_mask_cache.to(device)
        attn_scores = attn_scores.masked_fill(~window_mask[None, None, :, :], float('-inf'))
        return attn_scores
    
    def _apply_causal_mask(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """Apply causal mask — cached per seq_len."""
        seq_len = attn_scores.size(-1)
        device  = attn_scores.device
        if self._cached_causal_mask_len != seq_len or self._causal_mask_cache is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            self._causal_mask_cache  = causal_mask
            self._cached_causal_mask_len = seq_len
        causal_mask = self._causal_mask_cache.to(device)
        attn_scores = attn_scores.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
        return attn_scores
    
    def get_compute_stats(self, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
        """Get compute statistics for this pathway"""

        # FLOPs estimation        
        qkv_flops = 3 * batch_size * seq_len * self.hidden_dim * self.hidden_dim
        attn_flops = batch_size * self.num_heads * seq_len * seq_len * self.head_dim * 2
        output_flops = batch_size * seq_len * self.hidden_dim * self.hidden_dim
        
        total_flops = qkv_flops + attn_flops + output_flops
        
        # Memory estimation
        param_memory = sum(p.numel() for p in self.parameters()) * 4  # 4 bytes per param (float32)
        activation_memory = batch_size * seq_len * self.hidden_dim * 4 * 10  # Rough estimate
        
        return {
            'flops_total': total_flops,
            'flops_per_token': total_flops / (batch_size * seq_len),
            'param_count': sum(p.numel() for p in self.parameters()),
            'param_memory_bytes': param_memory,
            'activation_memory_bytes': activation_memory,
            'window_size': self.window_size,
        }


class LowRankGlobalPathway(nn.Module):
    """
    Low-Rank Global Attention Pathway.
    Efficient global context via low-rank approximation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        low_rank_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.low_rank_dim = low_rank_dim
        self.num_heads = num_heads
        
        # Low-rank projections
        self.to_low_rank = nn.Linear(hidden_dim, low_rank_dim * num_heads)
        self.from_low_rank = nn.Linear(low_rank_dim * num_heads, hidden_dim)
        
        # Context aggregation (weighted pooling)
        self.context_weights = nn.Parameter(torch.randn(1, 1, low_rank_dim * num_heads))
        
        # Layer norms
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_low_rank = nn.LayerNorm(low_rank_dim * num_heads)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self._init_weights()
        
        logger.debug("hass", 
                    f"LowRankGlobal: hidden={hidden_dim}, low_rank={low_rank_dim}, "
                    f"heads={num_heads}")
    
    def _init_weights(self):
        """Initialize low-rank pathway weights."""
        # Low-rank projections
        nn.init.xavier_uniform_(self.to_low_rank.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.to_low_rank.bias)
        
        nn.init.xavier_uniform_(self.from_low_rank.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.from_low_rank.bias)
        
        # Context weights
        nn.init.normal_(self.context_weights, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with low-rank global attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            
        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Layer norm input
        x_norm = self.ln_input(x)
        
        # Project to low-rank space
        low_rank = self.to_low_rank(x_norm)  # [batch, seq_len, low_rank*heads]
        low_rank = self.ln_low_rank(low_rank)
        low_rank = F.gelu(low_rank)
        
        # Compute global context (weighted average)
        # Attention weights from context_weights
        attn_weights = torch.matmul(low_rank, self.context_weights.transpose(-1, -2))
        attn_weights = F.softmax(attn_weights / math.sqrt(self.low_rank_dim), dim=1)
        
        # Apply attention to get global context
        global_context = torch.matmul(attn_weights.transpose(-1, -2), low_rank)
        global_context = global_context.expand(-1, seq_len, -1)  # [batch, seq_len, low_rank*heads]
        
        # Combine local and global
        combined = low_rank + global_context
        
        # Project back to hidden dimension
        output = self.from_low_rank(combined)
        output = self.dropout(output)
        
        return output
    
    def get_compute_stats(self, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
        """Get compute statistics for this pathway."""
        # FLOPs estimation
        to_low_rank_flops = batch_size * seq_len * self.hidden_dim * self.low_rank_dim * self.num_heads
        context_flops = batch_size * seq_len * self.low_rank_dim * self.num_heads  # Softmax
        from_low_rank_flops = batch_size * seq_len * self.low_rank_dim * self.num_heads * self.hidden_dim
        
        total_flops = to_low_rank_flops + context_flops + from_low_rank_flops
        
        # Memory estimation
        param_memory = sum(p.numel() for p in self.parameters()) * 4
        activation_memory = batch_size * seq_len * self.low_rank_dim * self.num_heads * 4 * 3
        
        return {
            'flops_total': total_flops,
            'flops_per_token': total_flops / (batch_size * seq_len),
            'param_count': sum(p.numel() for p in self.parameters()),
            'param_memory_bytes': param_memory,
            'activation_memory_bytes': activation_memory,
            'low_rank_dim': self.low_rank_dim,
            'compression_ratio': self.hidden_dim / (self.low_rank_dim * self.num_heads),
        }


class SSMPathway(nn.Module):
    """
    State Space Model (SSM) Pathway.
    Efficient sequential processing with linear complexity.
    Mamba-style diagonal-A SSM with input-dependent discretisation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_conv: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.kernel_size = kernel_size
        self.use_conv = use_conv
        
        # State transition parameters — diagonal A only (no matrix_exp overhead)
        self.A_log  = nn.Parameter(torch.zeros(state_dim))     # log(-a), init -> a=-1
        self.dt_proj = nn.Linear(hidden_dim, state_dim, bias=True)  # input-dep dt
        self.B_proj = nn.Linear(hidden_dim, state_dim)
        self.C_proj = nn.Linear(hidden_dim, state_dim)
        
        # Output projection
        self.D_proj = nn.Linear(state_dim, hidden_dim)
        
        # Conv for local patterns (optional)
        if use_conv:
            self.conv = nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim  # Depthwise separable
            )
        else:
            self.conv = None
        
        # Gating mechanism
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        # Layer norms
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_state = nn.LayerNorm(state_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self._init_weights()
        
        logger.debug("hass", 
                    f"SSMPathway: hidden={hidden_dim}, state={state_dim}, "
                    f"kernel={kernel_size}, conv={use_conv}")
    
    def _init_weights(self):
        """Initialize SSM weights."""
        # A_log = log(-a); init 0 -> a = -exp(0) = -1 (stable decay)
        with torch.no_grad():
            nn.init.zeros_(self.A_log)
            nn.init.zeros_(self.dt_proj.bias)
        
        # B and C projections
        nn.init.xavier_uniform_(self.B_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.B_proj.bias)
        
        nn.init.xavier_uniform_(self.C_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.C_proj.bias)
        
        # D projection
        nn.init.xavier_uniform_(self.D_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.D_proj.bias)
        
        # Gate projection
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.gate_proj.bias)
        
        # Conv initialization
        if self.conv is not None:
            nn.init.xavier_uniform_(self.conv.weight, gain=1.0 / math.sqrt(2))
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)
    
    def _scan_method(self) -> str:
        """Pick scan method based on config/env."""
        return getattr(self, "_scan_method_str", "chunked")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SSM.

        Implements the diagonal state-space recurrence with **proper ZOH
        discretization of both A and B**:

            a    = -exp(A_log)                          # [N], negative
            dt   = softplus(dt_proj(x))                 # [B, T, N]
            A_bar = exp(dt * a)                         # [B, T, N] in (0, 1)
            B_bar = ((A_bar - 1) / a) * B_proj(x)       # ZOH for diagonal A
            h_t  = A_bar_t * h_{t-1} + B_bar_t          # scan
            y_t  = C_t * h_t                            # NO double-C

        The scan uses ``chunked_scan`` (T/chunk Python iterations, vectorized
        within each chunk) by default. For T <= 64 it falls back to the plain
        sequential scan. The Python loop count is T/chunk_size (e.g. 8 for
        T=2048, chunk=256), NOT T.

        Args:
            x: Input tensor [batch, seq_len, hidden]

        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        from xorzen.model.components.ssm_scan import (
            discretize_zoh, select_scan,
        )

        batch_size, seq_len, _ = x.shape

        # Layer norm input
        x_norm = self.ln_input(x)

        # Apply conv if enabled
        if self.conv is not None:
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv)
            x_conv = x_conv.transpose(1, 2)
            x_norm = x_norm + x_conv

        # Compute gates
        gate = self.gate_proj(x_norm)
        gate, input_gate = gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        # Prepare continuous-time SSM inputs
        Bv = self.B_proj(x_norm * torch.sigmoid(input_gate))  # [B, T, N]
        C  = self.C_proj(x_norm)                              # [B, T, N]

        # Input-dependent discretization: ZOH for BOTH A and B (diagonal A).
        dt = F.softplus(self.dt_proj(x_norm))                 # [B, T, N]
        a  = -torch.exp(self.A_log)                           # [N], negative
        A_bar, B_bar = discretize_zoh(a, Bv, dt)              # both [B, T, N]

        # Scan: h_t = A_bar_t * h_{t-1} + B_bar_t
        # Returns states h_t of shape [B, T, N].
        states = select_scan(
            A_bar, B_bar,
            init_state=None,
            method=self._scan_method(),
        )

        # Layer-norm the states (stabilizes gradients over long sequences).
        states = self.ln_state(states)

        # Output: y_t = C_t * h_t  (single multiplication, NO double-C bug).
        ssm_output = C * states                                 # [B, T, N]

        # Project to hidden dimension
        ssm_output = self.D_proj(ssm_output)                   # [B, T, H]

        # Apply gate
        output = ssm_output * gate

        # Dropout
        output = self.dropout(output)

        return output

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parallel-scan forward pass: uses the Blelloch associative scan
        (O(log T) parallel depth) when T > 64, falls back to ``forward``
        otherwise. All SSM parameters receive gradients.

        This is NOT a fake "parallel" label — it actually calls
        ``parallel_scan`` which is the pure-PyTorch Blelloch scan.
        """
        from xorzen.model.components.ssm_scan import (
            discretize_zoh, parallel_scan, sequential_scan,
        )

        batch_size, seq_len, _ = x.shape

        x_norm = self.ln_input(x)
        if self.conv is not None:
            x_conv = x_norm.transpose(1, 2)
            x_conv = self.conv(x_conv)
            x_conv = x_conv.transpose(1, 2)
            x_norm = x_norm + x_conv

        gate = self.gate_proj(x_norm)
        gate, input_gate = gate.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        Bv = self.B_proj(x_norm * torch.sigmoid(input_gate))
        C  = self.C_proj(x_norm)
        dt = F.softplus(self.dt_proj(x_norm))
        a  = -torch.exp(self.A_log)
        A_bar, B_bar = discretize_zoh(a, Bv, dt)

        # Use the real parallel scan for T > 64, sequential otherwise.
        if seq_len > 64:
            states = parallel_scan(A_bar, B_bar)
        else:
            states = sequential_scan(A_bar, B_bar)

        states = self.ln_state(states)
        ssm_output = C * states
        ssm_output = self.D_proj(ssm_output)
        output = ssm_output * gate
        output = self.dropout(output)
        return output
    
    def get_compute_stats(self, seq_len: int, batch_size: int = 1) -> Dict[str, float]:
        """Get compute statistics for this pathway."""
        # FLOPs estimation (sequential version)
        proj_flops = 2 * batch_size * seq_len * self.hidden_dim * self.state_dim  # B and C
        state_update_flops = batch_size * seq_len * self.state_dim * self.state_dim  # A * s
        output_flops = batch_size * seq_len * self.state_dim  # C * s
        
        total_flops = proj_flops + state_update_flops + output_flops
        
        # Add conv flops if used
        if self.conv is not None:
            conv_flops = batch_size * seq_len * self.hidden_dim * self.kernel_size * 2
            total_flops += conv_flops
        
        # Memory estimation
        param_memory = sum(p.numel() for p in self.parameters()) * 4
        activation_memory = batch_size * seq_len * self.state_dim * 4 * 5
        
        return {
            'flops_total': total_flops,
            'flops_per_token': total_flops / (batch_size * seq_len),
            'param_count': sum(p.numel() for p in self.parameters()),
            'param_memory_bytes': param_memory,
            'activation_memory_bytes': activation_memory,
            'state_dim': self.state_dim,
            'complexity': 'O(seq_len * state_dim)',
        }


# ==================== FEED-FORWARD NETWORK ====================

class AdaptiveFFN(nn.Module):
    """
    Adaptive Feed-Forward Network.
    Width can be adjusted based on router decisions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_multiplier: float = 4.0,
        activation: str = "gelu",
        dropout: float = 0.0,
        width_choices: Optional[List[int]] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_ffn_dim = int(hidden_dim * ffn_multiplier)
        self.activation = activation
        self.width_choices = width_choices or [hidden_dim]
        
        # Base FFN (always computed)
        self.fc1 = nn.Linear(hidden_dim, self.base_ffn_dim)
        self.fc2 = nn.Linear(self.base_ffn_dim, hidden_dim)
        
        # Width adapters (for different compute levels)
        self.width_adapters = nn.ModuleDict()
        for width in self.width_choices:
            if width != hidden_dim:
                adapter = nn.Sequential(
                    nn.Linear(hidden_dim, width),
                    self._get_activation(),
                    nn.Linear(width, hidden_dim),
                )
                self.width_adapters[str(width)] = adapter
        
        # Layer norms
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_hidden = nn.LayerNorm(self.base_ffn_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        self._init_weights()
        
        logger.debug("hass", 
                    f"AdaptiveFFN: hidden={hidden_dim}, ffn={self.base_ffn_dim}, "
                    f"widths={len(self.width_adapters)}")
    
    def _get_activation(self):
        """Get activation function."""
        if self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "silu":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def _init_weights(self):
        """Initialize FFN weights."""
        # Base FFN
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0 / math.sqrt(2))
        nn.init.zeros_(self.fc2.bias)
        
        # Width adapters
        for adapter in self.width_adapters.values():
            for layer in adapter:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=1.0 / math.sqrt(2))
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        width_multiplier: Optional[torch.Tensor] = None,
        width_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional width adaptation.
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            width_multiplier: Width multiplier per token [batch, seq_len, 1]
            width_idx: Width index per token [batch, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Layer norm input
        x_norm = self.ln_input(x)
        
        # Base FFN
        hidden = self.fc1(x_norm)
        hidden = self._get_activation()(hidden)
        hidden = self.ln_hidden(hidden)
        hidden = self.ffn_dropout(hidden)
        
        base_output = self.fc2(hidden)
        
        # Apply width adaptation if requested
        if width_multiplier is not None and width_idx is not None:
            adaptive_output = self._apply_width_adaptation(x_norm, width_idx, width_multiplier)
            
            # Blend base and adaptive output based on width multiplier
            # width_multiplier = 1.0 means full width, 0.0 means minimal
            output = base_output * width_multiplier + adaptive_output * (1 - width_multiplier)
        else:
            output = base_output
        
        # Dropout and residual
        output = self.dropout(output)
        
        return output
    
    def _apply_width_adaptation(self, x: torch.Tensor, width_idx: torch.Tensor, width_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply width-specific adaptation with soft mixing during training."""
        batch_size, seq_len, _ = x.shape
        
        # TRAINING: Soft mix all adapters using width_probs
        if self.training and width_probs is not None:
            if len(self.width_adapters) == 0:
                return torch.zeros_like(x)
            
            # Compute all adapter outputs
            adapter_outputs = []
            for width_key in sorted(self.width_adapters.keys()):
                adapted = self.width_adapters[width_key](x)  # [B, S, H]
                adapter_outputs.append(adapted)
            
            # Stack and weight by width_probs
            if adapter_outputs:
                stacked = torch.stack(adapter_outputs, dim=-1)  # [B, S, H, num_adapters]
                # width_probs: [B, S, num_widths] - use only adapter dimensions
                w = width_probs[..., :len(adapter_outputs)].unsqueeze(-2)  # [B, S, 1, num_adapters]
                output = (stacked * w).sum(dim=-1)  # [B, S, H]
            else:
                output = torch.zeros_like(x)
            
            return output
        
        # INFERENCE: Hard selection
        output = torch.zeros_like(x)
        unique_widths = torch.unique(width_idx)
        
        for width_val in unique_widths:
            width_val = width_val.item()
            mask = (width_idx == width_val)
            if not mask.any():
                continue
            
            tokens = x[mask]
            
            if str(width_val) in self.width_adapters:
                adapter = self.width_adapters[str(width_val)]
                adapted = adapter(tokens)
            else:
                adapted = tokens
            
            output[mask] = adapted
        
        return output
    
    def get_compute_stats(
        self, 
        seq_len: int, 
        batch_size: int = 1,
        width_multiplier: float = 1.0
    ) -> Dict[str, float]:
        """Get compute statistics for FFN."""
        # Base FFN FLOPs
        fc1_flops = batch_size * seq_len * self.hidden_dim * self.base_ffn_dim
        fc2_flops = batch_size * seq_len * self.base_ffn_dim * self.hidden_dim
        
        base_flops = fc1_flops + fc2_flops
        
        # Adaptive FLOPs (average)
        adaptive_flops = 0
        for width in self.width_choices:
            if str(width) in self.width_adapters:
                adapter = self.width_adapters[str(width)]
                # Estimate adapter FLOPs
                fc1_adapter = batch_size * seq_len * self.hidden_dim * width
                fc2_adapter = batch_size * seq_len * width * self.hidden_dim
                adaptive_flops += (fc1_adapter + fc2_adapter) / len(self.width_choices)
        
        # Weighted total
        total_flops = base_flops * width_multiplier + adaptive_flops * (1 - width_multiplier)
        
        # Memory
        param_memory = sum(p.numel() for p in self.parameters()) * 4
        
        return {
            'flops_total': total_flops,
            'flops_per_token': total_flops / (batch_size * seq_len),
            'param_count': sum(p.numel() for p in self.parameters()),
            'param_memory_bytes': param_memory,
            'base_ffn_dim': self.base_ffn_dim,
            'width_choices': len(self.width_choices),
        }


# ==================== HASS BLOCK CORE ====================

class HASSBlock(nn.Module):
    """
    Hybrid Attention-Shard Switch Block.
    Combines 3 pathways with adaptive routing.
    """
    
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_size
        
        # Initialize pathways
        self.pathways = nn.ModuleDict({
            'local': LocalAttentionPathway(
                hidden_dim=config.hidden_size,
                num_heads=config.num_attention_heads // 2,  # Half heads for local
                window_size=config.local_window_size,
                dropout=config.dropout,
                causal=True
            ),
            'low_rank': LowRankGlobalPathway(
                hidden_dim=config.hidden_size,
                low_rank_dim=config.low_rank_dim,
                num_heads=4,  # Fixed for low-rank
                dropout=config.dropout
            ),
            'ssm': SSMPathway(
                hidden_dim=config.hidden_size,
                state_dim=config.ssm_state_dim,
                kernel_size=config.ssm_kernel_size,
                dropout=config.dropout,
                use_conv=True
            )
        })
        
        # Pathway gating (learns to combine pathways)
        self.pathway_gate = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3),  # 3 pathways
        )
        
        # Adaptive FFN — v0.4: choose between SlicedFFN (genuine per-token
        # width sparsity via nested slicing) and the legacy AdaptiveFFN
        # (compute-then-blend, NOT genuine sparsity).
        # SlicedFFN is the v0.4 default; set config.use_sliced_ffn=False to
        # preserve the v0.3 behavior for ablation comparison.
        from xorzen.model.components.sliced_ffn import SlicedFFN
        self.use_sliced_ffn = bool(getattr(config, 'use_sliced_ffn', True))
        if self.use_sliced_ffn:
            # max_width = hidden * ffn_multiplier (matches base_ffn_dim of AdaptiveFFN)
            max_width = int(config.hidden_size * 4.0)
            # width_choices are nested subsets of max_width. SlicedFFN requires
            # all widths to be <= max_width; the config's width_choices may
            # already include max_width. If not, we add it.
            wc = list(config.width_choices)
            if max_width not in wc:
                wc = sorted(set(wc + [max_width]))
            self.ffn = SlicedFFN(
                hidden_dim=config.hidden_size,
                max_width=max_width,
                activation=config.hidden_act,
                dropout=config.dropout,
                width_choices=wc,
            )
            logger.info("hass",
                       f"HASSBlock {layer_idx}: using SlicedFFN "
                       f"(genuine width sparsity), widths={wc}")
        else:
            self.ffn = AdaptiveFFN(
                hidden_dim=config.hidden_size,
                ffn_multiplier=4.0,
                activation=config.hidden_act,
                dropout=config.dropout,
                width_choices=config.width_choices
            )
            logger.info("hass",
                       f"HASSBlock {layer_idx}: using legacy AdaptiveFFN "
                       f"(compute-then-blend, no genuine sparsity)")
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Initialize
        self._init_weights()
        
        logger.info("hass", 
                   f"HASSBlock {layer_idx}: hidden={config.hidden_size}, "
                   f"local_window={config.local_window_size}, "
                   f"low_rank={config.low_rank_dim}, ssm={config.ssm_state_dim}")
    
    def _init_weights(self):
        """Initialize block weights."""
        # Pathway gate
        for layer in self.pathway_gate:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.0 / math.sqrt(2))
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        routing_decision: Optional[RoutingDecision] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        compute_all_pathways: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through HASS block — **genuine sparse pathway dispatch**.

        The previous implementation always computed all 3 pathways during
        training (claiming "gradient flow") and blended them with soft
        weights. That is not conditional computation — it is dense
        computation with routing metadata.

        The new implementation:

          - Reads ``self.config.pathway_top_k`` (default 2).
          - At INFERENCE: only the top-k pathways per token are invoked.
            Pathways that no token selects are NOT called at all.
            (Implemented via ``sparse_pathway_dispatch``.)
          - At TRAINING: the straight-through estimator is used — the
            forward uses the hard top-k mask (so only k pathways' forward
            functions are called per token), but the backward pass sees
            the soft probabilities. This is the classic Bengio STE and is
            differentiable.

        The pathway_call_counter (if attached to self by the parent model)
        records how many times each pathway's forward was actually invoked,
        so sparsity can be verified at the execution level.

        Args:
            x: [B, T, H]
            routing_decision: RoutingDecision (must provide path_probs).
            attention_mask, position_bias: passed through to local pathway.
            compute_all_pathways: if True, ignore routing and compute all 3
                (backward-compat / analysis only).

        Returns:
            [B, T, H]
        """
        from xorzen.model.components.sparse_dispatch import (
            sparse_pathway_dispatch, topk_pathway_mask,
        )

        batch_size, seq_len, _ = x.shape
        x_attn = self.ln1(x)

        # ===== ATTENTION PHASE =====
        if routing_decision is None or compute_all_pathways:
            # No routing — compute all 3 pathways and combine with learned gate.
            pathway_outputs = []
            local_out = self.pathways['local'](x_attn, attention_mask, position_bias)
            pathway_outputs.append(local_out)
            low_rank_out = self.pathways['low_rank'](x_attn)
            pathway_outputs.append(low_rank_out)
            ssm_out = self.pathways['ssm'].forward_parallel(x_attn)
            pathway_outputs.append(ssm_out)

            if routing_decision is None:
                gate_logits = self.pathway_gate(x_attn)
                gate_weights = F.softmax(gate_logits, dim=-1)
                combined = sum(
                    out * gate_weights[..., i:i+1]
                    for i, out in enumerate(pathway_outputs)
                )
            else:
                path_probs = routing_decision.path_probs
                combined = sum(
                    out * path_probs[..., i:i+1]
                    for i, out in enumerate(pathway_outputs)
                )
        else:
            # GENUINE SPARSE DISPATCH.
            path_probs = routing_decision.path_probs  # [B, T, 3]
            top_k = getattr(self.config, 'pathway_top_k', 2)

            if top_k >= 3:
                # All pathways selected — no sparsity, but use the new path
                # so the code is exercised. Equivalent to the old training
                # branch (compute all, weighted sum).
                local_out = self.pathways['local'](x_attn, attention_mask, position_bias)
                low_rank_out = self.pathways['low_rank'](x_attn)
                ssm_out = self.pathways['ssm'].forward_parallel(x_attn)
                combined = (
                    local_out    * path_probs[..., 0:1] +
                    low_rank_out * path_probs[..., 1:2] +
                    ssm_out      * path_probs[..., 2:3]
                )
                # Record call counts
                if hasattr(self, '_pathway_call_counter'):
                    self._pathway_call_counter['local'] = self._pathway_call_counter.get('local', 0) + 1
                    self._pathway_call_counter['low_rank'] = self._pathway_call_counter.get('low_rank', 0) + 1
                    self._pathway_call_counter['ssm'] = self._pathway_call_counter.get('ssm', 0) + 1
            else:
                # Top-k sparse dispatch.
                # We need per-pathway forward functions that accept a sliced
                # tensor [n, H] and return [n, H]. The local pathway takes
                # extra args (attention_mask, position_bias); but those are
                # batch-level so we can't easily slice them per-token. To keep
                # the dispatch correct, we pass the FULL x_attn to local
                # attention (it is the only pathway with mask args) but only
                # for the tokens that selected it. We do this by slicing the
                # input and padding the attention_mask.
                #
                # For simplicity and correctness, we use the
                # ``sparse_pathway_dispatch`` helper which handles the
                # token-level dispatch. The local pathway's extra args are
                # handled by wrapping it in a closure.
                extra_args = {}
                local_extra = ()
                if attention_mask is not None:
                    # Local attention uses attention_mask of shape [B, T] or [B, T, T].
                    # For per-token slicing we'd need to slice this too; for now,
                    # pass None (the local pathway will use its causal mask only).
                    # This is a known limitation when sparsifying the local pathway.
                    local_extra = (None, position_bias)
                extra_args['local'] = local_extra

                pathway_fns = {
                    'local':    lambda x_in, *a: self.pathways['local'](x_in, *(a if a else (None, None))),
                    'low_rank': lambda x_in, *a: self.pathways['low_rank'](x_in),
                    'ssm':      lambda x_in, *a: self.pathways['ssm'].forward_parallel(x_in),
                }
                # sparse_pathway_dispatch expects x as [B, T, H] and slices
                # it internally. But the local/low_rank/ssm pathways all
                # expect [B', T', H] (3D). So we need to reshape the slice
                # [n, H] back to [1, n, H] before calling each pathway,
                # then flatten the output back to [n, H].
                def _wrap(fn):
                    def wrapped(x_slice):
                        # x_slice: [n, H] -> [1, n, H] -> fn -> [1, n, H] -> [n, H]
                        x3d = x_slice.unsqueeze(0)
                        y3d = fn(x3d)
                        return y3d.squeeze(0)
                    return wrapped

                wrapped_fns = {k: _wrap(fn) for k, fn in pathway_fns.items()}
                # sparse_pathway_dispatch doesn't take extra_args in this shape;
                # rewrite to use wrapped_fns which ignore extra args.
                combined_flat, call_counts = sparse_pathway_dispatch(
                    x_attn,
                    path_probs,
                    wrapped_fns,
                    ['local', 'low_rank', 'ssm'],
                    top_k,
                    training=self.training,
                    extra_args=None,
                    pathway_call_counter=getattr(self, '_pathway_call_counter', None),
                )
                # combined_flat is [B, T, H]
                combined = combined_flat

        x = x + self.dropout(combined)

        # ===== FFN PHASE =====
        x_ffn = self.ln2(x)

        if routing_decision is not None:
            if self.use_sliced_ffn:
                # SlicedFFN API: takes width_idx (inference) or width_probs (training).
                # At training, SlicedFFN uses STE — forward uses hard argmax width
                # per token (genuine sparsity), backward sees soft probs.
                # At inference, per-token width grouping is used.
                if self.training:
                    ffn_out = self.ffn(
                        x_ffn,
                        width_probs=routing_decision.width_probs,
                    )
                else:
                    ffn_out = self.ffn(
                        x_ffn,
                        width_idx=routing_decision.width_idx,
                    )
            else:
                # Legacy AdaptiveFFN API
                width_multiplier = routing_decision.width_multiplier
                width_idx = routing_decision.width_idx
                ffn_out = self.ffn(x_ffn, width_multiplier, width_idx)
        else:
            # No routing decision — use max width (dense).
            if self.use_sliced_ffn:
                ffn_out = self.ffn(x_ffn)  # defaults to max_width
            else:
                ffn_out = self.ffn(x_ffn)

        x = x + self.dropout(ffn_out)
        return x
    
    def forward_with_depth(
        self,
        x: torch.Tensor,
        depth_mask: torch.Tensor,
        routing_decision: Optional[RoutingDecision] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with depth masking.
        Only processes tokens where depth_mask indicates this layer should be active.
        
        Args:
            x: Input tensor [batch, seq_len, hidden]
            depth_mask: Depth mask [batch, seq_len] indicating active tokens
            routing_decision: Routing decision
            attention_mask: Attention mask
            
        Returns:
            Output tensor [batch, seq_len, hidden]
        """
        batch_size, seq_len, _ = x.shape
        
        # Check if any tokens are active for this layer
        if depth_mask.sum() == 0:
            # No active tokens, return input unchanged
            return x
        
        # Create mask for active tokens
        active_mask = depth_mask.bool()  # [batch, seq_len]
        
        # Extract active tokens
        active_x = x[active_mask]  # [num_active, hidden]
        
        # Reshape for processing (treat as batch)
        active_x_reshaped = active_x.unsqueeze(0)  # [1, num_active, hidden]
        
        # Process active tokens
        if routing_decision is not None:
            # Extract routing for active tokens
            active_path_probs = routing_decision.path_probs[active_mask]  # [num_active, 3]
            active_width_mult = routing_decision.width_multiplier[active_mask]  # [num_active, 1]
            active_width_idx = routing_decision.width_idx[active_mask]  # [num_active]
            
            # Create mini routing decision
            mini_decision = RoutingDecision(
                depth_logits=torch.zeros(1, active_x_reshaped.shape[1], 1, device=x.device),
                depth_probs=torch.ones(1, active_x_reshaped.shape[1], 1, device=x.device),
                depth_mask=torch.ones(1, active_x_reshaped.shape[1], 1, device=x.device, dtype=torch.bool),
                width_logits=torch.zeros(1, active_x_reshaped.shape[1], len(self.config.width_choices), device=x.device),
                width_probs=F.one_hot(active_width_idx, num_classes=len(self.config.width_choices)).float().unsqueeze(0),
                width_idx=active_width_idx.unsqueeze(0),
                width_multiplier=active_width_mult.unsqueeze(0),
                path_logits=torch.zeros(1, active_x_reshaped.shape[1], 3, device=x.device),
                path_probs=active_path_probs.unsqueeze(0),
                expert_logits=torch.zeros(1, active_x_reshaped.shape[1], self.config.expert_count, device=x.device),
                expert_probs=torch.zeros(1, active_x_reshaped.shape[1], self.config.expert_count, device=x.device),
                expert_indices=torch.zeros(1, active_x_reshaped.shape[1], self.config.top_k_experts, device=x.device, dtype=torch.long),
                expert_weights=torch.zeros(1, active_x_reshaped.shape[1], self.config.top_k_experts, device=x.device),
                complexity=torch.ones(1, active_x_reshaped.shape[1], 1, device=x.device),
                uncertainty=torch.zeros(1, active_x_reshaped.shape[1], 1, device=x.device),
            )
            
            processed = self.forward(
                active_x_reshaped, 
                mini_decision,
                attention_mask=None,  # No mask for extracted tokens
                compute_all_pathways=False
            )
        else:
            # No routing, compute all pathways
            processed = self.forward(
                active_x_reshaped,
                routing_decision=None,
                attention_mask=None,
                compute_all_pathways=True
            )
        
        # Extract processed tokens
        processed_tokens = processed.squeeze(0)  # [num_active, hidden]
        
        # Create output tensor
        output = x.clone()
        output[active_mask] = processed_tokens
        
        return output
    
    def get_compute_stats(
        self,
        seq_len: int,
        batch_size: int = 1,
        routing_decision: Optional[RoutingDecision] = None,
        depth_active_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get compute statistics for this block.
        
        Args:
            seq_len: Sequence length
            batch_size: Batch size
            routing_decision: Routing decision for accurate stats
            depth_active_ratio: Ratio of tokens active for this layer
            
        Returns:
            Dictionary of compute statistics
        """
        # Adjust for depth routing
        active_tokens = int(batch_size * seq_len * depth_active_ratio)
        
        # Pathway statistics
        pathway_stats = {}
        total_flops = 0
        total_params = 0
        
        for name, pathway in self.pathways.items():
            stats = pathway.get_compute_stats(seq_len, batch_size)
            pathway_stats[name] = stats
            
            # Adjust for active tokens and pathway probability
            if routing_decision is not None:
                # Estimate pathway usage from routing decision
                if hasattr(routing_decision, 'path_probs'):
                    pathway_prob = routing_decision.path_probs[..., 
                        list(self.pathways.keys()).index(name)].mean().item()
                else:
                    pathway_prob = 1.0 / len(self.pathways)
            else:
                pathway_prob = 1.0  # All pathways computed
            
            pathway_flops = stats['flops_total'] * pathway_prob * depth_active_ratio
            total_flops += pathway_flops
            
            total_params += stats['param_count']
        
        # FFN statistics
        if routing_decision is not None:
            width_mult = routing_decision.width_multiplier.mean().item()
        else:
            width_mult = 1.0
        
        ffn_stats = self.ffn.get_compute_stats(seq_len, batch_size, width_mult)
        ffn_flops = ffn_stats['flops_total'] * depth_active_ratio
        total_flops += ffn_flops
        total_params += ffn_stats['param_count']
        
        # Block overhead
        overhead_flops = batch_size * seq_len * self.hidden_dim * 10  # Layer norms, etc.
        total_flops += overhead_flops
        
        # Memory
        param_memory = total_params * 4
        activation_memory = batch_size * seq_len * self.hidden_dim * 4 * 20  # Rough estimate
        
        return {
            'total_flops': total_flops,
            'flops_per_active_token': total_flops / max(active_tokens, 1),
            'flops_per_all_token': total_flops / (batch_size * seq_len),
            'total_params': total_params,
            'param_memory_bytes': param_memory,
            'activation_memory_bytes': activation_memory,
            'pathway_stats': pathway_stats,
            'ffn_stats': ffn_stats,
            'depth_active_ratio': depth_active_ratio,
            'active_tokens': active_tokens,
            'width_multiplier': width_mult,
            'efficiency_gain': 1.0 / depth_active_ratio if depth_active_ratio > 0 else 1.0,
        }
    
    def analyze_pathway_usage(
        self,
        routing_decision: RoutingDecision
    ) -> Dict[str, float]:
        """Analyze pathway usage from routing decision."""
        path_probs = routing_decision.path_probs  # [batch, seq_len, 3]
        
        # Average probabilities
        local_prob = path_probs[..., 0].mean().item()
        low_rank_prob = path_probs[..., 1].mean().item()
        ssm_prob = path_probs[..., 2].mean().item()
        
        # Entropy (diversity of pathway usage)
        entropy = -torch.sum(
            path_probs * torch.log(path_probs + 1e-12), 
            dim=-1
        ).mean().item()
        
        # Dominant pathway
        dominant = torch.argmax(path_probs.mean(dim=(0, 1))).item()
        dominant_names = ['local', 'low_rank', 'ssm']
        
        return {
            'local_prob': local_prob,
            'low_rank_prob': low_rank_prob,
            'ssm_prob': ssm_prob,
            'pathway_entropy': entropy,
            'dominant_pathway': dominant_names[dominant],
            'dominant_prob': path_probs.mean(dim=(0, 1))[dominant].item(),
        }


# ==================== HASS BLOCK MANAGER ====================

class HASSBlockManager:
    """
    Manages multiple HASS blocks with depth routing.
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.blocks = nn.ModuleList([
            HASSBlock(config, layer_idx=i)
            for i in range(config.max_depth)
        ])
        
        # Statistics
        self.compute_history = []
        self.pathway_history = []
        
        logger.info("hass", f"HASSBlockManager: {len(self.blocks)} blocks")
    
    def forward(
        self,
        x: torch.Tensor,
        routing_decision: RoutingDecision,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        collect_stats: bool = False,
    ) -> torch.Tensor:
        """
        Forward through blocks with depth routing.
        
        Args:
            x: Input tensor
            routing_decision: Routing decision with depth_mask
            attention_mask: Attention mask
            position_bias: Position bias
            collect_stats: Whether to collect compute statistics
            
        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape
        
        # Process through blocks based on depth routing
        current = x
        
        for i, block in enumerate(self.blocks):
            # Get depth mask for this layer
            if i < routing_decision.depth_mask.shape[-1]:
                depth_mask = routing_decision.depth_mask[..., i]  # [batch, seq_len]
            else:
                # If more blocks than depth decisions, all tokens go through
                depth_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)
            
            # Skip if no tokens active for this layer
            if depth_mask.sum() == 0:
                continue
            
            # Forward through block with depth masking
            current = block.forward_with_depth(
                current, depth_mask, routing_decision, attention_mask
            )
            
            # Collect statistics if requested
            if collect_stats:
                depth_active_ratio = depth_mask.float().mean().item()
                stats = block.get_compute_stats(
                    seq_len, batch_size, routing_decision, depth_active_ratio
                )
                stats['layer'] = i
                stats['depth_active_ratio'] = depth_active_ratio
                
                self.compute_history.append(stats)
                
                # Pathway usage
                pathway_stats = block.analyze_pathway_usage(routing_decision)
                pathway_stats['layer'] = i
                self.pathway_history.append(pathway_stats)
        
        return current
    
    def get_compute_summary(self) -> Dict[str, Any]:
        """Get compute summary across all blocks."""
        if not self.compute_history:
            return {}
        
        # Aggregate statistics
        total_flops = sum(stats['total_flops'] for stats in self.compute_history)
        total_params = sum(stats['total_params'] for stats in self.compute_history)
        
        # Average depth active ratio
        avg_depth_active = np.mean([stats['depth_active_ratio'] 
                                   for stats in self.compute_history])
        
        # Pathway usage
        if self.pathway_history:
            avg_local = np.mean([stats['local_prob'] for stats in self.pathway_history])
            avg_low_rank = np.mean([stats['low_rank_prob'] for stats in self.pathway_history])
            avg_ssm = np.mean([stats['ssm_prob'] for stats in self.pathway_history])
            avg_entropy = np.mean([stats['pathway_entropy'] for stats in self.pathway_history])
        else:
            avg_local = avg_low_rank = avg_ssm = avg_entropy = 0.0
        
        return {
            'total_blocks': len(self.blocks),
            'total_flops': total_flops,
            'total_params': total_params,
            'avg_depth_active_ratio': avg_depth_active,
            'pathway_usage': {
                'local': avg_local,
                'low_rank': avg_low_rank,
                'ssm': avg_ssm,
                'entropy': avg_entropy,
            },
            'efficiency_gain': 1.0 / avg_depth_active if avg_depth_active > 0 else 1.0,
            'compute_history': self.compute_history[-10:],  # Last 10
            'pathway_history': self.pathway_history[-10:],
        }
    
    def reset_statistics(self):
        """Reset collected statistics."""
        self.compute_history.clear()
        self.pathway_history.clear()


# ==================== TESTING ====================

__all__ = [
    'LocalAttentionPathway',
    'LowRankGlobalPathway',
    'SSMPathway',
    'AdaptiveFFN',
    'HASSBlock',
    'HASSBlockManager',
]
