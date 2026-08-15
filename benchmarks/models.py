"""
Baseline Model Implementations for Benchmarking
Implements standard architectures to compare against XORZENX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ==================== Vanilla Transformer ====================

class VanillaTransformer(nn.Module):
    """
    Standard GPT-style Transformer for baseline comparison.
    Based on Vaswani et al. 2017 with decoder-only architecture.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        x = self.dropout(token_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            # Handle both 1D (classification) and 2D (sequence) labels
            if labels.dim() == 1:
                # Classification: use last position logits
                loss = F.cross_entropy(logits[:, -1, :], labels)
            else:
                # Sequence prediction: shift for next-token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
            }
        return logits, loss


class TransformerBlock(nn.Module):
    """Standard Transformer block with attention + FFN."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, hidden_size * 4, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for t in qkv]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ==================== MoE Transformer ====================

class MoETransformer(nn.Module):
    """
    Transformer with Mixture of Experts.
    Based on Switch Transformer (Fedus et al. 2021).
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        top_k: int = 2,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # MoE Transformer blocks
        self.blocks = nn.ModuleList([
            MoETransformerBlock(hidden_size, num_heads, num_experts, top_k, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)
        x = self.dropout(token_emb + pos_emb)
        
        # MoE blocks
        total_aux_loss = 0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss
        loss = None
        if labels is not None:
            # Handle both 1D (classification) and 2D (sequence) labels
            if labels.dim() == 1:
                # Classification: use last position logits
                lm_loss = F.cross_entropy(logits[:, -1, :], labels)
            else:
                # Sequence prediction: shift for next-token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            loss = lm_loss + 0.01 * total_aux_loss  # Add MoE auxiliary loss
        
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "aux_loss": total_aux_loss,
            }
        return logits, loss


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.moe = SparseMoE(hidden_size, num_experts, top_k, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss


class SparseMoE(nn.Module):
    """Sparse Mixture of Experts with top-k routing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        dropout: float,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            FeedForward(hidden_size, hidden_size * 4, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = x.shape
        
        # Flatten
        x_flat = x.view(-1, hidden_size)
        
        # Router
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_mask = top_k_indices[:, k]
            for expert_id in range(self.num_experts):
                mask = (expert_mask == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_output * top_k_probs[mask, k:k+1]
        
        # Reshape
        output = output.view(batch_size, seq_len, hidden_size)
        
        # Load balancing loss
        aux_loss = self._compute_load_balance_loss(router_probs)
        
        return output, aux_loss
    
    def _compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Encourage balanced expert usage."""
        expert_usage = router_probs.mean(dim=0)
        target = 1.0 / self.num_experts
        loss = ((expert_usage - target) ** 2).sum()
        return loss


# ==================== Simplified Mamba ====================

class MambaLM(nn.Module):
    """
    Simplified Mamba-style SSM language model.
    Note: This is a simplified version for benchmarking purposes.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layer)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Handle both 1D (classification) and 2D (sequence) labels
            if labels.dim() == 1:
                # Classification: use last position logits
                loss = F.cross_entropy(logits[:, -1, :], labels)
            else:
                # Sequence prediction: shift for next-token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        if return_dict:
            return {"logits": logits, "loss": loss}
        return logits, loss


class SimplifiedMambaBlock(nn.Module):
    """Simplified Mamba block (not full implementation)."""
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv-1)
        self.ssm = nn.GRU(d_inner, d_state, batch_first=True)  # Simplified SSM
        self.out_proj = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        # Project
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv
        x = self.conv1d(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x = F.silu(x)
        
        # SSM (simplified with GRU)
        x, _ = self.ssm(x)
        
        # Gate and project
        x = x * F.silu(z)
        x = self.out_proj(x)
        
        return residual + x


# ==================== Jamba-Style Hybrid ====================

class JambaStyle(nn.Module):
    """
    Jamba-style hybrid model (Attention + SSM + MoE).
    Simplified implementation for benchmarking.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        ssm_ratio: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Hybrid blocks (alternating attention and SSM)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:  # Attention layers
                self.layers.append(TransformerBlock(hidden_size, num_heads, 0.1))
            else:  # SSM layers with MoE
                self.layers.append(SimplifiedMambaBlock(hidden_size, 16, 4, 2))
        
        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Handle both 1D (classification) and 2D (sequence) labels
            if labels.dim() == 1:
                # Classification: use last position logits
                loss = F.cross_entropy(logits[:, -1, :], labels)
            else:
                # Sequence prediction: shift for next-token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
        
        if return_dict:
            return {"logits": logits, "loss": loss}
        return logits, loss