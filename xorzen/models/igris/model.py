
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from xorzen.model.base import BaseModel, ModelOutput
from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from .config import IGRISConfig

logger = get_logger()

# --- AGENTIC BIT UPGRADES ---

class InternalLatentCoT(nn.Module):
    """
    Bit Upgrade: Maintains a hidden 'reasoning state' that evolves.
    This allows the model to 'think' in latent space across tokens.
    """
    def __init__(self, d_model, cot_dim):
        super().__init__()
        self.cot_dim = cot_dim
        self.update_gate = nn.Linear(d_model + cot_dim, cot_dim)
        self.transform = nn.Sequential(
            nn.Linear(cot_dim, cot_dim),
            nn.GELU(),
            nn.Linear(cot_dim, cot_dim)
        )
    
    def forward(self, x, prev_cot=None):
        # x: [B, L, D]
        B, L, D = x.shape
        if prev_cot is None:
            prev_cot = torch.zeros(B, L, self.cot_dim, device=x.device)
        
        # Simple GRU-like update for latent reasoning
        combined = torch.cat([x, prev_cot], dim=-1)
        gate = torch.sigmoid(self.update_gate(combined))
        thought = self.transform(prev_cot)
        
        new_cot = (1 - gate) * prev_cot + gate * thought
        return new_cot

class ActionHead(nn.Module):
    """
    Bit Upgrade: Natively predicts agentic actions.
    0: Output Token, 1: Search Memory, 2: Use Tool, 3: Internal Reasoning
    """
    def __init__(self, d_model, num_actions):
        super().__init__()
        self.proj = nn.Linear(d_model, num_actions)
    
    def forward(self, x):
        return self.proj(x)

class CritiqueModule(nn.Module):
    """
    Bit Upgrade: Identifies 'uncertainty' or 'errors' in the current latent state.
    Used to drive further recursion if the state is low quality.
    """
    def __init__(self, d_model):
        super().__init__()
        self.quality_eval = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.quality_eval(x)

# --- CORE ARCHITECTURE (SSM + GLA) ---

class FlashSSM(nn.Module):
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3)
        self.x_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        u = self.in_proj(x)
        x_ssm, z = u.chunk(2, dim=-1)
        x_ssm = self.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2)
        y = x_ssm * torch.sigmoid(self.x_proj(x_ssm))
        return self.out_proj(y * F.silu(z))

class GatedLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v, g = self.q_proj(x), self.k_proj(x), self.v_proj(x), torch.sigmoid(self.g_proj(x))
        # Simplified GLA for demonstration
        kv = torch.einsum('bld,ble->blde', F.relu(k), v)
        kv_cum = torch.cumsum(kv, dim=1)
        y = torch.einsum('bld,blde->ble', F.relu(q), kv_cum)
        return self.out_proj(y * g)

class RecursiveRouter(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.halt_proj = nn.Linear(d_model, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.halt_proj(x))

class IGRISBlock(nn.Module):
    def __init__(self, config: IGRISConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.ssm = FlashSSM(config.hidden_size)
        self.attn = GatedLinearAttention(config.hidden_size, config.num_attention_heads)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.mix_gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x_mix = (self.ssm(x) * self.mix_gate) + (self.attn(x) * (1 - self.mix_gate))
        x = residual + x_mix
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x

# --- MAIN AGENTIC MODEL ---

class IGRISModel(BaseModel):
    def __init__(self, config: IGRISConfig):
        super().__init__(config)
        self.config = config
        
        logger.info("igris", f"Initializing Agentic IGRIS ({config.hidden_size} dim, {config.num_layers} layers)")
        
        # 1. Base Components
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.blocks = nn.ModuleList([IGRISBlock(config) for _ in range(config.num_layers)])
        
        # 2. Agentic Components (The Bit Upgrades)
        if config.use_latent_cot:
            self.latent_cot = InternalLatentCoT(config.hidden_size, config.latent_cot_dim)
            self.cot_to_hidden = nn.Linear(config.latent_cot_dim, config.hidden_size)
            
        self.action_head = ActionHead(config.hidden_size, config.num_action_slots)
        self.critique = CritiqueModule(config.hidden_size)
        self.router = RecursiveRouter(config.hidden_size)
        
        # 3. Persistent Memory
        self.memory_vault = nn.Parameter(torch.randn(1, config.memory_slots, config.hidden_size))
        
        # 4. Output
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        B, L = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding(torch.arange(L, device=device))
        
        # Initialize Latent CoT (Monologue)
        latent_state = None
        if self.config.use_latent_cot:
            latent_state = self.latent_cot(x)
            x = x + self.cot_to_hidden(latent_state)
        
        total_recurrence_steps = 0
        ponder_cost = 0.0
        
        # Layer Processing with Agentic Recursion
        for block in self.blocks:
            x = block(x)
            
            if self.config.recurrence_depth > 1:
                for _ in range(self.config.recurrence_depth - 1):
                    # 1. Decide Halt Probability
                    halt_prob = self.router(x) # [B, L, 1]
                    
                    # 2. Evaluate State Quality (Self-Critique)
                    quality = self.critique(x)
                    
                    # If quality is low, we ignore the halt_prob and keep thinking
                    agentic_halt = halt_prob * quality
                    ponder_cost = ponder_cost + agentic_halt.mean()
                    
                    # 3. Recursive thought step
                    new_x = block(x)
                    x = agentic_halt * x + (1 - agentic_halt) * new_x
                    
                    # 4. Update internal monologue during thinking
                    if self.config.use_latent_cot:
                        latent_state = self.latent_cot(x, latent_state)
                        x = x + 0.1 * self.cot_to_hidden(latent_state) # Residual thought injection
                    
                    if agentic_halt.mean() > 0.8:
                        break
                    total_recurrence_steps += 1
        
        # Map to actions
        actions = self.action_head(x)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            loss += 0.01 * ponder_cost # Penalize over-thinking
        
        if not return_dict:
            return (logits, actions, loss) if loss is not None else (logits, actions)
            
        return ModelOutput(
            logits=logits,
            loss=loss,
            expert_stats={
                "recurrence_steps": total_recurrence_steps,
                "agentic_actions": actions,
                "latent_thought_dim": self.config.latent_cot_dim
            }
        )
