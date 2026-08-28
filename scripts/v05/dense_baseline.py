"""
Dense baseline transformer for the 10M-scale comparison.

A vanilla decoder-only transformer with:
- Token + position embeddings
- L transformer blocks (causal self-attention + MLP FFN)
- Layer norms
- Tied LM head

This is the SCIENTIFIC BASELINE against which Xorzen's adaptive
routing is compared. Same parameter budget, same training data,
same optimizer, same training tokens.

The question we answer:
    «Does Xorzen's adaptive routing achieve better quality per unit
     of compute than a dense model of similar total parameter count?»
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DenseConfig:
    vocab_size: int = 10000
    context_length: int = 512
    hidden_size: int = 192
    num_layers: int = 6
    num_heads: int = 8
    ffn_hidden: int = 1152  # 6x hidden
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0


class DenseBlock(nn.Module):
    """Standard pre-norm transformer block."""
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        H = cfg.hidden_size
        self.ln1 = nn.LayerNorm(H)
        self.attn = nn.MultiheadAttention(
            H, cfg.num_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(H)
        self.ffn = nn.Sequential(
            nn.Linear(H, cfg.ffn_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_hidden, H),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        # Pre-norm FFN
        h = self.ln2(x)
        f = self.ffn(h)
        x = x + f
        return x


class DenseTransformer(nn.Module):
    """Vanilla decoder-only transformer — the scientific baseline."""
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.token_embedding = nn.Embedding(cfg.vocab_size, H, padding_idx=cfg.pad_token_id)
        self.position_embedding = nn.Embedding(cfg.context_length, H)
        self.blocks = nn.ModuleList([DenseBlock(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(H)
        # Tied LM head (weight shared with token_embedding)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        # Causal mask buffer
        mask = torch.triu(torch.ones(cfg.context_length, cfg.context_length) * -1e9, diagonal=1)
        self.register_buffer('causal_mask', mask)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        B, T = input_ids.shape
        device = input_ids.device
        # Embeddings
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
        # Causal mask for nn.MultiheadAttention (bool mask: True = masked)
        attn_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )
        # Blocks
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        # Loss
        loss = None
        lm_loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.cfg.pad_token_id,
                reduction='mean',
            )
            lm_loss = loss.detach()
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits, loss=loss, lm_loss=lm_loss)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def build_dense_baseline_to_match(target_params: int, vocab_size: int = 10000,
                                   context_length: int = 512) -> DenseTransformer:
    """Build a dense transformer with approximately target_params parameters.

    Tries hidden_size in {128, 192, 256, 320, 384} and num_layers in {4, 6, 8}
    to find the closest match.
    """
    best = None
    best_diff = float('inf')
    for H in [128, 160, 192, 224, 256, 288, 320, 352, 384]:
        for L in [4, 5, 6, 7, 8]:
            for ffn_mult in [4, 6, 8]:
                cfg = DenseConfig(
                    vocab_size=vocab_size,
                    context_length=context_length,
                    hidden_size=H,
                    num_layers=L,
                    num_heads=max(4, H // 64),
                    ffn_hidden=H * ffn_mult,
                )
                # Estimate param count without building
                embed = vocab_size * H + context_length * H
                per_layer = 4 * H * H + 2 * H * cfg.ffn_hidden + 4 * H
                ln = 2 * H * (L + 1)
                total = embed + L * per_layer + ln
                diff = abs(total - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best = (cfg, total)
    cfg, total = best
    print(f"Dense baseline: H={cfg.hidden_size}, L={cfg.num_layers}, "
          f"ffn={cfg.ffn_hidden}, heads={cfg.num_heads}, "
          f"est_params={total:,}")
    return DenseTransformer(cfg)


if __name__ == "__main__":
    # Quick sanity check
    cfg = DenseConfig()
    model = DenseTransformer(cfg)
    n = model.num_parameters()
    print(f"Dense default config: {n:,} params")
    # Test forward
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    out = model(ids, labels=ids)
    print(f"forward OK: logits {out.logits.shape}, loss {out.loss.item():.4f}")
    out.loss.backward()
    print("backward OK")
