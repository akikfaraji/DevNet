"""Diagnose: print actual router logits to understand path 1 = 0%."""
import sys, os
sys.path.insert(0, "/home/z/my-project/xorzen_dev")
import torch
import numpy as np

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase

torch.manual_seed(42)
np.random.seed(42)

H = 64
cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
cfg.update(
    vocab_size=128, context_length=64, hidden_size=H,
    num_layers=4, num_attention_heads=4, max_depth=4, min_depth=1,
    width_choices=(H // 2, H), expert_count=4, top_k_experts=2,
    router_hidden_dim=H // 4, dropout=0.0, pad_token_id=0,
    load_balancing_weight=0.001, shard_experts=False,
    pathway_top_k=2, gradient_checkpointing=False,
    use_sliced_ffn=True, width_div_weight=0.1,
    path_div_weight=0.2, unify_load_balance=True, cost_aware_routing=True,
)
model = zeroBase(config=cfg, test_mode=True)
model.train()

rng = np.random.RandomState(42)
sequences = []
for i in range(8):
    offset = rng.randint(1, cfg.vocab_size)
    start = rng.randint(0, cfg.vocab_size)
    seq = [(start + offset * t) % cfg.vocab_size for t in range(cfg.context_length)]
    sequences.append(seq)
sequences = torch.tensor(sequences, dtype=torch.long)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01
)
for step in range(200):
    optimizer.zero_grad()
    out = model(input_ids=sequences, labels=sequences, output_routing_info=True)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# Eval mode — inspect raw logits
model.eval()
with torch.no_grad():
    # Run router manually to get raw logits
    hidden_states = model.token_embedding(sequences)
    pos_ids = torch.arange(sequences.size(1)).unsqueeze(0).expand_as(sequences)
    hidden_states = hidden_states + model.position_embedding(pos_ids)
    cot_total_dim = cfg.cot_dim * cfg.cot_components
    cot_vector_seq = torch.zeros(sequences.size(0), sequences.size(1), cot_total_dim)
    rd = model.router(hidden_states, cot_vector_seq, training=False)

print("Path logits (mean over batch,seq):", rd.path_logits.mean(dim=(0,1)).tolist())
print("Path logits (std  over batch,seq):", rd.path_logits.std(dim=(0,1)).tolist())
print("Path probs  (mean over batch,seq):", rd.path_probs.mean(dim=(0,1)).tolist())
print()
print("Width logits (mean):", rd.width_logits.mean(dim=(0,1)).tolist())
print("Width probs  (mean):", rd.width_probs.mean(dim=(0,1)).tolist())
print()
print("Depth logits (mean over batch,seq):", rd.depth_logits.mean(dim=(0,1)).tolist())
print()
print("Expert logits (mean over batch,seq):", rd.expert_logits.mean(dim=(0,1)).tolist())
