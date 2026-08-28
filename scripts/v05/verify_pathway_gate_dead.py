"""
Verify whether HASSBlock.pathway_gate receives useful gradients during
normal model training (routing_decision always passed).

If pathway_gate.grad is None or all-zero after a backward pass, the
parameter is dead code and can be safely removed.
"""
import sys
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
    vocab_size=128, context_length=32, hidden_size=H,
    num_layers=3, num_attention_heads=4, max_depth=3, min_depth=1,
    width_choices=(H // 2, H), expert_count=4, top_k_experts=2,
    router_hidden_dim=H // 4, dropout=0.0, pad_token_id=0,
    load_balancing_weight=0.001, shard_experts=False,
    pathway_top_k=2, gradient_checkpointing=False,
)
model = zeroBase(config=cfg, test_mode=True)
model.train()

ids = torch.randint(0, cfg.vocab_size, (2, 16))
out = model(input_ids=ids, labels=ids, output_routing_info=True)
out.loss.backward()

# Check pathway_gate grads across all blocks
total_grad_norm = 0.0
total_grad_count = 0
total_grad_zero = 0
for i, block in enumerate(model.blocks):
    pg = block.pathway_gate
    has_grad = all(p.grad is not None for p in pg.parameters())
    if has_grad:
        gn = sum(p.grad.abs().sum().item() for p in pg.parameters())
        nz = sum((p.grad.abs() > 0).sum().item() for p in pg.parameters())
        tz = sum(p.grad.numel() for p in pg.parameters())
        print(f"Block {i}: pathway_gate grad sum={gn:.6e}  nonzero={nz}/{tz}")
        total_grad_norm += gn
        total_grad_count += nz
        total_grad_zero += (tz - nz)
    else:
        print(f"Block {i}: pathway_gate has NO grad attribute (never touched by backward)")

print(f"\nTOTAL: grad_sum={total_grad_norm:.6e}  nonzero={total_grad_count}  zero={total_grad_zero}")
if total_grad_count == 0:
    print("VERDICT: pathway_gate receives NO useful gradients → DEAD CODE, safe to remove.")
else:
    print("VERDICT: pathway_gate receives SOME gradients → NOT dead, do not remove.")
