"""
Profile where the sparse-dispatch overhead actually lives.

Hypothesis: the Python for-loop over pathways (3 iterations) and the
per-call closure creation (_wrap) in HASSBlock.forward are the main
overhead. We measure:
  1. Total forward time
  2. Time spent in sparse_pathway_dispatch
  3. Time spent in closure creation (_wrap)
  4. Time spent in pathway forward calls (local/low_rank/ssm)

We then compare against a vectorized implementation that pre-creates
the wrapped functions.
"""
import sys, os, time, json
sys.path.insert(0, "/home/z/my-project/xorzen_dev")
import torch
import numpy as np

from xorzen.config import ConfigFactory, ModelSize
from xorzen.models.zero.variants import zeroBase

torch.manual_seed(42); np.random.seed(42)

H = 96
cfg = ConfigFactory.get_config(ModelSize.TINY_23K)
cfg.update(
    vocab_size=256, context_length=128, hidden_size=H,
    num_layers=4, num_attention_heads=4, max_depth=4, min_depth=1,
    width_choices=(H // 2, H), expert_count=4, top_k_experts=2,
    router_hidden_dim=H // 4, dropout=0.0, pad_token_id=0,
    load_balancing_weight=0.001, shard_experts=False,
    pathway_top_k=2, gradient_checkpointing=False,
    eval_routing_noise=0.15,
)
model = zeroBase(config=cfg, test_mode=True)
model.eval()

ids = torch.randint(0, cfg.vocab_size, (4, 128))

# Warm up
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids=ids, output_routing_info=True)

# Measure total forward time
N = 20
torch.manual_seed(0)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N):
        _ = model(input_ids=ids, output_routing_info=True)
t1 = time.perf_counter()
total_ms = (t1 - t0) / N * 1000
print(f"Total forward (eval, no grad): {total_ms:.2f} ms / call")

# Now profile with torch.profiler
from torch.profiler import profile, ProfilerActivity
activities = [ProfilerActivity.CPU]
with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=ids, output_routing_info=True)

# Print top 20 events by CPU time
print("\n=== Top 20 CPU-time events ===")
print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time_total", row_limit=20))

# Save events to JSON for further analysis
out_path = "/home/z/my-project/xorzen_dev/reports/v05/phase5_profile.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
events = prof.key_averages()
event_list = []
for ev in events:
    event_list.append({
        "key": ev.key,
        "cpu_time_us": ev.cpu_time_total,
        "cpu_memory": ev.cpu_memory_usage,
        "self_cpu_time_us": ev.self_cpu_time_total,
        "calls": ev.count,
    })
event_list.sort(key=lambda x: -x["cpu_time_us"])
with open(out_path, "w") as f:
    json.dump(event_list[:30], f, indent=2)
print(f"\nTop 30 events saved to {out_path}")

# Measure just the sparse_pathway_dispatch path
print("\n=== Isolated sparse_pathway_dispatch timing ===")
from xorzen.model.components.sparse_dispatch import sparse_pathway_dispatch
from xorzen.model.components.hass_block import HASSBlock

block = model.blocks[0]
B, T = 4, 128
H_actual = cfg.hidden_size
x = torch.randn(B, T, H_actual)
# Simulate path_probs with mild bias (realistic)
path_probs = torch.randn(B, T, 3).softmax(dim=-1)

# Build wrapped_fns exactly as HASSBlock.forward does
def _wrap(fn):
    def wrapped(x_slice):
        x3d = x_slice.unsqueeze(0)
        y3d = fn(x3d)
        return y3d.squeeze(0)
    return wrapped

pathway_fns = {
    'local':    lambda x_in, *a: block.pathways['local'](x_in, *(a if a else (None, None))),
    'low_rank': lambda x_in, *a: block.pathways['low_rank'](x_in),
    'ssm':      lambda x_in, *a: block.pathways['ssm'].forward_parallel(x_in),
}
wrapped_fns = {k: _wrap(fn) for k, fn in pathway_fns.items()}

# Time the dispatch
N_isolated = 100
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_isolated):
        _ = sparse_pathway_dispatch(
            x, path_probs, wrapped_fns, ['local', 'low_rank', 'ssm'],
            top_k=2, training=False, extra_args=None,
            pathway_call_counter=None,
        )
t1 = time.perf_counter()
dispatch_ms = (t1 - t0) / N_isolated * 1000
print(f"sparse_pathway_dispatch (1 block): {dispatch_ms:.3f} ms / call")
print(f"  × {cfg.num_layers} blocks = {dispatch_ms * cfg.num_layers:.3f} ms / forward (estimated)")
print(f"  Total forward was {total_ms:.2f} ms → dispatch is ~{100*dispatch_ms*cfg.num_layers/total_ms:.1f}% of forward")
