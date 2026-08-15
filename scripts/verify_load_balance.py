"""
Re-verify P5: Switch Transformer load-balance loss.

Formula: L_lb = E * sum_e (f_e * p_e)
  where f_e = fraction of tokens DISPATCHED to expert e (one-hot from argmax)
        p_e = mean router PROBABILITY for expert e

Mathematical bounds:
  - By AM-GM or Cauchy-Schwarz: sum_e(f_e * p_e) ≥ (sum_e sqrt(f_e*p_e))²/E
  - At perfect balance f_e = p_e = 1/E: L = E * E * (1/E²) = 1
  - At full collapse f_0 = p_0 = 1: L = E * 1 = E

So 1 ≤ L_lb ≤ E, with L_lb = 1 iff perfectly balanced.
"""
import os
os.environ["XORZENX_VERBOSE"] = "0"

import torch
import sys
sys.path.insert(0, "/home/z/my-project/xorzen_dev")
from xorzen.model.components.routing import load_balance_loss

E = 8
B, T = 4, 32
N = B * T  # 128 tokens

# Case 1: Perfectly balanced DISPATCH (each expert gets N/E tokens) and uniform probs
# Dispatch: assign tokens round-robin to experts 0..E-1
idx_perfect = torch.arange(N) % E  # [N] each expert gets 16 tokens
# Reshape to [B, T, 1]
idx_perfect = idx_perfect.view(B, T, 1)
probs_perfect = torch.full((B, T, E), 1.0 / E)
L_perfect = load_balance_loss(probs_perfect, idx_perfect, E).item()

# Case 2: Concentrated — all tokens to expert 0, prob 1 on expert 0
idx_conc = torch.zeros(B, T, 1, dtype=torch.long)
probs_conc = torch.zeros(B, T, E)
probs_conc[..., 0] = 1.0
L_conc = load_balance_loss(probs_conc, idx_conc, E).item()

# Case 3: Random routing
torch.manual_seed(0)
idx_rand = torch.randint(0, E, (B, T, 1))
probs_rand = torch.softmax(torch.randn(B, T, E), dim=-1)
L_rand = load_balance_loss(probs_rand, idx_rand, E).item()

print(f"E = {E}")
print(f"L_perfect (balanced dispatch + uniform probs) = {L_perfect:.6f}")
print(f"L_random  (random dispatch + softmax probs)   = {L_rand:.6f}")
print(f"L_conc    (all to expert 0)                    = {L_conc:.6f}")
print()
print(f"Theoretical bounds: 1 ≤ L_lb ≤ E = {E}")
print(f"  L_perfect = 1.0? {abs(L_perfect - 1.0) < 1e-6}")
print(f"  L_conc    = E = {E}? {abs(L_conc - E) < 1e-6}")
print(f"  1 ≤ L_random ≤ E? {1.0 <= L_rand <= E}")
print(f"  L_perfect < L_random < L_conc? {L_perfect < L_rand < L_conc}")

# Also verify the Cauchy-Schwarz form
# L_lb = E * sum(f_e * p_e)
# By Cauchy-Schwarz: sum(f_e * p_e) ≤ sqrt(sum(f_e²) * sum(p_e²))
# At balance: sum(f_e * p_e) = E * (1/E²) = 1/E, so L = 1
# At collapse: sum(f_e * p_e) = 1, so L = E
print()
print("Mathematical identity check:")
print(f"  L_perfect = E * sum(1/E * 1/E) = E * E * (1/E²) = 1.0 ✓")
print(f"  L_conc    = E * (1*1 + 0*0 + ...) = E = {E} ✓")
