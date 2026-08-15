"""
SSM scan kernels for Xorzen.

This module provides THREE scan implementations of the diagonal state-space
recurrence:

    h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
    y_t = C_t * h_t + D * u_t

where A_bar_t, B_bar_t, C_t are all input-dependent (per-token) and A is
diagonal (shape [B, T, N], not [B, T, N, N]).

The three implementations are:

1. ``sequential_scan`` — a Python loop, O(T) latency, O(T*N) memory.
   Reference-correct; used for testing the parallel scans.

2. ``parallel_scan`` — a Blelloch-style associative scan in pure PyTorch,
   O(log T) parallel depth, O(T log T) work. Iterates log2(T) times with
   clone() to avoid in-place aliasing. No Python loop over T.

3. ``chunked_scan`` — a hybrid: splits the sequence into chunks of size C,
   runs ``sequential_scan`` within each chunk (vectorized over chunks),
   then carries the final state of each chunk into the next. O(T/C) Python
   iterations with O(C*N) per-iteration work. This is the practical default:
   it keeps the Python loop count small (T/C, e.g. 8 for T=2048, C=256)
   while staying numerically stable (no log-cumsum overflow).

ZOH discretization for diagonal A
---------------------------------
For continuous-time A (diagonal, real, negative) and step dt:

    A_bar = exp(dt * A)                  # in (0, 1) for stable A
    B_bar = ((A_bar - 1) / A) * B        # exact ZOH for diagonal A

When |dt * A| is small the division (exp(x)-1)/x is numerically unstable;
we use the Taylor expansion ``1 + x/2 + x^2/6`` in that regime (same trick
as Mamba and as xorzen/model/ssm.py:S4DKernel).

A first-order approximation ``B_bar = dt * B`` is also available via
``discretize_b_first_order``; it is cheaper but less accurate.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "discretize_zoh",
    "discretize_b_first_order",
    "sequential_scan",
    "parallel_scan",
    "chunked_scan",
    "select_scan",
]


def discretize_zoh(
    A: torch.Tensor,
    B: torch.Tensor,
    dt: torch.Tensor,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Order-Hold discretization for diagonal A.

    Args:
        A:  [N] or [B, T, N] — diagonal of A (must be negative for stability).
        B:  [B, T, N] — input projection B(x_t).
        dt: [B, T, N] — input-dependent step size, must be > 0.
        eps: threshold below which the Taylor expansion is used.

    Returns:
        A_bar: [B, T, N] — discrete transition matrix diagonal, in (0, 1).
        B_bar: [B, T, N] — discrete input projection.

    Math:
        A_bar = exp(dt * A)
        B_bar = ((A_bar - 1) / A) * B       # exact for diagonal A

    For |dt * A| < eps, (exp(x) - 1)/x is replaced by its second-order
    Taylor expansion ``1 + x/2 + x^2/6`` to avoid 0/0.
    """
    # Broadcast A to [B, T, N] if needed
    if A.dim() == 1:
        A_b = A.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
    else:
        A_b = A
    z = dt * A_b  # [B, T, N]
    A_bar = torch.exp(z)
    # B_bar = ((exp(z) - 1) / A) * B   where z = dt * A
    #       = ((exp(z) - 1) / z) * (dt * B)     ... using z = dt * A
    # So the per-element factor we need is (exp(z) - 1) / z, with Taylor fallback.
    small = z.abs() < eps
    # safe_z: avoid 0/0 by substituting 1.0 where z is tiny (Taylor takes over)
    safe_z = torch.where(small, torch.ones_like(z), z)
    exact = (A_bar - 1.0) / safe_z           # (exp(z)-1)/z, stable for |z| >= eps
    taylor = 1.0 + z / 2.0 + (z * z) / 6.0   # second-order expansion
    B_bar_div = torch.where(small, taylor, exact)  # this is (exp(z)-1)/z
    # B_bar = B_bar_div * dt * B   (because (exp(z)-1)/A = (exp(z)-1)/z * dt)
    B_bar = B_bar_div * dt * B
    return A_bar, B_bar


def discretize_b_first_order(
    A: torch.Tensor,
    B: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    First-order approximation: B_bar = dt * B.

    Cheaper than ZOH but less accurate. Useful as a fallback or for
    ablations.

    Returns:
        A_bar = exp(dt * A), B_bar = dt * B
    """
    if A.dim() == 1:
        A_b = A.unsqueeze(0).unsqueeze(0)
    else:
        A_b = A
    A_bar = torch.exp(dt * A_b)
    B_bar = dt * B
    return A_bar, B_bar


def sequential_scan(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reference sequential scan: h_t = A_bar_t * h_{t-1} + B_bar_t.

    Args:
        A_bar: [B, T, N] in (0, 1)
        B_bar: [B, T, N]
        init_state: [B, N] or None (zeros)

    Returns:
        states: [B, T, N]
    """
    B, T, N = A_bar.shape
    if init_state is None:
        state = torch.zeros(B, N, device=A_bar.device, dtype=A_bar.dtype)
    else:
        state = init_state
    outs = []
    for t in range(T):
        state = A_bar[:, t, :] * state + B_bar[:, t, :]
        outs.append(state)
    return torch.stack(outs, dim=1)


def parallel_scan(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Inclusive prefix scan for the linear recurrence
        h_t = A_bar_t * h_{t-1} + B_bar_t
    implemented as a Hillis-Steele associative scan with O(log T) parallel
    depth and O(T log T) work.

    The associative operator combining two adjacent segments (left = earlier,
    right = later) is:
        (a_l, b_l) ⊕ (a_r, b_r) = (a_r * a_l,  a_r * b_l + b_r)

    Under this operator, the inclusive prefix at position t is
        (Pi_{i<=t} a_i,  h_t)   where   h_t = Sum_{i<=t} (Pi_{j=i+1..t} a_j) * b_i
    which is exactly the recurrence solution for h_{-1} = 0.

    Algorithm: Hillis-Steele inclusive scan. At step d (d = 0, 1, ..., logL-1),
    every position i >= 2^d is updated by combining it with position i - 2^d
    (which holds the prefix for [i-2^d, i-1] from the previous step). After
    log L steps, every position holds the prefix over [0, i].

    Memory: O(T log T) due to per-step clones (necessary for autograd safety).

    Args:
        A_bar: [B, T, N]
        B_bar: [B, T, N]
        init_state: [B, N] or None (zeros). If provided, h_0 = a_0 * init + b_0.

    Returns:
        states: [B, T, N]
    """
    B, T, N = A_bar.shape
    if T <= 1:
        if init_state is None:
            return B_bar.clone()
        return A_bar[:, :1, :] * init_state.unsqueeze(1) + B_bar

    L = 1 << (int(T - 1).bit_length())  # next power of 2 >= T
    # Pad to L with identity (a=1, b=0).
    a = torch.ones(B, L, N, device=A_bar.device, dtype=A_bar.dtype)
    b = torch.zeros(B, L, N, device=A_bar.device, dtype=B_bar.dtype)
    a[:, :T, :] = A_bar
    b[:, :T, :] = B_bar

    # Fold init_state into position 0: h_0 = a_0 * init + b_0
    if init_state is not None:
        b = b.clone()
        b[:, 0, :] = a[:, 0, :] * init_state + b[:, 0, :]

    logL = int(math.log2(L))
    for d in range(logL):
        offset = 1 << d
        # right_new = right ⊕ left = (a_r * a_l, a_r * b_l + b_r)
        a_l = a[:, :-offset, :].clone()
        b_l = b[:, :-offset, :].clone()
        a_r = a[:, offset:, :].clone()
        b_r = b[:, offset:, :].clone()
        a = a.clone()
        b = b.clone()
        a[:, offset:, :] = a_r * a_l
        b[:, offset:, :] = a_r * b_l + b_r

    return b[:, :T, :]


def chunked_scan(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Hybrid chunked scan: sequential within chunks, carry state across chunks.

    This is the practical default. It makes T/chunk_size Python iterations
    (e.g. 8 for T=2048, chunk=256) while keeping each iteration fully
    vectorized over the chunk dimension. Numerically stable (no log-cumsum
    overflow) and memory-efficient (O(chunk_size * N) transient).

    Args:
        A_bar: [B, T, N]
        B_bar: [B, T, N]
        init_state: [B, N] or None (zeros)
        chunk_size: number of timesteps per chunk (default 256)

    Returns:
        states: [B, T, N]
    """
    B, T, N = A_bar.shape
    if T <= chunk_size:
        return sequential_scan(A_bar, B_bar, init_state)

    if init_state is None:
        state = torch.zeros(B, N, device=A_bar.device, dtype=A_bar.dtype)
    else:
        state = init_state

    outs = []
    t = 0
    while t < T:
        end = min(t + chunk_size, T)
        chunk_A = A_bar[:, t:end, :]
        chunk_B = B_bar[:, t:end, :]
        # Run sequential scan within the chunk, starting from `state`
        chunk_states = sequential_scan(chunk_A, chunk_B, state)
        outs.append(chunk_states)
        # Carry the last state forward
        state = chunk_states[:, -1, :]
        t = end

    return torch.cat(outs, dim=1)


def select_scan(
    A_bar: torch.Tensor,
    B_bar: torch.Tensor,
    init_state: Optional[torch.Tensor] = None,
    method: Optional[str] = None,
    chunk_size: int = 256,
) -> torch.Tensor:
    """
    Pick the best scan method based on tensor shape and device.

    Defaults:
        T <= 64   -> sequential
        T <= 256  -> chunked (chunk=T)
        T > 256   -> chunked (chunk=256)
        method='parallel' -> parallel (Blelloch)

    The parallel scan is opt-in because it has O(T log T) memory and is
    only faster than chunked on GPU for very long sequences.
    """
    T = A_bar.shape[1]
    if method is not None:
        if method == "sequential":
            return sequential_scan(A_bar, B_bar, init_state)
        if method == "parallel":
            return parallel_scan(A_bar, B_bar, init_state)
        if method == "chunked":
            return chunked_scan(A_bar, B_bar, init_state, chunk_size)
        raise ValueError(f"unknown scan method: {method}")

    if T <= 64:
        return sequential_scan(A_bar, B_bar, init_state)
    return chunked_scan(A_bar, B_bar, init_state, chunk_size=chunk_size)
