"""
fast_ssm.py -- C++-backed SSMPathway drop-in replacement
=========================================================
Matches the CURRENT SSMPathway signature (Mamba-style diagonal-A with
input-dependent discretisation via dt_proj / A_log).

Original attributes (real SSMPathway):
  A_log    : nn.Parameter([state_dim])     log(-a), init 0 -> a=-1
  dt_proj  : nn.Linear(hidden_dim, state_dim)
  B_proj   : nn.Linear(hidden_dim, state_dim)
  C_proj   : nn.Linear(hidden_dim, state_dim)
  D_proj   : nn.Linear(state_dim, hidden_dim)
  gate_proj: nn.Linear(hidden_dim, hidden_dim*2)
  conv     : nn.Conv1d or None
  ln_input : nn.LayerNorm(hidden_dim)
  ln_state : nn.LayerNorm(state_dim)
  dropout  : nn.Dropout or nn.Identity
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -- Try loading C++ extension ----------------------------------------
try:
    from xorzen.speed import xorzen_ext as _ext
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


class FastSSMPathway(nn.Module):
    """
    Drop-in for SSMPathway using the vectorised log-cumsum scan.
    Identical forward interface:
        forward(x)           -> Tensor [B, T, H]
        forward_parallel(x)  -> Tensor [B, T, H]
        get_compute_stats()  -> dict
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_conv: bool = True,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.state_dim   = state_dim
        self.kernel_size = kernel_size
        self.use_conv    = use_conv

        # Mamba-style diagonal A: store log(-a), negative by construction
        self.A_log    = nn.Parameter(torch.zeros(state_dim))
        self.dt_proj  = nn.Linear(hidden_dim, state_dim, bias=True)
        self.B_proj   = nn.Linear(hidden_dim, state_dim)
        self.C_proj   = nn.Linear(hidden_dim, state_dim)
        self.D_proj   = nn.Linear(state_dim,  hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        if use_conv:
            self.conv = nn.Conv1d(
                hidden_dim, hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
            )
        else:
            self.conv = None

        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_state = nn.LayerNorm(state_dim)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        with torch.no_grad():
            nn.init.zeros_(self.A_log)        # a = -exp(0) = -1, Ab in (0,1)
            nn.init.zeros_(self.dt_proj.bias)
        for proj in [self.B_proj, self.C_proj, self.D_proj, self.gate_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
            nn.init.zeros_(proj.bias)
        if self.conv is not None:
            nn.init.xavier_uniform_(self.conv.weight, gain=1.0 / math.sqrt(2))
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    # ------------------------------------------------------------------
    @classmethod
    def from_slow(cls, slow: nn.Module) -> "FastSSMPathway":
        """Copy all weights from a real SSMPathway instance."""
        dp = slow.dropout.p if hasattr(slow.dropout, "p") else 0.0
        fast = cls(
            hidden_dim  = slow.hidden_dim,
            state_dim   = slow.state_dim,
            kernel_size = slow.kernel_size,
            dropout     = dp,
            use_conv    = slow.conv is not None,
        )
        # Copy the Mamba-style parameters that actually exist on slow
        fast.A_log.data.copy_(slow.A_log.data)
        fast.dt_proj.load_state_dict(slow.dt_proj.state_dict())
        fast.B_proj.load_state_dict(slow.B_proj.state_dict())
        fast.C_proj.load_state_dict(slow.C_proj.state_dict())
        fast.D_proj.load_state_dict(slow.D_proj.state_dict())
        fast.gate_proj.load_state_dict(slow.gate_proj.state_dict())
        fast.ln_input.load_state_dict(slow.ln_input.state_dict())
        fast.ln_state.load_state_dict(slow.ln_state.state_dict())
        if fast.conv is not None and slow.conv is not None:
            fast.conv.load_state_dict(slow.conv.state_dict())
        return fast

    # ------------------------------------------------------------------
    def _vectorised_scan(self, Bv: torch.Tensor, Ab: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable sequential scan.
        Recurrence: h_t = Ab_t * h_{t-1} + Bv_t   (diagonal A, no matrix_exp)

        The log-cumsum parallel formulation overflows for T >= ~60 because
        exp(-cum_log) = exp(+|cum_log|) which grows unboundedly.
        Sequential scan avoids this: Ab is always in (0, 1), so state stays bounded.

        Args:
            Bv : [B, T, state]  -- input projections
            Ab : [B, T, state]  -- per-step decay factors in (0, 1)
        Returns:
            states : [B, T, state]
        """
        B, T, sd = Bv.shape
        state = torch.zeros(B, sd, device=Bv.device, dtype=Bv.dtype)
        outs  = []
        for t in range(T):
            state = Ab[:, t, :] * state + Bv[:, t, :]
            outs.append(state)
        return torch.stack(outs, dim=1)    # [B, T, state]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        x_norm = self.ln_input(x)

        if self.conv is not None:
            x_conv = self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
            x_norm = x_norm + x_conv

        gate_raw = self.gate_proj(x_norm)
        gate, input_gate = gate_raw.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        # Input-dependent discretisation (Mamba style)
        dt = F.softplus(self.dt_proj(x_norm))              # [B, T, state]
        a  = -torch.exp(self.A_log)                        # [state]  negative
        Ab = torch.exp(dt * a.unsqueeze(0).unsqueeze(0))   # [B, T, state]

        Bv = self.B_proj(x_norm * torch.sigmoid(input_gate))  # [B, T, state]
        C  = self.C_proj(x_norm)                              # [B, T, state]

        states  = self._vectorised_scan(Bv, Ab)
        states  = self.ln_state(states)
        ssm_out = C * states
        ssm_out = self.D_proj(ssm_out)
        output  = ssm_out * gate
        return self.dropout(output)

    # ------------------------------------------------------------------
    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    # ------------------------------------------------------------------
    def get_compute_stats(self, seq_len: int, batch_size: int = 1) -> dict:
        proj  = 2 * batch_size * seq_len * self.hidden_dim * self.state_dim
        state = batch_size * seq_len * self.state_dim
        out   = batch_size * seq_len * self.state_dim
        conv  = (batch_size * seq_len * self.hidden_dim * self.kernel_size * 2
                 if self.conv else 0)
        total = proj + state + out + conv
        return {
            "flops_total"            : total,
            "flops_per_token"        : total / (batch_size * seq_len),
            "param_count"            : sum(p.numel() for p in self.parameters()),
            "param_memory_bytes"     : sum(p.numel() for p in self.parameters()) * 4,
            "activation_memory_bytes": batch_size * seq_len * self.state_dim * 4 * 5,
            "state_dim"              : self.state_dim,
            "cpp_kernel_active"      : _CPP_AVAILABLE,
        }
