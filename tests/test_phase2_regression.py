"""Phase 2 regression tests — executable evidence for each claim.

These tests are designed to fail on the original (pre-fix) implementation
and pass on the corrected version. They serve as permanent guards against
regression.

Run: pytest tests/test_phase2_regression.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xorzen.config import ConfigFactory
from xorzen.model.components.ssm_scan import (
    discretize_zoh, sequential_scan, parallel_scan, chunked_scan,
)
from xorzen.model.components.sparse_dispatch import topk_pathway_mask
from xorzen.model.components.hass_block import SSMPathway, LocalAttentionPathway
from xorzen.model.components.routing import RoutingDecision
from xorzen.model.zmoe import LRUExpertCache, ExpertFFN


# ==================== SSM SCAN EQUIVALENCE ====================

class TestSSMScanEquivalence:
    """Regression: all 3 scan methods must produce identical output."""

    @pytest.mark.parametrize("T", [8, 32, 64, 128, 256])
    def test_seq_par_chunked_match(self, T):
        torch.manual_seed(42)
        B, N = 2, 8
        A_bar = torch.rand(B, T, N) * 0.9 + 0.05
        B_bar = torch.randn(B, T, N) * 0.1

        seq = sequential_scan(A_bar, B_bar)
        par = parallel_scan(A_bar, B_bar)
        chk = chunked_scan(A_bar, B_bar, chunk_size=16)

        assert torch.allclose(seq, par, atol=1e-5), f"T={T}: seq vs par diff={(seq-par).abs().max():.2e}"
        assert torch.allclose(seq, chk, atol=1e-5), f"T={T}: seq vs chk diff={(seq-chk).abs().max():.2e}"


# ==================== ZOH DISCRETIZATION ====================

class TestZOHDiscretization:
    """Regression: ZOH must match independent reference."""

    def test_zoh_matches_reference(self):
        torch.manual_seed(42)
        N = 16
        A = -torch.rand(N) * 2.0 - 0.1
        B = torch.randn(1, 8, N) * 0.5
        dt = torch.rand(1, 8, N) * 0.5 + 0.01

        A_bar, B_bar = discretize_zoh(A, B, dt)

        # Independent reference
        with torch.no_grad():
            z = dt * A.unsqueeze(0).unsqueeze(0)
            A_exp = torch.exp(z)
            eps = 1e-4
            small = z.abs() < eps
            safe_z = torch.where(small, torch.ones_like(z), z)
            ref_factor = torch.where(small, 1.0 + z/2 + z**2/6, (A_exp - 1.0) / safe_z)
            ref_B_bar = ref_factor * dt * B

        assert torch.allclose(B_bar, ref_B_bar, atol=1e-5), f"ZOH diff={(B_bar - ref_B_bar).abs().max():.2e}"

    def test_abar_stable(self):
        N = 16
        A = -torch.rand(N) * 2.0 - 0.1
        B = torch.randn(2, 10, N) * 0.5
        dt = torch.rand(2, 10, N) * 0.5 + 0.01
        A_bar, _ = discretize_zoh(A, B, dt)
        assert (A_bar > 0).all(), "A_bar must be positive"
        assert (A_bar < 1).all(), "A_bar must be less than 1"


# ==================== SSM GRADIENTS ====================

class TestSSMGradients:
    """Regression: all SSM parameters must receive gradients."""

    def test_all_params_get_grad(self):
        cfg = ConfigFactory.get_config('1M')
        ssm = SSMPathway(
            hidden_dim=cfg.hidden_size,
            state_dim=cfg.ssm_state_dim,
            kernel_size=cfg.ssm_kernel_size,
            dropout=0.0,
            use_conv=True,
        )
        ssm.train()

        x = torch.randn(1, 8, cfg.hidden_size, requires_grad=True)
        y = ssm.forward(x)
        y.sum().backward()

        for name, p in ssm.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert p.grad.abs().sum() > 0, f"{name} has zero gradient"


# ==================== CAUSAL ATTENTION ====================

class TestCausalAttention:
    """Regression: causal attention must not leak future tokens."""

    def test_no_future_leakage(self):
        cfg = ConfigFactory.get_config('1M')
        attn = LocalAttentionPathway(
            hidden_dim=cfg.hidden_size,
            num_heads=cfg.num_attention_heads // 2,
            window_size=cfg.local_window_size,
            causal=True,
        )
        attn.eval()

        T = 16
        x = torch.zeros(1, T, cfg.hidden_size)
        torch.manual_seed(42)
        for t in range(T):
            x[0, t, 0] = float(t)

        with torch.no_grad():
            out = attn(x)

        x2 = x.clone()
        x2[0, 8:, 0] = 999.0
        with torch.no_grad():
            out2 = attn(x2)

        # Positions 0-7 must be identical
        diff = (out[0, :8, :] - out2[0, :8, :]).abs().max().item()
        assert diff < 1e-6, f"Future token leakage detected: diff={diff:.2e}"

    @pytest.mark.parametrize("window", [1, 2, 4])
    def test_window_edge_cases(self, window):
        cfg = ConfigFactory.get_config('1M')
        attn = LocalAttentionPathway(
            hidden_dim=cfg.hidden_size,
            num_heads=2,
            window_size=window,
            causal=True,
        )
        x = torch.randn(1, 8, cfg.hidden_size)
        out = attn(x)
        assert torch.isfinite(out).all()
        assert out.shape == (1, 8, cfg.hidden_size)


# ==================== ROUTING PROPERTIES ====================

class TestRoutingProperties:
    """Regression: routing probabilities must be valid simplexes, deterministic at eval."""

    def test_path_simplex(self):
        from xorzen.models.zero.model import zeroModel
        cfg = ConfigFactory.get_config('1M')
        torch.manual_seed(42)
        model = zeroModel(cfg, test_mode=True)
        model.eval()

        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids, output_routing_info=True)

        pp = out.routing_info.path_probs
        sums = pp.sum(dim=-1)
        assert (sums - 1.0).abs().max() < 1e-5, f"Path probs don't sum to 1: {sums.min():.6f}-{sums.max():.6f}"
        assert (pp >= -1e-6).all(), "Path probs have negative values"

    def test_expert_simplex(self):
        from xorzen.models.zero.model import zeroModel
        cfg = ConfigFactory.get_config('1M')
        torch.manual_seed(42)
        model = zeroModel(cfg, test_mode=True)
        model.eval()

        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids, output_routing_info=True)

        ep = out.routing_info.expert_probs
        sums = ep.sum(dim=-1)
        assert (sums - 1.0).abs().max() < 1e-5

    def test_width_simplex(self):
        from xorzen.models.zero.model import zeroModel
        cfg = ConfigFactory.get_config('1M')
        torch.manual_seed(42)
        model = zeroModel(cfg, test_mode=True)
        model.eval()

        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids, output_routing_info=True)

        wp = out.routing_info.width_probs
        sums = wp.sum(dim=-1)
        assert (sums - 1.0).abs().max() < 1e-5

    def test_determinism(self):
        from xorzen.models.zero.model import zeroModel
        cfg = ConfigFactory.get_config('1M')
        torch.manual_seed(42)
        model = zeroModel(cfg, test_mode=True)
        model.eval()

        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        with torch.no_grad():
            out1 = model(ids, output_routing_info=True)
            out2 = model(ids, output_routing_info=True)

        assert (out1.routing_info.depth_mask == out2.routing_info.depth_mask).all()
        assert (out1.routing_info.path_probs - out2.routing_info.path_probs).abs().max() < 1e-8
        assert (out1.routing_info.expert_indices == out2.routing_info.expert_indices).all()
        assert (out1.routing_info.width_idx == out2.routing_info.width_idx).all()

    def test_pathway_topk_mask(self):
        """Top-k pathway mask must have exactly K ones per token."""
        pp = torch.softmax(torch.randn(2, 16, 3), dim=-1)
        mask = topk_pathway_mask(pp, top_k=2, training=False)
        ones = mask.sum(dim=-1)
        assert (ones == 2).all(), f"Expected 2 ones/token, got min={ones.min()}, max={ones.max()}"


# ==================== LRU CACHE ====================

class TestLRUCache:
    """Regression: LRU cache must evict least-recently-used experts."""

    def test_lru_eviction_order(self):
        cache = LRUExpertCache(capacity=3)
        experts = [ExpertFFN(hidden_dim=32, intermediate_dim=64) for _ in range(5)]

        for i in range(3):
            cache.put(i, experts[i])

        # Add expert 3 → evicts expert 0 (LRU)
        cache.put(3, experts[3])
        assert 0 not in cache.cache, "Expert 0 should be evicted"
        assert 3 in cache.cache, "Expert 3 should be cached"

        # Access expert 1 → moves to end
        cache.get(1)
        cache.put(4, experts[4])
        assert 2 not in cache.cache, "Expert 2 should be evicted (not 1)"
        assert 1 in cache.cache, "Expert 1 should survive"

    def test_params_per_expert_correct(self):
        from xorzen.model.zmoe import estimate_expert_memory_mb
        mem = estimate_expert_memory_mb(num_experts=8, hidden_dim=64, intermediate_dim=256, cached_experts=3)
        expected = 64*256 + 64*256 + 256*64  # gate + up + down
        assert mem['params_per_expert'] == expected


# ==================== SPPQ ====================

class TestSPPQ:
    """Regression: SPPQ quantization properties."""

    def test_bitwidth_affects_error(self):
        from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig
        m = nn.Linear(64, 128)
        w_orig = m.weight.data.clone()

        errors = {}
        for bits in [4, 8, 16, 32]:
            m.weight.data = w_orig.clone()
            cfg = QuantizationConfig(bits=bits, observe_iterations=0)
            q = SPPQQuantizer(model=m, config=cfg)
            q.calibrate()
            q.apply_quantization()
            errors[bits] = (w_orig - m.weight.data).abs().mean().item()

        # Monotonic: more bits → less error
        assert errors[4] > errors[8] > errors[16] > errors[32], \
            f"Not monotonic: {errors}"

    def test_gradient_flows(self):
        from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig
        for bits in [4, 8]:
            m = nn.Linear(64, 128)
            cfg = QuantizationConfig(bits=bits, observe_iterations=0)
            q = SPPQQuantizer(model=m, config=cfg)
            q.calibrate()
            q.apply_quantization()
            x = torch.randn(4, 64, requires_grad=True)
            y = m(x).sum()
            y.backward()
            assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0, \
                f"No gradient at bits={bits}"

    def test_no_memory_reduction(self):
        from xorzen.utils.sppq import SPPQQuantizer, QuantizationConfig
        m = nn.Linear(64, 128)
        mem_before = m.weight.data.element_size() * m.weight.data.numel()
        cfg = QuantizationConfig(bits=4, observe_iterations=0)
        q = SPPQQuantizer(model=m, config=cfg)
        q.calibrate()
        q.apply_quantization()
        mem_after = m.weight.data.element_size() * m.weight.data.numel()
        assert mem_after == mem_before, "QAT should not change weight dtype/size"
