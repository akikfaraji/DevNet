"""
xorzen Speed Booster — Core Orchestrator
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import torch
import torch.nn as nn


class SpeedProfile(Enum):
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    LIGHTNING    = "lightning"


@dataclass
class BoosterStats:
    attention_blocks_patched: int = 0
    ssm_blocks_patched: int = 0
    router_cached: bool = False
    expert_fabric_preloaded: bool = False
    compiled: bool = False
    rmsnorm_patched: int = 0

    def report(self) -> str:
        lines = [
            "",
            "  +==========================================+",
            "  |   XORZEN SPEED BOOSTER - PATCH REPORT   |",
            "  +==========================================+",
            f"  LocalAttention blocks  : {self.attention_blocks_patched} patched",
            f"  SSM pathway blocks     : {self.ssm_blocks_patched} patched",
            f"  RMSNorm layers         : {self.rmsnorm_patched} patched (C++ kernel)",
            f"  Router encoder         : {'C++ cached' if self.router_cached else 'unchanged'}",
            f"  Expert fabric          : {'prefetch + C++ dispatch' if self.expert_fabric_preloaded else 'unchanged'}",
            f"  torch.compile          : {'yes' if self.compiled else 'no'}",
            "",
        ]
        return "\n".join(lines)


# ── Fast RMSNorm wrapper ──────────────────────────────────────────

class _FastRMSNorm(nn.Module):
    """Drop-in for RMSNorm using the C++ AVX2 kernel."""
    def __init__(self, original: nn.Module):
        super().__init__()
        self.weight = original.weight  # share parameter
        self.eps = original.eps if hasattr(original, 'eps') else 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from xorzen.speed import xorzen_ext as _ext
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous().float()
        out = _ext.cpp_rms_norm(x2d, self.weight.float(), self.eps)
        return out.to(x.dtype).reshape(shape)


class SpeedBooster:
    def __init__(self, config, profile: SpeedProfile = SpeedProfile.BALANCED,
                 device: Optional[str] = None):
        self.config  = config
        self.profile = profile
        self.device  = device

    def apply(self, model: nn.Module) -> BoosterStats:
        from xorzen.speed.fast_attention import FlashLocalAttention
        from xorzen.speed.fast_ssm import FastSSMPathway
        from xorzen.speed.fast_router import CachedRouter
        from xorzen.speed.fast_moe import PreloadedExpertFabric

        # Check if C++ extension is available for RMSNorm patching
        try:
            from xorzen.speed import xorzen_ext as _ext
            _cpp_ok = True
        except ImportError:
            _cpp_ok = False

        stats = BoosterStats()

        for name, module in list(model.named_modules()):
            cls_name = type(module).__name__
            if cls_name == "LocalAttentionPathway":
                _replace(model, name, FlashLocalAttention.from_slow(module))
                stats.attention_blocks_patched += 1
            elif cls_name == "SSMPathway":
                _replace(model, name, FastSSMPathway.from_slow(module))
                stats.ssm_blocks_patched += 1
            elif cls_name == "AdaptiveRouter" and self.profile != SpeedProfile.CONSERVATIVE:
                _replace(model, name, CachedRouter.from_slow(module))
                stats.router_cached = True
            elif cls_name == "ShardedExpertFabric" and self.profile != SpeedProfile.CONSERVATIVE:
                _replace(model, name, PreloadedExpertFabric.from_slow(module))
                stats.expert_fabric_preloaded = True
            elif cls_name == "RMSNorm" and _cpp_ok and self.profile == SpeedProfile.LIGHTNING:
                _replace(model, name, _FastRMSNorm(module))
                stats.rmsnorm_patched += 1

        if self.profile == SpeedProfile.LIGHTNING and hasattr(torch, "compile"):
            import sys
            _on_cpu_windows = (sys.platform == "win32" and
                               not torch.cuda.is_available())
            if _on_cpu_windows:
                print("  [SpeedBooster] torch.compile skipped (CPU-only Windows — no MSVC cl.exe)")
            else:
                try:
                    self.compiled_model = torch.compile(model, mode="reduce-overhead")
                    stats.compiled = True
                except Exception as e:
                    print(f"[SpeedBooster] torch.compile failed: {e}")

        if self.device:
            model.to(self.device)

        print(stats.report())
        return stats


def boost_model(model: nn.Module, config, profile: SpeedProfile = SpeedProfile.BALANCED,
                device: Optional[str] = None) -> nn.Module:
    booster = SpeedBooster(config, profile, device)
    booster.apply(model)
    if hasattr(booster, "compiled_model"):
        return booster.compiled_model
    return model


def _replace(root: nn.Module, path: str, replacement: nn.Module):
    parts  = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], replacement)
