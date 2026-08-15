"""
XORZENX Benchmarking Suite
Comprehensive benchmark system for comparing XORZENX against baseline models.
"""

from .config import (
    BenchmarkConfig,
    ModelSpec,
    ReportConfig,
    QUICK_BENCHMARK,
    FULL_BENCHMARK,
    SCALE_BENCHMARK,
)

__all__ = [
    "BenchmarkConfig",
    "ModelSpec",
    "ReportConfig",
    "QUICK_BENCHMARK",
    "FULL_BENCHMARK",
    "SCALE_BENCHMARK",
]
