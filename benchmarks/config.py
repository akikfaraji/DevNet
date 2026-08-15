"""
XORZENX Benchmarking Configuration
Defines all benchmark settings, model configurations, and comparison baselines.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # Dataset settings
    dataset_name: str = "mnist"  # mnist, cifar10, etc.
    data_dir: str = "./data"
    num_workers: int = 4
    
    # Training settings
    batch_size: int = 32
    eval_batch_size: int = 64
    max_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Evaluation settings
    eval_interval: int = 100  # Steps between evaluations
    save_interval: int = 500  # Steps between checkpoints
    
    # Hardware
    device: str = "cpu"  # or "cuda"
    mixed_precision: bool = False
    
    # Benchmarking
    num_runs: int = 3  # Average over multiple runs for stability
    measure_memory: bool = True
    measure_throughput: bool = True
    measure_inference_speed: bool = True
    
    # Output
    output_dir: str = "./benchmarks/results"
    log_dir: str = "./benchmarks/logs"
    save_predictions: bool = False
    
    def __post_init__(self):
        """Create necessary directories."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""
    name: str
    model_class: str  # Import path or factory function
    params: Dict
    description: str = ""
    paper_reference: Optional[str] = None
    expected_params: Optional[int] = None  # For validation


# ==================== XORZENX Models ====================

XORZENX_1M = ModelSpec(
    name="XORZENX-1M",
    model_class="xorzen.zero_1M",
    params={},
    description="XORZENX hybrid architecture with 1M parameters (4 layers, 128 hidden)",
    expected_params=4_021_477
)

XORZENX_10M = ModelSpec(
    name="XORZENX-10M",
    model_class="xorzen.zero_10M",
    params={},
    description="XORZENX hybrid architecture with 10M parameters (8 layers, 512 hidden)",
    expected_params=None  # Will be calculated
)


# ==================== Baseline Transformers ====================

VANILLA_TRANSFORMER_SMALL = ModelSpec(
    name="Vanilla-Transformer-1M",
    model_class="benchmarks.models.VanillaTransformer",
    params={
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "vocab_size": 4069,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    description="Standard Transformer baseline (GPT-style)",
    paper_reference="Vaswani et al. 2017 - Attention Is All You Need"
)

VANILLA_TRANSFORMER_MEDIUM = ModelSpec(
    name="Vanilla-Transformer-10M",
    model_class="benchmarks.models.VanillaTransformer",
    params={
        "hidden_size": 512,
        "num_layers": 8,
        "num_heads": 8,
        "vocab_size": 20000,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
    description="Standard Transformer baseline (GPT-style)",
    paper_reference="Vaswani et al. 2017"
)


# ==================== Mixture of Experts ====================

MOE_TRANSFORMER = ModelSpec(
    name="MoE-Transformer-1M",
    model_class="benchmarks.models.MoETransformer",
    params={
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "num_experts": 4,
        "top_k": 2,
        "vocab_size": 4069,
        "max_seq_len": 1024,
        "dropout": 0.1,
    },
    description="Transformer with Mixture of Experts (Switch Transformer style)",
    paper_reference="Fedus et al. 2021 - Switch Transformers"
)


# ==================== State Space Models ====================

MAMBA_SMALL = ModelSpec(
    name="Mamba-1M",
    model_class="benchmarks.models.MambaLM",
    params={
        "d_model": 128,
        "n_layer": 4,
        "vocab_size": 4069,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
    },
    description="Mamba SSM language model",
    paper_reference="Gu & Dao 2023 - Mamba: Linear-Time Sequence Modeling"
)


# ==================== Hybrid Models ====================

JAMBA_STYLE = ModelSpec(
    name="Jamba-Style-1M",
    model_class="benchmarks.models.JambaStyle",
    params={
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 4,
        "num_experts": 4,
        "top_k": 2,
        "vocab_size": 4069,
        "ssm_ratio": 0.5,  # 50% SSM layers, 50% attention
    },
    description="Jamba-style hybrid (Attention + SSM + MoE)",
    paper_reference="AI21 Labs 2024 - Jamba"
)


# ==================== Benchmark Suites ====================

QUICK_BENCHMARK = [
    XORZENX_1M,
    VANILLA_TRANSFORMER_SMALL,
    MOE_TRANSFORMER,
]

FULL_BENCHMARK = [
    XORZENX_1M,
    VANILLA_TRANSFORMER_SMALL,
    MOE_TRANSFORMER,
    MAMBA_SMALL,
    JAMBA_STYLE,
]

SCALE_BENCHMARK = [
    XORZENX_1M,
    XORZENX_10M,
    VANILLA_TRANSFORMER_SMALL,
    VANILLA_TRANSFORMER_MEDIUM,
]


# ==================== Metrics to Track ====================

METRICS = {
    "performance": [
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_perplexity",
    ],
    "efficiency": [
        "params_total",
        "params_active",  # For sparse models
        "flops_per_token",
        "memory_peak_mb",
        "memory_allocated_mb",
        "throughput_tokens_per_sec",
        "training_time_sec",
    ],
    "quality": [
        "convergence_speed",  # Steps to reach target loss
        "final_accuracy",
        "sample_quality",  # Perplexity of generated samples
    ],
    "inference": [
        "inference_latency_ms",
        "inference_throughput",
        "memory_inference_mb",
    ]
}


# ==================== Helper Functions ====================

def get_benchmark_suite(suite_name: str) -> List[ModelSpec]:
    """Get a predefined benchmark suite."""
    suites = {
        "quick": QUICK_BENCHMARK,
        "full": FULL_BENCHMARK,
        "scale": SCALE_BENCHMARK,
    }
    
    if suite_name not in suites:
        raise ValueError(f"Unknown suite: {suite_name}. Choose from: {list(suites.keys())}")
    
    return suites[suite_name]


def get_model_by_name(name: str) -> Optional[ModelSpec]:
    """Get a model spec by name."""
    all_models = FULL_BENCHMARK + [VANILLA_TRANSFORMER_MEDIUM]
    
    for model in all_models:
        if model.name == name:
            return model
    
    return None


# ==================== Reporting Configuration ====================

@dataclass
class ReportConfig:
    """Configuration for benchmark reports."""
    
    # Report formats
    generate_markdown: bool = True
    generate_html: bool = True
    generate_json: bool = True
    generate_csv: bool = True
    
    # Charts
    plot_loss_curves: bool = True
    plot_throughput: bool = True
    plot_memory: bool = True
    plot_pareto: bool = True  # Accuracy vs efficiency
    
    # Comparison tables
    include_summary_table: bool = True
    include_detailed_metrics: bool = True
    include_model_cards: bool = True
    
    # Output
    report_dir: str = "./benchmarks/reports"
    
    def __post_init__(self):
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)