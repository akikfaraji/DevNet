"""
XORZENX Quick Benchmark
Runs a fast benchmark on MNIST with XORZENX vs baseline models.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.config import BenchmarkConfig, QUICK_BENCHMARK
from benchmarks.runner import BenchmarkRunner

def main():
    print("="*80)
    print("XORZENX QUICK BENCHMARK - MNIST")
    print("="*80)
    print()
    print("This will run a quick comparison of:")
    print("  1. XORZENX-1M (hybrid architecture)")
    print("  2. Vanilla Transformer (baseline)")
    print("  3. MoE Transformer (MoE baseline)")
    print()
    print("Training: 5 epochs on MNIST (60k samples)")
    print("Expected time: ~10-15 minutes on CPU")
    print()
    
    # Quick configuration
    config = BenchmarkConfig(
        dataset_name="mnist",
        device="cpu",
        batch_size=64,
        max_epochs=5,
        num_runs=1,  # Single run for speed
        eval_interval=100,
    )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    runner.run_benchmark_suite(QUICK_BENCHMARK)
    
    print()
    print("="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print()
    print("Results saved to:")
    print(f"  - JSON: {config.output_dir}/benchmark_results.json")
    print(f"  - Reports: benchmarks/reports/")
    print()
    print("Check the Markdown report for detailed analysis:")
    print("  benchmarks/reports/benchmark_report.md")
    print()

if __name__ == "__main__":
    main()
