"""
XORZENX Benchmark Reporting
Generates comprehensive reports, charts, and comparisons.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from config import BenchmarkConfig, ReportConfig


class BenchmarkReporter:
    """Generates reports from benchmark results."""
    
    def __init__(self, results: Dict[str, List[Dict]], config: BenchmarkConfig):
        self.results = results
        self.config = config
        self.report_config = ReportConfig()
        
        # Create report directory
        Path(self.report_config.report_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_all_reports(self):
        """Generate all configured reports."""
        if self.report_config.generate_markdown:
            self.generate_markdown_report()
        
        if self.report_config.plot_loss_curves:
            self.plot_training_curves()
        
        if self.report_config.plot_throughput:
            self.plot_throughput_comparison()
        
        if self.report_config.plot_memory:
            self.plot_memory_comparison()
        
        if self.report_config.plot_pareto:
            self.plot_pareto_frontier()
        
        if self.report_config.include_summary_table:
            self.generate_summary_table()
    
    def aggregate_runs(self, model_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple runs for a model."""
        if not model_results:
            return {}
        
        # Extract final metrics from each run
        final_metrics = []
        for run in model_results:
            if "final" in run:
                final_metrics.append(run["final"])
        
        if not final_metrics:
            return {}
        
        # Compute mean and std
        accuracies = [m["accuracy"] for m in final_metrics]
        losses = [m["loss"] for m in final_metrics]
        
        return {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "loss_mean": np.mean(losses),
            "loss_std": np.std(losses),
            "num_runs": len(final_metrics),
        }
    
    def generate_markdown_report(self):
        """Generate Markdown summary report."""
        report_path = Path(self.report_config.report_dir) / "benchmark_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# XORZENX Benchmark Results\n\n")
            f.write(f"**Dataset:** {self.config.dataset_name}\n")
            f.write(f"**Device:** {self.config.device}\n")
            f.write(f"**Epochs:** {self.config.max_epochs}\n")
            f.write(f"**Runs per model:** {self.config.num_runs}\n\n")
            
            f.write("---\n\n")
            f.write("## Summary Table\n\n")
            f.write("| Model | Parameters | Final Accuracy | Final Loss | Training Time |\n")
            f.write("|-------|-----------|----------------|------------|---------------|\n")
            
            for model_name, model_results in self.results.items():
                if not model_results:
                    continue
                
                agg = self.aggregate_runs(model_results)
                params = model_results[0]["params"]["total"]
                
                # Average training time
                avg_time = np.mean([
                    sum(e["train"]["time_sec"] for e in r["epochs"])
                    for r in model_results
                ])
                
                f.write(f"| {model_name} ")
                f.write(f"| {params:,} ")
                f.write(f"| {agg['accuracy_mean']:.2f}% ± {agg['accuracy_std']:.2f} ")
                f.write(f"| {agg['loss_mean']:.4f} ± {agg['loss_std']:.4f} ")
                f.write(f"| {avg_time:.1f}s |\n")
            
            f.write("\n---\n\n")
            f.write("## Detailed Results\n\n")
            
            for model_name, model_results in self.results.items():
                f.write(f"### {model_name}\n\n")
                
                agg = self.aggregate_runs(model_results)
                
                f.write(f"**Parameters:** {model_results[0]['params']['total']:,}\n\n")
                f.write(f"**Final Accuracy:** {agg['accuracy_mean']:.2f}% ± {agg['accuracy_std']:.2f}\n\n")
                f.write(f"**Final Loss:** {agg['loss_mean']:.4f} ± {agg['loss_std']:.4f}\n\n")
                
                # Training curves
                f.write("**Training Progress:**\n\n")
                f.write("| Epoch | Train Loss | Val Loss | Val Accuracy |\n")
                f.write("|-------|------------|----------|-------------|\n")
                
                # Average across runs
                num_epochs = len(model_results[0]["epochs"])
                for epoch in range(num_epochs):
                    train_losses = [r["epochs"][epoch]["train"]["loss"] for r in model_results]
                    val_losses = [r["epochs"][epoch]["eval"]["loss"] for r in model_results]
                    val_accs = [r["epochs"][epoch]["eval"]["accuracy"] for r in model_results]
                    
                    f.write(f"| {epoch + 1} ")
                    f.write(f"| {np.mean(train_losses):.4f} ")
                    f.write(f"| {np.mean(val_losses):.4f} ")
                    f.write(f"| {np.mean(val_accs):.2f}% |\n")
                
                f.write("\n")
        
        print(f"    Markdown report: {report_path}")
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        for model_name, model_results in self.results.items():
            if not model_results:
                continue
            
            # Average across runs
            num_epochs = len(model_results[0]["epochs"])
            epochs = list(range(1, num_epochs + 1))
            
            train_losses = []
            val_losses = []
            val_accs = []
            
            for epoch in range(num_epochs):
                train_losses.append(np.mean([
                    r["epochs"][epoch]["train"]["loss"] for r in model_results
                ]))
                val_losses.append(np.mean([
                    r["epochs"][epoch]["eval"]["loss"] for r in model_results
                ]))
                val_accs.append(np.mean([
                    r["epochs"][epoch]["eval"]["accuracy"] for r in model_results
                ]))
            
            # Plot losses
            ax1.plot(epochs, train_losses, label=f"{model_name} (train)", linestyle='--')
            ax1.plot(epochs, val_losses, label=f"{model_name} (val)")
            
            # Plot accuracy
            ax2.plot(epochs, val_accs, label=model_name)
        
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(self.report_config.report_dir) / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"    Training curves: {plot_path}")
    
    def plot_throughput_comparison(self):
        """Plot throughput comparison."""
        model_names = []
        throughputs = []
        throughput_stds = []
        
        for model_name, model_results in self.results.items():
            if not model_results:
                continue
            
            # Average throughput from last epoch
            tp_values = [r["epochs"][-1]["train"]["throughput"] for r in model_results]
            
            model_names.append(model_name)
            throughputs.append(np.mean(tp_values))
            throughput_stds.append(np.std(tp_values))
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(model_names))
        bars = ax.bar(x, throughputs, yerr=throughput_stds, capsize=5)
        
        # Color XORZENX bars differently
        for i, name in enumerate(model_names):
            if "XORZENX" in name:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('steelblue')
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Training Throughput Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(self.report_config.report_dir) / "throughput_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"    Throughput comparison: {plot_path}")
    
    def plot_memory_comparison(self):
        """Plot peak memory usage comparison."""
        model_names = []
        peak_memories = []
        memory_stds = []
        
        for model_name, model_results in self.results.items():
            if not model_results:
                continue
            
            # Peak memory from all epochs
            peak_values = []
            for run in model_results:
                peak = max(e["memory"]["peak_mb"] for e in run["epochs"])
                peak_values.append(peak)
            
            model_names.append(model_name)
            peak_memories.append(np.mean(peak_values))
            memory_stds.append(np.std(peak_values))
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(model_names))
        bars = ax.bar(x, peak_memories, yerr=memory_stds, capsize=5)
        
        # Color XORZENX bars differently
        for i, name in enumerate(model_names):
            if "XORZENX" in name:
                bars[i].set_color('orange')
            else:
                bars[i].set_color('steelblue')
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Peak Memory Usage Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(self.report_config.report_dir) / "memory_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"    Memory comparison: {plot_path}")
    
    def plot_pareto_frontier(self):
        """Plot accuracy vs efficiency (Pareto frontier)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, model_results in self.results.items():
            if not model_results:
                continue
            
            agg = self.aggregate_runs(model_results)
            params = model_results[0]["params"]["total"]
            accuracy = agg["accuracy_mean"]
            
            # Plot point
            color = 'orange' if "XORZENX" in model_name else 'steelblue'
            marker = 'D' if "XORZENX" in model_name else 'o'
            
            ax.scatter(params / 1e6, accuracy, s=200, c=color, marker=marker,
                      edgecolors='black', linewidths=1.5, alpha=0.7,
                      label=model_name)
        
        ax.set_xlabel("Parameters (Millions)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy vs Model Size (Pareto Frontier)")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits with some padding
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=100)
        
        plt.tight_layout()
        plot_path = Path(self.report_config.report_dir) / "pareto_frontier.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"    Pareto frontier: {plot_path}")
    
    def generate_summary_table(self):
        """Generate CSV summary table."""
        import csv
        
        csv_path = Path(self.report_config.report_dir) / "summary.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Parameters", "Accuracy Mean", "Accuracy Std",
                "Loss Mean", "Loss Std", "Throughput", "Peak Memory (MB)"
            ])
            
            for model_name, model_results in self.results.items():
                if not model_results:
                    continue
                
                agg = self.aggregate_runs(model_results)
                params = model_results[0]["params"]["total"]
                
                # Throughput
                throughputs = [r["epochs"][-1]["train"]["throughput"] for r in model_results]
                avg_throughput = np.mean(throughputs)
                
                # Peak memory
                peak_memories = [max(e["memory"]["peak_mb"] for e in r["epochs"]) for r in model_results]
                avg_peak_memory = np.mean(peak_memories)
                
                writer.writerow([
                    model_name,
                    params,
                    f"{agg['accuracy_mean']:.2f}",
                    f"{agg['accuracy_std']:.2f}",
                    f"{agg['loss_mean']:.4f}",
                    f"{agg['loss_std']:.4f}",
                    f"{avg_throughput:.1f}",
                    f"{avg_peak_memory:.1f}",
                ])
        
        print(f"    CSV summary: {csv_path}")