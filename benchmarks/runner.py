"""
XORZENX Benchmark Runner
Main script to run comparative benchmarks on MNIST and other datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import psutil
import gc

# Relative imports within benchmarks package
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmarks.config import (
        BenchmarkConfig, ModelSpec, ReportConfig,
        QUICK_BENCHMARK, FULL_BENCHMARK, METRICS
    )
    import benchmarks.models as baseline_models
else:
    from .config import (
        BenchmarkConfig, ModelSpec, ReportConfig,
        QUICK_BENCHMARK, FULL_BENCHMARK, METRICS
    )
    from . import models as baseline_models


class BenchmarkRunner:
    """Runs benchmarks and collects metrics."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results = {}
        
        print(f"="*80)
        print(f"XORZENX BENCHMARK SUITE")
        print(f"="*80)
        print(f"Device: {self.device}")
        print(f"Dataset: {config.dataset_name}")
        print(f"Runs per model: {config.num_runs}")
        print()
    
    def load_dataset(self):
        """Load MNIST dataset."""
        print(f"[1/5] Loading {self.config.dataset_name} dataset...")
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        if self.config.dataset_name == "mnist":
            train_dataset = datasets.MNIST(
                self.config.data_dir,
                train=True,
                download=True,
                transform=transform
            )
            test_dataset = datasets.MNIST(
                self.config.data_dir,
                train=False,
                transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print()
        
        return train_loader, test_loader
    
    def instantiate_model(self, spec: ModelSpec) -> nn.Module:
        """Create model from specification."""
        if spec.model_class.startswith("xorzen."):
            # XORZENX model
            import xorzen
            model_fn = getattr(xorzen, spec.model_class.split(".")[-1])
            model = model_fn()
        elif spec.model_class.startswith("benchmarks.models."):
            # Baseline model
            model_class_name = spec.model_class.split(".")[-1]
            model_class = getattr(baseline_models, model_class_name)
            model = model_class(**spec.params)
        else:
            raise ValueError(f"Unknown model class: {spec.model_class}")
        
        return model.to(self.device)
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # For sparse models, estimate active parameters
        active = trainable
        if hasattr(model, 'moe'):
            # Rough estimate for MoE models
            active = trainable * 0.1  # Assume ~10% active
        
        return {
            "total": total,
            "trainable": trainable,
            "active": active,
        }
    
    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage."""
        if self.device.type == "cuda":
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
            }
        else:
            process = psutil.Process()
            mem_info = process.memory_info()
            return {
                "allocated_mb": mem_info.rss / 1024**2,
                "reserved_mb": mem_info.vms / 1024**2,
                "peak_mb": mem_info.rss / 1024**2,
            }
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Flatten images for sequence models
            data = data.view(data.size(0), -1)  # [B, 784]
            
            # Convert to token indices (simple binning for MNIST)
            data = (data * 255).long().clamp(0, 255)
            
            optimizer.zero_grad()
            
            # Forward - handle different model interfaces
            is_xorzen = 'xorzen' in model.__class__.__module__ or 'zero' in model.__class__.__name__.lower()
            
            if is_xorzen:
                # XORZENX expects labels same shape as input for next-token prediction
                # For classification, we create dummy labels (shift right by 1)
                # Ensure target is 2D: [B] -> [B, 1]
                if target.dim() == 1:
                    target = target.unsqueeze(1)
                labels = torch.cat([data[:, 1:], target], dim=1)
                output = model(input_ids=data, labels=labels, return_dict=True)
                
                # Extract loss if available, otherwise compute from logits
                if hasattr(output, 'loss') and output.loss is not None:
                    loss = output.loss
                else:
                    # Use last position logits for classification
                    logits = output.logits[:, -1, :]
                    loss = F.cross_entropy(logits, target)
            else:
                # Baseline models: standard classification interface
                try:
                    output = model(data, labels=target, return_dict=True)
                    loss = output["loss"]
                except (TypeError, KeyError):
                    # Model doesn't support labels, compute manually
                    output = model(data, return_dict=True)
                    logits = output["logits"]
                    # Average over sequence for classification
                    if logits.dim() == 3:
                        logits = logits.mean(dim=1)
                    loss = F.cross_entropy(logits, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        return {
            "loss": avg_loss,
            "time_sec": epoch_time,
            "throughput": len(train_loader.dataset) / epoch_time,
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate model."""
        model.eval()
        total_loss = 0
        correct = 0
        num_samples = 0
        
        start_time = time.time()
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            data = data.view(data.size(0), -1)
            data = (data * 255).long().clamp(0, 255)
            
            # Forward - handle different model interfaces
            is_xorzen = 'xorzen' in model.__class__.__module__ or 'zero' in model.__class__.__name__.lower()
            
            if is_xorzen:
                # XORZENX model
                output = model(input_ids=data, return_dict=True)
                logits = output.logits
                # Use last position logits for classification
                logits = logits[:, -1, :] if logits.dim() == 3 else logits
            else:
                # Baseline model
                output = model(data, return_dict=True)
                logits = output.logits if hasattr(output, 'logits') else output['logits']
                # Average over sequence for classification
                logits = logits.mean(dim=1) if logits.dim() == 3 else logits
            
            loss = F.cross_entropy(logits, target)
            pred = logits.argmax(dim=1)
            
            total_loss += loss.item() * data.size(0)
            correct += pred.eq(target).sum().item()
            num_samples += data.size(0)
        
        eval_time = time.time() - start_time
        
        return {
            "loss": total_loss / num_samples,
            "accuracy": 100.0 * correct / num_samples,
            "time_sec": eval_time,
            "throughput": num_samples / eval_time,
        }
    
    def benchmark_model(
        self,
        spec: ModelSpec,
        train_loader: DataLoader,
        test_loader: DataLoader,
        run_id: int = 0,
    ) -> Dict[str, Any]:
        """Benchmark a single model."""
        print(f"  Run {run_id + 1}/{self.config.num_runs}")
        
        # Clear memory
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Instantiate model
        print(f"    Instantiating model...")
        model = self.instantiate_model(spec)
        
        # Count parameters
        param_counts = self.count_parameters(model)
        print(f"    Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Training metrics
        results = {
            "model_name": spec.name,
            "run_id": run_id,
            "params": param_counts,
            "epochs": [],
        }
        
        # Training loop
        print(f"    Training for {self.config.max_epochs} epochs...")
        for epoch in range(self.config.max_epochs):
            print(f"    Epoch {epoch + 1}/{self.config.max_epochs}")
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
            
            # Evaluate
            eval_metrics = self.evaluate(model, test_loader)
            
            # Memory
            memory_metrics = self.measure_memory()
            
            results["epochs"].append({
                "epoch": epoch,
                "train": train_metrics,
                "eval": eval_metrics,
                "memory": memory_metrics,
            })
            
            print(f"      Train Loss: {train_metrics['loss']:.4f}")
            print(f"      Val Loss: {eval_metrics['loss']:.4f}")
            print(f"      Val Acc: {eval_metrics['accuracy']:.2f}%")
            print(f"      Memory: {memory_metrics['peak_mb']:.1f} MB")
        
        # Final evaluation
        final_eval = self.evaluate(model, test_loader)
        results["final"] = final_eval
        
        return results
    
    def run_benchmark_suite(
        self,
        model_specs: List[ModelSpec],
    ):
        """Run benchmarks on multiple models."""
        # Load dataset
        train_loader, test_loader = self.load_dataset()
        
        print(f"[2/5] Running benchmarks...")
        
        for idx, spec in enumerate(model_specs, 1):
            print(f"\n[Model {idx}/{len(model_specs)}] {spec.name}")
            print(f"  Description: {spec.description}")
            
            model_results = []
            for run in range(self.config.num_runs):
                try:
                    result = self.benchmark_model(spec, train_loader, test_loader, run)
                    model_results.append(result)
                except Exception as e:
                    print(f"  ❌ Error in run {run}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if model_results:
                self.results[spec.name] = model_results
                print(f"  ✓ Completed {len(model_results)}/{self.config.num_runs} runs")
            else:
                print(f"  ❌ All runs failed for {spec.name}")
        
        print(f"\n[3/5] Saving results...")
        self.save_results()
        
        print(f"\n[4/5] Generating reports...")
        self.generate_reports()
        
        print(f"\n[5/5] Done!")
    
    def save_results(self):
        """Save benchmark results to JSON."""
        output_path = Path(self.config.output_dir) / "benchmark_results.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  Results saved to: {output_path}")
    
    def generate_reports(self):
        """Generate benchmark reports."""
        from benchmarks.reporting import BenchmarkReporter
        
        reporter = BenchmarkReporter(self.results, self.config)
        reporter.generate_all_reports()

