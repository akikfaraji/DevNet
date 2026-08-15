"""
xorzen Trainer - Production-Grade Training Loop
Version: 2.0

Comprehensive trainer for xorzen-zero models with support for:
- Mixed precision training (FP16/BF16)
- Gradient accumulation and clipping
- Distributed training
- Checkpoint management
- Evaluation and metrics
- Early stopping
- Learning rate scheduling
- Progress tracking and logging
- CPU and GPU optimization
"""

import os
import time
import math
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import traceback

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available - trainer functionality limited")

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import the official TrainingState from xorzen.training.state
from xorzen.training.state import TrainingState


# ==================== TRAINER ====================

class XORZENXTrainer:
    """
    Production-grade trainer for xorzen-zero models.
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        config: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            callbacks: List of callback functions
            **kwargs: Additional arguments
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
        
        # Core components
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.callbacks = callbacks or []
        
        # Training state
        self.state = TrainingState()
        
        # Device setup
        self.device = self._setup_device()
        if self.model is not None:
            self.model.to(self.device)
        
        # Mixed precision
        self.scaler = None
        self.autocast_dtype = None
        self._setup_mixed_precision()
        
        # Data loaders
        self.train_dataloader = None
        self.eval_dataloader = None
        self._setup_dataloaders()
        
        # Logging
        self.logger = None
        self.tensorboard_writer = None
        self.wandb_run = None
        self._setup_logging()
        
        # Checkpointing
        self.checkpoint_dir = Path(kwargs.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics
        self.metrics = defaultdict(list)
        self.best_checkpoint_path = None
        
        # Print setup info
        self._print_setup_info()
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config and hasattr(self.config, 'training'):
            device_str = self.config.training.device
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        
        device = torch.device(device_str)
        
        if device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Using CPU for training")
            # CPU optimizations
            torch.set_num_threads(os.cpu_count())
        
        return device
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.config and hasattr(self.config, 'training'):
            precision = self.config.training.mixed_precision
        else:
            precision = "fp32"
        
        if precision == "fp16":
            self.scaler = torch.cuda.amp.GradScaler()
            self.autocast_dtype = torch.float16
            print("Using FP16 mixed precision")
        elif precision == "bf16":
            self.scaler = None  # BF16 doesn't need scaler
            self.autocast_dtype = torch.bfloat16
            print("Using BF16 mixed precision")
        else:
            self.scaler = None
            self.autocast_dtype = None
            print("Using FP32 precision")
    
    def _setup_dataloaders(self):
        """Setup data loaders."""
        if self.train_dataset is None:
            return
        
        batch_size = 1
        num_workers = 0
        pin_memory = False
        
        if self.config and hasattr(self.config, 'training'):
            batch_size = self.config.training.batch_size
            num_workers = self.config.training.num_workers
            pin_memory = self.config.training.pin_memory and self.device.type == 'cuda'
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False
            )
    
    def _setup_logging(self):
        """Setup logging."""
        if self.config is None:
            return
        
        logging_config = self.config.logging if hasattr(self.config, 'logging') else {}
        
        # TensorBoard
        if logging_config.get('tensorboard', True) and TENSORBOARD_AVAILABLE:
            log_dir = Path(logging_config.get('output_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logging to: {log_dir}")
        
        # Weights & Biases
        if logging_config.get('wandb', False) and WANDB_AVAILABLE:
            wandb_config = {
                'model': self.config.model.to_dict() if hasattr(self.config, 'model') else {},
                'training': self.config.training.to_dict() if hasattr(self.config, 'training') else {},
            }
            
            self.wandb_run = wandb.init(
                project=logging_config.get('project', 'xorzen-zero'),
                name=logging_config.get('run_name'),
                config=wandb_config
            )
            print("Weights & Biases logging enabled")
    
    def _print_setup_info(self):
        """Print training setup information."""
        print("\n" + "="*80)
        print("xorzen Trainer Setup")
        print("="*80)
        
        if self.model is not None:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        if self.train_dataloader is not None:
            print(f"Training Batches: {len(self.train_dataloader):,}")
        
        if self.eval_dataloader is not None:
            print(f"Evaluation Batches: {len(self.eval_dataloader):,}")
        
        if self.optimizer is not None:
            print(f"Optimizer: {self.optimizer.__class__.__name__}")
        
        if self.scheduler is not None:
            print(f"Scheduler: {self.scheduler.__class__.__name__}")
        
        print("="*80 + "\n")
    
    def train(
        self,
        epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Run training loop.
        
        Args:
            epochs: Number of epochs to train
            max_steps: Maximum number of steps
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training metrics
        """
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Determine training length
        if epochs is None and self.config and hasattr(self.config, 'training'):
            epochs = self.config.training.epochs
        elif epochs is None:
            epochs = 3  # Default
        
        if max_steps is None and self.config and hasattr(self.config, 'training'):
            max_steps = self.config.training.total_steps
        
        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        self.state.start_time = time.time()
        
        try:
            for epoch in range(self.state.epoch, epochs):
                self.state.epoch = epoch
                self.state.epoch_start_time = time.time()
                
                # Train one epoch
                epoch_metrics = self._train_epoch(max_steps)
                
                # Evaluate
                if self.eval_dataloader is not None:
                    eval_metrics = self.evaluate()
                    self.state.eval_loss = eval_metrics['eval_loss']
                    
                    # Check for best model
                    if eval_metrics['eval_loss'] < self.state.best_eval_loss:
                        self.state.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint(is_best=True)
                
                # Epoch summary
                self._print_epoch_summary(epoch, epoch_metrics)
                
                # Check early stopping
                if self.state.should_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
                
                # Save checkpoint
                if self.config and hasattr(self.config, 'training'):
                    save_steps = self.config.training.save_steps
                else:
                    save_steps = 1000
                
                if (epoch + 1) % max(1, epochs // 5) == 0:  # Save 5 times per training
                    self.save_checkpoint()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint(is_interrupted=True)
        
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            traceback.print_exc()
            self.save_checkpoint(is_interrupted=True)
            raise
        
        finally:
            # Cleanup
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()
            
            if self.wandb_run is not None:
                self.wandb_run.finish()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print(f"Total Time: {self.state.total_time / 3600:.2f} hours")
        print(f"Best Eval Loss: {self.state.best_eval_loss:.4f}")
        print("="*80)
        
        return self.metrics
    
    def _train_epoch(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_steps = 0
        
        # Progress bar
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.state.epoch}",
                leave=True
            )
            dataloader = pbar
        else:
            dataloader = self.train_dataloader
        
        gradient_accumulation_steps = 1
        if self.config and hasattr(self.config, 'training'):
            gradient_accumulation_steps = self.config.training.gradient_accumulation_steps
        
        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass
            loss = self._training_step(batch, step % gradient_accumulation_steps == 0)
            
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # Update state
            self.state.global_step += 1
            self.state.steps_in_epoch += 1
            
            # Log metrics
            if step % self._get_logging_steps() == 0:
                self._log_metrics({
                    'train_loss': loss.item(),
                    'learning_rate': self._get_current_lr(),
                    'epoch': self.state.epoch,
                    'step': self.state.global_step,
                })
            
            # Update progress bar
            if pbar is not None:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self._get_current_lr():.2e}',
                })
            
            # Check max steps
            if max_steps and self.state.global_step >= max_steps:
                break
        
        if pbar is not None:
            pbar.close()
        
        avg_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'epoch_steps': epoch_steps,
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor], should_step: bool = True) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            batch: Input batch
            should_step: Whether to perform optimizer step
            
        Returns:
            Loss tensor
        """
        # Mixed precision context
        if self.autocast_dtype is not None:
            with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Scale loss for gradient accumulation
        gradient_accumulation_steps = 1
        if self.config and hasattr(self.config, 'training'):
            gradient_accumulation_steps = self.config.training.gradient_accumulation_steps
        
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step
        if should_step:
            # Gradient clipping
            if self.config and hasattr(self.config, 'training') and self.config.training.gradient_clipping:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                max_grad_norm = self.config.training.max_grad_norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
                self.state.grad_norm = grad_norm.item()
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        return loss * gradient_accumulation_steps  # Return unscaled loss
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.
        
        Returns:
            Evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", disable=not TQDM_AVAILABLE):
                batch = self._move_to_device(batch)
                
                if self.autocast_dtype is not None:
                    with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                        outputs = self.model(**batch)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                total_steps += 1
        
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': math.exp(avg_loss) if avg_loss < 20 else float('inf'),
        }
        
        # Log metrics
        self._log_metrics(metrics)
        
        return metrics
    
    def save_checkpoint(
        self,
        is_best: bool = False,
        is_interrupted: bool = False
    ):
        """
        Save checkpoint.
        
        Args:
            is_best: Whether this is the best checkpoint
            is_interrupted: Whether training was interrupted
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_state': self.state.to_dict(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
            'metrics': dict(self.metrics),
        }
        
        # Determine filename
        if is_best:
            filename = 'best_model.pt'
            self.best_checkpoint_path = self.checkpoint_dir / filename
        elif is_interrupted:
            filename = f'interrupted_step_{self.state.global_step}.pt'
        else:
            filename = f'checkpoint_epoch_{self.state.epoch}_step_{self.state.global_step}.pt'
        
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        print(f"\nCheckpoint saved: {filepath}")
        
        # Save metadata
        metadata = {
            'epoch': self.state.epoch,
            'step': self.state.global_step,
            'train_loss': self.state.train_loss,
            'eval_loss': self.state.eval_loss,
            'best_eval_loss': self.state.best_eval_loss,
            'elapsed_time': self.state.total_time, # Access as property
        }
        
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model
        if self.model and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        if 'training_state' in checkpoint:
            state_dict = checkpoint['training_state']
            for key, value in state_dict.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
        
        # Load metrics
        if 'metrics' in checkpoint:
            self.metrics = defaultdict(list, checkpoint['metrics'])
        
        print(f"Resumed from epoch {self.state.epoch}, step {self.state.global_step}")
    
    def _move_to_device(self, batch: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        """Move batch to device."""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]['lr']
    
    def _get_logging_steps(self) -> int:
        """Get logging frequency."""
        if self.config and hasattr(self.config, 'training'):
            return self.config.training.logging_steps
        return 10
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to all enabled backends."""
        # Store metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, self.state.global_step)
        
        # Weights & Biases
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=self.state.global_step)
    
    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """Print epoch summary."""
        epoch_time = time.time() - self.state.epoch_start_time
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {metrics.get('train_loss', 0.0):.4f}")
        print(f"  Eval Loss: {self.state.eval_loss:.4f}")
        print(f"  Learning Rate: {self._get_current_lr():.2e}")
        print(f"  Epoch Time: {epoch_time / 60:.2f} minutes")
        print(f"  Steps/Second: {self.state.steps_per_second():.2f}")

__all__ = ['XORZENXTrainer']

