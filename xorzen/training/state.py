"""
xorzen Training State Tracker
Comprehensive tracking of training progress and metrics.

This module provides state management for training runs:
- Track epochs, steps, and timing
- Monitor metrics history
- Calculate training statistics
- Serialize/deserialize state

Example:
    >>> from xorzen.training.state import TrainingState
    >>> 
    >>> state = TrainingState()
    >>> state.on_epoch_start(epoch=1)
    >>> state.update_metrics({'train_loss': 2.5})
    >>> state.on_epoch_end()
    >>> 
    >>> print(f"Epoch {state.epoch} took {state.last_epoch_time:.2f}s")
"""

import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import json
from pathlib import Path

from xorzen.utils.logger import get_logger

logger = get_logger()


# =============================================================================
# TRAINING STATE
# =============================================================================

@dataclass
class TrainingState:
    """
    Comprehensive training state tracker.
    
    Tracks all aspects of training progress including:
    - Current position (epoch, step)
    - Metrics history
    - Timing information
    - Best model tracking
    - Early stopping state
    
    Example:
        >>> state = TrainingState()
        >>> 
        >>> # Start training
        >>> state.on_training_start()
        >>> 
        >>> # Epoch loop
        >>> for epoch in range(10):
        ...     state.on_epoch_start(epoch)
        ...     
        ...     # Step loop
        ...     for step in range(1000):
        ...         state.on_step_start()
        ...         # ... training code ...
        ...         state.update_metrics({'loss': loss_value})
        ...         state.on_step_end()
        ...     
        ...     state.on_epoch_end()
        >>> 
        >>> state.on_training_end()
        >>> print(f"Training took {state.total_time:.2f}s")
    """
    
    # === Position ===
    epoch: int = 0
    global_step: int = 0
    steps_in_epoch: int = 0
    
    # === Metrics ===
    train_loss: float = 0.0
    eval_loss: float = 0.0
    current_lr: float = 0.0
    grad_norm: float = 0.0
    
    # === Best Model Tracking ===
    best_eval_loss: float = float('inf')
    best_metric: float = 0.0
    best_epoch: int = 0
    best_step: int = 0
    
    # === Early Stopping ===
    early_stop_counter: int = 0
    should_stop: bool = False
    
    # === Timing ===
    start_time: float = field(default_factory=time.time)
    epoch_start_time: float = field(default_factory=time.time)
    step_start_time: float = field(default_factory=time.time)
    last_epoch_time: float = 0.0
    last_step_time: float = 0.0
    
    # === Statistics ===
    tokens_seen: int = 0
    samples_seen: int = 0
    total_steps: int = 0
    
    # === History ===
    _metrics_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    _epoch_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    _recent_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # === Flags ===
    is_training: bool = False
    in_epoch: bool = False
    in_step: bool = False
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure mutable defaults are properly initialized
        if not hasattr(self, '_metrics_history') or not isinstance(self._metrics_history, defaultdict):
            self._metrics_history = defaultdict(list)
        if not hasattr(self, '_epoch_metrics') or not isinstance(self._epoch_metrics, dict):
            self._epoch_metrics = {}
        if not hasattr(self, '_recent_losses') or not isinstance(self._recent_losses, deque):
            self._recent_losses = deque(maxlen=100)
    
    # =========================================================================
    # EVENT CALLBACKS
    # =========================================================================
    
    def on_training_start(self):
        """Call when training starts."""
        self.is_training = True
        self.start_time = time.time()
        logger.info("training.state", "Training started")
    
    def on_training_end(self):
        """Call when training ends."""
        self.is_training = False
        logger.info("training.state", 
                   f"Training ended. Total time: {self.total_time:.2f}s")
    
    def on_epoch_start(self, epoch: int):
        """Call when an epoch starts."""
        self.epoch = epoch
        self.in_epoch = True
        self.steps_in_epoch = 0
        self.epoch_start_time = time.time()
        logger.debug("training.state", f"Epoch {epoch} started")
    
    def on_epoch_end(self):
        """Call when an epoch ends."""
        self.in_epoch = False
        self.last_epoch_time = time.time() - self.epoch_start_time
        
        # Store epoch metrics
        self._epoch_metrics[self.epoch] = {
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'lr': self.current_lr,
            'time': self.last_epoch_time,
        }
        
        logger.debug("training.state", 
                    f"Epoch {self.epoch} ended. Time: {self.last_epoch_time:.2f}s")
    
    def on_step_start(self):
        """Call when a step starts."""
        self.in_step = True
        self.step_start_time = time.time()
    
    def on_step_end(self):
        """Call when a step ends."""
        self.in_step = False
        self.global_step += 1
        self.steps_in_epoch += 1
        self.total_steps += 1
        self.last_step_time = time.time() - self.step_start_time
    
    # =========================================================================
    # METRICS
    # =========================================================================
    
    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update current metrics.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        for key, value in metrics.items():
            # Update current value
            if hasattr(self, key):
                setattr(self, key, value)
            
            # Add to history
            self._metrics_history[key].append(value)
            
            # Track recent losses
            if 'loss' in key.lower():
                self._recent_losses.append(value)
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history for a specific metric."""
        return self._metrics_history.get(metric_name, [])
    
    def get_recent_loss_avg(self, window: int = 100) -> float:
        """Get average of recent losses."""
        if not self._recent_losses:
            return 0.0
        
        window = min(window, len(self._recent_losses))
        return sum(list(self._recent_losses)[-window:]) / window
    
    def get_epoch_metrics(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Get metrics for a specific epoch (or current)."""
        epoch = epoch if epoch is not None else self.epoch
        return self._epoch_metrics.get(epoch, {})
    
    # =========================================================================
    # TIMING & STATISTICS
    # =========================================================================
    
    @property
    def total_time(self) -> float:
        """Get total training time in seconds."""
        if self.is_training:
            return time.time() - self.start_time
        return 0.0
    
    @property
    def current_epoch_time(self) -> float:
        """Get current epoch elapsed time."""
        if self.in_epoch:
            return time.time() - self.epoch_start_time
        return self.last_epoch_time
    
    def steps_per_second(self) -> float:
        """Get training speed in steps/second."""
        if self.total_time > 0:
            return self.global_step / self.total_time
        return 0.0
    
    def tokens_per_second(self) -> float:
        """Get training speed in tokens/second."""
        if self.total_time > 0:
            return self.tokens_seen / self.total_time
        return 0.0
    
    def samples_per_second(self) -> float:
        """Get training speed in samples/second."""
        if self.total_time > 0:
            return self.samples_seen / self.total_time
        return 0.0
    
    def estimated_time_remaining(self, total_steps: int) -> float:
        """
        Estimate remaining training time.
        
        Args:
            total_steps: Total number of training steps
            
        Returns:
            Estimated seconds remaining
        """
        if self.global_step == 0:
            return 0.0
        
        steps_remaining = max(0, total_steps - self.global_step)
        avg_step_time = self.total_time / self.global_step
        
        return steps_remaining * avg_step_time
    
    # =========================================================================
    # BEST MODEL TRACKING
    # =========================================================================
    
    def update_best(
        self,
        metric_value: float,
        metric_name: str = 'eval_loss',
        mode: str = 'min'
    ) -> bool:
        """
        Update best model tracking.
        
        Args:
            metric_value: Current metric value
            metric_name: Name of metric
            mode: 'min' for loss, 'max' for accuracy
            
        Returns:
            True if this is a new best
        """
        is_better = False
        
        if mode == 'min':
            if metric_value < self.best_eval_loss:
                self.best_eval_loss = metric_value
                is_better = True
        else:
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                is_better = True
        
        if is_better:
            self.best_epoch = self.epoch
            self.best_step = self.global_step
            logger.info("training.state", 
                       f"New best {metric_name}: {metric_value:.4f}")
        
        return is_better
    
    # =========================================================================
    # EARLY STOPPING
    # =========================================================================
    
    def check_early_stopping(
        self,
        patience: int,
        metric_value: float,
        mode: str = 'min',
        min_delta: float = 0.0
    ) -> bool:
        """
        Check if training should stop early.
        
        Args:
            patience: Number of epochs to wait
            metric_value: Current metric value
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as improvement
            
        Returns:
            True if training should stop
        """
        # Check if improved
        improved = False
        
        if mode == 'min':
            if metric_value < self.best_eval_loss - min_delta:
                improved = True
        else:
            if metric_value > self.best_metric + min_delta:
                improved = True
        
        # Update counter
        if improved:
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
        
        # Check if should stop
        if self.early_stop_counter >= patience:
            self.should_stop = True
            logger.info("training.state", 
                       f"Early stopping triggered after {patience} epochs without improvement")
            return True
        
        return False
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        
        # Convert non-serializable fields
        data['_metrics_history'] = dict(self._metrics_history)
        data['_recent_losses'] = list(self._recent_losses)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        # Convert metrics history back to defaultdict
        if '_metrics_history' in data and isinstance(data['_metrics_history'], dict):
            data['_metrics_history'] = defaultdict(list, data['_metrics_history'])
        
        # Convert recent losses back to deque
        if '_recent_losses' in data and isinstance(data['_recent_losses'], list):
            data['_recent_losses'] = deque(data['_recent_losses'], maxlen=100)
        
        return cls(**data)
    
    def save(self, path: Union[str, Path]):
        """Save state to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.debug("training.state", f"State saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingState':
        """Load state from JSON file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        state = cls.from_dict(data)
        logger.debug("training.state", f"State loaded from {path}")
        
        return state
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'best_eval_loss': self.best_eval_loss,
            'current_lr': self.current_lr,
            'total_time': self.total_time,
            'steps_per_second': self.steps_per_second(),
            'tokens_per_second': self.tokens_per_second(),
            'early_stop_counter': self.early_stop_counter,
            'should_stop': self.should_stop,
        }
    
    def print_summary(self):
        """Print training summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("Training State Summary")
        print("="*70)
        print(f"Epoch: {summary['epoch']}")
        print(f"Global Step: {summary['global_step']:,}")
        print(f"Train Loss: {summary['train_loss']:.4f}")
        print(f"Eval Loss: {summary['eval_loss']:.4f}")
        print(f"Best Eval Loss: {summary['best_eval_loss']:.4f}")
        print(f"Learning Rate: {summary['current_lr']:.2e}")
        print(f"Total Time: {summary['total_time'] / 3600:.2f} hours")
        print(f"Speed: {summary['steps_per_second']:.2f} steps/sec")
        if summary['tokens_per_second'] > 0:
            print(f"Throughput: {summary['tokens_per_second']:,.0f} tokens/sec")
        print("="*70 + "\n")
    
    def __repr__(self) -> str:
        return (f"TrainingState(epoch={self.epoch}, step={self.global_step}, "
                f"loss={self.train_loss:.4f})")


# =============================================================================
# UTILITIES
# =============================================================================

from typing import Union


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        return f"{int(hours)}h {int(minutes)}m"
    else:
        days = seconds / 86400
        hours = (seconds % 86400) / 3600
        return f"{int(days)}d {int(hours)}h"


def format_number(number: Union[int, float]) -> str:
    """
    Format large numbers with K/M/B suffixes.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted string (e.g., "1.5M")
    """
    if number < 1000:
        return str(int(number))
    elif number < 1_000_000:
        return f"{number/1000:.1f}K"
    elif number < 1_000_000_000:
        return f"{number/1_000_000:.1f}M"
    else:
        return f"{number/1_000_000_000:.1f}B"


__all__ = [
    'TrainingState',
    'format_time',
    'format_number',
]

