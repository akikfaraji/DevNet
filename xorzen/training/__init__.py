"""
xorzen Training Module
Comprehensive training infrastructure for xorzen models.

New in v0.2.2:
- CheckpointManager with versioning
- TrainingState tracker
- continue_train() function for seamless training continuation
- Enhanced trainer with better resume support

Usage:
    # Initial training
    >>> from xorzen.training import Trainer, TrainingConfig
    >>> from xorzen.models.zero import zero_277M
    >>> from xorzen.data import load_from_bin
    >>> 
    >>> model = zero_277M()
    >>> data = load_from_bin('train.bin')
    >>> 
    >>> trainer = Trainer(
    ...     model=model,
    ...     train_data=data,
    ...     output_dir='checkpoints/run1'
    ... )
    >>> trainer.train(epochs=10)
    
    # Continue training
    >>> from xorzen.training import continue_train
    >>> trainer = continue_train(
    ...     model=model,
    ...     train_data=data,
    ...     checkpoint='checkpoints/run1/checkpoint_epoch_10.pt',
    ...     additional_epochs=5
    ... )
    >>> trainer.train()
"""
from typing import Dict, Union, List, Any, Optional, Iterator, Tuple
from pathlib import Path
import warnings

# === CORE COMPONENTS ===
from .trainer import XORZENXTrainer, TrainingState as LegacyTrainingState
from .checkpoint import CheckpointManager, CheckpointMetadata
from .state import TrainingState, format_time, format_number
from .curriculum import CurriculumTrainer, CurriculumConfig

# === ALIASES ===
# Make Trainer the primary name
Trainer = XORZENXTrainer


# === HIGH-LEVEL API FUNCTIONS ===

def train(
    model,
    train_data,
    eval_data=None,
    epochs: int = 3,
    output_dir: str = 'checkpoints',
    **kwargs
) -> Trainer:
    """
    High-level training function.
    
    Convenience function that creates and configures a trainer.
    
    Args:
        model: Model to train
        train_data: Training dataset
        eval_data: Evaluation dataset (optional)
        epochs: Number of epochs
        output_dir: Directory for checkpoints and logs
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured trainer (not yet started)
        
    Example:
        >>> from xorzen.training import train
        >>> from xorzen.models.zero import zero_277M
        >>> from xorzen.data import load_from_bin
        >>> 
        >>> model = zero_277M()
        >>> data = load_from_bin('train.bin', batch_size=32)
        >>> 
        >>> trainer = train(
        ...     model=model,
        ...     train_data=data,
        ...     epochs=10,
        ...     output_dir='checkpoints/run1'
        ... )
        >>> trainer.train()
    """
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        **kwargs
    )
    
    # Don't start training automatically - let user call train()
    return trainer


def continue_train(
    model,
    train_data,
    checkpoint: str,
    eval_data=None,
    additional_epochs: int = None,
    output_dir: str = None,
    **kwargs
) -> Trainer:
    """
    Continue training from a checkpoint.
    
    This is the PRIMARY way to resume/continue training in xorzen.
    Automatically loads checkpoint, validates epoch continuity,
    and configures trainer for continuation.
    
    Args:
        model: Model to train
        train_data: Training dataset
        checkpoint: Path to checkpoint file
        eval_data: Evaluation dataset (optional)
        additional_epochs: Number of additional epochs to train
        output_dir: Directory for new checkpoints (uses checkpoint dir if None)
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured trainer ready to continue training
        
    Raises:
        CheckpointNotFoundError: If checkpoint doesn't exist
        EpochContinuityError: If epoch validation fails
        
    Example:
        >>> from xorzen.training import continue_train
        >>> from xorzen.models.zero import zero_277M
        >>> from xorzen.data import load_from_bin
        >>> 
        >>> model = zero_277M()
        >>> data = load_from_bin('train.bin', batch_size=32)
        >>> 
        >>> # Continue from epoch 10
        >>> trainer = continue_train(
        ...     model=model,
        ...     train_data=data,
        ...     checkpoint='checkpoints/run1/checkpoint_epoch_10.pt',
        ...     additional_epochs=5
        ... )
        >>> trainer.train()  # Will train epochs 11-15
    """
    from pathlib import Path
    
    checkpoint_path = Path(checkpoint)
    
    # Determine output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        **kwargs
    )
    
    # Load the checkpoint into the trainer. This will restore model, optimizer, scheduler, and training state.
    trainer.load_checkpoint(checkpoint_path)

    # Extract epoch info from the restored state
    starting_epoch = trainer.state.epoch
    global_step = trainer.state.global_step # Also get global step from state

    # Calculate total epochs if additional_epochs specified
    if additional_epochs is not None:
        total_epochs = starting_epoch + additional_epochs
    else:
        total_epochs = starting_epoch + 1  # At least one more epoch
    
    # Store continuation info
    trainer._continuation_info = {
        'resumed_from': str(checkpoint_path),
        'starting_epoch': starting_epoch,
        'target_epochs': total_epochs,
        'resumed_global_step': global_step,
    }
    
    from xorzen.utils.logger import get_logger
    logger = get_logger()
    logger.info("training", 
               f"Continuing training from epoch {starting_epoch}, "
               f"will train until epoch {total_epochs}")
    
    return trainer


def evaluate(
    model,
    eval_data,
    checkpoint: str = None,
    **kwargs
) -> Dict:
    """
    Evaluate a model.
    
    Args:
        model: Model to evaluate
        eval_data: Evaluation dataset
        checkpoint: Optional checkpoint to load
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of evaluation metrics
        
    Example:
        >>> from xorzen.training import evaluate
        >>> from xorzen.models.zero import zero_277M
        >>> from xorzen.data import load_from_bin
        >>> 
        >>> model = zero_277M()
        >>> eval_data = load_from_bin('val.bin', batch_size=32)
        >>> 
        >>> metrics = evaluate(
        ...     model=model,
        ...     eval_data=eval_data,
        ...     checkpoint='checkpoints/best_model.pt'
        ... )
        >>> print(f"Eval Loss: {metrics['eval_loss']:.4f}")
    """
    # Load checkpoint if provided
    if checkpoint:
        manager = CheckpointManager(Path(checkpoint).parent)
        manager.load(checkpoint, model=model)
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        eval_dataset=eval_data,
        **kwargs
    )
    
    # Run evaluation
    metrics = trainer.evaluate()
    
    return metrics


__all__ = [
    # === Core Classes ===
    'Trainer',
    'XORZENXTrainer',  # Original name for backward compatibility
    'CheckpointManager',
    'CheckpointMetadata',
    'TrainingState',
    'LegacyTrainingState',  # Old TrainingState from trainer.py
    
    # === High-Level API ===
    'train',
    'continue_train',
    'evaluate',
    
    # === Curriculum Training ===
    'CurriculumTrainer',
    'CurriculumConfig',

    # === Utilities ===
    'format_time',
    'format_number',
]


# === CONVENIENCE IMPORTS ===

# Try to import optional components
try:
    from .callbacks import (
        Callback,
        EarlyStoppingCallback,
        CheckpointCallback,
        MetricsCallback,
    )
    __all__.extend([
        'Callback',
        'EarlyStoppingCallback',
        'CheckpointCallback',
        'MetricsCallback',
    ])
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False

try:
    from .metrics import MetricsTracker
    __all__.append('MetricsTracker')
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


# === INITIALIZATION ===

def _print_training_tips():
    """Print helpful tips on first import (only once)."""
    import os
    
    # Only print once per session
    if os.environ.get('XORZENX_TRAINING_TIPS_SHOWN'):
        return
    
    os.environ['XORZENX_TRAINING_TIPS_SHOWN'] = '1'
    
    from xorzen.utils.logger import get_logger
    logger = get_logger()
    
    logger.debug("training", 
                "Training module loaded. Use xorzen.training.train() or "
                "xorzen.training.continue_train() for high-level API")


# Show tips on import
_print_training_tips()

