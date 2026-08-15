"""
Training Continuation and Resumption System
Handles train/continue/resume lifecycle with robust checkpoint management.

This module extends the base XORZENXTrainer with explicit continuation capabilities:
- continue_train(): Continue from a checkpoint with validation
- Resume interrupted training automatically
- Epoch continuity checking
- State restoration
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
import warnings

try:
    import torch
    import torch.nn as nn
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from xorzen.exceptions import CheckpointNotFoundError, CheckpointLoadError, EpochContinuityError, TrainingError
from xorzen.utils.logger import get_logger
from .trainer import xorzenTrainer, TrainingState


class TrainingContinuationManager:
    """
    Manages training continuation and resumption logic.
    
    Responsibilities:
    - Validate checkpoints before continuing
    - Check epoch continuity  
    - Restore complete training state
    - Handle interrupted training recovery
    
    Example:
        >>> manager = TrainingContinuationManager()
        >>> trainer = manager.continue_from_checkpoint(
        ...     checkpoint_path='run1/epoch_10.pt',
        ...     additional_epochs=5
        ... )
    """
    
    def __init__(self):
        self.logger = get_logger()
    
    def validate_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate checkpoint file and extract metadata.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            Checkpoint metadata
            
        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointLoadError: If checkpoint is corrupted
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(str(checkpoint_path))
        
        try:
            if TORCH_AVAILABLE:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            else:
                raise ImportError("PyTorch required for checkpoint loading")
            
            # Extract metadata
            metadata = {
                'path': str(checkpoint_path),
                'has_model': 'model_state_dict' in checkpoint,
                'has_optimizer': 'optimizer_state_dict' in checkpoint,
                'has_scheduler': 'scheduler_state_dict' in checkpoint,
                'has_training_state': 'training_state' in checkpoint,
                'epoch': checkpoint.get('training_state', {}).get('epoch', 0),
                'step': checkpoint.get('training_state', {}).get('global_step', 0),
                'best_loss': checkpoint.get('training_state', {}).get('best_eval_loss', float('inf')),
            }
            
            # Validate required components
            if not metadata['has_model']:
                raise CheckpointLoadError(
                    str(checkpoint_path),
                    "Checkpoint missing model_state_dict"
                )
            
            self.logger.info("training.continuation", 
                           f"Validated checkpoint: epoch={metadata['epoch']}, step={metadata['step']}")
            
            return metadata
        
        except Exception as e:
            if isinstance(e, (CheckpointNotFoundError, CheckpointLoadError)):
                raise
            raise CheckpointLoadError(str(checkpoint_path), str(e))
    
    def check_epoch_continuity(
        self,
        checkpoint_epoch: int,
        requested_start_epoch: Optional[int] = None
    ):
        """
        Check if continuation makes sense epoch-wise.
        
        Args:
            checkpoint_epoch: Epoch from checkpoint
            requested_start_epoch: Epoch user wants to start from
            
        Raises:
            EpochContinuityError: If epochs don't align
        """
        if requested_start_epoch is not None:
            if requested_start_epoch < checkpoint_epoch:
                raise EpochContinuityError(
                    checkpoint_epoch=checkpoint_epoch,
                    requested_epoch=requested_start_epoch
                )
            elif requested_start_epoch > checkpoint_epoch + 1:
                warnings.warn(
                    f"Gap detected: checkpoint at epoch {checkpoint_epoch}, "
                    f"continuing from epoch {requested_start_epoch}. "
                    f"This may indicate a mismatch."
                )
    
    def create_continued_trainer(
        self,
        checkpoint_path: Union[str, Path],
        model: Optional[nn.Module] = None,
        train_dataset = None,
        eval_dataset = None,
        config = None,
        additional_epochs: int = 0,
        **trainer_kwargs
    ) -> XORZENXTrainer:
        """
        Create trainer instance configured for continuation.
        
        Args:
            checkpoint_path: Path to checkpoint to continue from
            model: Model instance (will be loaded from checkpoint if None)
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset  
            config: Training config
            additional_epochs: How many more epochs to train
            **trainer_kwargs: Additional trainer arguments
            
        Returns:
            Configured trainer ready to continue training
        """
        # Validate checkpoint
        metadata = self.validate_checkpoint(checkpoint_path)
        
        # Create trainer
        trainer = XORZENXTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=config,
            **trainer_kwargs
        )
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        self.logger.info("training.continuation", 
                        f"Loaded checkpoint from epoch {metadata['epoch']}, "
                        f"step {metadata['step']}")
        
        # Calculate total epochs
        if additional_epochs > 0:
            trainer.state.total_steps = (
                trainer.state.global_step + 
                additional_epochs * len(trainer.train_dataloader)
            )
        
        return trainer


# === HIGH-LEVEL API FUNCTIONS ===

def train(
    model: nn.Module,
    train_data,
    eval_data = None,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    output_dir: str = "checkpoints",
    config = None,
    **kwargs
) -> XORZENXTrainer:
    """
    Start fresh training run.
    
    This is the primary entry point for new training runs.
    
    Args:
        model: Model to train
        train_data: Training dataset or data path
        eval_data: Evaluation dataset (optional)
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory for checkpoints
        config: Training configuration
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance (already trained)
        
    Example:
        >>> from xorzen.training import train
        >>> from xorzen.models.zero import zero_277M
        >>> from xorzen.data import load_from_bin
        >>> 
        >>> model = zero_277M()
        >>> data = load_from_bin('train.bin')
        >>> trainer = train(model, data, epochs=10, batch_size=4)
    """
    logger = get_logger()
    logger.info("training.api", f"Starting new training run: {epochs} epochs")
    
    # Setup config if needed
    if config is None:
        from xorzen.config import TrainingConfig
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
    
    # Create trainer
    trainer = XORZENXTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        config=config,
        checkpoint_dir=output_dir,
        **kwargs
    )
    
    # Train
    trainer.train(epochs=epochs)
    
    return trainer


def continue_train(
    model: nn.Module,
    train_data,
    checkpoint_path: Union[str, Path],
    eval_data = None,
    additional_epochs: int = 0,
    total_epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
    config = None,
    **kwargs
) -> XORZENXTrainer:
    """
    Continue training from a checkpoint.
    
    This function:
    1. Validates the checkpoint
    2. Restores model, optimizer, and scheduler states
    3. Continues training for additional epochs
    4. Maintains training continuity
    
    Args:
        model: Model instance (architecture must match checkpoint)
        train_data: Training dataset
        checkpoint_path: Path to checkpoint to continue from
        eval_data: Evaluation dataset (optional)
        additional_epochs: Number of additional epochs to train
        total_epochs: Total epochs (alternative to additional_epochs)
        output_dir: Output directory (uses checkpoint dir if None)
        config: Training configuration
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance (already trained)
        
    Example:
        >>> from xorzen.training import continue_train
        >>> from xorzen.models.zero import zero_277M
        >>> from xorzen.data import load_from_bin
        >>> 
        >>> model = zero_277M()
        >>> data = load_from_bin('train.bin')
        >>> trainer = continue_train(
        ...     model, data,
        ...     checkpoint_path='checkpoints/epoch_10.pt',
        ...     additional_epochs=5
        ... )
    """
    logger = get_logger()
    checkpoint_path = Path(checkpoint_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent
    
    logger.info("training.api", 
               f"Continuing training from {checkpoint_path}")
    
    # Use continuation manager
    manager = TrainingContinuationManager()
    
    # Calculate epochs
    metadata = manager.validate_checkpoint(checkpoint_path)
    checkpoint_epoch = metadata['epoch']
    
    if total_epochs is not None:
        if total_epochs <= checkpoint_epoch:
            raise ValueError(
                f"total_epochs ({total_epochs}) must be > checkpoint epoch ({checkpoint_epoch})"
            )
        additional_epochs = total_epochs - checkpoint_epoch
    
    if additional_epochs <= 0:
        raise ValueError("additional_epochs must be > 0")
    
    logger.info("training.api", 
               f"Will train {additional_epochs} additional epochs "
               f"(from epoch {checkpoint_epoch} to {checkpoint_epoch + additional_epochs})")
    
    # Create continued trainer
    trainer = manager.create_continued_trainer(
        checkpoint_path=checkpoint_path,
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        config=config,
        additional_epochs=additional_epochs,
        checkpoint_dir=output_dir,
        **kwargs
    )
    
    # Continue training
    trainer.train(epochs=checkpoint_epoch + additional_epochs)
    
    return trainer


def resume_train(
    checkpoint_path: Union[str, Path],
    train_data,
    model: Optional[nn.Module] = None,
    eval_data = None,
    epochs: Optional[int] = None,
    **kwargs
) -> XORZENXTrainer:
    """
    Resume interrupted training.
    
    Automatically detects interruption and resumes from last checkpoint.
    Similar to continue_train but handles interrupted runs specifically.
    
    Args:
        checkpoint_path: Path to checkpoint (can be interrupted checkpoint)
        train_data: Training dataset
        model: Model instance (optional, can load from checkpoint)
        eval_data: Evaluation dataset
        epochs: Total epochs (if None, uses original config)
        **kwargs: Additional trainer arguments
        
    Returns:
        Trainer instance
        
    Example:
        >>> from xorzen.training import resume_train
        >>> from xorzen.data import load_from_bin
        >>> 
        >>> data = load_from_bin('train.bin')
        >>> trainer = resume_train(
        ...     checkpoint_path='checkpoints/interrupted_step_1500.pt',
        ...     train_data=data
        ... )
    """
    logger = get_logger()
    checkpoint_path = Path(checkpoint_path)
    
    logger.info("training.api", f"Resuming training from {checkpoint_path}")
    
    # Validate checkpoint
    manager = TrainingContinuationManager()
    metadata = manager.validate_checkpoint(checkpoint_path)
    
    # Determine remaining epochs
    if epochs is None:
        # Try to get from checkpoint
        try:
            if TORCH_AVAILABLE:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                config = checkpoint.get('config')
                if config:
                    epochs = config.get('epochs', 10)  # Default to 10
                else:
                    epochs = 10
            else:
                epochs = 10
        except:
            epochs = 10
    
    checkpoint_epoch = metadata['epoch']
    remaining_epochs = epochs - checkpoint_epoch
    
    if remaining_epochs <= 0:
        logger.warning("training.api", 
                      f"Training already complete (epoch {checkpoint_epoch}/{epochs})")
        remaining_epochs = 1  # Train at least 1 more epoch
    
    logger.info("training.api", 
               f"Resuming with {remaining_epochs} remaining epochs")
    
    # Use continue_train
    return continue_train(
        model=model,
        train_data=train_data,
        checkpoint_path=checkpoint_path,
        eval_data=eval_data,
        additional_epochs=remaining_epochs,
        **kwargs
    )


def evaluate(
    model: nn.Module,
    eval_data,
    checkpoint_path: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        eval_data: Evaluation dataset
        checkpoint_path: Optional checkpoint to load
        batch_size: Batch size
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of evaluation metrics
        
    Example:
        >>> from xorzen.training import evaluate
        >>> metrics = evaluate(model, eval_data, checkpoint_path='best_model.pt')
        >>> print(f"Loss: {metrics['eval_loss']:.4f}")
    """
    logger = get_logger()
    logger.info("training.api", "Running evaluation")
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            manager = TrainingContinuationManager()
            manager.validate_checkpoint(checkpoint_path)
            
            if TORCH_AVAILABLE:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("training.api", f"Loaded model from {checkpoint_path}")
    
    # Create trainer for evaluation
    from xorzen.config import TrainingConfig
    config = TrainingConfig(batch_size=batch_size)
    
    trainer = XORZENXTrainer(
        model=model,
        eval_dataset=eval_data,
        config=config,
        **kwargs
    )
    
    # Run evaluation
    metrics = trainer.evaluate()
    
    logger.info("training.api", f"Evaluation complete: {metrics}")
    
    return metrics


__all__ = [
    'TrainingContinuationManager',
    'train',
    'continue_train',
    'resume_train',
    'evaluate',
]
