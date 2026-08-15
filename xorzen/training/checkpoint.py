"""
xorzen Checkpoint Manager
Comprehensive checkpoint management for training with versioning and validation.

This module provides:
- Save/load checkpoints with validation
- Version checking and migration
- Automatic cleanup of old checkpoints
- Best model tracking
- Resume/continue training support

Example:
    >>> from xorzen.training import CheckpointManager
    >>> 
    >>> manager = CheckpointManager('checkpoints/run1')
    >>> 
    >>> # Save checkpoint
    >>> manager.save(
    ...     model=model,
    ...     optimizer=optimizer,
    ...     epoch=10,
    ...     step=1000,
    ...     metrics={'loss': 2.5}
    ... )
    >>> 
    >>> # Load checkpoint
    >>> checkpoint = manager.load('checkpoint_epoch_10.pt')
    >>> model.load_state_dict(checkpoint['model_state_dict'])
"""

import json
import shutil
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings

from xorzen.exceptions import (
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointLoadError,
    CheckpointSaveError,
    CheckpointVersionError,
)
from xorzen.utils.logger import get_logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = get_logger()


# =============================================================================
# CHECKPOINT METADATA
# =============================================================================

@dataclass
class CheckpointMetadata:
    """
    Metadata for a checkpoint.
    
    Stores essential information about the checkpoint for validation
    and tracking.
    """
    
    # Identification
    checkpoint_name: str
    checkpoint_path: str
    
    # Training state
    epoch: int
    global_step: int
    
    # Metrics
    train_loss: float
    eval_loss: Optional[float] = None
    best_metric: Optional[float] = None
    
    # Timing
    created_at: str = ""
    training_time_seconds: float = 0.0
    
    # Version
    xorzen_version: str = "0.2.2"
    checkpoint_version: str = "1.0"
    
    # Model info
    model_name: Optional[str] = None
    model_config: Optional[Dict] = None
    
    # Additional info
    is_best: bool = False
    notes: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: Union[str, Path]):
        """Save metadata to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CheckpointMetadata':
        """Load metadata from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Manage model checkpoints with versioning and validation.
    
    Features:
    - Automatic checkpoint naming
    - Version checking
    - Old checkpoint cleanup
    - Best model tracking
    - Resume/continue support
    
    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='checkpoints/run1',
        ...     max_checkpoints=5,
        ...     save_best=True
        ... )
        >>> 
        >>> # Save checkpoint
        >>> manager.save(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=10,
        ...     metrics={'loss': 2.5, 'accuracy': 0.85}
        ... )
        >>> 
        >>> # Load latest
        >>> checkpoint = manager.load_latest()
        >>> 
        >>> # Continue training
        >>> checkpoint = manager.load_for_continue(epoch=10)
    """
    
    CHECKPOINT_VERSION = "1.0"
    XORZENX_VERSION = "0.2.2"
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_best: bool = True,
        best_metric: str = 'eval_loss',
        best_mode: str = 'min',  # 'min' or 'max'
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Save best model separately
            best_metric: Metric to track for best model
            best_mode: 'min' for loss, 'max' for accuracy
            save_optimizer: Save optimizer state
            save_scheduler: Save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        # Best model tracking
        self.best_value = float('inf') if best_mode == 'min' else float('-inf')
        self.best_checkpoint_path: Optional[Path] = None
        
        # Load existing best checkpoint info if available
        self._load_best_checkpoint_info()
        
        logger.info("checkpoint", 
                   f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save(
        self,
        model,
        optimizer=None,
        scheduler=None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        model_config: Optional[Dict] = None,
        is_best: bool = False,
        notes: str = "",
        **kwargs
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer (optional)
            scheduler: LR scheduler (optional)
            epoch: Current epoch
            global_step: Global training step
            metrics: Dictionary of metrics
            model_config: Model configuration
            is_best: Mark as best checkpoint
            notes: Additional notes
            **kwargs: Additional data to save
            
        Returns:
            Path to saved checkpoint
            
        Raises:
            CheckpointSaveError: If saving fails
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for checkpoint management")
        
        metrics = metrics or {}
        
        # Generate checkpoint name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{global_step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        logger.info("checkpoint", 
                   f"Saving checkpoint: epoch={epoch}, step={global_step}")
        
        try:
            # Create checkpoint data
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
                'metrics': metrics,
                'checkpoint_version': self.CHECKPOINT_VERSION,
                'xorzen_version': self.XORZENX_VERSION,
                'created_at': datetime.now().isoformat(),
            }
            
            # Add optimizer state
            if self.save_optimizer and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            # Add scheduler state
            if self.save_scheduler and scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # Add model config
            if model_config:
                checkpoint['model_config'] = model_config
            elif hasattr(model, 'config'):
                try:
                    checkpoint['model_config'] = model.config.to_dict() if hasattr(model.config, 'to_dict') else str(model.config)
                except:
                    pass
            
            # Add any additional kwargs
            checkpoint.update(kwargs)
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_name=checkpoint_name,
                checkpoint_path=str(checkpoint_path),
                epoch=epoch,
                global_step=global_step,
                train_loss=metrics.get('train_loss', 0.0),
                eval_loss=metrics.get('eval_loss'),
                best_metric=metrics.get(self.best_metric),
                model_name=model.__class__.__name__ if hasattr(model, '__class__') else None,
                model_config=model_config,
                is_best=is_best,
                notes=notes
            )
            
            # Save metadata
            metadata_path = checkpoint_path.with_suffix('.meta.json')
            metadata.save(metadata_path)
            
            logger.info("checkpoint", f"Checkpoint saved: {checkpoint_path}")
            
            # Check if this is the best checkpoint
            if self.save_best and self.best_metric in metrics:
                metric_value = metrics[self.best_metric]
                is_new_best = self._is_better(metric_value, self.best_value)
                
                if is_new_best:
                    self.best_value = metric_value
                    self._save_best_checkpoint(checkpoint_path, metadata)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return checkpoint_path
        
        except Exception as e:
            raise CheckpointSaveError(f"Failed to save checkpoint: {e}")
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        model=None,
        optimizer=None,
        scheduler=None,
        strict: bool = True,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            strict: Strict state dict loading
            map_location: Device to map tensors to
            
        Returns:
            Checkpoint dictionary
            
        Raises:
            CheckpointLoadError: If loading fails
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for checkpoint loading")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(str(checkpoint_path))
        
        logger.info("checkpoint", f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Validate version
            self._validate_version(checkpoint)
            
            # Load model state
            if model is not None and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
                logger.debug("checkpoint", "Model state loaded")
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.debug("checkpoint", "Optimizer state loaded")
            
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.debug("checkpoint", "Scheduler state loaded")
            
            logger.info("checkpoint", 
                       f"Checkpoint loaded: epoch={checkpoint.get('epoch')}, "
                       f"step={checkpoint.get('global_step')}")
            
            return checkpoint
        
        except CheckpointVersionError:
            raise
        except Exception as e:
            raise CheckpointLoadError(str(checkpoint_path), str(e))
    
    def load_latest(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            **kwargs: Additional arguments for load()
            
        Returns:
            Checkpoint dictionary
            
        Raises:
            CheckpointNotFoundError: If no checkpoints found
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            raise CheckpointNotFoundError(
                f"No checkpoints found in {self.checkpoint_dir}"
            )
        
        # Get latest checkpoint (sorted by modification time)
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        return self.load(latest, model, optimizer, scheduler, **kwargs)
    
    def load_best(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            **kwargs: Additional arguments for load()
            
        Returns:
            Checkpoint dictionary
            
        Raises:
            CheckpointNotFoundError: If no best checkpoint found
        """
        best_path = self.checkpoint_dir / 'best_model.pt'
        
        if not best_path.exists():
            raise CheckpointNotFoundError(
                f"No best checkpoint found in {self.checkpoint_dir}"
            )
        
        return self.load(best_path, model, optimizer, scheduler, **kwargs)
    
    def load_for_continue(
        self,
        epoch: int,
        model=None,
        optimizer=None,
        scheduler=None,
        validate_epoch: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load checkpoint for continuing training.
        
        This method validates epoch continuity to prevent training errors.
        
        Args:
            epoch: Expected epoch to continue from
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            validate_epoch: Validate epoch matches
            **kwargs: Additional arguments for load()
            
        Returns:
            Checkpoint dictionary
            
        Raises:
            CheckpointNotFoundError: If checkpoint not found
            EpochContinuityError: If epoch mismatch
        """
        from xorzen.exceptions import EpochContinuityError
        
        # Find checkpoint with matching epoch
        checkpoints = self.list_checkpoints()
        
        matching_checkpoints = []
        for cp_path in checkpoints:
            metadata = self._load_metadata(cp_path)
            if metadata and metadata.epoch == epoch:
                matching_checkpoints.append((cp_path, metadata))
        
        if not matching_checkpoints:
            raise CheckpointNotFoundError(
                f"No checkpoint found for epoch {epoch} in {self.checkpoint_dir}"
            )
        
        # Get the latest checkpoint for this epoch (in case multiple exist)
        cp_path, metadata = max(
            matching_checkpoints,
            key=lambda x: x[0].stat().st_mtime
        )
        
        # Load checkpoint
        checkpoint = self.load(cp_path, model, optimizer, scheduler, **kwargs)
        
        # Validate epoch if requested
        if validate_epoch and checkpoint.get('epoch') != epoch:
            raise EpochContinuityError(
                checkpoint_epoch=checkpoint.get('epoch', -1),
                requested_epoch=epoch
            )
        
        logger.info("checkpoint", 
                   f"Loaded checkpoint for continuing training from epoch {epoch}")
        
        return checkpoint
    
    def list_checkpoints(self) -> List[Path]:
        """
        List all checkpoint files.
        
        Returns:
            List of checkpoint file paths
        """
        return sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
    
    def list_checkpoints_detailed(self) -> List[Dict[str, Any]]:
        """
        List checkpoints with detailed information.
        
        Returns:
            List of dictionaries with checkpoint info
        """
        checkpoints = []
        
        for cp_path in self.list_checkpoints():
            metadata = self._load_metadata(cp_path)
            
            info = {
                'path': str(cp_path),
                'name': cp_path.name,
                'size_mb': cp_path.stat().st_size / (1024 ** 2),
                'modified': datetime.fromtimestamp(cp_path.stat().st_mtime).isoformat(),
            }
            
            if metadata:
                info.update({
                    'epoch': metadata.epoch,
                    'global_step': metadata.global_step,
                    'train_loss': metadata.train_loss,
                    'eval_loss': metadata.eval_loss,
                    'is_best': metadata.is_best,
                })
            
            checkpoints.append(info)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Delete a specific checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to delete
        """
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("checkpoint", f"Deleted checkpoint: {checkpoint_path}")
            
            # Delete metadata if exists
            metadata_path = checkpoint_path.with_suffix('.meta.json')
            if metadata_path.exists():
                metadata_path.unlink()
    
    def delete_all_checkpoints(self, keep_best: bool = True):
        """
        Delete all checkpoints.
        
        Args:
            keep_best: Keep the best checkpoint
        """
        for cp_path in self.list_checkpoints():
            # Skip best model if requested
            if keep_best and cp_path.name == 'best_model.pt':
                continue
            
            self.delete_checkpoint(cp_path)
        
        logger.info("checkpoint", "All checkpoints deleted")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = self.list_checkpoints()
        
        # Filter out best model
        regular_checkpoints = [
            cp for cp in checkpoints
            if cp.name != 'best_model.pt'
        ]
        
        # Sort by modification time (oldest first)
        regular_checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Delete oldest checkpoints
        while len(regular_checkpoints) > self.max_checkpoints:
            oldest = regular_checkpoints.pop(0)
            self.delete_checkpoint(oldest)
            logger.debug("checkpoint", f"Cleaned up old checkpoint: {oldest.name}")
    
    def _save_best_checkpoint(self, checkpoint_path: Path, metadata: CheckpointMetadata):
        """Save a copy as the best checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        best_metadata_path = self.checkpoint_dir / 'best_model.meta.json'
        
        # Copy checkpoint
        shutil.copy2(checkpoint_path, best_path)
        
        # Update and save metadata
        metadata.is_best = True
        metadata.checkpoint_name = 'best_model.pt'
        metadata.checkpoint_path = str(best_path)
        metadata.save(best_metadata_path)
        
        self.best_checkpoint_path = best_path
        
        # Save best checkpoint info
        self._save_best_checkpoint_info()
        
        logger.info("checkpoint", 
                   f"Saved best checkpoint: {self.best_metric}={self.best_value:.4f}")
    
    def _is_better(self, new_value: float, old_value: float) -> bool:
        """Check if new value is better than old value."""
        if self.best_mode == 'min':
            return new_value < old_value
        else:
            return new_value > old_value
    
    def _validate_version(self, checkpoint: Dict[str, Any]):
        """Validate checkpoint version compatibility."""
        checkpoint_version = checkpoint.get('checkpoint_version', '0.0')
        xorzen_version = checkpoint.get('xorzen_version', 'unknown')
        
        # Check major version compatibility
        if checkpoint_version.split('.')[0] != self.CHECKPOINT_VERSION.split('.')[0]:
            raise CheckpointVersionError(
                expected_version=self.CHECKPOINT_VERSION,
                found_version=checkpoint_version
            )
        
        # Warn about xorzen version mismatch
        if xorzen_version != self.XORZENX_VERSION:
            warnings.warn(
                f"Checkpoint was created with xorzen version {xorzen_version}, "
                f"but current version is {self.XORZENX_VERSION}. "
                "Some features may not work correctly."
            )
    
    def _load_metadata(self, checkpoint_path: Path) -> Optional[CheckpointMetadata]:
        """Load metadata for a checkpoint."""
        metadata_path = checkpoint_path.with_suffix('.meta.json')
        
        if metadata_path.exists():
            try:
                return CheckpointMetadata.load(metadata_path)
            except Exception as e:
                logger.warning("checkpoint", 
                              f"Failed to load metadata for {checkpoint_path.name}: {e}")
        
        return None
    
    def _save_best_checkpoint_info(self):
        """Save best checkpoint info to file."""
        info_path = self.checkpoint_dir / 'best_checkpoint_info.json'
        
        info = {
            'best_value': self.best_value,
            'best_metric': self.best_metric,
            'best_checkpoint_path': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            'updated_at': datetime.now().isoformat(),
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def _load_best_checkpoint_info(self):
        """Load best checkpoint info from file."""
        info_path = self.checkpoint_dir / 'best_checkpoint_info.json'
        
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                self.best_value = info.get('best_value', self.best_value)
                
                best_path = info.get('best_checkpoint_path')
                if best_path:
                    self.best_checkpoint_path = Path(best_path)
            
            except Exception as e:
                logger.warning("checkpoint", 
                              f"Failed to load best checkpoint info: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        checkpoints = self.list_checkpoints()
        
        total_size = 0
        for cp in checkpoints:
            total_size += cp.stat().st_size
            meta_path = cp.with_suffix('.meta.json')
            if meta_path.exists():
                total_size += meta_path.stat().st_size
        
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'num_checkpoints': len(checkpoints),
            'total_size_mb': total_size / (1024 ** 2),
            'max_checkpoints': self.max_checkpoints,
            'has_best_checkpoint': (self.checkpoint_dir / 'best_model.pt').exists(),
            'best_value': self.best_value if self.best_value != float('inf') and self.best_value != float('-inf') else None,
            'best_metric': self.best_metric,
        }
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"CheckpointManager(dir={self.checkpoint_dir.name}, "
                f"checkpoints={stats['num_checkpoints']}/{self.max_checkpoints})")


__all__ = [
    'CheckpointMetadata',
    'CheckpointManager',
]

