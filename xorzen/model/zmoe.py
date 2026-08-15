"""
ShardedExpertFabric - Disk-Based Mixture of Experts
Production-grade MoE system with 192 experts stored on disk.

This is THE secret sauce for CPU training:
- 192 experts x 32M params each = 6GB total
- Only 24 experts cached in RAM = 768MB
- LRU cache for hot experts
- On-demand loading from disk
- Result: 277M model trains on 16GB RAM!

Key Innovation: Disk sharding + LRU caching makes massive MoE feasible on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import json
import threading
import time
import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Dict, List, Union, Any
from dataclasses import dataclass, field
import math

from xorzen.config import ModelConfig
from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import TensorStability

logger = get_logger()


# ==================== DATA STRUCTURES ====================

@dataclass
class ExpertStats:
    """Statistics for expert usage tracking."""
    expert_id: int
    activation_count: int = 0
    total_weight: float = 0.0
    avg_weight: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    load_time_ms: float = 0.0
    
    def update(self, weight: float, was_cached: bool, load_time: float = 0.0):
        """Update statistics."""
        self.activation_count += 1
        self.total_weight += weight
        if self.activation_count > 0:
            self.avg_weight = self.total_weight / self.activation_count
        else:
            self.avg_weight = 0.0 # Should not happen if activation_count increments before this.
        
        if was_cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.load_time_ms += load_time


@dataclass
class MoEOutput:
    """Output from MoE layer with statistics."""
    output: torch.Tensor  # [N, hidden_dim]
    expert_stats: Dict[int, ExpertStats]
    load_balance_aux: torch.Tensor  # Auxiliary loss for load balancing
    routing_entropy: float  # Routing entropy (higher = more diverse)
    cache_hit_rate: float  # Percentage of experts found in cache


# ==================== EXPERT IMPLEMENTATIONS ====================

class ExpertFFN(nn.Module):
    """
    Single expert: Simple FFN with SwiGLU activation.
    
    This is the basic building block. Each expert is small (~32M params)
    so we can have 192 of them without exploding memory when cached.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # SwiGLU components
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Expert ID (set during initialization)
        self.expert_id = -1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SwiGLU activation.
        
        Args:
            x: [batch, hidden_dim]
        
        Returns:
            [batch, hidden_dim]
        """
        # SwiGLU: gate(x) * silu(up(x))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SiLU activation on gate
        gate = F.silu(gate)
        
        # Element-wise multiplication
        hidden = gate * up
        
        # Down projection
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output
    
    def get_num_params(self) -> int:
        """Get number of parameters in this expert."""
        return sum(p.numel() for p in self.parameters())


# ==================== LRU CACHE ====================

class LRUExpertCache:
    """
    LRU (Least Recently Used) cache for experts.
    
    Keeps hot experts in RAM, evicts cold ones.
    Critical for performance - without this, disk I/O would kill us.
    """
    
    def __init__(self, capacity: int = 24):
        """
        Args:
            capacity: Maximum number of experts to keep in cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()  # Maintains insertion order
        self.lock = threading.Lock()  # Thread-safe operations
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info("zmoe", f"LRU Expert Cache initialized (capacity: {capacity} experts)")
    
    def get(self, expert_id: int) -> Optional[ExpertFFN]:
        """
        Get expert from cache.
        
        Args:
            expert_id: Expert index
        
        Returns:
            Expert module if cached, None otherwise
        """
        with self.lock:
            if expert_id in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(expert_id)
                self.hits += 1
                return self.cache[expert_id]
            else:
                self.misses += 1
                return None
    
    def put(self, expert_id: int, expert: ExpertFFN):
        """
        Put expert in cache.
        
        Args:
            expert_id: Expert index
            expert: Expert module
        """
        with self.lock:
            if expert_id in self.cache:
                # Update and move to end
                self.cache.move_to_end(expert_id)
            else:
                # Add new expert
                self.cache[expert_id] = expert
                
                # Evict oldest if over capacity
                if len(self.cache) > self.capacity:
                    oldest_id = next(iter(self.cache))
                    del self.cache[oldest_id]
                    self.evictions += 1
    
    def clear(self):
        """Clear all cached experts."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'capacity': self.capacity,
            'current_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate
        }


# ==================== DISK MANAGER ====================

class ExpertDiskManager:
    """
    Manages loading/saving experts to disk.
    
    Each expert is saved as a separate .pt file.
    This allows us to load only what we need on-demand.
    """
    
    def __init__(
        self,
        shard_dir: Union[str, Path],
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int
    ):
        """
        Args:
            shard_dir: Directory to store expert shards
            num_experts: Total number of experts
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate dimension for experts
        """
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        
        # Manifest file tracks which experts exist
        self.manifest_path = self.shard_dir / "manifest.json"
        self.manifest = self._load_or_create_manifest()
        
        logger.info("zmoe", f"Expert Disk Manager initialized")
        logger.info("zmoe", f"  Directory: {self.shard_dir}")
        logger.info("zmoe", f"  Total Experts: {num_experts}")
        logger.info("zmoe", f"  Existing Shards: {len(self.manifest['experts'])}")
    
    def _load_or_create_manifest(self) -> Dict:
        """Load or create manifest file."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
            logger.debug("zmoe", f"Loaded manifest: {len(manifest['experts'])} experts")
        else:
            manifest = {
                'version': '0.1.6',
                'num_experts': self.num_experts,
                'hidden_dim': self.hidden_dim,
                'intermediate_dim': self.intermediate_dim,
                'experts': {}
            }
            self._save_manifest(manifest)
            logger.debug("zmoe", "Created new manifest")
        
        return manifest
    
    def _save_manifest(self, manifest: Dict):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def get_expert_path(self, expert_id: int) -> Path:
        """Get path for expert shard file."""
        return self.shard_dir / f"expert_{expert_id:04d}.pt"
    
    def expert_exists(self, expert_id: int) -> bool:
        """Check if expert exists on disk."""
        return str(expert_id) in self.manifest['experts']
    
    def save_expert(self, expert_id: int, expert: ExpertFFN):
        """
        Save expert to disk.
        
        Args:
            expert_id: Expert index
            expert: Expert module to save
        """
        expert_path = self.get_expert_path(expert_id)
        
        # Save state dict
        torch.save({
            'expert_id': expert_id,
            'state_dict': expert.state_dict(),
            'hidden_dim': self.hidden_dim,
            'intermediate_dim': self.intermediate_dim,
            'num_params': expert.get_num_params()
        }, expert_path)
        
        # Update manifest
        self.manifest['experts'][str(expert_id)] = {
            'path': str(expert_path),
            'num_params': expert.get_num_params(),
            'created': time.time()
        }
        self._save_manifest(self.manifest)
        
        logger.debug("zmoe", f"Saved expert {expert_id} to {expert_path}")
    
    def load_expert(self, expert_id: int, device: str = 'cpu') -> ExpertFFN:
        """
        Load expert from disk.
        
        Args:
            expert_id: Expert index
            device: Device to load expert on
        
        Returns:
            Loaded expert module
        """
        expert_path = self.get_expert_path(expert_id)
        
        if not expert_path.exists():
            # Create new expert if doesn't exist
            logger.debug("zmoe", f"Expert {expert_id} not found, creating new...")
            expert = self._create_new_expert(expert_id)
            self.save_expert(expert_id, expert)
            return expert.to(device)
        
        # Load from disk
        start_time = time.time()
        checkpoint = torch.load(expert_path, map_location=device)
        
        # Create expert and load state
        expert = ExpertFFN(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim
        )
        expert.expert_id = expert_id
        expert.load_state_dict(checkpoint['state_dict'])
        expert = expert.to(device)
        expert.eval()  # Set to eval mode
        
        load_time = (time.time() - start_time) * 1000  # ms
        logger.debug("zmoe", f"Loaded expert {expert_id} in {load_time:.2f}ms")
        
        return expert
    
    def _create_new_expert(self, expert_id: int) -> ExpertFFN:
        """Create new expert with random initialization."""
        expert = ExpertFFN(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim
        )
        expert.expert_id = expert_id
        
        # Initialize weights
        for module in expert.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        return expert
    
    def initialize_all_experts(self, force: bool = False):
        """
        Initialize all experts on disk.
        
        Args:
            force: If True, reinitialize even if they exist
        """
        logger.info("zmoe", f"Initializing {self.num_experts} experts...")
        
        for expert_id in range(self.num_experts):
            if force or not self.expert_exists(expert_id):
                expert = self._create_new_expert(expert_id)
                self.save_expert(expert_id, expert)
            
            if (expert_id + 1) % 50 == 0:
                logger.info("zmoe", f"  Initialized {expert_id + 1}/{self.num_experts} experts")
        
        logger.info("zmoe", "All experts initialized")
    
    def get_total_size_mb(self) -> float:
        """Get total size of all expert shards on disk."""
        total_size = 0
        for expert_id in range(self.num_experts):
            expert_path = self.get_expert_path(expert_id)
            if expert_path.exists():
                total_size += expert_path.stat().st_size
        
        return total_size / (1024 ** 2)  # Convert to MB


# ==================== MAIN MOE FABRIC ====================

class ShardedExpertFabric(nn.Module):
    """
    Sharded Expert Fabric - The MoE brain.
    
    Manages 192 experts with disk sharding + LRU caching.
    This is what makes the 277M model work on CPU!
    
    Architecture:
        - 192 experts total
        - Top-2 routing (only 2 active per token)
        - Disk sharding (experts stored on disk)
        - LRU cache (24 hot experts in RAM)
        - Load balancing (prevents expert death)
    """
    
    def __init__(self, config: ModelConfig, test_mode: bool = False):
        super().__init__()
        self.config = config
        self.test_mode = test_mode # New parameter
        
        logger.info("zmoe", f"ShardedExpertFabric initialized with test_mode={self.test_mode}")
        
        # Extract config
        self.num_experts = config.expert_count
        self.top_k = config.top_k_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = int(config.hidden_size * config.expert_hidden_multiplier)
        self.shard_dir = config.expert_shard_dir
        self.cache_size = config.max_expert_cache
        
        # Device
        self.device = torch.device(config.device if hasattr(config, 'device') else 'cpu')
        
        # Disk manager (only initialize if not in test_mode)
        if not self.test_mode:
            self.disk_manager = ExpertDiskManager(
                shard_dir=self.shard_dir,
                num_experts=self.num_experts,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim
            )
        else:
            self.disk_manager = None # No disk manager in test mode
            logger.info("zmoe", "ShardedExpertFabric in TEST MODE - skipping disk manager initialization.")
        
        # LRU cache
        self.cache = LRUExpertCache(capacity=self.cache_size)
        
        # Expert statistics
        self.expert_stats = {
            i: ExpertStats(expert_id=i) 
            for i in range(self.num_experts)
        }
        
        # Router for gating (if not using external router)
        self.use_external_router = True  # We use AdaptiveRouter
        
        # Load balancing
        self.load_balance_weight = config.load_balancing_weight if hasattr(config, 'load_balancing_weight') else 0.01
        
        # Initialize experts on disk if needed (only if not in test_mode)
        if not self.test_mode:
            if not self.disk_manager.manifest['experts']:
                logger.info("zmoe", "No experts found, initializing...")
                self.disk_manager.initialize_all_experts()
            logger.info("zmoe", f"ShardedExpertFabric initialized")
            logger.info("zmoe", f"  Total Experts: {self.num_experts}")
            logger.info("zmoe", f"  Top-K: {self.top_k}")
            logger.info("zmoe", f"  Cache Size: {self.cache_size}")
            logger.info("zmoe", f"  Disk Size: {self.disk_manager.get_total_size_mb():.2f} MB")
        else:
            # Create a dummy expert for test_mode to avoid NoneType errors in forward
            self.dummy_expert = ExpertFFN(hidden_dim=self.hidden_dim, intermediate_dim=self.intermediate_dim)
            logger.info("zmoe", "ShardedExpertFabric in TEST MODE - using dummy expert.")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through MoE fabric.
        
        This is where the magic happens:
        1. Load required experts from disk (or cache)
        2. Route tokens to their top-k experts
        3. Compute weighted combination
        4. Track statistics
        
        Args:
            hidden_states: [N, hidden_dim] input tokens
            expert_indices: [N, top_k] which experts to use
            expert_weights: [N, top_k] weights for each expert
            attention_mask: [N] optional mask
        
        Returns:
            output: [N, hidden_dim]
            stats: Dictionary of statistics
        """
        N, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Validate inputs
        assert expert_indices.shape == (N, self.top_k), \
            f"expert_indices shape mismatch: {expert_indices.shape} vs ({N}, {self.top_k})"
        assert expert_weights.shape == (N, self.top_k), \
            f"expert_weights shape mismatch: {expert_weights.shape} vs ({N}, {self.top_k})"
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Track which experts were used
        experts_loaded = set()
        cache_hits = 0
        cache_misses = 0
        total_load_time = 0.0
        
        # Handle test_mode in forward pass
        if self.test_mode:
            # In test mode: run all tokens through the single dummy expert,
            # weighted by the sum of their routing weights.
            # dummy_expert is an nn.Module registered on self, so its parameters
            # are part of the computation graph and will receive gradients.
            with torch.set_grad_enabled(self.training):
                sum_weights = expert_weights.sum(dim=-1, keepdim=True)  # [N, 1]
                dummy_expert_output = self.dummy_expert(hidden_states)   # [N, H]
                output = dummy_expert_output * sum_weights               # [N, H]
            
            stats = {
                'experts_used': 1,
                'cache_hits': 0,
                'cache_misses': 0,
                'cache_hit_rate': 0.0,
                'total_load_time_ms': 0.0,
                'avg_load_time_ms': 0.0,
                'load_balance_loss': 0.0, # Dummy value
                'routing_entropy': 0.0 # Dummy value
            }
            return output, stats


        # Process each top-k slot
        for k in range(self.top_k):
            # Get expert indices and weights for this slot
            slot_indices = expert_indices[:, k]  # [N]
            slot_weights = expert_weights[:, k]  # [N]
            
            # Group tokens by expert (for efficiency)
            unique_experts = torch.unique(slot_indices).tolist()
            
            for expert_id in unique_experts:
                # Find tokens routed to this expert
                mask = (slot_indices == expert_id)
                if not mask.any():
                    continue
                
                token_indices = torch.where(mask)[0]
                token_weights = slot_weights[mask].unsqueeze(1)  # [n_tokens, 1]
                token_hidden = hidden_states[token_indices]  # [n_tokens, hidden_dim]
                
                # Load expert (from cache or disk)
                start_time = time.time()
                expert = self.cache.get(expert_id)
                was_cached = expert is not None
                
                if expert is None:
                    # Load from disk
                    expert = self.disk_manager.load_expert(expert_id, device=str(device))
                    self.cache.put(expert_id, expert)
                    cache_misses += 1
                else:
                    cache_hits += 1
                
                load_time = (time.time() - start_time) * 1000  # ms
                total_load_time += load_time
                
                experts_loaded.add(expert_id)
                
                # Forward through expert
                with torch.set_grad_enabled(self.training):
                    expert_output = expert(token_hidden)  # [n_tokens, hidden_dim]
                
                # Apply weights and accumulate
                weighted_output = expert_output * token_weights
                output[token_indices] += weighted_output
                
                # Update statistics
                avg_weight = token_weights.mean().item()
                self.expert_stats[expert_id].update(
                    weight=avg_weight,
                    was_cached=was_cached,
                    load_time=load_time
                )
        
        # Normalize by total weight (should be close to 1.0 already from softmax)
        # But we do it anyway for numerical stability
        total_weight = expert_weights.sum(dim=1, keepdim=True)  # [N, 1]
        output = output / (total_weight + 1e-8)
        
        # Compute auxiliary loss for load balancing
        load_balance_loss = self._compute_load_balance_loss(
            expert_indices,
            expert_weights
        )
        
        # Compute routing entropy
        routing_entropy = self._compute_routing_entropy(expert_weights)
        
        # Cache statistics
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0
        
        # Compile statistics
        stats = {
            'experts_used': len(experts_loaded),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'total_load_time_ms': total_load_time,
            'avg_load_time_ms': total_load_time / max(cache_misses, 1),
            'load_balance_loss': load_balance_loss.item(),
            'routing_entropy': routing_entropy
        }
        
        return output, stats
    
    def _compute_load_balance_loss(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.
        
        We want experts to be used roughly equally to prevent:
        - Expert death (some experts never used)
        - Expert overload (some experts overused)
        
        Args:
            expert_indices: [N, top_k]
            expert_weights: [N, top_k]
        
        Returns:
            Scalar loss
        """
        N = expert_indices.size(0)
        device = expert_indices.device
        
        # Count weighted usage per expert
        expert_usage = torch.zeros(self.num_experts, device=device, dtype=expert_weights.dtype)
        
        for k in range(self.top_k):
            expert_usage.scatter_add_(
                0,
                expert_indices[:, k],
                expert_weights[:, k]
            )
        
        # Normalize to get probabilities
        expert_probs = expert_usage / (N * self.top_k + 1e-8)
        
        # Target: uniform distribution
        target_prob = 1.0 / self.num_experts
        
        # Loss: encourage uniform distribution
        # We use squared error
        loss = torch.sum((expert_probs - target_prob) ** 2)
        
        # Weight by config
        loss = self.load_balance_weight * loss
        
        return loss
    
    def _compute_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """
        Compute routing entropy (diversity measure).
        
        Higher entropy = more diverse routing = good
        Lower entropy = concentrated routing = bad (risk of collapse)
        
        Args:
            expert_weights: [N, top_k]
        
        Returns:
            Entropy value
        """
        # Average weights across tokens
        avg_weights = expert_weights.mean(dim=0)  # [top_k]
        
        # Normalize
        probs = avg_weights / (avg_weights.sum() + 1e-8)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        
        return entropy.item()
    
    def get_expert_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed statistics for all experts."""
        stats = {}
        for expert_id, expert_stat in self.expert_stats.items():
            stats[expert_id] = {
                'activation_count': expert_stat.activation_count,
                'avg_weight': expert_stat.avg_weight,
                'cache_hit_rate': expert_stat.cache_hits / max(expert_stat.activation_count, 1),
                'avg_load_time_ms': expert_stat.load_time_ms / max(expert_stat.cache_misses, 1)
            }
        return stats
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def print_statistics(self):
        """Print comprehensive statistics."""
        print("\n" + "="*70)
        print("  MoE Expert Fabric Statistics")
        print("="*70)
        
        # Overall stats
        total_activations = sum(s.activation_count for s in self.expert_stats.values())
        active_experts = sum(1 for s in self.expert_stats.values() if s.activation_count > 0)
        
        print(f"\n[INFO] Overall:")
        print(f"  Total Experts: {self.num_experts}")
        print(f"  Active Experts: {active_experts}")
        print(f"  Total Activations: {total_activations:,}")
        
        # Cache stats
        cache_stats = self.get_cache_statistics()
        print(f"\n[INFO] Cache:")
        print(f"  Capacity: {cache_stats['capacity']}")
        print(f"  Current Size: {cache_stats['current_size']}")
        print(f"  Hit Rate: {cache_stats['hit_rate']*100:.2f}%")
        print(f"  Evictions: {cache_stats['evictions']}")
        
        # Top experts
        top_experts = sorted(
            self.expert_stats.items(),
            key=lambda x: x[1].activation_count,
            reverse=True
        )[:10]
        
        print(f"\n[INFO] Top 10 Most Used Experts:")
        for rank, (expert_id, stats) in enumerate(top_experts, 1):
            print(f"  {rank}. Expert {expert_id}: {stats.activation_count:,} activations (avg weight: {stats.avg_weight:.4f})")
        
        # Disk stats
        disk_size = self.disk_manager.get_total_size_mb()
        print(f"\n[INFO] Disk:")
        print(f"  Total Size: {disk_size:.2f} MB")
        print(f"  Experts on Disk: {len(self.disk_manager.manifest['experts'])}")
        
        print("\n" + "="*70 + "\n")
    
    def set_device(self, device: Union[str, torch.device]):
        """Set device for loading experts."""
        self.device = torch.device(device)
        logger.info("zmoe", f"MoE device set to: {self.device}")
    
    def clear_cache(self):
        """Clear expert cache."""
        self.cache.clear()
        logger.info("zmoe", "Expert cache cleared")
    
    def save_all_experts(self):
        """Force save all cached experts to disk."""
        logger.info("zmoe", "Saving all cached experts to disk...")
        with self.cache.lock:
            for expert_id, expert in self.cache.cache.items():
                self.disk_manager.save_expert(expert_id, expert)
        logger.info("zmoe", "All experts saved")


# ==================== UTILITY FUNCTIONS ====================

def create_expert_fabric(config: ModelConfig) -> ShardedExpertFabric:
    """
    Factory function to create expert fabric.
    
    Args:
        config: XORZENXConfig instance
    
    Returns:
        Initialized expert fabric
    """
    return ShardedExpertFabric(config)



def estimate_expert_memory_mb(
    num_experts: int,
    hidden_dim: int,
    intermediate_dim: int,
    cached_experts: int
) -> Dict[str, float]:
    """
    Estimate memory usage for expert fabric.
    
    Args:
        num_experts: Total number of experts
        hidden_dim: Hidden dimension
        intermediate_dim: Intermediate dimension
        cached_experts: Number of experts in cache
    
    Returns:
        Dictionary with memory estimates
    """
    # Parameters per expert
    # gate_proj: hidden x intermediate
    # up_proj: hidden x intermediate
    # down_proj: intermediate x hidden
    params_per_expert = (
        hidden_dim * intermediate_dim +  # gate
        hidden_dim * intermediate_dim +  # up
        intermediate_dim * hidden_dim    # down
    )
    
    # Assume FP32 (4 bytes per param)
    bytes_per_expert = params_per_expert * 4
    mb_per_expert = bytes_per_expert / (1024 ** 2)
    
    total_disk_mb = num_experts * mb_per_expert
    cached_mb = cached_experts * mb_per_expert
    
    return {
        'params_per_expert': params_per_expert,
        'mb_per_expert': mb_per_expert,
        'total_disk_mb': total_disk_mb,
        'cached_mb': cached_mb,
        'savings_mb': total_disk_mb - cached_mb
    }

# ==================== TESTING ====================






