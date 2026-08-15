"""
This module implements a production-grade sharding system specifically designed
for managing Mixture-of-Experts (MoE) within the `xorzen-zero` framework.
It provides robust functionalities for efficient handling of a large number
of experts, primarily optimized for CPU-first deployments with disk-backed
storage and intelligent memory management.

Key Components and Features:
- **Expert Shard Manager (`ExpertShardManager`)**: Manages the lifecycle of
  expert parameters stored as shards on disk, including creation, saving,
  and loading, with support for memory-mapped I/O.
- **LRU Cache (`LRUCache`)**: A thread-safe, memory-limited cache for frequently
  accessed expert shards, optimizing retrieval times by keeping active experts
  in memory.
- **Memory-Mapped I/O (`MemoryMappedShard`)**: Enables efficient, direct access
  to expert parameters on disk, reducing memory overhead and improving I/O performance.
- **Gradient-Driven Sharding (`GradientDrivenSharding`)**: Implements an intelligent
  eviction and prefetching strategy based on expert importance derived from
  training gradients, ensuring that the most relevant experts are prioritized in cache.
- **Batched Expert Dispatcher (`BatchedExpertDispatcher`)**: Optimizes the routing
  and processing of tokens through experts by dispatching them in batches,
  thereby improving computational efficiency.

This sharding system is critical for scaling MoE models to billions or trillions
of parameters, allowing them to operate effectively even on hardware with
limited RAM by intelligently leveraging disk storage.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import copy
import pickle
import time
import threading
import queue
from collections import OrderedDict, defaultdict
import hashlib
import mmap
import struct
import warnings
from concurrent.futures import ThreadPoolExecutor, Future
import atexit
import gc

from xorzen.utils.logger import get_logger
from xorzen.utils.math_utils import TensorStability

logger = get_logger()

# ==================== CORE DATA STRUCTURES ====================

@dataclass
class ShardMetadata:
    """
    A dataclass storing comprehensive metadata for a single sharded expert.
    This includes its unique identifier, file path on disk, parameter count,
    memory footprint, timestamps for creation and last access, access counts,
    and current in-memory status. This metadata is crucial for efficient cache
    management and expert lifecycle tracking.
    """
    expert_id: int
    shard_path: Path
    parameter_count: int
    byte_size: int
    creation_time: float
    last_access_time: float
    access_count: int = 0
    in_memory: bool = False
    memory_address: Optional[int] = None
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `ShardMetadata` instance into a dictionary, suitable
        for serialization (e.g., to JSON). It ensures that path objects
        are converted to strings and omits internal pointers like
        `memory_address` if not relevant for persistence.
        
        Returns:
            A dictionary representation of the shard's metadata.
        """
        return {
            "expert_id": self.expert_id,
            "shard_path": str(self.shard_path),
            "parameter_count": self.parameter_count,
            "byte_size": self.byte_size,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "in_memory": self.in_memory,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ShardMetadata':
        """
        Constructs a `ShardMetadata` instance from a dictionary representation.
        This class method is used for deserialization, parsing string paths
        back into `Path` objects and handling default values for optional fields.
        
        Args:
            data (`Dict[str, Any]`): A dictionary containing the shard metadata.
            
        Returns:
            `ShardMetadata`: A new instance of `ShardMetadata` populated from the dictionary.
        """
        return cls(
            expert_id=data["expert_id"],
            shard_path=Path(data["shard_path"]),
            parameter_count=data["parameter_count"],
            byte_size=data["byte_size"],
            creation_time=data["creation_time"],
            last_access_time=data["last_access_time"],
            access_count=data.get("access_count", 0),
            in_memory=data.get("in_memory", False),
            checksum=data.get("checksum", ""),
        )


@dataclass
class MemoryStats:
    """
    A dataclass for accumulating and reporting memory usage statistics related
    to the Expert Shard Manager. It tracks total, used, cached, and disk memory,
    along with cache hit/miss ratios, load/save operations, and eviction counts.
    """
    total_memory: int = 0
    used_memory: int = 0
    cache_memory: int = 0
    disk_memory: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    load_operations: int = 0
    save_operations: int = 0
    eviction_count: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """
        Calculates the cache hit ratio (hits / (hits + misses)).
        This metric indicates the effectiveness of the cache in serving requests
        from memory rzero than disk.
        
        Returns:
            `float`: The cache hit ratio, a value between 0.0 and 1.0.
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def memory_usage_ratio(self) -> float:
        """
        Calculates the ratio of currently used memory to the total allocated memory.
        This metric provides an indication of how much of the available memory
        is actively being consumed.
        
        Returns:
            `float`: The memory usage ratio, a value between 0.0 and 1.0.
        """
        return self.used_memory / self.total_memory if self.total_memory > 0 else 0.0
    
    def update_usage(self, used_memory: int):
        """
        Updates the tracked amount of currently used memory.
        
        Args:
            used_memory (`int`): The new value for the amount of memory
                                 currently in use, in bytes.
        """
        self.used_memory = used_memory
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `MemoryStats` instance into a dictionary representation,
        including computed properties like `hit_ratio` and `memory_usage_ratio`.
        
        Returns:
            A dictionary representing the current memory statistics.
        """
        return {
            "total_memory": self.total_memory,
            "used_memory": self.used_memory,
            "cache_memory": self.cache_memory,
            "disk_memory": self.disk_memory,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_ratio": self.hit_ratio,
            "load_operations": self.load_operations,
            "save_operations": self.save_operations,
            "eviction_count": self.eviction_count,
            "memory_usage_ratio": self.memory_usage_ratio,
        }


# ==================== LRU CACHE IMPLEMENTATION ====================

class LRUCache:
    """
    Implements a thread-safe Least Recently Used (LRU) cache with configurable
    memory limits and eviction policies. It is designed to efficiently store
    and retrieve items (e.g., expert shards) in memory, automatically
    evicting less-used items when the memory limit is approached.
    """
    
    def __init__(self, max_memory_bytes: int, eviction_policy: str = "lru"):
        """
        Initializes the LRU cache with a specified memory limit and eviction policy.
        
        Args:
            max_memory_bytes (`int`): The maximum amount of memory (in bytes) that
                                      the cache is allowed to consume.
            eviction_policy (`str`, *optional*): The strategy to use when the cache
                                                 memory limit is exceeded.
                                                 Supported: "lru" (Least Recently Used),
                                                 "lfu" (Least Frequently Used),
                                                 "lifo" (Last-In, First-Out, non-standard).
                                                 Defaults to "lru".
        """
        self.max_memory_bytes = max_memory_bytes
        self.eviction_policy = eviction_policy
        
        # Core data structures
        self.cache: OrderedDict[int, Any] = OrderedDict()  # key -> (value, size, access_time)
        self.access_times: Dict[int, float] = {}
        self.access_counts: Dict[int, int] = defaultdict(int)
        
        # Memory tracking
        self.current_memory = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.debug("sharding", f"LRU Cache initialized with {max_memory_bytes/1e9:.2f} GB limit")
    
    def get(self, key: int) -> Optional[Any]:
        """
        Retrieves an item from the cache using its unique key. If the item is
        found, its access time is updated (for LRU policy), and it's moved
        to the end of the cache (marking it as most recently used).
        
        Args:
            key (`int`): The unique identifier of the item to retrieve.
            
        Returns:
            `Optional[Any]`: The cached item if found, otherwise `None`.
        """
        with self.lock:
            if key in self.cache:
                # Update access
                value, size, _ = self.cache[key]
                self.cache[key] = (value, size, time.time())
                self.access_times[key] = time.time()
                self.access_counts[key] += 1
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
    
    def put(self, key: int, value: Any, size: int):
        """
        Adds or updates an item in the cache. If the item already exists,
        its value and size are updated. If adding a new item causes the
        cache to exceed its `max_memory_bytes` limit, items are
        automatically evicted based on the configured `eviction_policy`.
        
        Args:
            key (`int`): The unique identifier for the item.
            value (`Any`): The item to be stored in the cache.
            size (`int`): The memory footprint of the item in bytes.
        """
        with self.lock:
            # Check if item already exists
            if key in self.cache:
                # Update existing
                old_value, old_size, _ = self.cache[key]
                self.current_memory -= old_size
                self.cache[key] = (value, size, time.time())
                self.current_memory += size
                
                # Move to end
                self.cache.move_to_end(key)
                return
            
            # Evict if necessary
            while self.current_memory + size > self.max_memory_bytes and self.cache:
                self._evict_one()
            
            # Add new item
            self.cache[key] = (value, size, time.time())
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
            self.current_memory += size
    
    def _evict_one(self):
        """
        Evicts a single item from the cache based on the configured `eviction_policy`.
        Supported policies include Least Recently Used (LRU), Least Frequently Used (LFU),
        and Last-In, First-Out (LIFO). This method is called internally when the
                    cache's memory limit is exceeded.
        """
        if not self.cache:
        
            return
        
        if self.eviction_policy == "lfu":
            # Least Frequently Used
            if not self.access_counts:
                # Fallback to LRU if no access counts recorded
                key = next(iter(self.cache))
            else:
                key = min(self.access_counts.items(), key=lambda x: x[1])[0]
        elif self.eviction_policy == "lifo":
             # Last In, First Out
            key = next(reversed(self.cache))
        else:
            # Default to LRU/FIFO
            key = next(iter(self.cache))
        
        # Remove from cache
        if key in self.cache:
            value, size, _ = self.cache.pop(key)
            self.current_memory -= size
            
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
            
            self.evictions += 1
            
            # Force garbage collection
            del value
            gc.collect()
    
    def remove(self, key: int) -> bool:
        """
        Remove item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if key in self.cache:
                value, size, _ = self.cache.pop(key)
                self.current_memory -= size
                
                if key in self.access_times:
                    del self.access_times[key]
                if key in self.access_counts:
                    del self.access_counts[key]
                
                del value
                return True
            return False
    
    def clear(self):
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.current_memory = 0
            gc.collect()
    
    def contains(self, key: int) -> bool:
        """Check if key is in cache."""
        with self.lock:
            return key in self.cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "memory_used": self.current_memory,
                "memory_limit": self.max_memory_bytes,
                "memory_usage_ratio": self.current_memory / self.max_memory_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                "evictions": self.evictions,
            }


# ==================== MEMORY-MAPPED I/O ====================

class MemoryMappedShard:
    """
    Memory-mapped shard for efficient disk I/O.
    """
    
    def __init__(self, shard_path: Path, mode: str = "r", 
                 storage_numpy_dtype: np.dtype = np.float32, storage_element_size: int = 4):
        """
        Initialize memory-mapped shard.
        
        Args:
            shard_path: Path to shard file
            mode: Access mode ('r' for read, 'r+' for read/write)
            storage_numpy_dtype: NumPy dtype used for storage
            storage_element_size: Element size in bytes for storage
        """
        self.shard_path = shard_path
        self.mode = mode
        self.storage_numpy_dtype = storage_numpy_dtype
        self.storage_element_size = storage_element_size
        self.mmap_obj = None
        self.file_handle = None
        self._open()
    
    def _open(self):
        """Open memory-mapped file."""
        try:
            # Open file
            if self.mode == "r":
                self.file_handle = open(self.shard_path, "rb")
            elif self.mode == "r+":
                self.file_handle = open(self.shard_path, "r+b")
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            
            # Create memory map
            self.mmap_obj = mmap.mmap(
                self.file_handle.fileno(),
                0,  # Map entire file
                access=mmap.ACCESS_READ if self.mode == "r" else mmap.ACCESS_WRITE
            )
            
        except Exception as e:
            logger.error("sharding", f"Failed to memory-map {self.shard_path}: {e}")
            self.close()
            raise
    
    def read_tensor(self, offset: int, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        Read tensor from memory-mapped file.
        
        Args:
            offset: Byte offset in file
            shape: Tensor shape
            dtype: Tensor data type
            
        Returns:
            Tensor read from file
        """
        if self.mmap_obj is None:
            raise RuntimeError("Memory map not initialized")
        
        # Calculate size based on storage element size
        element_size_storage = self.storage_element_size
            
        num_elements = np.prod(shape)
        size_bytes = num_elements * element_size_storage # Use storage element size
        
        # Read bytes
        self.mmap_obj.seek(offset)
        buffer = self.mmap_obj.read(size_bytes)
        
        if len(buffer) != size_bytes:
            raise RuntimeError(f"Read {len(buffer)} bytes, expected {size_bytes}")
        
        # Convert to tensor
        numpy_array = np.frombuffer(buffer, dtype=self.storage_numpy_dtype)
        numpy_array = numpy_array.reshape(shape)
        
        torch_tensor = torch.from_numpy(numpy_array).clone() # Use .clone() to ensure a new tensor
        
        # Convert back to original dtype if it was bfloat16
        if dtype == torch.bfloat16:
            return torch_tensor.to(torch.bfloat16)
        
        return torch_tensor
    
    def write_tensor(self, offset: int, tensor: torch.Tensor):
        """
        Write tensor to memory-mapped file.
        
        Args:
            offset: Byte offset in file
            tensor: Tensor to write
        """
        if self.mode != "r+":
            raise RuntimeError("File not opened in write mode")
        
        if self.mmap_obj is None:
            raise RuntimeError("Memory map not initialized")
        
        # Convert to bytes
        # Ensure tensor is float32 for numpy conversion if original was bfloat16
        if tensor.dtype == torch.bfloat16:
            numpy_array = tensor.cpu().to(torch.float32).numpy()
        else:
            numpy_array = tensor.cpu().numpy()
        buffer = numpy_array.tobytes()
        
        # Write bytes
        self.mmap_obj.seek(offset)
        self.mmap_obj.write(buffer)
        self.mmap_obj.flush()
    
    def close(self):
        """Close memory map and file."""
        if self.mmap_obj is not None:
            try:
                self.mmap_obj.flush()
                self.mmap_obj.close()
            except:
                pass
            self.mmap_obj = None
        
        if self.file_handle is not None:
            try:
                self.file_handle.close()
            except:
                pass
            self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ==================== EXPERT SHARD MANAGER ====================

class ExpertShardManager:
    """
    Manages disk-backed expert shards with LRU caching.
    """
    
    @staticmethod
    def _torch_to_numpy_dtype(dtype: torch.dtype) -> np.dtype:
        """Convert torch dtype to numpy dtype."""
        dtype_map = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.float32,  # Map bfloat16 to float32 for NumPy compatibility
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        return dtype_map[dtype]
    
    def __init__(
        self,
        shard_dir: Union[str, Path],
        max_cache_memory_gb: float = 4.0,
        dtype: torch.dtype = torch.float32,
        use_memory_map: bool = True,
        preload_frequent: bool = True,
        num_io_workers: int = 4
    ):
        """
        Initialize expert shard manager.
        
        Args:
            shard_dir: Directory for shard files
            max_cache_memory_gb: Maximum cache memory in GB
            dtype: Data type for parameters
            use_memory_map: Use memory-mapped I/O
            preload_frequent: Preload frequently used experts
            num_io_workers: Number of I/O worker threads
        """
        self.shard_dir = Path(shard_dir)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        
        self.dtype = dtype # Original desired dtype
        # Determine the actual size of elements on disk
        if self.dtype == torch.bfloat16:
            self.storage_element_size = np.dtype(np.float32).itemsize # 4 bytes for float32
            self.storage_numpy_dtype = np.float32
        else:
            self.storage_element_size = torch.tensor(0, dtype=dtype).element_size()
            self.storage_numpy_dtype = self._torch_to_numpy_dtype(self.dtype)
        
        self.use_memory_map = use_memory_map        
        # Initialize cache
        max_memory_bytes = int(max_cache_memory_gb * 1024**3)
        self.cache = LRUCache(max_memory_bytes, eviction_policy="lru")
        
        # Metadata storage
        self.metadata_file = self.shard_dir / "metadata.json"
        self.metadata: Dict[int, ShardMetadata] = self._load_metadata()
        
        # Expert templates (for creating new experts)
        self.expert_templates: Dict[str, nn.Module] = {}
        
        # I/O workers
        self.io_executor = ThreadPoolExecutor(max_workers=num_io_workers)
        self.pending_operations: Dict[int, Future] = {}
        
        # Statistics
        self.stats = MemoryStats()
        self.stats.total_memory = max_memory_bytes
        
        # Preload frequent experts if enabled
        if preload_frequent and self.metadata:
            self._preload_frequent_experts()
        
        # Register cleanup
        atexit.register(self.cleanup)
        
        logger.info("sharding", 
                   f"ExpertShardManager initialized: {len(self.metadata)} experts, "
                   f"{max_cache_memory_gb:.1f} GB cache")
    
    def _load_metadata(self) -> Dict[int, ShardMetadata]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error("sharding", f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            metadata_dict = {
                str(expert_id): meta.to_dict()
                for expert_id, meta in self.metadata.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            logger.error("sharding", f"Failed to save metadata: {e}")
    
    def _preload_frequent_experts(self, top_k: int = 10):
        """Preload most frequently used experts."""
        # Sort by access count
        sorted_meta = sorted(
            self.metadata.values(),
            key=lambda m: m.access_count,
            reverse=True
        )
        
        for meta in sorted_meta[:top_k]:
            if meta.access_count > 10:  # Only preload truly frequent
                self.io_executor.submit(self._load_expert_sync, meta.expert_id)
    
    def register_expert_template(self, name: str, template: nn.Module):
        """
        Register an expert template.
        
        Args:
            name: Template name
            template: Expert template module
        """
        self.expert_templates[name] = template
        logger.debug("sharding", f"Registered expert template: {name}")
    
    def create_expert(
        self,
        expert_id: int,
        template_name: str,
        initialization: str = "xavier_uniform"
    ) -> nn.Module:
        """
        Create a new expert.
        
        Args:
            expert_id: Expert ID
            template_name: Name of template to use
            initialization: Weight initialization method
            
        Returns:
            Created expert module
        """
        if template_name not in self.expert_templates:
            raise ValueError(f"Unknown expert template: {template_name}")
        
        # Create expert from template
        expert = copy.deepcopy(self.expert_templates[template_name])
        
        # Initialize weights
        self._initialize_expert(expert, initialization)
        
        # Save to disk
        self.save_expert(expert_id, expert)
        
        logger.info("sharding", f"Created new expert {expert_id} from template {template_name}")
        
        return expert
    
    def _initialize_expert(self, expert: nn.Module, method: str):
        """Initialize expert weights."""
        for name, param in expert.named_parameters():
            if 'weight' in name:
                if method == "xavier_uniform":
                    nn.init.xavier_uniform_(param)
                elif method == "xavier_normal":
                    nn.init.xavier_normal_(param)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(param)
                elif method == "kaiming_normal":
                    nn.init.kaiming_normal_(param)
                else:
                    nn.init.xavier_uniform_(param)  # Default
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def save_expert(self, expert_id: int, expert: nn.Module):
        """
        Save expert to disk.
        
        Args:
            expert_id: Expert ID
            expert: Expert module to save
        """
        # Create shard path
        shard_path = self.shard_dir / f"expert_{expert_id:06d}.pt"
        
        # Save using memory map if enabled
        if self.use_memory_map:
            self._save_expert_memory_mapped(expert_id, expert, shard_path)
        else:
            self._save_expert_pytorch(expert_id, expert, shard_path)
        
        # Update metadata
        metadata = ShardMetadata(
            expert_id=expert_id,
            shard_path=shard_path,
            parameter_count=sum(p.numel() for p in expert.parameters()),
            byte_size=shard_path.stat().st_size,
            creation_time=time.time(),
            last_access_time=time.time(),
            access_count=1,
            checksum=self._calculate_checksum(expert),
        )
        
        self.metadata[expert_id] = metadata
        self._save_metadata()
        
        self.stats.save_operations += 1
        self.stats.disk_memory += metadata.byte_size
        
        logger.debug("sharding", f"Saved expert {expert_id} to {shard_path}")
    
    def _save_expert_memory_mapped(self, expert_id: int, expert: nn.Module, shard_path: Path):
        """Save expert using memory-mapped I/O."""
        # Collect all parameters
        parameters = []
        shapes = []
        
        for param in expert.parameters():
            parameters.append(param.detach().cpu())
            shapes.append(param.shape)
        
        # Calculate total size
        total_elements = sum(p.numel() for p in parameters)
        total_bytes = total_elements * self.storage_element_size
        
        # Create file with appropriate size
        with open(shard_path, 'wb') as f:
            f.seek(total_bytes - 1)
            f.write(b'\x00')
        
        # Write parameters using memory map
        with MemoryMappedShard(shard_path, mode='r+', 
                               storage_numpy_dtype=self.storage_numpy_dtype, 
                               storage_element_size=self.storage_element_size) as mmap_shard:
            offset = 0
            for param, shape in zip(parameters, shapes):
                size_bytes = param.numel() * self.storage_element_size
                mmap_shard.write_tensor(offset, param)
                offset += size_bytes
    
    def _save_expert_pytorch(self, expert_id: int, expert: nn.Module, shard_path: Path):
        """Save expert using PyTorch serialization."""
        torch.save(expert.state_dict(), shard_path)
    
    def load_expert(self, expert_id: int) -> nn.Module:
        """
        Load expert from disk or cache.
        
        Args:
            expert_id: Expert ID
            
        Returns:
            Loaded expert module
        """
        # Check cache first
        cached = self.cache.get(expert_id)
        if cached is not None:
            self.stats.cache_hits += 1
            self._update_metadata_access(expert_id)
            return cached
        
        # Not in cache, load from disk
        self.stats.cache_misses += 1
        
        # Check if expert exists
        if expert_id not in self.metadata:
            raise ValueError(f"Expert {expert_id} not found")
        
        # Load synchronously or asynchronously based on policy
        expert = self._load_expert_sync(expert_id)
        
        # Update cache
        expert_size = self._calculate_expert_size(expert)
        self.cache.put(expert_id, expert, expert_size)
        
        # Update statistics
        self.stats.cache_memory = self.cache.current_memory
        self.stats.load_operations += 1
        
        return expert
    
    def _load_expert_sync(self, expert_id: int) -> nn.Module:
        """Synchronously load expert from disk."""
        metadata = self.metadata[expert_id]
        
        # Check if we have a template
        if not self.expert_templates:
            raise RuntimeError("No expert templates registered")
        
        # Create empty expert from first template
        template_name = list(self.expert_templates.keys())[0]
        expert = copy.deepcopy(self.expert_templates[template_name])
        
        # Load parameters
        if self.use_memory_map and metadata.shard_path.suffix == '.pt':
            self._load_expert_memory_mapped(expert_id, expert, metadata)
        else:
            self._load_expert_pytorch(expert_id, expert, metadata)
        
        # Update metadata
        self._update_metadata_access(expert_id)
        
        return expert
    
    def _load_expert_memory_mapped(self, expert_id: int, expert: nn.Module, metadata: ShardMetadata):
        """Load expert using memory-mapped I/O."""
        with MemoryMappedShard(metadata.shard_path, mode='r', 
                               storage_numpy_dtype=self.storage_numpy_dtype, 
                               storage_element_size=self.storage_element_size) as mmap_shard:
            offset = 0
            
            for name, param in expert.named_parameters():
                shape = param.shape
                num_elements = param.numel()
                size_bytes = num_elements * self.storage_element_size
                
                # Read tensor
                tensor = mmap_shard.read_tensor(offset, shape, self.dtype)
                
                # Update parameter
                param.data.copy_(tensor)
                
                offset += size_bytes
    
    def _load_expert_pytorch(self, expert_id: int, expert: nn.Module, metadata: ShardMetadata):
        """Load expert using PyTorch serialization."""
        state_dict = torch.load(metadata.shard_path, map_location='cpu')
        expert.load_state_dict(state_dict)
    
    def _update_metadata_access(self, expert_id: int):
        """Update metadata access statistics."""
        if expert_id in self.metadata:
            metadata = self.metadata[expert_id]
            metadata.last_access_time = time.time()
            metadata.access_count += 1
            metadata.in_memory = True
            
            # Save metadata periodically
            if metadata.access_count % 100 == 0:
                self._save_metadata()
    
    def _calculate_expert_size(self, expert: nn.Module) -> int:
        """Calculate memory size of expert in bytes."""
        total_params = sum(p.numel() for p in expert.parameters())
        return total_params * self.storage_element_size
    
    def _calculate_checksum(self, expert: nn.Module) -> str:
        """Calculate checksum of expert parameters."""
        # Concatenate all parameter data
        data = b""
        for param in expert.parameters():
            # Convert to bytes
            param_bytes = param.detach().cpu().numpy().tobytes()
            data += param_bytes
        
        # Calculate SHA256 hash
        return hashlib.sha256(data).hexdigest()
    
    def prefetch_experts(self, expert_ids: List[int]):
        """
        Prefetch experts asynchronously.
        
        Args:
            expert_ids: List of expert IDs to prefetch
        """
        for expert_id in expert_ids:
            if not self.cache.contains(expert_id):
                future = self.io_executor.submit(self._load_expert_sync, expert_id)
                self.pending_operations[expert_id] = future
    
    def get_expert_if_cached(self, expert_id: int) -> Optional[nn.Module]:
        """
        Get expert only if it's in cache.
        
        Args:
            expert_id: Expert ID
            
        Returns:
            Expert if cached, None otherwise
        """
        return self.cache.get(expert_id)
    
    def unload_expert(self, expert_id: int):
        """
        Unload expert from cache.
        
        Args:
            expert_id: Expert ID to unload
        """
        self.cache.remove(expert_id)
        
        if expert_id in self.metadata:
            self.metadata[expert_id].in_memory = False
    
    def cleanup(self):
        """Cleanup resources."""
        # Wait for pending operations
        for future in self.pending_operations.values():
            try:
                future.result(timeout=1.0)
            except:
                pass
        
        # Shutdown executor
        self.io_executor.shutdown(wait=True)
        
        # Save metadata
        self._save_metadata()
        
        logger.info("sharding", "ExpertShardManager cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        cache_stats = self.cache.get_stats()
        memory_stats = self.stats.to_dict()
        
        return {
            "cache": cache_stats,
            "memory": memory_stats,
            "experts": {
                "total": len(self.metadata),
                "in_memory": sum(1 for m in self.metadata.values() if m.in_memory),
                "on_disk": len(self.metadata),
            },
            "io": {
                "pending_operations": len(self.pending_operations),
                "io_workers": self.io_executor._max_workers,
            }
        }


# ==================== GDS (GRADIENT-DRIVEN SHARDING) ====================

class GradientDrivenSharding:
    """
    Gradient-driven sharding for intelligent expert placement.
    """
    
    def __init__(
        self,
        shard_manager: ExpertShardManager,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ):
        """
        Initialize GDS.
        
        Args:
            shard_manager: Expert shard manager
            temperature: Softmax temperature for expert selection
            learning_rate: Learning rate for gradient updates
            momentum: Momentum for gradient updates
        """
        self.shard_manager = shard_manager
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Expert statistics
        self.expert_gradients: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.expert_importance: Dict[int, float] = defaultdict(float)
        self.expert_momentum: Dict[int, float] = defaultdict(float)
        
        # Placement decisions
        self.placement_cache: Dict[int, bool] = {}  # expert_id -> in_memory
        
        # Thread safety
        self.lock = threading.RLock()
    
    def record_gradient(self, expert_id: int, gradient: torch.Tensor):
        """
        Record gradient for expert.
        
        Args:
            expert_id: Expert ID
            gradient: Gradient tensor
        """
        with self.lock:
            if expert_id not in self.expert_gradients:
                self.expert_gradients[expert_id] = []
            
            # Store gradient norm as importance measure
            importance = gradient.norm().item()
            self.expert_gradients[expert_id].append(importance)
            
            # Keep only recent gradients
            if len(self.expert_gradients[expert_id]) > 100:
                self.expert_gradients[expert_id].pop(0)
    
    def update_importance(self, expert_id: int):
        """Update importance score for expert."""
        with self.lock:
            if expert_id in self.expert_gradients and self.expert_gradients[expert_id]:
                # Average of recent gradient norms
                recent_grads = self.expert_gradients[expert_id][-10:]  # Last 10
                new_importance = np.mean(recent_grads)
                
                # Update with momentum
                old_importance = self.expert_importance.get(expert_id, 0.0)
                momentum = self.expert_momentum.get(expert_id, 0.0)
                
                # Momentum update
                momentum = self.momentum * momentum + (1 - self.momentum) * new_importance
                importance = old_importance + self.learning_rate * momentum
                
                self.expert_importance[expert_id] = importance
                self.expert_momentum[expert_id] = momentum
    
    def decide_placement(self, expert_ids: List[int], available_memory: float) -> Dict[int, bool]:
        """
        Decide which experts to keep in memory.
        
        Args:
            expert_ids: List of expert IDs to consider
            available_memory: Available memory ratio (0-1)
            
        Returns:
            Dictionary of expert_id -> in_memory
        """
        with self.lock:
            # Update importance for all experts
            for expert_id in expert_ids:
                self.update_importance(expert_id)
            
            # Get importance scores
            importance_scores = {}
            for expert_id in expert_ids:
                importance_scores[expert_id] = self.expert_importance.get(expert_id, 0.0)
            
            # Softmax over temperature
            scores_array = np.array(list(importance_scores.values()))
            if self.temperature > 0:
                scores_array = scores_array / self.temperature
                exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
                probs = exp_scores / exp_scores.sum()
            else:
                # Hard selection (temperature = 0)
                probs = np.zeros_like(scores_array)
                probs[np.argmax(scores_array)] = 1.0
            
            # Decide placement based on probabilities and available memory
            decisions = {}
            sorted_indices = np.argsort(probs)[::-1]  # Descending
            
            experts_to_keep = int(len(expert_ids) * available_memory)
            experts_to_keep = max(1, experts_to_keep)  # Always keep at least one
            
            for i, idx in enumerate(sorted_indices):
                expert_id = expert_ids[idx]
                decisions[expert_id] = (i < experts_to_keep)
            
            # Update placement cache
            self.placement_cache.update(decisions)
            
            return decisions
    
    def prefetch_based_on_placement(self, decisions: Dict[int, bool]):
        """Prefetch experts based on placement decisions."""
        experts_to_prefetch = [expert_id for expert_id, in_memory in decisions.items() 
                              if in_memory]
        
        self.shard_manager.prefetch_experts(experts_to_prefetch)
    
    def get_importance_scores(self) -> Dict[int, float]:
        """Get importance scores for all experts."""
        with self.lock:
            return dict(self.expert_importance)
    
    def reset(self):
        """Reset GDS state."""
        with self.lock:
            self.expert_gradients.clear()
            self.expert_importance.clear()
            self.expert_momentum.clear()
            self.placement_cache.clear()


# ==================== BATCHED EXPERT DISPATCH ====================

class BatchedExpertDispatcher:
    """
    Efficient batched dispatch of tokens to experts.
    """
    
    def __init__(
        self,
        shard_manager: ExpertShardManager,
        gds: Optional[GradientDrivenSharding] = None,
        batch_size: int = 32,
        use_bfloat16: bool = True
    ):
        """
        Initialize batched dispatcher.
        
        Args:
            shard_manager: Expert shard manager
            gds: Gradient-driven sharding (optional)
            batch_size: Batch size for expert computation
            use_bfloat16: Use bfloat16 for expert computation
        """
        self.shard_manager = shard_manager
        self.gds = gds
        self.batch_size = batch_size
        self.use_bfloat16 = use_bfloat16
        
        # Buffers for batching
        self.expert_buffers: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self.expert_indices: Dict[int, List[int]] = defaultdict(list)  # Original token indices
        
        # Statistics
        self.tokens_processed = 0
        self.expert_calls = 0
        self.batch_efficiency = 0.0
        
    def dispatch(
        self,
        x: torch.Tensor,  # [batch_size, seq_len, hidden_dim]
        expert_weights: torch.Tensor,  # [batch_size, seq_len, top_k]
        expert_indices: torch.Tensor,  # [batch_size, seq_len, top_k]
    ) -> torch.Tensor:
        """
        Dispatch tokens to experts and return combined output.
        
        Args:
            x: Input tensor
            expert_weights: Expert weights per token
            expert_indices: Expert indices per token
            
        Returns:
            Combined output tensor
        """
        batch_size, seq_len, hidden_dim = x.shape
        top_k = expert_weights.shape[-1]
        
        # Flatten for processing
        x_flat = x.view(-1, hidden_dim)
        weights_flat = expert_weights.view(-1, top_k)
        indices_flat = expert_indices.view(-1, top_k)
        num_tokens = x_flat.shape[0]
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Process each top-k slot
        for k in range(top_k):
            # Get tokens for this slot
            slot_weights = weights_flat[:, k]
            slot_indices = indices_flat[:, k]
            
            # Group tokens by expert
            expert_to_tokens = defaultdict(list)
            expert_to_positions = defaultdict(list)
            
            for token_idx in range(num_tokens):
                expert_id = slot_indices[token_idx].item()
                if slot_weights[token_idx] > 0.01:  # Threshold
                    expert_to_tokens[expert_id].append(x_flat[token_idx])
                    expert_to_positions[expert_id].append(token_idx)
            
            # Process each expert
            for expert_id, tokens in expert_to_tokens.items():
                if not tokens:
                    continue
                
                positions = expert_to_positions[expert_id]
                tokens_tensor = torch.stack(tokens)
                weights_tensor = slot_weights[positions].unsqueeze(-1)
                
                # Load expert
                expert = self.shard_manager.load_expert(expert_id)
                if self.use_bfloat16:
                    expert = expert.to(torch.bfloat16)
                
                # Apply expert
                with torch.no_grad():
                    if self.use_bfloat16 and tokens_tensor.dtype != torch.bfloat16:
                        tokens_tensor = tokens_tensor.to(torch.bfloat16)
                    
                    expert_output = expert(tokens_tensor)
                    
                    if self.use_bfloat16 and expert_output.dtype != x_flat.dtype:
                        expert_output = expert_output.to(x_flat.dtype)
                
                # Apply weights and accumulate
                weighted_output = expert_output * weights_tensor
                
                # Scatter back
                for i, pos in enumerate(positions):
                    output_flat[pos] += weighted_output[i]
                
                # Update statistics
                self.tokens_processed += len(tokens)
                self.expert_calls += 1
        
        # Reshape back
        output = output_flat.view(batch_size, seq_len, hidden_dim)
        
        # Update batch efficiency
        self.batch_efficiency = self.tokens_processed / (self.expert_calls * self.batch_size + 1e-12)
        
        return output
    
    def dispatch_batched(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched dispatch with accumulation.
        
        Args:
            x: Input tensor
            expert_weights: Expert weights
            expert_indices: Expert indices
            
        Returns:
            Combined output
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten
        x_flat = x.view(-1, hidden_dim)
        weights_flat = expert_weights.view(-1, expert_weights.shape[-1])
        indices_flat = expert_indices.view(-1, expert_indices.shape[-1])
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Group by expert for each top-k
        for k in range(expert_weights.shape[-1]):
            output_flat += self._process_expert_slot(
                x_flat, weights_flat[:, k], indices_flat[:, k], k
            )
        
        return output_flat.view(batch_size, seq_len, hidden_dim)
    
    def _process_expert_slot(
        self,
        x_flat: torch.Tensor,
        slot_weights: torch.Tensor,
        slot_indices: torch.Tensor,
        slot_idx: int
    ) -> torch.Tensor:
        """Process one expert slot."""
        output_slot = torch.zeros_like(x_flat)
        
        # Find unique experts in this slot
        unique_experts = torch.unique(slot_indices).tolist()
        
        for expert_id in unique_experts:
            # Get tokens for this expert
            mask = (slot_indices == expert_id) & (slot_weights > 0.01)
            if not mask.any():
                continue
            
            token_indices = torch.where(mask)[0]
            tokens = x_flat[token_indices]
            weights = slot_weights[token_indices].unsqueeze(-1)
            
            # Load expert
            expert = self.shard_manager.load_expert(expert_id)
            if self.use_bfloat16:
                expert = expert.to(torch.bfloat16)
            
            # Process in batches
            num_tokens = tokens.shape[0]
            for i in range(0, num_tokens, self.batch_size):
                batch_tokens = tokens[i:i+self.batch_size]
                batch_weights = weights[i:i+self.batch_size]
                batch_indices = token_indices[i:i+self.batch_size]
                
                # Process batch
                with torch.no_grad():
                    if self.use_bfloat16:
                        batch_tokens = batch_tokens.to(torch.bfloat16)
                    
                    batch_output = expert(batch_tokens)
                    
                    if self.use_bfloat16:
                        batch_output = batch_output.to(x_flat.dtype)
                
                # Weight and accumulate
                weighted_output = batch_output * batch_weights
                output_slot[batch_indices] += weighted_output
        
        return output_slot
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "tokens_processed": self.tokens_processed,
            "expert_calls": self.expert_calls,
            "batch_efficiency": self.batch_efficiency,
            "average_tokens_per_expert": self.tokens_processed / (self.expert_calls + 1e-12),
        }


# ==================== MAIN SHARDING SYSTEM ====================

class XORZENXShardingSystem:
    """
    Main sharding system for xorzen-zero.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu"
    ):
        """
        Initialize sharding system.
        
        Args:
            config: Configuration dictionary
            device: Device for computation
        """
        self.config = config
        self.device = device
        
        # Extract config values
        shard_dir = config.get("shard_dir", "experts/")
        max_cache_gb = config.get("max_cache_gb", 4.0)
        num_experts = config.get("num_experts", 192)
        expert_hidden = config.get("expert_hidden", 2048)
        expert_multiplier = config.get("expert_multiplier", 4.0)
        
        # Initialize shard manager
        self.shard_manager = ExpertShardManager(
            shard_dir=shard_dir,
            max_cache_memory_gb=max_cache_gb,
            dtype=torch.bfloat16 if config.get("use_bfloat16", True) else torch.float32,
            use_memory_map=config.get("use_memory_map", True),
            preload_frequent=config.get("preload_frequent", True),
            num_io_workers=config.get("num_io_workers", 4)
        )
        
        # Register expert template
        hidden_dim = config.get("hidden_size", 512) # Default for safety
        self._register_expert_template(hidden_dim, expert_multiplier)
        
        # Initialize GDS if enabled
        if config.get("use_gds", True):
            self.gds = GradientDrivenSharding(
                shard_manager=self.shard_manager,
                temperature=config.get("gds_temperature", 1.0),
                learning_rate=config.get("gds_learning_rate", 0.01),
                momentum=config.get("gds_momentum", 0.9)
            )
        else:
            self.gds = None
        
        # Initialize dispatcher
        self.dispatcher = BatchedExpertDispatcher(
            shard_manager=self.shard_manager,
            gds=self.gds,
            batch_size=config.get("batch_size", 32),
            use_bfloat16=config.get("use_bfloat16", True)
        )
        
        # Create experts if they don't exist
        self._initialize_experts(num_experts)
        
        logger.info("sharding", f"XORZENXShardingSystem initialized with {num_experts} experts")
    
    def _register_expert_template(self, hidden_dim: int, multiplier: float):
        """Register expert template."""
        expert_hidden = int(hidden_dim * multiplier)
        
        template = nn.Sequential(
            nn.Linear(hidden_dim, expert_hidden),
            nn.GELU(),
            nn.Linear(expert_hidden, hidden_dim)
        )
        
        # Initialize
        for layer in template:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        self.shard_manager.register_expert_template("default", template)
    
    def _initialize_experts(self, num_experts: int):
        """Initialize experts on disk."""
        existing_experts = len(self.shard_manager.metadata)
        
        if existing_experts >= num_experts:
            logger.info("sharding", f"Found {existing_experts} existing experts")
            return
        
        # Create missing experts
        for expert_id in range(existing_experts, num_experts):
            self.shard_manager.create_expert(
                expert_id=expert_id,
                template_name="default",
                initialization="xavier_uniform"
            )
        
        logger.info("sharding", f"Created {num_experts - existing_experts} new experts")
    
    def dispatch(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main dispatch method.
        
        Args:
            x: Input tensor
            expert_weights: Expert weights
            expert_indices: Expert indices
            
        Returns:
            Combined output
        """
        # Update GDS if enabled
        if self.gds is not None:
            # Record gradients (simplified - in reality would hook into backward)
            unique_experts = torch.unique(expert_indices).tolist()
            for expert_id in unique_experts:
                # Simplified gradient recording
                self.gds.record_gradient(expert_id, torch.tensor(1.0))
        
        # Dispatch through batched dispatcher
        output = self.dispatcher.dispatch_batched(x, expert_weights, expert_indices)
        
        return output
    
    def prefetch_experts(self, expert_ids: List[int]):
        """Prefetch experts asynchronously."""
        self.shard_manager.prefetch_experts(expert_ids)
    
    def get_expert(self, expert_id: int) -> nn.Module:
        """Get expert by ID."""
        return self.shard_manager.load_expert(expert_id)
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save sharding system checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "config": self.config,
            "shard_manager_stats": self.shard_manager.get_stats(),
            "dispatcher_stats": self.dispatcher.get_stats(),
        }
        
        if self.gds is not None:
            checkpoint["gds_importance"] = self.gds.get_importance_scores()
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info("sharding", f"Saved checkpoint to {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "shard_manager": self.shard_manager.get_stats(),
            "dispatcher": self.dispatcher.get_stats(),
            "config": self.config,
        }
        
        if self.gds is not None:
            stats["gds"] = {
                "num_tracked_experts": len(self.gds.expert_importance),
                "average_importance": np.mean(list(self.gds.expert_importance.values())) 
                if self.gds.expert_importance else 0.0,
            }
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        self.shard_manager.cleanup()
        logger.info("sharding", "XORZENXShardingSystem cleaned up")


# ==================== TESTING ====================

__all__ = [
    'ShardMetadata',
    'MemoryStats',
    'LRUCache',
    'MemoryMappedShard',
    'ExpertShardManager',
    'GradientDrivenSharding',
    'BatchedExpertDispatcher',
    'XORZENXShardingSystem',
]