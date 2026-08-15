"""
Tokenizer Caching Module

Provides comprehensive caching mechanisms for tokenizers:
- In-memory caching with LRU and custom policies
- Disk-based persistent caching
- Distributed caching support
- Cache statistics and management
- Hash-based cache keys
"""

import os
import pickle
import hashlib
import json
import sqlite3
import time
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import OrderedDict, defaultdict
from enum import Enum
import threading
import warnings

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


# ==================== ENUMS ====================

class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"  # Based on entry size


class CacheBackend(Enum):
    """Cache storage backends."""
    MEMORY = "memory"
    DISK = "disk"
    SQLITE = "sqlite"
    REDIS = "redis"  # Placeholder for future implementation


# ==================== CACHE ENTRY ====================

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        """Calculate miss rate."""
        return 1.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"CacheStats("
            f"hits={self.hits:,}, "
            f"misses={self.misses:,}, "
            f"hit_rate={self.hit_rate:.2%}, "
            f"entries={self.total_entries:,}, "
            f"size={self.total_size_bytes:,} bytes)"
        )


# ==================== BASE CACHE ====================

class BaseCache:
    """Base class for cache implementations."""
    
    def __init__(
        self,
        max_size: int = 10000,
        policy: CachePolicy = CachePolicy.LRU
    ):
        """
        Initialize base cache.
        
        Args:
            max_size: Maximum number of entries
            policy: Eviction policy
        """
        self.max_size = max_size
        self.policy = policy
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        raise NotImplementedError
    
    def clear(self):
        """Clear entire cache."""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        raise NotImplementedError
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


# ==================== MEMORY CACHE ====================

class MemoryCache(BaseCache):
    """In-memory cache with various eviction policies."""
    
    def __init__(
        self,
        max_size: int = 10000,
        policy: CachePolicy = CachePolicy.LRU,
        max_memory_bytes: Optional[int] = None
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            policy: Eviction policy
            max_memory_bytes: Maximum memory usage in bytes
        """
        super().__init__(max_size, policy)
        self.max_memory_bytes = max_memory_bytes
        self.entries: OrderedDict[str, CacheEntry] = OrderedDict() # Always use OrderedDict
        # self.access_order = [] if policy == CachePolicy.LFU else None # This is not needed if using OrderedDict and sorting
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.entries:
                self.stats.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired():
                self.delete(key)
                self.stats.misses += 1
                return None
            
            # Update access metadata
            entry.touch()
            
            # Update order for policy
            if self.policy == CachePolicy.LRU: # Only LRU policy moves to end on access
                self.entries.move_to_end(key)
            
            self.stats.hits += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self.lock:
            current_time = time.time()
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=current_time,
                accessed_at=current_time,
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            # Check if key exists
            if key in self.entries:
                old_entry = self.entries[key]
                self.stats.total_size_bytes -= old_entry.size_bytes
            
            # Check if eviction needed
            while len(self.entries) >= self.max_size:
                self._evict()
            
            # Check memory limit
            if self.max_memory_bytes:
                while (self.stats.total_size_bytes + size_bytes) > self.max_memory_bytes:
                    self._evict()
            
            # Add entry
            self.entries[key] = entry
            self.stats.total_entries = len(self.entries)
            self.stats.total_size_bytes += size_bytes
            
            if self.policy == CachePolicy.LRU: # Only LRU policy moves to end on set if not already existing
                self.entries.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key not in self.entries:
                return False
            
            entry = self.entries.pop(key)
            self.stats.total_entries = len(self.entries)
            self.stats.total_size_bytes -= entry.size_bytes
            return True
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self.entries.clear()
            self.stats = CacheStats()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self.lock:
            return key in self.entries
    
    def _evict(self):
        """Evict entry based on policy."""
        if not self.entries:
            return
        
        if self.policy == CachePolicy.LRU or self.policy == CachePolicy.FIFO:
            # LRU and FIFO both evict the oldest entry. For OrderedDict, this is the first item.
            key, entry = self.entries.popitem(last=False)
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self.entries.keys(), key=lambda k: self.entries[k].access_count)
            entry = self.entries.pop(key)
        elif self.policy == CachePolicy.TTL:
            # Remove oldest or expired
            expired_keys = [k for k, e in self.entries.items() if e.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.entries.keys(), key=lambda k: self.entries[k].created_at)
            entry = self.entries.pop(key)
        elif self.policy == CachePolicy.SIZE_BASED:
            # Remove largest entry
            key = max(self.entries.keys(), key=lambda k: self.entries[k].size_bytes)
            entry = self.entries.pop(key)
        else: # Default to LRU (which is equivalent to FIFO if move_to_end is only on access/set)
            key, entry = self.entries.popitem(last=False)
        
        self.stats.evictions += 1
        self.stats.total_entries = len(self.entries)
        self.stats.total_size_bytes -= entry.size_bytes


# ==================== DISK CACHE ====================

class DiskCache(BaseCache):
    """Persistent disk-based cache."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_size: int = 100000,
        policy: CachePolicy = CachePolicy.LRU
    ):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size: Maximum number of entries
            policy: Eviction policy
        """
        super().__init__(max_size, policy)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file
        self.index_file = self.cache_dir / "cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_index(self):
        """Save cache index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.index:
                self.stats.misses += 1
                return None
            
            entry_meta = self.index[key]
            
            # Check expiration
            if entry_meta.get('ttl'):
                if (time.time() - entry_meta['created_at']) > entry_meta['ttl']:
                    self.delete(key)
                    self.stats.misses += 1
                    return None
            
            # Load from disk
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                self.delete(key)
                self.stats.misses += 1
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access metadata
                entry_meta['accessed_at'] = time.time()
                entry_meta['access_count'] = entry_meta.get('access_count', 0) + 1
                self._save_index()
                
                self.stats.hits += 1
                return value
            except Exception as e:
                warnings.warn(f"Failed to load cache entry: {e}")
                self.delete(key)
                self.stats.misses += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self.lock:
            current_time = time.time()
            
            # Check if eviction needed
            while len(self.index) >= self.max_size:
                self._evict()
            
            # Save to disk
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = cache_path.stat().st_size
                
                # Update index
                self.index[key] = {
                    'created_at': current_time,
                    'accessed_at': current_time,
                    'access_count': 1,
                    'size_bytes': size_bytes,
                    'ttl': ttl,
                    'path': str(cache_path)
                }
                
                self._save_index()
                self.stats.total_entries = len(self.index)
                
            except Exception as e:
                warnings.warn(f"Failed to save cache entry: {e}")
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key not in self.index:
                return False
            
            # Remove file
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            
            # Remove from index
            self.index.pop(key)
            self._save_index()
            self.stats.total_entries = len(self.index)
            return True
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            # Remove all cache files
            for key in list(self.index.keys()):
                cache_path = self._get_cache_path(key)
                if cache_path.exists():
                    cache_path.unlink()
            
            self.index.clear()
            self._save_index()
            self.stats = CacheStats()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self.lock:
            return key in self.index
    
    def _evict(self):
        """Evict entry based on policy."""
        if not self.index:
            return
        
        if self.policy == CachePolicy.LRU:
            key = min(self.index.keys(), key=lambda k: self.index[k]['accessed_at'])
        elif self.policy == CachePolicy.LFU:
            key = min(self.index.keys(), key=lambda k: self.index[k].get('access_count', 0))
        elif self.policy == CachePolicy.FIFO:
            key = min(self.index.keys(), key=lambda k: self.index[k]['created_at'])
        else:
            key = next(iter(self.index))
        
        self.delete(key)
        self.stats.evictions += 1


# ==================== SQLITE CACHE ====================

class SQLiteCache(BaseCache):
    """SQLite-based persistent cache."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        max_size: int = 1000000,
        policy: CachePolicy = CachePolicy.LRU
    ):
        """
        Initialize SQLite cache.
        
        Args:
            db_path: Path to SQLite database
            max_size: Maximum number of entries
            policy: Eviction policy
        """
        super().__init__(max_size, policy)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                accessed_at REAL,
                access_count INTEGER,
                size_bytes INTEGER,
                ttl REAL
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache(accessed_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON cache(access_count)')
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT value, created_at, ttl FROM cache WHERE key = ?',
                (key,)
            )
            
            result = cursor.fetchone()
            
            if result is None:
                conn.close()
                self.stats.misses += 1
                return None
            
            value_bytes, created_at, ttl = result
            
            # Check expiration
            if ttl and (time.time() - created_at) > ttl:
                cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                conn.commit()
                conn.close()
                self.stats.misses += 1
                return None
            
            # Deserialize value
            try:
                value = pickle.loads(value_bytes)
            except:
                conn.close()
                self.stats.misses += 1
                return None
            
            # Update access metadata
            cursor.execute('''
                UPDATE cache 
                SET accessed_at = ?, access_count = access_count + 1 
                WHERE key = ?
            ''', (time.time(), key))
            
            conn.commit()
            conn.close()
            
            self.stats.hits += 1
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """Set value in cache."""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if eviction needed
            cursor.execute('SELECT COUNT(*) FROM cache')
            count = cursor.fetchone()[0]
            
            while count >= self.max_size:
                self._evict(cursor)
                count -= 1
            
            # Serialize value
            value_bytes = pickle.dumps(value)
            size_bytes = len(value_bytes)
            current_time = time.time()
            
            # Insert or replace
            cursor.execute('''
                INSERT OR REPLACE INTO cache 
                (key, value, created_at, accessed_at, access_count, size_bytes, ttl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (key, value_bytes, current_time, current_time, 1, size_bytes, ttl))
            
            conn.commit()
            conn.close()
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            deleted = cursor.rowcount > 0
            
            conn.commit()
            conn.close()
            
            return deleted
    
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM cache')
            
            conn.commit()
            conn.close()
            
            self.stats = CacheStats()
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT 1 FROM cache WHERE key = ? LIMIT 1', (key,))
            exists = cursor.fetchone() is not None
            
            conn.close()
            return exists
    
    def _evict(self, cursor):
        """Evict entry based on policy."""
        if self.policy == CachePolicy.LRU:
            cursor.execute('SELECT key FROM cache ORDER BY accessed_at ASC LIMIT 1')
        elif self.policy == CachePolicy.LFU:
            cursor.execute('SELECT key FROM cache ORDER BY access_count ASC LIMIT 1')
        elif self.policy == CachePolicy.FIFO:
            cursor.execute('SELECT key FROM cache ORDER BY created_at ASC LIMIT 1')
        elif self.policy == CachePolicy.TTL:
            now = time.time()
            cursor.execute(f'SELECT key FROM cache WHERE ttl IS NOT NULL AND (created_at + ttl) < {now} ORDER BY created_at ASC LIMIT 1')
            result = cursor.fetchone()
            if result:
                key = result[0]
                cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
                self.stats.evictions += 1
                return
            # If no expired items, evict oldest by creation time
            cursor.execute('SELECT key FROM cache ORDER BY created_at ASC LIMIT 1')
        else: # Default to LRU if policy not explicitly handled
            cursor.execute('SELECT key FROM cache ORDER BY accessed_at ASC LIMIT 1')
        
        result = cursor.fetchone()
        if result:
            key = result[0]
            cursor.execute('DELETE FROM cache WHERE key = ?', (key,))
            self.stats.evictions += 1


# ==================== TOKENIZATION CACHE ====================

class TokenizationCache:
    """
    Specialized cache for tokenization results.
    """
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        cache_dir: Optional[Union[str, Path]] = None,
        max_size: int = 10000,
        policy: CachePolicy = CachePolicy.LRU,
        hash_algorithm: str = 'md5'
    ):
        """
        Initialize tokenization cache.
        
        Args:
            backend: Cache backend type
            cache_dir: Directory for persistent caches
            max_size: Maximum cache size
            policy: Eviction policy
            hash_algorithm: Hash algorithm for keys
        """
        self.backend = backend
        self.hash_algorithm = hash_algorithm
        
        if backend == CacheBackend.MEMORY:
            self.cache = MemoryCache(max_size=max_size, policy=policy)
        elif backend == CacheBackend.DISK:
            if not cache_dir:
                raise ValueError("cache_dir required for disk backend")
            self.cache = DiskCache(cache_dir=cache_dir, max_size=max_size, policy=policy)
        elif backend == CacheBackend.SQLITE:
            if not cache_dir:
                raise ValueError("cache_dir required for sqlite backend")
            db_path = Path(cache_dir) / "tokenization_cache.db"
            self.cache = SQLiteCache(db_path=db_path, max_size=max_size, policy=policy)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def _compute_key(self, text: str, tokenizer_id: Optional[str] = None) -> str:
        """Compute cache key for text."""
        if tokenizer_id:
            text = f"{tokenizer_id}:{text}"
        
        if self.hash_algorithm == 'md5':
            return hashlib.md5(text.encode()).hexdigest()
        elif self.hash_algorithm == 'sha256':
            return hashlib.sha256(text.encode()).hexdigest()
        else:
            return text
    
    def get_tokenized(
        self,
        text: str,
        tokenizer_id: Optional[str] = None
    ) -> Optional[List[int]]:
        """
        Get tokenized text from cache.
        
        Args:
            text: Input text
            tokenizer_id: Tokenizer identifier
            
        Returns:
            Token IDs or None if not cached
        """
        key = self._compute_key(text, tokenizer_id)
        return self.cache.get(key)
    
    def set_tokenized(
        self,
        text: str,
        token_ids: List[int],
        tokenizer_id: Optional[str] = None,
        ttl: Optional[float] = None
    ):
        """
        Cache tokenized text.
        
        Args:
            text: Input text
            token_ids: Token IDs
            tokenizer_id: Tokenizer identifier
            ttl: Time to live
        """
        key = self._compute_key(text, tokenizer_id)
        self.cache.set(key, token_ids, ttl=ttl)
    
    def tokenize_with_cache(
        self,
        tokenizer: Tokenizer,
        text: str,
        tokenizer_id: Optional[str] = None
    ) -> List[int]:
        """
        Tokenize with caching.
        
        Args:
            tokenizer: Tokenizer instance
            text: Input text
            tokenizer_id: Tokenizer identifier
            
        Returns:
            Token IDs
        """
        # Try cache first
        cached = self.get_tokenized(text, tokenizer_id)
        if cached is not None:
            return cached
        
        # Tokenize
        encoding = tokenizer.encode(text)
        token_ids = encoding.ids
        
        # Cache result
        self.set_tokenized(text, token_ids, tokenizer_id)
        
        return token_ids
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()


# ==================== CACHED TOKENIZER WRAPPER ====================

class CachedTokenizer:
    """
    Tokenizer wrapper with automatic caching.
    """
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        cache: Optional[TokenizationCache] = None,
        tokenizer_id: Optional[str] = None
    ):
        """
        Initialize cached tokenizer.
        
        Args:
            tokenizer: Base tokenizer
            cache: Tokenization cache
            tokenizer_id: Tokenizer identifier
        """
        if not TOKENIZERS_AVAILABLE:
            raise ImportError("tokenizers library is required")
        
        self.tokenizer = tokenizer
        self.cache = cache or TokenizationCache()
        self.tokenizer_id = tokenizer_id or str(id(tokenizer))
    
    def encode(self, text: str) -> List[int]:
        """Encode text with caching."""
        return self.cache.tokenize_with_cache(self.tokenizer, text, self.tokenizer_id)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode batch with caching."""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache
        for i, text in enumerate(texts):
            cached = self.cache.get_tokenized(text, self.tokenizer_id)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Tokenize uncached
        if uncached_texts:
            encodings = self.tokenizer.encode_batch(uncached_texts)
            
            for i, encoding in zip(uncached_indices, encodings):
                token_ids = encoding.ids
                results[i] = token_ids
                # Cache result
                self.cache.set_tokenized(texts[i], token_ids, self.tokenizer_id)
        
        return results
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode tokens (no caching)."""
        return self.tokenizer.decode(token_ids)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Clear cache."""
        self.cache.clear()

__all__ = [
    'CachePolicy',
    'CacheBackend',
    'CacheEntry',
    'CacheStats',
    'TokenizationCache',
    'CachedTokenizer',
]
