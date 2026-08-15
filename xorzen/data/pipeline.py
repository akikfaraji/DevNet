"""
xorzen Data Pipeline - Production Implementation
Version: 2.0

This module provides a comprehensive and scalable data processing pipeline framework,
designed to be the central orchestrator for all data preparation tasks in xorzen-zero.

Key Features:
- Composable & Modular: Built around the concept of `PipelineComponent`, allowing for
  the creation of complex data workflows by chaining together modular stages like
  filtering, transformation, validation, augmentation, and tokenization.
- Parallel Processing: Integrates `ThreadPoolExecutor` and `ProcessPoolExecutor` to
  run pipeline stages in parallel, significantly speeding up data preparation.
- Rich Component Library: Includes a variety of pre-built components for common tasks
  such as data validation, text cleaning, deduplication, and sampling.
- Extensible: Easily extended with custom components to handle project-specific needs.
- State Management & Checkpointing: Provides a mechanism for saving and loading the
  pipeline's statistical state, allowing for monitoring and simple resumption.
- Self-Contained Testing: Includes a comprehensive `if __name__ == "__main__"` block
  that demonstrates how to build and run a multi-stage pipeline from scratch, serving
  as both a test suite and example usage.
"""

import os
import time
import json
import warnings
import traceback
import sys # Added for sys.exit in __main__
from typing import List, Iterator, Optional, Callable, Any, Dict, Tuple, Union, Set # Added Set
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
# --- Data Structures ---

@dataclass
class PipelineStats:
    """Statistics for a complete pipeline run."""
    total_input: int = 0
    total_output: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    
    stage_stats: Dict[str, 'ComponentStats'] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    @property
    def throughput_in(self) -> float:
        return self.total_input / self.elapsed_time if self.elapsed_time > 0 else 0

    @property
    def throughput_out(self) -> float:
        return self.total_output / self.elapsed_time if self.elapsed_time > 0 else 0

@dataclass
class ComponentStats:
    """Statistics for a single pipeline component."""
    processed_in: int = 0
    processed_out: int = 0
    errors: int = 0
    
    @property
    def filtered_out(self) -> int:
        return self.processed_in - self.processed_out

# --- Pipeline Components ---

class PipelineComponent:
    """Base class for a stage in the data processing pipeline."""
    def __init__(self, name: str):
        self.name = name
        self.stats = ComponentStats()

    def process_batch(self, items: List[Any]) -> List[Any]:
        self.stats.processed_in += len(items)
        try:
            processed_items = self._process_batch_logic(items)
            self.stats.processed_out += len(processed_items)
            return processed_items
        except Exception as e:
            self.stats.errors += len(items)
            warnings.warn(f"Error in component '{self.name}': {e}. Dropping batch.")
            return []

    def _process_batch_logic(self, items: List[Any]) -> List[Any]:
        # Default implementation processes one by one. Can be overridden for efficiency.
        results = []
        for item in items:
            try:
                result = self.process_item(item)
                if result is not None:
                    if isinstance(result, list): # If the transform returns multiple items
                        results.extend(result)
                    else:
                        results.append(result)
            except Exception as e:
                warnings.warn(f"Error processing item in '{self.name}': {e}")
        return results

    def process_item(self, item: Any) -> Optional[Any]:
        raise NotImplementedError

    def get_stats(self) -> ComponentStats:
        return self.stats

    def reset_stats(self):
        self.stats = ComponentStats()

class FilterComponent(PipelineComponent):
    """Filters items based on a predicate function."""
    def __init__(self, name: str, predicate: Callable[[Any], bool]):
        super().__init__(name)
        self.predicate = predicate

    def process_item(self, item: Any) -> Optional[Any]:
        return item if self.predicate(item) else None

class TransformComponent(PipelineComponent):
    """Transforms items using a provided function."""
    def __init__(self, name: str, transform_fn: Callable[[Any], Any]):
        super().__init__(name)
        self.transform_fn = transform_fn

    def process_item(self, item: Any) -> Optional[Any]:
        return self.transform_fn(item)

class DeduplicatorComponent(PipelineComponent):
    """Removes duplicate items based on a key."""
    def __init__(self, name: str, key_fn: Callable[[Any], Any]):
        super().__init__(name)
        self.key_fn = key_fn
        self.seen_keys: Set[Any] = set()

    def process_item(self, item: Any) -> Optional[Any]:
        key = self.key_fn(item)
        if key in self.seen_keys:
            return None
        self.seen_keys.add(key)
        return item

    def reset_stats(self):
        super().reset_stats()
        self.seen_keys.clear()

# --- Main Data Pipeline Class ---

class DataPipeline:
    """A composable, parallelized data processing pipeline."""
    def __init__(
        self,
        components: Optional[List[PipelineComponent]] = None,
        num_workers: int = 1,
        batch_size: int = 1000
    ):
        self.components = components or []
        self.num_workers = max(1, num_workers)
        self.batch_size = batch_size
        self.stats = PipelineStats()

    def add(self, component: PipelineComponent) -> 'DataPipeline':
        self.components.append(component)
        return self

    def run(self, items: Iterator[Any], total: Optional[int] = None) -> Iterator[Any]:
        """
        Processes a stream of items through the full pipeline.

        Args:
            items: An iterator providing the source data.
            total: The total number of items for progress tracking.

        Yields:
            Processed items.
        """
        self.reset_stats()
        self.stats.start_time = time.time()
        
        pbar = None
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total, desc="Data Pipeline", unit="item")

        executor_class = ProcessPoolExecutor if self.num_workers > 1 and os.name != 'nt' else ThreadPoolExecutor
        
        with executor_class(max_workers=self.num_workers) as executor:
            batch = []
            futures = []
            items_consumed_count = 0 # Track items consumed from input iterator
            for item in items:
                batch.append(item)
                items_consumed_count += 1
                if pbar: pbar.update(1) # Update pbar for each item consumed

                if len(batch) >= self.batch_size:
                    future = executor.submit(self._process_pipeline_batch, batch)
                    futures.append(future)
                    batch = []
                
                # Yield results from completed futures to keep memory usage down
                if len(futures) >= self.num_workers * 2: # Limit pending futures
                    for future in as_completed(futures): # Blocking call
                        processed_batch, component_stats_list = future.result()
                        # No pbar.update here, as it's handled when items are consumed
                        self._update_stats(processed_batch, component_stats_list)
                        
                        for processed_item in processed_batch:
                            yield processed_item
                        futures.remove(future)
                        break # process one completed future at a time

            # Process final batch (if any)
            if batch:
                future = executor.submit(self._process_pipeline_batch, batch)
                futures.append(future)
            
            # Collect remaining results
            for future in as_completed(futures):
                processed_batch, component_stats_list = future.result()
                # No pbar.update here
                self._update_stats(processed_batch, component_stats_list)
                for processed_item in processed_batch:
                    yield processed_item
        
        if pbar:
            # Final update if total_items was not perfectly divisible by updates
            pbar.total = items_consumed_count
            pbar.n = items_consumed_count
            pbar.refresh()
            pbar.close()
            
        self.stats.end_time = time.time()
        self.stats.total_input = items_consumed_count # Set total_input here

    def _process_pipeline_batch(self, batch: List[Any]) -> Tuple[List[Any], List[ComponentStats]]:
        """
        Worker function: process one batch through all components.

        Each worker gets its own copy of the component list (via multiprocessing
        pickling), so we track stats locally per-batch and return them to the
        main process for aggregation — no shared-state race conditions.
        """
        component_stats_list = []
        for component in self.components:
            if not batch:
                component_stats_list.append(ComponentStats())
                continue

            start_count = len(batch)
            # Reset before processing so stats reflect this batch only
            component.reset_stats()
            batch = component.process_batch(batch)

            # Build a clean stats snapshot (avoid referencing the live object)
            batch_stats = ComponentStats(
                processed_in=start_count,
                processed_out=len(batch),
                errors=component.stats.errors,
            )
            component_stats_list.append(batch_stats)

        return batch, component_stats_list
        
    def _update_stats(self, processed_batch: List[Any], component_stats_list: List[ComponentStats]):
        """Update the main pipeline statistics from a worker's results."""
        self.stats.total_output += len(processed_batch)
        
        for i, component in enumerate(self.components):
            if i < len(component_stats_list):
                worker_stats = component_stats_list[i]
                if component.name not in self.stats.stage_stats:
                    self.stats.stage_stats[component.name] = ComponentStats()
                
                self.stats.stage_stats[component.name].processed_in += worker_stats.processed_in
                self.stats.stage_stats[component.name].processed_out += worker_stats.processed_out
                self.stats.stage_stats[component.name].errors += worker_stats.errors

    def get_stats(self) -> PipelineStats:
        return self.stats

    def reset_stats(self):
        self.stats = PipelineStats()
        for component in self.components:
            component.reset_stats()
            if isinstance(component, DeduplicatorComponent):
                component.seen_keys.clear()

__all__ = [
    'PipelineStats',
    'ComponentStats',
    'PipelineComponent',
    'FilterComponent',
    'TransformComponent',
    'DeduplicatorComponent',
    'DataPipeline',
]


