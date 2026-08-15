"""
This module implements a production-grade, highly configurable logging system
for the `xorzen` deep learning framework. It provides functionalities for:
- **Asynchronous Logging**: High-performance log message processing without blocking the main thread.
- **Structured Logging**: Log entries are formatted as structured data (JSON) for easy parsing and analysis.
- **Metrics Aggregation**: Capabilities for collecting, aggregating, and reporting numerical metrics over time.
- **Real-time Performance Monitoring**: Integration with performance metrics collection (e.g., CPU, GPU, memory).
- **Distributed Training Support**: Special handlers for coordinating logging and metrics across multiple processes in distributed environments.
- **File and Console Output**: Configurable handlers for directing log output to console and/or files.
- **Checkpointing**: Ability to save and load logger state for continuity.

The system is designed to provide rich, actionable insights into model training and operation,
facilitating debugging, performance analysis, and experiment tracking.
"""

import logging
import sys
import os
import json
import time
import csv
import threading
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
import traceback
import atexit

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available, some features disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available, some features disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available, some performance monitoring features disabled")


class LogLevel(Enum):
    """
    Defines standard logging levels with associated numeric values,
    consistent with Python's `logging` module. These levels control
    the verbosity and severity of log messages.
    """    
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEntry:
    """
    Represents a single, structured log entry. Each entry encapsulates
    metadata such as timestamp, logging level, source module, and the
    actual message, along with optional arbitrary data, exception details,
    and stack trace for detailed debugging.
    """    
    timestamp: str
    level: str
    module: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `LogEntry` instance into a dictionary representation.
        Optional fields (`exception`, `stack_trace`) are omitted if their
        values are `None`, ensuring a cleaner and more compact dictionary.
        
        Returns:
            A dictionary representing the log entry's data.
        """
        result = asdict(self)
        if self.exception is None:
            result.pop('exception')
        if self.stack_trace is None:
            result.pop('stack_trace')
        return result
    
    def to_json(self) -> str:
        """
        Serializes the `LogEntry` instance into a JSON formatted string.
        This facilitates easy storage and parsing of log data by external
        systems or log analysis tools.
        
        Returns:
            A JSON string representation of the log entry.
        """        
        return json.dumps(self.to_dict(), default=str)


@dataclass
class MetricEntry:
    """
    Represents a single metric entry, primarily used for time-series data
    collection. It captures the metric's name, its numerical value, the
    associated training step and epoch, and optional contextual tags.
    """    
    timestamp: str
    metric_name: str
    value: float
    step: int
    epoch: int
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the `MetricEntry` instance into a dictionary representation.
        
        Returns:
            A dictionary representing the metric entry's data.
        """        
        return asdict(self)


class AsyncLogHandler:
    """
    Manages asynchronous processing of log and metric entries to ensure
    high-performance logging without blocking the main application thread.
    Entries are buffered and processed in batches by a dedicated worker thread,
    which then dispatches them to registered handlers.
    """    
    def __init__(self, max_queue_size: int = 10000, batch_size: int = 100):
        """
        Initializes the asynchronous log handler.
        
        Args:
            max_queue_size (`int`, *optional*): The maximum number of log/metric
                                                 entries that can be held in the
                                                 internal queue. If the queue
                                                 becomes full, subsequent `log`
                                                 calls may block or fall back
                                                 to synchronous processing. Defaults to 10000.
            batch_size (`int`, *optional*): The number of entries to process
                                            together in a single batch. Larger
                                            batch sizes can improve throughput
                                            but may introduce slightly higher latency
                                            for individual entries. Defaults to 100.
        """        
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.batch_size = batch_size
        self.handlers: List[Callable] = []
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.batch_buffer: List[Union[LogEntry, MetricEntry]] = []
        
    def start(self):
        """
        Initiates the asynchronous processing thread. This thread continuously
        monitors the internal queue for new log/metric entries and dispatches
        them to registered handlers in batches. This method is idempotent;
        calling it multiple times has no additional effect if the thread is already running.
        """        
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
            name="AsyncLogHandler"
        )
        self.worker_thread.start()
        
        # Register cleanup
        atexit.register(self.stop)
    
    def stop(self):
        """
        Signals the asynchronous processing thread to terminate.
        This method waits for the worker thread to finish processing
        any remaining entries in its queue before fully stopping.
        """        
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def add_handler(self, handler: Callable):
        """
        Registers a callable object as a handler for processing log and metric batches.
        Each registered handler will be invoked with a list of `LogEntry` or `MetricEntry`
        objects whenever a batch is ready for processing.
        
        Args:
            handler (`Callable`): A callable that accepts a list of `Union[LogEntry, MetricEntry]`
                                  as its sole argument.
        """        
        self.handlers.append(handler)
    
    def log(self, entry: Union[LogEntry, MetricEntry]):
        """
        Adds a new `LogEntry` or `MetricEntry` to the asynchronous processing queue.
        If the queue is full, the entry may be processed synchronously as a fallback
        to prevent data loss, though this can introduce a brief blocking operation.
        
        Args:
            entry (`Union[LogEntry, MetricEntry]`): The log or metric entry to add.
        """        
        try:
            self.queue.put(entry, block=False)
        except queue.Full:
            # Fallback to synchronous logging if queue is full
            self._process_entry(entry)
    
    def _process_loop(self):
        """
        The main loop of the asynchronous processing thread. This method
        continuously retrieves entries from the queue, buffers them, and
        dispatches batches to registered handlers when the `batch_size`
        is reached or the queue is empty after a timeout.
        """        
        while self.running:
            try:
                # Wait for entry with timeout
                entry = self.queue.get(timeout=0.1)
                self.batch_buffer.append(entry)
                
                # Process batch if full
                if len(self.batch_buffer) >= self.batch_size:
                    self._process_batch()
                    
            except queue.Empty:
                # Process any remaining entries in buffer
                if self.batch_buffer:
                    self._process_batch()
            except Exception as e:
                # Log error but don't crash
                print(f"Error in async log handler: {e}")
    
    def _process_batch(self):
        """
        Processes a buffered batch of log and metric entries.
        Each registered handler is invoked with the current batch,
        and the internal buffer is then cleared. Errors in individual
        handlers are caught to prevent the entire logging process from failing.
        """        
        if not self.batch_buffer:
            return
        
        for handler in self.handlers:
            try:
                handler(self.batch_buffer)
            except Exception as e:
                print(f"Error in log handler: {e}")
        
        self.batch_buffer = []
    
    def _process_entry(self, entry: Union[LogEntry, MetricEntry]):
        """
        Processes a single log or metric entry synchronously. This method is
        typically used as a fallback when the asynchronous queue is full,
        or when asynchronous logging is disabled.
        
        Args:
            entry (`Union[LogEntry, MetricEntry]`): The log or metric entry to process.
        """        
        for handler in self.handlers:
            try:
                handler([entry])
            except Exception as e:
                print(f"Error in log handler: {e}")


class XORZENXLogger:
    """
    The central logging class for the `xorzen` framework, providing a highly
    configurable interface for emitting structured logs and metrics.
    It supports multiple output targets (console, file), asynchronous processing,
    and integrates with performance monitoring. Designed for both single-process
    and distributed training environments.
    """
    
    def __init__(
        self,
        name: str = "xorzen",
        log_dir: Union[str, Path] = "logs",
        level: LogLevel = LogLevel.INFO,
        enable_async: bool = True,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_metrics: bool = True,
        distributed_rank: int = 0
    ):
        """
        Initializes the `XORZENXLogger` instance, setting up its name, output
        directory, logging level, and various handlers for console, file,
        and metrics output. Asynchronous processing is enabled by default
        for improved performance.
        
        Args:
            name (`str`, *optional*): The logical name of the logger.
                                      Defaults to "xorzen".
            log_dir (`Union[str, Path]`, *optional*): The base directory for
                                                      log files. If a full
                                                      file path is provided,
                                                      it will be used directly.
                                                      Defaults to "logs".
            level (`LogLevel`, *optional*): The minimum logging level for
                                            messages to be processed.
                                            Defaults to `LogLevel.INFO`.
            enable_async (`bool`, *optional*): If `True`, logging operations
                                               are offloaded to a separate
                                               thread for non-blocking execution.
                                               Defaults to `True`.
            enable_console (`bool`, *optional*): If `True`, log messages
                                                 will be printed to the console.
                                                 Defaults to `True`.
            enable_file (`bool`, *optional*): If `True`, log messages
                                              will be written to a file.
                                              Defaults to `True`.
            enable_metrics (`bool`, *optional*): If `True`, enables the
                                                collection and aggregation
                                                of performance metrics.
                                                Defaults to `True`.
            distributed_rank (`int`, *optional*): The rank of the current
                                                  process in a distributed
                                                  training setup. Used to
                                                  control console output
                                                  (typically only rank 0 prints).
                                                  Defaults to 0.
        """        
        self.name = name
        self.log_dir = Path(log_dir)
        self.level = level
        self.distributed_rank = distributed_rank
        
        # Create log directory if log_dir is a directory
        if self.log_dir.is_dir() or '.' not in self.log_dir.name:
             self.log_dir.mkdir(parents=True, exist_ok=True)
        else: # it's a file
             self.log_dir.parent.mkdir(parents=True, exist_ok=True)

        
        # Setup async handler if enabled
        self.async_handler = AsyncLogHandler() if enable_async else None
        
        # Initialize handlers
        self.handlers: Dict[str, Any] = {}
        self.metric_writers: Dict[str, Any] = {}
        
        
        # Setup outputs
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handler()
        
        if enable_metrics:
            self._setup_metrics_handler()
        
        # Start async handler if enabled
        if enable_async and self.async_handler:
            self.async_handler.start()
        
        # Statistics
        self.stats = {
            "log_count": 0,
            "metric_count": 0,
            "error_count": 0,
            "last_error": None
        }
        
        # Performance monitoring
        self.performance_stats = {
            "avg_log_time": 0.0,
            "total_log_time": 0.0,
            "max_log_time": 0.0
        }
        
        # Register cleanup
        atexit.register(self.cleanup)
    
    def _setup_console_handler(self):
        """
        Configures and registers a handler for directing log messages to the console.
        In distributed environments, only the process with `distributed_rank == 0`
        will output logs to the console, and log levels are color-coded for readability.
        """        
        class ConsoleHandler:
            def __init__(self, rank: int):
                self.rank = rank
            
            def __call__(self, entries: List[Union[LogEntry, MetricEntry]]):
                for entry in entries:
                    if isinstance(entry, LogEntry):
                        if self.rank == 0:  # Only rank 0 prints to console
                            level_color = {
                                "DEBUG": "\033[90m",      # Gray
                                "INFO": "\033[94m",       # Blue
                                "WARNING": "\033[93m",    # Yellow
                                "ERROR": "\033[91m",      # Red
                                "CRITICAL": "\033[95m"    # Magenta
                            }
                            reset = "\033[0m"
                            
                            color = level_color.get(entry.level, "\033[0m")
                            msg = f"{color}[{entry.timestamp}] [{entry.level}] [{entry.module}] {entry.message}{reset}"
                            
                            if entry.exception:
                                msg += f"\n{color}Exception: {entry.exception}{reset}"
                                if entry.stack_trace and entry.level == LogLevel.CRITICAL.name:
                                    msg += f"\n{color}Stack Trace:\n{entry.stack_trace}{reset}"
                            
                            print(msg)
        
        handler = ConsoleHandler(self.distributed_rank)
        self.handlers["console"] = handler
        
        if self.async_handler:
            self.async_handler.add_handler(handler)
    
    def _setup_file_handler(self):
        """
        Configures and registers a handler for writing log and metric entries to files.
        Log messages are written to a `.log` file in JSON format, while metrics
        are appended to a `.csv` file. File names are timestamped for easy organization.
        """        
        if self.log_dir.is_dir():
            log_file = self.log_dir / f"xorzen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            metrics_file = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else: # it's a file path
            log_file = self.log_dir
            metrics_file = self.log_dir.with_suffix('.csv')

        class FileHandler:
            def __init__(self, log_path: Path, metrics_path: Path):
                self.log_path = log_path
                self.metrics_path = metrics_path
                self.log_file = None
                self.metrics_file = None
                self.metrics_writer = None
                self._initialize_files()

            def _initialize_files(self):
                # This logic is now inside a method to be called after the handler is created
                self.log_file = open(self.log_path, 'a', encoding='utf-8')
                
                # Ensure metrics file has header
                if not self.metrics_path.exists() or os.stat(self.metrics_path).st_size == 0:
                    with open(self.metrics_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'timestamp', 'metric_name', 'value', 'step', 'epoch', 'tags'
                        ])
                        writer.writeheader()
            
            def __call__(self, entries: List[Union[LogEntry, MetricEntry]]):
                log_entries_to_write = []
                metric_entries_to_write = []

                for entry in entries:
                    if isinstance(entry, LogEntry):
                        log_entries_to_write.append(entry.to_json())
                    elif isinstance(entry, MetricEntry):
                        metric_entries_to_write.append(entry.to_dict())

                if log_entries_to_write:
                    with open(self.log_path, 'a', encoding='utf-8') as f:
                        f.write('\n'.join(log_entries_to_write) + '\n')

                if metric_entries_to_write:
                    # Open in append mode, create writer if needed
                    if self.metrics_file is None or self.metrics_file.closed:
                        self.metrics_file = open(self.metrics_path, 'a', newline='', encoding='utf-8')
                        self.metrics_writer = csv.DictWriter(
                            self.metrics_file,
                            fieldnames=['timestamp', 'metric_name', 'value', 'step', 'epoch', 'tags']
                        )
                    
                    self.metrics_writer.writerows(metric_entries_to_write)
                    self.metrics_file.flush() # Ensure it's written
            
            def close(self):
                """Close file handles."""
                if self.log_file and not self.log_file.closed:
                    self.log_file.close()
                if self.metrics_file and not self.metrics_file.closed:
                    self.metrics_file.close()
        
        # Create and add the handler instance first
        handler = FileHandler(log_file, metrics_file)
        self.handlers["file"] = handler
        
        if self.async_handler:
            self.async_handler.add_handler(handler)
    
    def _setup_metrics_handler(self):
        """
        Configures and registers an internal handler for aggregating incoming metrics.
        This handler collects metric entries, stores recent values, and computes
        aggregated statistics (mean, std, min, max, count) for real-time monitoring.
        """        
        class MetricsHandler:
            def __init__(self):
                self.metrics: Dict[str, List[MetricEntry]] = {}
                self.aggregated: Dict[str, Dict[str, float]] = {}
            
            def __call__(self, entries: List[Union[LogEntry, MetricEntry]]):
                for entry in entries:
                    if isinstance(entry, MetricEntry):
                        # Store metric
                        if entry.metric_name not in self.metrics:
                            self.metrics[entry.metric_name] = []
                        self.metrics[entry.metric_name].append(entry)
                        
                        # Aggregate last 100 values
                        recent = self.metrics[entry.metric_name][-100:]
                        values = [m.value for m in recent]
                        
                        if values:
                            self.aggregated[entry.metric_name] = {
                                'mean': np.mean(values) if NUMPY_AVAILABLE else sum(values)/len(values),
                                'std': np.std(values) if NUMPY_AVAILABLE else 0.0,
                                'min': min(values),
                                'max': max(values),
                                'count': len(values)
                            }
        
        handler = MetricsHandler()
        self.metric_writers["aggregator"] = handler
        
        if self.async_handler:
            self.async_handler.add_handler(handler)
    
    def log(
        self,
        level: LogLevel,
        module: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """
        Emits a structured log message. Messages are processed asynchronously
        if `enable_async` is active during initialization, otherwise synchronously.
        The message is only processed if its `level` is greater than or equal
        to the logger's configured minimum level.
        
        Args:
            level (`LogLevel`): The severity level of the log message.
            module (`str`): The name of the module or component originating the log.
            message (`str`): The main textual content of the log entry.
            data (`Dict[str, Any]`, *optional*): An optional dictionary for
                                                including additional structured
                                                data relevant to the log message.
            exception (`Exception`, *optional*): An optional exception object
                                                 to be included in the log entry,
                                                 including its string representation
                                                 and stack trace (for CRITICAL errors).
        """        # Skip if below log level
        if level.value < self.level.value:
            return
        
        start_time = time.time()
        
        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.name,
            module=module,
            message=message,
            data=data or {},
            exception=str(exception) if exception else None,
            stack_trace="".join(traceback.format_exception(type(exception), exception, exception.__traceback__)) if exception else None
        )
        
        # Update statistics
        self.stats["log_count"] += 1
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.stats["error_count"] += 1
            self.stats["last_error"] = entry
        
        # Send to async handler or process directly
        if self.async_handler:
            self.async_handler.log(entry)
        else:
            # Process synchronously
            for handler in self.handlers.values():
                handler([entry])
        
        # Update performance stats
        log_time = time.time() - start_time
        self.performance_stats["total_log_time"] += log_time
        self.performance_stats["avg_log_time"] = (
            self.performance_stats["total_log_time"] / self.stats["log_count"]
        )
        self.performance_stats["max_log_time"] = max(
            self.performance_stats["max_log_time"], log_time
        )
    
    def debug(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Emits a log message with `LogLevel.DEBUG` severity.
        These messages are typically used for fine-grained informational events
        that are most useful when debugging an application.
        
        Args:
            module (`str`): The name of the module or component originating the log.
            message (`str`): The main textual content of the debug entry.
            data (`Dict[str, Any]`, *optional*): Additional structured data.
        """        
        self.log(LogLevel.DEBUG, module, message, data)
    
    def info(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Emits a log message with `LogLevel.INFO` severity.
        These messages provide general confirmation that things are working as expected.
        
        Args:
            module (`str`): The name of the module or component originating the log.
            message (`str`): The main textual content of the info entry.
            data (`Dict[str, Any]`, *optional*): Additional structured data.
        """        
        self.log(LogLevel.INFO, module, message, data)
    
    def warning(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Emits a log message with `LogLevel.WARNING` severity.
        These messages indicate that something unexpected happened, or
        indicative of a problem, but the software is still working as expected.
        
        Args:
            module (`str`): The name of the module or component originating the log.
            message (`str`): The main textual content of the warning entry.
            data (`Dict[str, Any]`, *optional*): Additional structured data.
        """        
        self.log(LogLevel.WARNING, module, message, data)
    
    def error(
        self,
        module: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """
        Emits a log message with `LogLevel.ERROR` severity.
        These messages indicate a more serious problem that prevented some
        functionality from completing.
        
        Args:
            module (`str`): The name of the module or component originating the log.
            message (`str`): The main textual content of the error entry.
            data (`Dict[str, Any]`, *optional*): Additional structured data.
            exception (`Exception`, *optional*): An optional exception object
                                                 to be included in the log entry.
        """        
        self.log(LogLevel.ERROR, module, message, data, exception)
    
    def critical(
        self,
        module: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """
        Emits a log message with `LogLevel.CRITICAL` severity.
        These messages indicate a severe error that has caused the application
        to stop or indicates an unrecoverable state. The full stack trace
        is typically included with critical logs.
        
        Args:
            module (`str`): The name of the module or component originating the log.
            message (`str`): The main textual content of the critical entry.
            data (`Dict[str, Any]`, *optional*): Additional structured data.
            exception (`Exception`, *optional*): An optional exception object
                                                 to be included in the log entry,
                                                 always including its stack trace.
        """        
        self.log(LogLevel.CRITICAL, module, message, data, exception)
    
    def metric(
        self,
        name: str,
        value: float,
        step: int,
        epoch: int,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Emits a structured metric entry for tracking time-series data
        related to training progress, model performance, or resource utilization.
        Metrics are typically processed asynchronously.
        
        Args:
            name (`str`): The unique identifier or name of the metric.
            value (`float`): The numerical value of the metric.
            step (`int`): The current global training step at which the metric was recorded.
            epoch (`int`): The current training epoch at which the metric was recorded.
            tags (`Dict[str, str]`, *optional*): An optional dictionary of
                                                key-value pairs for categorizing
                                                or filtering the metric (e.g., {"subset": "validation"}).
        """        
        entry = MetricEntry(
            timestamp=datetime.now().isoformat(),
            metric_name=name,
            value=value,
            step=step,
            epoch=epoch,
            tags=tags or {}
        )
        
        # Update statistics
        self.stats["metric_count"] += 1
        
        # Send to async handler or process directly
        if self.async_handler:
            self.async_handler.log(entry)
        else:
            # Process synchronously
            for handler in self.handlers.values():
                handler([entry])
    
    def log_model_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: int,
        prefix: str = "train"
    ):
        """
        Logs a collection of model-related metrics. This is a convenience
        method for logging multiple metrics from a dictionary, automatically
        applying a common `step`, `epoch`, and an optional `prefix` to each.
        
        Args:
            metrics (`Dict[str, float]`): A dictionary where keys are metric names
                                         (e.g., "loss", "accuracy") and values are
                                         their corresponding numerical values.
            step (`int`): The current global training step.
            epoch (`int`): The current training epoch.
            prefix (`str`, *optional*): A string prefix to prepend to each metric
                                        name (e.g., "train" results in "train/loss").
                                        Defaults to "train".
        """        
        for name, value in metrics.items():
            self.metric(f"{prefix}/{name}", value, step, epoch)
    
    def log_router_stats(
        self,
        stats: Dict[str, Any],
        step: int,
        epoch: int
    ):
        """
        Logs various statistics related to the model's adaptive router.
        This provides insights into routing decisions, activation patterns,
        and load distribution within the Mixture-of-Experts (MoE) system.
        
        Args:
            stats (`Dict[str, Any]`): A dictionary containing router statistics.
                                      Values can be scalar (int/float) or lists
                                      of numerical values.
            step (`int`): The current global training step.
            epoch (`int`): The current training epoch.
        """        
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.metric(f"router/{key}", value, step, epoch)
            elif isinstance(value, list):
                for i, val in enumerate(value):
                    if isinstance(val, (int, float)):
                        self.metric(f"router/{key}_{i}", val, step, epoch)
    
    def log_expert_stats(
        self,
        expert_id: int,
        activation_count: int,
        load: float,
        step: int,
        epoch: int
    ):
        """
        Logs detailed statistics for a specific expert within a Mixture-of-Experts (MoE) system.
        This includes the total number of times the expert has been activated and its
        current computational load, offering insights into individual expert utilization.
        
        Args:
            expert_id (`int`): The unique identifier of the expert.
            activation_count (`int`): The cumulative number of times this expert
                                      has been activated.
            load (`float`): The current estimated computational load or utilization
                            of the expert.
            step (`int`): The current global training step.
            epoch (`int`): The current training epoch.
        """        
        self.metric(f"expert/{expert_id}/activation_count", activation_count, step, epoch)
        self.metric(f"expert/{expert_id}/load", load, step, epoch)
    
    def log_memory_stats(
        self,
        stats: Dict[str, float],
        step: int,
        epoch: int
    ):
        """
        Logs various memory-related statistics, such as allocated memory,
        reserved memory, or memory utilization, typically collected from
        the underlying hardware (e.g., GPU memory).
        
        Args:
            stats (`Dict[str, float]`): A dictionary where keys are memory
                                        statistic names (e.g., "allocated_mb")
                                        and values are their numerical measurements.
            step (`int`): The current global training step.
            epoch (`int`): The current training epoch.
        """        
        for key, value in stats.items():
            self.metric(f"memory/{key}", value, step, epoch)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieves a comprehensive summary of the logger's operational statistics,
        including counts of logs and metrics emitted, error occurrences,
        performance benchmarks for logging operations, and aggregated metric values.
        
        Returns:
            A dictionary containing various statistics and performance metrics
            related to the logger's activity.
        """        
        stats = self.stats.copy()
        stats.update(self.performance_stats)
        
        # Get aggregated metrics if available
        if "aggregator" in self.metric_writers:
            stats["aggregated_metrics"] = self.metric_writers["aggregator"].aggregated
        
        return stats
    
    def create_checkpoint(self, path: Union[str, Path]):
        """
        Creates a checkpoint of the logger's internal state, including
        statistics and performance metrics, and saves it to a specified
        file path in JSON format. This allows for persistent tracking
        of logging activity across application runs.
        
        Args:
            path (`Union[str, Path]`): The file path where the logger checkpoint
                                        should be saved.
        """        
        stats_for_dump = self.stats.copy()
        if stats_for_dump["last_error"] is not None:
            stats_for_dump["last_error"] = stats_for_dump["last_error"].to_dict()

        checkpoint = {
            "stats": stats_for_dump,
            "performance_stats": self.performance_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self, path: Union[str, Path]):
        """
        Loads the logger's internal state (statistics and performance metrics)
        from a previously saved checkpoint file. This enables resuming
        logging activity and tracking from a consistent historical point.
        
        Args:
            path (`Union[str, Path]`): The file path to the logger checkpoint
                                        to be loaded.
        """        
        try:
            with open(path, 'r') as f:
                checkpoint = json.load(f)
            
            self.stats.update(checkpoint.get("stats", {}))
            self.performance_stats.update(checkpoint.get("performance_stats", {}))
            
            self.info(
                "logger",
                f"Loaded checkpoint from {path}",
                {"checkpoint_timestamp": checkpoint.get("timestamp")}
            )
        except Exception as e:
            self.error("logger", f"Failed to load checkpoint: {e}", exception=e)
    
    def cleanup(self):
        """
        Performs necessary cleanup operations, such as stopping the
        asynchronous logging thread and closing any open file handles,
        to ensure all resources are properly released.
        """        
        if self.async_handler:
            self.async_handler.stop()
        
        # Close file handlers
        for handler in self.handlers.values():
            if hasattr(handler, 'close'):
                handler.close()
    
    def __enter__(self):
        """
        Enables the `XORZENXLogger` to be used as a context manager.
        Upon entering the context, it simply returns the logger instance itself.
        """        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the `XORZENXLogger` context manager. This method ensures that
        all allocated resources are properly cleaned up via `cleanup()`.
        If an exception occurred within the context, it is logged before
        being propagated.
        
        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      `None` if no exception occurred.
            exc_val: The exception instance that occurred. `None` if no exception occurred.
            exc_tb: The traceback object. `None` if no exception occurred.
        
        Returns:
            `False` if an exception occurred and should be propagated, `True` otherwise.
        """        
        self.cleanup()
        if exc_type is not None:
            self.error(
                "logger",
                f"Context manager exited with exception: {exc_type.__name__}",
                exception=exc_val
            )


# ==================== DISTRIBUTED LOGGING ====================

class DistributedLogger:
    """
    Extends `XORZENXLogger` functionality to specifically handle logging and
    metric aggregation in distributed training environments. It coordinates
    metric collection across multiple processes, ensuring that aggregated
    statistics (e.g., mean loss) are correctly reported, typically by rank 0.
    """
    
    def __init__(
        self,
        base_logger: XORZENXLogger,
        world_size: int,
        rank: int,
        enable_all_reduce: bool = True
    ):
        """
        Initializes the `DistributedLogger` instance.
        
        Args:
            base_logger (`XORZENXLogger`): The underlying `XORZENXLogger` instance
                                         used for local logging and reporting
                                         aggregated metrics.
            world_size (`int`): The total number of processes participating
                                in the distributed training.
            rank (`int`): The unique identifier (rank) of the current process
                          within the distributed group.
            enable_all_reduce (`bool`, *optional*): If `True`, enables automatic
                                                    aggregation (all-reduce) of
                                                    metrics across all processes
                                                    before logging. Defaults to `True`.
        """        
        self.base_logger = base_logger
        self.world_size = world_size
        self.rank = rank
        self.enable_all_reduce = enable_all_reduce
        
        # Buffer for metrics to aggregate
        self.metric_buffer: Dict[str, List[float]] = {}
    
    def log_metric_all_reduce(
        self,
        name: str,
        value: float,
        step: int,
        epoch: int,
        reduction: str = "mean"
    ):
        """
        Logs a metric that requires aggregation across all distributed processes
        before being reported. The local `value` from each process is buffered,
        and once all processes have reported for a given step/epoch, the values
        are combined using the specified `reduction` method (e.g., mean, sum).
        Only the rank 0 process then logs the aggregated result.
        
        Args:
            name (`str`): The name of the metric to log.
            value (`float`): The local value of the metric from the current process.
            step (`int`): The current global training step.
            epoch (`int`): The current training epoch.
            reduction (`str`, *optional*): The method to use for aggregating
                                          metric values across processes.
                                          Supported: "mean", "sum", "max", "min".
                                          Defaults to "mean".
        """        
        if not self.enable_all_reduce or self.world_size == 1:
            # Single process or all-reduce disabled
            self.base_logger.metric(name, value, step, epoch)
            return
        
        # Buffer metric for aggregation
        if name not in self.metric_buffer:
            self.metric_buffer[name] = []
        
        self.metric_buffer[name].append((value, step, epoch))
        
        # Aggregate when buffer is full
        if len(self.metric_buffer[name]) >= self.world_size:
            values = [v for v, _, _ in self.metric_buffer[name]]
            steps = [s for _, s, _ in self.metric_buffer[name]]
            epochs = [e for _, _, e in self.metric_buffer[name]]
            
            # All processes should have same step and epoch
            step_consistent = all(s == steps[0] for s in steps)
            epoch_consistent = all(e == epochs[0] for e in epochs)
            
            if step_consistent and epoch_consistent:
                # Apply reduction
                if reduction == "mean":
                    reduced_value = sum(values) / len(values)
                elif reduction == "sum":
                    reduced_value = sum(values)
                elif reduction == "max":
                    reduced_value = max(values)
                elif reduction == "min":
                    reduced_value = min(values)
                else:
                    reduced_value = sum(values) / len(values)  # Default to mean
                
                # Only rank 0 logs the aggregated metric
                if self.rank == 0:
                    self.base_logger.metric(
                        f"distributed/{name}",
                        reduced_value,
                        steps[0],
                        epochs[0]
                    )
                
                # Clear buffer
                del self.metric_buffer[name]
    
    def log_router_stats_distributed(
        self,
        stats: Dict[str, Any],
        step: int,
        epoch: int
    ):
        """
        Logs router-specific statistics, aggregating numerical values across
        all distributed processes using a "mean" reduction before reporting.
        This provides a consolidated view of router behavior in a distributed setup.
        
        Args:
            stats (`Dict[str, Any]`): A dictionary containing router statistics.
                                      Only scalar (int/float) values will be aggregated.
            step (`int`): The current global training step.
            epoch (`int`): The current training epoch.
        """        
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.log_metric_all_reduce(
                    f"router/{key}",
                    value,
                    step,
                    epoch,
                    reduction="mean"
                )
    
    def flush(self):
        """
        Forces the processing and logging of any buffered metrics that
        are awaiting aggregation. This ensures that all collected metrics,
        even those that haven't reached the `world_size` threshold for
        `all-reduce`, are reported.
        """        
        for name, buffer in list(self.metric_buffer.items()):
            if buffer:
                # Log whatever we have
                values = [v for v, _, _ in buffer]
                steps = [s for _, s, _ in buffer]
                epochs = [e for _, _, e in buffer]
                
                # Take most common step and epoch
                from collections import Counter
                step = Counter(steps).most_common(1)[0][0]
                epoch = Counter(epochs).most_common(1)[0][0]
                
                # Log average
                avg_value = sum(values) / len(values)
                
                if self.rank == 0:
                    self.base_logger.metric(
                        f"distributed/flushed/{name}",
                        avg_value,
                        step,
                        epoch
                    )
                
                # Clear buffer
                del self.metric_buffer[name]


# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """
    Provides real-time monitoring and reporting of various performance metrics
    during model training and evaluation. This includes throughput (tokens/sec),
    memory utilization (CPU/GPU), CPU/GPU usage percentages, and optional
    PyTorch profiling.
    """
    
    def __init__(
        self,
        logger: XORZENXLogger,
        update_interval: int = 100,  # steps
        enable_profiling: bool = False
    ):
        """
        Initializes the `PerformanceMonitor`.
        
        Args:
            logger (`XORZENXLogger`): An instance of the `XORZENXLogger` to which
                                   performance metrics will be reported.
            update_interval (`int`, *optional*): The frequency, in training steps,
                                                 at which performance metrics
                                                 are updated and logged. Defaults to 100.
            enable_profiling (`bool`, *optional*): If `True`, enables PyTorch's
                                                   profiler to capture detailed
                                                   CPU and CUDA operation traces.
                                                   Defaults to `False`.
        """        
        self.logger = logger
        self.update_interval = update_interval
        self.enable_profiling = enable_profiling
        
        # Performance metrics
        self.metrics = {
            "throughput_tokens_per_sec": 0.0,
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "cpu_usage_percent": 0.0,
            "gpu_usage_percent": 0.0,
            "io_wait_time_sec": 0.0
        }
        
        # Counters
        self.step_count = 0
        self.token_count = 0
        self.start_time = time.time()
        
        # Profiling
        self.profiler = None
        if enable_profiling and TORCH_AVAILABLE:
            try:
                self.profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA
                    ] if torch.cuda.is_available() else [
                        torch.profiler.ProfilerActivity.CPU
                    ],
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=3,
                        repeat=1
                    ),
                    on_trace_ready=self._on_trace_ready
                )
            except Exception as e:
                self.logger.warning(
                    "performance_monitor",
                    f"Failed to initialize profiler: {e}"
                )
    
    def start_step(self):
        """
        Marks the beginning of a new training step for performance monitoring.
        This method records the start time of the step and, if profiling is
        enabled, initiates the PyTorch profiler for the current step.
        """        
        self.step_start_time = time.time()
        
        if self.profiler:
            self.profiler.start()
    
    def end_step(self, tokens_processed: int):
        """
        Marks the end of a training step, recording its duration and updating
        internal counters for total steps and tokens processed. Performance
        metrics are periodically updated and logged at intervals defined by
        `update_interval`.
        
        Args:
            tokens_processed (`int`): The number of tokens processed during
                                      the just-completed training step.
        """        
        step_time = time.time() - self.step_start_time
        
        # Update counters
        self.step_count += 1
        self.token_count += tokens_processed
        
        # Update profiler
        if self.profiler:
            self.profiler.step()
        
        # Update metrics periodically
        if self.step_count % self.update_interval == 0:
            self._update_metrics(step_time, tokens_processed)
    
    def _update_metrics(self, step_time: float, tokens_processed: int):
        """
        Calculates and updates various performance metrics, including
        overall throughput (tokens per second), memory usage (allocated/reserved),
        and CPU/GPU utilization. These metrics are then logged via the
        `XORZENXLogger` instance.
        
        Args:
            step_time (`float`): The time taken to complete the last training step in seconds.
            tokens_processed (`int`): The number of tokens processed in the last step.
        """        # Compute throughput
        total_time = time.time() - self.start_time
        self.metrics["throughput_tokens_per_sec"] = self.token_count / total_time
        
        # Memory usage
        if TORCH_AVAILABLE:
            self.metrics["memory_allocated_mb"] = (
                torch.cuda.memory_allocated() / 1e6
                if torch.cuda.is_available()
                else 0.0
            )
            self.metrics["memory_reserved_mb"] = (
                torch.cuda.memory_reserved() / 1e6
                if torch.cuda.is_available()
                else 0.0
            )
        
        # CPU usage (simplified)
        self.metrics["cpu_usage_percent"] = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
        
        # GPU usage (if available)
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.metrics["gpu_usage_percent"] = torch.cuda.utilization()
        
        # Log metrics
        self.logger.log_model_metrics(
            self.metrics,
            self.step_count,
            epoch=0,  # Will be updated by trainer
            prefix="performance"
        )
    
    def _on_trace_ready(self, prof):
        """
        Callback function invoked by the PyTorch profiler when a trace is ready.
        This method is responsible for saving the generated trace to a file,
        typically in Chrome Trace Format, for detailed analysis.
        
        Args:
            prof: The PyTorch profiler instance containing the trace data.
        """        # Save profiler trace
        trace_file = self.logger.log_dir / f"trace_step_{self.step_count}.json"
        prof.export_chrome_trace(str(trace_file))
        
        self.logger.info(
            "performance_monitor",
            f"Saved profiler trace to {trace_file}"
        )
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive performance report summarizing various
        metrics and efficiency indicators collected by the monitor.
        
        Returns:
            A dictionary containing:
            - `metrics`: Current performance metrics (throughput, memory, CPU/GPU usage).
            - `counters`: Accumulated step and token counts, and total monitoring time.
            - `efficiency`: Derived efficiency metrics such as tokens per second,
                            memory efficiency, and compute utilization.
        """        
        return {
            "metrics": self.metrics,
            "counters": {
                "step_count": self.step_count,
                "token_count": self.token_count,
                "total_time_sec": time.time() - self.start_time
            },
            "efficiency": {
                "tokens_per_second": self.metrics["throughput_tokens_per_sec"],
                "memory_efficiency": self.token_count / (self.metrics["memory_allocated_mb"] + 1e-6),
                "compute_utilization": min(
                    self.metrics["cpu_usage_percent"],
                    self.metrics.get("gpu_usage_percent", 0.0)
                ) / 100.0
            }
        }
    
    def reset(self):
        """
        Resets all internal counters and metrics of the performance monitor
        to their initial states. This is useful for starting a new measurement
        period without re-instantiating the monitor.
        """        
        self.step_count = 0
        self.token_count = 0
        self.start_time = time.time()
        self.metrics = {k: 0.0 for k in self.metrics}


# ==================== GLOBAL LOGGER INSTANCE ====================

# Global logger instance
_GLOBAL_LOGGER: Optional[XORZENXLogger] = None
_GLOBAL_DISTRIBUTED_LOGGER: Optional[DistributedLogger] = None
_GLOBAL_PERFORMANCE_MONITOR: Optional[PerformanceMonitor] = None


def setup_global_logger(
    name: str = "xorzen",
    log_dir: Union[str, Path] = "logs",
    level: LogLevel = LogLevel.INFO,
    enable_async: bool = True,
    distributed_world_size: int = 1,
    distributed_rank: int = 0,
    enable_performance_monitoring: bool = True
):
    """
    Initializes and configures the global `XORZENXLogger`, `DistributedLogger` (if applicable),
    and `PerformanceMonitor` instances. This function should be called once at the
    beginning of the application lifecycle to establish the central logging and
    monitoring infrastructure.
    
    Args:
        name (`str`, *optional*): The logical name for the global logger. Defaults to "xorzen".
        log_dir (`Union[str, Path]`, *optional*): The base directory for all log files.
                                                  Defaults to "logs".
        level (`LogLevel`, *optional*): The minimum logging level for messages.
                                        Defaults to `LogLevel.INFO`.
        enable_async (`bool`, *optional*): If `True`, enables asynchronous logging
                                           for improved performance. Defaults to `True`.
        distributed_world_size (`int`, *optional*): The total number of processes
                                                    in a distributed training setup.
                                                    If > 1, a `DistributedLogger` is also initialized.
                                                    Defaults to 1.
        distributed_rank (`int`, *optional*): The rank of the current process within
                                              the distributed training environment.
                                              Defaults to 0.
        enable_performance_monitoring (`bool`, *optional*): If `True`, a `PerformanceMonitor`
                                                            instance will be set up to collect
                                                            runtime performance metrics.
                                                            Defaults to `True`.
    """
    global _GLOBAL_LOGGER, _GLOBAL_DISTRIBUTED_LOGGER, _GLOBAL_PERFORMANCE_MONITOR
    
    # Create base logger
    _GLOBAL_LOGGER = XORZENXLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        enable_async=enable_async,
        enable_file=True, # Explicitly enable file logging
        enable_metrics=True, # Explicitly enable metrics logging
        distributed_rank=distributed_rank
    )
    
    # Create distributed logger if needed
    if distributed_world_size > 1:
        _GLOBAL_DISTRIBUTED_LOGGER = DistributedLogger(
            _GLOBAL_LOGGER,
            world_size=distributed_world_size,
            rank=distributed_rank
        )
    
    # Create performance monitor if enabled
    if enable_performance_monitoring:
        _GLOBAL_PERFORMANCE_MONITOR = PerformanceMonitor(_GLOBAL_LOGGER)
    
    # Log initialization
    get_logger().info(
        "global_logger",
        "Global logger initialized",
        {
            "log_dir": str(log_dir),
            "level": level.name,
            "distributed_world_size": distributed_world_size,
            "distributed_rank": distributed_rank
        }
    )


def get_logger() -> XORZENXLogger:
    """
    Retrieves the singleton instance of the global `XORZENXLogger`.
    If the logger has not yet been initialized via `setup_global_logger()`,
    it will be initialized with default parameters.
    
    Returns:
        `XORZENXLogger`: The active global logger instance.
    """
    global _GLOBAL_LOGGER
    
    if _GLOBAL_LOGGER is None:
        # Setup default logger
        setup_global_logger()
    
    return _GLOBAL_LOGGER


def get_distributed_logger() -> Optional[DistributedLogger]:
    """
    Retrieves the singleton instance of the global `DistributedLogger`.
    This logger is only available if `setup_global_logger()` was called
    with `distributed_world_size > 1`.
    
    Returns:
        `Optional[DistributedLogger]`: The active global distributed logger instance,
                                      or `None` if distributed logging was not
                                      initialized.
    """    
    global _GLOBAL_DISTRIBUTED_LOGGER
    return _GLOBAL_DISTRIBUTED_LOGGER


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """
    Retrieves the singleton instance of the global `PerformanceMonitor`.
    This monitor is only available if `setup_global_logger()` was called
    with `enable_performance_monitoring=True`.
    
    Returns:
        `Optional[PerformanceMonitor]`: The active global performance monitor instance,
                                        or `None` if performance monitoring was not
                                        initialized.
    """    
    global _GLOBAL_PERFORMANCE_MONITOR
    return _GLOBAL_PERFORMANCE_MONITOR


# ==================== CONTEXT MANAGERS ====================

class timed_block:
    """
    A context manager designed to measure and log the execution time
    of arbitrary code blocks. Upon exiting the block, it logs the elapsed
    time, providing insights into performance bottlenecks.
    """    
    def __init__(self, name: str, logger: Optional[XORZENXLogger] = None):
        """
        Initializes the `timed_block` context manager.
        
        Args:
            name (`str`): A descriptive name for the code block being timed.
            logger (`XORZENXLogger`, *optional*): An optional `XORZENXLogger` instance
                                               to use for logging the elapsed time.
                                               If `None`, the global logger (`get_logger()`)
                                               will be used.
        """        
        self.name = name
        self.logger = logger or get_logger()
    
    def __enter__(self):
        """
        Enters the context, recording the starting time of the block.
        
        Returns:
            The `timed_block` instance.
        """
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, calculates the elapsed time, and logs it.
        
        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      `None` if no exception occurred.
            exc_val: The exception instance that occurred. `None` if no exception occurred.
            exc_tb: The traceback object. `None` if no exception occurred.
        """
        elapsed = time.time() - self.start_time
        self.logger.info(
            "timed_block",
            f"Block '{self.name}' completed in {elapsed:.3f}s",
            {"elapsed_seconds": elapsed}
        )


class log_exceptions:
    """
    A context manager designed to catch and log any exceptions that occur
    within its scope. It provides a standardized way to log unexpected errors
    without suppressing them, allowing for centralized error reporting.
    """    
    def __init__(self, module: str, logger: Optional[XORZENXLogger] = None):
        """
        Initializes the `log_exceptions` context manager.
        
        Args:
            module (`str`): The name of the module or component associated with
                            the exceptions caught by this context manager.
            logger (`XORZENXLogger`, *optional*): An optional `XORZENXLogger` instance
                                               to use for logging exceptions.
                                               If `None`, the global logger (`get_logger()`)
                                               will be used.
        """        
        self.module = module
        self.logger = logger or get_logger()
    
    def __enter__(self):
        """
        Enters the context, returning the context manager instance itself.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context, logging any exceptions that occurred within the block.
        The exception is logged at `LogLevel.ERROR` severity, including its details.
        The exception is not suppressed and will propagate normally.
        
        Args:
            exc_type: The type of the exception that caused the context to be exited.
                      `None` if no exception occurred.
            exc_val: The exception instance that occurred. `None` if no exception occurred.
            exc_tb: The traceback object. `None` if no exception occurred.
            
        Returns:
            `False` to indicate that if an exception occurred, it should be propagated
            after being logged.
        """
        if exc_type is not None:
            self.logger.error(
                self.module,
                f"Exception occurred: {exc_type.__name__}",
                exception=exc_val
            )
        # Don't suppress the exception
        return False


# ==================== TESTING ====================

__all__ = [
    'LogLevel',
    'LogEntry',
    'MetricEntry',
    'XORZENXLogger',
    'DistributedLogger',
    'PerformanceMonitor',
    'setup_global_logger',
    'get_logger',
    'get_distributed_logger',
    'get_performance_monitor',
    'timed_block',
    'log_exceptions',
]

