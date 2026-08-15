"""
This module defines a comprehensive hierarchy of custom exception classes
for the `xorzen` deep learning framework. These exceptions are designed to
provide production-grade error handling with actionable and context-rich
messages, facilitating easier debugging and robust application development.

The exception structure categorizes errors by their domain (e.g., Model, Data,
Tokenizer, Training, Checkpoint, Config), allowing for precise error identification
and handling. Each custom exception inherits from `XORZENXError`, which provides
a standardized mechanism for formatting messages, including details and suggestions.
"""

class XORZENXError(Exception):
    """
    Base exception class for all errors originating from the `xorzen` framework.
    This class standardizes error reporting by allowing detailed messages,
    structured additional information (`details`), and user-friendly `suggestion`s
    to be included, facilitating more effective error diagnosis and resolution.
    """
    
    def __init__(self, message: str, details: dict = None, suggestion: str = None):
        """
        Initializes a new instance of `XORZENXError`.
        
        Args:
            message: A concise description of the error.
            details: An optional dictionary containing additional context or data
                     relevant to the error.
            suggestion: An optional string providing a user-friendly suggestion
                        or action to resolve the error.
        """
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """
        Constructs a human-readable and informative error message by
        combining the base error message with any provided details
        and suggestions for resolution.
        
        Returns:
            A string representing the formatted error message.
        """
        msg = f"{self.__class__.__name__}: {self.message}"
        
        if self.details:
            msg += f"\nDetails: {self.details}"
        
        if self.suggestion:
            msg += f"\n💡 Suggestion: {self.suggestion}"
        
        return msg


# === MODEL ERRORS ===

class ModelError(XORZENXError):
    """
    Base exception for all errors specifically related to model operations
    within the `xorzen` framework, including loading, configuration, or runtime issues.
    """
    pass


class ModelNotFoundError(ModelError):
    """
    Raised when a requested model variant or a specific named model
    cannot be located or identified within the framework's registry.
    """
    
    def __init__(self, model_name: str, available_models: list = None):
        super().__init__(
            message=f"Model '{model_name}' not found",
            details={"requested": model_name, "available": available_models},
            suggestion=f"Use xorzen.list_models() to see available models"
        )


class ModelConfigError(ModelError):
    """
    Raised when the configuration for a model is found to be invalid,
    inconsistent, or does not meet the required specifications.
    This may occur during model instantiation or validation.
    """
    pass


class ModelLoadError(ModelError):
    """
    Raised when an attempt to load a model (e.g., from a checkpoint file
    or a pretrained state) encounters an issue, preventing successful
    model initialization or restoration.
    """
    pass


# === DATA ERRORS ===

class DataError(XORZENXError):
    """
    Base exception for all errors encountered during data processing,
    loading, validation, or conversion within the `xorzen` framework.
    """
    pass


class DataFormatError(DataError):
    """
    Raised when input data is provided in an unrecognized, invalid,
    or unsupported file format, preventing successful parsing or processing.
    """
    
    def __init__(self, format_type: str, reason: str, supported_formats: list = None):
        super().__init__(
            message=f"Invalid data format: {format_type}",
            details={"format": format_type, "reason": reason, "supported": supported_formats},
            suggestion="Check that your data files are properly formatted and the format is supported"
        )


class DataValidationError(DataError):
    """
    Raised when data fails to meet predefined quality, consistency, or
    structural requirements during a validation phase. This indicates
    that the data is not suitable for further processing or model training.
    """
    
    def __init__(self, validation_errors: list):
        super().__init__(
            message="Data validation failed",
            details={"errors": validation_errors},
            suggestion="Review the validation errors and fix your data accordingly"
        )


class DataLoadError(DataError):
    """
    Raised when an operation to load data into memory or a processing pipeline
    encounters an issue, preventing access to or utilization of the dataset.
    This may include issues like file corruption or access permissions.
    """
    pass


class DataConversionError(DataError):
    """
    Raised when an attempt to convert data from one format to another
    (e.g., from text to a binary format) encounters an unrecoverable error.
    """
    
    def __init__(self, from_format: str, to_format: str, reason: str):
        super().__init__(
            message=f"Failed to convert {from_format} to {to_format}",
            details={"from": from_format, "to": to_format, "reason": reason},
            suggestion="Check that input files are valid and accessible"
        )


# === TOKENIZER ERRORS ===

class TokenizerError(XORZENXError):
    """
    Base exception for all errors encountered during tokenizer operations,
    including loading, training, or application of tokenization.
    """
    pass


class TokenizerNotFoundError(TokenizerError):
    """
    Raised when a requested tokenizer (either by name for a pretrained one,
    or by path for a local file) cannot be located.
    """
    
    def __init__(self, tokenizer_name: str, available_tokenizers: list = None):
        super().__init__(
            message=f"Tokenizer '{tokenizer_name}' not found",
            details={"requested": tokenizer_name, "available": available_tokenizers},
            suggestion="Use xorzen.list_pretrained() to see available tokenizers"
        )


class TokenizerTrainingError(TokenizerError):
    """
    Raised when the process of training a new tokenizer from a corpus
    encounters an unrecoverable error or fails to converge.
    """
    pass


class TokenizerLoadError(TokenizerError):
    """
    Raised when an attempt to load a tokenizer from a file path encounters
    an issue, preventing successful initialization of the tokenizer.
    This may be due to file corruption, incorrect format, or access problems.
    """
    
    def __init__(self, path: str, reason: str):
        super().__init__(
            message=f"Failed to load tokenizer from {path}",
            details={"path": path, "reason": reason},
            suggestion="Check that the tokenizer file exists and is a valid tokenizer.json file"
        )


# === CONFIG ERRORS ===

class ConfigError(XORZENXError):
    """
    Base exception for all errors related to configuration management,
    including validation, loading, or saving of configuration objects.
    """
    pass


class ConfigValidationError(ConfigError):
    """
    Raised when a configuration object fails to pass internal validation checks,
    indicating that its parameters are inconsistent, incomplete, or do not
    adhere to predefined rules.
    """
    
    def __init__(self, validation_errors: list):
        super().__init__(
            message="Configuration validation failed",
            details={"errors": validation_errors},
            suggestion="Review the configuration errors and fix them accordingly"
        )


class ConfigLoadError(ConfigError):
    """
    Raised when an attempt to load a configuration from a file or other source
    encounters an issue, preventing successful deserialization or parsing of
    the configuration data.
    """
    pass


class ConfigSaveError(ConfigError):
    """
    Raised when an attempt to save a configuration object to a persistent
    storage (e.g., a file) encounters an issue, preventing the successful
    serialization and writing of the configuration data.
    """
    pass


# === CHECKPOINT ERRORS ===

class CheckpointError(XORZENXError):
    """
    Base exception for all errors pertaining to the management of model checkpoints,
    including issues with saving, loading, or verifying checkpoint integrity.
    """
    pass


class CheckpointNotFoundError(CheckpointError):
    """
    Raised when a requested model checkpoint file or directory
    cannot be located at the specified path.
    """
    
    def __init__(self, checkpoint_path: str):
        super().__init__(
            message=f"Checkpoint not found: {checkpoint_path}",
            details={"path": checkpoint_path},
            suggestion="Check that the checkpoint path is correct and the file exists"
        )


class CheckpointLoadError(CheckpointError):
    """
    Raised when an attempt to load a model checkpoint from storage
    encounters an issue, preventing the successful restoration of
    the model's state. This may be due to data corruption or
    incompatible formats.
    """
    
    def __init__(self, checkpoint_path: str, reason: str):
        super().__init__(
            message=f"Failed to load checkpoint from {checkpoint_path}",
            details={"path": checkpoint_path, "reason": reason},
            suggestion="The checkpoint file may be corrupted. Try an earlier checkpoint."
        )


class CheckpointSaveError(CheckpointError):
    """
    Raised when an attempt to save a model checkpoint to persistent storage
    encounters an issue, preventing the successful serialization and writing
    of the model's current state.
    """
    pass


class CheckpointVersionError(CheckpointError):
    """
    Raised when a loaded checkpoint is found to be incompatible with the
    current version of the `xorzen` framework, indicating potential
    structural or API changes that prevent direct use.
    """
    
    def __init__(self, expected_version: str, found_version: str):
        super().__init__(
            message="Checkpoint version mismatch",
            details={"expected": expected_version, "found": found_version},
            suggestion="This checkpoint was created with a different version of xorzen. You may need to migrate it."
        )


# === TRAINING ERRORS ===

class TrainingError(XORZENXError):
    """
    Base exception for all errors encountered during the model training lifecycle,
    including issues with the training loop, optimization, or evaluation phases.
    """
    pass


class TrainingInterruptedError(TrainingError):
    """
    Raised when the model training process is explicitly interrupted
    (e.g., by a user signal or system shutdown) before completion.
    """
    pass


class EpochContinuityError(TrainingError):
    """
    Raised when an attempt is made to continue training from a checkpoint
    with an epoch number that is incompatible with the requested starting epoch.
    This ensures that training continuation is aligned with the checkpoint's progress.
    """
    
    def __init__(self, checkpoint_epoch: int, requested_epoch: int):
        super().__init__(
            message="Epoch continuity error",
            details={"checkpoint_epoch": checkpoint_epoch, "requested_epoch": requested_epoch},
            suggestion=f"The checkpoint is from epoch {checkpoint_epoch}. Continue from that epoch or use a different checkpoint."
        )


# === UTILITY FUNCTIONS ===

def handle_error(error: Exception, logger=None, raise_error: bool = True):
    """
    Provides a centralized mechanism for handling exceptions within the `xorzen` framework.
    It optionally logs the error (either a formatted `XORZENXError` or an unexpected exception)
    and can re-raise the exception, allowing for consistent error management across the codebase.
    
    Args:
        error: The exception object that needs to be handled.
        logger: An optional logger instance (e.g., from Python's `logging` module)
                to record the error message. If `None`, no logging occurs.
        raise_error: A boolean flag indicating whether the exception should be
                     re-raised after processing. Defaults to `True`.
    """
    if logger:
        if isinstance(error, XORZENXError):
            logger.error("error", str(error))
        else:
            logger.error("error", f"Unexpected error: {error}")
    
    if raise_error:
        raise error


__all__ = [
    # Base
    'XORZENXError',
    
    # Model
    'ModelError',
    'ModelNotFoundError',
    'ModelConfigError',
    'ModelLoadError',
    
    # Data
    'DataError',
    'DataFormatError',
    'DataValidationError',
    'DataLoadError',
    'DataConversionError',
    
    # Tokenizer
    'TokenizerError',
    'TokenizerNotFoundError',
    'TokenizerTrainingError',
    'TokenizerLoadError',
    
    # Config
    'ConfigError',
    'ConfigValidationError',
    'ConfigLoadError',
    'ConfigSaveError',
    
    # Checkpoint
    'CheckpointError',
    'CheckpointNotFoundError',
    'CheckpointLoadError',
    'CheckpointSaveError',
    'CheckpointVersionError',
    
    # Training
    'TrainingError',
    'TrainingInterruptedError',
    'EpochContinuityError',
    
    # Utils
    'handle_error',
]