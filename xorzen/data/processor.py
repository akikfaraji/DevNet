import os
import sys
import json
import argparse
from typing import Union, Iterator, List, Dict, Any, Optional
from pathlib import Path

# Import the xorzen logger
from xorzen.utils.logger import get_logger, setup_global_logger, LogLevel
# Import Tokenizer
from tokenizers import Tokenizer

# Optional imports for saving formats
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    get_logger().warning("data.processor", "NumPy not available. .npy format saving will not be supported.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    get_logger().warning("data.processor", "PyTorch not available. .pt format saving will not be supported.")


def _read_txt_file(filepath: Union[str, Path]) -> Iterator[str]:
    """Reads a text file line by line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

def _read_jsonl_data(filepath: Union[str, Path], text_keys: Union[str, List[str]] = 'text') -> Iterator[str]:
    """Reads a JSONL file, yielding the content of the specified text_key(s)."""
    logger = get_logger()
    if isinstance(text_keys, str):
        text_keys = [text_keys] # Ensure it's always a list

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    found_key = False
                    for key in text_keys:
                        if key in record:
                            yield str(record[key])
                            found_key = True
                            break
                    if not found_key:
                        logger.warning("data.processor", f"Record missing all specified text keys {text_keys} in {filepath}. Skipping text extraction for this record.")
                except json.JSONDecodeError:
                    logger.warning("data.processor", f"Malformed JSONL line in {filepath}. Skipping record.")

def _read_json_data(filepath: Union[str, Path], text_keys: Union[str, List[str]] = 'text') -> Iterator[str]:
    """Reads a JSON file (expected to be an array of objects), yielding the content of the specified text_key(s)."""
    logger = get_logger()
    if isinstance(text_keys, str):
        text_keys = [text_keys] # Ensure it's always a list

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            logger.error("data.processor", f"Input JSON file '{filepath}' is not a JSON array. Cannot process.")
            return

        for record in data:
            found_key = False
            for key in text_keys:
                if key in record:
                    yield str(record[key])
                    found_key = True
                    break
            if not found_key:
                logger.warning("data.processor", f"Record missing all specified text keys {text_keys} in {filepath}. Skipping text extraction for this record.")


def load_tokenizer(tokenizer_json_path: Union[str, Path]) -> Tokenizer:
    """Loads a tokenizer from a JSON file."""
    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_json_path}")
    
    return Tokenizer.from_file(str(tokenizer_json_path))

def tokenize_and_save(
    input_paths: List[Union[str, Path]],
    tokenizer_path: Union[str, Path],
    output_filepath: Union[str, Path],
    output_format: str = 'npy', # 'bin', 'pt', 'npy'
    text_keys: Union[str, List[str]] = 'text', # For JSON/JSONL, now accepts list
    max_length: Optional[int] = None, # For truncation
    stride: int = 0 # For overlapping chunks
):
    """
    Tokenizes input text data using a specified tokenizer and saves the tokenized IDs
    to a file in NumPy, PyTorch, or raw binary format.

    Args:
        input_paths: List of paths to input data files (.txt, .json, .jsonl) or directories containing them.
        tokenizer_path: Path to the trained tokenizer JSON file.
        output_filepath: Path where the tokenized data will be saved.
        output_format: Desired output format ('bin', 'pt', 'npy').
        text_keys: Key(s) to extract text from JSON/JSONL records (default: 'text'). Can be a string or a list of strings.
        max_length: Maximum sequence length for tokenization. If None, no truncation.
        stride: When max_length is set, the stride to use for overlapping chunks.
    """
    logger = get_logger()
    output_filepath = Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(tokenizer_path):
        logger.error("data.processor", f"Tokenizer file not found at: {tokenizer_path}")
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tokenizer = load_tokenizer(tokenizer_path)
    logger.info("data.processor", f"Tokenizer loaded from: {tokenizer_path}")

    # Configure tokenizer for truncation and padding if max_length is provided
    if max_length is not None:
        tokenizer.enable_truncation(max_length=max_length, stride=stride)
        tokenizer.enable_padding(length=max_length) # Assuming padding to max_length is desired
    
    all_token_ids = []
    total_texts_processed = 0
    supported_file_extensions = {'.txt', '.json', '.jsonl'}

    # Determine optimal dtype based on vocab size
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size <= 2**16: # Max value for uint16 is 65535
        output_numpy_dtype = np.uint16
        output_byte_size = 2
    elif vocab_size <= 2**32: # Max value for uint32 is 4294967295
        output_numpy_dtype = np.uint32
        output_byte_size = 4
    else:
        logger.warning("data.processor", f"Tokenizer vocab size ({vocab_size}) exceeds uint32 capacity. Using int64.")
        output_numpy_dtype = np.int64
        output_byte_size = 8

    files_to_process = []
    for input_path_item in input_paths:
        path_obj = Path(input_path_item)
        if path_obj.is_file():
            if path_obj.suffix.lower() in supported_file_extensions:
                files_to_process.append(path_obj)
            else:
                logger.warning("data.processor", f"Unsupported file type for tokenization: {path_obj.suffix}. Skipping {path_obj}.")
        elif path_obj.is_dir():
            logger.info("data.processor", f"Scanning directory: {path_obj} for supported files.")
            for ext in supported_file_extensions:
                files_to_process.extend(path_obj.rglob(f"*{ext}"))
        else:
            logger.warning("data.processor", f"Input path '{input_path_item}' not found or is not a supported file/directory type. Skipping.")
            
    if not files_to_process:
        logger.error("data.processor", "No supported input files found for tokenization.")
        raise ValueError("No supported input files found for tokenization.")


    for input_path in files_to_process:
        logger.info("data.processor", f"Processing input file: {input_path}")
        
        file_reader = None
        if input_path.suffix.lower() == '.txt':
            file_reader = _read_txt_file(input_path)
        elif input_path.suffix.lower() == '.jsonl':
            file_reader = _read_jsonl_data(input_path, text_keys=text_keys)
        elif input_path.suffix.lower() == '.json':
            file_reader = _read_json_data(input_path, text_keys=text_keys)
        # No need for else, as files_to_process already filtered for supported extensions.
        
        if file_reader:
            for text_content in file_reader:
                if not text_content: # Skip empty texts
                    continue
                
                # Tokenize the text
                # The tokenizer's truncation and padding settings are now enabled beforehand.
                # encode_kwargs can now be simpler or empty if only max_length/stride were there.
                encoding = tokenizer.encode(text_content)
                
                # If return_overflowing_tokens is True, encoding will be a list of encodings
                # Note: `enable_truncation` with `stride` implicitly handles `return_overflowing_tokens` 
                # for batch encoding, but for single `encode` it usually gives one Encoding object.
                # If the goal is to get multiple chunks from a single text, `tokenizer.encode` 
                # does not return a list of encodings unless specifically configured with a 
                # `Sequence` post-processor or similar.
                # For now, let's assume `tokenizer.encode` directly returns a single Encoding object.
                all_token_ids.extend(encoding.ids)
                
                total_texts_processed += 1
                if total_texts_processed % 1000 == 0:
                    logger.info("data.processor", f"Processed {total_texts_processed} texts from {input_path}...")

    logger.info("data.processor", f"Finished tokenizing all input files. Total texts processed: {total_texts_processed}")
    logger.info("data.processor", f"Total token IDs collected: {len(all_token_ids)}")

    # Save tokenized data
    if output_format == 'npy':
        if not NUMPY_AVAILABLE:
            logger.error("data.processor", "NumPy is not available. Cannot save to .npy format.")
            raise ImportError("NumPy is required for .npy output format.")
        
        token_array = np.array(all_token_ids, dtype=output_numpy_dtype) # Use determined dtype
        np.save(output_filepath, token_array)
        logger.info("data.processor", f"Tokenized data saved to '{output_filepath}' in NumPy format.")
    elif output_format == 'pt':
        if not TORCH_AVAILABLE:
            logger.error("data.processor", "PyTorch is not available. Cannot save to .pt format.")
            raise ImportError("PyTorch is required for .pt output format.")
        
        token_tensor = torch.tensor(all_token_ids, dtype=torch.int32) # Use int32 for PyTorch
        torch.save(token_tensor, output_filepath)
        logger.info("data.processor", f"Tokenized data saved to '{output_filepath}' in PyTorch format.")
    elif output_format == 'bin':
        # Raw binary format: write as sequence of bytes per token ID based on determined size
        with open(output_filepath, 'wb') as f:
            for token_id in all_token_ids:
                f.write(token_id.to_bytes(output_byte_size, 'little')) # Use determined byte size
        logger.info("data.processor", f"Tokenized data saved to '{output_filepath}' in raw binary (.bin) format (using {output_byte_size} bytes per token, little-endian).")
    else:
        logger.error("data.processor", f"Unsupported output format: {output_format}. Supported: 'npy', 'pt', 'bin'.")
        raise ValueError(f"Unsupported output format: {output_format}")


if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) 
    setup_global_logger(name="xorzen-data-processor", log_dir="logs_processor", level=LogLevel.INFO, enable_async=False)
    
    parser = argparse.ArgumentParser(description="Tokenize and save dataset for xorzen.")
    parser.add_argument("input_paths", nargs='+', type=str, 
                        help="One or more paths to input data files (.txt, .json, .jsonl) or directories containing them.")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the trained tokenizer JSON file.")
    parser.add_argument("--output_filepath", type=str, default="tokenized_data.npy",
                        help="Path where the tokenized data will be saved (e.g., tokenized_data.npy).")
    parser.add_argument("--output_format", type=str, choices=['npy', 'pt', 'bin'], default="npy",
                        help="Desired output format: 'npy' (NumPy), 'pt' (PyTorch), or 'bin' (raw binary).")
    parser.add_argument("--text_keys", nargs='+', type=str, default=["text"], 
                        help="Key(s) to extract text from JSON/JSONL records (default: ['text']). Pass multiple keys separated by spaces.")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum sequence length for tokenization. If set, sequences will be truncated.")
    parser.add_argument("--stride", type=int, default=0,
                        help="When max_length is set, the stride to use for overlapping chunks (default: 0).")
    
    args = parser.parse_args()

    try:
        tokenize_and_save(
            input_paths=args.input_paths,
            tokenizer_path=args.tokenizer_path,
            output_filepath=args.output_filepath,
            output_format=args.output_format,
            text_keys=args.text_keys,
            max_length=args.max_length,
            stride=args.stride
        )
    except Exception as e:
        get_logger().critical("data.processor", f"An unhandled error occurred during data processing: {e}", exc_info=True)
    finally:
        get_logger().cleanup()


__all__ = [
    'tokenize_and_save',
    'load_tokenizer',
]

