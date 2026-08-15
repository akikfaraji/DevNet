"""
Command-Line Interface for xorzen Data Processing
Unified CLI for inspect, clean, convert, and process operations.
"""

import argparse
import sys
import json
from pathlib import Path

from xorzen.utils.logger import setup_global_logger, get_logger, LogLevel
from xorzen.data.inspector import DataInspector
from xorzen.data.cleaner import DataCleaner
from xorzen.data.converter import DataConverter
from xorzen.data.processor import tokenize_and_save


def setup_logger(verbose: bool = False):
    """Setup global logger."""
    level = LogLevel.DEBUG if verbose else LogLevel.INFO
    setup_global_logger(name="xorzen-data-cli", log_dir="logs_cli", level=level, enable_async=False)


def cmd_inspect(args):
    """Handle inspect command."""
    inspector = DataInspector()
    
    path = Path(args.path)
    if path.is_file():
        stats = inspector.inspect_file(path, args.text_keys, args.sample_size)
    elif path.is_dir():
        stats = inspector.inspect_directory(path, args.text_keys, args.recursive, args.sample_size)
    else:
        print(f"Error: Invalid path: {args.path}")
        sys.exit(1)
    
    inspector.generate_report(stats, args.output)
    print(f"\n✓ Inspection complete!")


def cmd_clean(args):
    """Handle clean command."""
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
        if args.min_length is not None:
            config["min_length"] = args.min_length
        if args.max_length is not None:
            config["max_length"] = args.max_length
        if args.min_words is not None:
            config["min_words"] = args.min_words
        if args.remove_duplicates: # This is a boolean flag, so if present, it's True
            config["remove_duplicates"] = True
        if args.remove_urls: # This is a boolean flag, so if present, it's True
            config["remove_urls"] = True
        if args.remove_emails: # This is a boolean flag, so if present, it's True
            config["remove_emails"] = True
    
    cleaner = DataCleaner(config)
    
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        stats = cleaner.clean_file(input_path, output_path, args.text_keys)
    elif input_path.is_dir():
        stats = cleaner.clean_directory(input_path, output_path, args.text_keys, args.recursive)
    else:
        print(f"Error: Invalid input path: {args.input}")
        sys.exit(1)
    
    # Save report
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Cleaning complete!")
    print(f"  Processed: {stats.get('total_processed', 0)}")
    print(f"  Removed: {stats.get('total_removed', 0)}")
    print(f"  Modified: {stats.get('total_modified', 0)}")


def cmd_convert(args):
    """Handle convert command."""
    converter = DataConverter(max_workers=args.workers)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        stats = converter.convert_file(
            input_path, output_path,
            args.input_format, args.output_format,
            args.text_column, args.chunk_size
        )
    elif input_path.is_dir():
        if not args.input_format or not args.output_format:
            print("Error: --input-format and --output-format required for directory conversion")
            sys.exit(1)
        
        stats = converter.convert_directory(
            input_path, output_path,
            args.input_format, args.output_format,
            args.text_column, args.recursive, args.parallel
        )
    else:
        print(f"Error: Invalid input path: {args.input}")
        sys.exit(1)
    
    # Save report
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Conversion complete!")
    if "total_files" in stats:
        print(f"  Files: {stats['total_files']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
    else:
        print(f"  Records: {stats.get('records', stats.get('lines', 0))}")


def cmd_tokenize(args):
    """Handle tokenize command."""
    try:
        tokenize_and_save(
            input_paths=args.inputs,
            tokenizer_path=args.tokenizer,
            output_filepath=args.output,
            output_format=args.format,
            text_keys=args.text_keys,
            max_length=args.max_length,
            stride=args.stride
        )
        print(f"\n✓ Tokenization complete! Output saved to: {args.output}")
    except Exception as e:
        print(f"\n✗ Tokenization failed: {e}")
        sys.exit(1)


def cmd_pipeline(args):
    """Handle pipeline command (inspect -> clean -> convert -> tokenize)."""
    print("Running full data pipeline...")
    
    # Step 1: Inspect
    print("\n[1/4] Inspecting data...")
    inspector = DataInspector()
    inspect_stats = inspector.inspect_directory(args.input, args.text_keys, True, 100)
    inspector.generate_report(inspect_stats, args.output_dir / "inspect_report.json")
    
    # Step 2: Clean
    print("\n[2/4] Cleaning data...")
    clean_config = {}
    if args.min_length is not None:
        clean_config["min_length"] = args.min_length
    if args.max_length is not None:
        clean_config["max_length"] = args.max_length
    if args.min_words is not None:
        clean_config["min_words"] = args.min_words
    if args.remove_duplicates:
        clean_config["remove_duplicates"] = True
    if args.remove_urls:
        clean_config["remove_urls"] = True
    if args.remove_emails:
        clean_config["remove_emails"] = True
    
    cleaner = DataCleaner(clean_config)
    cleaned_dir = args.output_dir / "cleaned"
    clean_stats = cleaner.clean_directory(args.input, cleaned_dir, args.text_keys, True)
    
    with open(args.output_dir / "clean_report.json", 'w') as f:
        json.dump(clean_stats, f, indent=2)
    
    # Step 3: Convert
    if args.convert_to:
        print(f"\n[3/4] Converting to {args.convert_to}...")
        converter = DataConverter()
        converted_dir = args.output_dir / "converted"
        
        # Detect input format from cleaned files
        input_format = None
        for f in cleaned_dir.iterdir():
            if f.is_file():
                input_format = f.suffix[1:]
                break
        
        if input_format:
            convert_stats = converter.convert_directory(
                cleaned_dir, converted_dir,
                input_format, args.convert_to,
                args.text_keys[0], True, True
            )
            with open(args.output_dir / "convert_report.json", 'w') as f:
                json.dump(convert_stats, f, indent=2)
    else:
        converted_dir = cleaned_dir
    
    # Step 4: Tokenize
    if args.tokenizer:
        print("\n[4/4] Tokenizing data...")
        tokenize_and_save(
            input_paths=[str(converted_dir)],
            tokenizer_path=args.tokenizer,
            output_filepath=str(args.output_dir / f"tokenized_data.{args.token_format}"),
            output_format=args.token_format,
            text_keys=args.text_keys
        )
    
    print(f"\n✓ Pipeline complete! Output in: {args.output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="xorzen Data Processing CLI - Unified tool for data operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a file
  python -m xorzen.data.cli inspect data.jsonl --output report.json
  
  # Clean data with custom settings
  python -m xorzen.data.cli clean input.jsonl output.jsonl --min-length 50 --remove-duplicates
  
  # Convert formats
  python -m xorzen.data.cli convert input.parquet output.jsonl
  
  # Tokenize data
  python -m xorzen.data.cli tokenize data/ --tokenizer tokenizer.json --output tokens.npy
  
  # Run full pipeline
  python -m xorzen.data.cli pipeline raw_data/ --output-dir processed/ --tokenizer tokenizer.json
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect and analyze data')
    inspect_parser.add_argument('path', help='File or directory to inspect')
    inspect_parser.add_argument('--text-keys', nargs='+', default=['text'], help='Text field keys')
    inspect_parser.add_argument('--sample-size', type=int, default=100, help='Sample size')
    inspect_parser.add_argument('--recursive', action='store_true', help='Recursive directory scan')
    inspect_parser.add_argument('--output', '-o', help='Output report path')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean and filter data')
    clean_parser.add_argument('input', help='Input file or directory')
    clean_parser.add_argument('output', help='Output file or directory')
    clean_parser.add_argument('--text-keys', nargs='+', default=['text'], help='Text field keys')
    clean_parser.add_argument('--config', help='JSON config file path')
    clean_parser.add_argument('--min-length', type=int, help='Minimum text length')
    clean_parser.add_argument('--max-length', type=int, help='Maximum text length')
    clean_parser.add_argument('--min-words', type=int, help='Minimum word count')
    clean_parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicates')
    clean_parser.add_argument('--remove-urls', action='store_true', help='Remove URLs')
    clean_parser.add_argument('--remove-emails', action='store_true', help='Remove emails')
    clean_parser.add_argument('--recursive', action='store_true', help='Recursive processing')
    clean_parser.add_argument('--report', help='Save report to file')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between formats')
    convert_parser.add_argument('input', help='Input file or directory')
    convert_parser.add_argument('output', help='Output file or directory')
    convert_parser.add_argument('--input-format', help='Input format (auto-detect if omitted)')
    convert_parser.add_argument('--output-format', help='Output format (auto-detect if omitted)')
    convert_parser.add_argument('--text-column', default='text', help='Text column name')
    convert_parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size')
    convert_parser.add_argument('--recursive', action='store_true', help='Recursive processing')
    convert_parser.add_argument('--parallel', action='store_true', help='Parallel processing')
    convert_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    convert_parser.add_argument('--report', help='Save report to file')
    
    # Tokenize command
    tokenize_parser = subparsers.add_parser('tokenize', help='Tokenize data for training')
    tokenize_parser.add_argument('inputs', nargs='+', help='Input files or directories')
    tokenize_parser.add_argument('--tokenizer', required=True, help='Tokenizer JSON path')
    tokenize_parser.add_argument('--output', '-o', required=True, help='Output file path')
    tokenize_parser.add_argument('--format', choices=['npy', 'pt', 'bin'], default='npy',
                                 help='Output format')
    tokenize_parser.add_argument('--text-keys', nargs='+', default=['text'], help='Text field keys')
    tokenize_parser.add_argument('--max-length', type=int, help='Max sequence length')
    tokenize_parser.add_argument('--stride', type=int, default=0, help='Stride for chunking')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full processing pipeline')
    pipeline_parser.add_argument('input', help='Input directory')
    pipeline_parser.add_argument('--output-dir', required=True, type=Path, help='Output directory')
    pipeline_parser.add_argument('--text-keys', nargs='+', default=['text'], help='Text field keys')
    pipeline_parser.add_argument('--min-length', type=int, help='Minimum text length for cleaning')
    pipeline_parser.add_argument('--max-length', type=int, help='Maximum text length for cleaning')
    pipeline_parser.add_argument('--min-words', type=int, help='Minimum word count for cleaning')
    pipeline_parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate texts during cleaning')
    pipeline_parser.add_argument('--remove-urls', action='store_true', help='Remove URLs during cleaning')
    pipeline_parser.add_argument('--remove-emails', action='store_true', help='Remove emails during cleaning')
    pipeline_parser.add_argument('--convert-to', choices=['jsonl', 'json', 'txt'], 
                                 help='Convert to format')
    pipeline_parser.add_argument('--tokenizer', help='Tokenizer path (optional)')
    pipeline_parser.add_argument('--token-format', choices=['npy', 'pt', 'bin'], default='npy',
                                 help='Token output format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logger
    setup_logger(args.verbose)
    
    try:
        if args.command == 'inspect':
            cmd_inspect(args)
        elif args.command == 'clean':
            cmd_clean(args)
        elif args.command == 'convert':
            cmd_convert(args)
        elif args.command == 'tokenize':
            cmd_tokenize(args)
        elif args.command == 'pipeline':
            cmd_pipeline(args)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        get_logger().critical("cli", f"Command failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    finally:
        get_logger().cleanup()


if __name__ == '__main__':
    main()

