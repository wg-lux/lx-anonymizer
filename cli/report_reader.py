#!/usr/bin/env python3
"""
LX-Anonymizer Report Reader CLI

A command-line interface for processing medical reports with OCR, metadata extraction,
and anonymization capabilities using the LX-Anonymizer library.

Usage Examples:
    # Process a single PDF
    report_reader process /path/to/report.pdf

    # Process with OCR fallback and ensemble mode
    report_reader process /path/to/report.pdf --use-ocr --ensemble

    # Batch process multiple PDFs
    report_reader batch /path/to/reports/ --output-dir /path/to/output/

    # Use specific LLM for extraction
    report_reader process /path/to/report.pdf --llm-extractor deepseek

    # Extract only metadata without anonymization
    report_reader extract /path/to/report.pdf --json-output
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import traceback

# Add the current directory to Python path to import lx_anonymizer
current_dir = Path(__file__).parent.parent  # Go up to lx-anonymizer directory
sys.path.insert(0, str(current_dir))

try:
    from lx_anonymizer.report_reader import ReportReader
    from lx_anonymizer.custom_logger import logger, configure_global_logger
    # Try to import settings, but make it optional
    try:
        from lx_anonymizer.settings import DEFAULT_SETTINGS
    except ImportError:
        DEFAULT_SETTINGS = None
except ImportError as e:
    print(f"Error importing lx_anonymizer modules: {e}")
    print("Make sure you're running this from the lx-anonymizer directory.")
    print("Current working directory:", os.getcwd())
    print("Python path:", sys.path[:3])
    sys.exit(1)


class ReportReaderCLI:
    """CLI wrapper for the ReportReader functionality."""
    
    def __init__(self):
        self.reader = None
    
    def setup_logging(self, level: str = "INFO"):
        """Setup logging for the CLI application."""
        verbose = level.upper() == "DEBUG"
        configure_global_logger(verbose=verbose)
        
        # Also set the root logger level
        root_logger = logging.getLogger()
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
    
    def create_reader(self, locale: str = "de_DE", 
                     first_names: Optional[List[str]] = None,
                     last_names: Optional[List[str]] = None,
                     text_date_format: str = "%d.%m.%Y") -> ReportReader:
        """Create and configure a ReportReader instance."""
        if not self.reader:
            self.reader = ReportReader(
                locale=locale,
                employee_first_names=first_names,
                employee_last_names=last_names,
                text_date_format=text_date_format
            )
        return self.reader
    
    def process_single_pdf(self, pdf_path: str, 
                          use_ensemble: bool = False,
                          use_llm_extractor: Optional[str] = None,
                          output_dir: Optional[str] = None,
                          save_meta: bool = True,
                          save_anonymized: bool = True,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Process a single PDF file and return results.
        
        Args:
            pdf_path: Path to the PDF file
            use_ensemble: Whether to use ensemble OCR
            use_llm_extractor: LLM model to use for extraction ('deepseek', 'medllama', 'llama3')
            output_dir: Directory to save output files
            save_meta: Whether to save metadata as JSON
            save_anonymized: Whether to save anonymized text
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing processing results
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Create reader instance
            reader = self.create_reader()
            
            # Process the report
            original_text, anonymized_text, metadata = reader.process_report(
                pdf_path=str(pdf_path),
                use_ensemble=use_ensemble,
                verbose=verbose,
                use_llm_extractor=use_llm_extractor
            )
            
            results = {
                "pdf_path": str(pdf_path),
                "original_text_length": len(original_text) if original_text else 0,
                "anonymized_text_length": len(anonymized_text) if anonymized_text else 0,
                "metadata": metadata,
                "processing_timestamp": datetime.now().isoformat(),
                "use_ensemble": use_ensemble,
                "use_llm_extractor": use_llm_extractor
            }
            
            # Save outputs if requested
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                base_name = pdf_path.stem
                
                if save_meta and metadata:
                    meta_file = output_path / f"{base_name}_metadata.json"
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f"Metadata saved to: {meta_file}")
                
                if save_anonymized and anonymized_text:
                    anon_file = output_path / f"{base_name}_anonymized.txt"
                    with open(anon_file, 'w', encoding='utf-8') as f:
                        f.write(anonymized_text)
                    logger.info(f"Anonymized text saved to: {anon_file}")
                
                # Save original text for comparison
                if original_text:
                    orig_file = output_path / f"{base_name}_original.txt"
                    with open(orig_file, 'w', encoding='utf-8') as f:
                        f.write(original_text)
                    logger.info(f"Original text saved to: {orig_file}")
                
                # Save full results
                results_file = output_path / f"{base_name}_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"Full results saved to: {results_file}")
            
            if verbose:
                self.print_processing_summary(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {
                "pdf_path": str(pdf_path),
                "error": error_msg,
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def batch_process(self, input_dir: str, 
                     output_dir: str,
                     pattern: str = "*.pdf",
                     use_ensemble: bool = False,
                     use_llm_extractor: Optional[str] = None,
                     max_files: Optional[int] = None,
                     continue_on_error: bool = True) -> List[Dict[str, Any]]:
        """
        Batch process multiple PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save output files
            pattern: File pattern to match (default: "*.pdf")
            use_ensemble: Whether to use ensemble OCR
            use_llm_extractor: LLM model to use for extraction
            max_files: Maximum number of files to process
            continue_on_error: Whether to continue processing after errors
            
        Returns:
            List of processing results for each file
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all PDF files
        pdf_files = list(input_path.glob(pattern))
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                result = self.process_single_pdf(
                    pdf_path=str(pdf_file),
                    use_ensemble=use_ensemble,
                    use_llm_extractor=use_llm_extractor,
                    output_dir=output_dir,
                    verbose=False  # Reduce verbosity for batch processing
                )
                results.append(result)
                
            except Exception as e:
                error_result = {
                    "pdf_path": str(pdf_file),
                    "error": str(e),
                    "processing_timestamp": datetime.now().isoformat()
                }
                results.append(error_result)
                
                if not continue_on_error:
                    logger.error(f"Stopping batch processing due to error: {e}")
                    break
                else:
                    logger.warning(f"Error processing {pdf_file}, continuing: {e}")
        
        # Save batch summary
        summary_file = Path(output_dir) / "batch_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "batch_timestamp": datetime.now().isoformat(),
                "total_files": len(pdf_files),
                "processed_files": len(results),
                "successful_files": len([r for r in results if "error" not in r]),
                "failed_files": len([r for r in results if "error" in r]),
                "settings": {
                    "use_ensemble": use_ensemble,
                    "use_llm_extractor": use_llm_extractor,
                    "pattern": pattern
                },
                "results": results
            }, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Batch processing complete. Summary saved to: {summary_file}")
        self.print_batch_summary(results)
        
        return results
    
    def extract_metadata_only(self, pdf_path: str,
                             use_llm_extractor: Optional[str] = None,
                             json_output: bool = False) -> Dict[str, Any]:
        """
        Extract only metadata from a PDF without anonymization.
        
        Args:
            pdf_path: Path to the PDF file
            use_llm_extractor: LLM model to use for extraction
            json_output: Whether to output as JSON
            
        Returns:
            Extracted metadata
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = self.create_reader()
            
            # Read text from PDF
            text = reader.read_pdf(str(pdf_path))
            
            # Extract metadata only
            if use_llm_extractor:
                if use_llm_extractor == 'deepseek':
                    metadata = reader.extract_report_meta_deepseek(text, str(pdf_path))
                elif use_llm_extractor == 'medllama':
                    metadata = reader.extract_report_meta_medllama(text, str(pdf_path))
                elif use_llm_extractor == 'llama3':
                    metadata = reader.extract_report_meta_llama3(text, str(pdf_path))
                else:
                    metadata = reader.extract_report_meta(text, str(pdf_path))
            else:
                metadata = reader.extract_report_meta(text, str(pdf_path))
            
            result = {
                "pdf_path": str(pdf_path),
                "metadata": metadata,
                "extraction_timestamp": datetime.now().isoformat(),
                "extractor_used": use_llm_extractor or "spacy_regex"
            }
            
            if json_output:
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            else:
                self.print_metadata(metadata)
            
            return result
            
        except Exception as e:
            error_msg = f"Error extracting metadata from {pdf_path}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def print_processing_summary(self, results: Dict[str, Any]):
        """Print a summary of processing results."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"PDF File: {results['pdf_path']}")
        print(f"Original Text Length: {results.get('original_text_length', 0)} characters")
        print(f"Anonymized Text Length: {results.get('anonymized_text_length', 0)} characters")
        print(f"Processing Time: {results.get('processing_timestamp', 'Unknown')}")
        
        if results.get('use_ensemble'):
            print("OCR Method: Ensemble OCR")
        
        if results.get('use_llm_extractor'):
            print(f"LLM Extractor: {results['use_llm_extractor']}")
        
        metadata = results.get('metadata', {})
        if metadata:
            print("\nExtracted Metadata:")
            self.print_metadata(metadata)
        
        if "error" in results:
            print(f"\nError: {results['error']}")
        
        print("="*60 + "\n")
    
    def print_metadata(self, metadata: Dict[str, Any]):
        """Print metadata in a formatted way."""
        if not metadata:
            print("  No metadata extracted")
            return
        
        for key, value in metadata.items():
            if value is not None:
                if key == "patient_dob" and hasattr(value, 'strftime'):
                    value = value.strftime('%Y-%m-%d')
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    def print_batch_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of batch processing results."""
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Files: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if failed:
            print("\nFailed Files:")
            for result in failed:
                print(f"  {Path(result['pdf_path']).name}: {result.get('error', 'Unknown error')}")
        
        print("="*60 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LX-Anonymizer Report Reader CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--locale",
        default="de_DE",
        help="Locale for text processing (default: de_DE)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process a single PDF file"
    )
    process_parser.add_argument("pdf_path", help="Path to the PDF file")
    process_parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save output files"
    )
    process_parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Force OCR processing even if text is available"
    )
    process_parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble OCR for better accuracy"
    )
    process_parser.add_argument(
        "--llm-extractor",
        choices=["deepseek", "medllama", "llama3"],
        help="Use specific LLM for metadata extraction"
    )
    process_parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Skip anonymization, only extract metadata"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple PDF files"
    )
    batch_parser.add_argument("input_dir", help="Directory containing PDF files")
    batch_parser.add_argument("--output-dir", "-o", required=True, help="Directory to save output files")
    batch_parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern to match (default: *.pdf)"
    )
    batch_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    batch_parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble OCR for better accuracy"
    )
    batch_parser.add_argument(
        "--llm-extractor",
        choices=["deepseek", "medllama", "llama3"],
        help="Use specific LLM for metadata extraction"
    )
    batch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing on first error"
    )
    
    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract metadata only (no anonymization)"
    )
    extract_parser.add_argument("pdf_path", help="Path to the PDF file")
    extract_parser.add_argument(
        "--llm-extractor",
        choices=["deepseek", "medllama", "llama3"],
        help="Use specific LLM for metadata extraction"
    )
    extract_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results as JSON"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize CLI
    cli = ReportReaderCLI()
    cli.setup_logging(args.log_level)
    
    try:
        if args.command == "process":
            result = cli.process_single_pdf(
                pdf_path=args.pdf_path,
                use_ensemble=args.ensemble,
                use_llm_extractor=args.llm_extractor,
                output_dir=args.output_dir,
                save_anonymized=not args.no_anonymize,
                verbose=True
            )
            if "error" in result:
                sys.exit(1)
        
        elif args.command == "batch":
            results = cli.batch_process(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                use_ensemble=args.ensemble,
                use_llm_extractor=args.llm_extractor,
                max_files=args.max_files,
                continue_on_error=not args.stop_on_error
            )
            failed_count = len([r for r in results if "error" in r])
            if failed_count > 0:
                sys.exit(1)
        
        elif args.command == "extract":
            result = cli.extract_metadata_only(
                pdf_path=args.pdf_path,
                use_llm_extractor=args.llm_extractor,
                json_output=args.json_output
            )
            if "error" in result:
                sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()