#!/usr/bin/env python3
"""
Frame Cleaner CLI

A command-line interface for cleaning video frames of sensitive information.
Mirrors the style of report_reader.py CLI.

Usage Examples:
    # Clean a single video
    frame_cleaner_cli clean /path/to/video.mp4

    # Clean with custom output directory
    frame_cleaner_cli clean /path/to/video.mp4 --output-dir /path/to/output/

    # Batch clean multiple videos
    frame_cleaner_cli batch /path/to/videos/ --output-dir /path/to/output/

    # Clean with verbose output
    frame_cleaner_cli clean /path/to/video.mp4 --verbose
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add the current directory to Python path to import lx_anonymizer
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

try:
    from lx_anonymizer.frame_cleaner import FrameCleaner
    from lx_anonymizer.report_reader import ReportReader
    from lx_anonymizer.custom_logger import logger, configure_global_logger
except ImportError as e:
    print(f"Error importing lx_anonymizer modules: {e}")
    print("Make sure you're running this from the lx-anonymizer directory.")
    sys.exit(1)


class FrameCleanerCLI:
    """CLI wrapper for frame cleaning functionality."""
    
    def __init__(self):
        self.report_reader = None
    
    def setup_logging(self, level: str = "INFO"):
        """Setup logging for the CLI application."""
        verbose = level.upper() == "DEBUG"
        configure_global_logger(verbose=verbose)
        
        # Also set the root logger level
        root_logger = logging.getLogger()
        log_level = getattr(logging, level.upper(), logging.INFO)
        root_logger.setLevel(log_level)
    
    def create_report_reader(self) -> ReportReader:
        """Create and configure a ReportReader instance for sensitive data detection."""
        if not self.report_reader:
            self.report_reader = ReportReader(
                locale="de_DE",
                text_date_format="%d.%m.%Y"
            )
        return self.report_reader
    
    def clean_single_video(self, 
                          video_path: str,
                          output_dir: str = None,
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Clean a single video file.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save cleaned video (optional)
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing processing results
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            logger.info(f"Processing video: {video_path}")
            
            # Create report reader for sensitive data detection
            report_reader = self.create_report_reader()
            
            # Clean the video
            cleaned_video_path = FrameCleaner.clean_video(video_path, report_reader)
            
            # Move to output directory if specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                final_path = output_path / cleaned_video_path.name
                import shutil
                shutil.move(str(cleaned_video_path), str(final_path))
                cleaned_video_path = final_path
            
            results = {
                "original_video": str(video_path),
                "cleaned_video": str(cleaned_video_path),
                "success": True,
                "original_size": video_path.stat().st_size,
                "cleaned_size": cleaned_video_path.stat().st_size if cleaned_video_path.exists() else 0
            }
            
            if verbose:
                self.print_cleaning_summary(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Error cleaning {video_path}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {
                "original_video": str(video_path),
                "error": error_msg,
                "success": False
            }
    
    def batch_clean(self,
                   input_dir: str,
                   output_dir: str,
                   pattern: str = "*.mp4",
                   max_files: int = None,
                   continue_on_error: bool = True) -> List[Dict[str, Any]]:
        """
        Batch clean multiple video files.
        
        Args:
            input_dir: Directory containing video files
            output_dir: Directory to save cleaned videos
            pattern: File pattern to match
            max_files: Maximum number of files to process
            continue_on_error: Whether to continue after errors
            
        Returns:
            List of processing results for each file
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all video files
        video_files = list(input_path.glob(pattern))
        if max_files:
            video_files = video_files[:max_files]
        
        logger.info(f"Found {len(video_files)} video files to process")
        
        results = []
        for i, video_file in enumerate(video_files, 1):
            logger.info(f"Processing file {i}/{len(video_files)}: {video_file.name}")
            
            try:
                result = self.clean_single_video(
                    video_path=str(video_file),
                    output_dir=output_dir,
                    verbose=False  # Reduce verbosity for batch processing
                )
                results.append(result)
                
            except Exception as e:
                error_result = {
                    "original_video": str(video_file),
                    "error": str(e),
                    "success": False
                }
                results.append(error_result)
                
                if not continue_on_error:
                    logger.error(f"Stopping batch processing due to error: {e}")
                    break
                else:
                    logger.warning(f"Error processing {video_file}, continuing: {e}")
        
        logger.info("Batch processing complete")
        self.print_batch_summary(results)
        
        return results
    
    def print_cleaning_summary(self, results: Dict[str, Any]):
        """Print a summary of video cleaning results."""
        print("\n" + "="*60)
        print("VIDEO CLEANING SUMMARY")
        print("="*60)
        print(f"Original Video: {results['original_video']}")
        
        if results.get('success'):
            print(f"Cleaned Video: {results['cleaned_video']}")
            print(f"Original Size: {results['original_size']:,} bytes")
            print(f"Cleaned Size: {results['cleaned_size']:,} bytes")
            
            size_diff = results['original_size'] - results['cleaned_size']
            if size_diff > 0:
                print(f"Size Reduction: {size_diff:,} bytes ({size_diff/results['original_size']*100:.1f}%)")
            else:
                print("Size Change: Minimal (no frames removed)")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        print("="*60 + "\n")
    
    def print_batch_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of batch processing results."""
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        print("\n" + "="*60)
        print("BATCH CLEANING SUMMARY")
        print("="*60)
        print(f"Total Files: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            total_original = sum(r.get('original_size', 0) for r in successful)
            total_cleaned = sum(r.get('cleaned_size', 0) for r in successful)
            print(f"Total Size Processed: {total_original:,} bytes")
            print(f"Total Size After Cleaning: {total_cleaned:,} bytes")
            
            if total_original > total_cleaned:
                reduction = total_original - total_cleaned
                print(f"Total Size Reduction: {reduction:,} bytes ({reduction/total_original*100:.1f}%)")
        
        if failed:
            print("\nFailed Files:")
            for result in failed[:5]:  # Show first 5 failures
                print(f"  {Path(result['original_video']).name}: {result.get('error', 'Unknown error')}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        print("="*60 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Frame Cleaner CLI - Remove sensitive information from video frames",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Clean command
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean a single video file"
    )
    clean_parser.add_argument("video_path", help="Path to the video file")
    clean_parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save cleaned video"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch clean multiple video files"
    )
    batch_parser.add_argument("input_dir", help="Directory containing video files")
    batch_parser.add_argument("--output-dir", "-o", required=True, help="Directory to save cleaned videos")
    batch_parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="File pattern to match (default: *.mp4)"
    )
    batch_parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process"
    )
    batch_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop processing on first error"
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
    cli = FrameCleanerCLI()
    cli.setup_logging(args.log_level)
    
    try:
        if args.command == "clean":
            result = cli.clean_single_video(
                video_path=args.video_path,
                output_dir=args.output_dir,
                verbose=True
            )
            if not result.get('success'):
                sys.exit(1)
        
        elif args.command == "batch":
            results = cli.batch_clean(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                pattern=args.pattern,
                max_files=args.max_files,
                continue_on_error=not args.stop_on_error
            )
            failed_count = len([r for r in results if not r.get('success')])
            if failed_count > 0:
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