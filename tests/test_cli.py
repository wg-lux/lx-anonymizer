"""
Comprehensive tests for the LX-Anonymizer Report Reader CLI.

This test suite verifies all CLI functionality including:
- Command parsing and validation
- PDF processing workflows
- Batch processing capabilities
- Metadata extraction
- Error handling
- Output generation
"""

import pytest
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the CLI module
from cli.report_reader import (
    ReportReaderCLI,
    create_parser,
    main
)

class TestReportReaderCLI:
    """Test suite for the ReportReaderCLI class."""
    
    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance for testing."""
        return ReportReaderCLI()
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_pdf_path(self, temp_dir):
        """Create a mock PDF file for testing."""
        pdf_path = Path(temp_dir) / "test_report.pdf"
        pdf_path.touch()  # Create empty file
        return str(pdf_path)
    
    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata for testing."""
        return {
            "patient_first_name": "John",
            "patient_last_name": "Doe",
            "patient_dob": "1985-06-15",
            "examination_date": "2024-03-20",
            "examiner_name": "Dr. Smith",
            "center_name": "Medical Center"
        }
    
    def test_cli_initialization(self, cli_instance):
        """Test CLI initialization."""
        assert cli_instance.reader is None
        assert isinstance(cli_instance, ReportReaderCLI)
    
    def test_setup_logging(self, cli_instance):
        """Test logging setup functionality."""
        # Test different log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            cli_instance.setup_logging(level)
            # Verify no exceptions are raised
    
    @patch('cli.report_reader.ReportReader')
    def test_create_reader(self, mock_report_reader, cli_instance):
        """Test ReportReader instance creation."""
        mock_instance = Mock()
        mock_report_reader.return_value = mock_instance
        
        reader = cli_instance.create_reader(
            locale="de_DE",
            first_names=["John", "Jane"],
            last_names=["Doe", "Smith"]
        )
        
        assert reader == mock_instance
        mock_report_reader.assert_called_once_with(
            locale="de_DE",
            employee_first_names=["John", "Jane"],
            employee_last_names=["Doe", "Smith"],
            text_date_format="%d.%m.%Y"
        )
    
    @patch('cli.report_reader.ReportReader')
    def test_process_single_pdf_success(self, mock_report_reader, cli_instance, sample_pdf_path, sample_metadata, temp_dir):
        """Test successful single PDF processing."""
        # Setup mocks
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            "Original text content",
            "Anonymized text content", 
            sample_metadata
        )
        
        # Test processing
        result = cli_instance.process_single_pdf(
            pdf_path=sample_pdf_path,
            use_ensemble=True,
            use_llm_extractor="deepseek",
            output_dir=temp_dir,
            verbose=False
        )
        
        # Verify results
        assert result["pdf_path"] == sample_pdf_path
        assert result["original_text_length"] == 21
        assert result["anonymized_text_length"] == 24
        assert result["metadata"] == sample_metadata
        assert result["use_ensemble"] is True
        assert result["use_llm_extractor"] == "deepseek"
        assert "processing_timestamp" in result
        
        # Verify ReportReader was called correctly
        mock_reader_instance.process_report.assert_called_once_with(
            pdf_path=sample_pdf_path,
            use_ensemble=True,
            verbose=False,
            use_llm_extractor="deepseek"
        )
        
        # Verify output files were created
        output_path = Path(temp_dir)
        assert (output_path / "test_report_metadata.json").exists()
        assert (output_path / "test_report_anonymized.txt").exists()
        assert (output_path / "test_report_original.txt").exists()
        assert (output_path / "test_report_results.json").exists()
    
    def test_process_single_pdf_file_not_found(self, cli_instance):
        """Test processing non-existent PDF file."""
        result = cli_instance.process_single_pdf(
            pdf_path="/nonexistent/file.pdf",
            verbose=False
        )
        
        assert "error" in result
        assert "PDF file not found" in result["error"]
        assert result["pdf_path"] == "/nonexistent/file.pdf"
    
    @patch('cli.report_reader.ReportReader')
    def test_batch_process_success(self, mock_report_reader, cli_instance, temp_dir):
        """Test successful batch processing."""
        # Create multiple test PDF files
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir()
        output_dir = Path(temp_dir) / "output"
        
        pdf_files = []
        for i in range(3):
            pdf_path = input_dir / f"test_{i}.pdf"
            pdf_path.touch()
            pdf_files.append(pdf_path)
        
        # Setup mock
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            "Original text", "Anonymized text", {"test": "metadata"}
        )
        
        # Test batch processing
        results = cli_instance.batch_process(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            max_files=2,
            use_ensemble=False,
            continue_on_error=True
        )
        
        # Verify results
        assert len(results) == 2  # max_files limit
        for result in results:
            assert "error" not in result
            assert result["original_text_length"] == 13
            assert result["anonymized_text_length"] == 16
        
        # Verify summary file was created
        summary_file = output_dir / "batch_processing_summary.json"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        assert summary["total_files"] == 2
        assert summary["processed_files"] == 2
        assert summary["successful_files"] == 2
        assert summary["failed_files"] == 0
    
    def test_batch_process_directory_not_found(self, cli_instance, temp_dir):
        """Test batch processing with non-existent input directory."""
        with pytest.raises(FileNotFoundError):
            cli_instance.batch_process(
                input_dir="/nonexistent/directory",
                output_dir=temp_dir
            )
    
    @patch('cli.report_reader.ReportReader')
    def test_extract_metadata_only(self, mock_report_reader, cli_instance, sample_pdf_path, sample_metadata):
        """Test metadata-only extraction."""
        # Setup mock
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.read_pdf.return_value = "PDF text content"
        mock_reader_instance.extract_report_meta_deepseek.return_value = sample_metadata
        
        # Test extraction
        result = cli_instance.extract_metadata_only(
            pdf_path=sample_pdf_path,
            use_llm_extractor="deepseek",
            json_output=False
        )
        
        # Verify results
        assert result["pdf_path"] == sample_pdf_path
        assert result["metadata"] == sample_metadata
        assert result["extractor_used"] == "deepseek"
        assert "extraction_timestamp" in result
        
        # Verify correct method was called
        mock_reader_instance.extract_report_meta_deepseek.assert_called_once_with(
            "PDF text content", sample_pdf_path
        )
    
    @patch('cli.report_reader.ReportReader')
    def test_extract_metadata_different_extractors(self, mock_report_reader, cli_instance, sample_pdf_path, sample_metadata):
        """Test metadata extraction with different LLM extractors."""
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.read_pdf.return_value = "PDF text content"
        mock_reader_instance.extract_report_meta_medllama.return_value = sample_metadata
        mock_reader_instance.extract_report_meta_llama3.return_value = sample_metadata
        mock_reader_instance.extract_report_meta.return_value = sample_metadata
        
        # Test different extractors
        extractors = ["medllama", "llama3", None]
        expected_methods = [
            "extract_report_meta_medllama",
            "extract_report_meta_llama3", 
            "extract_report_meta"
        ]
        
        for extractor, expected_method in zip(extractors, expected_methods):
            result = cli_instance.extract_metadata_only(
                pdf_path=sample_pdf_path,
                use_llm_extractor=extractor,
                json_output=False
            )
            
            assert result["extractor_used"] == (extractor or "spacy_regex")
            getattr(mock_reader_instance, expected_method).assert_called()


class TestCLIArgumentParsing:
    """Test suite for CLI argument parsing."""
    
    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser.prog.endswith("report_reader.py")
        assert parser.description == "LX-Anonymizer Report Reader CLI"
    
    def test_parser_no_command(self):
        """Test parser with no command provided."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None
    
    def test_parser_process_command(self):
        """Test parsing process command arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "process", 
            "/path/to/file.pdf",
            "--output-dir", "/output",
            "--ensemble",
            "--llm-extractor", "deepseek",
            "--log-level", "DEBUG"
        ])
        
        assert args.command == "process"
        assert args.pdf_path == "/path/to/file.pdf"
        assert args.output_dir == "/output"
        assert args.ensemble is True
        assert args.llm_extractor == "deepseek"
        assert args.log_level == "DEBUG"
    
    def test_parser_batch_command(self):
        """Test parsing batch command arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "batch",
            "/input/dir",
            "--output-dir", "/output/dir",
            "--pattern", "*.pdf",
            "--max-files", "10",
            "--ensemble",
            "--stop-on-error"
        ])
        
        assert args.command == "batch"
        assert args.input_dir == "/input/dir"
        assert args.output_dir == "/output/dir"
        assert args.pattern == "*.pdf"
        assert args.max_files == 10
        assert args.ensemble is True
        assert args.stop_on_error is True
    
    def test_parser_extract_command(self):
        """Test parsing extract command arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "extract",
            "/path/to/file.pdf",
            "--llm-extractor", "llama3",
            "--json-output"
        ])
        
        assert args.command == "extract"
        assert args.pdf_path == "/path/to/file.pdf"
        assert args.llm_extractor == "llama3"
        assert args.json_output is True
    
    def test_parser_invalid_llm_extractor(self):
        """Test parser rejects invalid LLM extractor."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args([
                "process",
                "/path/to/file.pdf",
                "--llm-extractor", "invalid_extractor"
            ])


class TestCLIOutputHandling:
    """Test suite for CLI output handling."""
    
    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance for testing."""
        return ReportReaderCLI()
    
    def test_print_metadata(self, cli_instance, capsys):
        """Test metadata printing functionality."""
        metadata = {
            "patient_first_name": "John",
            "patient_last_name": "Doe",
            "patient_dob": datetime(1985, 6, 15),
            "examination_date": "2024-03-20",
            "examiner_name": None  # Test None handling
        }
        
        cli_instance.print_metadata(metadata)
        captured = capsys.readouterr()
        
        assert "Patient First Name: John" in captured.out
        assert "Patient Last Name: Doe" in captured.out
        assert "Patient Dob: 1985-06-15" in captured.out
        assert "Examination Date: 2024-03-20" in captured.out
        assert "Examiner Name" not in captured.out  # None values should be skipped
    
    def test_print_metadata_empty(self, cli_instance, capsys):
        """Test printing empty metadata."""
        cli_instance.print_metadata({})
        captured = capsys.readouterr()
        assert "No metadata extracted" in captured.out
    
    def test_print_processing_summary(self, cli_instance, capsys):
        """Test processing summary printing."""
        results = {
            "pdf_path": "/test/file.pdf",
            "original_text_length": 100,
            "anonymized_text_length": 95,
            "processing_timestamp": "2024-03-20T10:30:00",
            "use_ensemble": True,
            "use_llm_extractor": "deepseek",
            "metadata": {"patient_first_name": "John"}
        }
        
        cli_instance.print_processing_summary(results)
        captured = capsys.readouterr()
        
        assert "PROCESSING SUMMARY" in captured.out
        assert "/test/file.pdf" in captured.out
        assert "Original Text Length: 100" in captured.out
        assert "Anonymized Text Length: 95" in captured.out
        assert "OCR Method: Ensemble OCR" in captured.out
        assert "LLM Extractor: deepseek" in captured.out
    
    def test_print_batch_summary(self, cli_instance, capsys):
        """Test batch processing summary printing."""
        results = [
            {"pdf_path": "/test/file1.pdf", "success": True},
            {"pdf_path": "/test/file2.pdf", "success": True},
            {"pdf_path": "/test/file3.pdf", "error": "Processing failed"}
        ]
        
        cli_instance.print_batch_summary(results)
        captured = capsys.readouterr()
        
        assert "BATCH PROCESSING SUMMARY" in captured.out
        assert "Total Files: 3" in captured.out
        assert "Successful: 2" in captured.out
        assert "Failed: 1" in captured.out
        assert "file3.pdf: Processing failed" in captured.out


class TestCLIIntegration:
    """Integration tests for the CLI."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @patch('cli.report_reader.ReportReader')
    @patch('sys.argv')
    def test_main_process_command(self, mock_argv, mock_report_reader, temp_dir):
        """Test main function with process command."""
        # Create a test PDF file
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.touch()
        
        # Setup command line arguments
        mock_argv.__getitem__.side_effect = [
            "report_reader.py",
            "process",
            str(pdf_path),
            "--output-dir", temp_dir
        ]
        mock_argv.__len__.return_value = 4
        
        # Setup mock ReportReader
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            "Original", "Anonymized", {"test": "data"}
        )
        
        # Test main function
        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_not_called()  # Should not exit on success
    
    @patch('sys.argv')
    def test_main_no_command(self, mock_argv):
        """Test main function with no command."""
        mock_argv.__getitem__.side_effect = ["report_reader.py"]
        mock_argv.__len__.return_value = 1
        
        with patch('sys.exit') as mock_exit:
            with patch('builtins.print'):  # Suppress help output
                main()
            mock_exit.assert_called_with(1)
    
    @patch('sys.argv')
    def test_main_keyboard_interrupt(self, mock_argv):
        """Test main function handles keyboard interrupt."""
        mock_argv.__getitem__.side_effect = [
            "report_reader.py",
            "process", 
            "/nonexistent.pdf"
        ]
        mock_argv.__len__.return_value = 3
        
        with patch('cli.report_reader.ReportReaderCLI.process_single_pdf') as mock_process:
            mock_process.side_effect = KeyboardInterrupt()
            
            with patch('sys.exit') as mock_exit:
                main()
                mock_exit.assert_called_with(1)


class TestErrorHandling:
    """Test suite for error handling in the CLI."""
    
    @pytest.fixture
    def cli_instance(self):
        """Create a CLI instance for testing."""
        return ReportReaderCLI()
    
    @patch('cli.report_reader.ReportReader')
    def test_process_pdf_with_exception(self, mock_report_reader, cli_instance, temp_dir):
        """Test handling of exceptions during PDF processing."""
        # Create a test PDF file
        pdf_path = Path(temp_dir) / "test.pdf"
        pdf_path.touch()
        
        # Setup mock to raise exception
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.side_effect = Exception("Processing failed")
        
        # Test processing with exception
        result = cli_instance.process_single_pdf(
            pdf_path=str(pdf_path),
            verbose=False
        )
        
        # Verify error handling
        assert "error" in result
        assert "Processing failed" in result["error"]
        assert result["pdf_path"] == str(pdf_path)
        assert "processing_timestamp" in result
    
    @patch('cli.report_reader.ReportReader')
    def test_batch_process_continue_on_error(self, mock_report_reader, cli_instance, temp_dir):
        """Test batch processing continues on error when configured."""
        # Create test files
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir()
        output_dir = Path(temp_dir) / "output"
        
        for i in range(3):
            (input_dir / f"test_{i}.pdf").touch()
        
        # Setup mock to fail on second file
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        
        def side_effect(*args, **kwargs):
            if "test_1.pdf" in args[0]:
                raise Exception("Processing failed")
            return ("Original", "Anonymized", {"test": "data"})
        
        mock_reader_instance.process_report.side_effect = side_effect
        
        # Test batch processing
        results = cli_instance.batch_process(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            continue_on_error=True
        )
        
        # Verify results
        assert len(results) == 3
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        assert len(successful) == 2
        assert len(failed) == 1
        assert "Processing failed" in failed[0]["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])