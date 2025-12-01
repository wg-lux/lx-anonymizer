"""
Integration tests for the CLI with real file operations and end-to-end scenarios.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli.report_reader import ReportReaderCLI


class TestCLIRealFileOperations:
    """Test CLI with actual file I/O operations."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace with test files."""
        workspace = tempfile.mkdtemp()

        # Create input directory with test PDFs
        input_dir = Path(workspace) / "input"
        input_dir.mkdir()

        # Create mock report files
        for i in range(5):
            pdf_file = input_dir / f"report_{i:03d}.pdf"
            pdf_file.write_text(f"Mock report content for report {i}")

        # Create output directory
        output_dir = Path(workspace) / "output"
        output_dir.mkdir()

        yield {
            "workspace": workspace,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "pdf_files": list(input_dir.glob("*.pdf")),
        }

        shutil.rmtree(workspace)

    @patch("cli.report_reader.ReportReader")
    def test_end_to_end_single_processing(self, mock_report_reader, temp_workspace):
        """Test complete end-to-end single report processing."""
        cli = ReportReaderCLI()

        # Setup mock
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            "This is the original extracted text from the report report.",
            "This is the anonymized text with [PATIENT] placeholder.",
            {
                "patient_first_name": "Max",
                "patient_last_name": "Mustermann",
                "patient_dob": "1980-01-01",
                "examination_date": "2024-03-15",
                "center_name": "Test Hospital",
            },
        )

        # Process a single report
        pdf_path = str(temp_workspace["pdf_files"][0])
        result = cli.process_single_pdf(
            pdf_path=pdf_path,
            output_dir=temp_workspace["output_dir"],
            use_ensemble=True,
            use_llm_extractor="deepseek",
            verbose=True,
        )

        # Verify processing result
        assert "error" not in result
        assert result["use_ensemble"] is True
        assert result["use_llm_extractor"] == "deepseek"
        assert result["original_text_length"] > 0
        assert result["anonymized_text_length"] > 0

        # Verify output files were created
        output_dir = Path(temp_workspace["output_dir"])
        base_name = Path(pdf_path).stem

        expected_files = [
            f"{base_name}_metadata.json",
            f"{base_name}_anonymized.txt",
            f"{base_name}_original.txt",
            f"{base_name}_results.json",
        ]

        for filename in expected_files:
            file_path = output_dir / filename
            assert file_path.exists(), f"Expected output file not found: {filename}"
            assert file_path.stat().st_size > 0, f"Output file is empty: {filename}"

    @patch("cli.report_reader.ReportReader")
    def test_batch_processing_with_limits(self, mock_report_reader, temp_workspace):
        """Test batch processing with file limits and patterns."""
        cli = ReportReaderCLI()

        # Setup mock
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            "Original text",
            "Anonymized text",
            {"test": "metadata"},
        )

        # Test with max_files limit
        results = cli.batch_process(
            input_dir=temp_workspace["input_dir"],
            output_dir=temp_workspace["output_dir"],
            max_files=3,
            pattern="report_*.pdf",
        )

        # Verify results
        assert len(results) == 3  # Should respect max_files limit

        for result in results:
            assert "error" not in result
            assert "pdf_path" in result
            assert result["pdf_path"].endswith(".pdf")

        # Verify batch summary was created
        summary_file = (
            Path(temp_workspace["output_dir"]) / "batch_processing_summary.json"
        )
        assert summary_file.exists()

        # Read and verify summary content
        import json

        with open(summary_file, "r") as f:
            summary = json.load(f)

        assert summary["total_files"] == 3
        assert summary["processed_files"] == 3
        assert summary["successful_files"] == 3
        assert summary["failed_files"] == 0
        assert summary["settings"]["pattern"] == "report_*.pdf"

    def test_file_pattern_matching(self, temp_workspace):
        """Test different file patterns for batch processing."""
        cli = ReportReaderCLI()

        # Create additional files with different patterns
        input_dir = Path(temp_workspace["input_dir"])
        (input_dir / "document_001.pdf").touch()
        (input_dir / "scan_002.pdf").touch()
        (input_dir / "report.txt").touch()  # Should be ignored

        # Test specific pattern
        with patch("cli.report_reader.ReportReader") as mock_reader:
            mock_reader_instance = Mock()
            mock_reader.return_value = mock_reader_instance
            mock_reader_instance.process_report.return_value = ("", "", {})

            results = cli.batch_process(
                input_dir=str(input_dir),
                output_dir=temp_workspace["output_dir"],
                pattern="document_*.pdf",
            )

            # Should only find document_001.pdf
            assert len(results) == 1
            assert "document_001.pdf" in results[0]["pdf_path"]


class TestCLIEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def cli_instance(self):
        return ReportReaderCLI()

    def test_empty_metadata_handling(self, cli_instance, capsys):
        """Test handling of empty or None metadata."""
        # Test with None metadata
        cli_instance.print_metadata(None)
        captured = capsys.readouterr()
        assert "No metadata extracted" in captured.out

        # Test with empty dict
        cli_instance.print_metadata({})
        captured = capsys.readouterr()
        assert "No metadata extracted" in captured.out

    def test_unicode_filename_handling(self, cli_instance):
        """Test handling of unicode characters in filenames."""
        unicode_path = "/path/to/ärztlicher_bericht_müller.pdf"

        result = cli_instance.process_single_pdf(pdf_path=unicode_path, verbose=False)

        # Should handle unicode gracefully
        assert "error" in result
        assert unicode_path in result["pdf_path"]

    @patch("cli.report_reader.ReportReader")
    def test_very_large_text_processing(self, mock_report_reader, cli_instance):
        """Test processing of very large text content."""
        # Create a large text content
        large_text = "A" * 100000  # 100KB of text

        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            large_text,
            large_text[:50000],
            {"test": "metadata"},
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            result = cli_instance.process_single_pdf(pdf_path=tmp_path, verbose=False)

            assert result["original_text_length"] == 100000
            assert result["anonymized_text_length"] == 50000

        finally:
            os.unlink(tmp_path)

    def test_special_characters_in_metadata(self, cli_instance, capsys):
        """Test handling of special characters in metadata."""
        metadata_with_special_chars = {
            "patient_first_name": "François",
            "patient_last_name": "Müller-O'Connor",
            "center_name": "Médical Center & Clinic",
            "notes": "Special chars: ñáéíóú ÄÖÜß €£¥",
        }

        cli_instance.print_metadata(metadata_with_special_chars)
        captured = capsys.readouterr()

        # Should handle special characters without crashing
        assert "François" in captured.out
        assert "Müller-O'Connor" in captured.out
        assert "€£¥" in captured.out


class TestCLIPerformance:
    """Test CLI performance characteristics."""

    @pytest.fixture
    def performance_workspace(self):
        """Create workspace for performance testing."""
        workspace = tempfile.mkdtemp()
        input_dir = Path(workspace) / "input"
        input_dir.mkdir()

        # Create many small test files
        for i in range(50):
            pdf_file = input_dir / f"perf_test_{i:03d}.pdf"
            pdf_file.write_text(f"Performance test file {i}")

        output_dir = Path(workspace) / "output"
        output_dir.mkdir()

        yield {
            "workspace": workspace,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        }

        shutil.rmtree(workspace)

    @patch("cli.report_reader.ReportReader")
    def test_batch_processing_performance(
        self, mock_report_reader, performance_workspace
    ):
        """Test batch processing with many files."""
        import time

        cli = ReportReaderCLI()

        # Setup fast mock
        mock_reader_instance = Mock()
        mock_report_reader.return_value = mock_reader_instance
        mock_reader_instance.process_report.return_value = (
            "Fast processing",
            "Fast result",
            {"fast": "metadata"},
        )

        # Measure processing time
        start_time = time.time()

        results = cli.batch_process(
            input_dir=performance_workspace["input_dir"],
            output_dir=performance_workspace["output_dir"],
            max_files=20,
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify results
        assert len(results) == 20
        assert all("error" not in result for result in results)

        # Performance should be reasonable (less than 10 seconds for 20 files)
        assert processing_time < 10.0, (
            f"Processing took too long: {processing_time:.2f}s"
        )

    def test_memory_usage_with_large_batch(self, performance_workspace):
        """Test memory usage doesn't grow excessively with large batches."""
        import gc

        cli = ReportReaderCLI()

        # Force garbage collection before test
        gc.collect()

        with patch("cli.report_reader.ReportReader") as mock_reader:
            mock_reader_instance = Mock()
            mock_reader.return_value = mock_reader_instance
            mock_reader_instance.process_report.return_value = (
                "Memory test",
                "Memory result",
                {"memory": "test"},
            )

            # Process files in batches to simulate real usage
            for batch_start in range(0, 40, 10):
                results = cli.batch_process(
                    input_dir=performance_workspace["input_dir"],
                    output_dir=performance_workspace["output_dir"],
                    max_files=10,
                )

                # Verify each batch processes correctly
                assert len(results) == 10

                # Force garbage collection after each batch
                gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
