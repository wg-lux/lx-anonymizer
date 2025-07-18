#!/usr/bin/env python3
"""
Simple functional test for the CLI to verify key functionality works.
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

# Add the project root to the Python path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

def test_cli_basic_functionality():
    """Test basic CLI functionality with mocked dependencies."""
    try:
        from cli.report_reader import ReportReaderCLI, create_parser
        print("✅ CLI module imported successfully")
        
        # Test parser creation
        parser = create_parser()
        print("✅ CLI parser created successfully")
        
        # Test CLI instance creation
        cli = ReportReaderCLI()
        print("✅ CLI instance created successfully")
        
        # Test argument parsing
        args = parser.parse_args(['process', '/test.pdf', '--ensemble', '--llm-extractor', 'deepseek'])
        assert args.command == 'process'
        assert args.pdf_path == '/test.pdf'
        assert args.ensemble is True
        assert args.llm_extractor == 'deepseek'
        print("✅ Argument parsing works correctly")
        
        # Test batch command parsing
        args = parser.parse_args(['batch', '/input', '--output-dir', '/output', '--max-files', '10'])
        assert args.command == 'batch'
        assert args.input_dir == '/input'
        assert args.output_dir == '/output'
        assert args.max_files == 10
        print("✅ Batch command parsing works correctly")
        
        # Test extract command parsing
        args = parser.parse_args(['extract', '/test.pdf', '--json-output'])
        assert args.command == 'extract'
        assert args.pdf_path == '/test.pdf'
        assert args.json_output is True
        print("✅ Extract command parsing works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_with_mock_data():
    """Test CLI with mocked ReportReader to verify processing logic."""
    try:
        from cli.report_reader import ReportReaderCLI
        
        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock PDF file
            pdf_path = Path(temp_dir) / "test.pdf"
            pdf_path.write_text("Mock PDF content")
            
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            cli = ReportReaderCLI()
            
            # Mock the ReportReader
            with patch('cli.report_reader.ReportReader') as mock_reader_class:
                mock_reader = Mock()
                mock_reader_class.return_value = mock_reader
                mock_reader.process_report.return_value = (
                    "Original text content",
                    "Anonymized text content",
                    {"patient_first_name": "Test", "patient_last_name": "Patient"}
                )
                
                # Test single PDF processing
                result = cli.process_single_pdf(
                    pdf_path=str(pdf_path),
                    output_dir=str(output_dir),
                    verbose=False
                )
                
                assert "error" not in result
                assert result["original_text_length"] > 0
                assert result["anonymized_text_length"] > 0
                assert result["metadata"]["patient_first_name"] == "Test"
                print("✅ Single PDF processing works correctly")
                
                # Verify output files were created
                expected_files = [
                    "test_metadata.json",
                    "test_anonymized.txt",
                    "test_original.txt",
                    "test_results.json"
                ]
                
                for filename in expected_files:
                    file_path = output_dir / filename
                    assert file_path.exists(), f"Expected file {filename} not created"
                    assert file_path.stat().st_size > 0, f"File {filename} is empty"
                
                print("✅ Output files created correctly")
                
                # Test metadata content
                with open(output_dir / "test_metadata.json", 'r') as f:
                    metadata = json.load(f)
                    assert metadata["patient_first_name"] == "Test"
                    print("✅ Metadata saved correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_error_handling():
    """Test CLI error handling with non-existent files."""
    try:
        from cli.report_reader import ReportReaderCLI
        
        cli = ReportReaderCLI()
        
        # Test with non-existent file
        result = cli.process_single_pdf(
            pdf_path="/nonexistent/file.pdf",
            verbose=False
        )
        
        assert "error" in result
        assert "PDF file not found" in result["error"]
        assert result["pdf_path"] == "/nonexistent/file.pdf"
        print("✅ Error handling works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all functional tests."""
    print("🧪 Testing CLI Functionality")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_cli_basic_functionality),
        ("Mock Data Processing", test_cli_with_mock_data),
        ("Error Handling", test_cli_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    print("\n" + "=" * 50)
    print(f"🏆 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CLI functionality tests PASSED!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)