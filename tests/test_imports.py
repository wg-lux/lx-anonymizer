"""
Tests for verifying imports from the lx_anonymizer package.
"""
import pytest
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_package_imports():
    """Test that all main components can be imported from the package."""
    try:
        from lx_anonymizer.lx_anonymizer.report_reader import ReportReader
        from lx_anonymizer.lx_anonymizer.spacy_extractor import (
            PatientDataExtractor, ExaminerDataExtractor, 
            EndoscopeDataExtractor, ExaminationDataExtractor
        )
        from lx_anonymizer.lx_anonymizer.text_anonymizer import anonymize_text
        from lx_anonymizer.lx_anonymizer.determine_gender import determine_gender
        import_success = True
    except ImportError as e:
        print(f"Import error: {e}")
        import_success = False
    
    assert import_success, "Failed to import required modules"

def test_shortcut_imports():
    """Test the shortcut imports from the top-level package."""
    try:
        from lx_anonymizer import (
            ReportReader, PatientDataExtractor, ExaminerDataExtractor, 
            EndoscopeDataExtractor, ExaminationDataExtractor
        )
        import_success = True
    except ImportError as e:
        print(f"Shortcut import error: {e}")
        import_success = False
    
    assert import_success, "Failed to import through shortcuts"
