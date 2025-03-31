"""
LX Anonymizer - A package for anonymizing medical reports and images.
"""

# Re-export main components for easier imports
from lx_anonymizer.lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.lx_anonymizer.spacy_extractor import (
    PatientDataExtractor, 
    ExaminerDataExtractor, 
    EndoscopeDataExtractor,
    ExaminationDataExtractor
)

__all__ = [
    'ReportReader',
    'PatientDataExtractor',
    'ExaminerDataExtractor', 
    'EndoscopeDataExtractor',
    'ExaminationDataExtractor'
]
