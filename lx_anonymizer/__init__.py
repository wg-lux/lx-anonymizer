"""
LX Anonymizer package for anonymizing medical reports and images.
"""

# Make all modules directly importable from this package
from .report_reader import ReportReader
from .spacy_extractor import PatientDataExtractor, ExaminerDataExtractor, EndoscopeDataExtractor, ExaminationDataExtractor
from .text_anonymizer import anonymize_text
from .frame_cleaner import FrameCleaner

__all__ = [
    'ReportReader',
    'PatientDataExtractor',
    'ExaminerDataExtractor', 
    'EndoscopeDataExtractor',
    'ExaminationDataExtractor',
    'anonymize_text',
    'FrameCleaner',
    'MetaExtraction'
]
