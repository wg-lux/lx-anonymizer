"""
LX Anonymizer package for anonymizing medical reports and images.
"""

# Make all modules directly importable from this package
from .frame_cleaner import FrameCleaner
from .report_reader import ReportReader
from .text_anonymizer import anonymize_text

# Import ollama_llm if available (requires ollama)
try:
    from . import ollama_llm

    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    ollama_llm = None

__all__ = [
    "ReportReader",
    "anonymize_text",
    "FrameCleaner",
]

if _OLLAMA_AVAILABLE:
    __all__.append("ollama_llm")
