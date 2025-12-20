"""
LX Anonymizer package for anonymizing medical reports and images.
"""

# Make all modules directly importable from this package
from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.report_reader import ReportReader

# Import ollama_llm if available (requires ollama)
try:
    from lx_anonymizer.ollama import ollama_llm

    _ollama_available = True
except ImportError:
    _ollama_available = False
    ollama_llm = None

__all__ = [
    "ReportReader",
    "FrameCleaner",
]

if _ollama_available:
    __all__.append("ollama_llm")
