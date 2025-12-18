"""
LX Anonymizer package for anonymizing medical reports and images.
"""

# Make all modules directly importable from this package
from .frame_cleaner import FrameCleaner
from .report_reader import ReportReader

# Import ollama_llm if available (requires ollama)
try:
    from .ollama import ollama_llm

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
