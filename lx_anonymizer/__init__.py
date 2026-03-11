"""
LX Anonymizer package for anonymizing medical reports and images.
"""

from importlib import import_module

__all__ = ["ReportReader", "FrameCleaner", "ollama_llm"]


def __getattr__(name: str):
    if name == "FrameCleaner":
        from lx_anonymizer.frame_cleaner import FrameCleaner

        return FrameCleaner
    if name == "ReportReader":
        from lx_anonymizer.report_reader import ReportReader

        return ReportReader
    if name == "ollama_llm":
        return import_module("lx_anonymizer.ollama.ollama_llm")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
