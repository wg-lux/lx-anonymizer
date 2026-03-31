"""
LX Anonymizer package for anonymizing medical reports and images.
"""

import importlib.metadata

__all__ = ["ReportReader", "FrameCleaner", "__version__"]

try:
    __version__ = importlib.metadata.version("lx-anonymizer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


def __getattr__(name: str):
    if name == "FrameCleaner":
        from lx_anonymizer.frame_cleaner import FrameCleaner

        return FrameCleaner
    if name == "ReportReader":
        from lx_anonymizer.report_reader import ReportReader

        return ReportReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
