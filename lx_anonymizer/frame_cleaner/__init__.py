"""Public video entry point; construct one cleaner per immutable-source attempt."""

from lx_anonymizer.processing_contracts import (
    VideoAnonymizationRequest,
    VideoAnonymizationResult,
)

from .cleaner import (
    FrameCleaner,
    FrameCleanerQualityProfile,
    FrameCleanerSamplingProfile,
    InvalidVideoInputError,
    VideoAnonymizationError,
    VideoOutputCollisionError,
)

__all__ = [
    "FrameCleaner",
    "FrameCleanerQualityProfile",
    "FrameCleanerSamplingProfile",
    "InvalidVideoInputError",
    "VideoAnonymizationError",
    "VideoAnonymizationRequest",
    "VideoAnonymizationResult",
    "VideoOutputCollisionError",
]
