from __future__ import annotations

import stat
from fractions import Fraction
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from lx_dtypes.models.contracts.image_processing import ImageProcessingResultPayload
from lx_dtypes.models.meta.VideoMeta import VideoMeta
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VideoAnonymizationTechnique = Literal[
    "mask_overlay",
    "remove_frames",
    "extract_only",
]


class ProcessingContractError(ValueError):
    """A public processing request violates the local invocation contract."""


def _validate_regular_source(path: Path, *, kind: str) -> Path:
    if path.is_symlink():
        raise ProcessingContractError(
            f"{kind} source must not be a symbolic link: {path}"
        )
    try:
        source_stat = path.stat()
    except OSError as exc:
        raise ProcessingContractError(
            f"{kind} source cannot be inspected: {path}"
        ) from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise ProcessingContractError(f"{kind} source must be a regular file: {path}")
    if source_stat.st_size <= 0:
        raise ProcessingContractError(f"{kind} source must not be empty: {path}")
    return path.resolve()


class ImageAnonymizationRequest(BaseModel):
    """One immutable image/PDF invocation with a caller-selected output directory."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    source_path: Path
    output_directory: Path
    east_model_path: Path | None = None
    device: str = "olympus_cv_1500"
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    detector_width: int = Field(default=320, gt=0)
    detector_height: int = Field(default=320, gt=0)

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: Path) -> Path:
        source = _validate_regular_source(value, kind="image")
        if source.suffix.casefold() not in {
            ".bmp",
            ".jpeg",
            ".jpg",
            ".pdf",
            ".png",
            ".tif",
            ".tiff",
        }:
            raise ProcessingContractError(
                f"unsupported image/PDF source suffix: {source.suffix or '<none>'}"
            )
        return source

    @field_validator("output_directory")
    @classmethod
    def validate_output_directory(cls, value: Path) -> Path:
        if value.is_symlink():
            raise ProcessingContractError(
                f"image output directory must not be a symbolic link: {value}"
            )
        if not value.is_dir():
            raise ProcessingContractError(
                f"image output directory must already exist: {value}"
            )
        return value.resolve()

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ProcessingContractError("device must not be blank")
        return normalized

    @field_validator("detector_width", "detector_height")
    @classmethod
    def validate_detector_dimension(cls, value: int) -> int:
        if value % 32:
            raise ProcessingContractError("detector dimensions must be multiples of 32")
        return value


class ImageAnonymizationResult(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    source_path: Path
    artifact_path: Path
    metadata: ImageProcessingResultPayload


class VideoAnonymizationRequest(BaseModel):
    """Typed input for one attempt-scoped FrameCleaner invocation."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    source_path: Path
    output_path: Path
    source_frame_rate: Fraction
    endoscope_image_roi: dict[str, object] | None = None
    endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None = None
    technique: VideoAnonymizationTechnique = "mask_overlay"
    device: str | None = "olympus_cv_1500"

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: Path) -> Path:
        return _validate_regular_source(value, kind="video")

    @field_validator("source_frame_rate")
    @classmethod
    def validate_frame_rate(cls, value: Fraction) -> Fraction:
        if value <= 0:
            raise ProcessingContractError(
                "source_frame_rate must be a positive rational value"
            )
        return value

    @model_validator(mode="after")
    def validate_output_ownership(self) -> VideoAnonymizationRequest:
        if self.technique == "extract_only":
            return self
        if not self.output_path.suffix:
            raise ProcessingContractError(
                "video output_path must retain a media suffix"
            )
        if self.output_path.absolute() == self.source_path.absolute():
            raise ProcessingContractError(
                "video output_path must differ from the immutable source"
            )
        if self.output_path.exists() or self.output_path.is_symlink():
            raise ProcessingContractError(
                f"video output_path already exists: {self.output_path}"
            )
        return self


class VideoAnonymizationResult(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    source_path: Path
    artifact_path: Path | None
    metadata: VideoMeta


RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)


class ProcessingStrand(Protocol[RequestT_contra, ResultT_co]):
    """Common user-facing shape implemented by every central strand."""

    def process(self, request: RequestT_contra) -> ResultT_co: ...


__all__ = [
    "ImageAnonymizationRequest",
    "ImageAnonymizationResult",
    "ProcessingContractError",
    "ProcessingStrand",
    "VideoAnonymizationRequest",
    "VideoAnonymizationResult",
    "VideoAnonymizationTechnique",
]
