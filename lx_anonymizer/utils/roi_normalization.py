from typing import Any, Mapping, TypedDict

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)


class NormalizedRoi(TypedDict):
    x: int
    y: int
    width: int
    height: int


class _NormalizedRoi(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x: int
    y: int
    width: int
    height: int

    @field_validator("x", "y", "width", "height", mode="before")
    @classmethod
    def _coerce_int_like(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid ROI integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return int(stripped)
                except ValueError as exc:
                    raise ValueError("string is not an integer") from exc
        raise ValueError("value is not an integer")

    @field_validator("x", "y", "width", "height")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("ROI values must be non-negative")
        return value


def _first_present(roi: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in roi and roi[key] is not None:
            return roi[key]
    return None


def normalize_roi_keys(roi: Mapping[str, Any] | None) -> NormalizedRoi | None:
    """
    Normalize ROI key naming conventions.

    Converts mixed field names (e.g., endoscope_image_x) to a strict ROI form:
    {"x": int, "y": int, "width": int, "height": int}.
    """
    if not roi or not isinstance(roi, Mapping):
        logger.warning("ROI normalization skipped: input is not a mapping.")
        return None

    payload = {
        "x": _first_present(
            roi,
            ("x", "endoscope_image_x", "endoscope_type_x", "patient_first_name_x"),
        ),
        "y": _first_present(
            roi,
            ("y", "endoscope_image_y", "endoscope_type_y", "patient_first_name_y"),
        ),
        "width": _first_present(
            roi,
            (
                "width",
                "endoscope_image_width",
                "endoscope_type_width",
                "patient_first_name_width",
            ),
        ),
        "height": _first_present(
            roi,
            (
                "height",
                "endoscope_image_height",
                "endoscope_type_height",
                "patient_first_name_height",
            ),
        ),
    }

    if any(value is None for value in payload.values()):
        logger.warning(
            "ROI normalization failed: missing required ROI keys: %s", payload
        )
        return None

    try:
        normalized = _NormalizedRoi.model_validate(payload)
    except ValidationError as exc:
        logger.warning("ROI normalization failed validation: %s", exc)
        return None

    return {
        "x": normalized.x,
        "y": normalized.y,
        "width": normalized.width,
        "height": normalized.height,
    }
