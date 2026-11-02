"""Colour screening that decides when a decoded frame needs fresh full-frame OCR.

Every frame is screened; only screening decides whether the previous OCR result
is reused. Reuse requires positive evidence that nothing text-bearing changed:
the endoscope image shows mucosa (inside the body), no patient-data sticker is
visible, the overlay outside the endoscope image is unchanged against the last
OCR'd frame, and the anchor interval has not elapsed. Every uncertain or
unusual frame (outside the body, NBI or dark frames, stickers, overlay changes)
receives OCR. Misclassification therefore costs inference, never coverage.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from lx_anonymizer.utils.roi_normalization import NormalizedRoi

# OpenCV hue spans 0..179; mucosa is red/pink and wraps around 0.
MUCOSA_RED_HUE_MAX = 20
MUCOSA_RED_HUE_MIN_WRAP = 150
MUCOSA_MIN_SATURATION = 70
MUCOSA_MIN_VALUE = 40
INSIDE_MIN_MUCOSA_FRACTION = 0.4
# A changed overlay character spans far more than this many strongly changed
# pixels; compression noise on a static overlay stays well below the delta.
OVERLAY_PIXEL_DELTA = 40
OVERLAY_MAX_CHANGED_PIXELS = 16


class OcrScreenReason(str, Enum):
    """Why a frame received OCR, or that it reused the previous result."""

    STICKER = "sticker"
    OUTSIDE_BODY = "outside_body"
    NO_REFERENCE = "no_reference"
    ANCHOR_INTERVAL = "anchor_interval"
    OVERLAY_CHANGED = "overlay_changed"
    REUSED = "reused"


@dataclass(frozen=True)
class ScreenReference:
    """Overlay pixels of the last OCR'd clean inside-body frame."""

    frame_shape: tuple[int, ...]
    overlay_pixels: np.ndarray
    frames_since_ocr: int

    def __post_init__(self) -> None:
        if self.frames_since_ocr < 0:
            raise ValueError("Frames since OCR must be non-negative")


@dataclass(frozen=True)
class ScreenDecision:
    reason: OcrScreenReason
    clean_inside: bool
    overlay_pixels: np.ndarray

    @property
    def requires_ocr(self) -> bool:
        return self.reason is not OcrScreenReason.REUSED


def _image_region(frame: np.ndarray, roi: NormalizedRoi) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Colour screening requires uint8 BGR frames, got {frame.shape}"
        )
    if roi["width"] <= 0 or roi["height"] <= 0 or roi["x"] < 0 or roi["y"] < 0:
        raise ValueError(f"Screening ROI must be non-negative and non-empty: {roi}")
    region = frame[
        roi["y"] : roi["y"] + roi["height"],
        roi["x"] : roi["x"] + roi["width"],
    ]
    if region.size == 0:
        raise ValueError(
            f"Screening ROI {roi} lies outside the {frame.shape[:2]} frame"
        )
    return region


def mucosa_fraction(frame: np.ndarray, roi: NormalizedRoi) -> float:
    """Share of saturated red/pink pixels inside the endoscope image area."""
    hsv = cv2.cvtColor(_image_region(frame, roi), cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mucosa = (
        ((hue <= MUCOSA_RED_HUE_MAX) | (hue >= MUCOSA_RED_HUE_MIN_WRAP))
        & (saturation >= MUCOSA_MIN_SATURATION)
        & (value >= MUCOSA_MIN_VALUE)
    )
    return float(np.count_nonzero(mucosa)) / float(mucosa.size)


def is_inside_body(frame: np.ndarray, roi: NormalizedRoi) -> bool:
    return mucosa_fraction(frame, roi) >= INSIDE_MIN_MUCOSA_FRACTION


def overlay_pixels(frame: np.ndarray, roi: NormalizedRoi) -> np.ndarray:
    """Copy of every pixel outside the endoscope image area."""
    _image_region(frame, roi)
    mask = np.ones(frame.shape[:2], dtype=bool)
    mask[roi["y"] : roi["y"] + roi["height"], roi["x"] : roi["x"] + roi["width"]] = (
        False
    )
    return frame[mask]


def overlay_changed(reference: np.ndarray, current: np.ndarray) -> bool:
    if reference.shape != current.shape:
        return True
    delta = np.abs(reference.astype(np.int16) - current.astype(np.int16)).max(axis=-1)
    return (
        int(np.count_nonzero(delta > OVERLAY_PIXEL_DELTA)) > OVERLAY_MAX_CHANGED_PIXELS
    )


def screen_frame(
    frame: np.ndarray,
    roi: NormalizedRoi,
    *,
    sticker_detected: bool,
    reference: ScreenReference | None,
    max_reuse_frames: int,
) -> ScreenDecision:
    """Decide whether ``frame`` may reuse the OCR result of ``reference``."""
    if max_reuse_frames <= 0:
        raise ValueError("Maximum OCR reuse frames must be positive")
    current_overlay = overlay_pixels(frame, roi)
    inside = is_inside_body(frame, roi)
    clean_inside = inside and not sticker_detected
    if sticker_detected:
        reason = OcrScreenReason.STICKER
    elif not inside:
        reason = OcrScreenReason.OUTSIDE_BODY
    elif reference is None:
        reason = OcrScreenReason.NO_REFERENCE
    elif reference.frames_since_ocr + 1 >= max_reuse_frames:
        reason = OcrScreenReason.ANCHOR_INTERVAL
    elif reference.frame_shape != frame.shape or overlay_changed(
        reference.overlay_pixels, current_overlay
    ):
        reason = OcrScreenReason.OVERLAY_CHANGED
    else:
        reason = OcrScreenReason.REUSED
    return ScreenDecision(
        reason=reason, clean_inside=clean_inside, overlay_pixels=current_overlay
    )


def emit_screening_disabled(logger: logging.Logger) -> None:
    """Named fail-closed path: without an endoscope image ROI, OCR every frame."""
    logger.warning(
        json.dumps(
            {
                "event": "frame_analysis.ocr_screening_disabled",
                "reason": "missing_endoscope_image_roi",
                "fallback": "ocr_every_frame",
            }
        )
    )
