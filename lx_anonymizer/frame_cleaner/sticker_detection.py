"""Colour-based detection of paper stickers inside the endoscope image area.

Patient-data stickers held into the endoscope view are bright and nearly
unsaturated, while mucosa is strongly saturated red/pink. Specular highlights
are also white but small and scattered, so the score is the largest connected
white region rather than a raw pixel count.

Overexposure glare and pale NBI tissue can form large white regions too, but
they are smooth. A sticker additionally needs dark print enclosed by the paper.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from lx_anonymizer.utils.roi_normalization import NormalizedRoi

WHITE_MIN_VALUE = 180
WHITE_MAX_SATURATION = 60
STICKER_MIN_ROI_FRACTION = 0.05
# Print is darker than this share of the paper's median brightness.
PRINT_MAX_RELATIVE_VALUE = 0.6
STICKER_MIN_PRINT_FRACTION = 0.01
_SPECULAR_KERNEL = np.ones((5, 5), dtype=np.uint8)


@dataclass(frozen=True)
class StickerScore:
    largest_region_pixels: int
    print_pixels: int
    roi_pixels: int

    def __post_init__(self) -> None:
        if self.roi_pixels <= 0:
            raise ValueError(f"ROI must contain pixels, got {self.roi_pixels}")
        if self.largest_region_pixels < 0 or self.print_pixels < 0:
            raise ValueError("Region and print pixel counts must be non-negative")
        if self.largest_region_pixels + self.print_pixels > self.roi_pixels:
            raise ValueError(
                f"Region of {self.largest_region_pixels} + {self.print_pixels} "
                f"pixels does not fit a {self.roi_pixels}-pixel ROI"
            )

    @property
    def region_fraction(self) -> float:
        return self.largest_region_pixels / self.roi_pixels

    @property
    def print_fraction(self) -> float:
        """Share of the paper outline covered by enclosed dark print."""
        outline = self.largest_region_pixels + self.print_pixels
        return self.print_pixels / outline if outline else 0.0

    @property
    def detected(self) -> bool:
        return (
            self.region_fraction >= STICKER_MIN_ROI_FRACTION
            and self.print_fraction >= STICKER_MIN_PRINT_FRACTION
        )

    @property
    def ranking_pixels(self) -> int:
        """Paper area of a detected sticker; 0 so glare never outranks a sticker."""
        return self.largest_region_pixels if self.detected else 0


def _validate_frame(frame: np.ndarray) -> None:
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got {frame.dtype}")
    if not (frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 3)):
        raise ValueError(f"Expected (H, W) or (H, W, 3) frame, got {frame.shape}")


def white_mask(frame: np.ndarray) -> np.ndarray:
    """Bright, low-saturation pixels of a BGR or grayscale uint8 image."""
    _validate_frame(frame)
    if frame.ndim == 2:
        # Grayscale carries no saturation; brightness is the only signal.
        return (frame >= WHITE_MIN_VALUE).astype(np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return (
        (hsv[..., 2] >= WHITE_MIN_VALUE) & (hsv[..., 1] <= WHITE_MAX_SATURATION)
    ).astype(np.uint8)


def score_sticker(frame: np.ndarray, roi: NormalizedRoi) -> StickerScore:
    """Score the largest white region inside ``roi`` (clipped to the frame)."""
    _validate_frame(frame)
    if roi["width"] <= 0 or roi["height"] <= 0:
        raise ValueError(f"ROI must have positive width and height: {roi}")
    if roi["x"] < 0 or roi["y"] < 0:
        raise ValueError(f"ROI origin must be non-negative: {roi}")
    region = frame[
        roi["y"] : roi["y"] + roi["height"],
        roi["x"] : roi["x"] + roi["width"],
    ]
    if region.size == 0:
        raise ValueError(
            f"Endoscope image ROI {roi} lies outside the {frame.shape[:2]} frame"
        )

    roi_pixels = int(region.shape[0] * region.shape[1])
    mask = cv2.morphologyEx(white_mask(region), cv2.MORPH_OPEN, _SPECULAR_KERNEL)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:  # Label 0 is the background.
        return StickerScore(
            largest_region_pixels=0, print_pixels=0, roi_pixels=roi_pixels
        )
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    paper = (labels == label).astype(np.uint8)
    return StickerScore(
        largest_region_pixels=int(stats[label, cv2.CC_STAT_AREA]),
        print_pixels=_enclosed_print_pixels(region, paper),
        roi_pixels=roi_pixels,
    )


def _enclosed_print_pixels(region: np.ndarray, paper: np.ndarray) -> int:
    """Count pixels inside the paper outline that are much darker than the paper."""
    contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = np.zeros_like(paper)
    cv2.drawContours(outline, contours, -1, 1, thickness=cv2.FILLED)
    brightness = region if region.ndim == 2 else region.max(axis=2)
    paper_value = float(np.median(brightness[paper.astype(bool)]))
    enclosed = outline.astype(bool) & ~paper.astype(bool)
    return int(
        np.count_nonzero(
            enclosed & (brightness < PRINT_MAX_RELATIVE_VALUE * paper_value)
        )
    )
