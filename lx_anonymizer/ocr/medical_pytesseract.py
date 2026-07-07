from __future__ import annotations

import argparse
import json
import logging
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Protocol, SupportsFloat, TypeAlias, cast

import cv2
import numpy as np
import numpy.typing as npt
import pytesseract  # type: ignore[import-untyped]
from PIL import Image

logger = logging.getLogger(__name__)

ImageArray: TypeAlias = npt.NDArray[np.uint8]
GrayArray: TypeAlias = npt.NDArray[np.uint8]
ImageInput: TypeAlias = str | Path | Image.Image | ImageArray
TesseractData: TypeAlias = Mapping[str, Sequence[object]]
Box: TypeAlias = tuple[int, int, int, int]

_ALLOWED_OCR_NOISE_RE = re.compile(r"[^0-9A-Za-zÄÖÜäöüßÀ-ÿ\s.,:;/()+\-\[\]#%]")
_SINGLE_CHAR_RUN_RE = re.compile(
    r"(?<!\S)(?:[0-9A-Za-zÄÖÜäöüßÀ-ÿ]\s+){2,}[0-9A-Za-zÄÖÜäöüßÀ-ÿ](?!\S)"
)
_DIGIT_CHAR_RUN_RE = re.compile(r"(?<!\S)(?:\d\s+)+\d(?!\S)")
_LOWER_TO_UPPER_RE = re.compile(r"(?<=[a-zäöüß])(?=[A-ZÄÖÜ])")
_STRUCTURED_MEDICAL_RE = re.compile(
    r"\b(?:"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{1,2}:\d{2}(?::\d{2})?|"
    r"\d{3,}/\d{2,4}|"
    r"[A-ZÄÖÜ]{1,5}\s?\d{2,}"
    r")\b"
)


class MedicalDocumentType(str, Enum):
    VIDEO = "video"
    REPORT = "report"


class RoiPolicy(str, Enum):
    FALLBACK_TO_FULL_IMAGE = "fallback_to_full_image"
    RAISE = "raise"


class TesseractOem(IntEnum):
    LSTM_ONLY = 1


class TesseractPsm(IntEnum):
    SINGLE_BLOCK = 6
    SINGLE_LINE = 7
    SPARSE_TEXT = 11


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_box(self) -> Box:
        return (self.x, self.y, self.right, self.bottom)

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class OcrRuntimeConfig:
    language: str = "deu+eng"
    oem: TesseractOem = TesseractOem.LSTM_ONLY
    report_psm: TesseractPsm = TesseractPsm.SINGLE_BLOCK
    video_roi_psm: TesseractPsm = TesseractPsm.SINGLE_LINE
    video_full_frame_psm: TesseractPsm = TesseractPsm.SPARSE_TEXT
    report_dpi: int = 300
    video_dpi: int = 400
    min_word_confidence: float = 0.0
    disable_dictionary_dawgs: bool = True


@dataclass(frozen=True)
class TesseractInvocation:
    config: str
    psm: TesseractPsm
    oem: TesseractOem
    dpi: int


@dataclass(frozen=True)
class PreprocessedImage:
    image: Image.Image
    document_type: MedicalDocumentType
    preprocessing: str
    thresholding: str
    scale: float
    deskew_angle: float
    inverted: bool
    roi_used: Rect | None
    roi_fallback_used: bool
    roi_fallback_reason: str | None


@dataclass(frozen=True)
class OcrWord:
    text: str
    confidence: float
    box: Box
    line_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "box": list(self.box),
            "line_index": self.line_index,
        }


@dataclass(frozen=True)
class MedicalOcrResult:
    text: str
    raw_text: str
    confidence: float
    words: tuple[OcrWord, ...]
    document_type: MedicalDocumentType
    tesseract_config: str
    psm: TesseractPsm
    oem: TesseractOem
    dpi: int
    preprocessing: str
    thresholding: str
    scale: float
    deskew_angle: float
    inverted: bool
    roi_used: Rect | None
    roi_fallback_used: bool
    roi_fallback_reason: str | None
    processing_time_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "words": [word.as_dict() for word in self.words],
            "document_type": self.document_type.value,
            "tesseract_config": self.tesseract_config,
            "psm": int(self.psm),
            "oem": int(self.oem),
            "dpi": self.dpi,
            "preprocessing": self.preprocessing,
            "thresholding": self.thresholding,
            "scale": self.scale,
            "deskew_angle": self.deskew_angle,
            "inverted": self.inverted,
            "roi_used": self.roi_used.as_dict() if self.roi_used else None,
            "roi_fallback_used": self.roi_fallback_used,
            "roi_fallback_reason": self.roi_fallback_reason,
            "processing_time_ms": self.processing_time_ms,
        }


class ImageToDataFn(Protocol):
    def __call__(
        self, image: Image.Image, *, lang: str, config: str
    ) -> TesseractData: ...


class _TesseractOutput(Protocol):
    DICT: object


class _TesseractModule(Protocol):
    Output: _TesseractOutput

    def image_to_data(
        self,
        image: Image.Image,
        *,
        lang: str,
        config: str,
        output_type: object,
    ) -> TesseractData: ...


class _AdaptiveThresholdFn(Protocol):
    def __call__(
        self,
        src: GrayArray,
        maxValue: float,
        adaptiveMethod: int,
        thresholdType: int,
        blockSize: int,
        C: float,
    ) -> GrayArray: ...


class _MinAreaRectFn(Protocol):
    def __call__(
        self, points: npt.NDArray[np.float32]
    ) -> tuple[tuple[float, float], tuple[float, float], float]: ...


class _GetRotationMatrix2DFn(Protocol):
    def __call__(
        self, center: tuple[float, float], angle: float, scale: float
    ) -> npt.NDArray[np.float64]: ...


_PYTESSERACT = cast(_TesseractModule, pytesseract)


def _image_to_data(image: Image.Image, *, lang: str, config: str) -> TesseractData:
    return _PYTESSERACT.image_to_data(
        image,
        lang=lang,
        config=config,
        output_type=_PYTESSERACT.Output.DICT,
    )


def build_tesseract_invocation(
    document_type: MedicalDocumentType | str,
    *,
    has_roi: bool,
    runtime_config: OcrRuntimeConfig | None = None,
) -> TesseractInvocation:
    """
    Build explicit Tesseract flags for hospital video overlays and reports.

    OEM is pinned to LSTM-only because legacy Tesseract models tend to hallucinate
    on noisy overlays. PSM is selected by layout: sparse full frames, single-line
    overlay ROIs, and single-block report text.
    """
    resolved_document_type = _coerce_document_type(document_type)
    cfg = runtime_config or OcrRuntimeConfig()

    if resolved_document_type is MedicalDocumentType.VIDEO:
        psm = cfg.video_roi_psm if has_roi else cfg.video_full_frame_psm
        dpi = cfg.video_dpi
    else:
        psm = cfg.report_psm
        dpi = cfg.report_dpi

    flags = [
        f"--oem {int(cfg.oem)}",
        f"--psm {int(psm)}",
        "-c preserve_interword_spaces=1",
        f"-c user_defined_dpi={dpi}",
        "-c tessedit_do_invert=0",
        "-c classify_enable_learning=0",
    ]

    if cfg.disable_dictionary_dawgs:
        flags.extend(
            [
                "-c load_system_dawg=0",
                "-c load_freq_dawg=0",
                "-c tessedit_enable_doc_dict=0",
            ]
        )

    if resolved_document_type is MedicalDocumentType.VIDEO:
        # Video overlays are mostly structured identifiers and timestamps.
        # The whitelist cuts OCR noise without constraining report diagnoses.
        flags.append(
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "ÄÖÜäöüß0123456789.,:;/-()[]"
        )

    return TesseractInvocation(
        config=" ".join(flags),
        psm=psm,
        oem=cfg.oem,
        dpi=dpi,
    )


def preprocess_for_medical_ocr(
    image_input: ImageInput,
    document_type: MedicalDocumentType | str,
    *,
    roi: Mapping[str, object] | None = None,
    roi_policy: RoiPolicy = RoiPolicy.FALLBACK_TO_FULL_IMAGE,
    deskew_reports: bool = True,
) -> PreprocessedImage:
    source = _load_image(image_input)
    crop, resolved_roi, fallback_used, fallback_reason = _crop_or_full(
        source,
        roi=roi,
        roi_policy=roi_policy,
    )
    gray = _to_grayscale(crop)
    resolved_document_type = _coerce_document_type(document_type)

    if resolved_document_type is MedicalDocumentType.VIDEO:
        processed, scale, inverted = _preprocess_video_overlay(
            gray,
            has_roi=resolved_roi is not None,
        )
        return PreprocessedImage(
            image=Image.fromarray(processed),
            document_type=resolved_document_type,
            preprocessing="video_overlay_bilateral_clahe_otsu_close",
            thresholding="otsu_binary",
            scale=scale,
            deskew_angle=0.0,
            inverted=inverted,
            roi_used=resolved_roi,
            roi_fallback_used=fallback_used,
            roi_fallback_reason=fallback_reason,
        )

    processed_report, report_scale, deskew_angle = _preprocess_report_page(
        gray,
        has_roi=resolved_roi is not None,
        deskew=deskew_reports,
    )
    return PreprocessedImage(
        image=Image.fromarray(processed_report),
        document_type=resolved_document_type,
        preprocessing="report_bilateral_clahe_deskew_adaptive",
        thresholding="adaptive_gaussian_binary",
        scale=report_scale,
        deskew_angle=deskew_angle,
        inverted=False,
        roi_used=resolved_roi,
        roi_fallback_used=fallback_used,
        roi_fallback_reason=fallback_reason,
    )


def extract_medical_text(
    image_input: ImageInput,
    document_type: MedicalDocumentType | str,
    *,
    roi: Mapping[str, object] | None = None,
    runtime_config: OcrRuntimeConfig | None = None,
    roi_policy: RoiPolicy = RoiPolicy.FALLBACK_TO_FULL_IMAGE,
    image_to_data: ImageToDataFn | None = None,
) -> MedicalOcrResult:
    started_at = time.perf_counter()
    cfg = runtime_config or OcrRuntimeConfig()
    preprocessed = preprocess_for_medical_ocr(
        image_input,
        document_type,
        roi=roi,
        roi_policy=roi_policy,
    )
    invocation = build_tesseract_invocation(
        preprocessed.document_type,
        has_roi=preprocessed.roi_used is not None,
        runtime_config=cfg,
    )
    tesseract_data = (image_to_data or _image_to_data)(
        preprocessed.image,
        lang=cfg.language,
        config=invocation.config,
    )

    words = _words_from_tesseract_data(
        tesseract_data,
        min_confidence=cfg.min_word_confidence,
    )
    raw_text = _join_words_by_line(words)
    cleaned_text = normalize_ocr_text_for_nlp(raw_text)
    confidence = sum(word.confidence for word in words) / len(words) if words else 0.0

    return MedicalOcrResult(
        text=cleaned_text,
        raw_text=raw_text,
        confidence=confidence,
        words=words,
        document_type=preprocessed.document_type,
        tesseract_config=invocation.config,
        psm=invocation.psm,
        oem=invocation.oem,
        dpi=invocation.dpi,
        preprocessing=preprocessed.preprocessing,
        thresholding=preprocessed.thresholding,
        scale=preprocessed.scale,
        deskew_angle=preprocessed.deskew_angle,
        inverted=preprocessed.inverted,
        roi_used=preprocessed.roi_used,
        roi_fallback_used=preprocessed.roi_fallback_used,
        roi_fallback_reason=preprocessed.roi_fallback_reason,
        processing_time_ms=(time.perf_counter() - started_at) * 1000,
    )


def normalize_ocr_text_for_nlp(text: str) -> str:
    """
    Normalize OCR into a stable SpaCy/regex input while keeping PHI-like tokens.

    The key repair is reconnecting runs of single OCR letters, for example
    ``J o h n D o e`` -> ``John Doe`` and ``2 0 2 4`` -> ``2024``.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = _ALLOWED_OCR_NOISE_RE.sub(" ", raw_line)
        line = _merge_single_character_runs(line)
        line = _merge_digit_character_runs(line)
        line = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "-", line)
        line = re.sub(r"(?<=\d)\s*([:./])\s*(?=\d)", r"\1", line)
        line = re.sub(r"\s*/\s*", "/", line)
        line = re.sub(r"\s+([,;.])", r"\1", line)
        line = re.sub(r"\s*:\s*", ": ", line)
        line = re.sub(r"(?<=\d): (?=\d)", ":", line)
        line = re.sub(r"([,;:.])\1{2,}", r"\1", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        if _line_has_signal(line):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _merge_single_character_runs(line: str) -> str:
    def replace(match: re.Match[str]) -> str:
        merged = re.sub(r"\s+", "", match.group(0))
        return _LOWER_TO_UPPER_RE.sub(" ", merged)

    previous = None
    current = line
    while previous != current:
        previous = current
        current = _SINGLE_CHAR_RUN_RE.sub(replace, current)
    return current


def _merge_digit_character_runs(line: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return _DIGIT_CHAR_RUN_RE.sub(replace, line)


def _line_has_signal(line: str) -> bool:
    if not line:
        return False
    if _STRUCTURED_MEDICAL_RE.search(line):
        return True
    alnum_count = sum(char.isalnum() for char in line)
    if alnum_count < 2:
        return False
    allowed_count = sum(
        char.isalnum() or char.isspace() or char in ".,:;/()+-[]#%" for char in line
    )
    return allowed_count / max(len(line), 1) >= 0.75


def _load_image(image_input: object) -> ImageArray:
    if isinstance(image_input, Image.Image):
        return np.asarray(image_input.convert("RGB"), dtype=np.uint8)
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    if not isinstance(image_input, np.ndarray):
        raise TypeError("image_input must be a path, PIL image, or uint8 ndarray")
    array_input = cast(npt.NDArray[np.generic], image_input)
    if array_input.dtype != np.uint8:
        raise TypeError("image_input ndarray must have dtype uint8")
    if array_input.ndim not in (2, 3):
        raise ValueError("image_input ndarray must be 2-D grayscale or 3-D color")
    if array_input.ndim == 3 and array_input.shape[2] not in (3, 4):
        raise ValueError("3-D image_input must have 3 or 4 channels")
    return cast(ImageArray, array_input)


def _to_grayscale(image: ImageArray) -> GrayArray:
    if image.ndim == 2:
        return image.copy()
    if image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        rgb = image
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _crop_or_full(
    source: ImageArray,
    *,
    roi: Mapping[str, object] | None,
    roi_policy: RoiPolicy,
) -> tuple[ImageArray, Rect | None, bool, str | None]:
    if roi is None:
        return source, None, False, None

    try:
        rect = _parse_roi(roi, source_shape=source.shape[:2])
    except (TypeError, ValueError) as exc:
        if roi_policy is RoiPolicy.RAISE:
            raise
        reason = str(exc)
        logger.warning("Invalid OCR ROI, falling back to full image: %s", reason)
        return source, None, True, reason

    cropped = source[rect.y : rect.bottom, rect.x : rect.right]
    return cropped, rect, False, None


def _parse_roi(
    roi: Mapping[str, object],
    *,
    source_shape: tuple[int, int],
) -> Rect:
    missing_keys = {"x", "y", "width", "height"} - set(roi)
    if missing_keys:
        raise ValueError(f"ROI is missing keys: {sorted(missing_keys)}")

    x = _boundary_int(roi["x"], "x")
    y = _boundary_int(roi["y"], "y")
    width = _boundary_int(roi["width"], "width")
    height = _boundary_int(roi["height"], "height")

    if x < 0 or y < 0:
        raise ValueError("ROI x and y must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be positive")

    source_height, source_width = source_shape
    if x >= source_width or y >= source_height:
        raise ValueError("ROI starts outside image bounds")

    right = min(source_width, x + width)
    bottom = min(source_height, y + height)
    if right <= x or bottom <= y:
        raise ValueError("ROI does not overlap the image")
    return Rect(x=x, y=y, width=right - x, height=bottom - y)


def _boundary_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"ROI {name} must be an integer, not bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    raise TypeError(f"ROI {name} must be an integer-like value")


def _preprocess_video_overlay(
    gray: GrayArray,
    *,
    has_roi: bool,
) -> tuple[GrayArray, float, bool]:
    scale = _scale_for_min_short_edge(gray, target=96 if has_roi else 720, maximum=3.0)
    scaled = _resize_gray(gray, scale)

    # Bilateral filtering removes compression noise while keeping glyph edges.
    denoised: GrayArray = cv2.bilateralFilter(scaled, 5, 45, 45)

    # CLAHE local contrast helps low-light overlay text without overexposing frames.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced: GrayArray = clahe.apply(denoised)

    mean_brightness = float(cast(SupportsFloat, np.mean(enhanced)))
    inverted = bool(mean_brightness < 127.0)
    if inverted:
        enhanced = cv2.bitwise_not(enhanced)

    sharpened = _unsharp_mask(enhanced)
    _, binary = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # A narrow close reconnects broken strokes without merging neighboring words.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    connected = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1,
    )
    return connected, scale, inverted


def _preprocess_report_page(
    gray: GrayArray,
    *,
    has_roi: bool,
    deskew: bool,
) -> tuple[GrayArray, float, float]:
    scale = _scale_for_min_short_edge(
        gray, target=120 if has_roi else 1400, maximum=2.0
    )
    scaled = _resize_gray(gray, scale)

    deskew_angle = _deskew_angle(scaled) if deskew else 0.0
    if abs(deskew_angle) >= 0.2:
        scaled = _rotate_gray(scaled, deskew_angle, border_value=255)
    else:
        deskew_angle = 0.0

    denoised: GrayArray = cv2.bilateralFilter(scaled, 7, 55, 55)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced: GrayArray = clahe.apply(denoised)

    block_size = 35 if min(enhanced.shape[:2]) >= 600 else 25
    adaptive_threshold = cast(
        _AdaptiveThresholdFn,
        cv2.adaptiveThreshold,  # pyright: ignore[reportUnknownMemberType]
    )
    binary = adaptive_threshold(
        enhanced,
        255.0,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        15.0,
    )
    return binary, scale, deskew_angle


def _scale_for_min_short_edge(
    gray: GrayArray,
    *,
    target: int,
    maximum: float,
) -> float:
    height, width = gray.shape[:2]
    short_edge = min(height, width)
    if short_edge <= 0 or short_edge >= target:
        return 1.0
    return min(maximum, target / short_edge)


def _resize_gray(gray: GrayArray, scale: float) -> GrayArray:
    if abs(scale - 1.0) < 0.01:
        return gray.copy()
    height, width = gray.shape[:2]
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)


def _unsharp_mask(gray: GrayArray) -> GrayArray:
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def _deskew_angle(gray: GrayArray) -> float:
    _, foreground = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    coords = np.column_stack(np.where(foreground > 0))
    if coords.shape[0] < 30:
        return 0.0

    min_area_rect = cast(
        _MinAreaRectFn,
        cv2.minAreaRect,  # pyright: ignore[reportUnknownMemberType]
    )
    angle = float(min_area_rect(coords.astype(np.float32))[2])
    correction = -(90.0 + angle) if angle < -45.0 else -angle
    if abs(correction) > 15.0:
        return 0.0
    return correction


def _rotate_gray(gray: GrayArray, angle: float, *, border_value: int) -> GrayArray:
    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    get_rotation_matrix = cast(
        _GetRotationMatrix2DFn,
        cv2.getRotationMatrix2D,  # pyright: ignore[reportUnknownMemberType]
    )
    matrix = get_rotation_matrix(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _words_from_tesseract_data(
    data: TesseractData,
    *,
    min_confidence: float,
) -> tuple[OcrWord, ...]:
    texts = _field(data, "text")
    confs = _field(data, "conf")
    lefts = _field(data, "left")
    tops = _field(data, "top")
    widths = _field(data, "width")
    heights = _field(data, "height")
    blocks = _field(data, "block_num")
    paragraphs = _field(data, "par_num")
    lines = _field(data, "line_num")

    line_indexes: dict[tuple[int, int, int], int] = {}
    words: list[OcrWord] = []

    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        confidence = _confidence_at(confs, index)
        if confidence is None or confidence < min_confidence:
            continue

        line_key = (
            _int_at(blocks, index, default=0),
            _int_at(paragraphs, index, default=0),
            _int_at(lines, index, default=0),
        )
        if line_key not in line_indexes:
            line_indexes[line_key] = len(line_indexes)
        line_index = line_indexes[line_key]

        left = _int_at(lefts, index, default=0)
        top = _int_at(tops, index, default=0)
        width = max(0, _int_at(widths, index, default=0))
        height = max(0, _int_at(heights, index, default=0))
        words.append(
            OcrWord(
                text=text,
                confidence=confidence / 100.0,
                box=(left, top, left + width, top + height),
                line_index=line_index,
            )
        )

    return tuple(words)


def _join_words_by_line(words: Sequence[OcrWord]) -> str:
    if not words:
        return ""

    lines: dict[int, list[str]] = {}
    for word in words:
        if word.line_index not in lines:
            lines[word.line_index] = []
        lines[word.line_index].append(word.text)
    return "\n".join(" ".join(lines[index]) for index in sorted(lines))


def _field(data: TesseractData, key: str) -> Sequence[object]:
    return data.get(key, ())


def _int_at(values: Sequence[object], index: int, *, default: int) -> int:
    if index >= len(values):
        return default
    value = values[index]
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _confidence_at(values: Sequence[object], index: int) -> float | None:
    if index >= len(values):
        return None
    value = values[index]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        confidence = float(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            confidence = float(stripped)
        except ValueError:
            return None
    else:
        return None
    if confidence < 0:
        return None
    return min(confidence, 100.0)


def _coerce_document_type(
    document_type: MedicalDocumentType | str,
) -> MedicalDocumentType:
    if isinstance(document_type, MedicalDocumentType):
        return document_type
    try:
        return MedicalDocumentType(document_type.lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in MedicalDocumentType)
        raise ValueError(f"document_type must be one of: {valid}") from exc


def _parse_roi_arg(value: str) -> dict[str, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x,y,width,height")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi values must be integers") from exc
    return {"x": x, "y": y, "width": width, "height": height}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OpenCV+pytesseract OCR for hospital video overlays and reports."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--document-type",
        choices=[item.value for item in MedicalDocumentType],
        default=MedicalDocumentType.VIDEO.value,
    )
    parser.add_argument("--roi", type=_parse_roi_arg, default=None)
    parser.add_argument(
        "--roi-policy",
        choices=[item.value for item in RoiPolicy],
        default=RoiPolicy.FALLBACK_TO_FULL_IMAGE.value,
    )
    parser.add_argument("--lang", default=OcrRuntimeConfig.language)
    args = parser.parse_args(argv)

    result = extract_medical_text(
        args.image,
        MedicalDocumentType(args.document_type),
        roi=cast(dict[str, int] | None, args.roi),
        roi_policy=RoiPolicy(args.roi_policy),
        runtime_config=OcrRuntimeConfig(language=cast(str, args.lang)),
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
