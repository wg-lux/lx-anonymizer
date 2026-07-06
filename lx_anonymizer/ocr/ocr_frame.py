import logging
import os
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, TypeAlias, Tuple, TypedDict, cast

import cv2
import numpy as np
import pytesseract  # type: ignore[import-untyped]
from PIL import Image, ImageEnhance, ImageFilter

from lx_anonymizer.config import settings
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.regex_patterns import STRUCTURED_OVERLAY_RE

_RapidOCR: Optional[type[Any]] = None
RAPIDOCR_BACKEND = "rapidocr"
RAPIDOCR_ACCELERATION_ENV = "LX_ANONYMIZER_RAPIDOCR_ACCELERATION"
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
rapidocr_available = False

try:
    from rapidocr import RapidOCR  # type: ignore[import-untyped]

    _RapidOCR = RapidOCR  # Assign the class to your internal variable
    rapidocr_available = True
except ImportError:
    rapidocr_available = False

_TesseOCRFrameProcessor: Optional[type[Any]] = None
tesserocr_available = False
try:
    from lx_anonymizer.ocr.ocr_frame_tesserocr import TesseOCRFrameProcessor

    _TesseOCRFrameProcessor = TesseOCRFrameProcessor
    tesserocr_available = True
except ImportError:
    tesserocr_available = False

FlatRoi: TypeAlias = dict[str, int | None]
NestedRoi: TypeAlias = dict[str, FlatRoi]
RoiInput: TypeAlias = NestedRoi | FlatRoi | list[object] | None


class PytesseractData(TypedDict):
    text: list[str]
    conf: list[str | int | float]


logger = logging.getLogger(__name__)


class FrameOCR:
    """
    High-performance OCR interface for medical video frames.
    - Uses RapidOCR when available (CPU ONNX Runtime, no GPU VRAM)
    - Uses tesserocr when RapidOCR is unavailable
    - Falls back to pytesseract
    - Handles both ROI and full-frame OCR
    - Includes medical pattern extraction helpers
    """

    def __init__(self):
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.rapidocr_engine: Optional[Any] = None
        self.rapidocr_params = self._rapidocr_init_params()
        self._rapidocr_lock = threading.Lock()
        self.tesserocr_processor: Optional[Any] = None
        self._rapidocr_available = rapidocr_available and _RapidOCR is not None

        if self._rapidocr_available:
            logger.info(
                "FrameOCR configured with lazy %s backend (acceleration=%s)",
                RAPIDOCR_BACKEND,
                self._resolved_rapidocr_acceleration(),
            )
        elif tesserocr_available and _TesseOCRFrameProcessor is not None:
            self._ensure_tesserocr_processor()
        else:
            logger.info("TesseOCR not available, using PyTesseract")

        self.pytesseract_config = {
            "lang": "deu+eng",
            "oem": 3,
            "psm": 6,
            "dpi": 300,
        }

    # ---------------- Public API ----------------
    def extract_text_from_frame(
        self,
        frame: np.ndarray,
        roi: RoiInput,
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text + confidence + meta from a single frame.
        Always returns a (text, confidence, meta) tuple.
        """
        # Prefer RapidOCR: CPU-bound ONNX Runtime avoids GPU VRAM pressure.
        if self._rapidocr_available:
            try:
                self._ensure_rapidocr_engine()
                return self._extract_text_rapidocr(frame, roi, high_quality)
            except Exception as e:
                logger.error(
                    "RapidOCR failed, falling back to TesseOCR/PyTesseract: %s",
                    e,
                )
                self._rapidocr_available = False

        if (
            self.tesserocr_processor is None
            and tesserocr_available
            and _TesseOCRFrameProcessor is not None
        ):
            self._ensure_tesserocr_processor()

        if self.tesserocr_processor:
            try:
                return self.tesserocr_processor.extract_text_from_frame(
                    frame, roi, high_quality
                )
            except Exception as e:
                logger.error(f"TesseOCR failed, falling back to PyTesseract: {e}")

        # Fallback to pytesseract
        flat_roi = roi if roi and "x" in roi else None
        return self._extract_text_pytesseract(
            frame, cast(Optional[FlatRoi], flat_roi), high_quality
        )

    def _ensure_rapidocr_engine(self) -> None:
        if self.rapidocr_engine is not None:
            return
        with self._rapidocr_lock:
            if self.rapidocr_engine is not None:
                return
            if not rapidocr_available or _RapidOCR is None:
                raise RuntimeError("RapidOCR is not available")
            rapidocr_params = cast(
                Dict[str, Any],
                getattr(self, "rapidocr_params", {}),
            )
            if rapidocr_params:
                self.rapidocr_engine = _RapidOCR(params=rapidocr_params)
            else:
                self.rapidocr_engine = _RapidOCR()
            logger.info(
                "FrameOCR initialized with %s backend params=%s",
                RAPIDOCR_BACKEND,
                rapidocr_params,
            )

    @staticmethod
    def _resolved_rapidocr_acceleration() -> str:
        configured = os.environ.get(RAPIDOCR_ACCELERATION_ENV, "").strip().lower()
        if configured:
            return configured
        return settings.RAPIDOCR_ACCELERATION

    @classmethod
    def _rapidocr_init_params(cls) -> Dict[str, Any]:
        acceleration = cls._resolved_rapidocr_acceleration()
        if acceleration not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "Invalid RapidOCR acceleration setting "
                f"{acceleration!r}; expected 'auto', 'cpu', or 'cuda'."
            )
        if acceleration == "cpu":
            return {"EngineConfig.onnxruntime.use_cuda": False}

        available_providers = cls._available_onnx_providers()
        if CUDA_EXECUTION_PROVIDER in available_providers:
            logger.info(
                "RapidOCR will request %s; available ONNX providers=%s",
                CUDA_EXECUTION_PROVIDER,
                list(available_providers),
            )
            return {"EngineConfig.onnxruntime.use_cuda": True}

        if acceleration == "cuda":
            raise RuntimeError(
                "RapidOCR CUDA acceleration was requested, but "
                f"{CUDA_EXECUTION_PROVIDER} is not available. "
                f"Available ONNX providers: {list(available_providers)}"
            )

        logger.info(
            "RapidOCR using CPU ONNX provider; available ONNX providers=%s",
            list(available_providers),
        )
        return {"EngineConfig.onnxruntime.use_cuda": False}

    @staticmethod
    def _available_onnx_providers() -> tuple[str, ...]:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError:
            return ()

        providers = cast(Sequence[object], cast(Any, ort).get_available_providers())
        return tuple(str(provider) for provider in providers)

    def _ensure_tesserocr_processor(self) -> None:
        if self.tesserocr_processor is not None:
            return
        try:
            if not _TesseOCRFrameProcessor:
                raise ImportError
            self.tesserocr_processor = _TesseOCRFrameProcessor(language="deu")
            logger.info("FrameOCR initialized with TesseOCR acceleration")
        except Exception as e:
            logger.warning(
                "Failed to initialize TesseOCR, falling back to PyTesseract: %s",
                e,
            )
            self.tesserocr_processor = None

    # ---------------- RapidOCR backend ----------------
    def _extract_text_rapidocr(
        self,
        frame: np.ndarray,
        roi: RoiInput = None,
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Run RapidOCR and normalize output to the FrameOCR API."""
        if self.rapidocr_engine is None:
            return "", 0.0, {}

        _ = high_quality
        t0 = time.time()
        roi_entries = self._normalize_roi_input(roi)

        if roi_entries:
            all_texts: list[str] = []
            all_confs: list[float] = []
            all_regions: list[dict[str, Any]] = []
            metadata: Dict[str, Any] = {
                "backend": RAPIDOCR_BACKEND,
                "method": "rapidocr+roi",
                "roi_count": len(roi_entries),
            }

            for idx, (_name, flat_roi) in enumerate(roi_entries):
                roi_text, roi_conf, roi_regions, roi_elapsed = self._run_rapidocr(
                    frame, flat_roi
                )
                metadata[f"roi_{idx}"] = roi_text
                if roi_elapsed is not None:
                    metadata[f"roi_{idx}_elapse"] = roi_elapsed

                if roi_text:
                    all_texts.append(roi_text)
                if roi_conf > 0:
                    all_confs.append(roi_conf)
                all_regions.extend(roi_regions)

            text = "\n".join(all_texts)
            avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
            metadata.update(self._rapidocr_metadata(text, avg_conf, all_regions, t0))
            return text, avg_conf, metadata

        text, avg_conf, regions, elapsed = self._run_rapidocr(frame, None)
        metadata = {
            "backend": RAPIDOCR_BACKEND,
            "method": "rapidocr",
            "elapse": elapsed,
        }
        metadata.update(self._rapidocr_metadata(text, avg_conf, regions, t0))
        return text, avg_conf, metadata

    def _run_rapidocr(
        self, frame: np.ndarray, roi: Optional[FlatRoi]
    ) -> tuple[str, float, list[dict[str, Any]], Optional[float]]:
        img, x_offset, y_offset = self._crop_frame(frame, roi)
        engine = self.rapidocr_engine
        if engine is None:
            raise RuntimeError("RapidOCR engine is not initialized")
        with self._rapidocr_lock:
            result = engine(img)
        entries, elapsed = self._parse_rapidocr_result(result, x_offset, y_offset)

        entries.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        texts = [entry["text"] for entry in entries if entry["text"]]
        confs = [entry["confidence"] for entry in entries if entry["confidence"] > 0]
        text = " ".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return text, avg_conf, entries, elapsed

    def _parse_rapidocr_result(
        self, result: Any, x_offset: int, y_offset: int
    ) -> tuple[list[dict[str, Any]], Optional[float]]:
        elapsed: Optional[float] = None

        if hasattr(result, "elapse"):
            elapsed = self._float_or_none(getattr(result, "elapse"))

        entries: list[dict[str, Any]] = []

        if hasattr(result, "boxes") and hasattr(result, "txts"):
            boxes = cast(Sequence[object] | None, getattr(result, "boxes"))
            if boxes is None:
                return entries, elapsed

            txts = cast(Sequence[object] | None, getattr(result, "txts"))
            if txts is None:
                txts = ()

            scores = cast(Sequence[object] | None, getattr(result, "scores", None))
            if scores is None:
                scores = tuple(0.0 for _ in txts)

            for box, text, score in zip(boxes, txts, scores, strict=False):
                entry = self._rapidocr_entry(box, text, score, x_offset, y_offset)
                if entry:
                    entries.append(entry)
            return entries, elapsed

        return entries, elapsed

    def _rapidocr_entry(
        self, box: Any, text: Any, score: Any, x_offset: int, y_offset: int
    ) -> Optional[dict[str, Any]]:
        cleaned_text = str(text or "").strip()
        if not cleaned_text:
            return None

        points = np.asarray(box, dtype=float).reshape(-1, 2)
        if points.size == 0:
            return None

        points[:, 0] += x_offset
        points[:, 1] += y_offset
        xs = points[:, 0]
        ys = points[:, 1]
        bbox = [
            int(round(float(xs.min()))),
            int(round(float(ys.min()))),
            int(round(float(xs.max()))),
            int(round(float(ys.max()))),
        ]
        return {
            "text": cleaned_text,
            "confidence": self._normalize_confidence(score),
            "box": [
                [int(round(float(x))), int(round(float(y)))] for x, y in points.tolist()
            ],
            "bbox": bbox,
        }

    @staticmethod
    def _normalize_confidence(score: Any) -> float:
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            return 0.0
        if confidence > 1.0:
            confidence /= 100.0
        return max(0.0, min(confidence, 1.0))

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _rapidocr_metadata(
        self,
        text: str,
        avg_conf: float,
        regions: list[dict[str, Any]],
        started_at: float,
    ) -> Dict[str, Any]:
        return {
            "words": len(text.split()),
            "avg_conf": avg_conf,
            "confidence": avg_conf,
            "regions": len(regions),
            "text_regions": regions,
            "processing_time": time.time() - started_at,
        }

    def _normalize_roi_input(self, roi: RoiInput) -> list[tuple[str, FlatRoi]]:
        rois: list[tuple[str, FlatRoi]] = []

        def collect(value: object, name: str = "") -> None:
            if not value:
                return
            if isinstance(value, dict) and "x" in value:
                flat_roi = cast(FlatRoi, value)
                if self._validate_roi(flat_roi):
                    rois.append((name, flat_roi))
                return
            if isinstance(value, Mapping):
                nested_mapping = cast(Mapping[str, object], value)
                for key, nested in nested_mapping.items():
                    collect(nested, str(key))
                return
            if isinstance(value, list):
                nested_values = cast(list[object], value)
                for idx, nested in enumerate(nested_values):
                    collect(nested, str(idx))

        collect(roi)
        return rois

    def _crop_frame(
        self, frame: np.ndarray, roi: Optional[FlatRoi]
    ) -> tuple[np.ndarray, int, int]:
        if not roi or not self._validate_roi(roi):
            return frame, 0, 0

        frame_height, frame_width = frame.shape[:2]
        x = max(0, int(cast(int, roi["x"])))
        y = max(0, int(cast(int, roi["y"])))
        width = int(cast(int, roi["width"]))
        height = int(cast(int, roi["height"]))
        x2 = min(frame_width, x + width)
        y2 = min(frame_height, y + height)

        if x >= frame_width or y >= frame_height or x2 <= x or y2 <= y:
            return frame, 0, 0
        return frame[y:y2, x:x2], x, y

    # ---------------- PyTesseract fallback ----------------
    def _extract_text_pytesseract(
        self,
        frame: np.ndarray,
        roi: Optional[FlatRoi] = None,
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Simplified pytesseract fallback for emergency OCR."""
        try:
            img = self._preprocess_frame(frame, roi)
            cfg = self.pytesseract_config
            config_str = f"--oem {cfg['oem']} --psm {cfg['psm']} --dpi {cfg['dpi']}"
            data = cast(
                PytesseractData,
                cast(Any, pytesseract).image_to_data(
                    img,
                    lang=cfg["lang"],
                    config=config_str,
                    output_type=pytesseract.Output.DICT,
                ),
            )

            words: list[str] = []
            confs: list[int] = []
            for text, conf in zip(data["text"], data["conf"]):
                stripped_text = text.strip()
                int_conf = int(conf)
                if stripped_text and int_conf > 0:
                    words.append(stripped_text)
                    confs.append(int_conf)

            text = " ".join(words)
            avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.0
            return text, avg_conf, {"words": len(words), "avg_conf": avg_conf}
        except Exception as e:
            logger.error(f"PyTesseract OCR failed: {e}")
            return "", 0.0, {}

    # ---------------- Preprocessing ----------------
    def _preprocess_frame(
        self, frame: np.ndarray, roi: Optional[FlatRoi]
    ) -> Image.Image:
        """Light preprocessing for pytesseract fallback."""
        try:
            img = frame
            if roi and self._validate_roi(roi):
                x = int(cast(int, roi["x"]))
                y = int(cast(int, roi["y"]))
                w = int(cast(int, roi["width"]))
                h = int(cast(int, roi["height"]))
                img = img[y : y + h, x : x + w]

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(img).convert("L")
            pil = ImageEnhance.Contrast(pil).enhance(2.0)
            pil = pil.filter(
                ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
            )
            return pil
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return Image.fromarray(frame)

    # ---------------- Validation ----------------
    @staticmethod
    def _validate_roi(roi: FlatRoi) -> bool:
        """Validate ROI dictionary structure."""
        try:
            width = roi["width"]
            height = roi["height"]
            return (
                all(k in roi for k in ("x", "y", "width", "height"))
                and width is not None
                and height is not None
                and width > 0
                and height > 0
            )
        except Exception:
            return False

    def _ocr_with_tesserocr(
        self,
        gray_frame: np.ndarray,
        endoscope_data_roi_nested: Optional[NestedRoi | list[object]] = None,
    ) -> tuple[str, float, Dict[str, Any], bool]:
        """
        OCR with TesserOCR and metadata extraction.
        Handles both dict-based and list-based ROI structures gracefully.
        Includes enhanced validation to filter gibberish output.
        """
        try:
            logger.debug("Using TesserOCR OCR engine with enhanced validation")
            frame_metadata: Dict[str, Any] = {}
            ocr_text = ""
            valid_texts: list[str] = []  # Store only validated text

            # --- Normalize input ROI structure ---
            rois: list[FlatRoi] = []

            if not endoscope_data_roi_nested:
                has_roi = False
            elif isinstance(endoscope_data_roi_nested, dict):
                # Original expected format
                rois = [
                    roi
                    for roi in endoscope_data_roi_nested.values()
                    if self._validate_roi(roi)
                ]
                has_roi = True
            else:
                # Flatten nested lists of dicts
                for item in endoscope_data_roi_nested:
                    if isinstance(item, dict):
                        flat_item = cast(FlatRoi, item)
                        if self._validate_roi(flat_item):
                            rois.append(flat_item)
                    elif isinstance(item, list):
                        sub_items = cast(list[object], item)
                        for sub in sub_items:
                            if isinstance(sub, dict):
                                flat_sub = cast(FlatRoi, sub)
                                if self._validate_roi(flat_sub):
                                    rois.append(flat_sub)
                has_roi = len(rois) > 0

            # --- Run OCR ---
            if not has_roi:
                ocr_text, ocr_conf, _ = self.extract_text_from_frame(
                    gray_frame, roi=None, high_quality=True
                )

                # Validate full-frame OCR
                if self._is_valid_ocr_text(ocr_text):
                    valid_texts.append(ocr_text)
                else:
                    logger.debug(
                        f"Full-frame OCR produced invalid text, filtering: {ocr_text[:100]}"
                    )
                    ocr_text = ""
            else:
                ocr_conf = 0.0
                for i, roi in enumerate(rois):
                    output, conf, _ = self.extract_text_from_frame(
                        gray_frame, roi=roi, high_quality=True
                    )

                    # Validate each ROI's OCR output
                    if self._is_valid_ocr_text(output):
                        valid_texts.append(output)
                        frame_metadata[f"roi_{i}"] = output
                        ocr_conf = max(ocr_conf, conf)
                    else:
                        logger.debug(
                            f"ROI {i} produced invalid text (filtered): {output[:50]}"
                        )
                        frame_metadata[f"roi_{i}"] = ""

                # Combine only valid texts
                ocr_text = "\n".join(valid_texts)

            logger.debug(
                f"TesserOCR extracted {len(valid_texts)} valid text regions, total length: {len(ocr_text)}, conf: {ocr_conf:.3f}"
            )

            # --- Metadata Extraction ---
            # Always try to extract metadata from the combined OCR text
            if ocr_text:
                logger.debug("Extracting metadata from combined OCR text")
                extracted_meta = (
                    self.frame_metadata_extractor.extract_metadata_from_frame_text(
                        ocr_text
                    )
                )
                # Merge extracted metadata with ROI metadata
                frame_metadata.update(extracted_meta)

            is_sensitive = self.frame_metadata_extractor.is_sensitive_content(
                frame_metadata
            )
            return ocr_text, ocr_conf, frame_metadata, is_sensitive

        except Exception as e:
            logger.exception(f"TesserOCR OCR failed: {e}")
            return "", 0.0, {}, False

    def _is_valid_ocr_text(
        self, text: str, min_alpha_ratio: float = 0.20, min_length: int = 3
    ) -> bool:
        """
        Validate OCR text to filter out gibberish.

        Args:
            text: OCR extracted text
            min_alpha_ratio: Minimum ratio of alphabetic characters (default 0.20, relaxed from 0.35)
            min_length: Minimum text length (default 3)

        Returns:
            True if text appears valid, False if likely gibberish
        """
        if not text or len(text.strip()) < min_length:
            return False

        # Special case: Allow date/time patterns (have few letters but are valid)
        # Examples: "09:53:32", "2024-01-15", "15702/2024", "E 15702/2024809:53:32"
        if STRUCTURED_OVERLAY_RE.search(text):
            # Contains structured data patterns - likely valid
            return True

        # Calculate alphabetic character ratio
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / len(text)

        # Relaxed from 0.35 to 0.20 to allow more numeric/mixed content
        if alpha_ratio < min_alpha_ratio:
            return False

        # Check for excessive special characters
        expected_chars = set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-()[]"
        )
        nonstandard = sum(1 for c in text if c not in expected_chars)

        # Relaxed from 0.3 to 0.4 to allow more special characters
        if nonstandard > 0.4 * len(text):
            return False

        # Check for reasonable word structure
        words = [w for w in text.split() if len(w) > 1]
        if not words:
            return False

        # Most words should have vowels (German/English)
        vowels = set("aeiouäöüAEIOUÄÖÜ")
        words_with_vowels = sum(1 for word in words if any(c in vowels for c in word))

        # Relaxed from 0.25 to 0.15 to allow more abbreviations/codes
        if len(words) > 2 and words_with_vowels < 0.15 * len(words):
            return False

        return True
