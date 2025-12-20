import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Import TesseOCR if available (requires tesserocr)
try:
    from lx_anonymizer.ocr_frame_tesserocr import TesseOCRFrameProcessor

    TESSEROCR_AVAILABLE = True
except ImportError:
    TESSEROCR_AVAILABLE = False
    TesseOCRFrameProcessor = None

logger = logging.getLogger(__name__)


class FrameOCR:
    """
    High-performance OCR interface for medical video frames.
    - Uses tesserocr when available (10–50× faster)
    - Falls back to pytesseract
    - Handles both ROI and full-frame OCR
    - Includes medical pattern extraction helpers
    """

    def __init__(self):
        if TESSEROCR_AVAILABLE and TesseOCRFrameProcessor:
            try:
                self.tesserocr_processor = TesseOCRFrameProcessor(language="deu")
                logger.info("FrameOCR initialized with TesseOCR acceleration")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize TesseOCR, falling back to PyTesseract: {e}"
                )
                self.tesserocr_processor = None
        else:
            logger.info("TesseOCR not available, using PyTesseract")
            self.tesserocr_processor = None

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
        roi: Optional[dict[str, dict[str, int | None]] | None],
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text + confidence + meta from a single frame.
        Always returns a (text, confidence, meta) tuple.
        """
        # Prefer tesserocr for performance
        if self.tesserocr_processor:
            try:
                return self.tesserocr_processor.extract_text_from_frame(
                    frame, roi, high_quality
                )
            except Exception as e:
                logger.error(f"TesseOCR failed, falling back to PyTesseract: {e}")

        # Fallback to pytesseract
        return self._extract_text_pytesseract(frame, roi, high_quality)

    # ---------------- PyTesseract fallback ----------------
    def _extract_text_pytesseract(
        self,
        frame: np.ndarray,
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Simplified pytesseract fallback for emergency OCR."""
        try:
            img = self._preprocess_frame(frame, roi)
            cfg = self.pytesseract_config
            config_str = f"--oem {cfg['oem']} --psm {cfg['psm']} --dpi {cfg['dpi']}"
            data = pytesseract.image_to_data(
                img,
                lang=cfg["lang"],
                config=config_str,
                output_type=pytesseract.Output.DICT,
            )

            words, confs = [], []
            for text, conf in zip(data["text"], data["conf"]):
                if text.strip() and int(conf) > 0:
                    words.append(text.strip())
                    confs.append(int(conf))

            text = " ".join(words)
            avg_conf = (sum(confs) / len(confs) / 100) if confs else 0.0
            return text, avg_conf, {"words": len(words), "avg_conf": avg_conf}
        except Exception as e:
            logger.error(f"PyTesseract OCR failed: {e}")
            return "", 0.0, {}

    # ---------------- Preprocessing ----------------
    def _preprocess_frame(
        self, frame: np.ndarray, roi: Optional[Dict[str, Any]]
    ) -> Image.Image:
        """Light preprocessing for pytesseract fallback."""
        try:
            img = frame
            if roi and self._validate_roi(roi):
                x, y, w, h = map(int, (roi["x"], roi["y"], roi["width"], roi["height"]))
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
    def _validate_roi(roi: Dict[str, Any]) -> bool:
        """Validate ROI dictionary structure."""
        try:
            return (
                all(k in roi for k in ("x", "y", "width", "height"))
                and roi["width"] > 0
                and roi["height"] > 0
            )
        except Exception:
            return False

    def _ocr_with_tesserocr(
        self,
        gray_frame: np.ndarray,
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | None],
    ) -> tuple[str, float, dict, bool]:
        """
        OCR with TesserOCR and metadata extraction.
        Handles both dict-based and list-based ROI structures gracefully.
        Includes enhanced validation to filter gibberish output.
        """
        try:
            logger.debug("Using TesserOCR OCR engine with enhanced validation")
            frame_metadata: Dict[str, Any] = {}
            ocr_text = ""
            valid_texts = []  # Store only validated text

            # --- Normalize input ROI structure ---
            rois: list[dict[str, int | None]] = []

            if not endoscope_data_roi_nested:
                has_roi = False
            elif isinstance(endoscope_data_roi_nested, dict):
                # Original expected format
                rois = list(endoscope_data_roi_nested.values())
                has_roi = True
            elif isinstance(endoscope_data_roi_nested, list):
                # Flatten nested lists of dicts
                for item in endoscope_data_roi_nested:
                    if isinstance(item, dict):
                        rois.append(item)
                    elif isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict):
                                rois.append(sub)
                has_roi = len(rois) > 0
            else:
                logger.warning(
                    f"Unexpected ROI type: {type(endoscope_data_roi_nested)}"
                )
                has_roi = False

            # --- Run OCR ---
            if not has_roi:
                ocr_text, ocr_conf, _ = self.extract_text_from_frame(
                    gray_frame, roi={}, high_quality=True
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
        import re

        # Time patterns: HH:MM:SS or HH:MM
        time_pattern = r"\d{1,2}:\d{2}(?::\d{2})?"
        # Date patterns: YYYY-MM-DD, DD.MM.YYYY, YYYY/MM/DD
        date_pattern = r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{4}"
        # Case number patterns: E 12345/2024 or similar
        case_pattern = r"[A-Z]\s*\d{4,}/\d{4}"
        # Device ID patterns: long numbers with optional separators
        device_pattern = r"\d{8,}"

        if (
            re.search(time_pattern, text)
            or re.search(date_pattern, text)
            or re.search(case_pattern, text)
            or re.search(device_pattern, text)
        ):
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
