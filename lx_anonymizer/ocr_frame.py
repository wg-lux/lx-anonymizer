import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Import TesseOCR if available (requires tesserocr)
try:
    from .ocr_frame_tesserocr import TesseOCRFrameProcessor

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
                logger.warning(f"Failed to initialize TesseOCR, falling back to PyTesseract: {e}")
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
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text + confidence + meta from a single frame.
        Always returns a (text, confidence, meta) tuple.
        """
        # Prefer tesserocr for performance
        if self.tesserocr_processor:
            try:
                return self.tesserocr_processor.extract_text_from_frame(frame, roi, high_quality)
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
    def _preprocess_frame(self, frame: np.ndarray, roi: Optional[Dict[str, Any]]) -> Image.Image:
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
            pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            return pil
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return Image.fromarray(frame)

    # ---------------- Validation ----------------
    @staticmethod
    def _validate_roi(roi: Dict[str, Any]) -> bool:
        """Validate ROI dictionary structure."""
        try:
            return all(k in roi for k in ("x", "y", "width", "height")) and roi["width"] > 0 and roi["height"] > 0
        except Exception:
            return False
