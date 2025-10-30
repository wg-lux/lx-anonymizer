"""
High-performance Frame-specific OCR module using Tesserocr for medical video anonymization.

New: Region-first OCR
- Detect text-like regions with OpenCV (morph gradient + close + contour filter)
- OCR only those ROIs; fallback to full-frame if none found
- Grayscale-safe, robust preprocessing, dynamic inversion
- German-biased decoding; optional anti-gibberish filter
"""

import glob
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tesserocr
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------- Global processor cache (per language) ----------------
_GLOBAL_PROCESSORS: Dict[str, "TesseOCRFrameProcessor"] = {}
_GLOBAL_LOCK = threading.Lock()


class TesseOCRFrameProcessor:
    """
    High-performance OCR processor using Tesserocr for video frames.
    Now with text-region detection to increase accuracy and reduce gibberish.
    """

    def __init__(self, language: str = "deu"):
        self.language = language
        self._lock = threading.Lock()
        self.api: Optional[tesserocr.PyTessBaseAPI] = None
        self._initialize_api()

        self.frame_config = {
            "dpi": 400,
            "high_quality_dpi": 600,
            "min_confidence": 40,  # Increased from 20 to filter more garbage
            "high_quality_min_confidence": 50,  # Increased from 30
        }

        # Heuristics for text-region detection
        self._roi_min_area = 50  # px^2
        self._roi_aspect_min = 1.2  # width / height
        self._roi_aspect_max = 20.0
        self._roi_min_height = 10  # px
        self._max_area_ratio = 0.5  # discard giant blobs

        # Preprocessing parameters
        self._contrast_gain = 2.0
        self._upscale_target_min_dim = 1000  # px
        self._upscale_max_factor = 4.0

        # Stats
        self.processed_frames = 0
        self.total_processing_time = 0.0

    # ---------------- Tesseract init ----------------
    def _initialize_api(self) -> None:
        tessdata_path = self._get_tessdata_path()
        self.api = tesserocr.PyTessBaseAPI(
            lang=self.language,  # German prioritized
            path=tessdata_path,
            oem=tesserocr.OEM.DEFAULT,  # allow legacy fallback for tricky fonts
        )
        # Default PSM: single block (we switch per-ROI later)
        self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

        # Performance & stability
        self.api.SetVariable("preserve_interword_spaces", "1")
        self.api.SetVariable("user_defined_dpi", "400")
        self.api.SetVariable("classify_enable_learning", "0")
        self.api.SetVariable("tessedit_do_invert", "0")
        self.api.SetVariable("tessedit_enable_doc_dict", "0")
        self.api.SetVariable("load_system_dawg", "0")
        self.api.SetVariable("load_freq_dawg", "1")
        self.api.SetVariable("textord_really_old_xheight", "1")  # small caps overlays

        # German bias & safe charset
        self.api.SetVariable("language_model_penalty_non_dict_word", "1")
        self.api.SetVariable("language_model_penalty_non_freq_dict_word", "1")
        self.api.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-")

        logger.info("TesseOCR initialized (OEM=DEFAULT, PSM=SINGLE_BLOCK, lang=%s, path=%s)", self.language, tessdata_path)

    # ---------------- tessdata discovery ----------------
    def _get_tessdata_path(self) -> Optional[str]:
        """
        Return tessdata/ directory itself (PyTessBaseAPI expects the dir containing *.traineddata).
        """
        env_tessdata_parent = os.environ.get("TESSDATA_PREFIX")
        if env_tessdata_parent:
            if env_tessdata_parent.endswith("/tessdata") and os.path.isdir(env_tessdata_parent):
                logger.info("Using TESSDATA_PREFIX directly: %s", env_tessdata_parent)
                return env_tessdata_parent
            tessdata_dir = os.path.join(env_tessdata_parent, "tessdata")
            if os.path.isdir(tessdata_dir):
                logger.info("Using tessdata from TESSDATA_PREFIX parent: %s", tessdata_dir)
                return tessdata_dir

        nix_patterns = [
            "/nix/store/*/share/tessdata",
            "/run/current-system/sw/share/tessdata",
        ]
        for pattern in nix_patterns:
            if "*" in pattern:
                for candidate in glob.glob(pattern):
                    if os.path.isdir(candidate):
                        if any(f.endswith(".traineddata") for f in os.listdir(candidate)):
                            logger.info("Using NixOS tessdata: %s", candidate)
                            return candidate
            elif os.path.isdir(pattern):
                logger.info("Using tessdata: %s", pattern)
                return pattern

        for p in ["/usr/share/tessdata", "/usr/local/share/tessdata"]:
            if os.path.isdir(p):
                logger.info("Using tessdata: %s", p)
                return p

        logger.warning("No tessdata path found; falling back to tesserocr default")
        return None

    # ---------------- Helpers ----------------
    @staticmethod
    def _validate_roi(roi: Dict[str, Any]) -> bool:
        req = ["x", "y", "width", "height"]
        return all(k in roi for k in req) and roi["width"] > 0 and roi["height"] > 0

    @staticmethod
    def _is_gibberish(text: str) -> bool:
        """Enhanced gibberish detection for video OCR with support for structured data"""
        if not text or len(text) < 3:
            return True

        # Special case: Allow date/time patterns and numeric IDs (common in medical videos)
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

        if re.search(time_pattern, text) or re.search(date_pattern, text) or re.search(case_pattern, text) or re.search(device_pattern, text):
            # Contains structured data patterns - likely valid
            return False

        # Count alphabetic characters
        alpha = sum(c.isalpha() for c in text)
        alpha_ratio = alpha / max(len(text), 1)

        # Reject if too few alphabetic characters (only for non-structured text)
        # Relaxed from 0.35 to 0.20 to allow more numeric/mixed content
        if alpha_ratio < 0.20:
            return True

        # Count non-standard characters (not in expected charset)
        expected_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-()[]")
        nonstandard = sum(1 for c in text if c not in expected_chars)

        # Reject if too many non-standard characters
        # Relaxed from 0.3 to 0.4 to allow more special characters
        if nonstandard > 0.4 * len(text):
            return True

        # Check for reasonable word structure
        words = text.split()
        if not words:
            return True

        # Most words should have vowels (German/English)
        vowels = set("aeiouäöüAEIOUÄÖÜ")
        words_with_vowels = sum(1 for word in words if any(c in vowels for c in word) and len(word) > 1)

        # At least 15% of multi-char words should have vowels (relaxed from 30%)
        multi_char_words = sum(1 for word in words if len(word) > 1)
        if multi_char_words > 0 and words_with_vowels < 0.15 * multi_char_words:
            return True

        # Check for excessive repetition (sign of bad OCR)
        unique_chars = len(set(text.replace(" ", "")))
        if unique_chars < len(text) * 0.1:  # Less than 10% unique chars
            return True

        return False

    def _choose_psm_for_box(self, w, h):
        """Choose optimal PSM based on ROI dimensions and expected content"""
        aspect_ratio = w / max(h, 1)

        # Very wide, short boxes are likely single lines (timestamps, IDs)
        if h < 50 or aspect_ratio > 8:
            return tesserocr.PSM.SINGLE_LINE

        # Tall narrow boxes might be vertical text or single words
        elif aspect_ratio < 2 and h < 100:
            return tesserocr.PSM.SINGLE_WORD

        # Medium boxes are likely single blocks of text
        elif h < 200:
            return tesserocr.PSM.SINGLE_BLOCK

        # Large boxes might have sparse text (device overlays)
        return tesserocr.PSM.SPARSE_TEXT

    # ---------------- Preprocessing ----------------
    def _preprocess_to_gray(self, frame, roi=None):
        if frame.ndim == 3:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            img = frame

        if roi and self._validate_roi(roi):
            x, y, w, h = map(int, (roi["x"], roi["y"], roi["width"], roi["height"]))
            img = img[y : y + h, x : x + w]
            logger.debug(f"ROI: {x},{y},{w},{h} (frame size now {img.shape})")
        pil_image = Image.fromarray(img)
        if min(pil_image.size) < 400:
            pil_image = pil_image.resize((int(pil_image.width * 2), int(pil_image.height * 2)), Image.Resampling.LANCZOS)

        gray = np.array(pil_image.convert("L"))

        # Enhanced preprocessing for video frames
        # 1. Denoise to remove video compression artifacts
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        # 2. CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. Sharpen text edges (helps with blurry video text)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

        # 4. Detect if we have white-on-black text (common in medical overlays)
        # Calculate mean brightness to determine if inversion is needed
        mean_brightness = np.mean(sharpened)
        is_dark_background = mean_brightness < 127  # More dark pixels than light

        if is_dark_background:
            # Invert for white-on-black text (makes it black-on-white for Tesseract)
            logger.debug("Detected dark background, inverting image for OCR")
            sharpened = cv2.bitwise_not(sharpened)

        # 5. Adaptive thresholding for varying lighting
        # Use THRESH_BINARY (not INV) since we already inverted if needed
        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # Changed from THRESH_BINARY_INV
            blockSize=21,
            C=8,
        )

        # 6. Morphological cleaning to connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary

    # ---------------- Text region detection ----------------
    def _detect_text_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect likely text regions in a grayscale/binary image.
        Returns list of (x, y, w, h) bounding boxes sorted top-to-bottom, left-to-right.
        """
        try:
            # Mild gradient to highlight character edges
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

            # Otsu on gradient
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Bridge gaps between characters into words/lines
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rect_kernel)

            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rois: List[Tuple[int, int, int, int]] = []
            H, W = gray.shape
            max_area = self._max_area_ratio * (W * H)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < self._roi_min_area or area > max_area or h < self._roi_min_height:
                    continue
                aspect = w / (h + 1e-3)
                if not (self._roi_aspect_min <= aspect <= self._roi_aspect_max):
                    continue
                # Optional padding to avoid clipping glyphs
                pad = 2
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(W, x + w + pad)
                y1 = min(H, y + h + pad)
                rois.append((x0, y0, x1 - x0, y1 - y0))

            rois.sort(key=lambda b: (b[1], b[0]))
            return rois
        except Exception as e:
            logger.warning("Text region detection failed: %s", e)
            return []

    # ---------------- OCR core ----------------
    def extract_text_from_frame(self, frame: np.ndarray, roi: Optional[Dict[str, Any]] = None, high_quality: bool = True) -> Tuple[str, float, Dict[str, Any]]:
        if not self.api:
            logger.error("TesseOCR API not initialized.")
            return "", 0.0, {}

        t0 = time.time()
        with self._lock:
            try:
                gray = self._preprocess_to_gray(frame, roi)

                # Choose DPI per mode
                dpi = self.frame_config["high_quality_dpi" if high_quality else "dpi"]
                self.api.SetVariable("user_defined_dpi", str(dpi))

                # Detect candidate text boxes

                if not roi:
                    regions = self._detect_text_regions(gray)
                    has_roi = False
                else:
                    regions = []
                    has_roi = True
                text_parts: List[str] = []
                if has_roi:
                    # if rois are present: OCR whole image, image was already cropped
                    self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
                    self.api.SetImage(Image.fromarray(gray))
                    txt = (self.api.GetUTF8Text() or "").strip()
                    if txt:
                        text_parts.append(txt)
                else:
                    # OCR each detected region, if any
                    for x, y, w, h in regions:
                        sub = gray[y : y + h, x : x + w]
                        self.api.SetPageSegMode(self._choose_psm_for_box(w, h))
                        self.api.SetImage(Image.fromarray(sub))
                        part = (self.api.GetUTF8Text() or "").strip()
                        if part:
                            text_parts.append(part)

                # Combine & normalize
                text = " ".join(text_parts)
                text = unicodedata.normalize("NFC", text)
                text = re.sub(r"[^\w\s.,:;/-ÄÖÜäöüß]", "", text)

                # Filter gibberish early
                if self._is_gibberish(text):
                    logger.debug(f"Detected gibberish, filtering out: {text[:100]}")
                    text = ""
                    conf = 0.0

                # Confidence (from last call). If multiple regions, MeanTextConf reflects last SetImage call.
                # We approximate by re-measuring on the full image if we produced text from >1 region.
                if len(regions) > 1 and text:
                    self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
                    self.api.SetImage(Image.fromarray(gray))
                conf = max(self.api.MeanTextConf(), 0) / 100.0

                # ---------------- Adaptive fallback ----------------
                if (not text or conf < 0.3) and regions:
                    # Choose largest detected region by area
                    largest = max(regions, key=lambda b: b[2] * b[3])
                    x, y, w, h = largest
                    sub = gray[y : y + h, x : x + w]

                    # Retry with high DPI and tighter PSM
                    logger.debug(f"Low confidence ({conf:.2f}) → retrying largest region {w}x{h} with SINGLE_BLOCK @600 dpi")
                    self.api.SetVariable("user_defined_dpi", "600")
                    self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
                    self.api.SetImage(Image.fromarray(sub))

                    retry_text = (self.api.GetUTF8Text() or "").strip()
                    retry_conf = max(self.api.MeanTextConf(), 0, 3) / 100.0

                    # Use the retry result only if it improves confidence
                    if retry_conf > conf and not self._is_gibberish(retry_text):
                        text, conf = retry_text, retry_conf
                        meta_source = "retry"
                    else:
                        meta_source = "initial"
                else:
                    meta_source = "initial"

                # Normalize and clean text
                text = unicodedata.normalize("NFC", text)
                text = re.sub(r"[^\w\s.,:;/-ÄÖÜäöüß]", "", text)
                texts, confs = [], []
                for x, y, w, h in regions:
                    sub = gray[y : y + h, x : x + w]
                    self.api.SetPageSegMode(self._choose_psm_for_box(w, h))
                    self.api.SetImage(Image.fromarray(sub))
                    t = (self.api.GetUTF8Text() or "").strip()
                    c = max(self.api.MeanTextConf(), 0) / 100.0
                    texts.append(t)
                    confs.append(c)
                # visualize_ocr_regions(gray, regions, texts, confs, title=f"Frame debug ({len(regions)} regions)")

                # Package metadata
                meta = {
                    "processing_time": time.time() - t0,
                    "dpi": dpi,
                    "regions": len(regions),
                    "psm": int(self.api.GetPageSegMode()),
                    "confidence": conf,
                    "method": f"tesserocr+regions({meta_source})",
                }

                # Optional: HOCR metadata (disabled by default for speed)
                # try:
                #     self.api.SetVariable("tessedit_create_hocr", "1")
                #     hocr = self.api.GetHOCRText(0)
                #     if "<span" in hocr:
                #         meta["hocr_len"] = len(hocr)
                # except Exception:
                #     pass

                self.processed_frames += 1
                self.total_processing_time += meta["processing_time"]
                return text, conf, meta

            except Exception as e:
                logger.error("TesseOCR failed: %s", e)
                return "", 0.0, {}

    # ---------------- Batch ----------------
    def extract_text_from_frame_batch(
        self, frames: List[np.ndarray], roi: Optional[Dict[str, Any]] = None, high_quality: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        t0 = time.time()
        for i, f in enumerate(frames):
            out.append(self.extract_text_from_frame(f, roi, high_quality))
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                logger.debug("Processed %d/%d frames, avg=%.3fs/frame", i + 1, len(frames), elapsed / (i + 1))
        logger.info("Batch complete: %d frames in %.2fs", len(frames), time.time() - t0)
        return out

    # ---------------- Stats ----------------
    def get_performance_stats(self) -> Dict[str, float]:
        if self.processed_frames == 0:
            return {"avg_time": 0.0, "frames": 0, "fps": 0.0}
        t = self.total_processing_time
        return {
            "avg_time": t / self.processed_frames,
            "frames": self.processed_frames,
            "fps": self.processed_frames / t if t > 0 else 0.0,
        }

    def reset_performance_stats(self) -> None:
        self.processed_frames = 0
        self.total_processing_time = 0.0


# ---------- Public helpers ----------
def get_tesseocr_processor(language: str = "deu"):
    with _GLOBAL_LOCK:
        proc = _GLOBAL_PROCESSORS.get(language)
        if proc is None:
            proc = TesseOCRFrameProcessor(language)
            _GLOBAL_PROCESSORS[language] = proc
        return proc


def extract_text_from_frame_fast(
    frame: np.ndarray, roi: Optional[Dict[str, Any]] = None, high_quality: bool = True, language: str = "deu"
) -> Tuple[str, float, Dict[str, Any]]:
    proc = get_tesseocr_processor(language)
    return proc.extract_text_from_frame(frame, roi, high_quality)
