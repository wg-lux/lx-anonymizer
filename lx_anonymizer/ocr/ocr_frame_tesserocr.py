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
import tesserocr  # type: ignore[import-untyped]
from PIL import Image

from lx_anonymizer._native import native as _native

logger = logging.getLogger(__name__)


_EXPECTED_OCR_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-()[]"
)
_OCR_VOWELS = set("aeiouäöüAEIOUÄÖÜ")


def _py_is_gibberish(text: str) -> bool:
    """Enhanced gibberish detection for video OCR with support for structured data."""
    if not text or len(text) < 3:
        return True

    time_pattern = r"\d{1,2}:\d{2}(?::\d{2})?"
    date_pattern = r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}[-./]\d{4}"
    case_pattern = r"[A-Z]\s*\d{4,}/\d{4}"
    compact_code_pattern = r"\b[A-Z]\s*\d{5,}\b|\b[A-Z]\d{5,}\b"
    device_pattern = r"\d{8,}"
    ratio_pattern = r"\b\d+(?:[.,]\d+)?/\d+(?:[.,]\d+)?\b"

    if (
        re.search(time_pattern, text)
        or re.search(date_pattern, text)
        or re.search(case_pattern, text)
        or re.search(compact_code_pattern, text)
        or re.search(device_pattern, text)
        or re.search(ratio_pattern, text)
    ):
        return False

    alpha = sum(c.isalpha() for c in text)
    alpha_ratio = alpha / max(len(text), 1)
    if alpha_ratio < 0.20:
        return True

    nonstandard = sum(1 for c in text if c not in _EXPECTED_OCR_CHARS)
    if nonstandard > 0.4 * len(text):
        return True

    words = text.split()
    if not words:
        return True

    words_with_vowels = sum(
        1 for word in words if any(c in _OCR_VOWELS for c in word) and len(word) > 1
    )
    multi_char_words = sum(1 for word in words if len(word) > 1)
    if multi_char_words > 0 and words_with_vowels < 0.15 * multi_char_words:
        return True

    unique_chars = len(set(text.replace(" ", "")))
    if unique_chars < len(text) * 0.1:
        return True

    return False


def _py_gibberish_score(text: str) -> float:
    if not text:
        return 1.0

    score = 0.0
    length = max(len(text), 1)

    alpha_ratio = sum(c.isalpha() for c in text) / length
    if alpha_ratio < 0.2:
        score += 0.35
    elif alpha_ratio < 0.35:
        score += 0.15

    nonstandard_ratio = sum(1 for c in text if c not in _EXPECTED_OCR_CHARS) / length
    score += min(nonstandard_ratio * 0.8, 0.4)

    punct_like = sum(
        1 for c in text if not c.isalnum() and not c.isspace() and c not in ".,:;/-"
    )
    score += min((punct_like / length) * 0.6, 0.2)

    words = [w for w in text.split() if len(w) > 1]
    if words:
        vowel_words = sum(1 for w in words if any(c in _OCR_VOWELS for c in w))
        ratio = vowel_words / len(words)
        if ratio < 0.15:
            score += 0.25
        elif ratio < 0.3:
            score += 0.1

    return max(0.0, min(score, 1.0))


def _py_looks_structured_overlay_text(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"\d{1,2}:\d{2}(?::\d{2})?",
        r"\b[A-Z]\s*\d{5,}\b|\b[A-Z]\d{5,}\b",
        r"\b\d+(?:[.,]\d+)?/\d+(?:[.,]\d+)?\b",
        r"\b\d{4,}\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _py_normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s.,:;/-ÄÖÜäöüß]", "", text)
    text = re.sub(r"([.,:;])\1{1,}", r"\1", text)
    text = re.sub(r"[.,:;]{2,}", lambda m: m.group(0)[0], text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _py_candidate_rank(text: str, conf: float) -> Tuple[int, int, float, float, int]:
    is_empty = 0 if text else 1
    is_gib = 1 if (text and _py_is_gibberish(text)) else 0
    gib_score = _py_gibberish_score(text)
    return (is_empty, is_gib, gib_score, -float(conf), -len(text or ""))


def _is_gibberish_impl(text: str) -> bool:
    if _native is not None:
        return _native.is_gibberish(text)
    return _py_is_gibberish(text)


def _gibberish_score_impl(text: str) -> float:
    if _native is not None:
        return _native.gibberish_score(text)
    return _py_gibberish_score(text)


def _looks_structured_overlay_text_impl(text: str) -> bool:
    if _native is not None:
        return _native.looks_structured_overlay_text(text)
    return _py_looks_structured_overlay_text(text)


def _normalize_ocr_text_impl(text: str) -> str:
    if _native is not None:
        return _native.normalize_ocr_text(text)
    return _py_normalize_ocr_text(text)


def _candidate_rank_impl(text: str, conf: float) -> Tuple[int, int, float, float, int]:
    if _native is not None:
        return _native.candidate_rank(text, conf)
    return _py_candidate_rank(text, conf)


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
        self.api: Any = None
        self.api_lstm: Any = None
        self._tessdata_path: Optional[str] = None
        self._default_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÄÖÜäöüß0123456789 .,:;/-"
        self._numeric_overlay_whitelist = "0123456789 .,:;/-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
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
        self._tessdata_path = tessdata_path
        self.api = tesserocr.PyTessBaseAPI(
            lang=self.language,  # German prioritized
            path=tessdata_path,
            oem=tesserocr.OEM.DEFAULT,  # allow legacy fallback for tricky fonts
        )
        self._configure_api_common(self.api)

        try:
            self.api_lstm = tesserocr.PyTessBaseAPI(
                lang=self.language,
                path=tessdata_path,
                oem=tesserocr.OEM.LSTM_ONLY,
            )
            self._configure_api_common(self.api_lstm)
            logger.info(
                "TesseOCR alternate API initialized (OEM=LSTM_ONLY, lang=%s)",
                self.language,
            )
        except Exception as e:
            logger.warning("Could not initialize LSTM_ONLY fallback API: %s", e)
            self.api_lstm = None

        logger.info(
            "TesseOCR initialized (OEM=DEFAULT, PSM=SINGLE_BLOCK, lang=%s, path=%s)",
            self.language,
            tessdata_path,
        )

    def _configure_api_common(self, api: tesserocr.PyTessBaseAPI) -> None:
        # Default PSM: single block (we switch per-ROI later)
        api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)

        # Performance & stability
        api.SetVariable("preserve_interword_spaces", "1")
        api.SetVariable("user_defined_dpi", "400")
        api.SetVariable("classify_enable_learning", "0")
        api.SetVariable("tessedit_do_invert", "0")
        api.SetVariable("tessedit_enable_doc_dict", "0")
        api.SetVariable("load_system_dawg", "0")
        api.SetVariable("load_freq_dawg", "1")
        api.SetVariable("textord_really_old_xheight", "1")  # small caps overlays

        # German bias & safe charset
        api.SetVariable("language_model_penalty_non_dict_word", "1")
        api.SetVariable("language_model_penalty_non_freq_dict_word", "1")
        api.SetVariable("tessedit_char_whitelist", self._default_whitelist)

    # ---------------- tessdata discovery ----------------
    def _get_tessdata_path(self) -> Optional[str]:
        """
        Return tessdata/ directory itself (PyTessBaseAPI expects the dir containing *.traineddata).
        """
        env_tessdata_parent = os.environ.get("TESSDATA_PREFIX")
        if env_tessdata_parent:
            if env_tessdata_parent.endswith("/tessdata") and os.path.isdir(
                env_tessdata_parent
            ):
                logger.info("Using TESSDATA_PREFIX directly: %s", env_tessdata_parent)
                return env_tessdata_parent
            tessdata_dir = os.path.join(env_tessdata_parent, "tessdata")
            if os.path.isdir(tessdata_dir):
                logger.info(
                    "Using tessdata from TESSDATA_PREFIX parent: %s", tessdata_dir
                )
                return tessdata_dir

        nix_patterns = [
            "/nix/store/*/share/tessdata",
            "/run/current-system/sw/share/tessdata",
        ]
        for pattern in nix_patterns:
            if "*" in pattern:
                for candidate in glob.glob(pattern):
                    if os.path.isdir(candidate):
                        if any(
                            f.endswith(".traineddata") for f in os.listdir(candidate)
                        ):
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
        return _is_gibberish_impl(text)

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

    @staticmethod
    def _gibberish_score(text: str) -> float:
        return _gibberish_score_impl(text)

    @staticmethod
    def _looks_structured_overlay_text(text: str) -> bool:
        return _looks_structured_overlay_text_impl(text)

    def _candidate_rank(
        self, text: str, conf: float
    ) -> Tuple[int, int, float, float, int]:
        return _candidate_rank_impl(text, conf)

    @staticmethod
    def _normalize_ocr_text(text: str) -> str:
        return _normalize_ocr_text_impl(text)

    def _choose_whitelist_for_box(self, w: int, h: int) -> str:
        """
        Use a stricter whitelist for obvious numeric overlays (timestamps/IDs).
        """
        aspect_ratio = w / max(h, 1)
        if h < 60 or aspect_ratio > 7:
            return self._numeric_overlay_whitelist
        return self._default_whitelist

    # ---------------- Preprocessing ----------------
    def _preprocess_to_gray(self, frame, roi=None, mode: str = "binary"):
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
            pil_image = pil_image.resize(
                (int(pil_image.width * 2), int(pil_image.height * 2)),
                Image.Resampling.LANCZOS,
            )

        gray = np.array(pil_image.convert("L"))

        # Enhanced preprocessing for video frames
        # 1. Denoise to remove video compression artifacts
        denoised = cv2.fastNlMeansDenoising(
            gray, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # 2. CLAHE for adaptive contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 4. Detect if we have white-on-black text (common in medical overlays)
        # Calculate mean brightness to determine if inversion is needed
        mean_brightness = np.mean(enhanced)
        is_dark_background = mean_brightness < 127  # More dark pixels than light

        if is_dark_background:
            # Invert for white-on-black text (makes it black-on-white for Tesseract)
            logger.debug("Detected dark background, inverting image for OCR")
            enhanced = cv2.bitwise_not(enhanced)

        if mode == "gray_clahe":
            return enhanced

        # 3b. Sharpen only for binary path (can introduce halos in grayscale OCR path)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

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

    def _ocr_processed_image(
        self, processed: np.ndarray, has_roi: bool, dpi: int
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        OCR a preprocessed frame/ROI image and return text, confidence, and metadata.
        """
        self.api.SetVariable("user_defined_dpi", str(dpi))

        if not has_roi:
            regions = self._detect_text_regions(processed)
        else:
            regions = []

        text_parts: List[str] = []
        accepted_regions = 0
        rejected_regions = 0
        best_region_text = ""
        best_region_conf = 0.0
        best_region_rank: Optional[Tuple[int, int, float, float, int]] = None
        if has_roi:
            h, w = processed.shape[:2]
            self.api.SetVariable(
                "tessedit_char_whitelist", self._choose_whitelist_for_box(w, h)
            )
            self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
            self.api.SetImage(Image.fromarray(processed))
            txt = (self.api.GetUTF8Text() or "").strip()
            if txt:
                text_parts.append(self._normalize_ocr_text(txt))
        else:
            for x, y, w, h in regions:
                sub = processed[y : y + h, x : x + w]
                self.api.SetVariable(
                    "tessedit_char_whitelist", self._choose_whitelist_for_box(w, h)
                )
                self.api.SetPageSegMode(self._choose_psm_for_box(w, h))
                self.api.SetImage(Image.fromarray(sub))
                part = (self.api.GetUTF8Text() or "").strip()
                part_conf = max(self.api.MeanTextConf(), 0) / 100.0
                if not part:
                    rejected_regions += 1
                    continue

                part = self._normalize_ocr_text(part)
                if not part:
                    rejected_regions += 1
                    continue

                part_rank = self._candidate_rank(part, part_conf)
                if best_region_rank is None or part_rank < best_region_rank:
                    best_region_rank = part_rank
                    best_region_text = part
                    best_region_conf = part_conf

                # Filter noisy regions early so one bad ROI doesn't poison merged text.
                structured_part = self._looks_structured_overlay_text(part)
                low_value_region = (
                    self._is_gibberish(part)
                    and not structured_part
                    and part_conf < 0.45
                    and len(part) < 48
                )
                if low_value_region:
                    rejected_regions += 1
                    logger.debug(
                        "Rejecting ROI OCR part (conf=%.2f, %sx%s): %s",
                        part_conf,
                        w,
                        h,
                        part[:80],
                    )
                    continue

                text_parts.append(part)
                accepted_regions += 1
                logger.debug(
                    "Accepted ROI OCR part (conf=%.2f, %sx%s): %s",
                    part_conf,
                    w,
                    h,
                    part[:80],
                )

        # Safety net: avoid dropping all OCR output when ROI filtering is too strict.
        if not has_roi and accepted_regions == 0 and best_region_text:
            text_parts = [best_region_text]
            accepted_regions = 1
            logger.debug(
                "All ROI parts were rejected; keeping best ROI candidate as safety net "
                "(conf=%.2f): %s",
                best_region_conf,
                best_region_text[:120],
            )

        text = " ".join(text_parts)
        text = self._normalize_ocr_text(text)

        if self._is_gibberish(text):
            logger.debug("Detected gibberish, filtering out: %s", text[:100])
            text = ""

        if text:
            if len(regions) > 1:
                self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
                self.api.SetImage(Image.fromarray(processed))
            conf = max(self.api.MeanTextConf(), 0) / 100.0
        else:
            conf = 0.0

        # Second safety net: merged text may be blanked as gibberish even when some ROIs were accepted.
        # In that case, keep the best normalized ROI candidate to avoid total text loss.
        if not text and best_region_text:
            text = best_region_text
            conf = best_region_conf
            logger.debug(
                "Merged OCR text was blank after filtering; restoring best ROI candidate "
                "(accepted=%d rejected=%d conf=%.2f): %s",
                accepted_regions,
                rejected_regions,
                best_region_conf,
                best_region_text[:120],
            )

        # If region-wise filtering removed too much, prefer the best single region candidate.
        if (not text or conf < 0.2) and best_region_text and best_region_conf >= conf:
            if (
                not self._is_gibberish(best_region_text)
                or best_region_conf >= 0.55
                or self._looks_structured_overlay_text(best_region_text)
            ):
                text = best_region_text
                conf = best_region_conf

        # ---------------- Adaptive fallback ----------------
        if (not text or conf < 0.3) and regions:
            largest = max(regions, key=lambda b: b[2] * b[3])
            x, y, w, h = largest
            sub = processed[y : y + h, x : x + w]

            logger.debug(
                "Low confidence (%.2f) -> retrying largest region %sx%s with SINGLE_BLOCK @600 dpi",
                conf,
                w,
                h,
            )
            self.api.SetVariable("user_defined_dpi", "600")
            self.api.SetVariable(
                "tessedit_char_whitelist", self._choose_whitelist_for_box(w, h)
            )
            self.api.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
            self.api.SetImage(Image.fromarray(sub))

            retry_text = (self.api.GetUTF8Text() or "").strip()
            retry_text = self._normalize_ocr_text(retry_text)
            retry_conf = max(self.api.MeanTextConf(), 0) / 100.0

            if retry_conf > conf and not self._is_gibberish(retry_text):
                text, conf = retry_text, retry_conf
                meta_source = "retry"
            else:
                meta_source = "initial"
                # Optional alternate OEM retry for hard overlay text (low-risk, only on weak ROI retries).
                if self.api_lstm is not None and (retry_conf < 0.6 or not retry_text):
                    try:
                        self.api_lstm.SetVariable("user_defined_dpi", "600")
                        self.api_lstm.SetVariable(
                            "tessedit_char_whitelist",
                            self._choose_whitelist_for_box(w, h),
                        )
                        self.api_lstm.SetPageSegMode(tesserocr.PSM.SINGLE_BLOCK)
                        self.api_lstm.SetImage(Image.fromarray(sub))
                        alt_text = self._normalize_ocr_text(
                            (self.api_lstm.GetUTF8Text() or "").strip()
                        )
                        alt_conf = max(self.api_lstm.MeanTextConf(), 0) / 100.0
                        self.api_lstm.SetVariable(
                            "tessedit_char_whitelist", self._default_whitelist
                        )

                        alt_better = self._candidate_rank(
                            alt_text, alt_conf
                        ) < self._candidate_rank(text, conf)
                        if alt_better and (
                            alt_text
                            and (
                                not self._is_gibberish(alt_text)
                                or alt_conf >= 0.55
                                or self._looks_structured_overlay_text(alt_text)
                            )
                        ):
                            text, conf = alt_text, alt_conf
                            meta_source = "retry_lstm"
                            logger.debug(
                                "LSTM_ONLY retry improved ROI OCR (conf %.2f -> %.2f): %s",
                                retry_conf,
                                alt_conf,
                                alt_text[:80],
                            )
                    except Exception as e:
                        logger.debug("LSTM_ONLY retry failed: %s", e)
        else:
            meta_source = "initial"

        text = self._normalize_ocr_text(text)
        # Restore default whitelist for subsequent calls.
        self.api.SetVariable("tessedit_char_whitelist", self._default_whitelist)

        meta = {
            "dpi": dpi,
            "regions": len(regions),
            "regions_accepted": accepted_regions,
            "regions_rejected": rejected_regions,
            "psm": int(self.api.GetPageSegMode()),
            "confidence": conf,
            "gibberish_score": self._gibberish_score(text),
            "method": f"tesserocr+regions({meta_source})",
        }
        logger.debug(
            "OCR summary: regions=%d accepted=%d rejected=%d text_len=%d conf=%.2f method=%s",
            len(regions),
            accepted_regions,
            rejected_regions,
            len(text or ""),
            conf,
            meta["method"],
        )
        return text, conf, meta

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

            contours, _ = cv2.findContours(
                connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            rois: List[Tuple[int, int, int, int]] = []
            H, W = gray.shape
            max_area = self._max_area_ratio * (W * H)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if (
                    area < self._roi_min_area
                    or area > max_area
                    or h < self._roi_min_height
                ):
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
    def extract_text_from_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True,
    ) -> Tuple[str, float, Dict[str, Any]]:
        if not self.api:
            logger.error("TesseOCR API not initialized.")
            return "", 0.0, {}

        t0 = time.time()
        with self._lock:
            try:
                # Choose DPI per mode
                dpi = self.frame_config["high_quality_dpi" if high_quality else "dpi"]
                has_roi = bool(roi)
                candidates: List[Tuple[str, str, float, Dict[str, Any]]] = []
                for preprocess_mode in ("gray_clahe", "binary"):
                    processed = self._preprocess_to_gray(
                        frame, roi, mode=preprocess_mode
                    )
                    c_text, c_conf, c_meta = self._ocr_processed_image(
                        processed, has_roi=has_roi, dpi=dpi
                    )
                    c_meta["preprocessing"] = preprocess_mode
                    candidates.append((preprocess_mode, c_text, c_conf, c_meta))

                best = min(candidates, key=lambda c: self._candidate_rank(c[1], c[2]))
                _, text, conf, meta = best
                meta["processing_time"] = time.time() - t0

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
        self,
        frames: List[np.ndarray],
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        t0 = time.time()
        for i, f in enumerate(frames):
            out.append(self.extract_text_from_frame(f, roi, high_quality))
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                logger.debug(
                    "Processed %d/%d frames, avg=%.3fs/frame",
                    i + 1,
                    len(frames),
                    elapsed / (i + 1),
                )
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
    frame: np.ndarray,
    roi: Optional[Dict[str, Any]] = None,
    high_quality: bool = True,
    language: str = "deu",
) -> Tuple[str, float, Dict[str, Any]]:
    proc = get_tesseocr_processor(language)
    return proc.extract_text_from_frame(frame, roi, high_quality)
