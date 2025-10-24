"""
High-performance Frame-specific OCR module using Tesserocr for medical video anonymization.

New: Region-first OCR
- Detect text-like regions with OpenCV (morph gradient + close + contour filter)
- OCR only those ROIs; fallback to full-frame if none found
- Grayscale-safe, robust preprocessing, dynamic inversion
- German-biased decoding; optional anti-gibberish filter
"""

import logging
import cv2
import numpy as np
import os
import tesserocr
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List
import re
import threading
import time
import glob
import unicodedata

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
            "min_confidence": 20,
            "high_quality_min_confidence": 30,
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
        if not text or len(text) < 8:
            return True
        alpha = sum(c.isalpha() for c in text)
        ratio = alpha / max(len(text), 1)
        nonlatin = len(re.findall(r"[^A-Za-zÄÖÜäöüß0-9\s.,:;/-]", text))
        return ratio < 0.3 or nonlatin > 0.2 * len(text)

    def _choose_psm_for_box(self, w, h):
        if h < 80:
            return tesserocr.PSM.SINGLE_LINE
        elif h < 200:
            return tesserocr.PSM.SINGLE_BLOCK
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
        gray = cv2.bilateralFilter(gray, 3, 50, 50)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5)
        kernel = np.ones((2, 2), np.uint8)
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
