"""
Enhanced FrameOCR with diagnostic capabilities for gibberish text detection.
"""
import os
import subprocess
import json
import logging
import unicodedata
import re
import cv2
import numpy as np
import pytesseract
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class DiagnosticFrameOCR:
    """Enhanced FrameOCR with diagnostic capabilities."""
    
    def __init__(self, enable_diagnostics: bool = False, diag_dir: Optional[str] = None):
        """Initialize with optional diagnostic mode."""
        self.enable_diagnostics = enable_diagnostics or bool(os.getenv("OCR_DIAG", "0"))
        self.diag_dir = Path(diag_dir or os.getenv("OCR_DIAG_DIR", "./debug/ocr"))
        
        if self.enable_diagnostics:
            self.diag_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"OCR Diagnostics enabled, output: {self.diag_dir}")
            self._check_tesseract_setup()
        
        # Enhanced OCR configuration with robust defaults
        self.ocr_config = self._get_robust_ocr_config()
        self.frame_count = 0
        self.diag_samples = []
    
    def _get_robust_ocr_config(self) -> Dict[str, Any]:
        """Get robust OCR configuration based on environment or defaults."""
        return {
            'lang': os.getenv('OCR_LANG', 'deu+eng'),
            'psm': int(os.getenv('OCR_PSM', '6')),
            'oem': int(os.getenv('OCR_OEM', '3')),
            'dpi': int(os.getenv('OCR_DPI', '300')),
        }
    
    def _check_tesseract_setup(self):
        """Check and log Tesseract setup for diagnostics."""
        try:
            version_output = subprocess.check_output(
                ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
            )
            logger.info(f"Tesseract version: {version_output.strip().split()[1]}")
            
            langs_output = subprocess.check_output(
                ["tesseract", "--list-langs"], stderr=subprocess.STDOUT, text=True
            )
            langs = langs_output.strip().split('\n')[1:]
            has_deu = 'deu' in langs
            has_eng = 'eng' in langs
            
            logger.info(f"Languages: {len(langs)} (deu: {has_deu}, eng: {has_eng})")
            
            if not has_deu:
                logger.warning("German language pack (deu) not available!")
            if not has_eng:
                logger.warning("English language pack (eng) not available!")
                
        except Exception as e:
            logger.error(f"Tesseract setup check failed: {e}")
    
    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze OCR text quality and detect gibberish patterns."""
        if not text:
            return {
                "len": 0, "letters": 0, "digits": 0, "punct": 0,
                "punct_ratio": 0.0, "words": 0, "readable_words": 0,
                "is_gibberish": True, "gibberish_score": 1.0
            }
        
        # Normalize Unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Character analysis
        punct_chars = "!@#$%^&*()_+{}|:\"<>?`~[]\\;',./§°^"
        special_chars = "‚''–—•…"
        all_punct = punct_chars + special_chars
        
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        punct = sum(1 for ch in text if ch in all_punct)
        whitespace = sum(1 for ch in text if ch.isspace())
        
        # Word analysis
        words = re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", text)
        readable_words = [w for w in words if len(w) >= 3 and not all(c == w[0] for c in w)]
        
        # Gibberish detection
        punct_ratio = punct / max(1, len(text))
        repeated_chars = len(re.findall(r"(.)\1{3,}", text))
        special_symbols = sum(1 for ch in text if ord(ch) > 127 and not ch.isalpha())
        
        # Calculate gibberish score
        gibberish_score = 0.0
        if punct_ratio > 0.5:
            gibberish_score += 0.4
        if len(readable_words) == 0:
            gibberish_score += 0.3
        if repeated_chars > 2:
            gibberish_score += 0.2
        if special_symbols > len(text) * 0.2:
            gibberish_score += 0.1
        
        is_gibberish = gibberish_score > 0.6 or punct_ratio > 0.7
        
        return {
            "len": len(text),
            "letters": letters,
            "digits": digits,
            "punct": punct,
            "whitespace": whitespace,
            "punct_ratio": punct_ratio,
            "words": len(words),
            "readable_words": len(readable_words),
            "repeated_chars": repeated_chars,
            "special_symbols": special_symbols,
            "is_gibberish": is_gibberish,
            "gibberish_score": gibberish_score
        }
    
    def _preprocess_for_ocr(self, frame: np.ndarray, roi: Optional[Dict] = None) -> np.ndarray:
        """Improved preprocessing with reduced aggressive filtering."""
        try:
            # Apply ROI if specified
            if roi and self._validate_roi(roi):
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                frame = frame[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Gentle enhancement instead of aggressive processing
            if os.getenv('OCR_FIX_V1', '0') == '1':
                # Improved preprocessing based on diagnostic results
                
                # Mild noise reduction
                denoised = cv2.bilateralFilter(gray, 9, 75, 75)
                
                # Contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                
                # Adaptive thresholding
                processed = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                if self.enable_diagnostics:
                    logger.debug("Using improved preprocessing (OCR_FIX_V1)")
                
            else:
                # Original processing
                processed = gray
            
            return processed
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def _validate_roi(self, roi: Dict[str, Any]) -> bool:
        """Validate ROI dictionary."""
        required_keys = ['x', 'y', 'width', 'height']
        if not all(key in roi for key in required_keys):
            return False
        
        try:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            return all(isinstance(val, (int, float)) and val >= 0 for val in [x, y, w, h])
        except (TypeError, ValueError):
            return False
    
    def extract_text_from_frame(
        self, 
        frame: np.ndarray, 
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True,
        frame_id: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Extract text with diagnostic capabilities and gibberish detection."""
        try:
            # Preprocess frame
            processed_frame = self._preprocess_for_ocr(frame, roi)
            
            # Build tesseract config
            config = self.ocr_config
            tesseract_config = f"--oem {config['oem']} --psm {config['psm']} --dpi {config['dpi']}"
            
            # Perform OCR
            ocr_data = pytesseract.image_to_data(
                processed_frame,
                lang=config['lang'],
                config=tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract and filter words
            words = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    conf = int(ocr_data['conf'][i])
                    if conf > 1:
                        words.append(word.strip())
                        confidences.append(conf)
            
            # Calculate metrics
            text = ' '.join(words)
            mean_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
            
            # Analyze text quality
            quality = self._analyze_text_quality(text)
            
            # Diagnostic logging
            if self.enable_diagnostics and frame_id is not None:
                self._log_diagnostic_sample(frame_id, processed_frame, text, mean_conf, quality, roi)
            
            # Warning for gibberish detection
            if quality['is_gibberish'] and quality['gibberish_score'] > 0.8:
                logger.warning(f"High gibberish score ({quality['gibberish_score']:.2f}) detected: {text[:50]}...")
                
                # Try alternative OCR config for gibberish cases
                if os.getenv('OCR_FIX_V1', '0') == '1':
                    alternative_text, alternative_conf = self._try_alternative_ocr(processed_frame)
                    alt_quality = self._analyze_text_quality(alternative_text)
                    
                    if alt_quality['gibberish_score'] < quality['gibberish_score']:
                        logger.info(f"Alternative OCR improved gibberish score: {alt_quality['gibberish_score']:.2f}")
                        text, mean_conf = alternative_text, alternative_conf
            
            logger.debug(f"OCR extracted {len(words)} words, conf: {mean_conf:.2f}, "
                        f"gibberish: {quality['gibberish_score']:.2f}")
            
            return text, mean_conf, ocr_data
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return "", 0.0, {}
    
    def _try_alternative_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Try alternative OCR configuration for better results."""
        try:
            # Try single line mode (PSM 7)
            alt_config = "--oem 3 --psm 7 --dpi 400"
            
            ocr_data = pytesseract.image_to_data(
                image,
                lang='deu+eng',
                config=alt_config,
                output_type=pytesseract.Output.DICT
            )
            
            words = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    conf = int(ocr_data['conf'][i])
                    if conf > 1:
                        words.append(word.strip())
                        confidences.append(conf)
            
            text = ' '.join(words)
            mean_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0
            
            return text, mean_conf
            
        except Exception as e:
            logger.error(f"Alternative OCR failed: {e}")
            return "", 0.0
    
    def _log_diagnostic_sample(self, frame_id: int, processed_frame: np.ndarray, 
                              text: str, confidence: float, quality: Dict, roi: Optional[Dict]):
        """Log diagnostic sample for analysis."""
        if len(self.diag_samples) >= 20:
            return
        
        sample_dir = self.diag_dir / f"frame_{frame_id:06d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Save processed frame
        cv2.imwrite(str(sample_dir / "processed.png"), processed_frame)
        
        # Log sample data
        sample = {
            "frame_id": frame_id,
            "text": text[:200],
            "confidence": confidence,
            "roi": str(roi) if roi else "None",
            **quality
        }
        
        self.diag_samples.append(sample)
        
        # Save as JSON
        with open(sample_dir / "analysis.json", 'w') as f:
            json.dump(sample, f, indent=2)
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get summary of diagnostic results."""
        if not self.diag_samples:
            return {"error": "No diagnostic samples collected"}
        
        total_samples = len(self.diag_samples)
        gibberish_samples = sum(1 for s in self.diag_samples if s['is_gibberish'])
        avg_confidence = sum(s['confidence'] for s in self.diag_samples) / total_samples
        avg_gibberish_score = sum(s['gibberish_score'] for s in self.diag_samples) / total_samples
        
        return {
            "total_samples": total_samples,
            "gibberish_samples": gibberish_samples,
            "gibberish_ratio": gibberish_samples / total_samples,
            "avg_confidence": avg_confidence,
            "avg_gibberish_score": avg_gibberish_score,
            "recommendation": self._get_recommendation(avg_confidence, avg_gibberish_score)
        }
    
    def _get_recommendation(self, avg_conf: float, avg_gibberish: float) -> str:
        """Get diagnostic recommendation based on results."""
        if avg_conf < 0.2:
            return "LOW_CONFIDENCE: Check ROI positioning and image quality"
        elif avg_gibberish > 0.7:
            return "HIGH_GIBBERISH: Enable OCR_FIX_V1 for improved preprocessing"
        elif avg_conf > 0.6 and avg_gibberish < 0.3:
            return "GOOD_QUALITY: Current configuration is working well"
        else:
            return "MODERATE_ISSUES: Consider adjusting OCR parameters"
