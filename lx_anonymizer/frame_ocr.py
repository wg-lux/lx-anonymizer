"""
Frame-specific OCR module for video processing.

This module provides specialized OCR functionality optimized for video frames:
- Higher quality OCR settings for small text detection
- Frame-specific preprocessing
- Medical text recognition patterns
- Optimized for endoscopy video overlays

Separated from PDF processing to maintain clean architecture.
"""

import logging
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FrameOCR:
    """
    Specialized OCR processor for video frames with medical text detection.
    
    Optimized for:
    - Small overlay text in endoscopy videos
    - Patient information detection
    - High confidence text extraction
    - German medical terminology
    """
    
    def __init__(self):
        # Frame-specific OCR configuration
        self.frame_ocr_config = {
            'lang': 'deu+eng',  # German + English for medical terms
            'oem': 3,           # Default OCR Engine Mode
            'psm': 6,           # Assume uniform block of text
            'dpi': 300,         # High DPI for better small text detection
        }
        
        # High-quality configuration for sensitive areas
        self.high_quality_config = {
            'lang': 'deu+eng',
            'oem': 3,
            'psm': 8,           # Single word mode for precise detection
            'dpi': 400,
            'custom_config': '--user-words /usr/share/tesseract-ocr/4.00/tessdata/configs/hocr'
        }
        
        # Medical text patterns for enhanced detection
        self.medical_patterns = [
            r'Patient[:\s]*([A-Za-zäöüÄÖÜß\s\-]+)',
            r'Pat[\.:\s]*([A-Za-zäöüÄÖÜß\s\-]+)',
            r'geb[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'Fall[nr]*[\.:\s]*(\d+)',
            r'Datum[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
        ]
    
    def preprocess_frame_for_ocr(self, frame: np.ndarray, roi: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Preprocess a video frame for optimal OCR results.
        
        Args:
            frame: Input frame as numpy array (BGR format from cv2)
            roi: Optional region of interest to focus OCR on
            
        Returns:
            Preprocessed PIL Image ready for OCR
        """
        try:
            # Convert BGR to RGB for PIL
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Apply ROI cropping if specified
            if roi and self._validate_roi(roi):
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                frame_rgb = frame_rgb[y:y+h, x:x+w]
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize for better OCR (upscale small text)
            original_size = pil_image.size
            scale_factor = max(2.0, 1200 / max(original_size))
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale
            pil_image = pil_image.convert('L')
            
            # Enhance contrast for text visibility
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.0)
            
            # Apply sharpening filter
            pil_image = pil_image.filter(ImageFilter.SHARPEN)
            
            # Threshold to clean binary image
            img_array = np.array(pil_image)
            _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Remove noise with morphological operations
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return Image.fromarray(cleaned)
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            # Fallback to simple conversion
            if len(frame.shape) == 3:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('L')
            else:
                return Image.fromarray(frame)
    
    def extract_text_from_frame(
        self, 
        frame: np.ndarray, 
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text from a video frame with confidence scoring.
        
        Args:
            frame: Input frame as numpy array
            roi: Optional region of interest for OCR
            high_quality: Use high-quality OCR settings
            
        Returns:
            Tuple of (extracted_text, confidence_score, ocr_data)
        """
        try:
            # Preprocess frame
            #processed_image = self.preprocess_frame_for_ocr(frame, roi)
            
            # Select OCR configuration
            config = self.high_quality_config if high_quality else self.frame_ocr_config
            
            # Build tesseract config string
            tesseract_config = f"--oem {config['oem']} --psm {config['psm']} --dpi {config['dpi']}"
            if 'custom_config' in config:
                tesseract_config += f" {config['custom_config']}"
            
            # Extract text with detailed data
            ocr_data = pytesseract.image_to_data(
                frame,
                lang=config['lang'],
                config=tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter and clean text
            words = []
            confidences = []
            
            for i, word in enumerate(ocr_data['text']):
                if word.strip():
                    confidence = int(ocr_data['conf'][i])
                    if confidence > 1:  # Higher threshold for frames
                        words.append(word.strip())
                        confidences.append(confidence)
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.0
            
            # Join words to form text
            extracted_text = ' '.join(words)
            
            logger.debug(f"Frame OCR extracted {len(words)} words with {avg_confidence:.2f} confidence")
            
            return extracted_text, avg_confidence, ocr_data
            
        except Exception as e:
            logger.error(f"Frame text extraction failed: {e}")
            return "", 0.0, {}
    
    def extract_medical_text_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extract medical information patterns from OCR text.
        
        Args:
            text: OCR-extracted text
            
        Returns:
            Dictionary with extracted medical information
        """
        import re
        
        extracted_info = {
            'patient_names': [],
            'dates': [],
            'case_numbers': [],
            'medical_terms': []
        }
        
        try:
            # Apply medical patterns
            for pattern in self.medical_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if 'Patient' in pattern or 'Pat' in pattern:
                        extracted_info['patient_names'].extend(matches)
                    elif 'geb' in pattern or 'Datum' in pattern:
                        extracted_info['dates'].extend(matches)
                    elif 'Fall' in pattern:
                        extracted_info['case_numbers'].extend(matches)
            
            # Clean and deduplicate
            for key in extracted_info:
                extracted_info[key] = list(set(extracted_info[key]))
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Medical pattern extraction failed: {e}")
            return extracted_info
    
    def detect_sensitive_content(
        self, 
        frame: np.ndarray, 
        roi: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Detect sensitive content in a video frame.
        
        Args:
            frame: Input frame as numpy array
            roi: Optional region of interest
            
        Returns:
            Tuple of (has_sensitive_content, ocr_text, extracted_patterns)
        """
        try:
            # Extract text from frame
            ocr_text, confidence, _ = self.extract_text_from_frame(frame, roi, high_quality=True)
            
            if not ocr_text or confidence < 0.3:
                return False, "", {}
            
            # Extract medical patterns
            patterns = self.extract_medical_text_patterns(ocr_text)
            
            # Check for sensitive content
            has_sensitive = any([
                patterns['patient_names'],
                patterns['dates'],
                patterns['case_numbers']
            ])
            
            if has_sensitive:
                logger.warning(f"Sensitive content detected in frame: {patterns}")
            
            return has_sensitive, ocr_text, patterns
            
        except Exception as e:
            logger.error(f"Sensitive content detection failed: {e}")
            return False, "", {}
    
    def _validate_roi(self, roi: Dict[str, Any]) -> bool:
        """Validate ROI parameters."""
        required_keys = ['x', 'y', 'width', 'height']
        if not all(key in roi for key in required_keys):
            return False
        
        try:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            return all(isinstance(v, (int, float)) and v >= 0 for v in [x, y, w, h]) and w > 0 and h > 0
        except (TypeError, ValueError):
            return False
    
    def process_frame_batch(
        self, 
        frames: list[np.ndarray], 
        roi: Optional[Dict[str, Any]] = None
    ) -> list[Tuple[str, float, bool]]:
        """
        Process multiple frames efficiently.
        
        Args:
            frames: List of frame arrays
            roi: Optional region of interest
            
        Returns:
            List of (text, confidence, has_sensitive) tuples
        """
        results = []
        
        for i, frame in enumerate(frames):
            try:
                has_sensitive, text, _ = self.detect_sensitive_content(frame, roi)
                # Get confidence from separate call (could be optimized)
                _, confidence, _ = self.extract_text_from_frame(frame, roi, high_quality=False)
                
                results.append((text, confidence, has_sensitive))
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(frames)} frames")
                    
            except Exception as e:
                logger.error(f"Failed to process frame {i}: {e}")
                results.append(("", 0.0, False))
        
        return results