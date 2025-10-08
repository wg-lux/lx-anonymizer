"""
High-performance Frame-specific OCR module using Tesserocr for video processing.

This module provides significantly faster OCR functionality using Tesserocr:
- Pre-loaded Tesseract model for memoization (10-50x faster)
- Direct C++ API access instead of CLI wrapper
- Optimized for video frame batch processing
- Memory-efficient frame handling
- Medical text recognition patterns
- Optimized for endoscopy video overlays

Performance Benefits over pytesseract:
- Model is loaded once at initialization
- No subprocess overhead per frame
- Direct memory access to Tesseract engine
- Batch processing optimizations
"""

import logging
import cv2
import numpy as np
import os
import tesserocr
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, Any, Tuple, Optional, List
import re
import threading
import time

logger = logging.getLogger(__name__)

class TesseOCRFrameProcessor:
    """
    High-performance OCR processor using Tesserocr for video frames.
    
    Key Features:
    - Pre-loaded Tesseract model for maximum speed
    - Batch processing for video frames
    - Thread-safe operations
    - Automatic memory management
    - German medical text optimization
    """
    
    def __init__(self, language: str = 'deu+eng'):
        """
        Initialize TesseOCR processor with pre-loaded model.
        
        Args:
            language: Tesseract language code (e.g., 'deu+eng' for German+English)
        """
        self.language = language
        self._lock = threading.Lock()
        
        # Determine tessdata path from environment or system defaults
        tessdata_path = self._get_tessdata_path()
        
        # Initialize Tesseract API with optimal settings for medical video OCR
        try:
            # OEM 1 = LSTM only (best for modern text recognition)
            self.api = tesserocr.PyTessBaseAPI(lang=language, path=tessdata_path, oem=tesserocr.OEM.LSTM_ONLY)
            logger.info(f"TesseOCR initialized with LSTM engine, language: {language}, tessdata_path: {tessdata_path}")
            
            # PSM 3 = Fully automatic page segmentation (best for mixed layouts)
            # This works better than SINGLE_BLOCK for medical overlays with multiple text regions
            self.api.SetPageSegMode(tesserocr.PSM.AUTO)
            
            # Remove char whitelist - it's too restrictive and causes fragmentation
            # Medical overlays may have various symbols we need to recognize
            # self.api.SetVariable('tessedit_char_whitelist', '')  # Commented out - no restrictions
            
            # Enable advanced dictionary and language models for better accuracy
            self.api.SetVariable('tessedit_enable_dict_correction', '1')
            self.api.SetVariable('tessedit_enable_bigram_correction', '1')
            self.api.SetVariable('language_model_penalty_non_dict_word', '0.5')  # Less aggressive penalties
            self.api.SetVariable('language_model_penalty_non_freq_dict_word', '0.5')
            
            # Improve segmentation quality
            self.api.SetVariable('textord_heavy_nr', '1')  # Better noise reduction
            self.api.SetVariable('preserve_interword_spaces', '1')  # Keep spaces between words
            
            # Optimize for better recognition quality
            self.api.SetVariable('tessedit_create_hocr', '0')  # Don't create HOCR (we don't need it)
            self.api.SetVariable('tessedit_pageseg_mode', '3')  # Auto page segmentation
            
            logger.info("TesseOCR configured with optimized settings: PSM=AUTO, OEM=LSTM_ONLY, no char restrictions")
            
        except Exception as e:
            logger.error(f"Failed to initialize TesseOCR: {e}")
            self.api = None
            raise
        
        # Frame-specific OCR configurations - optimized for high quality
        self.frame_config = {
            'dpi': 400,  # Increased from 300 for better text clarity
            'min_confidence': 20,  # Lowered to catch more text (we filter later)
            'high_quality_dpi': 600,  # Significantly increased for maximum quality
            'high_quality_min_confidence': 30  # Lowered threshold for high quality mode
        }
        
        # Medical text patterns for enhanced detection
        self.medical_patterns = [
            r'Patient[:\s]*([A-Za-zäöüÄÖÜß\s\-]+)',
            r'Pat[\.:\s]*([A-Za-zäöüÄÖÜß\s\-]+)',
            r'geb[\.:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'Fall[nr]*[\.:\s]*(\d+)',
            r'Datum[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'Geschlecht[:\s]*(männlich|weiblich|m|w|male|female)',
            r'Geburtsdatum[:\s]*(\d{1,2}\.\d{1,2}\.\d{2,4})',
            r'(\d{1,2}\.\d{1,2}\.\d{2,4})',  # Generic date pattern
            r'(\d{2,})',  # Case numbers
        ]
        
        # Performance metrics
        self.processed_frames = 0
        self.total_processing_time = 0.0
        
    def __del__(self):
        """Clean up Tesseract API on destruction."""
        if hasattr(self, 'api') and self.api:
            try:
                self.api.End()
            except Exception as e:
                logger.debug(f"Error cleaning up Tesseract API: {e}")
    
    def _get_tessdata_path(self) -> Optional[str]:
        """
        Determine the correct tessdata directory for tesserocr.
        
        IMPORTANT: tesserocr's PyTessBaseAPI(path=...) expects the tessdata/ directory itself!
        This is DIFFERENT from TESSDATA_PREFIX (which points to the parent for CLI).
        
        Returns:
            Path to tessdata/ directory itself, or None to use system default
        """
        import glob
        
        # 1. Check environment variable (set by devenv.nix)
        # TESSDATA_PREFIX points to parent (/nix/store/.../share), so append /tessdata
        env_tessdata_parent = os.environ.get('TESSDATA_PREFIX')
        if env_tessdata_parent:
            # If it already ends with /tessdata, use as-is
            if env_tessdata_parent.endswith('/tessdata') and os.path.isdir(env_tessdata_parent):
                logger.info(f"Using TESSDATA_PREFIX directly: {env_tessdata_parent}")
                return env_tessdata_parent
            # If it's the parent, append /tessdata
            tessdata_dir = os.path.join(env_tessdata_parent, 'tessdata')
            if os.path.isdir(tessdata_dir):
                logger.info(f"Using tessdata from TESSDATA_PREFIX parent: {tessdata_dir} (parent: {env_tessdata_parent})")
                return tessdata_dir
        
        # 2. Check common NixOS locations - look for tessdata directory
        nix_patterns = [
            '/nix/store/*/share/tessdata',
            '/run/current-system/sw/share/tessdata',
        ]
        
        for nix_pattern in nix_patterns:
            if '*' in nix_pattern:
                # Expand glob pattern for NixOS store paths
                matches = glob.glob(nix_pattern)
                if matches:
                    # Prefer paths with language files
                    for tessdata_dir in matches:
                        if os.path.isdir(tessdata_dir):
                            # Quick check for language files
                            traineddata_files = [f for f in os.listdir(tessdata_dir) if f.endswith('.traineddata')]
                            if traineddata_files:
                                logger.info(f"Using NixOS tessdata directory: {tessdata_dir} (found {len(traineddata_files)} language files)")
                                return tessdata_dir
                            else:
                                logger.warning(f"Skipping {tessdata_dir} - no .traineddata files found")
            elif os.path.isdir(nix_pattern):
                logger.info(f"Using NixOS tessdata directory: {nix_pattern}")
                return nix_pattern
        
        # 3. Check standard Linux paths - tessdata dir itself
        standard_tessdata_paths = [
            '/usr/share/tessdata',
            '/usr/local/share/tessdata',
        ]
        
        for tessdata_path in standard_tessdata_paths:
            if os.path.isdir(tessdata_path):
                logger.info(f"Using standard tessdata directory: {tessdata_path}")
                return tessdata_path
        
        # 4. Let tesserocr use its default
        logger.warning("No tessdata path found in environment or standard locations, using tesserocr default")
        return None
    
    def preprocess_frame_for_ocr(self, frame: np.ndarray, roi: Optional[Dict[str, Any]] = None) -> Image.Image:
        """
        Preprocess a video frame for optimal OCR results with aggressive quality enhancement.
        
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
            
            # AGGRESSIVE UPSCALING for tiny text in video frames
            # Target at least 2400px on the largest dimension for optimal OCR
            original_size = pil_image.size
            target_size = 2400
            scale_factor = max(3.0, target_size / max(original_size))  # Minimum 3x upscale
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            logger.debug(f"Upscaled frame from {original_size} to {new_size} (factor: {scale_factor:.2f}x)")
            
            # Convert to grayscale
            pil_image = pil_image.convert('L')
            
            # ENHANCED contrast adjustment with adaptive limits
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2.5)  # Increased from 2.0
            
            # Apply STRONGER sharpening with UnsharpMask
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            # Additional edge enhancement
            pil_image = pil_image.filter(ImageFilter.EDGE_ENHANCE)
            
            # Convert to numpy for advanced processing
            img_array = np.array(pil_image)
            
            # Apply Gaussian blur BEFORE thresholding to reduce noise
            img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
            
            # Adaptive thresholding for better handling of varying lighting
            # Use both Otsu AND adaptive for comparison
            _, binary_otsu = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_adaptive = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Use Otsu by default (generally better for uniform overlays)
            binary = binary_otsu
            
            # REFINED morphological operations with smaller kernel
            kernel = np.ones((1, 1), np.uint8)  # Reduced from (2,2) to preserve thin text
            
            # Close small gaps in characters
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Remove small noise
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Optional: Dilate very slightly to thicken thin characters
            kernel_dilate = np.ones((1, 1), np.uint8)
            cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)
            
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
        Extract text from a video frame using pre-loaded Tesseract model.
        
        Args:
            frame: Input frame as numpy array
            roi: Optional region of interest for OCR
            high_quality: Use high-quality OCR settings
            
        Returns:
            Tuple of (extracted_text, confidence_score, ocr_data)
        """
        if not self.api:
            logger.error("TesseOCR API not initialized")
            return "", 0.0, {}
            
        start_time = time.time()
        
        try:
            with self._lock:  # Thread safety
                # Preprocess frame
                processed_image = self.preprocess_frame_for_ocr(frame, roi)
                
                # Set DPI for better recognition
                dpi = self.frame_config['high_quality_dpi'] if high_quality else self.frame_config['dpi']
                
                # Set image in Tesseract API
                self.api.SetImage(processed_image)
                
                # Extract text - this is much faster than pytesseract!
                extracted_text = self.api.GetUTF8Text().strip()
                
                # Get confidence score
                confidence = self.api.MeanTextConf()
                confidence_normalized = confidence / 100.0 if confidence > 0 else 0.0
                
                # Get detailed text structure information
                ocr_data = {}
                try:
                    # The primary extracted text is already from GetUTF8Text() above
                    # Just add metadata about the extraction process
                    ocr_data = {
                        'processing_time': time.time() - start_time,
                        'dpi': dpi,
                        'method': 'tesserocr',
                        'engine': 'LSTM',
                        'psm': 'AUTO',
                        'text_length': len(extracted_text),
                        'word_count': len(extracted_text.split()) if extracted_text else 0
                    }
                    
                    # Optionally get bounding box data if needed (but don't rely on unpacking)
                    # This is for advanced use cases, not primary text extraction
                    try:
                        # Get iterator to text components
                        self.api.SetImage(processed_image)  # Ensure image is set
                        ri = self.api.GetIterator()
                        
                        if ri:
                            # Extract text at line level for structure
                            lines = []
                            level = tesserocr.RIL.TEXTLINE
                            
                            # Iterate through text lines
                            for _ in range(50):  # Limit iterations to prevent infinite loops
                                try:
                                    line_text = ri.GetUTF8Text(level)
                                    line_conf = ri.Confidence(level)
                                    
                                    if line_text and line_text.strip():
                                        lines.append({
                                            'text': line_text.strip(),
                                            'confidence': line_conf
                                        })
                                    
                                    if not ri.Next(level):
                                        break
                                except Exception:
                                    break
                            
                            if lines:
                                ocr_data['lines'] = lines
                                logger.debug(f"Extracted {len(lines)} text lines via iterator")
                    
                    except Exception as iter_error:
                        logger.debug(f"Could not extract line structure via iterator: {iter_error}")
                    
                except Exception as e:
                    logger.debug(f"Could not extract detailed OCR data: {e}")
            
            # Update performance metrics
            self.processed_frames += 1
            self.total_processing_time += time.time() - start_time
            
            logger.debug(f"TesseOCR extracted text: '{extracted_text[:50]}...' "
                        f"with confidence {confidence_normalized:.2f} "
                        f"in {time.time() - start_time:.3f}s")
            
            return extracted_text, confidence_normalized, ocr_data
            
        except Exception as e:
            logger.error(f"TesseOCR text extraction failed: {e}")
            return "", 0.0, {}
    
    def extract_text_from_frame_batch(
        self, 
        frames: List[np.ndarray], 
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Process multiple frames in batch for maximum efficiency.
        
        Args:
            frames: List of input frames as numpy arrays
            roi: Optional region of interest for OCR
            high_quality: Use high-quality OCR settings
            
        Returns:
            List of tuples (extracted_text, confidence_score, ocr_data)
        """
        if not self.api:
            logger.error("TesseOCR API not initialized")
            return [("", 0.0, {})] * len(frames)
        
        results = []
        batch_start = time.time()
        
        logger.info(f"Processing batch of {len(frames)} frames with TesseOCR...")
        
        for i, frame in enumerate(frames):
            try:
                result = self.extract_text_from_frame(frame, roi, high_quality)
                results.append(result)
                
                if i % 10 == 0 and i > 0:
                    avg_time = (time.time() - batch_start) / (i + 1)
                    logger.debug(f"Processed {i+1}/{len(frames)} frames, "
                               f"avg time: {avg_time:.3f}s/frame")
                    
            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                results.append(("", 0.0, {}))
        
        total_time = time.time() - batch_start
        logger.info(f"Batch processing complete: {len(frames)} frames in {total_time:.2f}s "
                   f"({total_time/len(frames):.3f}s/frame)")
        
        return results
    
    def detect_sensitive_content(
        self, 
        frame: np.ndarray, 
        roi: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Detect sensitive medical information in frame using TesseOCR.
        
        Args:
            frame: Input frame as numpy array
            roi: Optional region of interest
            
        Returns:
            Tuple of (has_sensitive_content, extracted_text, pattern_matches)
        """
        try:
            # Extract text using high-quality OCR
            ocr_text, confidence, ocr_data = self.extract_text_from_frame(
                frame, roi, high_quality=True
            )
            
            if not ocr_text or confidence < 0.3:
                return False, "", {}
            
            # Extract medical patterns
            patterns = self.extract_medical_text_patterns(ocr_text)
            
            # Determine if sensitive content is present
            has_sensitive = bool(patterns)
            
            logger.debug(f"Sensitive content detection: {has_sensitive}, "
                        f"patterns found: {len(patterns)}")
            
            return has_sensitive, ocr_text, patterns
            
        except Exception as e:
            logger.error(f"Sensitive content detection failed: {e}")
            return False, "", {}
    
    def extract_medical_text_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extract medical information patterns from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of extracted medical information
        """
        extracted_info = {
            'patient_name': [],
            'birth_date': [],
            'case_number': [],
            'examination_date': [],
            'gender': [],
            'dates': [],
            'numbers': []
        }
        
        try:
            for pattern in self.medical_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    matched_text = match.group(1) if match.groups() else match.group(0)
                    
                    # Categorize matches
                    if any(keyword in pattern.lower() for keyword in ['patient', 'pat']):
                        extracted_info['patient_name'].append(matched_text.strip())
                    elif 'geb' in pattern.lower() or 'geburtsdatum' in pattern.lower():
                        extracted_info['birth_date'].append(matched_text.strip())
                    elif 'fall' in pattern.lower():
                        extracted_info['case_number'].append(matched_text.strip())
                    elif 'datum' in pattern.lower():
                        extracted_info['examination_date'].append(matched_text.strip())
                    elif 'geschlecht' in pattern.lower():
                        extracted_info['gender'].append(matched_text.strip())
                    elif re.match(r'\d{1,2}\.\d{1,2}\.\d{2,4}', matched_text):
                        extracted_info['dates'].append(matched_text.strip())
                    elif re.match(r'\d{2,}', matched_text):
                        extracted_info['numbers'].append(matched_text.strip())
            
            # Remove duplicates
            for key in extracted_info:
                extracted_info[key] = list(set(extracted_info[key]))
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Medical pattern extraction failed: {e}")
            return extracted_info
    
    def _validate_roi(self, roi: Dict[str, Any]) -> bool:
        """Validate ROI parameters."""
        required_keys = ['x', 'y', 'width', 'height']
        return all(key in roi for key in required_keys) and all(
            isinstance(roi[key], (int, float)) and roi[key] >= 0 
            for key in required_keys
        )
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if self.processed_frames == 0:
            return {'avg_processing_time': 0.0, 'total_frames': 0}
        
        return {
            'avg_processing_time': self.total_processing_time / self.processed_frames,
            'total_frames': self.processed_frames,
            'total_time': self.total_processing_time,
            'frames_per_second': self.processed_frames / self.total_processing_time if self.total_processing_time > 0 else 0
        }
    
    def reset_performance_stats(self):
        """Reset performance metrics."""
        self.processed_frames = 0
        self.total_processing_time = 0.0


# Global instance for easy access (optional - can be instantiated as needed)
_global_tesseocr_processor = None

def get_global_tesseocr_processor(language: str = 'deu+eng') -> TesseOCRFrameProcessor:
    """
    Get or create global TesseOCR processor instance.
    
    This allows for model reuse across multiple video processing sessions.
    """
    global _global_tesseocr_processor
    
    if _global_tesseocr_processor is None:
        _global_tesseocr_processor = TesseOCRFrameProcessor(language)
    
    return _global_tesseocr_processor


# Convenience function for drop-in replacement
def extract_text_from_frame_fast(
    frame: np.ndarray, 
    roi: Optional[Dict[str, Any]] = None,
    high_quality: bool = True,
    language: str = 'deu+eng'
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Fast text extraction from frame using global TesseOCR processor.
    
    Drop-in replacement for pytesseract-based functions with significant speed improvement.
    """
    processor = get_global_tesseocr_processor(language)
    return processor.extract_text_from_frame(frame, roi, high_quality)
