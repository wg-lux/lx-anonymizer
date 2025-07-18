"""
MiniCPM-o 2.6 Vision-Language OCR Integration Module

This module provides a unified OCR + reasoning solution using MiniCPM-o 2.6's 
omni-modal capabilities to replace the traditional Tesseract + LLM pipeline.

Key features:
- Single model for both OCR and sensitivity detection using model.chat()
- High-resolution text transcription (state-of-the-art on OCRBench < 25B params)
- Immediate reasoning over extracted text without fragile handoffs
- Fallback to Tesseract for edge cases or batch processing
- Optimized for medical/endoscopy frame analysis
"""

import logging
import math
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import numpy as np
from PIL import Image
import torch
import gc
import os

logger = logging.getLogger(__name__)

min_storage_gb = 200.0  # Minimum free storage required before download

class StorageError(Exception):
    """Raised when there's insufficient storage for model operations."""
    pass

def _can_load_model() -> bool:
    """Check if the model can be loaded based on current storage."""
    storage_info = min_storage_gb
    
    if storage_info['free_gb'] < min_storage_gb:
        logger.error(f"Insufficient storage: {storage_info['free_gb']:.1f}GB free, "
                        f"required: {min_storage_gb}GB")
        return False
    
    if storage_info['hf_cache_gb'] > self.max_cache_size_gb:
        logger.warning(f"HuggingFace cache too large: {storage_info['hf_cache_gb']:.1f}GB, "
                        f"max allowed: {max_cache_size_gb}GB")
        return False
    
    return True

def _get_storage_info(self) -> Dict[str, float]:
    """Get current storage information in GB."""
    try:
        # Get filesystem stats
        total, used, free = shutil.disk_usage('/')
        
        # Get HuggingFace cache size
        hf_cache_size = 0
        if self.hf_cache_dir.exists():
            hf_cache_size = sum(
                f.stat().st_size for f in self.hf_cache_dir.rglob('*') 
                if f.is_file()
            )
        
        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'hf_cache_gb': hf_cache_size / (1024**3),
            'usage_percent': (used / total) * 100
        }
    except Exception as e:
        logger.error(f"Failed to get storage info: {e}")
        return {
            'total_gb': 0,
            'used_gb': 0,
            'free_gb': 0,
            'hf_cache_gb': 0,
            'usage_percent': 100
        }
    

class MiniCPMVisionOCR:
    """
    MiniCPM-o 2.6 based OCR and sensitivity detection for medical frames.
    
    Uses model.chat() for stateless frame-by-frame processing, combining 
    OCR and reasoning in a single model call for optimal efficiency.
    
    Now includes storage management to prevent filesystem overflow.
    """
    
    def __init__(
        self, 
        model_name: str = "openbmb/MiniCPM-o-2_6",
        max_image_size: Tuple[int, int] = (1344, 1344),
        device: Optional[str] = None,
        fallback_to_tesseract: bool = True,
        confidence_threshold: float = 0.7,
        min_storage_gb: float = 200.0,  # Minimum free storage required
        max_cache_size_gb: float = 200.0,  # Maximum HuggingFace cache size
        auto_cleanup: bool = True  # Whether to auto-cleanup on low storage
    ):
        """
        Initialize MiniCPM-o 2.6 vision OCR system with storage management.
        
        Args:
            model_name: HuggingFace model identifier (using int4 for efficiency)
            max_image_size: Maximum input resolution (1.8M px limit)
            device: Compute device (auto-detected if None)
            fallback_to_tesseract: Whether to use Tesseract fallback
            confidence_threshold: Minimum confidence for model outputs
            min_storage_gb: Minimum free storage required before download (default: 50GB)
            max_cache_size_gb: Maximum HuggingFace cache size before cleanup (default: 200GB)
            auto_cleanup: Whether to automatically clean up cache on low storage
        """
        self.model_name = model_name
        self.max_image_size = max_image_size
        self.fallback_to_tesseract = fallback_to_tesseract
        self.confidence_threshold = confidence_threshold
        self.min_storage_gb = min_storage_gb
        self.max_cache_size_gb = max_cache_size_gb
        self.auto_cleanup = auto_cleanup
        
        # Storage paths
        self.hf_cache_dir = Path.home() / '.cache' / 'huggingface'
        self.model_cache_dir = self.hf_cache_dir / 'hub'
        
        # Device detection
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.warning("CUDA not available, using CPU")
        else:
            self.device = device
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        
        # Check storage and load model
        self._check_and_manage_storage()
        self._load_model()
        

    def _check_and_manage_storage(self):
        """Check storage capacity and clean up if necessary."""
        storage_info = _get_storage_info()
        
        logger.info(f"Storage check: {storage_info['free_gb']:.1f}GB free, "
                   f"HF cache: {storage_info['hf_cache_gb']:.1f}GB")
        
        # Check if we have minimum required storage
        if storage_info['free_gb'] < self.min_storage_gb:
            if self.auto_cleanup:
                logger.warning(f"Low storage ({storage_info['free_gb']:.1f}GB < {self.min_storage_gb}GB), "
                              "attempting cleanup...")
                self._cleanup_hf_cache()
                
                # Re-check after cleanup
                storage_info = _get_storage_info()
                
            if storage_info['free_gb'] < self.min_storage_gb:
                error_msg = (
                    f"Insufficient storage for MiniCPM model. "
                    f"Available: {storage_info['free_gb']:.1f}GB, "
                    f"Required: {self.min_storage_gb}GB. "
                    f"Consider cleaning up HuggingFace cache ({storage_info['hf_cache_gb']:.1f}GB) "
                    f"or use fallback_to_tesseract=True"
                )
                logger.error(error_msg)
                
                if self.fallback_to_tesseract:
                    logger.info("Insufficient storage, will use Tesseract fallback only")
                    return
                else:
                    raise StorageError(error_msg)
        
        # Check if HuggingFace cache is too large
        if storage_info['hf_cache_gb'] > self.max_cache_size_gb:
            if self.auto_cleanup:
                logger.warning(f"HF cache too large ({storage_info['hf_cache_gb']:.1f}GB > {self.max_cache_size_gb}GB), "
                              "cleaning up...")
                self._cleanup_hf_cache()
    
    def _cleanup_hf_cache(self) -> float:
        """
        Clean up HuggingFace cache to free storage space.
        
        Returns:
            Amount of space freed in GB
        """
        if not self.hf_cache_dir.exists():
            return 0.0
        
        try:
            from huggingface_hub import scan_cache_dir
            
            # Scan cache to find deletable items
            cache_info = scan_cache_dir(self.hf_cache_dir)
            
            # Get size before cleanup
            size_before = sum(repo.size_on_disk for repo in cache_info.repos)
            
            # Find old/unused models to delete (keep only recent ones)
            repos_to_delete = []
            for repo in cache_info.repos:
                # Skip our current model
                if self.model_name in repo.repo_id:
                    continue
                    
                # Mark old models for deletion (> 30 days unused or large size)
                if (repo.size_on_disk > 10 * 1024**3):  # > 10GB
                    repos_to_delete.append(repo)
            
            # Delete old models
            if repos_to_delete:
                logger.info(f"Deleting {len(repos_to_delete)} cached models to free space...")
                for repo in repos_to_delete:
                    try:
                        repo.delete()
                        logger.info(f"Deleted cached model: {repo.repo_id} ({repo.size_on_disk / 1024**3:.1f}GB)")
                    except Exception as e:
                        logger.warning(f"Failed to delete {repo.repo_id}: {e}")
            
            # Get size after cleanup
            cache_info_after = scan_cache_dir(self.hf_cache_dir)
            size_after = sum(repo.size_on_disk for repo in cache_info_after.repos)
            
            freed_gb = (size_before - size_after) / (1024**3)
            logger.info(f"HuggingFace cache cleanup completed: {freed_gb:.1f}GB freed")
            
            return freed_gb
            
        except ImportError:
            # Fallback: manual cleanup of old files
            logger.warning("huggingface_hub not available, using manual cleanup")
            return self._manual_cache_cleanup()
        except Exception as e:
            logger.error(f"HuggingFace cache cleanup failed: {e}")
            return 0.0
    
    def _manual_cache_cleanup(self) -> float:
        """Manual cleanup of HuggingFace cache when hub library is not available."""
        try:
            if not self.hf_cache_dir.exists():
                return 0.0
            
            size_before = sum(
                f.stat().st_size for f in self.hf_cache_dir.rglob('*') 
                if f.is_file()
            )
            
            # Remove old temporary files and incomplete downloads
            patterns_to_clean = [
                '*.tmp',
                '*.incomplete',
                '**/tmp*',
                '**/temp*'
            ]
            
            files_deleted = 0
            for pattern in patterns_to_clean:
                for file_path in self.hf_cache_dir.rglob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            files_deleted += 1
                    except Exception as e:
                        logger.debug(f"Failed to delete {file_path}: {e}")
            
            size_after = sum(
                f.stat().st_size for f in self.hf_cache_dir.rglob('*') 
                if f.is_file()
            )
            
            freed_gb = (size_before - size_after) / (1024**3)
            logger.info(f"Manual cache cleanup completed: {files_deleted} files deleted, {freed_gb:.1f}GB freed")
            
            return freed_gb
            
        except Exception as e:
            logger.error(f"Manual cache cleanup failed: {e}")
            return 0.0
    
    def _estimate_model_size(self) -> float:
        """
        Estimate the download size of the MiniCPM model in GB.
        
        Returns:
            Estimated model size in GB
        """
        # MiniCPM-o 2.6 is approximately 8-12GB depending on precision
        # Being conservative with estimate
        model_size_estimates = {
            "openbmb/MiniCPM-o-2_6": 12.0,  # ~12GB for full model
            "openbmb/MiniCPM-o-2_6-int4": 4.0,  # ~4GB for quantized
        }
        
        return model_size_estimates.get(self.model_name, 10.0)  # Default 10GB
    
    def _load_model(self):
        """Load MiniCPM-o 2.6 model and tokenizer with storage checks."""
        try:
            # Check if we should skip model loading due to storage constraints
            storage_info = _get_storage_info()
            estimated_model_size = self._estimate_model_size()
            
            if storage_info['free_gb'] < estimated_model_size + 5:  # 5GB buffer
                logger.warning(f"Insufficient storage for model loading. "
                              f"Available: {storage_info['free_gb']:.1f}GB, "
                              f"Estimated model size: {estimated_model_size:.1f}GB")
                if self.fallback_to_tesseract:
                    logger.info("Skipping model loading, will use Tesseract fallback")
                    return
                else:
                    raise StorageError(f"Insufficient storage for model loading")
            
            logger.info(f"Loading MiniCPM-o 2.6 model: {self.model_name}")
            
            # Import here to avoid dependency issues if not available
            from transformers import AutoModel, AutoTokenizer
            
            # Load model in vision-only mode for faster loading and less VRAM usage
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation='sdpa',  # Scaled Dot Product Attention for efficiency
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                init_vision=True,   # Enable vision processing
                init_audio=False,   # Disable audio for frame processing
                init_tts=False,     # Disable TTS for frame processing
                cache_dir=str(self.hf_cache_dir)  # Use managed cache directory
            )
            
            # Load tokenizer (use base model name for tokenizer)
            tokenizer_name = "openbmb/MiniCPM-o-2_6"  # Use base model for tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                trust_remote_code=True,
                cache_dir=str(self.hf_cache_dir)
            )
            
            # Move to device and set to eval mode
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            
            # Log final storage status
            final_storage = _get_storage_info()
            logger.info(f"MiniCPM-o 2.6 model loaded successfully. "
                       f"Storage: {final_storage['free_gb']:.1f}GB free, "
                       f"HF cache: {final_storage['hf_cache_gb']:.1f}GB")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            self.model = None
            self.tokenizer = None
            if not self.fallback_to_tesseract:
                raise RuntimeError(f"transformers not available and fallback disabled: {e}")
        except StorageError:
            # Re-raise storage errors
            self.model = None
            self.tokenizer = None
            raise
        except Exception as e:
            logger.error(f"Failed to load MiniCPM-o 2.6 model: {e}")
            # Ensure both are None on failure
            self.model = None
            self.tokenizer = None
            if not self.fallback_to_tesseract:
                raise RuntimeError(f"MiniCPM-o 2.6 initialization failed: {e}")
    
    def _prepare_image(self, image: Image.Image) -> Image.Image:
        """
        Prepare image for MiniCPM-o processing with optimal sizing.
        
        Args:
            image: Input PIL image
            
        Returns:
            Prepared image within size constraints
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check if image exceeds maximum pixel count (1.8M pixels)
        width, height = image.size
        total_pixels = width * height
        max_pixels = self.max_image_size[0] * self.max_image_size[1]  # ~1.8M
        
        if total_pixels > max_pixels:
            # Calculate scaling factor to fit within limit
            scale_factor = math.sqrt(max_pixels / total_pixels)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def transcribe_image(self, image: Image.Image, extract_verbatim: bool = True) -> str:
        """
        Extract text from image using MiniCPM-o 2.6 vision encoder.
        
        Args:
            image: Input PIL image
            extract_verbatim: Whether to request exact transcription
            
        Returns:
            Extracted text string
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("MiniCPM-o model not available, using fallback")
            return self._fallback_transcribe(image)
        
        try:
            # Prepare image
            image = self._prepare_image(image)
            
            # Create prompt for exact transcription
            if extract_verbatim:
                prompt = (
                    "You are a strict OCR engine. Extract all visible text from this image "
                    "exactly as it appears, line by line. Output only the raw text without "
                    "any interpretation, formatting, or additional comments."
                )
            else:
                prompt = "Please transcribe the text content visible in this image."
            
            # Build message for model.chat()
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            # Generate transcription using model.chat()
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=False,  # Deterministic for OCR
                temperature=0.0,
                max_new_tokens=512,
                use_image_id=False
            )
            
            logger.debug(f"MiniCPM transcription: {response[:100]}...")
            return response.strip()
            
        except Exception as e:
            logger.error(f"MiniCPM transcription failed: {e}")
            if self.fallback_to_tesseract:
                return self._fallback_transcribe(image)
            return ""
    
    def detect_sensitivity_unified(
        self, 
        image: Image.Image, 
        context: str = "endoscopy video frame"
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Unified OCR + sensitivity detection in single model call.
        
        This is the key advantage of MiniCPM-o: no fragile handoffs between
        OCR and LLM components.
        
        Args:
            image: Input PIL image
            context: Context description for better reasoning
            
        Returns:
            Tuple of (is_sensitive, metadata_dict, transcribed_text)
        """
        if self.model is None or self.tokenizer is None:
            logger.warning("MiniCPM-o model not available, using fallback")
            return self._fallback_detect_sensitivity(image)
        
        try:
            # Prepare image
            image = self._prepare_image(image)
            
            # Create unified prompt for OCR + sensitivity detection
            prompt = f"""Analyze this {context} image and:

1. Extract all visible text exactly as it appears
2. Identify any patient-identifying information (PII)
3. Determine if this frame should be considered sensitive

Look specifically for:
- Patient names (first name, last name)
- Date of birth (DOB)
- Case/patient numbers  
- Examination dates and times
- Medical record numbers

Respond in this exact JSON format:
{{
  "text": "extracted text here",
  "has_pii": true/false,
  "meta": {{
    "patient_first_name": "name or null",
    "patient_last_name": "name or null",
    "patient_dob": "date or null",
    "casenumber": "number or null",
    "examination_date": "date or null",
    "examination_time": "time or null"
  }}
}}"""

            # Build message for model.chat()
            msgs = [{'role': 'user', 'content': [image, prompt]}]
            
            # Generate response using model.chat()
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=0.1,  # Low temperature for consistent structure
                max_new_tokens=768,
                use_image_id=False
            )
            
            # Parse JSON response
            is_sensitive, metadata, transcribed_text = self._parse_json_response(response)
            
            logger.debug(f"MiniCPM unified analysis: sensitive={is_sensitive}, text={transcribed_text[:50]}...")
            return is_sensitive, metadata, transcribed_text
            
        except Exception as e:
            logger.error(f"MiniCPM unified detection failed: {e}")
            if self.fallback_to_tesseract:
                return self._fallback_detect_sensitivity(image)
            return False, {}, ""
    
    def _parse_json_response(self, response: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Parse JSON response from unified detection prompt.
        
        Args:
            response: Raw model response
            
        Returns:
            Tuple of (is_sensitive, metadata_dict, transcribed_text)
        """
        try:
            # Try to extract JSON from response
            # Sometimes the model adds extra text, so look for JSON block
            response = response.strip()
            
            # Find JSON block in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Extract fields
                transcribed_text = data.get('text', '')
                is_sensitive = data.get('has_pii', False)
                metadata = data.get('meta', {})
                
                # Add source annotation
                metadata['source'] = 'minicpm_unified'
                
                return is_sensitive, metadata, transcribed_text
            else:
                logger.warning("No valid JSON found in response")
                return False, {}, response
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract at least the text content
            return False, {}, response
        except Exception as e:
            logger.error(f"Failed to parse unified response: {e}")
            return False, {}, ""
    
    def _fallback_transcribe(self, image: Image.Image) -> str:
        """Fallback OCR using Tesseract."""
        if not self.fallback_to_tesseract:
            return ""
        
        try:
            import pytesseract
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            return pytesseract.image_to_string(image, lang='deu')
        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            return ""
    
    def _fallback_detect_sensitivity(self, image: Image.Image) -> Tuple[bool, Dict[str, Any], str]:
        """Fallback sensitivity detection using traditional pipeline."""
        if not self.fallback_to_tesseract:
            return False, {}, ""
        
        try:
            # Use traditional OCR
            transcribed_text = self._fallback_transcribe(image)
            if not transcribed_text:
                return False, {}, ""
            
            # Basic pattern matching for PII (simplified fallback)
            # In production, this would use the full FrameMetadataExtractor
            is_sensitive = any(pattern in transcribed_text.lower() for pattern in [
                'patient:', 'name:', 'dob:', 'geburt', 'fall-nr', 'case'
            ])
            
            metadata = {'source': 'tesseract_fallback'}
            
            return is_sensitive, metadata, transcribed_text
            
        except Exception as e:
            logger.error(f"Fallback sensitivity detection failed: {e}")
            return False, {}, ""
    
    def extract_text_from_frame(
        self, 
        frame: np.ndarray, 
        roi: Optional[Dict[str, Any]] = None,
        high_quality: bool = True
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Extract text from video frame with optional ROI cropping.
        
        Compatible interface with existing FrameOCR for drop-in replacement.
        
        Args:
            frame: Input frame as numpy array
            roi: Optional region of interest for cropping
            high_quality: Whether to use high-quality processing
            
        Returns:
            Tuple of (extracted_text, confidence, metadata)
        """
        try:
            # Convert numpy array to PIL Image
            if len(frame.shape) == 2:  # Grayscale
                image = Image.fromarray(frame, mode='L').convert('RGB')
            else:  # Color
                image = Image.fromarray(frame, mode='RGB')
            
            # Apply ROI cropping if specified
            if roi and self._validate_roi(roi):
                x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
                image = image.crop((x, y, x + w, y + h))
            
            # Use unified detection for best results
            if high_quality:
                is_sensitive, metadata, transcribed_text = self.detect_sensitivity_unified(image)
                # Calculate a pseudo-confidence based on text length and metadata richness
                confidence = min(0.95, max(0.1, len(transcribed_text) / 100 + len(metadata) * 0.1))
            else:
                # Use basic transcription for speed
                transcribed_text = self.transcribe_image(image)
                confidence = 0.8 if transcribed_text else 0.0
                metadata = {}
            
            return transcribed_text, confidence, metadata
            
        except Exception as e:
            logger.error(f"Frame text extraction failed: {e}")
            return "", 0.0, {}
    
    def _validate_roi(self, roi: Dict[str, Any]) -> bool:
        """Validate ROI dictionary format."""
        required_keys = ['x', 'y', 'width', 'height']
        if not all(key in roi for key in required_keys):
            return False
        
        try:
            return all(isinstance(roi[key], (int, float)) and roi[key] >= 0 for key in required_keys)
        except (TypeError, ValueError):
            return False
    
    def cleanup(self):
        """Clean up GPU memory and model resources."""
        try:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            logger.info("MiniCPM-o resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor


# Factory function for easy integration
def create_minicpm_ocr(**kwargs) -> MiniCPMVisionOCR:
    """
    Factory function to create MiniCPM-o 2.6 OCR instance.
    
    Args:
        **kwargs: Arguments passed to MiniCPMVisionOCR constructor
        
    Returns:
        Configured MiniCPMVisionOCR instance
    """
    return MiniCPMVisionOCR(**kwargs)


# Quick test function
def test_minicpm_functionality():
    """Quick test of MiniCPM-o functionality."""
    try:
        # Create test image with some text
        from PIL import Image, ImageDraw, ImageFont
        
        test_image = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(test_image)
        
        # Try to use a basic font, fallback to default if not available
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((10, 30), "Test Patient: John Doe", fill='black', font=font)
        
        # Initialize MiniCPM OCR
        with create_minicpm_ocr() as ocr:
            # Test transcription
            text = ocr.transcribe_image(test_image)
            logger.info(f"Test transcription: {text}")
            
            # Test unified detection
            is_sensitive, metadata, extracted_text = ocr.detect_sensitivity_unified(test_image)
            logger.info(f"Test detection: sensitive={is_sensitive}, metadata={metadata}")
        
        return True
        
    except Exception as e:
        logger.error(f"MiniCPM test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic functionality test
    test_minicpm_functionality()