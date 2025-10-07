"""
Enhanced BestFrameText with improved selection criteria to avoid gibberish.

This module addresses the gibberish text issue by implementing better
text quality scoring and selection criteria.
"""

import logging
import random
import re
import unicodedata
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedBestFrameText:
    """
    Improved frame text selector that prioritizes quality over quantity.
    
    Fixes the gibberish issue by:
    1. Using confidence-based selection instead of length-based
    2. Filtering out low-quality/gibberish text
    3. Implementing text quality scoring
    4. Providing diagnostic information
    """
    
    def __init__(
        self,
        reservoir_size: int = 500,
        min_conf: float = 0.3,  # Lowered for more samples
        min_len: int = 10,
        max_punct_ratio: float = 0.5,  # New: filter high punctuation
        min_readable_words: int = 2,    # New: require readable words
    ):
        """
        Initialize enhanced text selector.
        
        Args:
            reservoir_size: Maximum number of text samples to keep
            min_conf: Minimum OCR confidence threshold
            min_len: Minimum text length
            max_punct_ratio: Maximum allowed punctuation ratio
            min_readable_words: Minimum number of readable words required
        """
        self.reservoir_size = reservoir_size
        self.min_conf = min_conf
        self.min_len = min_len
        self.max_punct_ratio = max_punct_ratio
        self.min_readable_words = min_readable_words
        
        # Storage for text samples with metadata
        self.samples: List[Dict[str, Any]] = []
        self.total_pushed = 0
        
        # Enable enhanced selection if flag is set
        self.use_enhanced_selection = self._should_use_enhanced_selection()
        
        if self.use_enhanced_selection:
            logger.info("Enhanced BestFrameText selection enabled (OCR_FIX_V1)")
    
    def _should_use_enhanced_selection(self) -> bool:
        """Check if enhanced selection should be used."""
        import os
        return os.getenv('OCR_FIX_V1', '0') == '1'
    
    def _calculate_text_quality_score(self, text: str, confidence: float) -> float:
        """
        Calculate a quality score for OCR text.
        
        Args:
            text: OCR text to evaluate
            confidence: OCR confidence score (0.0 - 1.0)
            
        Returns:
            Quality score (0.0 - 1.0, higher is better)
        """
        if not text or not text.strip():
            return 0.0
        
        # Normalize text
        text = unicodedata.normalize("NFKC", text.strip())
        
        # Base score from confidence
        score = confidence * 0.4
        
        # Character analysis
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        punct_chars = r"""!@#$%^&*()_+{}|:"<>?`~[]\;',./§°^""‚''–—•…"""
        punct = sum(1 for ch in text if ch in punct_chars)
        
        # Letter ratio bonus (medical text should have many letters)
        letter_ratio = letters / total_chars
        score += letter_ratio * 0.2
        
        # Punctuation penalty (too much punctuation indicates gibberish)
        punct_ratio = punct / total_chars
        if punct_ratio > 0.5:
            score -= (punct_ratio - 0.5) * 0.3
        
        # Word analysis
        words = re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", text)
        readable_words = [w for w in words if len(w) >= 3 and not all(c == w[0] for c in w)]
        
        if len(words) > 0:
            readable_ratio = len(readable_words) / len(words)
            score += readable_ratio * 0.2
        
        # Medical/German text patterns bonus
        medical_patterns = [
            r'\b(?:Patient|Herr|Frau|Dr\.?|Prof\.?)\b',
            r'\b(?:geb\.?|geboren|Geburt)\b',
            r'\b(?:Datum|Untersuchung|Fall|Case)\b',
            r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # Date patterns
            r'\b[A-ZÄÖÜ][a-zäöüß]+\b',      # Capitalized words
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.05
        
        # Penalty for repeated characters (gibberish indicator)
        repeated_chars = len(re.findall(r"(.)\1{3,}", text))
        if repeated_chars > 0:
            score -= repeated_chars * 0.1
        
        # Penalty for excessive special characters
        special_chars = sum(1 for ch in text if ord(ch) > 127 and not ch.isalpha())
        if special_chars > total_chars * 0.1:
            score -= (special_chars / total_chars) * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _is_quality_text(self, text: str, confidence: float) -> bool:
        """
        Determine if text meets quality criteria.
        
        Args:
            text: OCR text to evaluate
            confidence: OCR confidence (0.0 - 1.0)
            
        Returns:
            True if text meets quality standards
        """
        if not text or len(text.strip()) < self.min_len:
            return False
        
        if confidence < self.min_conf:
            return False
        
        # Check punctuation ratio
        punct_chars = r"""!@#$%^&*()_+{}|:"<>?`~[]\;',./§°^""‚''–—•…"""
        punct_count = sum(1 for ch in text if ch in punct_chars)
        punct_ratio = punct_count / len(text)
        
        if punct_ratio > self.max_punct_ratio:
            return False
        
        # Check readable words
        words = re.findall(r"[A-Za-zÄÖÜäöüß]{3,}", text)
        readable_words = [w for w in words if not all(c == w[0] for c in w)]
        
        if len(readable_words) < self.min_readable_words:
            return False
        
        return True
    
    def push(
        self,
        text: str,
        ocr_conf: float,
        is_sensitive: bool = None,
    ) -> None:
        """
        Add OCR text sample with enhanced quality filtering.
        
        Args:
            text: OCR-extracted text
            ocr_conf: OCR confidence score (0.0 - 1.0)
            is_sensitive: Whether text contains sensitive information
        """
        self.total_pushed += 1
        
        # Always keep sensitive text (higher priority)
        force_keep = is_sensitive is True
        
        # Check quality criteria
        if not force_keep and not self._is_quality_text(text, ocr_conf):
            return
        
        # Calculate quality score
        quality_score = self._calculate_text_quality_score(text, ocr_conf)
        
        # Create sample record
        sample = {
            'text': text,
            'confidence': ocr_conf,
            'quality_score': quality_score,
            'is_sensitive': is_sensitive or False,
            'length': len(text),
            'word_count': len(re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", text)),
        }
        
        # Reservoir sampling with quality preference
        if len(self.samples) < self.reservoir_size:
            self.samples.append(sample)
        else:
            # Enhanced selection: prefer higher quality samples
            if self.use_enhanced_selection:
                # Find lowest quality sample to potentially replace
                min_quality_idx = min(range(len(self.samples)), 
                                    key=lambda i: self.samples[i]['quality_score'])
                min_quality = self.samples[min_quality_idx]['quality_score']
                
                # Replace if new sample is significantly better
                if quality_score > min_quality + 0.1 or force_keep:
                    self.samples[min_quality_idx] = sample
                elif random.random() < self.reservoir_size / self.total_pushed:
                    # Standard reservoir sampling as fallback
                    replace_idx = random.randrange(self.reservoir_size)
                    self.samples[replace_idx] = sample
            else:
                # Original reservoir sampling
                if random.random() < self.reservoir_size / self.total_pushed:
                    replace_idx = random.randrange(self.reservoir_size)
                    self.samples[replace_idx] = sample
    
    def reduce(self, preview_size: int = 5) -> Dict[str, str]:
        """
        Select best representative text samples.
        
        Args:
            preview_size: Number of samples for average preview
            
        Returns:
            Dictionary with 'best' and 'average' text samples
        """
        if not self.samples:
            return {"best": "", "average": ""}
        
        if self.use_enhanced_selection:
            return self._reduce_enhanced(preview_size)
        else:
            return self._reduce_original(preview_size)
    
    def _reduce_enhanced(self, preview_size: int) -> Dict[str, str]:
        """Enhanced reduction based on quality scores."""
        # Sort by quality score (highest first)
        sorted_samples = sorted(self.samples, key=lambda x: x['quality_score'], reverse=True)
        
        # Best sample: highest quality
        best_sample = sorted_samples[0]
        best_text = best_sample['text']
        
        # Average: top quality samples
        top_samples = sorted_samples[:min(preview_size, len(sorted_samples))]
        average_text = "\n\n".join(sample['text'] for sample in top_samples)[:1500]
        
        # Log diagnostic info
        logger.info(f"Selected best text (quality: {best_sample['quality_score']:.2f}, "
                   f"conf: {best_sample['confidence']:.2f}): {best_text[:50]}...")
        
        return {"best": best_text, "average": average_text}
    
    def _reduce_original(self, preview_size: int) -> Dict[str, str]:
        """Original reduction method (by length)."""
        best_text = max(self.samples, key=lambda x: x['length'])['text']
        
        preview_samples = random.sample(self.samples, min(preview_size, len(self.samples)))
        average_text = "\n\n".join(sample['text'] for sample in preview_samples)[:1500]
        
        return {"best": best_text, "average": average_text}
    
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information about collected samples."""
        if not self.samples:
            return {"total_samples": 0, "message": "No samples collected"}
        
        qualities = [s['quality_score'] for s in self.samples]
        confidences = [s['confidence'] for s in self.samples]
        
        return {
            "total_samples": len(self.samples),
            "total_pushed": self.total_pushed,
            "acceptance_rate": len(self.samples) / max(1, self.total_pushed),
            "avg_quality_score": sum(qualities) / len(qualities),
            "avg_confidence": sum(confidences) / len(confidences),
            "quality_range": (min(qualities), max(qualities)),
            "enhanced_selection": self.use_enhanced_selection,
            "sensitive_samples": sum(1 for s in self.samples if s['is_sensitive']),
        }
