import os
import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EnhancedBestFrameText:
    def __init__(self, max_candidates: int = 5):
        self.max_candidates = max_candidates
        self.candidates = []
        self.enable_quality_mode = bool(os.getenv('OCR_FIX_V1', '0') == '1')
        
    def _calculate_text_quality_score(self, text: str, confidence: float) -> float:
        if not text:
            return 0.0
        
        # Base score from confidence
        score = confidence * 0.4
        
        # Word quality analysis
        words = re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", text)
        if words:
            # Medical text patterns
            medical_patterns = [
                r"Patient|Untersuchung|Diagnose|Befund|Behandlung",
                r"mm|cm|Grad|°|%",
                r"links|rechts|lateral|medial|anterior|posterior",
                r"Jahr|Jahre|Tag|Tage|Monat|Monate"
            ]
            
            medical_score = 0.0
            for pattern in medical_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    medical_score += 0.1
            
            score += min(medical_score, 0.3)  # Cap medical bonus
            
            # Word length bonus
            avg_word_len = sum(len(word) for word in words) / len(words)
            if avg_word_len >= 4:
                score += 0.1
            
            # Readable word ratio
            readable_words = [w for w in words if len(w) >= 3]
            readable_ratio = len(readable_words) / len(words) if words else 0
            score += readable_ratio * 0.2
        
        # Penalty for excessive punctuation
        punct_chars = "!@#$%^&*()_+{}|:\"<>?`~[]\\;',./§°^‚''–—•…"
        punct_count = sum(1 for ch in text if ch in punct_chars)
        punct_ratio = punct_count / len(text)
        
        if punct_ratio > 0.5:
            score *= 0.3  # Heavy penalty for gibberish
        elif punct_ratio > 0.3:
            score *= 0.7  # Moderate penalty
        
        # Length bonus
        length_bonus = min(len(text) / 100, 0.1)
        score += length_bonus
        
        return min(score, 1.0)
    
    def _is_quality_text(self, text: str) -> bool:
        if not text or len(text) < 3:
            return False
        
        # Check for excessive special characters
        special_chars = sum(1 for ch in text if ord(ch) > 127 and not ch.isalpha())
        if special_chars > len(text) * 0.3:
            return False
        
        # Check for readable words
        words = re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", text)
        readable_words = [w for w in words if len(w) >= 3 and not all(c == w[0] for c in w)]
        
        return len(readable_words) > 0
    
    def push(self, text: str, confidence: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.enable_quality_mode:
            # Quality-based selection
            if not self._is_quality_text(text):
                logger.debug(f"Rejected low-quality text: {text[:50]}...")
                return
            
            quality_score = self._calculate_text_quality_score(text, confidence)
            
            candidate = {
                'text': text,
                'confidence': confidence,
                'quality_score': quality_score,
                'metadata': metadata or {}
            }
            
            self.candidates.append(candidate)
            
            # Sort by quality score and keep best candidates
            self.candidates.sort(key=lambda x: x['quality_score'], reverse=True)
            if len(self.candidates) > self.max_candidates:
                self.candidates = self.candidates[:self.max_candidates]
            
            logger.debug(f"Added candidate with quality score {quality_score:.3f}")
        
        else:
            # Original length-based selection
            candidate = {
                'text': text,
                'confidence': confidence,
                'quality_score': len(text),
                'metadata': metadata or {}
            }
            
            self.candidates.append(candidate)
            
            # Sort by text length
            self.candidates.sort(key=lambda x: len(x['text']), reverse=True)
            if len(self.candidates) > self.max_candidates:
                self.candidates = self.candidates[:self.max_candidates]
    
    def get_best_text(self) -> str:
        if not self.candidates:
            return ""
        
        best_candidate = self.candidates[0]
        
        if self.enable_quality_mode:
            logger.info(f"Best text selected with quality score: {best_candidate['quality_score']:.3f}")
        
        return best_candidate['text']
    
    def get_all_candidates(self) -> list:
        return self.candidates.copy()
    
    def clear(self) -> None:
        self.candidates.clear()
    
    def get_quality_stats(self) -> Dict[str, Any]:
        if not self.candidates:
            return {"total_candidates": 0}
        
        quality_scores = [c['quality_score'] for c in self.candidates]
        confidences = [c['confidence'] for c in self.candidates]
        
        return {
            "total_candidates": len(self.candidates),
            "avg_quality_score": sum(quality_scores) / len(quality_scores),
            "max_quality_score": max(quality_scores),
            "avg_confidence": sum(confidences) / len(confidences),
            "best_candidate_length": len(self.candidates[0]['text']) if self.candidates else 0,
            "quality_mode_enabled": self.enable_quality_mode
        }
