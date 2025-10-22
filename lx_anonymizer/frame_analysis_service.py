# lx_anonymizer/frame_cleaner/frame_analysis_service.py
import logging
from typing import Any, Dict, Optional

import numpy as np

from lx_anonymizer.best_frame_text import BestFrameText
from lx_anonymizer.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ocr_frame import FrameOCR
from lx_anonymizer.ollama_llm_meta_extraction_optimized import \
    OllamaOptimizedExtractor
from lx_anonymizer.roi_processor import ROIProcessor
from lx_anonymizer.spacy_extractor import PatientDataExtractor

from .sensitive_meta_interface import SensitiveMeta

logger = logging.getLogger(__name__)

class FrameAnalysisService:
    """
    Handles OCR + metadata extraction for individual frames.
    Supports ROI-based OCR and unified metadata extraction pipeline.
    """

    def __init__(self, use_llm: bool = False):
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.patient_data_extractor = PatientDataExtractor()
        self.best_frame_text = BestFrameText()
        self.roi_processor = ROIProcessor()
        self.use_llm = use_llm
        self.ollama_extractor = OllamaOptimizedExtractor() if use_llm else None

    # --- Core OCR ---
    def run_ocr(
        self,
        gray_frame: np.ndarray,
        endoscope_data_roi_nested: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> tuple[str, float, Dict[str, Any], bool]:
        """OCR + heuristic metadata extraction"""
        ocr_text, ocr_conf, frame_metadata = "", 0.0, {}

        if not endoscope_data_roi_nested:
            ocr_text, ocr_conf, _ = self.frame_ocr.extract_text_from_frame(
                gray_frame, roi=None, high_quality=True
            )
            frame_metadata = self.frame_metadata_extractor.extract_metadata_from_frame_text(ocr_text)
        else:
            conf_acc, len_acc = 0.0, 0
            for key, roi in endoscope_data_roi_nested.items():
                if not roi:
                    continue
                text, conf, _ = self.frame_ocr.extract_text_from_frame(gray_frame, roi=roi, high_quality=True)
                frame_metadata[key] = text
                ocr_text += f"\n{text}"
                conf_acc += conf * len(text)
                len_acc += len(text)
            ocr_conf = conf_acc / len_acc if len_acc else 0.0

        if hasattr(self.best_frame_text, "push"):
            self.best_frame_text.push(ocr_text, ocr_conf)

        is_sensitive = self.frame_metadata_extractor.is_sensitive_content(frame_metadata)
        return ocr_text, ocr_conf, frame_metadata, is_sensitive

    # --- Unified Metadata Extraction ---
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """LLM → spaCy → regex fallback"""
        if not text or not text.strip():
            return {}
        if self.use_llm and self.ollama_extractor:
            try:
                data = self.ollama_extractor.extract_metadata(text)
                if data:
                    return data.model_dump()
            except Exception as e:
                logger.warning(f"LLM extractor failed: {e}")
        try:
            return self.patient_data_extractor(text)
        except Exception:
            return self.frame_metadata_extractor.extract_metadata_from_frame_text(text)

    # --- SensitiveMeta merging ---
    def update_sensitive_meta(self, sensitive_meta: SensitiveMeta, metadata: Dict[str, Any]) -> SensitiveMeta:
        """Merge metadata into SensitiveMeta (in place)."""
        for k, v in (metadata or {}).items():
            if v and hasattr(sensitive_meta, k):
                setattr(sensitive_meta, k, v)
        return sensitive_meta