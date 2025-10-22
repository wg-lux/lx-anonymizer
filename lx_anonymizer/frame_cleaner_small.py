# lx_anonymizer/frame_cleaner/frame_cleaner.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
from lx_anonymizer.frame_analysis_service import FrameAnalysisService
from lx_anonymizer.video_processing_service import VideoProcessingService
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
from lx_anonymizer.video_encoder import VideoEncoder

logger = logging.getLogger(__name__)

class FrameCleaner:
    """
    Orchestrates video cleaning: frame sampling, OCR, metadata aggregation, and masking.
    """

    def __init__(self, use_llm: bool = False):
        self.analysis = FrameAnalysisService(use_llm=use_llm)
        self.video_encoder = VideoEncoder(mask_video_streaming=False, create_mask_config_from_roi=False)
        self.video_service = VideoProcessingService(self.video_encoder.preferred_encoder)
        self.sensitive_meta = SensitiveMeta()

    def clean_video(
        self,
        video_path: Path,
        endoscope_image_roi: Optional[Dict[str, int]] = None,
        endoscope_data_roi_nested: Optional[Dict[str, Dict[str, int]]] = None,
        output_path: Optional[Path] = None,
        technique: str = "mask_overlay"
    ) -> Tuple[Path, SensitiveMeta]:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        output_video = output_path or video_path.with_stem(f"{video_path.stem}_anony")
        metadata_accumulated: Dict[str, Any] = {}
        sensitive_frames = []

        logger.info(f"Processing {total_frames} frames from {video_path}")
        for idx, gray_frame in self._iter_video(video_path, total_frames):
            ocr_text, conf, meta, is_sensitive = self.analysis.run_ocr(gray_frame, endoscope_data_roi_nested)
            metadata_accumulated.update(meta)
            if is_sensitive:
                sensitive_frames.append(idx)

        # Apply masking
        mask_cfg = endoscope_image_roi or {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "image_width": 1920,
            "image_height": 1080
        }
        if technique == "mask_overlay":
            self.video_service.mask_video(video_path, mask_cfg, output_video)
        if technique == "remove_frames":
            self.video_service.remove_frames(video_path, sensitive_frames, output_video)

        # Merge metadata
        self.sensitive_meta = self.analysis.update_sensitive_meta(self.sensitive_meta, metadata_accumulated)
        logger.info(f"SensitiveMeta: {self.sensitive_meta.to_dict()}")

        return output_video, self.sensitive_meta

    def _iter_video(self, video_path: Path, total_frames: int):
        cap = cv2.VideoCapture(str(video_path))
        skip = max(1, total_frames // 50)
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % skip == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                yield idx, gray
            idx += 1
        cap.release()
        
    def remove_frames(
        self,
        input_video: Path,
        frames_to_remove: list[int],
        output_video: Path
    ) -> bool:
        """
        Remove specified frames from the video.
        """
        return self.video_service.remove_frames(input_video, frames_to_remove, output_video)
