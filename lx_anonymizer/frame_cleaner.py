"""
Frame-level anonymization module for video processing.

This module provides functionality to:
- Extract frames from videos using ffmpeg
- Apply specialized frame OCR to detect sensitive information
- Remove or mask frames containing sensitive data
- Re-encode cleaned videos

Uses specialized frame processing components separated from PDF logic.
"""

import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Mapping, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image

from lx_anonymizer.anonymization.masking import MaskApplication
from lx_anonymizer.huggingface_cache.can_load_model import HF_Cache
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor
from lx_anonymizer.ocr.ocr_frame import FrameOCR

# from lx_anonymizer.ocr_minicpm import (
#     _can_load_model,
#     create_minicpm_ocr,
# )
from lx_anonymizer.config import settings
from lx_anonymizer.ollama.ollama_llm_meta_extraction import (
    EnrichedMetadataExtractor,
    FrameSamplingOptimizer,
    OllamaOptimizedExtractor,
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
from lx_anonymizer.text_detection.roi_processor import ROIProcessor
from lx_anonymizer.utils.ollama import ensure_ollama
from lx_anonymizer.utils.roi_normalization import normalize_roi_keys
from lx_anonymizer.video_processing import video_encoder, video_processor, video_utils

logger = logging.getLogger(__name__)


class FrameCleaner:
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.

    This class provides methods to extract frames from a video, detect sensitive information
    using specialized frame OCR (including MiniCPM-o 2.6), and re-encode the video without sensitive frames.

    New features:
    - Stream-based processing with FFmpeg -c copy to avoid full transcoding
    - Named pipe (FIFO) support for in-memory video streaming
    - Pixel format conversion optimization for minimal re-encoding
    - NVIDIA NVENC hardware acceleration with CPU fallback
    """

    def __init__(
        self,
        use_minicpm: Optional[bool] = False,
        minicpm_config: Optional[Dict[str, Any]] = None,
        use_llm: Optional[bool] = None,
    ):
        # Initialize specialized frame processing components
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.patient_data_extractor = PatientDataExtractor()
        self.roi_processor = ROIProcessor()

        # Logging cache info
        logger.info("Huggingface cache status:")
        hf = HF_Cache()
        hf.log_hf_cache_info()

        # Enhanced OCR integration - use enhanced components if available
        logger.info("Initializing with Enhanced OCR components (OCR_FIX_V1 enabled)")
        self.use_enhanced_ocr = True

        # LLM usage flag (guard)
        self.use_llm = settings.LLM_ENABLED if use_llm is None else bool(use_llm)

        # Initialize MiniCPM-o 2.6 if enabled
        # self.use_minicpm = use_minicpm
        # self.minicpm_ocr = create_minicpm_ocr() if use_minicpm else None
        # self._log_hf_cache_info()

        # Initialize the optimized ollama processing pipeline (guarded)
        self.ollama_proc = None
        self.ollama_extractor = None
        self.frame_sampling_optimizer = None
        self.enriched_extractor = None
        if self.use_llm:
            try:
                # Initialize Ollama for LLM processing
                self.ollama_proc = ensure_ollama(
                    auto_pull_model=getattr(settings, "LLM_MODEL", None)
                )

                # Try to initialize OllamaOptimizedExtractor
                # If it fails (no models available), it will raise an exception caught below
                self.ollama_extractor = OllamaOptimizedExtractor(
                    preferred_model=settings.LLM_MODEL,
                    model_timeout=settings.LLM_TIMEOUT,
                )

                # Only initialize other components if ollama_extractor succeeded
                if self.ollama_extractor and self.ollama_extractor.current_model:
                    # Initialize enriched metadata extraction components
                    self.frame_sampling_optimizer = FrameSamplingOptimizer(
                        max_frames=100, skip_similar_threshold=0.85
                    )
                    self.enriched_extractor = EnrichedMetadataExtractor(
                        ollama_extractor=self.ollama_extractor,
                        frame_optimizer=self.frame_sampling_optimizer,
                    )
                else:
                    logger.warning(
                        "Ollama models not available, disabling LLM features"
                    )
                    self.use_llm = False
                    self.ollama_extractor = None

            except Exception as e:
                logger.warning(f"Ollama/LLM unavailable, disabling LLM features: {e}")
                self.use_llm = False
                self.ollama_proc = None
                self.ollama_extractor = None
                self.frame_sampling_optimizer = None
                self.enriched_extractor = None

        # Frame data collection for batch processing
        self.frame_collection: List[Dict[str, Any]] = []
        self.ocr_text_collection: List[str] = []
        self.current_video_total_frames = 0

        # Hardware acceleration detection, Encoder setup
        self.video_encoder = video_encoder.VideoEncoder()
        self.video_processor = video_processor.VideoProcessor(self.video_encoder)
        self.nvenc_available = self.video_encoder.nvenc_available
        self.preferred_encoder = self.video_encoder.preferred_encoder
        self.build_encoder_cmd = self.video_encoder.build_encoder_cmd

        # Masking
        self.mask_application = MaskApplication(self.preferred_encoder)
        self._load_mask = self.mask_application._load_mask()
        self._mask_video_streaming = self.mask_application.mask_video_streaming
        self._create_mask_config_from_roi = (
            self.mask_application.create_mask_config_from_roi
        )

        # Sensitive metadata dictionary
        self.sensitive_meta: SensitiveMeta = SensitiveMeta()
        self._reset_frame_collection()  # clear ocr and sm states

        # ROI Processor - dict traversal for convenience
        self.roi_processor = ROIProcessor()

        logger.info(
            f"Hardware acceleration: NVENC {'available' if self.nvenc_available else 'not available'}"
        )
        logger.info(f"Using encoder: {self.preferred_encoder}")

    def clean_video(
        self,
        video_path: Path,
        endoscope_image_roi: Optional[dict[str, int]],
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | None],
        output_path: Optional[Path] = None,
        technique: str = "mask_overlay",
        device: Optional[str] = "olympus_cv_1500",
    ) -> tuple[Path, Dict[str, Any]]:
        """
        Handles the cleaning of a video by removing or masking frames with sensitive information.
        Args:
            video_path: Path to the input video file
            endoscope_image_roi: ROI for image masking
            endoscope_data_roi_nested: Nested ROI for data extraction
            output_path: Optional path for the output cleaned video
            technique: 'remove_frames', 'mask_overlay', or 'extract_only'
            extended: Whether to perform extended metadata extraction

        Returns:
            A tuple containing the output video path and a dictionary of sensitive metadata.

        Refactored version: single code path, fewer duplicated branches. Batch Metadata logic preserves previous output based on confidence.
        """
        # Reset per-run state to avoid metadata/text bleed across videos.
        self.sensitive_meta = SensitiveMeta()
        self._reset_frame_collection()

        default_center = os.environ.get("DEFAULT_CENTER", "Endoscopy Center")
        output_video = output_path or video_path.with_stem(f"{video_path.stem}_anony")
        accumulated: dict[str, Any] = {
            "file_path": str(object=video_path),
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender_name": None,
            "examination_date": None,
            "examination_time": None,
            "examiner_first_name": None,
            "examiner_last_name": None,
            "center": default_center,
            "text": None,
            "source": "frame_extraction",
        }
        cap = cv2.VideoCapture(str(video_path))  # type: ignore[call-arg]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        target_samples = max(1, settings.MAX_FRAMES_TO_SAMPLE)
        max_samples = min(target_samples, total_frames) if total_frames else 0
        logger.info(
            "Video detected (%d frames). Sampling ≤%d frames.",
            total_frames,
            max_samples,
        )
        sensitive_idx: list[int] = []
        frames_processed = 0
        best_ocr_text = ""
        best_ocr_conf = -1.0
        for idx, gray_frame, stride in self._iter_video(video_path, total_frames):
            if frames_processed >= max_samples:
                logger.info("Reached maximum frame sample limit. Stopping analysis.")
                break
            is_sensitive, frame_meta, ocr_text, ocr_conf = self._process_frame_single(
                gray_frame,
                endoscope_image_roi=endoscope_image_roi,
                endoscope_data_roi_nested=endoscope_data_roi_nested,
                frame_id=idx,
                collect_for_batch=True,
            )
            accumulated = self.frame_metadata_extractor.merge_metadata(
                accumulated, frame_meta
            )
            if ocr_text and ocr_text.strip():
                candidate = ocr_text.strip()
                # Prefer higher confidence; break ties by longer text to avoid tiny overlays dominating.
                if ocr_conf > best_ocr_conf or (
                    abs(ocr_conf - best_ocr_conf) < 1e-6
                    and len(candidate) > len(best_ocr_text)
                ):
                    best_ocr_text = candidate
                    best_ocr_conf = float(ocr_conf)
            if is_sensitive:
                sensitive_idx.append(idx)
                self.sensitive_meta.safe_update(accumulated)
            frames_processed += 1
            if (
                settings.SMART_EARLY_STOPPING
                and technique == "extract_only"
                and self.frame_metadata_extractor.is_complete(accumulated)
            ):
                logger.info("Critical metadata found. Early stopping enabled.")
                break

        # Batch-Metadaten-Anreicherung nach Frame-Loop
        if self.use_llm and self.frame_collection:
            batch_enriched = self._extract_enriched_metadata_batch()
            if batch_enriched:
                accumulated = self.frame_metadata_extractor.merge_metadata(
                    accumulated, batch_enriched
                )
        elif self.frame_collection:
            logger.debug(
                "Skipping batch enrichment because LLM is disabled (frames=%d).",
                len(self.frame_collection),
            )
        sensitive_ratio = len(sensitive_idx) / total_frames if total_frames else 0.0
        logger.info(
            "Sensitive frames: %d/%d (%.1f %%)",
            len(sensitive_idx),
            total_frames,
            100 * sensitive_ratio,
        )
        if technique == "remove_frames":
            logger.info("Using frame‑removal strategy.")
            ok = self.remove_frames_from_video_streaming(
                video_path,
                sensitive_idx,
                output_video,
                total_frames=total_frames,
            )
            if not ok:
                logger.error("Frame removal failed.")
        elif technique == "mask_overlay":
            logger.info("Using masking strategy.")
            if endoscope_image_roi and self._validate_roi(endoscope_image_roi):
                mask_cfg = self._create_mask_config_from_roi(endoscope_image_roi)
            else:
                mask_cfg = self.mask_application._load_mask()
            self._mask_video_streaming(
                video_path, mask_cfg, output_video, use_named_pipe=True
            )
        elif technique == "extract_only":
            logger.info("Extraction-only mode: skipping video modification.")
            output_video = video_path

        # ----------------------- persist metadata, apply type checking---------------------------
        if best_ocr_text:
            accumulated["text"] = best_ocr_text
        elif not accumulated.get("text"):
            # Fallback for cases where OCR text was fully filtered but metadata was extracted.
            fallback_text = self._build_representative_text_from_meta(accumulated)
            if fallback_text:
                accumulated["text"] = fallback_text
        self.sensitive_meta.safe_update(accumulated)

        return output_video, self.sensitive_meta.to_dict()

    @staticmethod
    def _build_representative_text_from_meta(meta: Dict[str, Any]) -> str:
        """
        Build a compact diagnostic text from extracted metadata when raw OCR text
        is unavailable (e.g. filtered as gibberish).
        """
        if not isinstance(meta, dict):
            return ""

        parts: list[str] = []

        first = str(meta.get("patient_first_name") or "").strip()
        last = str(meta.get("patient_last_name") or "").strip()
        if first or last:
            parts.append(" ".join(p for p in [first, last] if p))

        for key, label in (
            ("casenumber", "Case"),
            ("patient_dob", "DOB"),
            ("examination_date", "Date"),
            ("examination_time", "Time"),
            ("examiner_last_name", "Examiner"),
            ("endoscope_type", "Scope"),
            ("endoscope_sn", "SN"),
        ):
            value = meta.get(key)
            if value:
                parts.append(f"{label}: {value}")

        return " | ".join(parts).strip()

    def remove_frames_from_video_streaming(
        self,
        original_video: Path,
        frames_to_remove: List[int],
        output_video: Path,
        total_frames: Optional[int] = None,
        use_named_pipe: bool = True,
    ) -> bool:
        """
        Remove frames using streaming approach with optional named pipes.

        Args:
            original_video: Path to original video
            frames_to_remove: List of frame numbers to remove (0-based)
            output_video: Path for output video
            total_frames: Total frame count (for optimization)
            use_named_pipe: Whether to use named pipes for streaming

        Returns:
            True if successful, False otherwise
        """
        filter_script_paths: List[Path] = []
        ffmpeg_timeout = max(10, int(getattr(settings, "LLM_TIMEOUT", 30)) * 10)
        try:
            if not frames_to_remove:
                logger.info("No frames to remove, using stream copy")
                return self.video_processor.stream_copy_video(
                    original_video, output_video
                )

            format_info = video_utils.detect_video_format(original_video)

            logger.info(
                f"Removing {len(frames_to_remove)} frames using streaming method"
            )

            vf, af = self._build_frame_drop_filters(frames_to_remove)
            vf_args, af_args, filter_script_paths = self._build_filter_args(vf, af)

            has_audio = bool(format_info.get("has_audio", True))

            if use_named_pipe and len(frames_to_remove) < (total_frames or 1000) * 0.1:
                # Use named pipe for small frame removal operations
                with video_utils.named_pipe() as pipe_path:
                    filter_proc: Optional[subprocess.Popen] = None
                    try:
                        # Pipeline: filter frames -> pipe -> stream copy to final output
                        filter_cmd = [
                            "ffmpeg",
                            "-nostdin",
                            "-y",
                            "-i",
                            str(original_video),
                            *vf_args,
                        ]
                        if has_audio:
                            filter_cmd.extend(af_args)
                        filter_cmd.extend(
                            [
                                "-f",
                                "matroska",  # Use MKV for better streaming compatibility
                                str(pipe_path),
                            ]
                        )

                        copy_cmd = [
                            "ffmpeg",
                            "-nostdin",
                            "-y",
                            "-fflags",
                            "nobuffer",
                            "-i",
                            str(pipe_path),
                            "-c",
                            "copy",  # Stream copy from pipe
                            "-movflags",
                            "+faststart",
                            str(output_video),
                        ]

                        logger.info(
                            "Using named pipe for frame removal streaming (MKV container)"
                        )
                        logger.debug(
                            f"Filter command with -nostdin: {' '.join(filter_cmd)}"
                        )
                        logger.debug(
                            f"Copy command with -nostdin: {' '.join(copy_cmd)}"
                        )

                        # Start filter process in background without PIPE buffering risk.
                        filter_proc = subprocess.Popen(
                            filter_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )

                        # Start copy process (blocks until complete)
                        copy_result = subprocess.run(
                            copy_cmd,
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=ffmpeg_timeout,
                        )
                        if (
                            isinstance(copy_result.returncode, int)
                            and copy_result.returncode != 0
                        ):
                            raise subprocess.CalledProcessError(
                                copy_result.returncode,
                                copy_cmd,
                                output=copy_result.stdout,
                                stderr=copy_result.stderr,
                            )

                        # Wait for filter to complete
                        filter_return = filter_proc.wait(timeout=ffmpeg_timeout)
                        if isinstance(filter_return, int) and filter_return != 0:
                            raise subprocess.CalledProcessError(
                                filter_return, filter_cmd
                            )

                        logger.debug("Streaming frame removal completed via named pipe")
                    except subprocess.TimeoutExpired as timeout_exc:
                        logger.error(
                            "Named-pipe frame removal timed out after %ss",
                            timeout_exc.timeout,
                        )
                        raise
                    finally:
                        if filter_proc is not None and filter_proc.poll() is None:
                            filter_proc.terminate()
                            try:
                                filter_proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                filter_proc.kill()
                                filter_proc.wait(timeout=10)

            else:
                # Direct processing for larger removals or when pipes unavailable
                encoder_args = self.build_encoder_cmd("balanced")
                cmd = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(original_video),
                    *vf_args,
                ]
                if has_audio:
                    # Keep A/V timelines aligned while dropping selected frames.
                    cmd.extend(
                        [
                            *af_args,
                            *encoder_args,
                            "-c:a",
                            "aac",
                            "-b:a",
                            "128k",
                            "-movflags",
                            "+faststart",
                            str(output_video),
                        ]
                    )
                else:
                    cmd.extend(
                        [
                            *encoder_args,
                            "-an",
                            str(output_video),
                        ]
                    )

                logger.info(
                    f"Direct frame removal processing using {self.preferred_encoder['type']}"
                )
                logger.debug(f"FFmpeg command with -nostdin: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=ffmpeg_timeout,
                )
                if isinstance(result.returncode, int) and result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        cmd,
                        output=result.stdout,
                        stderr=result.stderr,
                    )
                logger.debug(f"Direct frame removal output: {result.stderr}")

            # Verify output
            if output_video.exists() and output_video.stat().st_size > 0:
                logger.info(f"Successfully removed frames: {output_video}")
                return True
            else:
                logger.error("Frame removal output is empty or missing")
                return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Streaming frame removal failed: {e.stderr}")

            # Fallback to CPU method without audio processing
            try:
                logger.warning(
                    "Retrying frame removal without audio processing using CPU..."
                )
                fallback_encoder_args = self.build_encoder_cmd("fast", fallback=True)
                cmd_no_audio = [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(original_video),
                    *vf_args,
                    "-an",  # No audio
                    *fallback_encoder_args,
                    str(output_video),
                ]
                result = subprocess.run(
                    cmd_no_audio,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=ffmpeg_timeout,
                )
                if isinstance(result.returncode, int) and result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode,
                        cmd_no_audio,
                        output=result.stdout,
                        stderr=result.stderr,
                    )
                logger.info(
                    "Successfully removed frames without audio using CPU fallback"
                )
                return True
            except subprocess.TimeoutExpired as timeout_exc:
                logger.error(
                    "Frame removal CPU fallback timed out after %ss",
                    timeout_exc.timeout,
                )
                return False
            except subprocess.CalledProcessError as e2:
                logger.error(f"Frame removal CPU fallback also failed: {e2.stderr}")
                return False

        except subprocess.TimeoutExpired as timeout_exc:
            logger.error(
                "Streaming frame removal timed out after %ss", timeout_exc.timeout
            )
            return False
        except Exception as e:
            logger.error(f"Streaming frame removal error: {e}")
            return False
        finally:
            for script_path in filter_script_paths:
                try:
                    script_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Could not remove temporary filter script %s", script_path
                    )

    def _unified_metadata_extract(self, text: str) -> Dict[str, Any]:
        """Hierarchische Metadaten-Extraktion: LLM → spaCy → Regex-Fallback."""
        meta: Optional[Dict[str, Any]] = {}
        if self.use_llm and self.ollama_extractor:
            try:
                meta_obj = self.ollama_extractor.extract_metadata(text)
                if meta_obj is not None:
                    candidate = (
                        meta_obj.to_dict()
                        if isinstance(meta_obj, SensitiveMeta)
                        else dict(meta_obj)
                    )
                    if self._metadata_has_signal(candidate):
                        self.sensitive_meta.safe_update(candidate)
                        meta = self.sensitive_meta.to_dict()
                    else:
                        meta = None
                else:
                    meta = None
            except Exception:
                meta = None
        if not meta and self.patient_data_extractor:
            try:
                patient_candidate: Mapping[str, Any] = self.patient_data_extractor(text)
                if self._metadata_has_signal(patient_candidate):
                    self.sensitive_meta.safe_update(patient_candidate)
                    meta = self.sensitive_meta.to_dict()
                else:
                    meta = None
            except Exception:
                meta = None
        if not meta:
            out = self.frame_metadata_extractor.extract_metadata_from_frame_text(text)
            self.sensitive_meta.safe_update(out)
            meta = self.sensitive_meta.to_dict()

        # Ensure Dict is returned
        if isinstance(meta, SensitiveMeta):
            return meta.to_dict()

        return meta or {}

    @staticmethod
    def _metadata_has_signal(meta: Any) -> bool:
        """
        Return True if extracted metadata contains at least one meaningful field.
        Prevents empty extractor dicts from blocking regex fallback.
        """
        if not isinstance(meta, dict):
            return False
        signal_keys = (
            "patient_first_name",
            "patient_last_name",
            "patient_dob",
            "casenumber",
            "patient_gender_name",
            "examination_date",
            "examination_time",
            "examiner_first_name",
            "examiner_last_name",
            "endoscope_type",
            "endoscope_sn",
        )
        return any(
            (
                (value := meta.get(key)) is not None
                and (not isinstance(value, str) or bool(value.strip()))
            )
            for key in signal_keys
        )

    def _process_frame_single(
        self,
        gray_frame: np.ndarray,
        endoscope_image_roi: Optional[dict | None],
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | None],
        frame_id: int | None = None,
        collect_for_batch: bool = False,
    ) -> tuple[bool, dict[str, Any], str, float]:
        """
        Konsolidierte Einzel-Frame-Verarbeitung mit OCR, Metadaten und optionaler Batch-Sammlung.
        """
        logger.debug(f"Processing frame_id={frame_id or 'unknown'}")
        ocr_text: str = ""
        ocr_conf: float = 0.0
        frame_metadata: dict[str, Any] = self.sensitive_meta.to_dict()
        is_sensitive = False

        ocr_text, ocr_conf, frame_metadata = self.frame_ocr.extract_text_from_frame(
            gray_frame, endoscope_data_roi_nested
        )

        is_sensitive = self.frame_metadata_extractor.is_sensitive_content(
            frame_metadata
        )

        # Unified metadata extraction
        if ocr_text:
            meta_unified = self._unified_metadata_extract(ocr_text)
            frame_metadata = self.frame_metadata_extractor.merge_metadata(
                frame_metadata, meta_unified
            )
        self.sensitive_meta.safe_update(frame_metadata)
        frame_metadata = self.sensitive_meta.to_dict()

        # Collect for batch enrichment
        if collect_for_batch and ocr_text:
            self.frame_collection.append(
                {
                    "frame_id": frame_id,
                    "ocr_text": ocr_text,
                    "meta": frame_metadata,
                    "is_sensitive": is_sensitive,
                }
            )
            self.ocr_text_collection.append(ocr_text)
        return is_sensitive, frame_metadata, ocr_text, ocr_conf

    def video_ocr_stream(
        self, frame_paths: List[Path]
    ) -> Generator[tuple[str, float], Any, None]:
        """
        Yield (ocr_text, avg_confidence) for every frame in frame_paths.

        Uses FrameOCR with preprocessing for better quality.
        Confidence is normalised to [0,1]. Empty-text frames are skipped.
        """
        for fp in frame_paths:
            # Load image and convert to numpy array
            img = Image.open(fp).convert("L")
            frame_array = np.array(img)

            # Use FrameOCR with preprocessing instead of raw pytesseract
            ocr_text, avg_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,  # Could be enhanced with ROI if available
                high_quality=True,
            )

            if not ocr_text.strip():
                continue  # nothing recognisable

            yield ocr_text, avg_conf

    def _iter_video(
        self, video_path: Path, total_frames: int
    ) -> Iterator[Tuple[int, np.ndarray, int]]:
        """
        Yields frames with higher density to allow temporal analysis in batch processing.
        """
        cap = cv2.VideoCapture(str(video_path))  # type: ignore[call-arg]
        if not cap.isOpened():
            logger.error("Cannot open %s", video_path)
            return

        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except (AttributeError, cv2.error):
            pass

        # Sampling density is configurable via settings to balance speed/accuracy.
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Calculate skip to hit target samples, clamp to avoid over/under-sampling.
        target_samples = max(1, settings.MAX_FRAMES_TO_SAMPLE)
        calculated_skip = (
            math.ceil(total_frames / target_samples) if total_frames else 1
        )

        # Clamps: Min skip 5 (density), Max skip = 2 seconds of video
        max_skip_limit = int(fps * 2)
        skip = max(5, min(calculated_skip, max_skip_limit))

        idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if idx % skip == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # Apply sharpening for OCR
                gray = cv2.filter2D(
                    gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                )
                yield idx, gray, skip
            idx += 1

        cap.release()

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """LLM-first mit automatischem Modell-Fallback; spaCy als Notanker."""
        if not text or not text.strip():
            return {}

        logger.debug(
            f"Extracting metadata from text of length {len(text)} with content: {text}..."
        )

        meta: Dict[str, Any] = {}
        # Nur versuchen, wenn LLM aktiviert und verfügbar ist
        if (
            getattr(self, "use_llm", False)
            and getattr(self, "ollama_extractor", None) is not None
        ):
            try:
                meta_obj = self.ollama_extractor.extract_metadata(text)  # type: ignore  # Pydantic-Objekt oder None
                if isinstance(meta_obj, SensitiveMeta):
                    meta = meta_obj.to_dict()
            except Exception as e:
                logger.warning(f"Ollama extraction failed: {e}")
                meta = {}

        # LLM fehlgeschlagen/leer oder nicht aktiv → spaCy-Extractor fallback
        if meta == {}:
            try:
                if callable(self.patient_data_extractor):
                    spacy_meta: Mapping[str, Any] = self.patient_data_extractor(text)
                elif hasattr(self.patient_data_extractor, "extract"):
                    spacy_meta = self.patient_data_extractor.extract(text)
                elif hasattr(self.patient_data_extractor, "patient_extractor"):
                    spacy_meta = self.patient_data_extractor.patient_extractor(text)
                else:
                    spacy_meta = {}
                if isinstance(spacy_meta, Mapping):
                    meta = dict(spacy_meta)
            except Exception as e:
                logger.error(f"spaCy fallback failed: {e}")
                meta = {}

        return meta or {}

    def _extract_enriched_metadata_batch(self) -> Dict[str, Any]:
        """
        Extrahiert erweiterte Metadaten aus gesammelten Frame-Daten.

        Returns:
            Erweiterte Metadaten-Dictionary
        """
        if not self.use_llm:
            logger.debug("Batch enrichment disabled because LLM is not enabled.")
            return {}

        if not self.frame_collection:
            logger.warning("Keine Frame-Daten für Batch-Extraktion gesammelt")
            return {}

        logger.info(
            f"Extracting enriched metadata from {len(self.frame_collection)} collected frames"
        )
        try:
            # Verwende EnrichedMetadataExtractor für Multi-Frame-Analyse
            if not self.frame_sampling_optimizer:
                self.frame_sampling_optimizer = FrameSamplingOptimizer()
            if not self.ollama_extractor:
                self.ollama_extractor = OllamaOptimizedExtractor(
                    preferred_model=settings.LLM_MODEL,
                    model_timeout=settings.LLM_TIMEOUT,
                )
            if not self.enriched_extractor:
                self.enriched_extractor = EnrichedMetadataExtractor(
                    ollama_extractor=self.ollama_extractor,
                    frame_optimizer=self.frame_sampling_optimizer,
                )
            enriched_metadata = self.enriched_extractor.extract_from_frame_sequence(
                frames_data=self.frame_collection, ocr_texts=self.ocr_text_collection
            )

            logger.info(
                "✅ Enriched metadata extraction successful: %d fields",
                len(enriched_metadata),
            )
            return enriched_metadata
        except Exception as exc:
            logger.warning(
                "Batch enrichment failed softly (returning empty metadata): %s",
                exc,
            )
            return {}

    def _reset_frame_collection(self):
        """Setzt die Frame-Sammlung für ein neues Video zurück."""
        self.frame_collection.clear()
        self.ocr_text_collection.clear()
        logger.debug("Frame collection reset for new video")

    @staticmethod
    def _frame_ranges(indices: List[int]) -> List[Tuple[int, int]]:
        """Convert sorted frame indices into inclusive ranges."""
        clean = sorted({int(idx) for idx in indices if int(idx) >= 0})
        if not clean:
            return []
        ranges: List[Tuple[int, int]] = []
        start = clean[0]
        end = clean[0]
        for value in clean[1:]:
            if value == end + 1:
                end = value
                continue
            ranges.append((start, end))
            start = value
            end = value
        ranges.append((start, end))
        return ranges

    def _build_frame_drop_filters(self, frames_to_remove: List[int]) -> Tuple[str, str]:
        """
        Build compact ffmpeg select/aselect filters using interval-based ranges.
        This avoids extremely long eq(n,...) chains for large frame sets.
        """
        clean = sorted({int(idx) for idx in frames_to_remove if int(idx) >= 0})
        if not clean:
            return "select='1',setpts=N/FRAME_RATE/TB", "aselect='1',asetpts=N/SR/TB"

        terms: List[str] = []
        # Keep explicit eq(...) terms for small lists to preserve existing behavior/tests.
        if len(clean) <= 64:
            terms = [f"eq(n\\,{idx})" for idx in clean]
        else:
            ranges = self._frame_ranges(clean)
            for start, end in ranges:
                if start == end:
                    terms.append(f"eq(n\\,{start})")
                else:
                    terms.append(f"between(n\\,{start}\\,{end})")

        condition = "+".join(terms)
        vf = f"select='not({condition})',setpts=N/FRAME_RATE/TB"
        af = f"aselect='not({condition})',asetpts=N/SR/TB"
        return vf, af

    def _build_filter_args(
        self, vf: str, af: str
    ) -> Tuple[List[str], List[str], List[Path]]:
        """
        Provide ffmpeg filter args either inline or as temporary script files
        when expressions get too long for safe CLI handling.
        """
        if len(vf) + len(af) < 8000:
            return ["-vf", vf], ["-af", af], []

        script_paths: List[Path] = []
        vf_script = self._write_filter_script(vf, "vf")
        af_script = self._write_filter_script(af, "af")
        script_paths.extend([vf_script, af_script])
        return (
            ["-filter_script:v", str(vf_script)],
            ["-filter_script:a", str(af_script)],
            script_paths,
        )

    @staticmethod
    def _write_filter_script(content: str, prefix: str) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f".{prefix}.ffscript", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            return Path(tmp.name)

    def _validate_roi(self, roi: Dict[str, Any]) -> bool:
        """
        Validate that ROI dictionary contains required fields and reasonable values.

        Args:
            roi: ROI dictionary with x, y, width, height keys

        Returns:
            True if ROI is valid, False otherwise
        """
        if not isinstance(roi, dict):
            return False

        normalized = normalize_roi_keys(roi)
        if not normalized:
            return False

        roi = cast(dict[str, int], normalized)

        # Check for reasonable values (non-negative, not too large)
        try:
            x, y, width, height = roi["x"], roi["y"], roi["width"], roi["height"]

            if any(val < 0 for val in [x, y, width, height]):
                logger.warning(f"ROI contains negative values: {roi}")
                return False

            if width == 0 or height == 0:
                logger.warning(f"ROI has zero width or height: {roi}")
                return False

            if any(val > 5000 for val in [x, y, width, height]):
                logger.warning(f"ROI values seem unreasonably large: {roi}")
                return False

            return True

        except (TypeError, ValueError) as e:
            logger.warning(f"ROI contains invalid values: {roi}, error: {e}")
            return False
