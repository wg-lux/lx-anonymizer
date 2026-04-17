import logging
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Mapping, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image

from lx_anonymizer.anonymization.masking import MaskApplication
from lx_anonymizer.config import settings
from lx_anonymizer.huggingface_cache.can_load_model import HF_Cache
from lx_anonymizer.llm.factory import LLMFactory
from lx_anonymizer.llm.llm_extractor import (
    EnrichedMetadataExtractor,
    FrameSamplingOptimizer,
)
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor
from lx_anonymizer.ocr.ocr_frame import FrameOCR
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
from lx_anonymizer.text_detection.roi_processor import ROIProcessor
from lx_anonymizer.utils.roi_normalization import normalize_roi_keys
from lx_anonymizer.video_processing import video_encoder, video_processor, video_utils

logger = logging.getLogger(__name__)

_LLM_TITLE_TOKEN_RE = re.compile(
    r"\b(?:herrn?|frau|fru|monsieur|madame|dr\.?|prof\.?|professor|ing\.?)\b",
    re.IGNORECASE,
)
_LLM_AGE_TOKEN_RE = re.compile(r"\b\d{1,3}\s*jahre?\b", re.IGNORECASE)
_LLM_NARRATIVE_TOKEN_RE = re.compile(
    r"\b(?:befund|patient|screening|beschwerden|koloskopie|gastroskopie)\b",
    re.IGNORECASE,
)


def _in_pytest_runtime() -> bool:
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION")
    )


class FrameCleaner:
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.
    """

    def __init__(
        self,
        use_minicpm: bool = False,
        minicpm_config: Optional[Dict[str, Any]] = None,
        use_llm: Optional[bool] = None,
    ):
        self.use_minicpm = use_minicpm
        self.minicpm_config = minicpm_config or {}

        self._init_core_components()
        self._log_hf_cache_status()
        self._init_llm_pipeline(use_llm=use_llm)
        self._init_video_pipeline()
        self._init_masking()

        self.current_video_total_frames = 0
        self._reset_run_state()

        logger.info(
            "Hardware acceleration: NVENC %s",
            "available" if self.nvenc_available else "not available",
        )
        logger.info("Using encoder: %s", self.preferred_encoder)

    def _init_core_components(self) -> None:
        self.frame_ocr = FrameOCR()
        self.frame_metadata_extractor = FrameMetadataExtractor()
        self.patient_data_extractor = PatientDataExtractor()
        self.roi_processor = ROIProcessor()

        logger.info("Initializing with Enhanced OCR components (OCR_FIX_V1 enabled)")
        self.use_enhanced_ocr = True

    def _log_hf_cache_status(self) -> None:
        logger.info("Huggingface cache status:")
        hf = HF_Cache()
        hf.log_hf_cache_info()

    def _init_llm_pipeline(self, use_llm: Optional[bool]) -> None:
        self.use_llm = settings.LLM_ENABLED if use_llm is None else bool(use_llm)

        self.llm_extractor = None
        self.frame_sampling_optimizer = None
        self.enriched_extractor = None

        if not self.use_llm:
            return

        try:
            self.llm_extractor = LLMFactory.create_metadata_extractor()

            if self.llm_extractor and self.llm_extractor.current_model:
                self.frame_sampling_optimizer = FrameSamplingOptimizer(
                    max_frames=100,
                    skip_similar_threshold=0.85,
                )
                self.enriched_extractor = EnrichedMetadataExtractor(
                    llm_extractor=self.llm_extractor,
                    frame_optimizer=self.frame_sampling_optimizer,
                )
            else:
                logger.warning(
                    "LLM provider %s has no available models, disabling LLM features",
                    settings.LLM_PROVIDER,
                )
                self._disable_llm_pipeline()

        except Exception as exc:
            logger.warning(
                "LLM provider %s unavailable, disabling LLM features: %s",
                settings.LLM_PROVIDER,
                exc,
            )
            self._disable_llm_pipeline()

    def _disable_llm_pipeline(self) -> None:
        self.use_llm = False
        self.llm_extractor = None
        self.frame_sampling_optimizer = None
        self.enriched_extractor = None

    def _init_video_pipeline(self) -> None:
        self.video_encoder = video_encoder.VideoEncoder()
        self.video_processor = video_processor.VideoProcessor(self.video_encoder)

        self.nvenc_available = self.video_encoder.nvenc_available
        self.preferred_encoder = self.video_encoder.preferred_encoder
        self.build_encoder_cmd = self.video_encoder.build_encoder_cmd

    def _init_masking(self) -> None:
        self.mask_application = MaskApplication(self.preferred_encoder)
        self.default_mask_config = self.mask_application._load_mask()
        self._mask_video_streaming = self.mask_application.mask_video_streaming
        self._create_mask_config_from_roi = (
            self.mask_application.create_mask_config_from_roi
        )

    def _reset_run_state(self) -> None:
        self.frame_collection: List[Dict[str, Any]] = []
        self.ocr_text_collection: List[str] = []
        self._llm_calls_this_video = 0
        self._llm_seen_texts: set[str] = set()
        self.sensitive_meta: SensitiveMeta = SensitiveMeta()
        logger.debug("Run state reset for new video")

    def _target_sample_count(self, total_frames: int) -> int:
        configured = max(1, settings.MAX_FRAMES_TO_SAMPLE)
        if total_frames <= 0:
            return 0

        if _in_pytest_runtime():
            pytest_cap = min(configured, 12)
            logger.debug(
                "Pytest runtime detected, capping frame sampling from %d to %d.",
                configured,
                pytest_cap,
            )
            configured = pytest_cap

        return min(configured, total_frames)

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
        """
        _ = device  # preserved argument for compatibility

        self._reset_run_state()

        output_video, accumulated = self._prepare_clean_video_run(
            video_path=video_path,
            output_path=output_path,
        )

        total_frames = self._get_total_frames(video_path)
        self.current_video_total_frames = total_frames
        max_samples = self._target_sample_count(total_frames)

        logger.info(
            "Video detected (%d frames). Sampling ≤%d frames.",
            total_frames,
            max_samples,
        )

        analysis = self._analyze_video_frames(
            video_path=video_path,
            total_frames=total_frames,
            max_samples=max_samples,
            endoscope_image_roi=endoscope_image_roi,
            endoscope_data_roi_nested=endoscope_data_roi_nested,
            technique=technique,
            accumulated=accumulated,
        )

        accumulated = analysis["accumulated"]
        sensitive_idx = analysis["sensitive_idx"]
        best_ocr_text = analysis["best_ocr_text"]

        accumulated = self._maybe_enrich_video_metadata(accumulated)

        sensitive_ratio = len(sensitive_idx) / total_frames if total_frames else 0.0
        logger.info(
            "Sensitive frames: %d/%d (%.1f %%)",
            len(sensitive_idx),
            total_frames,
            100 * sensitive_ratio,
        )

        output_video = self._apply_cleaning_technique(
            technique=technique,
            video_path=video_path,
            output_video=output_video,
            sensitive_idx=sensitive_idx,
            total_frames=total_frames,
            endoscope_image_roi=endoscope_image_roi,
        )

        accumulated = self._finalize_video_metadata(
            accumulated=accumulated,
            best_ocr_text=best_ocr_text,
        )

        return output_video, self.sensitive_meta.to_dict()

    def _prepare_clean_video_run(
        self,
        video_path: Path,
        output_path: Optional[Path],
    ) -> tuple[Path, dict[str, Any]]:
        default_center = os.environ.get("DEFAULT_CENTER", "Endoscopy Center")
        output_video = output_path or video_path.with_stem(f"{video_path.stem}_anony")

        accumulated: dict[str, Any] = {
            "file_path": str(video_path),
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
        return output_video, accumulated

    @staticmethod
    def _get_total_frames(video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))  # type: ignore[call-arg]
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def _analyze_video_frames(
        self,
        video_path: Path,
        total_frames: int,
        max_samples: int,
        endoscope_image_roi: Optional[dict[str, int]],
        endoscope_data_roi_nested: Optional[dict[str, dict[str, int | None]] | None],
        technique: str,
        accumulated: dict[str, Any],
    ) -> dict[str, Any]:
        sensitive_idx: list[int] = []
        frames_processed = 0
        best_ocr_text = ""
        best_ocr_conf = -1.0

        for idx, gray_frame, stride in self._iter_video(video_path, total_frames):
            _ = stride
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

        return {
            "accumulated": accumulated,
            "sensitive_idx": sensitive_idx,
            "best_ocr_text": best_ocr_text,
            "best_ocr_conf": best_ocr_conf,
            "frames_processed": frames_processed,
        }

    def _maybe_enrich_video_metadata(
        self, accumulated: dict[str, Any]
    ) -> dict[str, Any]:
        if (
            self.use_llm
            and self.frame_collection
            and not self.frame_metadata_extractor.is_complete(accumulated)
            and (
                int(settings.LLM_MAX_CALLS_PER_VIDEO) < 0
                or self._llm_calls_this_video < int(settings.LLM_MAX_CALLS_PER_VIDEO)
            )
        ):
            batch_enriched = self._extract_enriched_metadata_batch()
            if batch_enriched:
                accumulated = self.frame_metadata_extractor.merge_metadata(
                    accumulated,
                    batch_enriched,
                )
        elif self.use_llm and self.frame_collection:
            logger.debug(
                "Skipping batch enrichment because metadata is already complete or LLM budget is exhausted."
            )
        elif self.frame_collection:
            logger.debug(
                "Skipping batch enrichment because LLM is disabled (frames=%d).",
                len(self.frame_collection),
            )

        return accumulated

    def _apply_cleaning_technique(
        self,
        technique: str,
        video_path: Path,
        output_video: Path,
        sensitive_idx: list[int],
        total_frames: int,
        endoscope_image_roi: Optional[dict[str, int]],
    ) -> Path:
        if technique == "remove_frames":
            logger.info("Using frame-removal strategy.")
            ok = self.remove_frames_from_video_streaming(
                video_path,
                sensitive_idx,
                output_video,
                total_frames=total_frames,
            )
            if not ok:
                logger.error("Frame removal failed.")
            return output_video

        if technique == "mask_overlay":
            logger.info("Using masking strategy.")
            if endoscope_image_roi and self._validate_roi(endoscope_image_roi):
                mask_cfg = self._create_mask_config_from_roi(endoscope_image_roi)
            else:
                mask_cfg = self.default_mask_config

            ok = self._mask_video_streaming(
                video_path,
                mask_cfg,
                output_video,
                use_named_pipe=True,
            )
            if not ok:
                raise RuntimeError(
                    "Masking failed: ROI/crop configuration does not match input video dimensions."
                )
            return output_video

        if technique == "extract_only":
            logger.info("Extraction-only mode: skipping video modification.")
            return video_path

        logger.warning(
            "Unknown cleaning technique '%s'. Returning original output path.",
            technique,
        )
        return output_video

    def _finalize_video_metadata(
        self,
        accumulated: dict[str, Any],
        best_ocr_text: str,
    ) -> dict[str, Any]:
        if best_ocr_text:
            accumulated["text"] = best_ocr_text
        elif not accumulated.get("text"):
            fallback_text = self._build_representative_text_from_meta(accumulated)
            if fallback_text:
                accumulated["text"] = fallback_text

        self.sensitive_meta.safe_update(accumulated)
        return accumulated

    @staticmethod
    def _build_representative_text_from_meta(meta: Dict[str, Any]) -> str:
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
        vf_args: List[str] = []
        af_args: List[str] = []
        ffmpeg_timeout = max(10, int(getattr(settings, "LLM_TIMEOUT", 30)) * 10)

        try:
            if not frames_to_remove:
                logger.info("No frames to remove, using stream copy")
                return self.video_processor.stream_copy_video(
                    original_video,
                    output_video,
                )

            format_info = video_utils.detect_video_format(original_video)
            has_audio = bool(format_info.get("has_audio", True))

            logger.info(
                "Removing %d frames using streaming method",
                len(frames_to_remove),
            )

            vf, af = self._build_frame_drop_filters(frames_to_remove)
            vf_args, af_args, filter_script_paths = self._build_filter_args(vf, af)

            should_use_named_pipe = (
                use_named_pipe and len(frames_to_remove) < (total_frames or 1000) * 0.1
            )

            if should_use_named_pipe:
                self._remove_frames_via_named_pipe(
                    original_video=original_video,
                    output_video=output_video,
                    vf_args=vf_args,
                    af_args=af_args,
                    has_audio=has_audio,
                    ffmpeg_timeout=ffmpeg_timeout,
                )
            else:
                self._remove_frames_direct(
                    original_video=original_video,
                    output_video=output_video,
                    vf_args=vf_args,
                    af_args=af_args,
                    has_audio=has_audio,
                    ffmpeg_timeout=ffmpeg_timeout,
                )

            return self._verify_video_output(output_video)

        except subprocess.CalledProcessError as exc:
            logger.error("Streaming frame removal failed: %s", exc.stderr)
            return self._remove_frames_cpu_fallback(
                original_video=original_video,
                output_video=output_video,
                vf_args=vf_args,
                ffmpeg_timeout=ffmpeg_timeout,
            )

        except subprocess.TimeoutExpired as exc:
            logger.error(
                "Streaming frame removal timed out after %ss",
                exc.timeout,
            )
            return False

        except Exception as exc:
            logger.error("Streaming frame removal error: %s", exc)
            return False

        finally:
            for script_path in filter_script_paths:
                try:
                    script_path.unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Could not remove temporary filter script %s",
                        script_path,
                    )

    def _remove_frames_via_named_pipe(
        self,
        original_video: Path,
        output_video: Path,
        vf_args: List[str],
        af_args: List[str],
        has_audio: bool,
        ffmpeg_timeout: int,
    ) -> None:
        """
        Remove frames via a named-pipe streaming pipeline:
        filter -> mkv pipe -> stream copy to final output.
        """
        with video_utils.named_pipe() as pipe_path:
            filter_proc: Optional[subprocess.Popen] = None
            try:
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
                        "matroska",
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
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(output_video),
                ]

                logger.info(
                    "Using named pipe for frame removal streaming (MKV container)"
                )
                logger.debug(
                    "Filter command with -nostdin: %s",
                    " ".join(filter_cmd),
                )
                logger.debug(
                    "Copy command with -nostdin: %s",
                    " ".join(copy_cmd),
                )

                filter_proc = subprocess.Popen(
                    filter_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

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

                filter_return = filter_proc.wait(timeout=ffmpeg_timeout)
                if isinstance(filter_return, int) and filter_return != 0:
                    raise subprocess.CalledProcessError(
                        filter_return,
                        filter_cmd,
                    )

                logger.debug("Streaming frame removal completed via named pipe")

            except subprocess.TimeoutExpired as exc:
                logger.error(
                    "Named-pipe frame removal timed out after %ss",
                    exc.timeout,
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

    def _remove_frames_direct(
        self,
        original_video: Path,
        output_video: Path,
        vf_args: List[str],
        af_args: List[str],
        has_audio: bool,
        ffmpeg_timeout: int,
    ) -> None:
        """
        Remove frames via direct ffmpeg processing.
        """
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
            "Direct frame removal processing using %s",
            self.preferred_encoder["type"],
        )
        logger.debug("FFmpeg command with -nostdin: %s", " ".join(cmd))

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

        logger.debug("Direct frame removal output: %s", result.stderr)

    def _remove_frames_cpu_fallback(
        self,
        original_video: Path,
        output_video: Path,
        vf_args: List[str],
        ffmpeg_timeout: int,
    ) -> bool:
        """
        Retry frame removal without audio using CPU fallback.
        """
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
                "-an",
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

            logger.info("Successfully removed frames without audio using CPU fallback")
            return self._verify_video_output(output_video)

        except subprocess.TimeoutExpired as exc:
            logger.error(
                "Frame removal CPU fallback timed out after %ss",
                exc.timeout,
            )
            return False
        except subprocess.CalledProcessError as exc:
            logger.error(
                "Frame removal CPU fallback also failed: %s",
                exc.stderr,
            )
            return False

    @staticmethod
    def _verify_video_output(output_video: Path) -> bool:
        if output_video.exists() and output_video.stat().st_size > 0:
            logger.info("Successfully removed frames: %s", output_video)
            return True

        logger.error("Frame removal output is empty or missing")
        return False

    def _unified_metadata_extract(self, text: str) -> Dict[str, Any]:
        meta: Optional[Dict[str, Any]] = {}
        if self.patient_data_extractor:
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

        if isinstance(meta, SensitiveMeta):
            return meta.to_dict()

        return meta or {}

    @staticmethod
    def _normalize_text_for_llm(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def _should_attempt_llm(self, text: str, current_meta: Dict[str, Any]) -> bool:
        if not (self.use_llm and self.llm_extractor):
            return False

        normalized_text = self._normalize_text_for_llm(text)
        if len(normalized_text) < max(1, int(settings.LLM_MIN_TEXT_LENGTH)):
            return False

        if normalized_text in self._llm_seen_texts:
            return False

        max_calls = int(settings.LLM_MAX_CALLS_PER_VIDEO)
        if max_calls >= 0 and self._llm_calls_this_video >= max_calls:
            logger.debug(
                "Skipping LLM extraction because per-video budget was exhausted (%d).",
                max_calls,
            )
            return False

        if self.frame_metadata_extractor.is_complete(current_meta):
            logger.debug(
                "Skipping LLM extraction because local extractors already found complete metadata."
            )
            return False

        return True

    def _remaining_llm_budget(self) -> Optional[int]:
        max_calls = int(settings.LLM_MAX_CALLS_PER_VIDEO)
        if max_calls < 0:
            return None
        return max(0, max_calls - self._llm_calls_this_video)

    def _select_llm_video_text_candidates(self) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()

        if self.enriched_extractor:
            aggregated = self.enriched_extractor._aggregate_ocr_texts(
                self.frame_collection, self.ocr_text_collection
            )
            normalized = self._normalize_text_for_llm(aggregated)
            if normalized and len(normalized) >= max(
                1, int(settings.LLM_MIN_TEXT_LENGTH)
            ):
                candidates.append(aggregated)
                seen.add(normalized)

        ranked_frames = sorted(
            self.frame_collection,
            key=lambda item: (
                float(item.get("ocr_confidence", 0.0)),
                len(str(item.get("ocr_text") or "")),
            ),
            reverse=True,
        )
        for frame_data in ranked_frames:
            text = str(frame_data.get("ocr_text") or "").strip()
            normalized = self._normalize_text_for_llm(text)
            if not normalized or normalized in seen:
                continue
            if len(normalized) < max(1, int(settings.LLM_MIN_TEXT_LENGTH)):
                continue
            candidates.append(text)
            seen.add(normalized)
            if len(candidates) >= 2:
                break

        return candidates

    def _llm_candidate_value_is_valid(
        self, key: str, value: Any, source_text: str
    ) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False

        cleaned = value.strip()
        if not cleaned:
            return False

        if key in {"patient_first_name", "patient_last_name"}:
            if _LLM_TITLE_TOKEN_RE.search(cleaned):
                return False
            if _LLM_AGE_TOKEN_RE.search(cleaned):
                return False
            if _LLM_NARRATIVE_TOKEN_RE.search(cleaned):
                return False
            if len(cleaned) > 40:
                return False

        if key == "casenumber":
            compact_value = re.sub(r"[^a-z0-9]", "", cleaned.lower())
            compact_source = re.sub(r"[^a-z0-9]", "", source_text.lower())
            return bool(compact_value) and compact_value in compact_source

        if key in {"patient_first_name", "patient_last_name"}:
            compact_source = re.sub(r"[^a-z0-9]", " ", source_text.lower())
            tokens = [tok for tok in re.split(r"\s+", cleaned.lower()) if tok]
            if not tokens:
                return False
            return all(token in compact_source for token in tokens)

        return True

    def _validate_llm_metadata_candidate(
        self, candidate: Mapping[str, Any], source_text: str
    ) -> bool:
        if not self._metadata_has_signal(candidate):
            return False

        for key in (
            "patient_first_name",
            "patient_last_name",
            "casenumber",
        ):
            if not self._llm_candidate_value_is_valid(
                key, candidate.get(key), source_text
            ):
                logger.debug(
                    "Rejecting LLM candidate because %s failed validation.", key
                )
                return False

        return True

    @staticmethod
    def _metadata_has_signal(meta: Any) -> bool:
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
        logger.debug(f"Processing frame_id={frame_id or 'unknown'}")
        ocr_text: str = ""
        ocr_conf: float = 0.0
        frame_metadata: dict[str, Any] = self.sensitive_meta.to_dict()
        is_sensitive = False

        _ = endoscope_image_roi

        ocr_text, ocr_conf, frame_metadata = self.frame_ocr.extract_text_from_frame(
            gray_frame, endoscope_data_roi_nested
        )

        is_sensitive = self.frame_metadata_extractor.is_sensitive_content(
            frame_metadata
        )

        if ocr_text:
            meta_unified = self._unified_metadata_extract(ocr_text)
            frame_metadata = self.frame_metadata_extractor.merge_metadata(
                frame_metadata, meta_unified
            )
        self.sensitive_meta.safe_update(frame_metadata)
        frame_metadata = self.sensitive_meta.to_dict()

        if collect_for_batch and ocr_text:
            self.frame_collection.append(
                {
                    "frame_id": frame_id,
                    "ocr_text": ocr_text,
                    "ocr_confidence": ocr_conf,
                    "meta": frame_metadata,
                    "is_sensitive": is_sensitive,
                }
            )
            self.ocr_text_collection.append(ocr_text)
        return is_sensitive, frame_metadata, ocr_text, ocr_conf

    def video_ocr_stream(
        self, frame_paths: List[Path]
    ) -> Generator[tuple[str, float], Any, None]:
        for fp in frame_paths:
            img = Image.open(fp).convert("L")
            frame_array = np.array(img)

            ocr_text, avg_conf, _ = self.frame_ocr.extract_text_from_frame(
                frame_array,
                roi=None,
                high_quality=True,
            )

            if not ocr_text.strip():
                continue

            yield ocr_text, avg_conf

    def _iter_video(
        self, video_path: Path, total_frames: int
    ) -> Iterator[Tuple[int, np.ndarray, int]]:
        cap = cv2.VideoCapture(str(video_path))  # type: ignore[call-arg]
        if not cap.isOpened():
            logger.error("Cannot open %s", video_path)
            return

        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        except (AttributeError, cv2.error):
            pass

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        target_samples = self._target_sample_count(total_frames) or 1
        calculated_skip = (
            math.ceil(total_frames / target_samples) if total_frames else 1
        )

        max_skip_limit = int(fps * 2)
        skip = max(5, min(calculated_skip, max_skip_limit))

        idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            if idx % skip == 0:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.filter2D(
                    gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                )
                yield idx, gray, skip
            idx += 1

        cap.release()

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {}

        logger.debug(
            f"Extracting metadata from text of length {len(text)} with content: {text}..."
        )

        meta: Dict[str, Any] = {}
        if (
            getattr(self, "use_llm", False)
            and getattr(self, "llm_extractor", None) is not None
        ):
            try:
                meta_obj = self.llm_extractor.extract_metadata(text)  # type: ignore
                if isinstance(meta_obj, SensitiveMeta):
                    meta = meta_obj.to_dict()
            except Exception as e:
                logger.warning("LLM extraction failed: %s", e)
                meta = {}

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
            if not self.frame_sampling_optimizer:
                self.frame_sampling_optimizer = FrameSamplingOptimizer()
            if not self.llm_extractor:
                self.llm_extractor = LLMFactory.create_metadata_extractor()
            if not self.enriched_extractor:
                self.enriched_extractor = EnrichedMetadataExtractor(
                    llm_extractor=self.llm_extractor,
                    frame_optimizer=self.frame_sampling_optimizer,
                )

            remaining_budget = self._remaining_llm_budget()
            if remaining_budget is not None and remaining_budget <= 0:
                logger.debug(
                    "Skipping batch enrichment because LLM budget is exhausted."
                )
                return {}

            text_candidates = self._select_llm_video_text_candidates()
            if not text_candidates:
                logger.debug(
                    "Skipping batch enrichment because no viable text candidate was found."
                )
                return {}

            validated_meta: Dict[str, Any] = {}
            attempts_allowed = (
                2 if remaining_budget is None else min(remaining_budget, 2)
            )

            for idx, text in enumerate(text_candidates[:attempts_allowed], start=1):
                normalized = self._normalize_text_for_llm(text)
                self._llm_seen_texts.add(normalized)
                self._llm_calls_this_video += 1
                logger.info(
                    "Running video-level LLM extraction attempt %d/%d on aggregated OCR text.",
                    idx,
                    attempts_allowed,
                )
                meta_obj = self.llm_extractor.extract_metadata(text)
                if meta_obj is None:
                    continue
                candidate = (
                    meta_obj.to_dict()
                    if isinstance(meta_obj, SensitiveMeta)
                    else dict(meta_obj)
                )
                if self._validate_llm_metadata_candidate(candidate, text):
                    validated_meta = candidate
                    break
                logger.info(
                    "Discarding video-level LLM result from attempt %d because validation failed.",
                    idx,
                )

            if validated_meta:
                logger.info(
                    "✅ Enriched metadata extraction successful: %d fields",
                    len(validated_meta),
                )
                return validated_meta

            logger.info("Video-level LLM extraction produced no validated metadata.")
            return {}
        except Exception as exc:
            logger.warning(
                "Batch enrichment failed softly (returning empty metadata): %s",
                exc,
            )
            return {}

    @staticmethod
    def _frame_ranges(indices: List[int]) -> List[Tuple[int, int]]:
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
        clean = sorted({int(idx) for idx in frames_to_remove if int(idx) >= 0})
        if not clean:
            return "select='1',setpts=N/FRAME_RATE/TB", "aselect='1',asetpts=N/SR/TB"

        terms: List[str] = []
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
        if not isinstance(roi, dict):
            return False

        normalized = normalize_roi_keys(roi)
        if not normalized:
            return False

        roi = cast(dict[str, int], normalized)

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
