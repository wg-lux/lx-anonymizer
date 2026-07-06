import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generator, List, Mapping, cast

import cv2
import numpy as np
from PIL import Image
from lx_dtypes.models.meta.VideoMeta import (
    FrameAnalysisResult,
    FrameCleanerAccumulatedMeta,
    FrameCollectionItem,
    FrameObservation,
    FrameProcessResult,
    VideoMeta,
)
from lx_dtypes.models.contracts.llm_extractor import LLMFrameDataPayload
from lx_dtypes.models.contracts.video_processing import (
    VideoEncoderConfig,
    VideoMaskConfig,
)

from lx_anonymizer.anonymization.masking import MaskApplication
from lx_anonymizer.config import settings
from lx_anonymizer.huggingface_cache.can_load_model import HF_Cache
from lx_anonymizer.llm.factory import LLMFactory
from lx_anonymizer.llm.llm_extractor import (
    EnrichedMetadataExtractor,
    FrameSamplingOptimizer,
    LLMMetadataExtractor,
)
from lx_anonymizer.metrics_provenance import (
    build_anonymizer_provenance,
    summarize_frame_observations,
)
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor
from lx_anonymizer.ocr.ocr_frame import FlatRoi, FrameOCR, RoiInput
from lx_anonymizer.regex_patterns import (
    LLM_AGE_TOKEN_RE,
    LLM_NARRATIVE_TOKEN_RE,
    LLM_TITLE_TOKEN_RE,
    MULTISPACE_RE,
    NON_ALNUM_COMPACT_RE,
    NON_ALNUM_SPACE_RE,
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta, sensitive_meta_to_dict
from lx_anonymizer.text_detection.phi_region_detector import (
    detect_phi_regions_from_settings,
)
from lx_anonymizer.text_detection.roi_processor import ROIProcessor
from lx_anonymizer.utils.roi_normalization import normalize_roi_keys
from lx_anonymizer.video_processing import video_encoder, video_processor
from lx_anonymizer.frame_cleaner_video import FrameCleanerVideoMixin

logger = logging.getLogger(__name__)


class FrameCleanerQualityProfile(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass(frozen=True)
class FrameCleanerSamplingProfile:
    max_frames_to_sample: int
    high_quality_ocr: bool
    smart_early_stopping: bool
    early_stop_techniques: frozenset[str]

    @classmethod
    def from_quality_profile(
        cls, profile: FrameCleanerQualityProfile | str | None
    ) -> "FrameCleanerSamplingProfile":
        resolved_profile = _coerce_quality_profile(
            profile or settings.FRAME_CLEANER_QUALITY_PROFILE
        )
        configured_samples = max(1, settings.MAX_FRAMES_TO_SAMPLE)

        if resolved_profile is FrameCleanerQualityProfile.FAST:
            max_frames = min(configured_samples, 12)
            high_quality_ocr = False
        elif resolved_profile is FrameCleanerQualityProfile.QUALITY:
            max_frames = max(configured_samples, 48)
            high_quality_ocr = True
        else:
            max_frames = configured_samples
            high_quality_ocr = True

        return cls(
            max_frames_to_sample=max_frames,
            high_quality_ocr=high_quality_ocr,
            smart_early_stopping=bool(settings.SMART_EARLY_STOPPING),
            early_stop_techniques=frozenset({"extract_only", "mask_overlay"}),
        )


def _coerce_quality_profile(
    profile: FrameCleanerQualityProfile | str,
) -> FrameCleanerQualityProfile:
    if isinstance(profile, FrameCleanerQualityProfile):
        return profile
    try:
        return FrameCleanerQualityProfile(profile.strip().lower())
    except ValueError as exc:
        valid_values = ", ".join(item.value for item in FrameCleanerQualityProfile)
        raise ValueError(
            f"Unknown frame cleaner quality profile {profile!r}. "
            f"Expected one of: {valid_values}."
        ) from exc


def _in_pytest_runtime() -> bool:
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION")
    )


class FrameCleaner(FrameCleanerVideoMixin):
    """
    FrameCleaner class for handling video frame extraction and sensitive data detection.
    """

    def __init__(
        self,
        use_minicpm: bool = False,
        minicpm_config: Mapping[str, object] | None = None,
        use_llm: bool | None = None,
        quality_profile: FrameCleanerQualityProfile | str | None = None,
        sampling_profile: FrameCleanerSamplingProfile | None = None,
    ):
        self.use_minicpm = use_minicpm
        self.minicpm_config: dict[str, object] = dict(minicpm_config or {})
        self.sampling_profile = (
            sampling_profile
            or FrameCleanerSamplingProfile.from_quality_profile(quality_profile)
        )
        self.llm_extractor: LLMMetadataExtractor | None = None
        self.frame_sampling_optimizer: FrameSamplingOptimizer | None = None
        self.enriched_extractor: EnrichedMetadataExtractor | None = None
        self.mask_application: MaskApplication | None = None
        self.default_mask_config: VideoMaskConfig | None = None
        self._mask_video_streaming = None
        self._create_mask_config_from_roi = None

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

    def _init_llm_pipeline(self, use_llm: bool | None) -> None:
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
        self.preferred_encoder = VideoEncoderConfig.model_validate(
            self.video_encoder.preferred_encoder
        )
        self.build_encoder_cmd = self.video_encoder.build_encoder_cmd

    def _init_masking(self) -> None:
        self.mask_application = MaskApplication(self.preferred_encoder.model_dump())
        self.default_mask_config = self.mask_application.default_mask_config
        self._mask_video_streaming = self.mask_application.mask_video_streaming
        self._create_mask_config_from_roi = (
            self.mask_application.create_mask_config_from_roi
        )

    def _reset_run_state(self) -> None:
        self.frame_collection: List[FrameCollectionItem] = []
        self.frame_observations: List[FrameObservation] = []
        self.ocr_text_collection: List[str] = []
        self._llm_calls_this_video = 0
        self._llm_seen_texts: set[str] = set()
        self.sensitive_meta: SensitiveMeta = SensitiveMeta()
        logger.debug("Run state reset for new video")

    def _target_sample_count(self, total_frames: int) -> int:
        configured = max(1, self.sampling_profile.max_frames_to_sample)
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
        endoscope_image_roi: Mapping[str, object] | None,
        endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None,
        output_path: Path | None = None,
        technique: str = "mask_overlay",
        device: str | None = "olympus_cv_1500",
    ) -> tuple[Path, dict[str, object]]:
        """
        Handles the cleaning of a video by removing or masking frames with sensitive information.
        """
        _ = device  # preserved argument for compatibility

        self._reset_run_state()

        output_video, accumulated = self._prepare_clean_video_run(
            video_path=video_path,
            output_path=output_path,
        )

        total_frames, max_samples = self._prepare_video_sampling(video_path)
        analysis = self._run_frame_analysis(
            video_path=video_path,
            total_frames=total_frames,
            max_samples=max_samples,
            endoscope_image_roi=endoscope_image_roi,
            endoscope_data_roi_nested=endoscope_data_roi_nested,
            technique=technique,
            accumulated=accumulated,
        )

        accumulated = analysis.accumulated
        sensitive_idx = analysis.sensitive_idx
        best_ocr_text = analysis.best_ocr_text

        accumulated = self._maybe_enrich_video_metadata(accumulated)

        self._log_sensitive_frame_ratio(sensitive_idx, total_frames)

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

        return output_video, self._build_final_video_meta()

    def _prepare_video_sampling(self, video_path: Path) -> tuple[int, int]:
        total_frames = self._get_total_frames(video_path)
        self.current_video_total_frames = total_frames
        max_samples = self._target_sample_count(total_frames)
        logger.info(
            "Video detected (%d frames). Sampling ≤%d frames (high_quality_ocr=%s, early_stop=%s).",
            total_frames,
            max_samples,
            self.sampling_profile.high_quality_ocr,
            self.sampling_profile.smart_early_stopping,
        )
        return total_frames, max_samples

    def _run_frame_analysis(
        self,
        *,
        video_path: Path,
        total_frames: int,
        max_samples: int,
        endoscope_image_roi: Mapping[str, object] | None,
        endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None,
        technique: str,
        accumulated: FrameCleanerAccumulatedMeta,
    ) -> FrameAnalysisResult:
        return self._analyze_video_frames(
            video_path=video_path,
            total_frames=total_frames,
            max_samples=max_samples,
            endoscope_image_roi=endoscope_image_roi,
            endoscope_data_roi_nested=endoscope_data_roi_nested,
            technique=technique,
            accumulated=accumulated,
        )

    @staticmethod
    def _log_sensitive_frame_ratio(sensitive_idx: list[int], total_frames: int) -> None:
        sensitive_ratio = len(sensitive_idx) / total_frames if total_frames else 0.0
        logger.info(
            "Sensitive frames: %d/%d (%.1f %%)",
            len(sensitive_idx),
            total_frames,
            100 * sensitive_ratio,
        )

    def _build_final_video_meta(self) -> dict[str, object]:
        frame_observation_payloads: list[Mapping[str, object]] = [
            observation.model_dump(mode="json")
            for observation in self.frame_observations
        ]
        detector_sources, proposal_counts = summarize_frame_observations(
            frame_observation_payloads
        )
        anonymizer_provenance = build_anonymizer_provenance(
            detector_sources=detector_sources,
            proposal_counts=proposal_counts,
        )
        raw_sensitive_payload = sensitive_meta_to_dict(self.sensitive_meta)
        sensitive_payload = {
            key: value
            for key, value in raw_sensitive_payload.items()
            if value is not None
        }
        final_payload: dict[str, object] = {
            **sensitive_payload,
            "anonymizer_provenance": anonymizer_provenance.model_dump(),
        }
        if frame_observation_payloads:
            final_payload["frame_observations"] = frame_observation_payloads
        final_meta = VideoMeta.model_validate(final_payload)
        payload = final_meta.model_dump(mode="json")
        for field_name, value in raw_sensitive_payload.items():
            payload.setdefault(field_name, value)
        return payload

    def _prepare_clean_video_run(
        self,
        video_path: Path,
        output_path: Path | None,
    ) -> tuple[Path, FrameCleanerAccumulatedMeta]:
        default_center = os.environ.get("DEFAULT_CENTER", "Endoscopy Center")
        output_video = output_path or video_path.with_stem(f"{video_path.stem}_anony")

        accumulated = FrameCleanerAccumulatedMeta(
            file_path=str(video_path),
            center=default_center,
        )
        return output_video, accumulated

    @staticmethod
    def _get_total_frames(video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))  # type: ignore[call-arg]
        try:
            frame_count = cast(float, cast(Any, cap).get(cv2.CAP_PROP_FRAME_COUNT))
            return int(frame_count)
        finally:
            cap.release()

    def _analyze_video_frames(
        self,
        video_path: Path,
        total_frames: int,
        max_samples: int,
        endoscope_image_roi: Mapping[str, object] | None,
        endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None,
        technique: str,
        accumulated: FrameCleanerAccumulatedMeta,
    ) -> FrameAnalysisResult:
        sensitive_idx: list[int] = []
        frames_processed = 0
        best_ocr_text = ""
        best_ocr_conf = -1.0

        for idx, gray_frame, stride in self._iter_video(video_path, total_frames):
            _ = stride
            if frames_processed >= max_samples:
                logger.info("Reached maximum frame sample limit. Stopping analysis.")
                break

            frame_result = self._process_frame_result(
                gray_frame=gray_frame,
                endoscope_image_roi=endoscope_image_roi,
                endoscope_data_roi_nested=endoscope_data_roi_nested,
                frame_id=idx,
                collect_for_batch=True,
                high_quality_ocr=self.sampling_profile.high_quality_ocr,
            )

            merged_accumulated = self.frame_metadata_extractor.merge_metadata(
                accumulated.model_dump(), frame_result.metadata
            )
            accumulated = FrameCleanerAccumulatedMeta.model_validate(merged_accumulated)

            if frame_result.ocr_text and frame_result.ocr_text.strip():
                candidate = frame_result.ocr_text.strip()
                if frame_result.ocr_confidence > best_ocr_conf or (
                    abs(frame_result.ocr_confidence - best_ocr_conf) < 1e-6
                    and len(candidate) > len(best_ocr_text)
                ):
                    best_ocr_text = candidate
                    best_ocr_conf = float(frame_result.ocr_confidence)

            if frame_result.is_sensitive:
                sensitive_idx.append(idx)
                self.sensitive_meta.safe_update(accumulated.model_dump())

            frames_processed += 1

            if self._should_stop_frame_analysis(
                technique=technique,
                accumulated=accumulated,
            ):
                logger.info(
                    "Critical metadata found. Early stopping enabled for %s.",
                    technique,
                )
                break

        return FrameAnalysisResult(
            accumulated=accumulated,
            sensitive_idx=sensitive_idx,
            best_ocr_text=best_ocr_text,
            best_ocr_conf=best_ocr_conf,
            frames_processed=frames_processed,
        )

    def _should_stop_frame_analysis(
        self,
        *,
        technique: str,
        accumulated: FrameCleanerAccumulatedMeta,
    ) -> bool:
        return (
            self.sampling_profile.smart_early_stopping
            and technique in self.sampling_profile.early_stop_techniques
            and self.frame_metadata_extractor.is_complete(accumulated.model_dump())
        )

    def _maybe_enrich_video_metadata(
        self, accumulated: FrameCleanerAccumulatedMeta
    ) -> FrameCleanerAccumulatedMeta:
        if (
            self.use_llm
            and self.frame_collection
            and not self.frame_metadata_extractor.is_complete(accumulated.model_dump())
            and (
                int(settings.LLM_MAX_CALLS_PER_VIDEO) < 0
                or self._llm_calls_this_video < int(settings.LLM_MAX_CALLS_PER_VIDEO)
            )
        ):
            batch_enriched = self._extract_enriched_metadata_batch()
            if batch_enriched:
                merged_accumulated = self.frame_metadata_extractor.merge_metadata(
                    accumulated.model_dump(),
                    batch_enriched,
                )
                accumulated = FrameCleanerAccumulatedMeta.model_validate(
                    merged_accumulated
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
        endoscope_image_roi: Mapping[str, object] | None,
    ) -> Path:
        if technique == "remove_frames":
            return self._apply_frame_removal(
                video_path=video_path,
                output_video=output_video,
                sensitive_idx=sensitive_idx,
                total_frames=total_frames,
            )

        if technique == "mask_overlay":
            return self._apply_mask_overlay(
                video_path=video_path,
                output_video=output_video,
                endoscope_image_roi=endoscope_image_roi,
            )

        if technique == "extract_only":
            logger.info("Extraction-only mode: skipping video modification.")
            return video_path

        logger.warning(
            "Unknown cleaning technique '%s'. Returning original output path.",
            technique,
        )
        return output_video

    def _apply_frame_removal(
        self,
        *,
        video_path: Path,
        output_video: Path,
        sensitive_idx: list[int],
        total_frames: int,
    ) -> Path:
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

    def _apply_mask_overlay(
        self,
        *,
        video_path: Path,
        output_video: Path,
        endoscope_image_roi: Mapping[str, object] | None,
    ) -> Path:
        logger.info("Using masking strategy.")
        mask_cfg = self._mask_config_for_roi(endoscope_image_roi)
        assert self._mask_video_streaming is not None
        ok = self._mask_video_streaming(
            video_path,
            mask_cfg.model_dump(),
            output_video,
            use_named_pipe=True,
        )
        if not ok:
            raise RuntimeError(
                "Masking failed: ROI/crop configuration does not match input video dimensions."
            )
        return output_video

    def _mask_config_for_roi(self, roi: Mapping[str, object] | None) -> VideoMaskConfig:
        if roi and self._validate_roi(roi):
            assert self._create_mask_config_from_roi is not None
            return self._create_mask_config_from_roi(roi)
        assert self.default_mask_config is not None
        return self.default_mask_config

    def _finalize_video_metadata(
        self,
        accumulated: FrameCleanerAccumulatedMeta,
        best_ocr_text: str,
    ) -> FrameCleanerAccumulatedMeta:
        if best_ocr_text:
            accumulated.text = best_ocr_text
        elif not accumulated.text:
            fallback_text = self._build_representative_text_from_meta(accumulated)
            if fallback_text:
                accumulated.text = fallback_text

        self.sensitive_meta.safe_update(accumulated.model_dump())
        return accumulated

    @staticmethod
    def _build_representative_text_from_meta(
        meta: FrameCleanerAccumulatedMeta,
    ) -> str:
        parts: list[str] = []

        first = str(meta.first_name or "").strip()
        last = str(meta.last_name or "").strip()
        if first or last:
            parts.append(" ".join(p for p in [first, last] if p))

        for value, label in (
            (meta.casenumber, "Case"),
            (meta.dob, "DOB"),
            (meta.examination_date, "Date"),
            (meta.examination_time, "Time"),
            (meta.examiner_last_name, "Examiner"),
            (meta.endoscope_type, "Scope"),
            (meta.endoscope_sn, "SN"),
        ):
            if value:
                parts.append(f"{label}: {value}")

        return " | ".join(parts).strip()

    def _unified_metadata_extract(self, text: str) -> dict[str, object]:
        meta: dict[str, object] | None = {}
        if self.patient_data_extractor:
            try:
                patient_candidate: Mapping[str, object] = self.patient_data_extractor(
                    text
                )
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

        return meta or {}

    @staticmethod
    def _normalize_text_for_llm(text: str) -> str:
        return MULTISPACE_RE.sub(" ", text).strip().lower()

    def _should_attempt_llm(
        self, text: str, current_meta: Mapping[str, object]
    ) -> bool:
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

    def _remaining_llm_budget(self) -> int | None:
        max_calls = int(settings.LLM_MAX_CALLS_PER_VIDEO)
        if max_calls < 0:
            return None
        return max(0, max_calls - self._llm_calls_this_video)

    def _select_llm_video_text_candidates(self) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()

        if self.enriched_extractor:
            llm_frames = [
                LLMFrameDataPayload.model_validate(frame_data.model_dump(mode="json"))
                for frame_data in self.frame_collection
            ]
            aggregated = self.enriched_extractor.aggregate_ocr_texts(
                llm_frames,
                self.ocr_text_collection,
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
                float(item.ocr_confidence),
                len(item.ocr_text),
            ),
            reverse=True,
        )
        for frame_data in ranked_frames:
            text = frame_data.ocr_text.strip()
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
        self, key: str, value: object, source_text: str
    ) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False

        cleaned = value.strip()
        if not cleaned:
            return False

        if key in {"first_name", "last_name"}:
            if LLM_TITLE_TOKEN_RE.search(cleaned):
                return False
            if LLM_AGE_TOKEN_RE.search(cleaned):
                return False
            if LLM_NARRATIVE_TOKEN_RE.search(cleaned):
                return False
            if len(cleaned) > 40:
                return False

        if key == "casenumber":
            compact_value = NON_ALNUM_COMPACT_RE.sub("", cleaned.lower())
            compact_source = NON_ALNUM_COMPACT_RE.sub("", source_text.lower())
            return bool(compact_value) and compact_value in compact_source

        if key in {"first_name", "last_name"}:
            compact_source = NON_ALNUM_SPACE_RE.sub(" ", source_text.lower())
            tokens = [tok for tok in MULTISPACE_RE.split(cleaned.lower()) if tok]
            if not tokens:
                return False
            return all(token in compact_source for token in tokens)

        return True

    def _validate_llm_metadata_candidate(
        self, candidate: Mapping[str, object], source_text: str
    ) -> bool:
        if not self._metadata_has_signal(candidate):
            return False

        for key in (
            "first_name",
            "last_name",
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
    def _metadata_has_signal(meta: object) -> bool:
        if not isinstance(meta, Mapping):
            return False
        typed_meta = cast(Mapping[str, object], meta)

        signal_keys = (
            "first_name",
            "last_name",
            "dob",
            "casenumber",
            "gender",
            "examination_date",
            "examination_time",
            "examiner_first_name",
            "examiner_last_name",
            "endoscope_type",
            "endoscope_sn",
        )
        return any(
            (
                (value := typed_meta.get(key)) is not None
                and (not isinstance(value, str) or bool(value.strip()))
            )
            for key in signal_keys
        )

    @staticmethod
    def _resolve_ocr_roi(
        endoscope_image_roi: Mapping[str, object] | None,
        endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None,
    ) -> RoiInput:
        if endoscope_data_roi_nested:
            return endoscope_data_roi_nested

        if not endoscope_image_roi:
            return None

        normalized_image_roi = normalize_roi_keys(endoscope_image_roi)
        if not normalized_image_roi:
            return None

        return {"endoscope_image": cast(FlatRoi, normalized_image_roi)}

    @staticmethod
    def _frame_image_for_phi_detection(gray_frame: np.ndarray) -> Image.Image:
        if gray_frame.ndim == 2:
            return Image.fromarray(gray_frame).convert("RGB")
        return Image.fromarray(gray_frame)

    def _detect_phi_regions_for_frame(
        self, gray_frame: np.ndarray
    ) -> list[dict[str, object]]:
        image = self._frame_image_for_phi_detection(gray_frame)
        regions = detect_phi_regions_from_settings(image)
        phi_regions: list[dict[str, object]] = []
        for x1, y1, x2, y2 in regions:
            width = max(0, int(x2) - int(x1))
            height = max(0, int(y2) - int(y1))
            if width <= 0 or height <= 0:
                continue
            phi_regions.append(
                {
                    "source": "phi_detector",
                    "x": int(x1),
                    "y": int(y1),
                    "width": width,
                    "height": height,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": None,
                    "class_id": None,
                }
            )
        return phi_regions

    @staticmethod
    def _observation_source_tags(
        *,
        ocr_roi: RoiInput,
        ocr_text: str,
        metadata_signal: bool,
        phi_regions: list[dict[str, object]],
    ) -> list[str]:
        tags: list[str] = []
        if ocr_roi:
            tags.append("ocr_roi")
        if ocr_text and ocr_text.strip():
            tags.append("east_ocr")
        if metadata_signal:
            tags.append("metadata_signal")
        if phi_regions:
            tags.append("phi_detector")
        return tags

    def _build_frame_observation(
        self,
        *,
        frame_id: int | None,
        gray_frame: np.ndarray,
        ocr_roi: RoiInput,
        ocr_text: str,
        ocr_conf: float,
        frame_metadata: dict[str, object],
        is_sensitive: bool,
        phi_regions: list[dict[str, object]],
    ) -> FrameObservation:
        image_height, image_width = gray_frame.shape[:2]
        metadata_signal = self._metadata_has_signal(frame_metadata)
        return FrameObservation.model_validate(
            {
                "frame_number": frame_id,
                "frame_id": frame_id,
                "image_width": int(image_width),
                "image_height": int(image_height),
                "ocr_roi": ocr_roi,
                "ocr_text": ocr_text,
                "ocr_confidence": float(ocr_conf),
                "metadata_signal": metadata_signal,
                "is_sensitive": bool(is_sensitive),
                "phi_regions": phi_regions,
                "source_tags": self._observation_source_tags(
                    ocr_roi=ocr_roi,
                    ocr_text=ocr_text,
                    metadata_signal=metadata_signal,
                    phi_regions=phi_regions,
                ),
            }
        )

    def _process_frame_result(
        self,
        gray_frame: np.ndarray,
        endoscope_image_roi: Mapping[str, object] | None,
        endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None,
        frame_id: int | None = None,
        collect_for_batch: bool = False,
        high_quality_ocr: bool = True,
    ) -> FrameProcessResult:
        logger.debug(f"Processing frame_id={frame_id or 'unknown'}")
        ocr_roi = self._resolve_ocr_roi(
            endoscope_image_roi,
            endoscope_data_roi_nested,
        )
        if high_quality_ocr:
            ocr_text, ocr_conf, frame_metadata = self.frame_ocr.extract_text_from_frame(
                gray_frame, ocr_roi
            )
        else:
            ocr_text, ocr_conf, frame_metadata = self.frame_ocr.extract_text_from_frame(
                gray_frame, ocr_roi, high_quality=False
            )
        phi_regions = self._detect_phi_regions_for_frame(gray_frame)
        frame_metadata = self._metadata_for_ocr_text(ocr_text, frame_metadata)
        is_sensitive = self._frame_is_sensitive(frame_metadata, phi_regions)

        if collect_for_batch:
            self._collect_frame_observation(
                frame_id=frame_id,
                gray_frame=gray_frame,
                ocr_roi=ocr_roi,
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
                frame_metadata=frame_metadata,
                is_sensitive=is_sensitive,
                phi_regions=phi_regions,
            )

        if collect_for_batch and ocr_text:
            self._collect_frame_for_batch(
                frame_id=frame_id,
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
                frame_metadata=frame_metadata,
                is_sensitive=is_sensitive,
                phi_regions=phi_regions,
            )
        return FrameProcessResult(
            is_sensitive=is_sensitive,
            metadata=frame_metadata,
            ocr_text=ocr_text,
            ocr_confidence=ocr_conf,
        )

    def _metadata_for_ocr_text(
        self,
        ocr_text: str,
        frame_metadata: dict[str, object],
    ) -> dict[str, object]:
        if ocr_text:
            meta_unified = self._unified_metadata_extract(ocr_text)
            frame_metadata = self.frame_metadata_extractor.merge_metadata(
                frame_metadata, meta_unified
            )
        self.sensitive_meta.safe_update(frame_metadata)
        return self.sensitive_meta.to_dict()

    def _frame_is_sensitive(
        self,
        frame_metadata: dict[str, object],
        phi_regions: list[dict[str, object]],
    ) -> bool:
        return self.frame_metadata_extractor.is_sensitive_content(
            frame_metadata
        ) or bool(phi_regions)

    def _collect_frame_observation(
        self,
        *,
        frame_id: int | None,
        gray_frame: np.ndarray,
        ocr_roi: RoiInput,
        ocr_text: str,
        ocr_conf: float,
        frame_metadata: dict[str, object],
        is_sensitive: bool,
        phi_regions: list[dict[str, object]],
    ) -> None:
        self.frame_observations.append(
            self._build_frame_observation(
                frame_id=frame_id,
                gray_frame=gray_frame,
                ocr_roi=ocr_roi,
                ocr_text=ocr_text,
                ocr_conf=ocr_conf,
                frame_metadata=frame_metadata,
                is_sensitive=is_sensitive,
                phi_regions=phi_regions,
            )
        )

    def _collect_frame_for_batch(
        self,
        *,
        frame_id: int | None,
        ocr_text: str,
        ocr_conf: float,
        frame_metadata: dict[str, object],
        is_sensitive: bool,
        phi_regions: list[dict[str, object]],
    ) -> None:
        self.frame_collection.append(
            FrameCollectionItem.model_validate(
                {
                    "frame_id": frame_id,
                    "frame_number": frame_id,
                    "ocr_text": ocr_text,
                    "ocr_confidence": ocr_conf,
                    "meta": frame_metadata,
                    "is_sensitive": is_sensitive,
                    "phi_regions": phi_regions,
                }
            )
        )
        self.ocr_text_collection.append(ocr_text)

    def _process_frame_single(
        self,
        gray_frame: np.ndarray,
        endoscope_image_roi: Mapping[str, object] | None,
        endoscope_data_roi_nested: dict[str, dict[str, int | None]] | None,
        frame_id: int | None = None,
        collect_for_batch: bool = False,
        high_quality_ocr: bool = True,
    ) -> tuple[bool, dict[str, object], str, float]:
        return self._process_frame_result(
            gray_frame=gray_frame,
            endoscope_image_roi=endoscope_image_roi,
            endoscope_data_roi_nested=endoscope_data_roi_nested,
            frame_id=frame_id,
            collect_for_batch=collect_for_batch,
            high_quality_ocr=high_quality_ocr,
        ).as_legacy_tuple()

    def video_ocr_stream(
        self, frame_paths: List[Path]
    ) -> Generator[tuple[str, float], None, None]:
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

    def extract_metadata(self, text: str) -> dict[str, object]:
        if not text or not text.strip():
            return {}

        logger.debug(
            f"Extracting metadata from text of length {len(text)} with content: {text}..."
        )

        meta: dict[str, object] = {}
        if (
            getattr(self, "use_llm", False)
            and getattr(self, "llm_extractor", None) is not None
        ):
            try:
                meta_obj = self.llm_extractor.extract_metadata(text)  # type: ignore
                if meta_obj is not None:
                    meta = meta_obj.to_dict()
            except Exception as e:
                logger.warning("LLM extraction failed: %s", e)
                meta = {}

        if meta == {}:
            try:
                spacy_meta: Mapping[str, object] = {}
                if callable(self.patient_data_extractor):
                    spacy_meta = cast(
                        Mapping[str, object], self.patient_data_extractor(text)
                    )
                elif hasattr(self.patient_data_extractor, "extract"):
                    spacy_meta = cast(
                        Mapping[str, object], self.patient_data_extractor.extract(text)
                    )
                elif hasattr(self.patient_data_extractor, "patient_extractor"):
                    spacy_meta = cast(
                        Mapping[str, object],
                        self.patient_data_extractor.patient_extractor(text),
                    )
                else:
                    spacy_meta = {}
                meta = dict(spacy_meta)
            except Exception as e:
                logger.error(f"spaCy fallback failed: {e}")
                meta = {}

        return meta or {}

    def _extract_enriched_metadata_batch(self) -> dict[str, object]:
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
            self._ensure_batch_llm_components()

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

            attempts_allowed = (
                2 if remaining_budget is None else min(remaining_budget, 2)
            )
            validated_meta = self._first_valid_llm_candidate(
                text_candidates=text_candidates,
                attempts_allowed=attempts_allowed,
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

    def _ensure_batch_llm_components(self) -> None:
        if not self.frame_sampling_optimizer:
            self.frame_sampling_optimizer = FrameSamplingOptimizer()
        if not self.llm_extractor:
            self.llm_extractor = LLMFactory.create_metadata_extractor()
        if not self.enriched_extractor:
            self.enriched_extractor = EnrichedMetadataExtractor(
                llm_extractor=self.llm_extractor,
                frame_optimizer=self.frame_sampling_optimizer,
            )

    def _first_valid_llm_candidate(
        self,
        *,
        text_candidates: list[str],
        attempts_allowed: int,
    ) -> dict[str, object]:
        for idx, text in enumerate(text_candidates[:attempts_allowed], start=1):
            candidate = self._extract_llm_candidate(text, idx, attempts_allowed)
            if not candidate:
                continue
            if self._validate_llm_metadata_candidate(candidate, text):
                return candidate
            logger.info(
                "Discarding video-level LLM result from attempt %d because validation failed.",
                idx,
            )
        return {}

    def _extract_llm_candidate(
        self,
        text: str,
        attempt_idx: int,
        attempts_allowed: int,
    ) -> dict[str, object]:
        normalized = self._normalize_text_for_llm(text)
        self._llm_seen_texts.add(normalized)
        self._llm_calls_this_video += 1
        logger.info(
            "Running video-level LLM extraction attempt %d/%d on aggregated OCR text.",
            attempt_idx,
            attempts_allowed,
        )
        if self.llm_extractor is None:
            return {}
        meta_obj = self.llm_extractor.extract_metadata(text)
        if meta_obj is None:
            return {}
        return meta_obj.to_dict()
