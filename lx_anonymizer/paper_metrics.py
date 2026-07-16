from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

MetricStatus: TypeAlias = Literal[
    "requires_human_validation",
    "proposals_only",
    "validated",
    "not_evaluated",
    "dry_run",
]

PAPER_METRICS_SCHEMA_VERSION = "1.0"
SENSITIVE_META_SIGNAL_FIELDS: tuple[str, ...] = (
    "first_name",
    "last_name",
    "dob",
    "casenumber",
    "gender",
    "examination_date",
    "examination_time",
    "examiner_first_name",
    "examiner_last_name",
    "center",
    "endoscope_type",
    "endoscope_sn",
    "pseudo_patient",
    "pseudo_examination",
)


class _StrictMetricModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoRuntimeMetrics(_StrictMetricModel):
    schema_version: str = PAPER_METRICS_SCHEMA_VERSION
    total_seconds: float = Field(ge=0.0)
    staging_seconds: float = Field(ge=0.0)
    anonymizer_seconds: float = Field(ge=0.0)
    process_cpu_seconds: float = Field(ge=0.0)
    max_rss_kib_delta: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    frames_processed: int = Field(ge=0)
    sampled_frame_count: int = Field(ge=0)
    sensitive_frame_count: int = Field(ge=0)
    throughput_fps: float | None = Field(default=None, ge=0.0)
    sampled_frame_throughput_fps: float | None = Field(default=None, ge=0.0)
    technique: str
    high_quality_ocr: bool
    early_stopping_enabled: bool

    @field_validator(
        "total_seconds",
        "staging_seconds",
        "anonymizer_seconds",
        "process_cpu_seconds",
        "throughput_fps",
        "sampled_frame_throughput_fps",
    )
    @classmethod
    def require_finite_float(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("metric floats must be finite")
        return value


class TemporalAccumulationMetrics(_StrictMetricModel):
    schema_version: str = PAPER_METRICS_SCHEMA_VERSION
    total_frames: int = Field(ge=0)
    sampled_frame_count: int = Field(ge=0)
    frames_processed: int = Field(ge=0)
    sensitive_frame_count: int = Field(ge=0)
    metadata_signal_frame_count: int = Field(ge=0)
    ocr_text_frame_count: int = Field(ge=0)
    phi_region_proposal_count: int = Field(ge=0)
    mean_ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    max_ocr_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    populated_sensitive_fields: list[str] = Field(default_factory=list)
    detector_sources: list[str] = Field(default_factory=list)


class DeidentificationQualityMetrics(_StrictMetricModel):
    schema_version: str = PAPER_METRICS_SCHEMA_VERSION
    measurement_status: MetricStatus = "requires_human_validation"
    residual_ocr_match_count: int | None = Field(default=None, ge=0)
    residual_phi_detected_count: int | None = Field(default=None, ge=0)
    phi_region_false_negative_count: int | None = Field(default=None, ge=0)
    phi_bounding_box_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    phi_bounding_box_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    phi_region_proposal_count: int = Field(ge=0)
    metadata_signal_frame_count: int = Field(ge=0)
    ocr_text_frame_count: int = Field(ge=0)
    detector_sources: list[str] = Field(default_factory=list)
    validation_note: str = (
        "Residual OCR and PHI box precision/recall require post-processing OCR "
        "and human annotation comparison."
    )


class KPseudonymityReleaseControlMetrics(_StrictMetricModel):
    schema_version: str = PAPER_METRICS_SCHEMA_VERSION
    measurement_status: MetricStatus = "not_evaluated"
    evaluated_record_count: int = Field(default=0, ge=0)
    qi_fields: list[str] = Field(default_factory=list)
    sensitive_attribute: str | None = None
    k_threshold: int | None = Field(default=None, ge=1)
    l_threshold: int | None = Field(default=None, ge=1)
    t_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    qi_projection_count: int = Field(default=0, ge=0)
    equivalence_class_count: int = Field(default=0, ge=0)
    minimum_equivalence_class_size: int | None = Field(default=None, ge=0)
    underprotected_pattern_count: int = Field(default=0, ge=0)
    frequency_deficit_total: int = Field(default=0, ge=0)
    synthetic_padding_lower_bound: int = Field(default=0, ge=0)
    synthetic_padding_record_count: int | None = Field(default=None, ge=0)
    l_diversity_violation_count: int | None = Field(default=None, ge=0)
    t_closeness_violation_count: int | None = Field(default=None, ge=0)
    max_t_closeness_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    loss_size: int | None = Field(default=None, ge=0)
    loss_sensitive_modifications: int | None = Field(default=0, ge=0)
    loss_distribution_distance: float | None = Field(default=None, ge=0.0)
    real_record_mutation_allowed: bool = False
    stopping_reason: str = "not_evaluated_in_lx_anonymizer"


class VideoPaperEvaluationMetrics(_StrictMetricModel):
    schema_version: str = PAPER_METRICS_SCHEMA_VERSION
    runtime: VideoRuntimeMetrics
    temporal_accumulation: TemporalAccumulationMetrics
    deidentification_quality: DeidentificationQualityMetrics
    release_control: KPseudonymityReleaseControlMetrics = Field(
        default_factory=KPseudonymityReleaseControlMetrics
    )


class ReportPaperEvaluationMetrics(_StrictMetricModel):
    schema_version: str = PAPER_METRICS_SCHEMA_VERSION
    measurement_status: MetricStatus = "requires_human_validation"
    metadata_field_count: int = Field(ge=0)
    redaction_region_count: int = Field(ge=0)
    detector_sources: list[str] = Field(default_factory=list)
    residual_ocr_match_count: int | None = Field(default=None, ge=0)
    residual_phi_detected_count: int | None = Field(default=None, ge=0)
    release_control: KPseudonymityReleaseControlMetrics = Field(
        default_factory=KPseudonymityReleaseControlMetrics
    )
    validation_note: str = (
        "Report residual identifier counts require downstream validation against "
        "the anonymized report artifact."
    )


def build_video_paper_evaluation_metrics(
    *,
    total_seconds: float,
    staging_seconds: float,
    anonymizer_seconds: float,
    process_cpu_seconds: float,
    max_rss_kib_delta: int,
    total_frames: int,
    frames_processed: int,
    sensitive_frame_count: int,
    technique: str,
    high_quality_ocr: bool,
    early_stopping_enabled: bool,
    frame_observations: Sequence[Mapping[str, object]],
    sensitive_meta_payload: Mapping[str, object],
) -> VideoPaperEvaluationMetrics:
    sampled_frame_count = len(frame_observations)
    metadata_signal_frame_count = sum(
        1
        for observation in frame_observations
        if bool(observation.get("metadata_signal"))
    )
    ocr_text_frame_count = sum(
        1
        for observation in frame_observations
        if _is_nonblank(observation.get("ocr_text"))
    )
    phi_region_proposal_count = _count_phi_regions(frame_observations)
    detector_sources = _detector_sources(frame_observations)
    ocr_confidences = _ocr_confidences(frame_observations)

    temporal = TemporalAccumulationMetrics(
        total_frames=total_frames,
        sampled_frame_count=sampled_frame_count,
        frames_processed=frames_processed,
        sensitive_frame_count=sensitive_frame_count,
        metadata_signal_frame_count=metadata_signal_frame_count,
        ocr_text_frame_count=ocr_text_frame_count,
        phi_region_proposal_count=phi_region_proposal_count,
        mean_ocr_confidence=_mean_or_none(ocr_confidences),
        max_ocr_confidence=max(ocr_confidences) if ocr_confidences else None,
        populated_sensitive_fields=_populated_sensitive_fields(sensitive_meta_payload),
        detector_sources=detector_sources,
    )
    runtime = VideoRuntimeMetrics(
        total_seconds=total_seconds,
        staging_seconds=staging_seconds,
        anonymizer_seconds=anonymizer_seconds,
        process_cpu_seconds=process_cpu_seconds,
        max_rss_kib_delta=max_rss_kib_delta,
        total_frames=total_frames,
        frames_processed=frames_processed,
        sampled_frame_count=sampled_frame_count,
        sensitive_frame_count=sensitive_frame_count,
        throughput_fps=_rate_or_none(float(total_frames), anonymizer_seconds),
        sampled_frame_throughput_fps=_rate_or_none(
            float(frames_processed), anonymizer_seconds
        ),
        technique=technique,
        high_quality_ocr=high_quality_ocr,
        early_stopping_enabled=early_stopping_enabled,
    )
    quality = DeidentificationQualityMetrics(
        phi_region_proposal_count=phi_region_proposal_count,
        metadata_signal_frame_count=metadata_signal_frame_count,
        ocr_text_frame_count=ocr_text_frame_count,
        detector_sources=detector_sources,
    )
    return VideoPaperEvaluationMetrics(
        runtime=runtime,
        temporal_accumulation=temporal,
        deidentification_quality=quality,
    )


def build_report_paper_evaluation_metrics(
    report_meta: Mapping[str, object],
) -> ReportPaperEvaluationMetrics:
    redaction_summary = _mapping_or_empty(report_meta.get("redaction_summary"))
    anonymizer_provenance = _mapping_or_empty(report_meta.get("anonymizer_provenance"))
    return ReportPaperEvaluationMetrics(
        metadata_field_count=_metadata_field_count(report_meta),
        redaction_region_count=_int_from_mapping(
            redaction_summary, "redaction_region_count"
        ),
        detector_sources=_string_list_from_mapping(
            anonymizer_provenance, "detector_sources"
        ),
    )


def audit_k_pseudonymity_records(
    records: Sequence[Mapping[str, object]],
    *,
    qi_fields: Sequence[str],
    sensitive_attribute: str | None = None,
    k_threshold: int = 3,
    l_threshold: int | None = None,
    t_threshold: float | None = None,
) -> KPseudonymityReleaseControlMetrics:
    if not qi_fields:
        raise ValueError("qi_fields must not be empty")
    if k_threshold < 1:
        raise ValueError("k_threshold must be >= 1")
    if l_threshold is not None and l_threshold < 1:
        raise ValueError("l_threshold must be >= 1")
    if t_threshold is not None and not 0.0 <= t_threshold <= 1.0:
        raise ValueError("t_threshold must be within [0, 1]")
    if (l_threshold is not None or t_threshold is not None) and not sensitive_attribute:
        raise ValueError(
            "sensitive_attribute is required for l-diversity or t-closeness"
        )

    qi_field_list = [str(field) for field in qi_fields]
    subsets = _qi_subsets(qi_field_list)
    class_counts: list[int] = []
    underprotected_pattern_count = 0
    frequency_deficit_total = 0
    synthetic_padding_lower_bound = 0
    l_diversity_violation_count = 0 if l_threshold is not None else None
    t_closeness_violation_count = 0 if t_threshold is not None else None
    max_t_closeness_distance = 0.0 if t_threshold is not None else None
    global_sensitive_distribution = (
        _sensitive_distribution(records, sensitive_attribute)
        if sensitive_attribute and t_threshold is not None
        else {}
    )

    for subset in subsets:
        groups = _group_records(records, subset)
        class_counts.extend(groups.values())
        for count in groups.values():
            if count >= k_threshold:
                continue
            deficit = k_threshold - count
            underprotected_pattern_count += 1
            frequency_deficit_total += deficit
            synthetic_padding_lower_bound = max(synthetic_padding_lower_bound, deficit)

        if sensitive_attribute and l_threshold is not None:
            assert l_diversity_violation_count is not None
            l_diversity_violation_count += _l_diversity_violations(
                records,
                subset=subset,
                sensitive_attribute=sensitive_attribute,
                l_threshold=l_threshold,
            )

        if sensitive_attribute and t_threshold is not None:
            assert t_closeness_violation_count is not None
            assert max_t_closeness_distance is not None
            distances = _t_closeness_distances(
                records,
                subset=subset,
                sensitive_attribute=sensitive_attribute,
                global_distribution=global_sensitive_distribution,
            )
            for distance in distances:
                if distance > t_threshold:
                    t_closeness_violation_count += 1
                max_t_closeness_distance = max(max_t_closeness_distance, distance)

    loss_distribution_distance = max_t_closeness_distance
    return KPseudonymityReleaseControlMetrics(
        measurement_status="dry_run",
        evaluated_record_count=len(records),
        qi_fields=qi_field_list,
        sensitive_attribute=sensitive_attribute,
        k_threshold=k_threshold,
        l_threshold=l_threshold,
        t_threshold=t_threshold,
        qi_projection_count=len(subsets),
        equivalence_class_count=len(class_counts),
        minimum_equivalence_class_size=min(class_counts) if class_counts else None,
        underprotected_pattern_count=underprotected_pattern_count,
        frequency_deficit_total=frequency_deficit_total,
        synthetic_padding_lower_bound=synthetic_padding_lower_bound,
        l_diversity_violation_count=l_diversity_violation_count,
        t_closeness_violation_count=t_closeness_violation_count,
        max_t_closeness_distance=max_t_closeness_distance,
        loss_size=synthetic_padding_lower_bound,
        loss_sensitive_modifications=0,
        loss_distribution_distance=loss_distribution_distance,
        real_record_mutation_allowed=False,
        stopping_reason="dry_run_no_synthetic_serialization",
    )


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        return cast(Mapping[str, object], value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _int_from_mapping(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value >= 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _string_list_from_mapping(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    items: list[str] = []
    sequence_value = cast(Sequence[object], value)
    for item in sequence_value:
        if isinstance(item, str) and item:
            items.append(item)
    return sorted(set(items))


def _metadata_field_count(payload: Mapping[str, object]) -> int:
    return sum(
        1 for field in SENSITIVE_META_SIGNAL_FIELDS if _is_nonblank(payload.get(field))
    )


def _count_phi_regions(observations: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for observation in observations:
        regions = observation.get("phi_regions")
        if isinstance(regions, list):
            region_values = cast(list[object], regions)
            count += len(region_values)
    return count


def _detector_sources(observations: Sequence[Mapping[str, object]]) -> list[str]:
    sources: set[str] = set()
    for observation in observations:
        source_tags = observation.get("source_tags")
        if isinstance(source_tags, Sequence) and not isinstance(
            source_tags, (str, bytes)
        ):
            source_tag_values = cast(Sequence[object], source_tags)
            for source_tag in source_tag_values:
                if isinstance(source_tag, str) and source_tag:
                    sources.add(source_tag)
    return sorted(sources)


def _ocr_confidences(observations: Sequence[Mapping[str, object]]) -> list[float]:
    values: list[float] = []
    for observation in observations:
        value = observation.get("ocr_confidence")
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            confidence = float(value)
            if math.isfinite(confidence) and 0.0 <= confidence <= 1.0:
                values.append(confidence)
    return values


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _rate_or_none(numerator: float, seconds: float) -> float | None:
    if seconds <= 0.0:
        return None
    rate = numerator / seconds
    return rate if math.isfinite(rate) else None


def _populated_sensitive_fields(payload: Mapping[str, object]) -> list[str]:
    return [
        field
        for field in SENSITIVE_META_SIGNAL_FIELDS
        if _is_nonblank(payload.get(field))
    ]


def _is_nonblank(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        stripped = value.strip()
        return bool(stripped) and stripped.casefold() not in {
            "unknown",
            "none",
            "n/a",
            "na",
            "null",
            "undefined",
            "-",
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence_value = cast(Sequence[object], value)
        return len(sequence_value) > 0
    if isinstance(value, Mapping):
        mapping_value = cast(Mapping[object, object], value)
        return len(mapping_value) > 0
    return True


def _qi_subsets(qi_fields: Sequence[str]) -> list[tuple[str, ...]]:
    return [
        tuple(subset)
        for width in range(1, len(qi_fields) + 1)
        for subset in combinations(qi_fields, width)
    ]


def _group_records(
    records: Sequence[Mapping[str, object]], subset: Sequence[str]
) -> Counter[tuple[str, ...]]:
    groups: Counter[tuple[str, ...]] = Counter()
    for record in records:
        groups[tuple(_value_key(record.get(field)) for field in subset)] += 1
    return groups


def _l_diversity_violations(
    records: Sequence[Mapping[str, object]],
    *,
    subset: Sequence[str],
    sensitive_attribute: str,
    l_threshold: int,
) -> int:
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for record in records:
        key = tuple(_value_key(record.get(field)) for field in subset)
        groups[key].add(_value_key(record.get(sensitive_attribute)))
    return sum(
        1 for sensitive_values in groups.values() if len(sensitive_values) < l_threshold
    )


def _sensitive_distribution(
    records: Sequence[Mapping[str, object]],
    sensitive_attribute: str | None,
) -> dict[str, float]:
    if sensitive_attribute is None:
        return {}
    counts: Counter[str] = Counter()
    for record in records:
        counts[_value_key(record.get(sensitive_attribute))] += 1
    return _normalize_counts(counts)


def _t_closeness_distances(
    records: Sequence[Mapping[str, object]],
    *,
    subset: Sequence[str],
    sensitive_attribute: str,
    global_distribution: Mapping[str, float],
) -> list[float]:
    grouped_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    for record in records:
        key = tuple(_value_key(record.get(field)) for field in subset)
        grouped_counts[key][_value_key(record.get(sensitive_attribute))] += 1
    return [
        _total_variation_distance(
            _normalize_counts(counts),
            global_distribution,
        )
        for counts in grouped_counts.values()
    ]


def _normalize_counts(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {key: count / total for key, count in counts.items()}


def _total_variation_distance(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _value_key(value: object) -> str:
    if value is None:
        return "<NULL>"
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "<EMPTY>"
    return str(value)


__all__ = [
    "DeidentificationQualityMetrics",
    "KPseudonymityReleaseControlMetrics",
    "ReportPaperEvaluationMetrics",
    "TemporalAccumulationMetrics",
    "VideoPaperEvaluationMetrics",
    "VideoRuntimeMetrics",
    "audit_k_pseudonymity_records",
    "build_report_paper_evaluation_metrics",
    "build_video_paper_evaluation_metrics",
]
