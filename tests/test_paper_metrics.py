import math
from copy import deepcopy

import pytest

from lx_anonymizer.paper_metrics import (
    KPseudonymityReleaseControlMetrics,
    audit_k_pseudonymity_records,
    build_report_paper_evaluation_metrics,
    build_video_paper_evaluation_metrics,
)
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def test_video_paper_metrics_summarize_runtime_and_frame_observations() -> None:
    metrics = build_video_paper_evaluation_metrics(
        total_seconds=4.0,
        staging_seconds=0.5,
        anonymizer_seconds=2.0,
        process_cpu_seconds=1.25,
        max_rss_kib_delta=128,
        total_frames=100,
        frames_processed=2,
        sensitive_frame_count=1,
        technique="mask_overlay",
        high_quality_ocr=True,
        early_stopping_enabled=False,
        frame_observations=[
            {
                "frame_number": 0,
                "ocr_text": "Patient: Max",
                "ocr_confidence": 0.8,
                "metadata_signal": True,
                "is_sensitive": True,
                "phi_regions": [{"source": "phi_detector"}],
                "source_tags": ["east_ocr", "metadata_signal", "phi_detector"],
            },
            {
                "frame_number": 30,
                "ocr_text": "",
                "ocr_confidence": 0.4,
                "metadata_signal": False,
                "is_sensitive": False,
                "phi_regions": [],
                "source_tags": [],
            },
        ],
        sensitive_meta_payload={
            "first_name": "Max",
            "last_name": "Mustermann",
            "dob": "1990-01-01",
        },
    )

    assert metrics.runtime.throughput_fps == 50.0
    assert metrics.runtime.sampled_frame_throughput_fps == 1.0
    assert metrics.temporal_accumulation.phi_region_proposal_count == 1
    assert metrics.temporal_accumulation.metadata_signal_frame_count == 1
    assert metrics.temporal_accumulation.ocr_text_frame_count == 1
    assert math.isclose(metrics.temporal_accumulation.mean_ocr_confidence or 0.0, 0.6)
    assert metrics.temporal_accumulation.populated_sensitive_fields == [
        "first_name",
        "last_name",
        "dob",
    ]
    assert metrics.deidentification_quality.measurement_status == (
        "requires_human_validation"
    )
    assert metrics.deidentification_quality.phi_bounding_box_precision is None


def test_report_paper_metrics_preserve_redaction_and_detector_counts() -> None:
    metrics = build_report_paper_evaluation_metrics(
        {
            "first_name": "Max",
            "last_name": "Mustermann",
            "redaction_summary": {"redaction_region_count": 3},
            "anonymizer_provenance": {"detector_sources": ["regex", "spacy", "regex"]},
        }
    )

    assert metrics.metadata_field_count == 2
    assert metrics.redaction_region_count == 3
    assert metrics.detector_sources == ["regex", "spacy"]
    assert metrics.residual_ocr_match_count is None


def test_report_reader_final_output_includes_report_and_paper_metrics() -> None:
    reader = ReportReader.__new__(ReportReader)
    reader.sensitive_meta = SensitiveMeta(first_name="Max", last_name="Mustermann")

    payload = reader._build_final_report_output_meta(  # pyright: ignore[reportPrivateUsage]
        {
            "first_name": "Max",
            "last_name": "Mustermann",
            "redaction_summary": {"redaction_region_count": 2},
        }
    )

    assert payload["first_name"] == "Max"
    assert payload["last_name"] == "Mustermann"
    assert payload["redaction_summary"] == {"redaction_region_count": 2}
    paper_metrics = payload["paper_evaluation_metrics"]
    assert isinstance(paper_metrics, dict)
    assert paper_metrics["metadata_field_count"] == 2
    assert paper_metrics["redaction_region_count"] == 2


def test_k_pseudonymity_dry_run_reports_frequency_diversity_and_t_closeness() -> None:
    records = [
        {
            "center": "A",
            "age_band": "50-59",
            "sex": "f",
            "diagnosis": "adenoma",
        },
        {
            "center": "A",
            "age_band": "50-59",
            "sex": "f",
            "diagnosis": "adenoma",
        },
        {
            "center": "B",
            "age_band": "70-79",
            "sex": "m",
            "diagnosis": "cancer",
        },
    ]

    metrics = audit_k_pseudonymity_records(
        records,
        qi_fields=("center", "age_band", "sex"),
        sensitive_attribute="diagnosis",
        k_threshold=3,
        l_threshold=2,
        t_threshold=0.2,
    )

    assert isinstance(metrics, KPseudonymityReleaseControlMetrics)
    assert metrics.measurement_status == "dry_run"
    assert metrics.evaluated_record_count == 3
    assert metrics.qi_projection_count == 7
    assert metrics.equivalence_class_count == 14
    assert metrics.minimum_equivalence_class_size == 1
    assert metrics.underprotected_pattern_count == 14
    assert metrics.frequency_deficit_total == 21
    assert metrics.synthetic_padding_lower_bound == 2
    assert metrics.loss_size == 2
    assert metrics.loss_sensitive_modifications == 0
    assert metrics.l_diversity_violation_count == 14
    assert metrics.t_closeness_violation_count == 14
    assert metrics.max_t_closeness_distance is not None
    assert math.isclose(metrics.max_t_closeness_distance, 2 / 3)


def test_k_pseudonymity_requires_sensitive_attribute_for_l_or_t() -> None:
    with pytest.raises(ValueError, match="sensitive_attribute"):
        audit_k_pseudonymity_records(
            [{"center": "A", "diagnosis": "adenoma"}],
            qi_fields=("center",),
            l_threshold=2,
        )


@pytest.mark.parametrize("l_threshold", [None, 1, 2])
@pytest.mark.parametrize("t_threshold, violations", [(None, None), (0.5, 2), (0.75, 0)])
def test_release_audit_exact_thresholds_and_no_mutation(
    l_threshold: int | None, t_threshold: float | None, violations: int | None
) -> None:
    records: list[dict[str, object]] = [
        {"center": " A ", "diagnosis": "x"},
        {"center": "A", "diagnosis": "x"},
        {"center": "B", "diagnosis": "y"},
        {"center": "C"},
    ]
    original = deepcopy(records)

    metrics = audit_k_pseudonymity_records(
        records,
        qi_fields=("center",),
        sensitive_attribute="diagnosis",
        k_threshold=2,
        l_threshold=l_threshold,
        t_threshold=t_threshold,
    )

    assert metrics == KPseudonymityReleaseControlMetrics(
        measurement_status="dry_run",
        evaluated_record_count=4,
        qi_fields=["center"],
        sensitive_attribute="diagnosis",
        k_threshold=2,
        l_threshold=l_threshold,
        t_threshold=t_threshold,
        qi_projection_count=1,
        equivalence_class_count=3,
        minimum_equivalence_class_size=1,
        underprotected_pattern_count=2,
        frequency_deficit_total=2,
        synthetic_padding_lower_bound=1,
        l_diversity_violation_count=None
        if l_threshold is None
        else (3 if l_threshold == 2 else 0),
        t_closeness_violation_count=violations,
        max_t_closeness_distance=None if t_threshold is None else 0.75,
        loss_size=1,
        loss_distribution_distance=None if t_threshold is None else 0.75,
        stopping_reason="dry_run_no_synthetic_serialization",
    )
    assert records == original


@pytest.mark.parametrize("enabled", [False, True])
def test_release_audit_empty_records(enabled: bool) -> None:
    metrics = audit_k_pseudonymity_records(
        [],
        qi_fields=("center", "age"),
        sensitive_attribute="diagnosis" if enabled else None,
        l_threshold=2 if enabled else None,
        t_threshold=0.0 if enabled else None,
    )

    assert metrics == KPseudonymityReleaseControlMetrics(
        measurement_status="dry_run",
        qi_fields=["center", "age"],
        sensitive_attribute="diagnosis" if enabled else None,
        k_threshold=3,
        l_threshold=2 if enabled else None,
        t_threshold=0.0 if enabled else None,
        qi_projection_count=3,
        l_diversity_violation_count=0 if enabled else None,
        t_closeness_violation_count=0 if enabled else None,
        max_t_closeness_distance=0.0 if enabled else None,
        loss_size=0,
        loss_distribution_distance=0.0 if enabled else None,
        stopping_reason="dry_run_no_synthetic_serialization",
    )


@pytest.mark.parametrize("values", [(None, None), ("", "  "), (1, "1"), (" A ", "A")])
def test_release_audit_preserves_group_value_normalization(
    values: tuple[object, object],
) -> None:
    records = [{"center": value, "diagnosis": value} for value in values]

    metrics = audit_k_pseudonymity_records(
        records,
        qi_fields=("center",),
        sensitive_attribute="diagnosis",
        k_threshold=2,
        l_threshold=2,
        t_threshold=0.0,
    )

    assert metrics.equivalence_class_count == 1
    assert metrics.minimum_equivalence_class_size == 2
    assert metrics.frequency_deficit_total == 0
    assert metrics.l_diversity_violation_count == 1
    assert metrics.t_closeness_violation_count == 0
    assert metrics.max_t_closeness_distance == 0.0


@pytest.mark.parametrize(
    "qi_fields,k_threshold,l_threshold,t_threshold,sensitive_attribute,message",
    [
        ((), 3, None, None, None, "qi_fields must not be empty"),
        (("center",), 0, None, None, None, "k_threshold must be >= 1"),
        (("center",), 3, 0, None, "diagnosis", "l_threshold must be >= 1"),
        *(
            (
                ("center",),
                3,
                None,
                value,
                "diagnosis",
                "t_threshold must be within [0, 1]",
            )
            for value in (-0.1, 1.1, math.nan, math.inf)
        ),
        *(
            (
                ("center",),
                3,
                l_value,
                t_value,
                attribute,
                "sensitive_attribute is required for l-diversity or t-closeness",
            )
            for l_value, t_value in ((2, None), (None, 0.0))
            for attribute in (None, "")
        ),
    ],
)
def test_release_audit_rejects_invalid_thresholds(
    qi_fields: tuple[str, ...],
    k_threshold: int,
    l_threshold: int | None,
    t_threshold: float | None,
    sensitive_attribute: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError) as error:
        audit_k_pseudonymity_records(
            [],
            qi_fields=qi_fields,
            k_threshold=k_threshold,
            l_threshold=l_threshold,
            t_threshold=t_threshold,
            sensitive_attribute=sensitive_attribute,
        )

    assert str(error.value) == message


def test_k_only_audit_does_not_inspect_sensitive_values() -> None:
    class UnreadableValue:
        def __str__(self) -> str:
            raise AssertionError("k-only auditing must not inspect sensitive values")

    records: list[dict[str, object]] = [
        {"diagnosis": UnreadableValue()},
        {"center": None, "diagnosis": UnreadableValue()},
    ]

    metrics = audit_k_pseudonymity_records(
        records,
        qi_fields=("center",),
        sensitive_attribute="diagnosis",
        k_threshold=2,
    )

    assert metrics.equivalence_class_count == 1
    assert metrics.minimum_equivalence_class_size == 2
    assert metrics.underprotected_pattern_count == 0
    assert metrics.synthetic_padding_lower_bound == 0
    assert metrics.sensitive_attribute == "diagnosis"
    assert metrics.l_diversity_violation_count is None
    assert metrics.t_closeness_violation_count is None


@pytest.mark.parametrize("reverse", [False, True])
def test_release_audit_counts_each_qi_projection_independently(reverse: bool) -> None:
    records = [
        {"center": "A", "age": "young", "diagnosis": "x"},
        {"center": "A", "age": "old", "diagnosis": "y"},
        {"center": "B", "age": "young", "diagnosis": "y"},
        {"center": "B", "age": "old", "diagnosis": "x"},
    ]

    metrics = audit_k_pseudonymity_records(
        records[::-1] if reverse else records,
        qi_fields=("center", "age"),
        sensitive_attribute="diagnosis",
        k_threshold=3,
        l_threshold=2,
        t_threshold=0.0,
    )

    assert metrics.qi_projection_count == 3
    assert metrics.equivalence_class_count == 8
    assert metrics.minimum_equivalence_class_size == 1
    assert metrics.underprotected_pattern_count == 8
    assert metrics.frequency_deficit_total == 12
    assert metrics.synthetic_padding_lower_bound == 2
    assert metrics.l_diversity_violation_count == 4
    assert metrics.t_closeness_violation_count == 4
    assert metrics.max_t_closeness_distance == 0.5
