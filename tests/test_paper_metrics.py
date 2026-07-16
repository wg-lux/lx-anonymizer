import math

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
    assert metrics.underprotected_pattern_count > 0
    assert metrics.synthetic_padding_lower_bound == 2
    assert metrics.loss_size == 2
    assert metrics.loss_sensitive_modifications == 0
    assert metrics.l_diversity_violation_count is not None
    assert metrics.l_diversity_violation_count > 0
    assert metrics.t_closeness_violation_count is not None
    assert metrics.t_closeness_violation_count > 0
    assert metrics.max_t_closeness_distance is not None
    assert metrics.max_t_closeness_distance > 0.0


def test_k_pseudonymity_requires_sensitive_attribute_for_l_or_t() -> None:
    with pytest.raises(ValueError, match="sensitive_attribute"):
        audit_k_pseudonymity_records(
            [{"center": "A", "diagnosis": "adenoma"}],
            qi_fields=("center",),
            l_threshold=2,
        )
