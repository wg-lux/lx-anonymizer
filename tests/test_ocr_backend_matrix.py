from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

import manage
from lx_anonymizer.evaluation.ocr_backend_matrix import (
    BoundingBox,
    EvaluationResult,
    GroundTruth,
    InputType,
    PHI_FIELDS,
    PipelineId,
    PipelinePrediction,
    SummaryRow,
    UtilityMetrics,
    aggregate_results,
    calculate_utility_metrics,
    load_golden_set,
)


def _ground_truth(
    *,
    first_name: str | None = "Alice",
    last_name: str | None = "Doe",
    dob: str | None = "1980-01-02",
    casenumber: str | None = "E 123",
    examination_date: str | None = "2024-03-04",
    boxes: Sequence[BoundingBox] = (BoundingBox(10, 20, 40, 60),),
    text: str = "Patient Alice Doe DOB 1980-01-02 Case E 123 2024-03-04",
) -> GroundTruth:
    return GroundTruth(
        fields={
            "first_name": first_name,
            "last_name": last_name,
            "dob": dob,
            "casenumber": casenumber,
            "examination_date": examination_date,
        },
        bounding_boxes=tuple(boxes),
        text=text,
    )


def _prediction(
    *,
    first_name: str | None = "Alice",
    last_name: str | None = "Doe",
    dob: str | None = "1980-01-02",
    casenumber: str | None = "E 123",
    examination_date: str | None = "2024-03-04",
    boxes: Sequence[BoundingBox] = (BoundingBox(10, 20, 40, 60),),
    text: str = "Patient Alice Doe DOB 1980-01-02 Case E 123 2024-03-04",
) -> PipelinePrediction:
    return PipelinePrediction(
        text=text,
        fields={
            "first_name": first_name,
            "last_name": last_name,
            "dob": dob,
            "casenumber": casenumber,
            "examination_date": examination_date,
        },
        bounding_boxes=tuple(boxes),
        details={},
    )


def test_load_golden_set_validates_absolute_manifest_schema(tmp_path: Path) -> None:
    source_path = tmp_path / "report.txt"
    source_path.write_text("Patient Alice Doe", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "report-001",
                "input_type": "report",
                "source_path": str(source_path),
                "ground_truth": {
                    "first_name": "Alice",
                    "last_name": "Doe",
                    "dob": "1980-01-02",
                    "casenumber": "E 123",
                    "examination_date": "2024-03-04",
                    "bounding_boxes": [{"x": 10, "y": 20, "width": 30, "height": 40}],
                    "text": "Patient Alice Doe",
                },
                "tags": ["native"],
            }
        ),
        encoding="utf-8",
    )

    items = load_golden_set(manifest_path)

    assert len(items) == 1
    item = items[0]
    assert item.sample_id == "report-001"
    assert item.input_type is InputType.TEXT_REPORT
    assert item.ground_truth.fields["first_name"] == "Alice"
    assert item.ground_truth.bounding_boxes == (BoundingBox(10, 20, 40, 60),)
    assert item.tags == ("native",)


def test_load_golden_set_rejects_missing_ground_truth_keys(tmp_path: Path) -> None:
    source_path = tmp_path / "frame.png"
    source_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "frame-001",
                "input_type": "video_frame",
                "source_path": str(source_path),
                "ground_truth": {
                    "first_name": "Alice",
                    "last_name": "Doe",
                    "dob": "1980-01-02",
                    "casenumber": "E 123",
                    "bounding_boxes": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="examination_date"):
        load_golden_set(manifest_path)


def test_phi_recall_hard_gate_zeroes_utility_when_identifier_is_missing() -> None:
    metrics = calculate_utility_metrics(
        prediction=_prediction(dob=None),
        ground_truth=_ground_truth(),
        latency_seconds=0.01,
        speed_target_seconds=1.0,
        process_succeeded=True,
        rss_delta_mb=0.0,
    )

    assert metrics.phi_field_recall == 0.8
    assert metrics.hard_gate_applied is True
    assert metrics.utility_score == 0.0


def test_clean_frame_without_phi_scores_as_complete_recall_for_empty_prediction() -> (
    None
):
    ground_truth = _ground_truth(
        first_name=None,
        last_name=None,
        dob=None,
        casenumber=None,
        examination_date=None,
        boxes=(),
        text="",
    )
    prediction = _prediction(
        first_name=None,
        last_name=None,
        dob=None,
        casenumber=None,
        examination_date=None,
        boxes=(),
        text="",
    )

    metrics = calculate_utility_metrics(
        prediction=prediction,
        ground_truth=ground_truth,
        latency_seconds=0.01,
        speed_target_seconds=1.0,
        process_succeeded=True,
        rss_delta_mb=0.0,
    )

    assert metrics.phi_field_recall == 1.0
    assert metrics.hard_gate_applied is False
    assert metrics.utility_score > 0.0


def test_aggregate_results_ranks_pipelines_by_utility_within_input_type() -> None:
    low = _result(PipelineId.R1, utility=0.2)
    high = _result(PipelineId.R2, utility=0.9)

    rows = aggregate_results([low, high])

    assert [row.pipeline_id for row in rows] == [PipelineId.R2, PipelineId.R1]
    assert isinstance(rows[0], SummaryRow)
    assert rows[0].mean_utility_score == 0.9


def test_manage_dispatches_evaluate_ocr_backend_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_args: list[str] = []

    def fake_main(argv: Sequence[str] | None = None) -> int:
        observed_args.extend(list(argv or ()))
        return 17

    monkeypatch.setattr(
        "lx_anonymizer.evaluation.ocr_backend_matrix.main",
        fake_main,
    )

    exit_code = manage.main(
        ["evaluate_ocr_backend_matrix", "--golden-set", "/tmp/manifest.jsonl"]
    )

    assert exit_code == 17
    assert observed_args == ["--golden-set", "/tmp/manifest.jsonl"]


def _result(pipeline_id: PipelineId, *, utility: float) -> EvaluationResult:
    metrics = UtilityMetrics(
        phi_field_recall=1.0,
        field_accuracy=1.0,
        text_score=1.0,
        box_coverage=1.0,
        speed_score=1.0,
        stability_score=1.0,
        utility_score=utility,
        hard_gate_applied=False,
    )
    return EvaluationResult(
        sample_id="report-001",
        input_type=InputType.TEXT_REPORT,
        source_path=Path("/tmp/report.txt"),
        pipeline_id=pipeline_id,
        pipeline_name=pipeline_id.value,
        status="ok",
        latency_seconds=0.01,
        rss_delta_mb=0.0,
        metrics=metrics,
        prediction=None,
        error=None,
    )


def test_phi_fields_are_the_required_manifest_identifiers() -> None:
    assert PHI_FIELDS == (
        "first_name",
        "last_name",
        "dob",
        "casenumber",
        "examination_date",
    )
