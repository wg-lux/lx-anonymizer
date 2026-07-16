from __future__ import annotations

import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Sequence

import pytest

import manage
from lx_anonymizer.evaluation.ocr_backend_matrix import (
    BoundingBox,
    EvaluationCanvas,
    EvaluationResult,
    GoldenSetItem,
    GroundTruth,
    InputType,
    OcrBackendMatrixEvaluator,
    PHI_FIELDS,
    PipelineId,
    PipelinePrediction,
    PipelineSpec,
    SummaryRow,
    UtilityMetrics,
    aggregate_results,
    audit_golden_set,
    calculate_utility_metrics,
    load_golden_set,
)
from lx_anonymizer.ocr.ocr_ensemble import OcrEngine, normalize_ocr_selection_score


def _ground_truth(
    *,
    first_name: str | None = "Alice",
    last_name: str | None = "Doe",
    dob: str | None = "1980-01-02",
    casenumber: str | None = "E 123",
    examination_date: str | None = "2024-03-04",
    boxes: Sequence[BoundingBox] = (BoundingBox(10, 20, 40, 60),),
    text: str = "Patient Alice Doe DOB 1980-01-02 Case E 123 2024-03-04",
    image_dimensions: Sequence[EvaluationCanvas] = (EvaluationCanvas(100, 100),),
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
        image_dimensions=tuple(image_dimensions),
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


def test_tesseract_selection_confidence_is_normalized_to_unit_interval() -> None:
    assert normalize_ocr_selection_score(OcrEngine.TESSERACT, 82.0) == 0.82
    assert normalize_ocr_selection_score(OcrEngine.TROCR, 0.9) == 0.9
    assert normalize_ocr_selection_score(OcrEngine.DONUT, 0.75) == 0.75
    with pytest.raises(ValueError, match="cannot be negative"):
        normalize_ocr_selection_score(OcrEngine.TESSERACT, -1.0)


def test_result_json_hides_nested_ocr_text_without_sensitive_output() -> None:
    result = EvaluationResult(
        sample_id="frame-001",
        input_type=InputType.VIDEO_FRAME,
        source_path=Path("/tmp/frame.png"),
        pipeline_id=PipelineId.V5,
        pipeline_name="Production FrameOCR Cascade",
        status="ok",
        latency_seconds=0.1,
        rss_delta_mb=0.0,
        metrics=_result(PipelineId.V5, utility=1.0).metrics,
        prediction=PipelinePrediction(
            text="Patient Alice Doe",
            fields={"first_name": "Alice", "last_name": "Doe"},
            bounding_boxes=(),
            details={"metadata": {"roi_0": "Patient Alice Doe"}},
        ),
        error=None,
    )

    safe_row = result.to_json_dict()
    sensitive_row = result.to_json_dict(include_sensitive_output=True)

    assert result.prediction is not None
    assert "pipeline_details" not in safe_row
    assert "recognized_text" not in safe_row
    assert "predicted_fields" not in safe_row
    assert sensitive_row["pipeline_details"] == result.prediction.details
    assert sensitive_row["recognized_text"] == "Patient Alice Doe"


def test_default_matrix_includes_exact_production_frame_ocr_configuration() -> None:
    specs = OcrBackendMatrixEvaluator().default_pipeline_specs()

    production_spec = next(spec for spec in specs if spec.pipeline_id is PipelineId.V5)

    assert production_spec.input_type is InputType.VIDEO_FRAME
    assert production_spec.name == "Production FrameOCR Cascade"
    assert "Exact public FrameOCR production call" in production_spec.description

    detector_spec = next(spec for spec in specs if spec.pipeline_id is PipelineId.V6)
    assert detector_spec.input_type is InputType.VIDEO_FRAME
    assert detector_spec.name == "Production FrameOCR + PHI Detector"
    assert "additive regions" in detector_spec.description


def test_over_redaction_penalizes_non_phi_area() -> None:
    metrics = calculate_utility_metrics(
        prediction=_prediction(boxes=(BoundingBox(0, 0, 100, 100),)),
        ground_truth=_ground_truth(boxes=(BoundingBox(10, 20, 40, 60),)),
        latency_seconds=0.01,
        speed_target_seconds=1.0,
        process_succeeded=True,
        rss_delta_mb=0.0,
    )

    assert metrics.false_positive_region_fraction == 0.88
    assert metrics.non_phi_area_removed_fraction == 1.0
    assert metrics.over_redaction_score == 0.06


def test_small_correlated_golden_set_is_not_benchmark_ready() -> None:
    fixture_path = Path(__file__).parent / "assets" / "ocr_backend_golden_set.jsonl"

    audit = audit_golden_set(load_golden_set(fixture_path))

    assert audit.benchmark_ready is False
    assert audit.sample_count == 5
    assert audit.subject_count == 1
    assert audit.source_group_count == 2
    assert any("samples" in issue for issue in audit.issues)
    assert any("independent subjects" in issue for issue in audit.issues)


def test_evaluate_one_uses_fresh_metadata_extractors_for_every_sample(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeReportReader:
        pass

    fake_report_reader_module = ModuleType("lx_anonymizer.report_reader")
    fake_report_reader_module.ReportReader = FakeReportReader  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "lx_anonymizer.report_reader", fake_report_reader_module
    )
    evaluator = OcrBackendMatrixEvaluator()
    report_readers: list[object] = []
    frame_extractors: list[object] = []

    def observe_extractors(item: GoldenSetItem) -> PipelinePrediction:
        _ = item
        report_readers.append(
            evaluator._report_reader_instance()  # pyright: ignore[reportPrivateUsage]
        )
        frame_extractors.append(
            evaluator._frame_metadata_extractor_instance()  # pyright: ignore[reportPrivateUsage]
        )
        return _prediction()

    spec = PipelineSpec(
        pipeline_id=PipelineId.R1,
        input_type=InputType.TEXT_REPORT,
        name="isolation regression",
        description="Observe evaluator-owned extractor instances.",
        runner=observe_extractors,
    )
    source_path = tmp_path / "report.txt"
    source_path.write_text("Patient Alice Doe", encoding="utf-8")
    first_item = GoldenSetItem(
        sample_id="report-001",
        input_type=InputType.TEXT_REPORT,
        source_path=source_path,
        ground_truth=_ground_truth(),
        roi=None,
        tags=(),
    )
    second_item = GoldenSetItem(
        sample_id="report-002",
        input_type=InputType.TEXT_REPORT,
        source_path=source_path,
        ground_truth=_ground_truth(),
        roi=None,
        tags=(),
    )

    evaluator.evaluate_one(first_item, spec)
    evaluator.evaluate_one(second_item, spec)

    assert len(report_readers) == 2
    assert report_readers[0] is not report_readers[1]
    assert len(frame_extractors) == 2
    assert frame_extractors[0] is not frame_extractors[1]


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
        false_positive_region_fraction=0.0,
        non_phi_area_removed_fraction=0.0,
        over_redaction_score=1.0,
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
