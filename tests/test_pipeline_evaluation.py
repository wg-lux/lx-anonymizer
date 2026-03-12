from __future__ import annotations

from pathlib import Path

import pytest

from lx_anonymizer.evaluation.pipeline_evaluation import (
    evaluate_feedback_alignment,
    evaluate_records,
    load_records,
)


def test_load_records_supports_json_array_and_jsonl_with_trailing_commas(
    tmp_path: Path,
) -> None:
    array_path = tmp_path / "array.json"
    jsonl_path = tmp_path / "records.jsonl"

    array_path.write_text(
        '[{"file": "a", "report_id": "1", "first_name": "Alice"}]',
        encoding="utf-8",
    )
    jsonl_path.write_text(
        "\n".join(
            [
                '{"file": "a", "report_id": "1", "first_name": "Alice"},',
                '{"file": "b", "report_id": "2", "first_name": "Bob"}',
            ]
        ),
        encoding="utf-8",
    )

    array_records = load_records(array_path)
    jsonl_records = load_records(jsonl_path)

    assert len(array_records) == 1
    assert array_records[0]["first_name"] == "Alice"
    assert len(jsonl_records) == 2
    assert jsonl_records[1]["first_name"] == "Bob"


def test_evaluate_records_returns_expected_core_metrics() -> None:
    gold = [
        {"file": "f1", "report_id": "1", "first_name": "Alice", "last_name": "Doe"},
        {"file": "f1", "report_id": "2", "first_name": "Bob", "last_name": "Smith"},
        {"file": "f2", "report_id": "1", "first_name": "Carla", "last_name": None},
    ]
    predictions = [
        {"file": "f1", "report_id": "1", "first_name": "Dr. Alice", "last_name": "Doe"},
        {"file": "f1", "report_id": "2", "first_name": None, "last_name": "Smyth"},
        {"file": "f3", "report_id": "1", "first_name": "Extra", "last_name": "Person"},
    ]

    result = evaluate_records(
        predictions=predictions,
        gold=gold,
        fields=("first_name", "last_name"),
        scenario="unit-test",
    )

    assert result.matched_records == 2
    assert result.missing_in_predictions == 1
    assert result.extra_in_predictions == 1

    first_name = result.field_metrics["first_name"]
    last_name = result.field_metrics["last_name"]

    assert first_name.support == 2
    assert first_name.exact_match_rate == pytest.approx(0.5)
    assert first_name.presence_precision == pytest.approx(1.0)
    assert first_name.presence_recall == pytest.approx(0.5)

    assert last_name.support == 2
    assert last_name.exact_match_rate == pytest.approx(0.5)
    assert last_name.mean_similarity > 0.5
    assert last_name.presence_f1 == pytest.approx(1.0)

    assert result.macro_exact_match_rate == pytest.approx(0.5)


def test_evaluate_feedback_alignment_uses_text_change_when_no_explicit_prediction() -> (
    None
):
    predictions = [
        {
            "file": "f1",
            "report_id": "1",
            "text": "Patient Alice",
            "anonymized_text": "Patient [NAME]",
        },
        {
            "file": "f1",
            "report_id": "2",
            "text": "No PHI",
            "anonymized_text": "No PHI",
        },
        {
            "file": "f1",
            "report_id": "3",
            "predicted_changed": True,
        },
    ]
    feedback = [
        {"file": "f1", "report_id": "1", "did_anonymization_change": True},
        {"file": "f1", "report_id": "2", "did_anonymization_change": False},
        {"file": "f1", "report_id": "3", "did_anonymization_change": False},
    ]

    agreement = evaluate_feedback_alignment(predictions=predictions, feedback=feedback)

    assert agreement.evaluated_pairs == 3
    assert agreement.tp == 1
    assert agreement.tn == 1
    assert agreement.fp == 1
    assert agreement.fn == 0
    assert agreement.accuracy == pytest.approx(2 / 3)
    assert agreement.precision == pytest.approx(0.5)
    assert agreement.recall == pytest.approx(1.0)
    assert agreement.f1 == pytest.approx(2 / 3)
