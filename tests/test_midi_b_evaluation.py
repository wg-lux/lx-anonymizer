from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lx_anonymizer.evaluation import midi_b
from lx_anonymizer.evaluation.midi_b import (
    MidiBEvaluationError,
    evaluate_midi_b_pixel_detector,
    load_midi_b_pixel_annotations,
)


class _FakeDataset:
    SOPInstanceUID: str
    Modality: str
    PhotometricInterpretation: str
    pixel_array: np.ndarray

    def __init__(self, uid: str):
        self.SOPInstanceUID = uid
        self.Modality = "DX"
        self.PhotometricInterpretation = "MONOCHROME2"
        self.pixel_array = np.arange(10_000, dtype=np.uint16).reshape(100, 100)


class _FakePydicom:
    dataset: _FakeDataset

    def __init__(self, dataset: _FakeDataset):
        self.dataset = dataset

    def dcmread(self, _path: str | Path, **_kwargs: object) -> _FakeDataset:
        return self.dataset


class _FixedDetector:
    name = "fixed"

    def detect(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        del image
        return [(10, 10, 30, 30), (70, 70, 90, 90)]


def _write_answer_db(path: Path, uid: str = "1.2.3") -> None:
    answer_data = {
        "0": {
            "action": "<pixels_hidden>",
            "action_text": "<"
            + json.dumps(
                {
                    "text": "PATIENT",
                    "top_left": [10, 10],
                    "bottom_right": [30, 30],
                }
            )
            + ">",
        },
        "1": {
            "action": "<pixels_hidden>",
            "action_text": "<"
            + json.dumps(
                {
                    "text": "IDENTIFIER",
                    "top_left": [40, 40],
                    "bottom_right": [60, 60],
                }
            )
            + ">",
        },
        "2": {"action": "<tag_retained>", "action_text": "<{}>"},
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE answer_data (SOPInstanceUID TEXT, AnswerData TEXT)"
        )
        connection.execute(
            "INSERT INTO answer_data VALUES (?, ?)", (uid, json.dumps(answer_data))
        )


def test_load_midi_b_pixel_annotations_extracts_only_hidden_pixels(
    tmp_path: Path,
) -> None:
    answer_db = tmp_path / "answers.db"
    _write_answer_db(answer_db)

    annotations = load_midi_b_pixel_annotations(answer_db)

    assert [annotation.text for annotation in annotations] == [
        "PATIENT",
        "IDENTIFIER",
    ]
    assert annotations[0].box == (10, 10, 30, 30)


def test_evaluate_midi_b_pixel_detector_reports_detection_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_db = tmp_path / "answers.db"
    _write_answer_db(answer_db)
    dataset_root = tmp_path / "dicom"
    dataset_root.mkdir()
    (dataset_root / "sample.dcm").write_bytes(b"DICOM")
    fake_pydicom = _FakePydicom(_FakeDataset("1.2.3"))
    monkeypatch.setattr(midi_b, "_load_pydicom", lambda: fake_pydicom)

    report = evaluate_midi_b_pixel_detector(
        dataset_root=dataset_root,
        answer_db=answer_db,
        detector=_FixedDetector(),
        iou_threshold=0.5,
    )

    assert report.evaluated_instances == 1
    assert report.overall.annotations == 2
    assert report.overall.predictions == 2
    assert report.overall.true_positives == 1
    assert report.overall.precision == 0.5
    assert report.overall.recall == 0.5
    assert report.overall.f1 == 0.5
    assert report.overall.mean_best_iou == 0.5
    assert report.overall.mean_ground_truth_coverage == 0.5
    assert report.by_modality["DX"].recall == 0.5


def test_load_midi_b_pixel_annotations_requires_expected_schema(
    tmp_path: Path,
) -> None:
    answer_db = tmp_path / "answers.db"
    with sqlite3.connect(answer_db) as connection:
        connection.execute("CREATE TABLE answer_data (wrong TEXT)")

    with pytest.raises(MidiBEvaluationError, match="must contain"):
        load_midi_b_pixel_annotations(answer_db)


def test_ground_truth_coverage_unions_split_predictions() -> None:
    coverage = midi_b._ground_truth_coverage(  # pyright: ignore[reportPrivateUsage]
        (0, 0, 20, 10),
        [(0, 0, 10, 10), (10, 0, 20, 10)],
    )

    assert coverage == 1.0
