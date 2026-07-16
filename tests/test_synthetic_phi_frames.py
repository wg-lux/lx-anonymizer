from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import pytest

from lx_anonymizer.training.synthetic_phi_frames import (
    SyntheticPhiFrameConfig,
    SyntheticPhiGenerationError,
    generate_synthetic_phi_dataset,
    render_synthetic_phi_frame,
)


class _ManifestRecord(TypedDict):
    patient_key: str
    split: str
    label_path: str
    negative: bool


def _write_sources(root: Path, patient_count: int = 5) -> None:
    for patient_index in range(patient_count):
        patient_dir = root / f"patient-{patient_index}"
        patient_dir.mkdir(parents=True)
        image = np.full((240, 320, 3), 40 + patient_index * 10, dtype=np.uint8)
        assert cv2.imwrite(str(patient_dir / "frame.png"), image)


def _read_manifest(path: Path) -> list[_ManifestRecord]:
    return [
        cast(_ManifestRecord, cast(object, json.loads(line)))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_render_synthetic_phi_frame_returns_exact_bounded_annotations() -> None:
    background = np.full((360, 640, 3), 80, dtype=np.uint8)

    rendered, annotations = render_synthetic_phi_frame(
        background,
        ["PATIENT: ANNA MUELLER", "DOB: 01.02.1980", "PID-123456"],
        random.Random(7),
    )

    assert len(annotations) == 3
    assert not np.array_equal(rendered, background)
    for annotation in annotations:
        x1, y1, x2, y2 = annotation["box"]
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 360
        assert np.any(rendered[y1:y2, x1:x2] != background[y1:y2, x1:x2])


def test_generate_dataset_writes_yolo_tree_without_patient_leakage(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    _write_sources(source_root)
    names = tmp_path / "names.jsonl"
    names.write_text('{"first_name":"Anna","last_name":"Mueller"}\n', encoding="utf-8")
    output_root = tmp_path / "dataset"

    report = generate_synthetic_phi_dataset(
        SyntheticPhiFrameConfig(
            source_root=source_root,
            output_root=output_root,
            names_source=names,
            seed=11,
            frames_per_patient=1,
            negative_fraction=0.0,
            max_dimension=320,
        )
    )

    records = _read_manifest(output_root / "manifest.jsonl")
    splits_by_patient: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_patient[record["patient_key"]].add(record["split"])
        labels = Path(record["label_path"]).read_text(encoding="utf-8").splitlines()
        assert labels
        assert all(line.startswith("0 ") and len(line.split()) == 5 for line in labels)
    assert all(len(splits) == 1 for splits in splits_by_patient.values())
    assert set(report.frames_by_split) == {"test", "train", "val"}
    assert report.frames == 5
    assert report.positive_frames == 5
    assert report.annotations >= 10
    assert (output_root / "dataset.yaml").is_file()
    assert (output_root / "summary.json").is_file()


def test_generate_dataset_keeps_empty_labels_for_negative_frames(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    _write_sources(source_root, patient_count=3)
    output_root = tmp_path / "dataset"

    report = generate_synthetic_phi_dataset(
        SyntheticPhiFrameConfig(
            source_root=source_root,
            output_root=output_root,
            seed=3,
            frames_per_patient=1,
            negative_fraction=1.0,
            max_dimension=320,
        )
    )

    records = _read_manifest(output_root / "manifest.jsonl")
    assert report.negative_frames == 3
    assert report.annotations == 0
    assert all(record["negative"] for record in records)
    assert all(
        Path(record["label_path"]).read_text(encoding="utf-8") == ""
        for record in records
    )


def test_generation_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    _write_sources(source_root, patient_count=1)
    output_root = tmp_path / "dataset"
    output_root.mkdir()
    (output_root / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(SyntheticPhiGenerationError, match="absent or empty"):
        SyntheticPhiFrameConfig(source_root=source_root, output_root=output_root)
