from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from lx_anonymizer.training import midi_b_phi_dataset
from lx_anonymizer.training.midi_b_phi_dataset import (
    MidiBTrainingDatasetConfig,
    MidiBTrainingDatasetError,
    generate_midi_b_phi_dataset,
)


def _pixel_action() -> str:
    return json.dumps(
        {
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
            }
        }
    )


def _write_answer_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE answer_data ("
            "SOPInstanceUID TEXT, PatientID INTEGER, Modality TEXT, AnswerData TEXT)"
        )
        connection.executemany(
            "INSERT INTO answer_data VALUES (?, ?, ?, ?)",
            [
                ("1.2.3", 100, "DX", _pixel_action()),
                ("1.2.4", 200, "CR", _pixel_action()),
                ("1.2.5", 100, "DX", json.dumps({})),
                ("1.2.6", 300, "CT", json.dumps({})),
            ],
        )


def test_generate_midi_b_dataset_keeps_patient_split_and_hard_negatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer_db = tmp_path / "answers.db"
    _write_answer_db(answer_db)
    dataset_root = tmp_path / "dicom"
    dataset_root.mkdir()
    source_paths = {
        uid: dataset_root / f"{uid}.dcm" for uid in ("1.2.3", "1.2.4", "1.2.5")
    }
    for path in source_paths.values():
        path.write_bytes(b"DICOM")

    def fake_index(_root: Path, requested: set[str]) -> dict[str, Path]:
        return {uid: source_paths[uid] for uid in requested if uid in source_paths}

    def fake_load_image(_path: Path) -> Image.Image:
        return Image.new("RGB", (100, 50))

    def fake_patient_split(
        patient: str, *, seed: int, validation_fraction: float
    ) -> str:
        del seed, validation_fraction
        return "val" if patient == "200" else "train"

    monkeypatch.setattr(
        midi_b_phi_dataset,
        "index_midi_b_dicom_instances",
        fake_index,
    )
    monkeypatch.setattr(
        midi_b_phi_dataset,
        "load_midi_b_dicom_image",
        fake_load_image,
    )
    monkeypatch.setattr(
        midi_b_phi_dataset,
        "_patient_split",
        fake_patient_split,
    )
    output_root = tmp_path / "output"

    report = generate_midi_b_phi_dataset(
        MidiBTrainingDatasetConfig(
            dataset_root=dataset_root,
            answer_db=answer_db,
            output_root=output_root,
        )
    )

    manifest = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text().splitlines()
    ]
    assert report.written_images == 3
    assert report.positive_images == 2
    assert report.negative_images == 1
    assert report.positive_images_by_split == {"train": 1, "val": 1}
    assert {
        row["split"]
        for row in manifest
        if row["patient_key_sha256"] == manifest[0]["patient_key_sha256"]
    } == {"train"}
    negative = next(row for row in manifest if not row["positive"])
    assert Path(negative["label_path"]).read_text() == ""
    positive = next(row for row in manifest if row["positive"])
    assert Path(positive["label_path"]).read_text().strip() == (
        "0 0.20000000 0.40000000 0.20000000 0.40000000"
    )
    assert (output_root / "dataset.yaml").is_file()


def test_generation_rejects_nonempty_output(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    (output_root / "keep.txt").write_text("keep")

    with pytest.raises(MidiBTrainingDatasetError, match="absent or empty"):
        MidiBTrainingDatasetConfig(
            dataset_root=tmp_path,
            answer_db=tmp_path / "missing.db",
            output_root=output_root,
        )
        midi_b_phi_dataset._prepare_output_root(output_root)  # pyright: ignore[reportPrivateUsage]


def test_patient_split_is_deterministic() -> None:
    first = midi_b_phi_dataset._patient_split(  # pyright: ignore[reportPrivateUsage]
        "patient-1", seed=4, validation_fraction=0.25
    )
    second = midi_b_phi_dataset._patient_split(  # pyright: ignore[reportPrivateUsage]
        "patient-1", seed=4, validation_fraction=0.25
    )

    assert first == second
    assert first in {"train", "val"}
