from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from lx_anonymizer.training.synthetic_endoscopy_stickers import (
    SyntheticEndoscopyStickerConfig,
    generate_synthetic_endoscopy_sticker_dataset,
    sanitize_endoscopy_background,
)


def _write_source(path: Path, value: int) -> None:
    image = np.full((180, 320, 3), value, dtype=np.uint8)
    image[:, :96] = 255
    assert cv2.imwrite(str(path), image)


def test_sanitize_endoscopy_background_removes_original_panel() -> None:
    source = np.full((180, 320, 3), 127, dtype=np.uint8)
    source[:, :100] = 255

    sanitized = sanitize_endoscopy_background(source, 640, 0.3)

    assert sanitized.shape == (360, 640, 3)
    assert np.count_nonzero(sanitized[:, :192]) == 0
    assert np.all(sanitized[:, 200:] == 127)


def test_generate_endoscopy_stickers_writes_exact_train_only_yolo_data(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "a.jpg"
    source_b = tmp_path / "b.jpg"
    _write_source(source_a, 70)
    _write_source(source_b, 90)
    output_root = tmp_path / "stickers"

    report = generate_synthetic_endoscopy_sticker_dataset(
        SyntheticEndoscopyStickerConfig(
            source_images=(source_a, source_b),
            output_root=output_root,
            seed=5,
            frames_per_source=3,
            negative_fraction=0.0,
            output_width=640,
        )
    )

    records = [
        cast(dict[str, object], cast(object, json.loads(line)))
        for line in (output_root / "manifest.jsonl").read_text().splitlines()
    ]
    assert report.frames == 6
    assert report.positive_frames == 6
    assert report.annotations == 6
    assert len(list((output_root / "images" / "train").glob("*.jpg"))) == 6
    assert len(list((output_root / "labels" / "train").glob("*.txt"))) == 6
    assert all(record["split"] == "train" for record in records)
    for record in records:
        label_path = Path(cast(str, record["label_path"]))
        fields = label_path.read_text().strip().split()
        assert fields[0] == "0"
        assert len(fields) == 5
        assert all(0.0 <= float(value) <= 1.0 for value in fields[1:])


def test_generate_endoscopy_stickers_keeps_empty_negative_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    _write_source(source, 80)
    output_root = tmp_path / "stickers"

    report = generate_synthetic_endoscopy_sticker_dataset(
        SyntheticEndoscopyStickerConfig(
            source_images=(source,),
            output_root=output_root,
            frames_per_source=2,
            negative_fraction=1.0,
            output_width=640,
        )
    )

    assert report.negative_frames == 2
    assert all(
        path.read_text() == ""
        for path in (output_root / "labels" / "train").glob("*.txt")
    )


def test_lens_placement_overlaps_camera_field_and_combines_train_roots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    _write_source(source, 80)
    dicom_root = tmp_path / "dicom"
    prior_stickers = tmp_path / "prior_stickers"
    (dicom_root / "images" / "train").mkdir(parents=True)
    (prior_stickers / "images" / "train").mkdir(parents=True)
    output_root = tmp_path / "lens_stickers"

    generate_synthetic_endoscopy_sticker_dataset(
        SyntheticEndoscopyStickerConfig(
            source_images=(source,),
            output_root=output_root,
            combined_dataset_root=dicom_root,
            additional_train_roots=(prior_stickers,),
            sticker_placement="lens",
            frames_per_source=2,
            negative_fraction=0.0,
            output_width=640,
        )
    )

    records = [
        cast(dict[str, object], cast(object, json.loads(line)))
        for line in (output_root / "manifest.jsonl").read_text().splitlines()
    ]
    for record in records:
        annotations = cast(list[dict[str, object]], record["annotations"])
        box = cast(list[int], annotations[0]["box"])
        assert box[0] > 640 * 0.3
        assert record["sticker_placement"] == "lens"
    dataset_yaml = (output_root / "dataset.yaml").read_text()
    assert "dicom/images/train" in dataset_yaml
    assert "prior_stickers/images/train" in dataset_yaml
    assert "lens_stickers/images/train" in dataset_yaml
