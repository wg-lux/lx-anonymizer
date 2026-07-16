from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from lx_anonymizer.training import radphi_dataset
from lx_anonymizer.training.radphi_dataset import (
    RadPhiDatasetConfig,
    RadPhiDatasetError,
    generate_radphi_yolo_dataset,
)


def _write_example(root: Path, stem: str, *, phi: int) -> None:
    images = root / "data" / "images"
    imprints = root / "data" / "imprints"
    images.mkdir(parents=True, exist_ok=True)
    imprints.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(images / f"{stem}.png")
    annotation = (
        {"line": {"phi": phi, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.1}}
        if phi in (0, 1)
        else {"line": {"phi": phi}}
    )
    (imprints / f"{stem}.json").write_text(json.dumps(annotation))


def test_generate_radphi_dataset_filters_non_phi_and_keeps_groups_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write_example(source, "ts_s0001_1_1", phi=1)
    _write_example(source, "ts_s0001_2_2", phi=0)
    _write_example(source, "nih_xray_00000002_001_0", phi=1)

    def fixed_split(group: str, *, seed: int, validation_fraction: float) -> str:
        del seed, validation_fraction
        return "val" if group.endswith("00000002") else "train"

    monkeypatch.setattr(radphi_dataset, "_group_split", fixed_split)
    output = tmp_path / "output"
    report = generate_radphi_yolo_dataset(
        RadPhiDatasetConfig(source_root=source, output_root=output)
    )

    assert report.images == 3
    assert report.positive_images == 2
    assert report.negative_images == 1
    assert report.groups_by_split == {"train": 1, "val": 1}
    assert (output / "labels/train/ts_s0001_2_2.txt").read_text() == ""
    assert (output / "labels/val/nih_xray_00000002_001_0.txt").read_text() == (
        "0 0.50000000 0.50000000 0.25000000 0.10000000\n"
    )


def test_radphi_dataset_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_example(source, "bs80k_1_0", phi=1)
    imprint = source / "data/imprints/bs80k_1_0.json"
    imprint.write_text(
        json.dumps({"line": {"phi": 1, "cx": 0.95, "cy": 0.5, "w": 0.2, "h": 0.1}})
    )

    with pytest.raises(RadPhiDatasetError, match="horizontal bounds"):
        generate_radphi_yolo_dataset(
            RadPhiDatasetConfig(source_root=source, output_root=tmp_path / "output")
        )


def test_radphi_group_split_is_deterministic() -> None:
    first = radphi_dataset._group_split(  # pyright: ignore[reportPrivateUsage]
        "ts_s0001", seed=3, validation_fraction=0.2
    )
    second = radphi_dataset._group_split(  # pyright: ignore[reportPrivateUsage]
        "ts_s0001", seed=3, validation_fraction=0.2
    )
    assert first == second
