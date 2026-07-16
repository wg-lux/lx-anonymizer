from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast


class RadPhiDatasetError(RuntimeError):
    """Raised when RadPHI cannot be converted into a valid YOLO dataset."""


@dataclass(frozen=True)
class RadPhiDatasetConfig:
    source_root: Path
    output_root: Path
    validation_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", self.source_root.expanduser().resolve())
        object.__setattr__(self, "output_root", self.output_root.expanduser().resolve())
        if not self.source_root.is_dir():
            raise RadPhiDatasetError(
                f"RadPHI source root not found: {self.source_root}"
            )
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise RadPhiDatasetError(
                f"output root must be absent or empty: {self.output_root}"
            )
        if not 0.0 < self.validation_fraction < 1.0:
            raise RadPhiDatasetError("validation_fraction must be between 0 and 1")
        if self.seed < 0:
            raise RadPhiDatasetError("seed must be non-negative")


@dataclass(frozen=True)
class RadPhiDatasetReport:
    schema_version: int
    source_root: str
    output_root: str
    seed: int
    validation_fraction: float
    images: int
    positive_images: int
    negative_images: int
    phi_annotations: int
    images_by_split: dict[str, int]
    positive_images_by_split: dict[str, int]
    groups_by_split: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


NormalizedBox = tuple[float, float, float, float]


def generate_radphi_yolo_dataset(
    config: RadPhiDatasetConfig,
) -> RadPhiDatasetReport:
    images_root = config.source_root / "data" / "images"
    imprints_root = config.source_root / "data" / "imprints"
    if not images_root.is_dir() or not imprints_root.is_dir():
        raise RadPhiDatasetError(
            "source root must contain data/images and data/imprints"
        )

    image_paths = sorted(images_root.glob("*.png"))
    if not image_paths:
        raise RadPhiDatasetError("RadPHI source contains no PNG images")
    _prepare_output_root(config.output_root)

    image_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    groups: dict[str, set[str]] = {"train": set(), "val": set()}
    annotations = 0
    for image_path in image_paths:
        imprint_path = imprints_root / f"{image_path.stem}.json"
        if not imprint_path.is_file():
            raise RadPhiDatasetError(f"missing imprint JSON: {imprint_path}")
        group = _source_group(image_path.stem)
        split = _group_split(
            group,
            seed=config.seed,
            validation_fraction=config.validation_fraction,
        )
        boxes = _load_phi_boxes(imprint_path)
        destination = config.output_root / "images" / split / image_path.name
        shutil.copy2(image_path, destination)
        _write_yolo_labels(
            config.output_root / "labels" / split / f"{image_path.stem}.txt",
            boxes,
        )
        image_counts[split] += 1
        groups[split].add(group)
        annotations += len(boxes)
        if boxes:
            positive_counts[split] += 1

    if not positive_counts["train"] or not positive_counts["val"]:
        raise RadPhiDatasetError(
            "group split must contain positive examples in train and val"
        )
    if groups["train"] & groups["val"]:
        raise RadPhiDatasetError("source groups overlap across train and val")

    _write_dataset_yaml(config.output_root)
    positive_images = sum(positive_counts.values())
    report = RadPhiDatasetReport(
        schema_version=1,
        source_root=str(config.source_root),
        output_root=str(config.output_root),
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        images=len(image_paths),
        positive_images=positive_images,
        negative_images=len(image_paths) - positive_images,
        phi_annotations=annotations,
        images_by_split=dict(sorted(image_counts.items())),
        positive_images_by_split=dict(sorted(positive_counts.items())),
        groups_by_split={key: len(value) for key, value in sorted(groups.items())},
    )
    (config.output_root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def _source_group(stem: str) -> str:
    patterns = (
        r"^(ts_s\d+)",
        r"^(nih_xray_\d+)",
        r"^(msd_brain_BRATS_\d+)",
        r"^(bs80k_\d+)",
    )
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match is not None:
            return match.group(1)
    raise RadPhiDatasetError(f"unsupported RadPHI image stem: {stem}")


def _group_split(group: str, *, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if fraction < validation_fraction else "train"


def _load_phi_boxes(path: Path) -> tuple[NormalizedBox, ...]:
    try:
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise RadPhiDatasetError(f"invalid imprint JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RadPhiDatasetError(f"imprint JSON must be an object: {path}")

    boxes: list[NormalizedBox] = []
    for raw_annotation in cast(Mapping[object, object], payload).values():
        if not isinstance(raw_annotation, Mapping):
            raise RadPhiDatasetError(f"imprint annotation must be an object: {path}")
        annotation = cast(Mapping[object, object], raw_annotation)
        phi = annotation.get("phi")
        if phi not in (0, 1):
            raise RadPhiDatasetError(f"imprint phi must be 0 or 1: {path}")
        if phi == 0:
            continue
        coordinates = tuple(
            _normalized_float(annotation.get(key), key=key, path=path)
            for key in ("cx", "cy", "w", "h")
        )
        cx, cy, width, height = coordinates
        if width <= 0.0 or height <= 0.0:
            raise RadPhiDatasetError(f"imprint box has non-positive size: {path}")
        if cx - width / 2.0 < 0.0 or cx + width / 2.0 > 1.0:
            raise RadPhiDatasetError(f"imprint box exceeds horizontal bounds: {path}")
        if cy - height / 2.0 < 0.0 or cy + height / 2.0 > 1.0:
            raise RadPhiDatasetError(f"imprint box exceeds vertical bounds: {path}")
        boxes.append((cx, cy, width, height))
    return tuple(boxes)


def _normalized_float(value: object, *, key: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RadPhiDatasetError(f"imprint {key} must be numeric: {path}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise RadPhiDatasetError(f"imprint {key} must be normalized: {path}")
    return result


def _prepare_output_root(output_root: Path) -> None:
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_yolo_labels(path: Path, boxes: tuple[NormalizedBox, ...]) -> None:
    lines = [
        f"0 {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
        for cx, cy, width, height in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_dataset_yaml(output_root: Path) -> None:
    (output_root / "dataset.yaml").write_text(
        f"path: {output_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: phi\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer-generate-radphi-data",
        description="Convert RadPHI imprint JSON into a group-disjoint YOLO dataset.",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_radphi_yolo_dataset(
        RadPhiDatasetConfig(
            source_root=cast(Path, args.source_root),
            output_root=cast(Path, args.output_root),
            validation_fraction=cast(float, args.validation_fraction),
            seed=cast(int, args.seed),
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
