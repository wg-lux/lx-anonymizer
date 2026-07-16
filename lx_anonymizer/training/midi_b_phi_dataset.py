from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from lx_anonymizer.evaluation.midi_b import (
    Box,
    index_midi_b_dicom_instances,
    load_midi_b_dicom_image,
    load_midi_b_pixel_annotations,
)


class MidiBTrainingDatasetError(RuntimeError):
    """Raised when a leakage-safe MIDI-B detector dataset cannot be built."""


@dataclass(frozen=True)
class MidiBTrainingDatasetConfig:
    dataset_root: Path
    answer_db: Path
    output_root: Path
    validation_fraction: float = 0.25
    seed: int = 0
    modalities: tuple[str, ...] = ("CR", "DX", "MG")
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not self.modalities:
            raise ValueError("modalities must not be empty")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class MidiBTrainingDatasetReport:
    schema_version: int
    output_root: str
    seed: int
    validation_fraction: float
    answer_rows: int
    indexed_instances: int
    written_images: int
    positive_images: int
    negative_images: int
    annotations: int
    missing_instances: int
    images_by_split: dict[str, int]
    positive_images_by_split: dict[str, int]
    patients_by_split: dict[str, int]
    modalities: dict[str, int]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class _AnswerRecord:
    instance_uid: str
    patient_key: str
    modality: str


def generate_midi_b_phi_dataset(
    config: MidiBTrainingDatasetConfig,
) -> MidiBTrainingDatasetReport:
    dataset_root = config.dataset_root.expanduser().resolve()
    answer_db = config.answer_db.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise MidiBTrainingDatasetError(f"dataset root not found: {dataset_root}")
    if not answer_db.is_file():
        raise MidiBTrainingDatasetError(f"answer database not found: {answer_db}")
    _prepare_output_root(output_root)

    records = _load_answer_records(answer_db, config.modalities)
    annotations_by_uid: dict[str, list[Box]] = defaultdict(list)
    for annotation in load_midi_b_pixel_annotations(answer_db):
        annotations_by_uid[annotation.instance_uid].append(annotation.box)
    positive_uids = set(annotations_by_uid)
    records = [
        record
        for record in records
        if record.instance_uid in positive_uids or record.modality in config.modalities
    ]
    paths_by_uid = index_midi_b_dicom_instances(
        dataset_root, {record.instance_uid for record in records}
    )

    split_by_patient = {
        record.patient_key: _patient_split(
            record.patient_key,
            seed=config.seed,
            validation_fraction=config.validation_fraction,
        )
        for record in records
    }
    image_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    patient_sets: dict[str, set[str]] = defaultdict(set)
    modality_counts: Counter[str] = Counter()
    failures: list[str] = []
    written_images = 0
    positive_images = 0
    negative_images = 0
    annotation_count = 0

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in sorted(records, key=lambda item: item.instance_uid):
            source_path = paths_by_uid.get(record.instance_uid)
            if source_path is None:
                failures.append(f"missing DICOM instance: {record.instance_uid}")
                continue
            try:
                image = load_midi_b_dicom_image(source_path).convert("RGB")
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                failures.append(f"{record.instance_uid}: {type(exc).__name__}: {exc}")
                continue
            split = split_by_patient[record.patient_key]
            stem = record.instance_uid.replace(".", "_")
            image_path = output_root / "images" / split / f"{stem}.jpg"
            label_path = output_root / "labels" / split / f"{stem}.txt"
            boxes = tuple(annotations_by_uid.get(record.instance_uid, ()))
            _write_image(image_path, image, config.jpeg_quality)
            _write_yolo_labels(label_path, boxes, image.size)

            is_positive = bool(boxes)
            written_images += 1
            annotation_count += len(boxes)
            image_counts[split] += 1
            modality_counts[record.modality] += 1
            patient_sets[split].add(record.patient_key)
            if is_positive:
                positive_images += 1
                positive_counts[split] += 1
            else:
                negative_images += 1
            manifest.write(
                json.dumps(
                    {
                        "instance_uid_sha256": hashlib.sha256(
                            record.instance_uid.encode("utf-8")
                        ).hexdigest(),
                        "patient_key_sha256": hashlib.sha256(
                            record.patient_key.encode("utf-8")
                        ).hexdigest(),
                        "modality": record.modality,
                        "split": split,
                        "source_path": str(source_path),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "positive": is_positive,
                        "annotation_count": len(boxes),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    if written_images == 0:
        raise MidiBTrainingDatasetError("no MIDI-B images could be written")
    if not positive_counts["train"] or not positive_counts["val"]:
        raise MidiBTrainingDatasetError(
            "patient split must contain positive examples in train and val"
        )
    _write_dataset_yaml(output_root)
    report = MidiBTrainingDatasetReport(
        schema_version=1,
        output_root=str(output_root),
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        answer_rows=len(records),
        indexed_instances=len(paths_by_uid),
        written_images=written_images,
        positive_images=positive_images,
        negative_images=negative_images,
        annotations=annotation_count,
        missing_instances=len(records) - len(paths_by_uid),
        images_by_split=dict(sorted(image_counts.items())),
        positive_images_by_split=dict(sorted(positive_counts.items())),
        patients_by_split={
            split: len(patients) for split, patients in sorted(patient_sets.items())
        },
        modalities=dict(sorted(modality_counts.items())),
        failures=tuple(failures),
    )
    (output_root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_answer_records(
    answer_db: Path, modalities: tuple[str, ...]
) -> list[_AnswerRecord]:
    placeholders = ",".join("?" for _ in modalities)
    query = (
        "SELECT SOPInstanceUID, PatientID, Modality FROM answer_data "
        f"WHERE Modality IN ({placeholders})"
    )
    with sqlite3.connect(answer_db) as connection:
        rows = connection.execute(query, modalities).fetchall()
    records: list[_AnswerRecord] = []
    for raw_uid, raw_patient, raw_modality in rows:
        if not isinstance(raw_uid, str) or not isinstance(raw_modality, str):
            raise MidiBTrainingDatasetError("answer database contains invalid fields")
        uid = raw_uid.strip().removeprefix("<").removesuffix(">")
        patient_key = str(raw_patient).strip()
        if not uid or not patient_key:
            raise MidiBTrainingDatasetError(
                "answer database contains blank identifiers"
            )
        records.append(
            _AnswerRecord(
                instance_uid=uid,
                patient_key=patient_key,
                modality=raw_modality.strip(),
            )
        )
    return records


def _patient_split(patient_key: str, *, seed: int, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{patient_key}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if fraction < validation_fraction else "train"


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise MidiBTrainingDatasetError(
            f"output directory must be absent or empty: {output_root}"
        )
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_image(path: Path, image: Image.Image, jpeg_quality: int) -> None:
    image.save(path, format="JPEG", quality=jpeg_quality, optimize=True)


def _write_yolo_labels(
    path: Path, boxes: tuple[Box, ...], size: tuple[int, int]
) -> None:
    width, height = size
    if width < 1 or height < 1:
        raise MidiBTrainingDatasetError(f"invalid image dimensions: {size}")
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes:
        clipped_x1 = min(max(x1, 0), width)
        clipped_y1 = min(max(y1, 0), height)
        clipped_x2 = min(max(x2, 0), width)
        clipped_y2 = min(max(y2, 0), height)
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            raise MidiBTrainingDatasetError(
                f"box outside image bounds: {(x1, y1, x2, y2)}"
            )
        box_width = clipped_x2 - clipped_x1
        box_height = clipped_y2 - clipped_y1
        center_x = clipped_x1 + box_width / 2.0
        center_y = clipped_y1 + box_height / 2.0
        lines.append(
            "0 "
            f"{center_x / width:.8f} {center_y / height:.8f} "
            f"{box_width / width:.8f} {box_height / height:.8f}"
        )
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
        prog="lx-anonymizer-generate-midi-b-phi-data",
        description="Convert MIDI-B pixel actions into a patient-split YOLO dataset.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--answer-db", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_midi_b_phi_dataset(
        MidiBTrainingDatasetConfig(
            dataset_root=cast(Path, args.dataset_root),
            answer_db=cast(Path, args.answer_db),
            output_root=cast(Path, args.output_root),
            validation_fraction=cast(float, args.validation_fraction),
            seed=cast(int, args.seed),
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
