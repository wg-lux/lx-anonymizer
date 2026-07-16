from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Protocol, TypedDict, cast

import cv2
import numpy as np
import numpy.typing as npt

Box = tuple[int, int, int, int]
SplitName = str
ImageArray = npt.NDArray[np.uint8]


class SyntheticPhiGenerationError(RuntimeError):
    """Raised when a reproducible synthetic dataset cannot be generated."""


class _PydicomModule(Protocol):
    def dcmread(self, path: str | Path, **kwargs: object) -> object: ...


class _AnnotationRecord(TypedDict):
    text: str
    box: Box


@dataclass(frozen=True)
class SyntheticPhiFrameConfig:
    source_root: Path
    output_root: Path
    names_source: Path | None = None
    seed: int = 0
    frames_per_patient: int = 2
    max_patients: int | None = None
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    negative_fraction: float = 0.15
    max_dimension: int = 1024
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", self.source_root.expanduser().resolve())
        object.__setattr__(self, "output_root", self.output_root.expanduser().resolve())
        if self.names_source is not None:
            object.__setattr__(
                self, "names_source", self.names_source.expanduser().resolve()
            )
        if not self.source_root.is_dir():
            raise SyntheticPhiGenerationError(
                f"source root does not exist: {self.source_root}"
            )
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise SyntheticPhiGenerationError(
                f"output root must be absent or empty: {self.output_root}"
            )
        if self.names_source is not None and not self.names_source.is_file():
            raise SyntheticPhiGenerationError(
                f"names source does not exist: {self.names_source}"
            )
        if self.seed < 0:
            raise SyntheticPhiGenerationError("seed must be non-negative")
        if self.frames_per_patient < 1:
            raise SyntheticPhiGenerationError("frames_per_patient must be at least 1")
        if self.max_patients is not None and self.max_patients < 1:
            raise SyntheticPhiGenerationError("max_patients must be at least 1")
        if not 0.0 < self.train_fraction < 1.0:
            raise SyntheticPhiGenerationError("train_fraction must be between 0 and 1")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise SyntheticPhiGenerationError(
                "validation_fraction must be at least 0 and less than 1"
            )
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise SyntheticPhiGenerationError(
                "train_fraction + validation_fraction must be less than 1"
            )
        if not 0.0 <= self.negative_fraction <= 1.0:
            raise SyntheticPhiGenerationError(
                "negative_fraction must be between 0 and 1"
            )
        if self.max_dimension < 128:
            raise SyntheticPhiGenerationError("max_dimension must be at least 128")
        if not 1 <= self.jpeg_quality <= 100:
            raise SyntheticPhiGenerationError("jpeg_quality must be between 1 and 100")


@dataclass(frozen=True)
class SyntheticPhiGenerationReport:
    schema_version: int
    output_root: str
    seed: int
    patients: int
    frames: int
    positive_frames: int
    negative_frames: int
    annotations: int
    failed_sources: int
    frames_by_split: dict[str, int]
    patients_by_split: dict[str, int]
    failures: list[str]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class _Identity:
    name: str
    birth_date: str
    patient_id: str
    accession: str
    examination_date: str


def generate_synthetic_phi_dataset(
    config: SyntheticPhiFrameConfig,
) -> SyntheticPhiGenerationReport:
    patient_sources = _discover_patient_sources(config.source_root)
    if not patient_sources:
        raise SyntheticPhiGenerationError(
            f"no DICOM or raster source frames found under {config.source_root}"
        )

    rng = random.Random(config.seed)
    patient_keys = sorted(patient_sources)
    rng.shuffle(patient_keys)
    if config.max_patients is not None:
        patient_keys = patient_keys[: config.max_patients]
    failures: list[str] = []
    selected_sources: dict[str, list[Path]] = {}
    for patient_key in patient_keys:
        selected = _select_manageable_sources(
            patient_sources[patient_key], config.frames_per_patient
        )
        if selected:
            selected_sources[patient_key] = selected
        else:
            failures.append(f"{patient_key}: no manageable single-frame source")
    patient_keys = [key for key in patient_keys if key in selected_sources]
    if not patient_keys:
        raise SyntheticPhiGenerationError(
            "no patients contain a manageable single-frame source"
        )
    split_by_patient = _assign_patient_splits(
        patient_keys, config.train_fraction, config.validation_fraction
    )
    names = _load_name_pool(config.names_source)
    _prepare_output_tree(config.output_root)

    manifest_path = config.output_root / "manifest.jsonl"
    frame_counts: dict[str, int] = defaultdict(int)
    patient_counts: dict[str, int] = defaultdict(int)
    patients_with_frames: set[str] = set()
    positive_frames = 0
    negative_frames = 0
    annotation_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for patient_key in patient_keys:
            split = split_by_patient[patient_key]
            for frame_index, source_path in enumerate(selected_sources[patient_key]):
                try:
                    background = _load_background(source_path, config.max_dimension)
                    is_negative = rng.random() < config.negative_fraction
                    annotations: list[_AnnotationRecord] = []
                    rendered = background
                    if not is_negative:
                        identity = _make_identity(rng, names)
                        rendered, annotations = render_synthetic_phi_frame(
                            background, _identity_lines(identity, rng), rng
                        )
                except (
                    AttributeError,
                    ImportError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    failures.append(f"{source_path}: {type(exc).__name__}: {exc}")
                    continue

                stem = f"{patient_key}_{frame_index:03d}_{source_path.stem}"
                safe_stem = _safe_filename(stem)
                image_path = config.output_root / "images" / split / f"{safe_stem}.jpg"
                label_path = config.output_root / "labels" / split / f"{safe_stem}.txt"
                _write_image(image_path, rendered, config.jpeg_quality)
                _write_yolo_labels(label_path, annotations, rendered.shape)
                if is_negative:
                    negative_frames += 1
                else:
                    positive_frames += 1
                    annotation_count += len(annotations)
                frame_counts[split] += 1
                if patient_key not in patients_with_frames:
                    patients_with_frames.add(patient_key)
                    patient_counts[split] += 1
                manifest.write(
                    json.dumps(
                        {
                            "patient_key": patient_key,
                            "split": split,
                            "source_path": str(source_path),
                            "image_path": str(image_path),
                            "label_path": str(label_path),
                            "negative": is_negative,
                            "annotations": annotations,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    total_frames = sum(frame_counts.values())
    if total_frames == 0:
        raise SyntheticPhiGenerationError("all selected source frames failed to render")
    _write_dataset_yaml(config.output_root)
    report = SyntheticPhiGenerationReport(
        schema_version=1,
        output_root=str(config.output_root),
        seed=config.seed,
        patients=len(patients_with_frames),
        frames=total_frames,
        positive_frames=positive_frames,
        negative_frames=negative_frames,
        annotations=annotation_count,
        failed_sources=len(failures),
        frames_by_split=dict(sorted(frame_counts.items())),
        patients_by_split=dict(sorted(patient_counts.items())),
        failures=failures,
    )
    (config.output_root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def render_synthetic_phi_frame(
    background: ImageArray,
    lines: Sequence[str],
    rng: random.Random,
) -> tuple[ImageArray, list[_AnnotationRecord]]:
    if background.ndim != 3 or background.shape[2] != 3:
        raise ValueError("background must be a BGR uint8 image with three channels")
    if background.dtype != np.uint8:
        raise ValueError("background must use uint8 pixels")
    cleaned_lines = [_ascii_text(line) for line in lines if line.strip()]
    if not cleaned_lines:
        raise ValueError("at least one non-empty PHI line is required")

    image = background.copy()
    height, width = image.shape[:2]
    font = rng.choice(
        [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN]
    )
    scale = rng.uniform(0.45, 1.15) * max(0.7, min(width, height) / 768.0)
    thickness = rng.choice([1, 1, 2])
    margin = max(8, int(min(width, height) * 0.025))
    gap = max(4, int(scale * 7))

    sizes: list[tuple[int, int, int, float]] = []
    for line in cleaned_lines:
        current_scale = scale
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, current_scale, thickness
        )
        while text_width > width - 2 * margin and current_scale > 0.25:
            current_scale *= 0.9
            (text_width, text_height), baseline = cv2.getTextSize(
                line, font, current_scale, thickness
            )
        sizes.append((text_width, text_height, baseline, current_scale))
    block_width = max(size[0] for size in sizes)
    block_height = sum(size[1] + size[2] for size in sizes) + gap * (len(sizes) - 1)
    available_height = height - 2 * margin
    if block_height > available_height:
        vertical_scale = max(0.1, available_height / float(block_height) * 0.92)
        resized: list[tuple[int, int, int, float]] = []
        for line, (_text_width, _text_height, _baseline, line_scale) in zip(
            cleaned_lines, sizes
        ):
            adjusted_scale = line_scale * vertical_scale
            (text_width, text_height), baseline = cv2.getTextSize(
                line, font, adjusted_scale, thickness
            )
            resized.append((text_width, text_height, baseline, adjusted_scale))
        sizes = resized
        gap = max(1, int(min(size[3] for size in sizes) * 7))
        block_width = max(size[0] for size in sizes)
        block_height = sum(size[1] + size[2] for size in sizes) + gap * (len(sizes) - 1)
    if block_width > width - 2 * margin or block_height > height - 2 * margin:
        raise ValueError("source frame is too small for the generated PHI overlay")

    anchor = rng.choice(("top-left", "top-right", "bottom-left", "bottom-right"))
    left = margin if anchor.endswith("left") else width - margin - block_width
    top = margin if anchor.startswith("top") else height - margin - block_height
    panel = rng.random() < 0.65
    light_text = rng.random() < 0.7
    text_color = (245, 245, 245) if light_text else (10, 10, 10)
    outline_color = (0, 0, 0) if light_text else (255, 255, 255)
    if panel:
        panel_color = (8, 8, 8) if light_text else (238, 238, 238)
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (max(0, left - margin // 2), max(0, top - margin // 2)),
            (
                min(width - 1, left + block_width + margin // 2),
                min(height - 1, top + block_height + margin // 2),
            ),
            panel_color,
            -1,
        )
        cv2.addWeighted(overlay, rng.uniform(0.55, 0.9), image, 1.0, 0.0, image)

    annotations: list[_AnnotationRecord] = []
    cursor_y = top
    for line, (text_width, text_height, baseline, line_scale) in zip(
        cleaned_lines, sizes
    ):
        baseline_y = cursor_y + text_height
        padding = thickness + 2
        cv2.putText(
            image,
            line,
            (left, baseline_y),
            font,
            line_scale,
            outline_color,
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (left, baseline_y),
            font,
            line_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        box = (
            max(0, left - padding),
            max(0, baseline_y - text_height - padding),
            min(width, left + text_width + padding),
            min(height, baseline_y + baseline + padding),
        )
        annotations.append({"text": line, "box": box})
        cursor_y += text_height + baseline + gap
    return image, annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer-generate-phi-data",
        description="Create patient-split YOLO PHI frames using OpenCV overlays.",
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--names-source", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames-per-patient", type=int, default=2)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--negative-fraction", type=float, default=0.15)
    parser.add_argument("--max-dimension", type=int, default=1024)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_synthetic_phi_dataset(
        SyntheticPhiFrameConfig(
            source_root=cast(Path, args.source_root),
            output_root=cast(Path, args.output_root),
            names_source=cast(Path | None, args.names_source),
            seed=cast(int, args.seed),
            frames_per_patient=cast(int, args.frames_per_patient),
            max_patients=cast(int | None, args.max_patients),
            train_fraction=cast(float, args.train_fraction),
            validation_fraction=cast(float, args.validation_fraction),
            negative_fraction=cast(float, args.negative_fraction),
            max_dimension=cast(int, args.max_dimension),
            jpeg_quality=cast(int, args.jpeg_quality),
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=True))
    return 0


def _discover_patient_sources(source_root: Path) -> dict[str, list[Path]]:
    supported = {".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in supported:
            continue
        relative = path.relative_to(source_root)
        patient_key = relative.parts[0] if len(relative.parts) > 1 else path.stem
        grouped[patient_key].append(path)
    return dict(grouped)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 2**63 - 1


def _select_manageable_sources(paths: Sequence[Path], limit: int) -> list[Path]:
    ordered = sorted(paths, key=lambda path: (_file_size(path), path))
    selected: list[Path] = []
    pydicom: _PydicomModule | None = None
    for path in ordered:
        if path.suffix.lower() == ".dcm":
            if pydicom is None:
                pydicom = _load_pydicom()
            try:
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=["Rows", "Columns", "NumberOfFrames"],
                )
                rows = int(str(getattr(dataset, "Rows", "0")))
                columns = int(str(getattr(dataset, "Columns", "0")))
                frames = int(str(getattr(dataset, "NumberOfFrames", "1")))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
            if rows < 1 or columns < 1 or frames != 1 or rows * columns > 4_194_304:
                continue
        selected.append(path)
        if len(selected) == limit:
            break
    return selected


def _assign_patient_splits(
    patient_keys: Sequence[str], train_fraction: float, validation_fraction: float
) -> dict[str, SplitName]:
    count = len(patient_keys)
    train_count = max(1, int(count * train_fraction))
    validation_count = int(count * validation_fraction)
    if count >= 3 and validation_fraction > 0.0:
        validation_count = max(1, validation_count)
    if train_count + validation_count >= count and count > 1:
        train_count = max(1, count - validation_count - 1)
    return {
        patient: (
            "train"
            if index < train_count
            else "val"
            if index < train_count + validation_count
            else "test"
        )
        for index, patient in enumerate(patient_keys)
    }


def _load_name_pool(names_source: Path | None) -> list[str]:
    if names_source is None:
        return [
            "Anna Mueller",
            "Thomas Becker",
            "Maria Schmidt",
            "David Wagner",
            "Sofia Fischer",
            "Jonas Weber",
        ]
    raw = names_source.read_text(encoding="utf-8")
    try:
        parsed = cast(object, json.loads(raw))
        records: list[object] = (
            cast(list[object], parsed) if isinstance(parsed, list) else [parsed]
        )
    except json.JSONDecodeError:
        records = [
            cast(object, json.loads(line)) for line in raw.splitlines() if line.strip()
        ]
    names: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        item = cast(Mapping[str, object], record)
        first = item.get("first_name")
        last = item.get("last_name")
        if isinstance(first, str) and isinstance(last, str) and first and last:
            names.append(_ascii_text(f"{first} {last}"))
    if not names:
        raise SyntheticPhiGenerationError("names source contains no complete names")
    return names


def _make_identity(rng: random.Random, names: Sequence[str]) -> _Identity:
    base_date = date(1940, 1, 1) + timedelta(days=rng.randrange(25_000))
    examination = date(2010, 1, 1) + timedelta(days=rng.randrange(5_500))
    return _Identity(
        name=rng.choice(list(names)),
        birth_date=base_date.strftime("%d.%m.%Y"),
        patient_id=f"PID-{rng.randrange(100000, 999999)}",
        accession=f"ACC-{rng.randrange(10000000, 99999999)}",
        examination_date=examination.strftime("%d.%m.%Y"),
    )


def _identity_lines(identity: _Identity, rng: random.Random) -> list[str]:
    layouts = [
        [
            f"PATIENT: {identity.name}",
            f"DOB: {identity.birth_date}",
            identity.patient_id,
        ],
        [
            identity.name.upper(),
            f"ID {identity.patient_id}",
            f"DATE {identity.examination_date}",
        ],
        [f"NAME {identity.name}", f"BORN {identity.birth_date}", identity.accession],
        [
            f"{identity.patient_id}  {identity.name}",
            f"EXAM {identity.examination_date}",
        ],
    ]
    return rng.choice(layouts)


def _load_background(path: Path, max_dimension: int) -> ImageArray:
    if path.suffix.lower() == ".dcm":
        pydicom = _load_pydicom()
        dataset = pydicom.dcmread(path)
        pixels = getattr(dataset, "pixel_array", None)
        if pixels is None:
            raise SyntheticPhiGenerationError("DICOM has no decodable pixel data")
        image = _pixels_to_bgr(np.asarray(pixels), dataset)
    else:
        decoded = cast(object, cv2.imread(str(path), cv2.IMREAD_COLOR))
        if not isinstance(decoded, np.ndarray):
            raise SyntheticPhiGenerationError("OpenCV could not decode source image")
        image = cast(ImageArray, decoded)
    height, width = image.shape[:2]
    maximum_scale = max_dimension / float(max(height, width))
    minimum_scale = 256.0 / float(min(height, width))
    scale = min(maximum_scale, max(1.0, minimum_scale))
    if scale != 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _pixels_to_bgr(pixels: np.ndarray, dataset: object) -> ImageArray:
    if pixels.ndim == 4:
        pixels = pixels[0]
    elif pixels.ndim == 3 and pixels.shape[-1] not in (3, 4):
        pixels = pixels[0]
    if pixels.ndim == 2:
        grayscale = _normalize_pixels(pixels)
        if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
            grayscale = 255 - grayscale
        return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        color = _normalize_pixels(pixels[..., :3])
        return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    raise SyntheticPhiGenerationError(f"unsupported DICOM pixel shape: {pixels.shape}")


def _normalize_pixels(pixels: np.ndarray) -> ImageArray:
    values = pixels.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise SyntheticPhiGenerationError("source pixels contain no finite values")
    lower, upper = np.percentile(finite, (0.5, 99.5))
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    normalized = np.clip((values - lower) * (255.0 / (upper - lower)), 0, 255)
    return normalized.astype(np.uint8)


def _prepare_output_tree(output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_image(path: Path, image: ImageArray, quality: int) -> None:
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise SyntheticPhiGenerationError(f"failed to write generated image: {path}")


def _write_yolo_labels(
    path: Path, annotations: Sequence[_AnnotationRecord], shape: tuple[int, ...]
) -> None:
    height, width = shape[:2]
    rows: list[str] = []
    for annotation in annotations:
        x1, y1, x2, y2 = annotation["box"]
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        rows.append(f"0 {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_dataset_yaml(output_root: Path) -> None:
    (output_root / "dataset.yaml").write_text(
        f"path: {output_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: phi\n",
        encoding="utf-8",
    )


def _ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii").strip()


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return safe[:180].strip("_") or "frame"


def _load_pydicom() -> _PydicomModule:
    try:
        import pydicom  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SyntheticPhiGenerationError(
            "DICOM frame generation requires pydicom; install the evaluation extra"
        ) from exc
    return cast(_PydicomModule, pydicom)


if __name__ == "__main__":
    raise SystemExit(main())
