from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

import cv2
import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]
Box = tuple[int, int, int, int]
StickerPlacement = Literal["panel", "lens"]


class SyntheticEndoscopyStickerError(RuntimeError):
    """Raised when a sticker dataset cannot be generated safely."""


class StickerAnnotation(TypedDict):
    text: str
    box: Box


@dataclass(frozen=True)
class SyntheticEndoscopyStickerConfig:
    source_images: tuple[Path, ...]
    output_root: Path
    combined_dataset_root: Path | None = None
    additional_train_roots: tuple[Path, ...] = ()
    sticker_placement: StickerPlacement = "panel"
    seed: int = 0
    frames_per_source: int = 60
    negative_fraction: float = 0.1
    output_width: int = 960
    masked_left_fraction: float = 0.30
    jpeg_quality: int = 92

    def __post_init__(self) -> None:
        sources = tuple(path.expanduser().resolve() for path in self.source_images)
        object.__setattr__(self, "source_images", sources)
        object.__setattr__(self, "output_root", self.output_root.expanduser().resolve())
        additional_roots = tuple(
            path.expanduser().resolve() for path in self.additional_train_roots
        )
        object.__setattr__(self, "additional_train_roots", additional_roots)
        if self.combined_dataset_root is not None:
            object.__setattr__(
                self,
                "combined_dataset_root",
                self.combined_dataset_root.expanduser().resolve(),
            )
        if any(not path.is_dir() for path in additional_roots):
            raise SyntheticEndoscopyStickerError(
                "every additional training root must be an existing directory"
            )
        if self.combined_dataset_root is None and additional_roots:
            raise SyntheticEndoscopyStickerError(
                "additional training roots require a combined dataset root"
            )
        if self.sticker_placement not in ("panel", "lens"):
            raise SyntheticEndoscopyStickerError(
                "sticker_placement must be either 'panel' or 'lens'"
            )
        if not sources or any(not path.is_file() for path in sources):
            raise SyntheticEndoscopyStickerError(
                "every configured endoscopy source must be an existing file"
            )
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise SyntheticEndoscopyStickerError(
                f"output root must be absent or empty: {self.output_root}"
            )
        if self.seed < 0:
            raise SyntheticEndoscopyStickerError("seed must be non-negative")
        if self.frames_per_source < 1:
            raise SyntheticEndoscopyStickerError("frames_per_source must be >= 1")
        if not 0.0 <= self.negative_fraction <= 1.0:
            raise SyntheticEndoscopyStickerError(
                "negative_fraction must be between 0 and 1"
            )
        if self.output_width < 320:
            raise SyntheticEndoscopyStickerError("output_width must be >= 320")
        if not 0.2 <= self.masked_left_fraction <= 0.5:
            raise SyntheticEndoscopyStickerError(
                "masked_left_fraction must be between 0.2 and 0.5"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise SyntheticEndoscopyStickerError(
                "jpeg_quality must be between 1 and 100"
            )


@dataclass(frozen=True)
class SyntheticEndoscopyStickerReport:
    schema_version: int
    output_root: str
    seed: int
    sources: int
    frames: int
    positive_frames: int
    negative_frames: int
    annotations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SYNTHETIC_NAMES = (
    "Anna Bergmann",
    "Jonas Keller",
    "Miriam Vogt",
    "David Neumann",
    "Sofia Hartmann",
    "Lukas Werner",
    "Nora Albrecht",
    "Emil Brandt",
    "Lea Sommer",
    "Felix Conrad",
)


def _decode_image(path: Path) -> object:
    """Keep OpenCV's dynamically nullable decode result at the integration edge."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def sanitize_endoscopy_background(
    source: ImageArray, output_width: int, masked_left_fraction: float
) -> ImageArray:
    """Resize a frame and irreversibly remove its original left metadata panel."""
    if source.ndim != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        raise ValueError("source must be a three-channel uint8 BGR image")
    height, width = source.shape[:2]
    output_height = max(1, round(height * output_width / width))
    resized = cv2.resize(
        source, (output_width, output_height), interpolation=cv2.INTER_AREA
    )
    panel_width = round(output_width * masked_left_fraction)
    resized[:, :panel_width] = 0
    return resized


def generate_synthetic_endoscopy_sticker_dataset(
    config: SyntheticEndoscopyStickerConfig,
) -> SyntheticEndoscopyStickerReport:
    rng = random.Random(config.seed)
    image_dir = config.output_root / "images" / "train"
    label_dir = config.output_root / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    positive_frames = 0
    negative_frames = 0

    for source_index, source_path in enumerate(config.source_images):
        decoded_object = _decode_image(source_path)
        if not isinstance(decoded_object, np.ndarray):
            raise SyntheticEndoscopyStickerError(
                f"OpenCV could not decode endoscopy source: {source_path}"
            )
        decoded = cast(ImageArray, decoded_object)
        base = sanitize_endoscopy_background(
            decoded, config.output_width, config.masked_left_fraction
        )
        for variant_index in range(config.frames_per_source):
            frame = _augment_background(base, rng)
            negative = rng.random() < config.negative_fraction
            annotations: list[StickerAnnotation] = []
            if not negative:
                identity = _synthetic_identity(rng)
                frame, annotation = _render_patient_sticker(
                    frame,
                    identity,
                    rng,
                    config.masked_left_fraction,
                    config.sticker_placement,
                )
                annotations.append(annotation)
                positive_frames += 1
            else:
                negative_frames += 1

            stem = f"endo_sticker_s{source_index:02d}_v{variant_index:03d}"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            if not cv2.imwrite(
                str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            ):
                raise SyntheticEndoscopyStickerError(
                    f"failed to write generated image: {image_path}"
                )
            _write_yolo_labels(label_path, annotations, frame.shape)
            records.append(
                {
                    "patient_key": f"synthetic-endo-{source_index:02d}-{variant_index:03d}",
                    "split": "train",
                    "source_path": str(source_path),
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "negative": negative,
                    "sticker_placement": config.sticker_placement,
                    "annotations": annotations,
                    "provenance": "synthetic_patient_sticker_on_sanitized_endoscopy_fixture",
                }
            )

    manifest_path = config.output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_dataset_yaml(config)
    report = SyntheticEndoscopyStickerReport(
        schema_version=1,
        output_root=str(config.output_root),
        seed=config.seed,
        sources=len(config.source_images),
        frames=len(records),
        positive_frames=positive_frames,
        negative_frames=negative_frames,
        annotations=positive_frames,
    )
    (config.output_root / "summary.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return report


def _augment_background(source: ImageArray, rng: random.Random) -> ImageArray:
    alpha = rng.uniform(0.82, 1.18)
    beta = rng.uniform(-14.0, 14.0)
    augmented = cv2.convertScaleAbs(source, alpha=alpha, beta=beta)
    if rng.random() < 0.35:
        kernel = rng.choice((3, 5))
        augmented = cv2.GaussianBlur(augmented, (kernel, kernel), rng.uniform(0.2, 1.2))
    if rng.random() < 0.35:
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        noise = noise_rng.normal(0.0, rng.uniform(1.0, 4.0), augmented.shape)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(
            np.uint8
        )
    return augmented


def _synthetic_identity(rng: random.Random) -> tuple[str, str, str, str]:
    name = rng.choice(_SYNTHETIC_NAMES)
    day = rng.randrange(1, 29)
    month = rng.randrange(1, 13)
    year = rng.randrange(1945, 2003)
    birth_date = f"{day:02d}.{month:02d}.{year}"
    patient_id = f"PID {rng.randrange(100000, 999999)}"
    case_id = f"CASE {rng.randrange(10000000, 99999999)}"
    return name, birth_date, patient_id, case_id


def _render_patient_sticker(
    frame: ImageArray,
    identity: tuple[str, str, str, str],
    rng: random.Random,
    masked_left_fraction: float,
    sticker_placement: StickerPlacement,
) -> tuple[ImageArray, StickerAnnotation]:
    height, width = frame.shape[:2]
    panel_width = round(width * masked_left_fraction)
    sticker_width = rng.randrange(
        max(170, panel_width - 90), max(171, panel_width - 18)
    )
    sticker_height = rng.randrange(92, min(158, max(93, height // 3)))
    pad = 9
    sticker = np.full(
        (sticker_height, sticker_width, 3),
        rng.choice(((238, 241, 235), (224, 238, 244), (244, 235, 218))),
        dtype=np.uint8,
    )
    cv2.rectangle(
        sticker, (0, 0), (sticker_width - 1, sticker_height - 1), (80, 80, 80), 1
    )
    name, birth_date, patient_id, case_id = identity
    lines = (name.upper(), f"DOB {birth_date}", patient_id, case_id)
    font = rng.choice((cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX))
    scale = rng.uniform(0.34, 0.48)
    thickness = 1
    line_height = max(16, (sticker_height - 2 * pad) // 4)
    for index, line in enumerate(lines):
        baseline_y = pad + (index + 1) * line_height - 3
        current_scale = scale
        text_width = cv2.getTextSize(line, font, current_scale, thickness)[0][0]
        while text_width > sticker_width - 2 * pad and current_scale > 0.25:
            current_scale *= 0.9
            text_width = cv2.getTextSize(line, font, current_scale, thickness)[0][0]
        cv2.putText(
            sticker,
            line,
            (pad, baseline_y),
            font,
            current_scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )
    _draw_barcode(sticker, rng)

    border = 10
    padded = cv2.copyMakeBorder(
        sticker, border, border, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    mask = np.zeros(padded.shape[:2], dtype=np.uint8)
    mask[border : border + sticker_height, border : border + sticker_width] = 255
    center = (padded.shape[1] / 2.0, padded.shape[0] / 2.0)
    matrix: npt.NDArray[np.float64] = cv2.getRotationMatrix2D(  # pyright: ignore[reportUnknownMemberType]
        center, rng.uniform(-4.0, 4.0), 1.0
    )
    rotated = cv2.warpAffine(padded, matrix, (padded.shape[1], padded.shape[0]))
    rotated_mask = cv2.warpAffine(mask, matrix, (padded.shape[1], padded.shape[0]))
    nonzero = cv2.findNonZero(rotated_mask)
    if nonzero is None:
        raise SyntheticEndoscopyStickerError("rotated sticker mask is empty")
    rect: tuple[int, int, int, int] = cv2.boundingRect(  # pyright: ignore[reportUnknownMemberType]
        nonzero
    )
    rx, ry, rw, rh = rect
    if sticker_placement == "lens":
        lens_start = panel_width + 12
        lens_end = width - padded.shape[1] - 4
        if lens_end < lens_start:
            raise SyntheticEndoscopyStickerError(
                "output frame is too narrow for lens sticker placement"
            )
        x = rng.randint(lens_start, lens_end)
    else:
        x = rng.randrange(2, max(3, panel_width - padded.shape[1]))
    y = rng.randrange(4, max(5, height - padded.shape[0] - 4))
    roi = frame[y : y + padded.shape[0], x : x + padded.shape[1]]
    selector = rotated_mask > 0
    roi[selector] = rotated[selector]
    text = " | ".join(lines)
    return frame, {"text": text, "box": (x + rx, y + ry, x + rx + rw, y + ry + rh)}


def _draw_barcode(sticker: ImageArray, rng: random.Random) -> None:
    height, width = sticker.shape[:2]
    x = width - 43
    top = max(5, height - 22)
    for _ in range(12):
        bar_width = rng.choice((1, 1, 2))
        cv2.rectangle(sticker, (x, top), (x + bar_width, height - 6), (30, 30, 30), -1)
        x += bar_width + rng.choice((1, 2))


def _write_yolo_labels(
    path: Path, annotations: Sequence[StickerAnnotation], shape: tuple[int, ...]
) -> None:
    height, width = shape[:2]
    rows: list[str] = []
    for annotation in annotations:
        x1, y1, x2, y2 = annotation["box"]
        rows.append(
            "0 "
            f"{((x1 + x2) / 2.0) / width:.8f} "
            f"{((y1 + y2) / 2.0) / height:.8f} "
            f"{(x2 - x1) / width:.8f} "
            f"{(y2 - y1) / height:.8f}"
        )
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_dataset_yaml(config: SyntheticEndoscopyStickerConfig) -> None:
    if config.combined_dataset_root is None:
        content = (
            f"path: {config.output_root}\n"
            "train: images/train\n"
            "val: images/train\n"
            "test: images/train\n"
        )
    else:
        root = config.combined_dataset_root
        content = f"path: {root.parent}\ntrain:\n  - {root.name}/images/train\n"
        content += "".join(
            f"  - {train_root.name}/images/train\n"
            for train_root in config.additional_train_roots
        )
        content += (
            f"  - {config.output_root.name}/images/train\n"
            f"val: {root.name}/images/val\n"
            f"test: {root.name}/images/test\n"
        )
    (config.output_root / "dataset.yaml").write_text(
        content + "names:\n  0: phi\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer-generate-endoscopy-stickers",
        description="Generate sanitized endoscopy frames with synthetic patient stickers.",
    )
    parser.add_argument("--source-image", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--combined-dataset-root", type=Path)
    parser.add_argument("--additional-train-root", type=Path, action="append")
    parser.add_argument(
        "--sticker-placement", choices=("panel", "lens"), default="panel"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames-per-source", type=int, default=60)
    parser.add_argument("--negative-fraction", type=float, default=0.1)
    parser.add_argument("--output-width", type=int, default=960)
    parser.add_argument("--masked-left-fraction", type=float, default=0.30)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = generate_synthetic_endoscopy_sticker_dataset(
        SyntheticEndoscopyStickerConfig(
            source_images=tuple(cast(list[Path], args.source_image)),
            output_root=cast(Path, args.output_root),
            combined_dataset_root=cast(Path | None, args.combined_dataset_root),
            additional_train_roots=tuple(
                cast(list[Path] | None, args.additional_train_root) or ()
            ),
            sticker_placement=cast(StickerPlacement, args.sticker_placement),
            seed=cast(int, args.seed),
            frames_per_source=cast(int, args.frames_per_source),
            negative_fraction=cast(float, args.negative_fraction),
            output_width=cast(int, args.output_width),
            masked_left_fraction=cast(float, args.masked_left_fraction),
            jpeg_quality=cast(int, args.jpeg_quality),
        )
    )
    print(json.dumps(report.to_dict(), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
