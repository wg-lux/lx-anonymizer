from __future__ import annotations

# pydicom intentionally exposes dynamic Dataset/DataElement values at this
# integration boundary.
# pyright: reportMissingTypeStubs=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

import argparse
import csv
import hashlib
import hmac
import json
import logging
import os
import tempfile
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pydicom  # type: ignore[import-untyped]
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset  # type: ignore[import-untyped]
from pydicom.uid import ExplicitVRLittleEndian, UID  # type: ignore[import-untyped]

from lx_anonymizer.region_processing.box_operations import Box, fill_boxes
from lx_anonymizer.text_detection.phi_region_detector import (
    CustomPhiRegionDetector,
    PhiRegionDetectorConfig,
)

logger = logging.getLogger(__name__)

PixelArray = npt.NDArray[np.generic]

_DIRECT_TEXT_KEYWORDS = frozenset(
    {
        "AccessionNumber",
        "AdditionalPatientHistory",
        "AdmittingDiagnosesDescription",
        "InstitutionAddress",
        "InstitutionName",
        "InstitutionalDepartmentName",
        "MedicalAlerts",
        "MilitaryRank",
        "Occupation",
        "OtherPatientIDs",
        "OtherPatientNames",
        "PatientAddress",
        "PatientComments",
        "PatientMotherBirthName",
        "PatientTelephoneNumbers",
        "ReferringPhysicianAddress",
        "ReferringPhysicianTelephoneNumbers",
        "RequestingService",
        "StudyID",
    }
)

_UID_KEYWORD_EXCLUSIONS = frozenset(
    {
        "CodingSchemeUID",
        "ContextGroupExtensionCreatorUID",
        "ImplementationClassUID",
        "MediaStorageSOPClassUID",
        "ReferencedSOPClassUID",
        "RelatedGeneralSOPClassUID",
        "SOPClassUID",
        "TransferSyntaxUID",
    }
)

_FREE_TEXT_VRS = frozenset({"AE", "LO", "LT", "SH", "ST", "UC", "UR", "UT"})


class DicomAnonymizationError(RuntimeError):
    """Raised when an output DICOM cannot satisfy anonymization invariants."""


@dataclass(frozen=True)
class DicomExportConfig:
    input_root: Path
    output_root: Path
    secret_file: Path
    model_path: Path | None
    confidence_threshold: float = 0.1
    nms_threshold: float = 0.45
    input_size: int = 960
    pixel_modalities: frozenset[str] = frozenset({"CR", "DX", "MG"})
    max_files: int | None = None

    def __post_init__(self) -> None:
        if not self.input_root.is_dir():
            raise DicomAnonymizationError(
                f"DICOM input root does not exist: {self.input_root}"
            )
        if not self.secret_file.is_file():
            raise DicomAnonymizationError(
                f"pseudonym secret file does not exist: {self.secret_file}"
            )
        if self.model_path is not None and not self.model_path.is_file():
            raise DicomAnonymizationError(
                f"PHI detector model does not exist: {self.model_path}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        if self.input_size < 1:
            raise ValueError("input_size must be positive")
        if self.max_files is not None and self.max_files < 1:
            raise ValueError("max_files must be positive")


@dataclass(frozen=True)
class DicomFileResult:
    input_path: str
    output_path: str
    original_sop_instance_uid: str
    anonymized_sop_instance_uid: str
    modality: str
    masked_regions: int


@dataclass(frozen=True)
class DicomExportSummary:
    schema_version: int
    input_root: str
    output_root: str
    files_discovered: int
    files_written: int
    patient_mappings: int
    uid_mappings: int
    masked_files: int
    masked_regions: int
    results: list[DicomFileResult]


class StablePseudonymRegistry:
    """Deterministic patient/UID/date mappings with optional persisted seeds."""

    def __init__(
        self,
        secret: bytes,
        *,
        patient_mappings: Mapping[str, str] | None = None,
        uid_mappings: Mapping[str, str] | None = None,
    ) -> None:
        if len(secret) < 32:
            raise DicomAnonymizationError(
                "pseudonym secret must contain at least 32 bytes"
            )
        self._secret = secret
        self._patient_mappings: dict[str, str] = dict(patient_mappings or {})
        self._uid_mappings: dict[str, str] = dict(uid_mappings or {})
        _validate_one_to_one(self._patient_mappings, "patient")
        _validate_one_to_one(self._uid_mappings, "UID")

    @classmethod
    def from_files(
        cls,
        secret_file: Path,
        *,
        patient_mapping_file: Path | None = None,
        uid_mapping_file: Path | None = None,
    ) -> StablePseudonymRegistry:
        secret = secret_file.read_bytes().strip()
        return cls(
            secret,
            patient_mappings=_read_mapping_csv(patient_mapping_file),
            uid_mappings=_read_mapping_csv(uid_mapping_file),
        )

    @property
    def patient_mappings(self) -> Mapping[str, str]:
        return self._patient_mappings

    @property
    def uid_mappings(self) -> Mapping[str, str]:
        return self._uid_mappings

    def patient_id(self, original: str) -> str:
        normalized = original.strip()
        if not normalized:
            raise DicomAnonymizationError("PatientID must not be empty")
        existing = self._patient_mappings.get(normalized)
        if existing is not None:
            return existing
        digest = self._digest("patient", normalized).hex()
        pseudonym = f"PAT_{digest[:32]}"
        self._insert_unique(self._patient_mappings, normalized, pseudonym, "patient")
        return pseudonym

    def uid(self, original: str) -> str:
        normalized = original.strip()
        if not normalized:
            raise DicomAnonymizationError("UID must not be empty")
        existing = self._uid_mappings.get(normalized)
        if existing is not None:
            return existing
        value = int.from_bytes(self._digest("uid", normalized)[:16], "big")
        pseudonym = f"2.25.{value}"
        self._insert_unique(self._uid_mappings, normalized, pseudonym, "UID")
        return pseudonym

    def date_shift_days(self, patient_id: str) -> int:
        value = int.from_bytes(self._digest("date-shift", patient_id)[:4], "big")
        magnitude = (value % 3650) + 1
        return -magnitude if value & 1 else magnitude

    def write_csvs(self, directory: Path) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        patient_path = directory / "patient_id_mapping.csv"
        uid_path = directory / "uid_mapping.csv"
        _write_mapping_csv(patient_path, self._patient_mappings)
        _write_mapping_csv(uid_path, self._uid_mappings)
        return patient_path, uid_path

    def _digest(self, namespace: str, value: str) -> bytes:
        return hmac.new(
            self._secret,
            f"{namespace}:{value}".encode("utf-8"),
            hashlib.sha256,
        ).digest()

    @staticmethod
    def _insert_unique(
        mappings: MutableMapping[str, str],
        original: str,
        pseudonym: str,
        label: str,
    ) -> None:
        conflicting = next(
            (
                old
                for old, new in mappings.items()
                if new == pseudonym and old != original
            ),
            None,
        )
        if conflicting is not None:
            raise DicomAnonymizationError(
                f"{label} pseudonym collision for {original!r} and {conflicting!r}"
            )
        mappings[original] = pseudonym


class DicomAnonymizer:
    def __init__(
        self,
        registry: StablePseudonymRegistry,
        detector: CustomPhiRegionDetector | None,
        *,
        pixel_modalities: frozenset[str],
    ) -> None:
        self._registry = registry
        self._detector = detector
        self._pixel_modalities = pixel_modalities

    def anonymize_dataset(self, dataset: Dataset) -> tuple[Dataset, int]:
        original_patient_id = _required_dataset_text(dataset, "PatientID")
        original_sop_uid = _required_dataset_text(dataset, "SOPInstanceUID")
        date_shift_days = self._registry.date_shift_days(original_patient_id)

        masked_regions = self._mask_pixels(dataset)
        _sanitize_dataset(
            dataset,
            registry=self._registry,
            date_shift_days=date_shift_days,
            forbidden_text=original_patient_id,
        )
        _synchronize_file_meta(dataset, self._registry, original_sop_uid)
        dataset.PatientIdentityRemoved = "YES"
        dataset.DeidentificationMethod = (
            "LX Anonymizer deterministic DICOM profile and PHI pixel detector"
        )
        _validate_dataset(
            dataset,
            original_patient_id=original_patient_id,
            original_sop_uid=original_sop_uid,
            expected_patient_id=self._registry.patient_id(original_patient_id),
            expected_sop_uid=self._registry.uid(original_sop_uid),
        )
        return dataset, masked_regions

    def process_file(self, input_path: Path, output_root: Path) -> DicomFileResult:
        dataset = pydicom.dcmread(input_path)
        original_sop_uid = _required_dataset_text(dataset, "SOPInstanceUID")
        dataset, masked_regions = self.anonymize_dataset(dataset)
        patient_id = _required_dataset_text(dataset, "PatientID")
        study_uid = _required_dataset_text(dataset, "StudyInstanceUID")
        series_uid = _required_dataset_text(dataset, "SeriesInstanceUID")
        sop_uid = _required_dataset_text(dataset, "SOPInstanceUID")
        destination = (
            output_root / patient_id / study_uid / series_uid / f"{sop_uid}.dcm"
        )
        _atomic_save_dataset(dataset, destination)
        return DicomFileResult(
            input_path=str(input_path),
            output_path=str(destination),
            original_sop_instance_uid=original_sop_uid,
            anonymized_sop_instance_uid=sop_uid,
            modality=str(getattr(dataset, "Modality", "") or ""),
            masked_regions=masked_regions,
        )

    def _mask_pixels(self, dataset: Dataset) -> int:
        modality = str(getattr(dataset, "Modality", "") or "").strip().upper()
        if (
            self._detector is None
            or modality not in self._pixel_modalities
            or "PixelData" not in dataset
        ):
            return 0

        pixels = cast(PixelArray, np.asarray(dataset.pixel_array))
        frames = _pixel_frames(dataset, pixels)
        boxes_by_frame: list[list[Box]] = []
        for frame in frames:
            boxes_by_frame.append(
                self._detector.detect(_frame_to_image(dataset, frame))
            )
        masked_regions = sum(len(boxes) for boxes in boxes_by_frame)
        if masked_regions == 0:
            return 0

        fill_value = _black_pixel_value(dataset)
        masked_pixels = pixels.copy()
        if len(frames) == 1 and _is_single_frame_array(dataset, pixels):
            masked_pixels = fill_boxes(
                masked_pixels,
                boxes_by_frame[0],
                fill_value=fill_value,
            )
        else:
            for frame_index, boxes in enumerate(boxes_by_frame):
                if boxes:
                    masked_pixels[frame_index] = fill_boxes(
                        cast(PixelArray, masked_pixels[frame_index]),
                        boxes,
                        fill_value=fill_value,
                    )
        _replace_pixel_data(dataset, masked_pixels)
        return masked_regions


def export_dicom_tree(
    config: DicomExportConfig,
    registry: StablePseudonymRegistry,
) -> DicomExportSummary:
    if config.output_root.exists() and any(config.output_root.iterdir()):
        raise DicomAnonymizationError(
            f"output root must be absent or empty: {config.output_root}"
        )
    config.output_root.mkdir(parents=True, exist_ok=True)
    detector = _build_detector(config)
    anonymizer = DicomAnonymizer(
        registry,
        detector,
        pixel_modalities=config.pixel_modalities,
    )
    paths = sorted(config.input_root.rglob("*.dcm"))
    discovered = len(paths)
    if config.max_files is not None:
        paths = paths[: config.max_files]

    results: list[DicomFileResult] = []
    for index, path in enumerate(paths, start=1):
        result = anonymizer.process_file(path, config.output_root)
        results.append(result)
        if index == 1 or index % 100 == 0 or index == len(paths):
            logger.info(
                "DICOM export progress: %s/%s masked_regions=%s",
                index,
                len(paths),
                sum(item.masked_regions for item in results),
            )

    summary = DicomExportSummary(
        schema_version=1,
        input_root=str(config.input_root),
        output_root=str(config.output_root),
        files_discovered=discovered,
        files_written=len(results),
        patient_mappings=len(registry.patient_mappings),
        uid_mappings=len(registry.uid_mappings),
        masked_files=sum(item.masked_regions > 0 for item in results),
        masked_regions=sum(item.masked_regions for item in results),
        results=results,
    )
    return summary


def write_export_artifacts(
    summary: DicomExportSummary,
    registry: StablePseudonymRegistry,
    artifact_root: Path,
) -> tuple[Path, Path, Path]:
    artifact_root.mkdir(parents=True, exist_ok=True)
    patient_path, uid_path = registry.write_csvs(artifact_root)
    summary_path = artifact_root / "dicom_export_summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(asdict(summary), indent=2, ensure_ascii=True) + "\n",
    )
    return patient_path, uid_path, summary_path


def write_validator_config(
    path: Path,
    *,
    run_name: str,
    dicom_root: Path,
    validator_output_root: Path,
    answer_db: Path,
    uid_mapping_file: Path,
    patient_mapping_file: Path,
    log_root: Path,
    multiprocessing_cpus: int,
) -> None:
    if multiprocessing_cpus < 1:
        raise ValueError("multiprocessing_cpus must be positive")
    payload: dict[str, str] = {
        "run_name": run_name,
        "input_data_path": str(dicom_root.resolve()),
        "output_data_path": str(validator_output_root.resolve()),
        "answer_db_file": str(answer_db.resolve()),
        "uid_mapping_file": str(uid_mapping_file.resolve()),
        "patid_mapping_file": str(patient_mapping_file.resolve()),
        "multiprocessing": "True",
        "multiprocessing_cpus": str(multiprocessing_cpus),
        "log_path": str(log_root.resolve()),
        "log_level": "info",
        "report_series": "False",
    }
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lx-anonymizer-export-dicom",
        description="Create a validator-ready anonymized DICOM tree and mappings.",
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--patient-mapping-input", type=Path)
    parser.add_argument("--uid-mapping-input", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--input-size", type=int, default=960)
    parser.add_argument("--pixel-modalities", default="CR,DX,MG")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--answer-db", type=Path)
    parser.add_argument("--validator-config", type=Path)
    parser.add_argument("--validator-output-root", type=Path)
    parser.add_argument("--validator-log-root", type=Path)
    parser.add_argument("--run-name", default="MIDI_B_Validation")
    parser.add_argument("--validator-cpus", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    # MIDI-B intentionally contains malformed source identifiers and character
    # set declarations. They are replaced or preserved according to the export
    # profile; repeating pydicom's source warning for every nested value obscures
    # actionable fail-closed errors.
    logging.getLogger("pydicom").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r"Invalid value for VR UI:.*",
        module=r"pydicom\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Unknown encoding .*",
        module=r"pydicom\..*",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    input_root = cast(Path, args.input_root).expanduser().resolve()
    output_root = cast(Path, args.output_root).expanduser().resolve()
    artifact_root = cast(Path, args.artifact_root).expanduser().resolve()
    config = DicomExportConfig(
        input_root=input_root,
        output_root=output_root,
        secret_file=cast(Path, args.secret_file).expanduser().resolve(),
        model_path=(
            cast(Path, args.model_path).expanduser().resolve()
            if args.model_path is not None
            else None
        ),
        confidence_threshold=cast(float, args.confidence_threshold),
        nms_threshold=cast(float, args.nms_threshold),
        input_size=cast(int, args.input_size),
        pixel_modalities=frozenset(
            part.strip().upper()
            for part in cast(str, args.pixel_modalities).split(",")
            if part.strip()
        ),
        max_files=cast(int | None, args.max_files),
    )
    registry = StablePseudonymRegistry.from_files(
        config.secret_file,
        patient_mapping_file=cast(Path | None, args.patient_mapping_input),
        uid_mapping_file=cast(Path | None, args.uid_mapping_input),
    )
    summary = export_dicom_tree(config, registry)
    patient_path, uid_path, summary_path = write_export_artifacts(
        summary,
        registry,
        artifact_root,
    )

    validator_config = cast(Path | None, args.validator_config)
    if validator_config is not None:
        answer_db = cast(Path | None, args.answer_db)
        validator_output_root = cast(Path | None, args.validator_output_root)
        validator_log_root = cast(Path | None, args.validator_log_root)
        if (
            answer_db is None
            or validator_output_root is None
            or validator_log_root is None
        ):
            parser.error(
                "--validator-config requires --answer-db, --validator-output-root, "
                "and --validator-log-root"
            )
        write_validator_config(
            validator_config.expanduser().resolve(),
            run_name=cast(str, args.run_name),
            dicom_root=output_root,
            validator_output_root=validator_output_root.expanduser().resolve(),
            answer_db=answer_db.expanduser().resolve(),
            uid_mapping_file=uid_path,
            patient_mapping_file=patient_path,
            log_root=validator_log_root.expanduser().resolve(),
            multiprocessing_cpus=cast(int, args.validator_cpus),
        )

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "patient_mapping": str(patient_path),
                "uid_mapping": str(uid_path),
                "files_written": summary.files_written,
                "masked_files": summary.masked_files,
                "masked_regions": summary.masked_regions,
            },
            ensure_ascii=True,
        )
    )
    return 0


def _build_detector(config: DicomExportConfig) -> CustomPhiRegionDetector | None:
    if config.model_path is None:
        return None
    return CustomPhiRegionDetector(
        PhiRegionDetectorConfig(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            nms_threshold=config.nms_threshold,
            input_size=config.input_size,
            box_format="yolo_xywh",
            score_format="class_scores",
            allowed_class_ids=frozenset(),
            resize_mode="letterbox",
            required=True,
        )
    )


def _sanitize_dataset(
    dataset: Dataset,
    *,
    registry: StablePseudonymRegistry,
    date_shift_days: int,
    forbidden_text: str,
) -> None:
    for element in list(dataset):
        if element.tag.is_private:
            del dataset[element.tag]
            continue
        if element.VR == "SQ":
            for item in element.value:
                _sanitize_dataset(
                    item,
                    registry=registry,
                    date_shift_days=date_shift_days,
                    forbidden_text=forbidden_text,
                )
            continue

        keyword = str(element.keyword or "")
        if keyword == "PatientID":
            element.value = registry.patient_id(str(element.value))
        elif element.VR == "PN":
            element.value = "ANON^PATIENT" if keyword == "PatientName" else ""
        elif element.VR == "UI" and _should_map_uid(keyword):
            element.value = _map_uid_value(element.value, registry)
        elif element.VR == "DA":
            element.value = _shift_date_value(element.value, date_shift_days)
        elif element.VR == "DT":
            element.value = _shift_datetime_value(element.value, date_shift_days)
        elif keyword in _DIRECT_TEXT_KEYWORDS:
            element.value = ""
        elif element.VR in _FREE_TEXT_VRS and forbidden_text in str(element.value):
            element.value = ""


def _should_map_uid(keyword: str) -> bool:
    return (
        bool(keyword)
        and keyword not in _UID_KEYWORD_EXCLUSIONS
        and not keyword.endswith("ClassUID")
    )


def _map_uid_value(value: object, registry: StablePseudonymRegistry) -> object:
    if isinstance(value, str):
        return registry.uid(value) if value.strip() else value
    if isinstance(value, Sequence):
        return [registry.uid(str(item)) for item in value if str(item).strip()]
    raw = str(value).strip()
    return registry.uid(raw) if raw else value


def _shift_date_value(value: object, days: int) -> object:
    if isinstance(value, str):
        return _shift_date_text(value, days)
    if isinstance(value, Sequence):
        return [_shift_date_text(str(item), days) for item in value]
    return _shift_date_text(str(value), days)


def _shift_date_text(value: str, days: int) -> str:
    raw = value.strip()
    if not raw:
        return raw
    try:
        parsed = date.fromisoformat(f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}")
    except (ValueError, IndexError):
        return ""
    return (parsed + timedelta(days=days)).strftime("%Y%m%d")


def _shift_datetime_value(value: object, days: int) -> object:
    if isinstance(value, str):
        return _shift_datetime_text(value, days)
    if isinstance(value, Sequence):
        return [_shift_datetime_text(str(item), days) for item in value]
    return _shift_datetime_text(str(value), days)


def _shift_datetime_text(value: str, days: int) -> str:
    raw = value.strip()
    if len(raw) < 8:
        return "" if raw else raw
    shifted = _shift_date_text(raw[:8], days)
    return f"{shifted}{raw[8:]}" if shifted else ""


def _synchronize_file_meta(
    dataset: Dataset,
    registry: StablePseudonymRegistry,
    original_sop_uid: str,
) -> None:
    if not hasattr(dataset, "file_meta"):
        dataset.file_meta = FileMetaDataset()
    expected_sop_uid = registry.uid(original_sop_uid)
    dataset.SOPInstanceUID = expected_sop_uid
    dataset.file_meta.MediaStorageSOPInstanceUID = UID(expected_sop_uid)
    if "SOPClassUID" in dataset:
        dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID


def _pixel_frames(dataset: Dataset, pixels: PixelArray) -> list[PixelArray]:
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    if pixels.ndim == 2:
        return [pixels]
    if pixels.ndim == 3 and samples > 1:
        return [pixels]
    if pixels.ndim == 3:
        return [cast(PixelArray, pixels[index]) for index in range(pixels.shape[0])]
    if pixels.ndim == 4 and samples > 1:
        return [cast(PixelArray, pixels[index]) for index in range(pixels.shape[0])]
    raise DicomAnonymizationError(f"unsupported decoded pixel shape: {pixels.shape}")


def _is_single_frame_array(dataset: Dataset, pixels: PixelArray) -> bool:
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    return pixels.ndim == 2 or (pixels.ndim == 3 and samples > 1)


def _frame_to_image(dataset: Dataset, frame: PixelArray) -> Image.Image:
    if frame.ndim == 3:
        return Image.fromarray(_normalize_color(frame)).convert("RGB")
    if frame.ndim != 2:
        raise DicomAnonymizationError(f"unsupported frame shape: {frame.shape}")
    grayscale = _normalize_grayscale(frame)
    if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
        grayscale = 255 - grayscale
    return Image.fromarray(grayscale).convert("RGB")


def _normalize_grayscale(pixels: PixelArray) -> npt.NDArray[np.uint8]:
    values = pixels.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise DicomAnonymizationError("DICOM pixels contain no finite values")
    lower, upper = np.percentile(finite, (0.5, 99.5))
    if upper <= lower:
        lower, upper = float(np.min(finite)), float(np.max(finite))
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lower) * (255.0 / (upper - lower))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _normalize_color(pixels: PixelArray) -> npt.NDArray[np.uint8]:
    color = pixels[..., :3]
    if color.dtype == np.uint8:
        return cast(npt.NDArray[np.uint8], color)
    values = color.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise DicomAnonymizationError("DICOM pixels contain no finite values")
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if upper <= lower:
        return np.zeros(values.shape, dtype=np.uint8)
    return np.clip((values - lower) * (255.0 / (upper - lower)), 0.0, 255.0).astype(
        np.uint8
    )


def _black_pixel_value(dataset: Dataset) -> int:
    bits_stored = int(getattr(dataset, "BitsStored", 8) or 8)
    signed = int(getattr(dataset, "PixelRepresentation", 0) or 0) == 1
    monochrome1 = (
        str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1"
    )
    if monochrome1:
        return (1 << (bits_stored - (1 if signed else 0))) - 1
    return -(1 << (bits_stored - 1)) if signed else 0


def _replace_pixel_data(dataset: Dataset, pixels: PixelArray) -> None:
    if not hasattr(dataset, "file_meta"):
        dataset.file_meta = FileMetaDataset()
    dtype = pixels.dtype.newbyteorder("<")
    raw = pixels.astype(dtype, copy=False).tobytes(order="C")
    if len(raw) % 2:
        raw += b"\x00"
    dataset.PixelData = raw
    dataset["PixelData"].is_undefined_length = False
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    if int(getattr(dataset, "SamplesPerPixel", 1) or 1) > 1:
        dataset.PlanarConfiguration = 0
        dataset.PhotometricInterpretation = "RGB"
    for keyword in ("LossyImageCompressionRatio", "LossyImageCompressionMethod"):
        if keyword in dataset:
            del dataset[keyword]
    dataset.LossyImageCompression = "00"


def _validate_dataset(
    dataset: Dataset,
    *,
    original_patient_id: str,
    original_sop_uid: str,
    expected_patient_id: str,
    expected_sop_uid: str,
) -> None:
    if _required_dataset_text(dataset, "PatientID") != expected_patient_id:
        raise DicomAnonymizationError("PatientID mapping invariant failed")
    if _required_dataset_text(dataset, "SOPInstanceUID") != expected_sop_uid:
        raise DicomAnonymizationError("SOPInstanceUID mapping invariant failed")
    file_meta_uid = str(dataset.file_meta.MediaStorageSOPInstanceUID).strip()
    if file_meta_uid != expected_sop_uid:
        raise DicomAnonymizationError(
            "MediaStorageSOPInstanceUID does not match SOPInstanceUID"
        )
    if (
        original_patient_id == expected_patient_id
        or original_sop_uid == expected_sop_uid
    ):
        raise DicomAnonymizationError("direct identifier was not transformed")
    _assert_value_absent(dataset, original_patient_id)


def _assert_value_absent(dataset: Dataset, forbidden: str) -> None:
    for element in dataset:
        if element.VR == "SQ":
            for item in element.value:
                _assert_value_absent(item, forbidden)
            continue
        if element.tag == 0x7FE00010:
            continue
        if forbidden and forbidden in str(element.value):
            raise DicomAnonymizationError(
                f"original PatientID remains in tag {element.tag}"
            )


def _required_dataset_text(dataset: Dataset, keyword: str) -> str:
    value = str(getattr(dataset, keyword, "") or "").strip()
    if not value:
        raise DicomAnonymizationError(f"DICOM dataset is missing {keyword}")
    return value


def _atomic_save_dataset(dataset: Dataset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        dataset.save_as(temp_path, enforce_file_format=True)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.debug(
        "%s",
        json.dumps(
            {"event": "dicom_written", "path": str(destination)},
            ensure_ascii=True,
        ),
    )


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _read_mapping_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DicomAnonymizationError(f"mapping CSV does not exist: {resolved}")
    mappings: dict[str, str] = {}
    with resolved.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id_old", "id_new"]:
            raise DicomAnonymizationError(
                f"mapping CSV must have id_old,id_new columns: {resolved}"
            )
        for row in reader:
            old, new = row["id_old"].strip(), row["id_new"].strip()
            if not old or not new:
                raise DicomAnonymizationError(
                    f"mapping CSV contains an empty value: {resolved}"
                )
            previous = mappings.get(old)
            if previous is not None and previous != new:
                raise DicomAnonymizationError(
                    f"mapping CSV contains conflicting rows for {old!r}"
                )
            mappings[old] = new
    _validate_one_to_one(mappings, "CSV")
    return mappings


def _write_mapping_csv(path: Path, mappings: Mapping[str, str]) -> None:
    rows = ["id_old,id_new"]
    for old, new in sorted(mappings.items()):
        rows.append(f"{_csv_cell(old)},{_csv_cell(new)}")
    _atomic_write_text(path, "\n".join(rows) + "\n")


def _csv_cell(value: str) -> str:
    if any(character in value for character in (",", '"', "\n", "\r")):
        return f'"{value.replace(chr(34), chr(34) * 2)}"'
    return value


def _validate_one_to_one(mappings: Mapping[str, str], label: str) -> None:
    if any(not old.strip() or not new.strip() for old, new in mappings.items()):
        raise DicomAnonymizationError(f"{label} mappings must not contain empty values")
    if len(set(mappings.values())) != len(mappings):
        raise DicomAnonymizationError(f"{label} mappings must be one-to-one")


if __name__ == "__main__":
    raise SystemExit(main())
