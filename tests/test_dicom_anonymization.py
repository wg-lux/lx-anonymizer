from __future__ import annotations

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pydicom  # type: ignore[import-untyped]
import pytest
from PIL import Image
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset  # type: ignore[import-untyped]
from pydicom.uid import (  # type: ignore[import-untyped]
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    UID,
)

from lx_anonymizer.dicom_anonymization import (
    DicomAnonymizationError,
    DicomAnonymizer,
    StablePseudonymRegistry,
    write_validator_config,
)
from lx_anonymizer.region_processing.box_operations import fill_boxes
from lx_anonymizer.text_detection.phi_region_detector import CustomPhiRegionDetector


class _FixedDetector:
    def detect(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        assert image.size == (16, 12)
        return [(2, 3, 8, 7)]


def _dataset(*, photometric_interpretation: str = "MONOCHROME2") -> FileDataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.826.0.1.3680043.10.1.3")
    dataset = FileDataset("", {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = "1.2.826.0.1.3680043.10.1.3"
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.10.1.1"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.10.1.2"
    dataset.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.1.4"
    dataset.PatientID = "ORIGINAL-123"
    dataset.PatientName = "Doe^Jane"
    dataset.ReferringPhysicianName = "Doctor^Private"
    dataset.RequestedProcedureDescription = "DX for ORIGINAL-123"
    dataset.StudyDate = "20260715"
    dataset.AcquisitionDateTime = "20260715112233.1234"
    dataset.Modality = "DX"
    dataset.Rows = 12
    dataset.Columns = 16
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric_interpretation
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.full((12, 16), 1000, dtype=np.uint16).tobytes()
    dataset.add_new((0x0011, 0x0010), "LO", "PRIVATE CREATOR")
    nested = Dataset()
    nested.ReferencedSOPClassUID = SecondaryCaptureImageStorage
    nested.ReferencedSOPInstanceUID = "1.2.826.0.1.3680043.10.1.9"
    nested.PatientID = "ORIGINAL-123"
    dataset.ReferencedImageSequence = [nested]
    return dataset


def _registry() -> StablePseudonymRegistry:
    return StablePseudonymRegistry(b"x" * 32)


def test_fill_boxes_returns_copy_and_clips_coordinates() -> None:
    source = np.full((10, 10), 7, dtype=np.uint16)

    result = fill_boxes(source, [(-2, 2, 4, 8)], fill_value=0)

    assert np.all(source == 7)
    assert np.all(result[2:8, 0:4] == 0)
    assert result.dtype == source.dtype


@pytest.mark.parametrize("box", [(1, 1, 1, 2), (20, 20, 30, 30)])
def test_fill_boxes_rejects_invalid_regions(box: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        fill_boxes(np.zeros((10, 10), dtype=np.uint8), [box])


def test_registry_is_deterministic_and_round_trips_csv(tmp_path: Path) -> None:
    first = StablePseudonymRegistry(b"a" * 32)
    second = StablePseudonymRegistry(b"a" * 32)

    assert first.patient_id("P1") == second.patient_id("P1")
    assert first.uid("1.2.3") == second.uid("1.2.3")
    assert first.date_shift_days("P1") == second.date_shift_days("P1")
    assert first.patient_id("P1") != StablePseudonymRegistry(b"b" * 32).patient_id("P1")

    patient_path, uid_path = first.write_csvs(tmp_path)
    secret_path = tmp_path / "secret"
    secret_path.write_bytes(b"a" * 32)
    loaded = StablePseudonymRegistry.from_files(
        secret_path,
        patient_mapping_file=patient_path,
        uid_mapping_file=uid_path,
    )
    assert loaded.patient_id("P1") == first.patient_id("P1")
    assert loaded.uid("1.2.3") == first.uid("1.2.3")
    with patient_path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [
            {"id_old": "P1", "id_new": first.patient_id("P1")}
        ]


def test_anonymize_dataset_recursively_transforms_identifiers_and_dates() -> None:
    dataset = _dataset()
    original_pixels = bytes(dataset.PixelData)
    registry = _registry()
    expected_shifted = (
        date(2026, 7, 15) + timedelta(days=registry.date_shift_days("ORIGINAL-123"))
    ).strftime("%Y%m%d")
    anonymizer = DicomAnonymizer(
        registry,
        detector=None,
        pixel_modalities=frozenset({"DX"}),
    )

    result, masked_regions = anonymizer.anonymize_dataset(dataset)

    assert masked_regions == 0
    assert bytes(result.PixelData) == original_pixels
    assert result.PatientID == registry.patient_id("ORIGINAL-123")
    assert result.PatientName == "ANON^PATIENT"
    assert result.ReferringPhysicianName == ""
    assert result.RequestedProcedureDescription == ""
    assert result.StudyDate == expected_shifted
    assert str(result.AcquisitionDateTime).startswith(expected_shifted)
    assert (0x0011, 0x0010) not in result
    nested = result.ReferencedImageSequence[0]
    assert nested.PatientID == registry.patient_id("ORIGINAL-123")
    assert nested.ReferencedSOPClassUID == SecondaryCaptureImageStorage
    assert nested.ReferencedSOPInstanceUID == registry.uid("1.2.826.0.1.3680043.10.1.9")
    assert result.SOPInstanceUID == result.file_meta.MediaStorageSOPInstanceUID
    assert result.PatientIdentityRemoved == "YES"


@pytest.mark.parametrize(
    ("photometric", "expected_fill"),
    [("MONOCHROME2", 0), ("MONOCHROME1", 4095)],
)
def test_pixel_detector_blackens_box_and_reencodes(
    photometric: str, expected_fill: int
) -> None:
    dataset = _dataset(photometric_interpretation=photometric)
    anonymizer = DicomAnonymizer(
        _registry(),
        detector=cast(CustomPhiRegionDetector, _FixedDetector()),
        pixel_modalities=frozenset({"DX"}),
    )

    result, masked_regions = anonymizer.anonymize_dataset(dataset)
    pixels = result.pixel_array

    assert masked_regions == 1
    assert np.all(pixels[3:7, 2:8] == expected_fill)
    assert np.all(pixels[0:2, 0:2] == 1000)
    assert result.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian


def test_process_file_writes_reloadable_consistent_dicom(tmp_path: Path) -> None:
    input_path = tmp_path / "input.dcm"
    output_root = tmp_path / "output"
    _dataset().save_as(input_path, enforce_file_format=True)
    registry = _registry()
    anonymizer = DicomAnonymizer(
        registry,
        detector=None,
        pixel_modalities=frozenset({"DX"}),
    )

    result = anonymizer.process_file(input_path, output_root)
    reloaded = pydicom.dcmread(result.output_path)

    assert Path(result.output_path).is_file()
    assert reloaded.PatientID == registry.patient_id("ORIGINAL-123")
    assert reloaded.SOPInstanceUID == reloaded.file_meta.MediaStorageSOPInstanceUID
    assert not list(output_root.rglob("*.tmp"))


def test_registry_rejects_non_unique_seed_mapping() -> None:
    with pytest.raises(DicomAnonymizationError, match="one-to-one"):
        StablePseudonymRegistry(
            b"x" * 32,
            patient_mappings={"A": "SAME", "B": "SAME"},
        )


def test_write_validator_config_uses_official_field_names(tmp_path: Path) -> None:
    config_path = tmp_path / "validator.json"
    answer_db = tmp_path / "answer.db"
    uid_map = tmp_path / "uid.csv"
    patient_map = tmp_path / "patient.csv"
    for path in (answer_db, uid_map, patient_map):
        path.touch()

    write_validator_config(
        config_path,
        run_name="run",
        dicom_root=tmp_path / "dicom",
        validator_output_root=tmp_path / "results",
        answer_db=answer_db,
        uid_mapping_file=uid_map,
        patient_mapping_file=patient_map,
        log_root=tmp_path / "logs",
        multiprocessing_cpus=4,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["uid_mapping_file"] == str(uid_map)
    assert payload["patid_mapping_file"] == str(patient_map)
    assert payload["multiprocessing_cpus"] == "4"
