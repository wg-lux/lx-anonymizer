from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType, TracebackType
from typing import Protocol, Self, cast
from unittest.mock import patch
from uuid import uuid4

import pymupdf  # type: ignore[import-untyped]
import pytest
from lx_dtypes.models.contracts.report_anonymization import (
    ReportAnonymizationRequest as SharedReportAnonymizationRequest,
)
from lx_dtypes.models.meta.ReportMeta import ReportProcessRequest, ReportProcessResult

from lx_anonymizer.anonymization.anonymizer import Anonymizer
from lx_anonymizer.report_contracts import (
    AnonymizationArtifactError,
    ArtifactAlreadyExistsError,
    OperationDeadlineExceededError,
    ReportAnonymizationRequest,
    SourceIdentityMismatchError,
)
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


class _WritablePdfPage(Protocol):
    def insert_text(self, point: tuple[int, int], text: str) -> int: ...


class _WritablePdfDocument(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def new_page(self) -> _WritablePdfPage: ...

    def save(self, filename: str) -> None: ...


def _request(
    *,
    source: Path,
    output_directory: Path,
) -> ReportAnonymizationRequest:
    payload = source.read_bytes()
    return ReportAnonymizationRequest(
        attempt_id=uuid4(),
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_size_bytes=len(payload),
        output_directory=output_directory,
    )


def _reader_with_fake_pipeline() -> ReportReader:
    reader = object.__new__(ReportReader)
    reader.llm_available = False

    def fake_process(
        self: ReportReader,
        request: ReportProcessRequest,
    ) -> ReportProcessResult:
        output_path = request.anonymized_pdf_output_path
        assert isinstance(output_path, Path)
        with cast(_WritablePdfDocument, pymupdf.open()) as document:
            page = document.new_page()
            page.insert_text((72, 72), "anonymized")
            document.save(str(output_path))
        return ReportProcessResult(
            text="original",
            anonymized_text="anonymized",
            report_meta={
                "patient_first_name": "ANON",
                "cropping_enabled": False,
                "paper_evaluation_metrics": {"schema_version": "1.0"},
            },
            anonymized_pdf_path=output_path,
        )

    reader._process_report_request = MethodType(  # pyright: ignore[reportPrivateUsage]
        fake_process,
        reader,
    )
    return reader


def test_process_report_is_the_canonical_processing_method() -> None:
    assert hasattr(ReportReader, "process_report")


def test_report_ocr_failure_preserves_cause_and_cleans_only_owned_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    original = source.read_bytes()
    output_directory = tmp_path / "attempt"
    output_directory.mkdir()
    unrelated = output_directory / "unrelated.pdf"
    unrelated.write_bytes(b"preserve")
    request = _request(source=source, output_directory=output_directory)
    reader = object.__new__(ReportReader)
    reader.llm_available = False
    reader.anonymizer = object.__new__(Anonymizer)
    failure = FileNotFoundError(
        "TESSDATA_PREFIX does not contain trained data for deu+eng"
    )

    def process(
        self: ReportReader, process_request: ReportProcessRequest
    ) -> ReportProcessResult:
        temporary = process_request.anonymized_pdf_output_path
        assert isinstance(temporary, Path)
        temporary.write_bytes(b"partial")
        self._maybe_create_anonymized_pdf(request=process_request, report_meta={})  # pyright: ignore[reportPrivateUsage]
        raise AssertionError("OCR configuration failure must propagate")

    reader._process_report_request = MethodType(process, reader)  # pyright: ignore[reportPrivateUsage]
    with (
        patch(
            "lx_anonymizer.anonymization.anonymizer.get_tessdata_path",
            side_effect=failure,
        ),
        pytest.raises(AnonymizationArtifactError, match="Report OCR runtime") as error,
    ):
        reader.process_report(request)

    assert error.value.__cause__ is failure
    assert source.read_bytes() == original
    assert list(output_directory.iterdir()) == [unrelated]
    assert unrelated.read_bytes() == b"preserve"


@pytest.mark.parametrize("late_name", [None, "", "unknown", "Conflicting"])
def test_process_report_returns_collected_best_prediction(
    tmp_path: Path, late_name: str | None
) -> None:
    # Arrange: collected evidence and a later partial extraction using report aliases.
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory = tmp_path / "attempt"
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)
    reader = object.__new__(ReportReader)
    reader.llm_available = False
    reader.patient_pseudonym_resolver = None
    reader.sensitive_meta = SensitiveMeta(first_name="Previous patient")

    def extract(text: str, pdf_path: Path | None) -> dict[str, object]:
        reader.sensitive_meta.safe_update(
            {"first_name": "Anna", "last_name": "Muster", "dob": "1980-02-03"}
        )
        return {
            "patient_first_name": late_name,
            "patient_last_name": late_name,
            "examination_date": "2026-09-01",
            "endoscope_sn": "SN-123",
        }

    def create_pdf(
        *, request: ReportProcessRequest, report_meta: dict[str, object]
    ) -> tuple[Path, dict[str, object]]:
        output_path = request.anonymized_pdf_output_path
        assert isinstance(output_path, Path)
        with cast(_WritablePdfDocument, pymupdf.open()) as document:
            page = document.new_page()
            page.insert_text((72, 72), "Anonymized report")
            document.save(str(output_path))
        return output_path, report_meta

    with (
        patch.object(
            reader, "_load_report_text", return_value="Anna Muster report text"
        ),
        patch.object(
            reader,
            "_apply_ocr_fallback_if_needed",
            return_value="Anna Muster report text",
        ),
        patch.object(reader, "extract_report_meta", side_effect=extract),
        patch.object(reader, "anonymize_report", return_value="Anonymized report"),
        patch.object(reader, "_maybe_create_anonymized_pdf", side_effect=create_pdf),
    ):
        # Act: exercise real collection, finalization and the public typed boundary.
        result = reader.process_report(request)

    # Assert: raw predictions survive anonymization and previous-run state is reset.
    prediction = result.extracted_metadata
    assert prediction.first_name == "Anna"
    assert prediction.last_name == "Muster"
    assert prediction.dob is not None
    assert prediction.dob.isoformat() == "1980-02-03"
    assert prediction.examination_date is not None
    assert prediction.examination_date.isoformat() == "2026-09-01"
    assert prediction.endoscope_sn == "SN-123"
    assert prediction.first_name == reader.sensitive_meta.first_name
    assert prediction.dob == reader.sensitive_meta.dob


def test_report_contract_import_is_the_shared_contract() -> None:
    assert ReportAnonymizationRequest is SharedReportAnonymizationRequest


def test_process_report_publishes_attempt_local_validated_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)

    result = _reader_with_fake_pipeline().process_report(request)

    assert result.contract_version == "report_anonymization"
    assert result.attempt_id == request.attempt_id
    assert result.source_sha256 == request.source_sha256
    assert result.artifact_path.parent == output_directory
    assert result.artifact_path.read_bytes().startswith(b"%PDF-")
    assert (
        result.artifact_sha256
        == hashlib.sha256(result.artifact_path.read_bytes()).hexdigest()
    )
    assert result.artifact_validation.page_count == 1
    assert result.artifact_validation.repaired is False
    assert result.provenance.anonymizer_version != "unknown"
    assert result.provenance.used_llm is False
    assert result.provenance.deterministic is True
    assert result.extracted_metadata.first_name == "ANON"
    assert not (output_directory / f".{request.attempt_id}.part.pdf").exists()


def test_process_report_rejects_source_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)
    source.write_bytes(b"%PDF-1.4\nchanged\n%%EOF\n")

    with pytest.raises(SourceIdentityMismatchError):
        _reader_with_fake_pipeline().process_report(request)


def test_process_report_refuses_existing_attempt_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)
    existing = output_directory / f"{request.attempt_id}.pdf"
    existing.write_bytes(b"existing")

    with pytest.raises(ArtifactAlreadyExistsError):
        _reader_with_fake_pipeline().process_report(request)

    assert existing.read_bytes() == b"existing"


def test_report_request_rejects_output_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    real_output = tmp_path / "real-output"
    linked_output = tmp_path / "linked-output"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    real_output.mkdir()
    linked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink"):
        _request(source=source, output_directory=linked_output)


def test_process_report_rejects_expired_deadline(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory).model_copy(
        update={"deadline_monotonic_ns": 1}
    )

    with pytest.raises(OperationDeadlineExceededError):
        _reader_with_fake_pipeline().process_report(request)


def test_process_report_rejects_structurally_invalid_pdf(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)
    reader = object.__new__(ReportReader)
    reader.llm_available = False

    def fake_process(
        self: ReportReader,
        process_request: ReportProcessRequest,
    ) -> ReportProcessResult:
        output_path = process_request.anonymized_pdf_output_path
        assert isinstance(output_path, Path)
        output_path.write_bytes(b"%PDF-this-is-not-structurally-valid")
        return ReportProcessResult(
            text="original",
            anonymized_text="anonymized",
            report_meta={},
            anonymized_pdf_path=output_path,
        )

    reader._process_report_request = MethodType(  # pyright: ignore[reportPrivateUsage]
        fake_process,
        reader,
    )

    with pytest.raises(AnonymizationArtifactError, match="structurally invalid"):
        reader.process_report(request)
