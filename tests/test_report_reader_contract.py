from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType, TracebackType
from typing import Protocol, Self, cast
from uuid import uuid4

import pymupdf  # type: ignore[import-untyped]
import pytest
from lx_dtypes.models.contracts.report_anonymization import (
    ReportAnonymizationRequest as SharedReportAnonymizationRequest,
)
from lx_dtypes.models.meta.ReportMeta import ReportProcessRequest, ReportProcessResult

from lx_anonymizer.report_contracts import (
    AnonymizationArtifactError,
    ArtifactAlreadyExistsError,
    OperationDeadlineExceededError,
    ReportAnonymizationRequest,
    SourceIdentityMismatchError,
)
from lx_anonymizer.report_reader import ReportReader


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
