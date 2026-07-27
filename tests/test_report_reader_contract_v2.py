from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType
from uuid import uuid4

import pytest
from lx_dtypes.models.meta.ReportMeta import ReportProcessRequest, ReportProcessResult

from lx_anonymizer.report_contracts import (
    ArtifactAlreadyExistsError,
    ReportAnonymizationRequestV2,
    SourceIdentityMismatchError,
)
from lx_anonymizer.report_reader import ReportReader


def _request(
    *,
    source: Path,
    output_directory: Path,
) -> ReportAnonymizationRequestV2:
    payload = source.read_bytes()
    return ReportAnonymizationRequestV2(
        attempt_id=uuid4(),
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_size_bytes=len(payload),
        output_directory=output_directory,
    )


def _reader_with_fake_pipeline() -> ReportReader:
    reader = object.__new__(ReportReader)

    def fake_process(
        self: ReportReader,
        request: ReportProcessRequest,
    ) -> ReportProcessResult:
        output_path = request.anonymized_pdf_output_path
        assert isinstance(output_path, Path)
        output_path.write_bytes(b"%PDF-1.4\nanonymized\n%%EOF\n")
        return ReportProcessResult(
            text="original",
            anonymized_text="anonymized",
            report_meta={"patient_first_name": "ANON"},
            anonymized_pdf_path=output_path,
        )

    reader._process_report_request = MethodType(  # pyright: ignore[reportPrivateUsage]
        fake_process,
        reader,
    )
    return reader


def test_process_report_v2_publishes_attempt_local_validated_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)

    result = _reader_with_fake_pipeline().process_report_v2(request)

    assert result.contract_version == "report_anonymization_v2"
    assert result.attempt_id == request.attempt_id
    assert result.source_sha256 == request.source_sha256
    assert result.artifact_path.parent == output_directory
    assert result.artifact_path.read_bytes().startswith(b"%PDF-")
    assert (
        result.artifact_sha256
        == hashlib.sha256(result.artifact_path.read_bytes()).hexdigest()
    )
    assert not (output_directory / f".{request.attempt_id}.part.pdf").exists()


def test_process_report_v2_rejects_source_identity_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)
    source.write_bytes(b"%PDF-1.4\nchanged\n%%EOF\n")

    with pytest.raises(SourceIdentityMismatchError):
        _reader_with_fake_pipeline().process_report_v2(request)


def test_process_report_v2_refuses_existing_attempt_output(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output_directory = tmp_path / "attempt"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    output_directory.mkdir()
    request = _request(source=source, output_directory=output_directory)
    existing = output_directory / f"{request.attempt_id}.pdf"
    existing.write_bytes(b"existing")

    with pytest.raises(ArtifactAlreadyExistsError):
        _reader_with_fake_pipeline().process_report_v2(request)

    assert existing.read_bytes() == b"existing"


def test_report_v2_request_rejects_output_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    real_output = tmp_path / "real-output"
    linked_output = tmp_path / "linked-output"
    source.write_bytes(b"%PDF-1.4\nsource\n%%EOF\n")
    real_output.mkdir()
    linked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink"):
        _request(source=source, output_directory=linked_output)
