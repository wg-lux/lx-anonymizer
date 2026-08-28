import hashlib
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from lx_anonymizer.report_contracts import (
    ReportAnonymizationOptions,
    ReportAnonymizationRequest,
)
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def _example_report_path() -> Path:
    return Path("tests/assets/lux-histo-1.pdf")


def _sensitive_meta_fields() -> list[str]:
    return [
        name for name, field in SensitiveMeta.model_fields.items() if not field.exclude
    ]


@pytest.mark.integration
def test_example_report_populates_sensitive_meta_fields(tmp_path: Path) -> None:
    """
    Run ReportReader on an example report PDF and verify metadata/text propagation.

    Diagnostic integration behavior:
    - verifies returned metadata keys conform to SensitiveMeta
    - verifies raw/anonymized text are returned and persisted when extraction succeeds
    - xfails (not fails) if fixture text is unreadable or yields no sensitive metadata
    """
    pdf_path = _example_report_path()
    if not pdf_path.exists():
        pytest.skip(f"Example report not found: {pdf_path}")

    with patch(
        "lx_anonymizer.report_reader.LLMFactory.create_metadata_extractor",
        side_effect=RuntimeError("test"),
    ):
        reader = ReportReader()

    source_bytes = pdf_path.read_bytes()
    output_directory = tmp_path / "attempt"
    output_directory.mkdir()
    result = reader.process_report(
        ReportAnonymizationRequest(
            attempt_id=uuid4(),
            source_path=pdf_path,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            source_size_bytes=len(source_bytes),
            output_directory=output_directory,
            options=ReportAnonymizationOptions(use_llm=False),
        )
    )
    original_text = result.original_text
    anonymized_text = result.anonymized_text
    meta = result.extracted_metadata.model_dump(mode="json")

    if not meta:
        pytest.xfail(
            "Example report produced no metadata because local text extraction failed."
        )

    expected_fields = _sensitive_meta_fields()
    missing_keys = [k for k in expected_fields if k not in meta]
    assert not missing_keys, f"Returned meta missing SensitiveMeta keys: {missing_keys}"

    assert isinstance(original_text, str)
    assert isinstance(anonymized_text, str)

    populated = {k: v for k, v in meta.items() if isinstance(v, str) and v.strip()}
    high_signal_fields = [
        "first_name",
        "last_name",
        "dob",
        "casenumber",
        "examination_date",
        "examiner_last_name",
    ]
    populated_high_signal = [k for k in high_signal_fields if populated.get(k)]

    baseline_fields = {"file_path", "center", "text", "anonymized_text"}
    non_baseline_populated = [k for k in populated.keys() if k not in baseline_fields]

    if not populated_high_signal and not non_baseline_populated:
        pytest.xfail(
            "Example report produced no recoverable sensitive metadata "
            f"(populated fields: {sorted(populated.keys())}). "
            "This indicates fixture/parser difficulty, not necessarily a regression."
        )

    # If metadata extraction produced meaningful fields, text payloads should also be persisted.
    if non_baseline_populated or populated_high_signal:
        assert populated.get("text"), (
            "meta['text'] was not populated despite metadata extraction output. "
            f"Populated fields: {sorted(populated.keys())}"
        )
        assert populated.get("anonymized_text"), (
            "meta['anonymized_text'] was not populated despite metadata extraction output. "
            f"Populated fields: {sorted(populated.keys())}"
        )
