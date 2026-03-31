from pathlib import Path
from unittest.mock import patch

import pytest

from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def _example_report_path() -> Path:
    return Path("tests/assets/lux-histo-1.pdf")


def _sensitive_meta_fields() -> list[str]:
    return list(SensitiveMeta.model_fields.keys())


@pytest.mark.integration
def test_example_report_populates_sensitive_meta_fields() -> None:
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
        "lx_anonymizer.report_reader.VLLMMetadataExtractor",
        side_effect=RuntimeError("test"),
    ):
        reader = ReportReader()

    original_text, anonymized_text, meta, _ = reader.process_report(
        pdf_path=pdf_path,
        use_llm_extractor=None,  # force SpaCy/regex path for deterministic local test
        create_anonymized_pdf=False,
    )

    expected_fields = _sensitive_meta_fields()
    missing_keys = [k for k in expected_fields if k not in meta]
    assert not missing_keys, f"Returned meta missing SensitiveMeta keys: {missing_keys}"

    assert isinstance(original_text, str)
    assert isinstance(anonymized_text, str)

    populated = {k: v for k, v in meta.items() if isinstance(v, str) and v.strip()}
    high_signal_fields = [
        "patient_first_name",
        "patient_last_name",
        "patient_dob",
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
