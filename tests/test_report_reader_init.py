from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.setup.private_settings import DEFAULT_SETTINGS


def _build_report_reader_without_heavy_init(*args, **kwargs) -> ReportReader:
    """
    Construct ReportReader while patching expensive/external initializers.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch("lx_anonymizer.report_reader.Faker", return_value=MagicMock())
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.gender_detector.Detector",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.PatientDataExtractor",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.ExaminerDataExtractor",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.EndoscopeDataExtractor",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.ExaminationDataExtractor",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.SensitiveRegionCropper",
                return_value=MagicMock(),
            )
        )
        stack.enter_context(
            patch("lx_anonymizer.report_reader.Anonymizer", return_value=MagicMock())
        )
        # Force graceful fallback path and avoid external process/model startup.
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.ensure_ollama",
                side_effect=RuntimeError("test"),
            )
        )
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.OllamaOptimizedExtractor",
                return_value=MagicMock(),
            )
        )
        return ReportReader(*args, **kwargs)


def test_report_reader_init_uses_passed_flags() -> None:
    custom_flags = {"x": True}

    reader = _build_report_reader_without_heavy_init(flags=custom_flags)

    assert reader.flags["x"] is True
    # Missing required parser/anonymizer keys are filled from defaults.
    assert (
        reader.flags["patient_info_line"]
        == DEFAULT_SETTINGS["flags"]["patient_info_line"]
    )
    assert reader.flags["cut_off_below"] == DEFAULT_SETTINGS["flags"]["cut_off_below"]


def test_report_reader_init_uses_default_flags() -> None:
    reader = _build_report_reader_without_heavy_init()

    assert reader.flags == DEFAULT_SETTINGS["flags"]
    assert isinstance(reader.flags, dict)
    assert "patient_info_line" in reader.flags
    assert "cut_off_above" in reader.flags


def test_report_reader_init_no_nameerror_regression() -> None:
    try:
        _build_report_reader_without_heavy_init()
    except NameError as exc:  # regression guard for invalid flags expression
        pytest.fail(f"ReportReader.__init__ raised unexpected NameError: {exc}")


def test_report_reader_init_merges_partial_flags_and_normalizes_cutoff_types() -> None:
    reader = _build_report_reader_without_heavy_init(
        flags={
            "patient_info_line": "Patientenzeile:",
            "cut_off_below": "----",
        }
    )

    assert reader.flags["patient_info_line"] == "Patientenzeile:"
    # str input is normalized to list[str]
    assert reader.flags["cut_off_below"] == ["----"]
    # Unspecified keys are still available from defaults
    assert reader.flags["cut_off_above"] == DEFAULT_SETTINGS["flags"]["cut_off_above"]
