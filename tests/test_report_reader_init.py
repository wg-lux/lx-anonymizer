from contextlib import ExitStack
from typing import Mapping, Sequence, cast
from unittest.mock import MagicMock, patch

import pytest

from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.setup.private_settings import DEFAULT_SETTINGS


def _build_report_reader_without_heavy_init(
    report_root_path: str | None = None,
    locale: str | None = None,
    employee_first_names: Sequence[str] | None = None,
    employee_last_names: Sequence[str] | None = None,
    flags: Mapping[str, object] | None = None,
    text_date_format: str | None = None,
) -> ReportReader:
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
        stack.enter_context(
            patch(
                "lx_anonymizer.report_reader.LLMFactory.create_metadata_extractor",
                return_value=MagicMock(),
            )
        )
        return ReportReader(
            report_root_path=report_root_path,
            locale=locale,
            employee_first_names=employee_first_names,
            employee_last_names=employee_last_names,
            flags=flags,
            text_date_format=text_date_format,
        )


def test_report_reader_init_uses_passed_flags() -> None:
    default_flags = cast(dict[str, object], DEFAULT_SETTINGS["flags"])
    custom_flags = {"x": True}

    reader = _build_report_reader_without_heavy_init(flags=custom_flags)

    assert reader.flags.model_extra is not None
    assert reader.flags.model_extra["x"] is True
    # Missing required parser/anonymizer keys are filled from defaults.
    assert reader.flags.patient_info_line == default_flags["patient_info_line"]
    assert reader.flags.cut_off_below == default_flags["cut_off_below"]


def test_report_reader_init_uses_default_flags() -> None:
    default_flags = cast(dict[str, object], DEFAULT_SETTINGS["flags"])
    reader = _build_report_reader_without_heavy_init()

    assert reader.flags.model_dump() == default_flags
    assert reader.flags.patient_info_line is not None
    assert reader.flags.cut_off_above


def test_report_reader_init_no_nameerror_regression() -> None:
    try:
        _build_report_reader_without_heavy_init()
    except NameError as exc:  # regression guard for invalid flags expression
        pytest.fail(f"ReportReader.__init__ raised unexpected NameError: {exc}")


def test_report_reader_init_merges_partial_flags_and_normalizes_cutoff_types() -> None:
    default_flags = cast(dict[str, object], DEFAULT_SETTINGS["flags"])
    reader = _build_report_reader_without_heavy_init(
        flags={
            "patient_info_line": "Patientenzeile:",
            "cut_off_below": "----",
        }
    )

    assert reader.flags.patient_info_line == "Patientenzeile:"
    # str input is normalized to list[str]
    assert reader.flags.cut_off_below == ["----"]
    # Unspecified keys are still available from defaults
    assert reader.flags.cut_off_above == default_flags["cut_off_above"]
