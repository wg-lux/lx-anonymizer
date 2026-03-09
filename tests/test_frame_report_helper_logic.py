from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


def _frame_cleaner_stub() -> FrameCleaner:
    fc = FrameCleaner.__new__(FrameCleaner)
    fc.use_llm = False
    fc.ollama_extractor = None
    fc.patient_data_extractor = None
    fc.frame_metadata_extractor = SimpleNamespace(
        extract_metadata_from_frame_text=Mock(return_value={}),
    )
    fc.sensitive_meta = SensitiveMeta()
    return fc


def _report_reader_stub() -> ReportReader:
    rr = ReportReader.__new__(ReportReader)
    rr.flags = {
        "endoscope_info_line": "scope",
        "cut_off_below": [],
        "cut_off_above": [],
    }
    rr.text_date_format = "%d.%m.%Y"
    rr.locale = "de_DE"
    rr.employee_first_names = []
    rr.employee_last_names = []
    rr.sensitive_meta = SensitiveMeta()
    rr.patient_extractor = Mock(return_value=PatientExtractorStub.blank())
    rr.examiner_extractor = SimpleNamespace(extract_examiner_info=Mock(return_value={}))
    rr.examination_extractor = SimpleNamespace(
        extract_examination_info=Mock(return_value={})
    )
    rr.endoscope_extractor = SimpleNamespace(extract_endoscope_info=Mock(return_value={}))
    rr.ollama_available = False
    rr.ollama_extractor = None
    return rr


class PatientExtractorStub:
    @staticmethod
    def blank():
        return {
            "patient_first_name": None,
            "patient_last_name": None,
            "patient_dob": None,
            "casenumber": None,
            "patient_gender_name": None,
        }


def test_frame_cleaner_build_representative_text_from_meta_compacts_fields():
    meta = {
        "patient_first_name": " Max ",
        "patient_last_name": "Muster",
        "casenumber": "E 123",
        "patient_dob": "1990-01-02",
        "endoscope_sn": "SN-9",
        "empty": None,
    }
    text = FrameCleaner._build_representative_text_from_meta(meta)
    assert text.startswith("Max Muster")
    assert "Case: E 123" in text
    assert "DOB: 1990-01-02" in text
    assert "SN: SN-9" in text


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({}, False),
        ({"patient_first_name": " "}, False),
        ({"casenumber": "E1"}, True),
        ({"patient_dob": None, "examiner_last_name": "DrX"}, True),
        ("not-a-dict", False),
    ],
)
def test_frame_cleaner_metadata_has_signal(meta, expected):
    assert FrameCleaner._metadata_has_signal(meta) is expected


def test_frame_cleaner_validate_roi_accepts_alias_keys_and_rejects_bad_values():
    fc = _frame_cleaner_stub()
    assert (
        fc._validate_roi(
            {
                "image_width": 1920,
                "image_height": 1080,
                "endoscope_image_x": 10,
                "endoscope_image_y": 5,
                "endoscope_image_width": 100,
                "endoscope_image_height": 50,
            }
        )
        is True
    )
    assert fc._validate_roi({"x": -1, "y": 0, "width": 10, "height": 10}) is False
    assert fc._validate_roi({"x": 0, "y": 0, "width": 0, "height": 10}) is False
    assert fc._validate_roi({"x": 0, "y": 0, "width": 6000, "height": 10}) is False


def test_unified_metadata_extract_prefers_llm_when_signal_present():
    fc = _frame_cleaner_stub()
    fc.use_llm = True
    fc.ollama_extractor = SimpleNamespace(
        extract_metadata=Mock(return_value={"patient_last_name": "LLM"})
    )
    fc.patient_data_extractor = Mock(return_value={"patient_last_name": "spacy"})
    fc.frame_metadata_extractor.extract_metadata_from_frame_text.return_value = {
        "patient_last_name": "regex"
    }

    meta = fc._unified_metadata_extract("text")

    assert meta["patient_last_name"] == "LLM"
    fc.patient_data_extractor.assert_not_called()
    fc.frame_metadata_extractor.extract_metadata_from_frame_text.assert_not_called()


def test_unified_metadata_extract_falls_back_when_llm_has_no_signal():
    fc = _frame_cleaner_stub()
    fc.use_llm = True
    fc.ollama_extractor = SimpleNamespace(extract_metadata=Mock(return_value={}))
    fc.patient_data_extractor = Mock(return_value={"patient_first_name": "SpaCy"})
    fc.frame_metadata_extractor.extract_metadata_from_frame_text.return_value = {
        "patient_first_name": "Regex"
    }

    meta = fc._unified_metadata_extract("text")

    assert meta["patient_first_name"] == "SpaCy"
    fc.patient_data_extractor.assert_called_once()
    fc.frame_metadata_extractor.extract_metadata_from_frame_text.assert_not_called()


def test_unified_metadata_extract_reaches_regex_fallback_when_extractors_fail():
    fc = _frame_cleaner_stub()
    fc.use_llm = False
    fc.patient_data_extractor = Mock(return_value={})
    fc.frame_metadata_extractor.extract_metadata_from_frame_text.return_value = {
        "patient_last_name": "Regex"
    }

    meta = fc._unified_metadata_extract("text")

    assert meta["patient_last_name"] == "Regex"
    fc.frame_metadata_extractor.extract_metadata_from_frame_text.assert_called_once_with(
        "text"
    )


def test_report_reader_resolve_flags_rejects_non_mapping():
    with pytest.raises(TypeError):
        ReportReader._resolve_flags(flags=["not", "a", "mapping"])


def test_report_reader_pdf_hash_is_deterministic():
    rr = _report_reader_stub()
    payload = b"pdf-bytes"
    assert rr.pdf_hash(payload) == rr.pdf_hash(payload)
    assert rr.pdf_hash(payload) != rr.pdf_hash(b"other")


def test_report_reader_anonymize_report_passes_config():
    rr = _report_reader_stub()
    with patch("lx_anonymizer.report_reader.anonymize_text", return_value="anon") as mock_anon:
        result = rr.anonymize_report("raw text", {"patient_first_name": "Max"})

    assert result == "anon"
    kwargs = mock_anon.call_args.kwargs
    assert kwargs["text"] == "raw text"
    assert kwargs["report_meta"]["patient_first_name"] == "Max"
    assert kwargs["lower_cut_off_flags"] == []
    assert kwargs["upper_cut_off_flags"] == []


def test_report_reader_shared_ollama_wrapper_success_and_failure_paths():
    rr = _report_reader_stub()
    rr.ollama_available = True
    rr.ollama_extractor = SimpleNamespace(
        extract_metadata=Mock(return_value={"patient_first_name": "LLM"})
    )

    ok = rr._extract_report_meta_via_ollama("txt", "DeepSeek")
    assert ok["patient_first_name"] == "LLM"

    rr.ollama_extractor.extract_metadata.return_value = None
    fail = rr._extract_report_meta_via_ollama("txt", "DeepSeek")
    assert fail == {}


def test_extract_report_meta_uses_line_fallback_parses_dob_and_enriches_fields():
    rr = _report_reader_stub()
    rr.patient_extractor.side_effect = [
        PatientExtractorStub.blank(),  # full-text attempt fails
        {
            "patient_first_name": "Max",
            "patient_last_name": "Muster",
            "patient_dob": "02.01.1990",
            "casenumber": "E 123",
            "patient_gender_name": "male",
        },
    ]
    rr.examiner_extractor.extract_examiner_info.return_value = {
        "examiner_last_name": "Arzt"
    }
    rr.examination_extractor.extract_examination_info.return_value = {
        "examination_date": "2024-01-01"
    }
    rr.endoscope_extractor.extract_endoscope_info.return_value = {"endoscope_sn": "SN1"}

    text = "\n".join(
        [
            "Intro line",
            "Patient: Max Muster geb. 02.01.1990",
            "Unters. Arzt: Dr Arzt",
            "U-Datum: 01.01.2024",
            "Scope: Olympus SN1",
        ]
    )

    with (
        patch("lx_anonymizer.report_reader.PatientDataExtractor", PatientExtractorStub),
        patch("lx_anonymizer.report_reader.dateparser.parse", return_value=datetime(1990, 1, 2)),
    ):
        meta = rr.extract_report_meta(text, pdf_path=None)

    assert meta["patient_first_name"] == "Max"
    assert meta["patient_last_name"] == "Muster"
    assert str(meta["patient_dob"]) == "1990-01-02"
    assert meta["casenumber"] == "E 123"
    assert meta["examiner_last_name"] == "Arzt"
    assert meta["examination_date"] == "2024-01-01"
    assert meta["endoscope_sn"] == "SN1"
    assert meta["pdf_hash"] is None
