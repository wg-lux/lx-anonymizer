from collections.abc import Mapping
from typing import cast
from unittest.mock import Mock, patch
import pytest
from lx_dtypes.models.meta.ReportMeta import ReportReaderFlags
from lx_dtypes.models.meta.VideoMeta import FrameCleanerAccumulatedMeta
from lx_anonymizer.frame_cleaner import FrameCleaner
from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor
from lx_anonymizer.ner.frame_metadata_extractor import FrameMetadataExtractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor
from lx_anonymizer.report_reader import ReportReader
from lx_anonymizer.report_reader_extraction import (
    LLMExtractorProtocol,
    _EndoscopeExtractor,  # pyright: ignore[reportPrivateUsage]
    _ExaminationExtractor,  # pyright: ignore[reportPrivateUsage]
    _ExaminerExtractor,  # pyright: ignore[reportPrivateUsage]
    _PatientExtractor,  # pyright: ignore[reportPrivateUsage]
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


class _FrameMetadataExtractorStub:
    def __init__(self) -> None:
        self.extract_metadata_from_frame_text = Mock(return_value={})


class _ExaminerExtractorStub:
    def __init__(self) -> None:
        self.extract_examiner_info = Mock(return_value={})


class _ExaminationExtractorStub:
    def __init__(self) -> None:
        self.extract_examination_info = Mock(return_value={})


class _EndoscopeExtractorStub:
    def __init__(self) -> None:
        self.extract_endoscope_info = Mock(return_value={})


class _ReportLlmExtractorStub:
    current_model: dict[str, object] | None = {"name": "test"}

    def __init__(self) -> None:
        self.extract_metadata = Mock(return_value=SensitiveMeta(first_name="LLM"))


def _frame_cleaner_stub() -> FrameCleaner:
    fc = FrameCleaner.__new__(FrameCleaner)
    fc.use_llm = False
    fc.llm_extractor = None
    fc.patient_data_extractor = cast(PatientDataExtractor, Mock(return_value={}))
    fc.frame_metadata_extractor = cast(
        FrameMetadataExtractor,
        _FrameMetadataExtractorStub(),
    )
    fc.sensitive_meta = SensitiveMeta()
    return fc


def _report_reader_stub() -> ReportReader:
    rr = ReportReader.__new__(ReportReader)
    rr.flags = ReportReaderFlags(
        endoscope_info_line="scope",
        cut_off_below=[],
        cut_off_above=[],
    )
    rr.text_date_format = "%d.%m.%Y"
    rr.locale = "de_DE"
    rr.employee_first_names = []
    rr.employee_last_names = []
    rr.sensitive_meta = SensitiveMeta()
    rr.patient_extractor = cast(
        _PatientExtractor,
        Mock(return_value=PatientExtractorStub.blank()),
    )
    rr.examiner_extractor = cast(_ExaminerExtractor, _ExaminerExtractorStub())
    rr.examination_extractor = cast(
        _ExaminationExtractor,
        _ExaminationExtractorStub(),
    )
    rr.endoscope_extractor = cast(_EndoscopeExtractor, _EndoscopeExtractorStub())
    rr.llm_available = False
    rr.llm_extractor = None
    return rr


class PatientExtractorStub:
    @staticmethod
    def blank():
        return {
            "first_name": None,
            "last_name": None,
            "dob": None,
            "casenumber": None,
            "gender": None,
        }


def test_frame_cleaner_build_representative_text_from_meta_compacts_fields():
    meta = FrameCleanerAccumulatedMeta(
        file_path="video.mp4",
        first_name=" Max ",
        last_name="Muster",
        casenumber="E 123",
        dob="1990-01-02",
        endoscope_sn="SN-9",
    )
    text = FrameCleaner._build_representative_text_from_meta(meta)  # pyright: ignore[reportPrivateUsage]
    assert text.startswith("Max Muster")
    assert "Case: E 123" in text
    assert "DOB: 1990-01-02" in text
    assert "SN: SN-9" in text


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({}, False),
        ({"first_name": " "}, False),
        ({"casenumber": "E1"}, True),
        ({"dob": None, "examiner_last_name": "DrX"}, True),
        ("not-a-dict", False),
    ],
)
def test_frame_cleaner_metadata_has_signal(meta: object, expected: bool) -> None:
    assert (
        FrameCleaner._metadata_has_signal(cast(Mapping[str, object], meta)) is expected
    )  # pyright: ignore[reportPrivateUsage]


def test_frame_cleaner_validate_roi_accepts_alias_keys_and_rejects_bad_values():
    fc = _frame_cleaner_stub()
    assert (
        fc._validate_roi(  # pyright: ignore[reportPrivateUsage]
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
    assert fc._validate_roi({"x": -1, "y": 0, "width": 10, "height": 10}) is False  # pyright: ignore[reportPrivateUsage]
    assert fc._validate_roi({"x": 0, "y": 0, "width": 0, "height": 10}) is False  # pyright: ignore[reportPrivateUsage]
    assert fc._validate_roi({"x": 0, "y": 0, "width": 6000, "height": 10}) is False  # pyright: ignore[reportPrivateUsage]


def test_unified_metadata_extract_falls_back_when_llm_has_no_signal():
    fc = _frame_cleaner_stub()
    fc.use_llm = True
    llm_extractor = Mock(spec=LLMMetadataExtractor)
    llm_extractor.extract_metadata.return_value = None
    fc.llm_extractor = cast(LLMMetadataExtractor, llm_extractor)
    patient_data_extractor = Mock(return_value={"first_name": "SpaCy"})
    fc.patient_data_extractor = patient_data_extractor
    frame_metadata_extractor = cast(
        _FrameMetadataExtractorStub,
        fc.frame_metadata_extractor,
    )
    frame_metadata_extractor.extract_metadata_from_frame_text.return_value = {
        "first_name": "Regex"
    }

    meta = fc._unified_metadata_extract("text")  # pyright: ignore[reportPrivateUsage]

    assert meta["first_name"] == "SpaCy"
    patient_data_extractor.assert_called_once()
    frame_metadata_extractor.extract_metadata_from_frame_text.assert_not_called()


def test_unified_metadata_extract_reaches_regex_fallback_when_extractors_fail():
    fc = _frame_cleaner_stub()
    fc.use_llm = False
    patient_data_extractor = Mock(return_value={})
    fc.patient_data_extractor = patient_data_extractor
    frame_metadata_extractor = cast(
        _FrameMetadataExtractorStub,
        fc.frame_metadata_extractor,
    )
    frame_metadata_extractor.extract_metadata_from_frame_text.return_value = {
        "last_name": "Regex"
    }

    meta = fc._unified_metadata_extract("text")  # pyright: ignore[reportPrivateUsage]

    assert meta["last_name"] == "Regex"
    frame_metadata_extractor.extract_metadata_from_frame_text.assert_called_once_with(
        "text"
    )


def test_report_reader_resolve_flags_rejects_non_mapping():
    with pytest.raises(TypeError):
        ReportReader._resolve_flags(  # pyright: ignore[reportPrivateUsage]
            flags=cast(Mapping[str, object], ["not", "a", "mapping"])
        )


def test_report_reader_pdf_hash_is_deterministic():
    rr = _report_reader_stub()
    payload = b"pdf-bytes"
    assert rr.pdf_hash(payload) == rr.pdf_hash(payload)
    assert rr.pdf_hash(payload) != rr.pdf_hash(b"other")


def test_report_reader_anonymize_report_passes_config():
    rr = _report_reader_stub()
    with patch(
        "lx_anonymizer.anonymization.text_anonymizer.anonymize_text",
        return_value="anon",
    ) as mock_anon:
        result = rr.anonymize_report("raw text", {"first_name": "Max"})

    assert result == "anon"
    kwargs = mock_anon.call_args.kwargs
    assert kwargs["text"] == "raw text"
    assert kwargs["report_meta"]["first_name"] == "Max"
    assert kwargs["text_date_format"] == "%d.%m.%Y"
    assert kwargs["lower_cut_off_flags"] == []
    assert kwargs["upper_cut_off_flags"] == []


def test_report_reader_shared_llm_wrapper_success_and_failure_paths():
    rr = _report_reader_stub()
    rr.llm_available = True
    llm_extractor = _ReportLlmExtractorStub()
    rr.llm_extractor = cast(LLMExtractorProtocol, llm_extractor)

    ok = rr._extract_report_meta_via_llm("txt", "DeepSeek")  # pyright: ignore[reportPrivateUsage]
    assert ok["first_name"] == "LLM"

    llm_extractor.extract_metadata.return_value = None
    fail = rr._extract_report_meta_via_llm("txt", "DeepSeek")  # pyright: ignore[reportPrivateUsage]
    assert fail == {}


def test_extract_report_meta_uses_line_fallback_parses_dob_and_enriches_fields():
    rr = _report_reader_stub()
    patient_extractor = Mock()
    patient_extractor.side_effect = [
        PatientExtractorStub.blank(),  # full-text attempt fails
        {
            "first_name": "Max",
            "last_name": "Muster",
            "dob": "02.01.1990",
            "casenumber": "E 123",
            "gender": "male",
        },
    ]
    rr.patient_extractor = cast(_PatientExtractor, patient_extractor)
    examiner_extractor = cast(_ExaminerExtractorStub, rr.examiner_extractor)
    examination_extractor = cast(_ExaminationExtractorStub, rr.examination_extractor)
    endoscope_extractor = cast(_EndoscopeExtractorStub, rr.endoscope_extractor)
    examiner_extractor.extract_examiner_info.return_value = {
        "examiner_last_name": "Arzt"
    }
    examination_extractor.extract_examination_info.return_value = {
        "examination_date": "2024-01-01"
    }
    endoscope_extractor.extract_endoscope_info.return_value = {"endoscope_sn": "SN1"}

    text = "\n".join(
        [
            "Intro line",
            "Patient: Max Muster geb. 02.01.1990",
            "Unters. Arzt: Dr Arzt",
            "U-Datum: 01.01.2024",
            "Scope: Olympus SN1",
        ]
    )

    with patch(
        "lx_anonymizer.report_reader.PatientDataExtractor", PatientExtractorStub
    ):
        meta = rr.extract_report_meta(text, pdf_path=None)

    assert meta["first_name"] == "Max"
    assert meta["last_name"] == "Muster"
    assert str(meta["dob"]) == "1990-01-02"
    assert meta["casenumber"] == "E 123"
    assert meta["examiner_last_name"] == "Arzt"
    assert meta["examination_date"] == "2024-01-01"
    assert meta["endoscope_sn"] == "SN1"
    assert meta["pdf_hash"] is None
