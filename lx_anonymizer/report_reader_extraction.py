from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import pdfplumber
from PIL import Image
from lx_dtypes.models.meta.ReportMeta import (
    ReportCropInfo,
    ReportCroppingRequest,
    ReportEndoscopeInfo,
    ReportExaminerInfo,
    ReportExaminationInfo,
    ReportMeta,
    ReportReaderFlags,
    ReportPatientInfo,
    ReportProcessRequest,
)
from lx_dtypes.models import SensitiveMeta

from lx_anonymizer.anonymization import text_anonymizer
from lx_anonymizer.config import settings
from lx_anonymizer.image_processing.pdf_operations import convert_pdf_to_images
from lx_anonymizer.ner.spacy_extractor import (
    EndoscopeInfo,
    ExaminationInfo,
    ExaminerInfo,
    PatientInfo,
)
from lx_anonymizer.ner.spacy_ner_fallback import extract_patient_info_from_text
from lx_anonymizer.ocr.ocr import tesseract_full_image_ocr
from lx_anonymizer.region_processing.box_operations import OcrResult
from lx_anonymizer.regex_patterns import (
    EXAMINATION_LINE_RE,
    EXAMINER_LINE_RE,
    PATIENT_LINE_RE,
)
from lx_anonymizer.setup.custom_logger import logger


class _PatientExtractor(Protocol):
    def __call__(self, text: str) -> PatientInfo: ...


class _ExaminerExtractor(Protocol):
    def extract_examiner_info(self, text: str) -> ExaminerInfo | None: ...


class _ExaminationExtractor(Protocol):
    def extract_examination_info(self, text: str) -> ExaminationInfo | None: ...


class _EndoscopeExtractor(Protocol):
    def extract_endoscope_info(self, text: str) -> EndoscopeInfo | None: ...


class _SensitiveCropper(Protocol):
    def crop_sensitive_regions(
        self, pdf_path: str, output_dir: str, page_numbers: Sequence[int] = ()
    ) -> dict[str, list[str]]: ...

    def create_anonymized_pdf_with_crops(
        self, pdf_path: str, crop_output_dir: str, anonymized_pdf_path: str
    ) -> None: ...

    def visualize_sensitive_regions(
        self, image: "Image.Image", word_boxes: Sequence[OcrResult], output_path: str
    ) -> None: ...


class _Anonymizer(Protocol):
    last_redaction_summary: dict[str, object] | None

    def create_anonymized_pdf(
        self,
        pdf_path: str,
        output_path: str | None = None,
        report_meta: Mapping[str, object] | None = None,
    ) -> str | None: ...


class LLMExtractorProtocol(Protocol):
    @property
    def current_model(self) -> object | None: ...

    def extract_metadata(self, text: str) -> SensitiveMeta | None: ...


class ReportReaderExtractionMixin:
    flags: ReportReaderFlags
    text_date_format: str
    locale: str
    employee_first_names: Sequence[str]
    employee_last_names: Sequence[str]
    sensitive_meta: SensitiveMeta
    patient_extractor: _PatientExtractor
    examiner_extractor: _ExaminerExtractor
    examination_extractor: _ExaminationExtractor
    endoscope_extractor: _EndoscopeExtractor
    sensitive_cropper: _SensitiveCropper
    anonymizer: _Anonymizer
    llm_available: bool
    llm_extractor: LLMExtractorProtocol | None

    def read_pdf(self, pdf_path: str | os.PathLike[str] | Path | None) -> str:
        """Read pdf file using pdfplumber and return the raw text content."""
        if pdf_path is None:
            logger.error("PDF path is None, cannot read PDF")
            return ""
        pdf_file = Path(pdf_path)

        # Disable verbose pdfminer logging
        logging.getLogger("pdfminer").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.psparser").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfdocument").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfinterp").setLevel(logging.WARNING)
        logging.getLogger("pdfminer.pdfpage").setLevel(logging.WARNING)

        try:
            with pdfplumber.open(str(pdf_file)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as exc:
            logger.error("Error reading PDF %s: %s", pdf_file, exc)
            return ""

    def extract_report_meta(
        self, text: str, pdf_path: str | os.PathLike[str] | Path | None
    ) -> dict[str, object]:
        """Extract metadata from report text using the spacy extractor classes."""
        logger.debug(f"Full text extracted from PDF: {text[:500]}...")
        lines = text.split("\n") if text else []
        patient_info = self._extract_patient_info(text, lines)
        final_patient_info = ReportPatientInfo.model_validate(patient_info).model_dump()

        self.sensitive_meta.safe_update(final_patient_info)
        report_meta = self.sensitive_meta.to_dict()
        report_meta = self._extract_line_report_meta(lines, report_meta)

        self.sensitive_meta.safe_update(report_meta)
        report_meta = self.sensitive_meta.to_dict()

        pdf_hash = self.pdf_hash_file(pdf_path) if pdf_path else None
        report_meta = self._merge_report_meta(
            report_meta, self._partial_report_meta({"pdf_hash": pdf_hash})
        )

        return self._validated_report_meta(report_meta)

    def _extract_patient_info(
        self, text: str, lines: list[str]
    ) -> Mapping[str, object]:
        patient_info = self.patient_extractor(text) if text else None
        if patient_info:
            logger.debug("Patient extractor result on full text: %s", patient_info)
        else:
            logger.debug("Skipping extraction on empty text.")

        if self._patient_info_has_signal(patient_info):
            assert patient_info is not None
            return patient_info

        line_info = self._extract_patient_info_from_lines(lines)
        if self._patient_info_has_signal(line_info):
            assert line_info is not None
            return line_info

        fallback_info = self._extract_patient_info_with_regex(text)
        if self._patient_info_has_signal(fallback_info):
            assert fallback_info is not None
            return fallback_info
        return {
            "first_name": None,
            "last_name": None,
            "dob": None,
            "casenumber": None,
            "gender": None,
        }

    @staticmethod
    def _patient_info_has_signal(
        patient_info: Mapping[str, object] | None,
    ) -> bool:
        if not patient_info:
            return False
        return (
            patient_info.get("first_name") is not None
            or patient_info.get("last_name") is not None
        )

    def _extract_patient_info_from_lines(
        self, lines: list[str]
    ) -> Mapping[str, object] | None:
        logger.debug("Extractor failed on full text, trying line by line.")
        for line in lines:
            if not PATIENT_LINE_RE.search(line):
                continue
            patient_info = self.patient_extractor(line)
            if self._patient_info_has_signal(patient_info):
                return patient_info
        return None

    @staticmethod
    def _extract_patient_info_with_regex(text: str) -> Mapping[str, object] | None:
        if not text:
            return None
        logger.debug("SpaCy extractor failed, using regex fallback extraction")
        fallback_info = extract_patient_info_from_text(text)
        if fallback_info is None:
            return None
        if not (fallback_info["first_name"] or fallback_info["last_name"]):
            return None
        return dict(fallback_info)

    def _extract_line_report_meta(
        self, lines: list[str], report_meta: Mapping[str, object]
    ) -> dict[str, object]:
        meta = dict(report_meta)
        for line in lines:
            meta = self._merge_line_examiner_meta(line, meta)
            meta = self._merge_line_examination_meta(line, meta)
            meta = self._merge_line_endoscope_meta(line, meta)
        return meta

    def _merge_line_examiner_meta(
        self, line: str, report_meta: Mapping[str, object]
    ) -> dict[str, object]:
        if not EXAMINER_LINE_RE.search(line):
            return dict(report_meta)
        examiner_info = self.examiner_extractor.extract_examiner_info(line)
        if not examiner_info:
            return dict(report_meta)
        typed_examiner = ReportExaminerInfo.model_validate(
            {
                "examiner_first_name": examiner_info.get("examiner_first_name"),
                "examiner_last_name": examiner_info.get("examiner_last_name"),
            }
        )
        return self._merge_report_meta(
            report_meta,
            self._partial_report_meta(typed_examiner.model_dump(exclude_none=True)),
        )

    def _merge_line_examination_meta(
        self, line: str, report_meta: Mapping[str, object]
    ) -> dict[str, object]:
        if not EXAMINATION_LINE_RE.search(line):
            return dict(report_meta)
        examination_info = self.examination_extractor.extract_examination_info(line)
        if not examination_info or not examination_info.get("examination_date"):
            return dict(report_meta)

        typed_examination = ReportExaminationInfo.model_validate(
            {
                "examination_date": examination_info.get("examination_date"),
                "examination_time": examination_info.get("examination_time"),
            }
        )
        meta = self._merge_report_meta(
            report_meta,
            self._partial_report_meta(typed_examination.model_dump(exclude_none=True)),
        )
        typed_examiner = ReportExaminerInfo.model_validate(
            {
                "examiner_first_name": examination_info.get("examiner_first_name"),
                "examiner_last_name": examination_info.get("examiner_last_name"),
            }
        )
        return self._merge_report_meta(
            meta,
            self._partial_report_meta(typed_examiner.model_dump(exclude_none=True)),
        )

    def _merge_line_endoscope_meta(
        self, line: str, report_meta: Mapping[str, object]
    ) -> dict[str, object]:
        marker = str(self.flags.endoscope_info_line).strip().lower()
        if not marker or marker not in line.lower():
            return dict(report_meta)
        endoscope_info = self.endoscope_extractor.extract_endoscope_info(line)
        if not endoscope_info:
            return dict(report_meta)
        typed_endoscope = ReportEndoscopeInfo.model_validate(
            {
                "endoscope_type": endoscope_info.get("endoscope_type")
                or endoscope_info.get("model_name"),
                "endoscope_sn": endoscope_info.get("endoscope_sn")
                or endoscope_info.get("serial_number"),
            }
        )
        return self._merge_report_meta(
            report_meta,
            self._partial_report_meta(typed_endoscope.model_dump(exclude_none=True)),
        )

    def extract_report_meta_with_llm(self, text: str) -> dict[str, object]:
        """Extract metadata using the shared structured-output LLM path."""
        return self._extract_report_meta_via_llm(
            text=text,
            extractor_name="default",
            unavailable_log_level="warning",
        )

    def _extract_report_meta_via_llm(
        self,
        text: str,
        extractor_name: str,
        unavailable_log_level: str = "debug",
    ) -> dict[str, object]:
        """Shared wrapper for provider-backed LLM metadata extraction variants."""
        if not self.llm_available or not self.llm_extractor:
            msg = (
                f"LLM provider {settings.LLM_PROVIDER} not available for "
                f"{extractor_name} extraction, returning empty dict."
            )
            log_fn = getattr(logger, unavailable_log_level, logger.debug)
            log_fn(msg)
            return {}

        logger.info(
            "Attempting metadata extraction with %s via provider %s",
            extractor_name,
            settings.LLM_PROVIDER,
        )
        try:
            meta_obj = self.llm_extractor.extract_metadata(text)
            if not meta_obj:
                logger.warning(
                    "%s LLM extraction failed, returning empty dict.",
                    extractor_name,
                )
                return {}

            self.sensitive_meta.safe_update(meta_obj)
            logger.info("%s LLM extraction successful.", extractor_name)
            return self.sensitive_meta.to_dict()
        except Exception as e:
            logger.warning(f"{extractor_name} LLM extraction error: {e}")
            return {}

    def anonymize_report(self, text: str, report_meta: Mapping[str, object]) -> str:
        """Anonymize the report text using the extracted metadata."""
        anonymized_text = text_anonymizer.anonymize_text(
            text=text,
            report_meta=dict(report_meta),
            text_date_format=self.text_date_format,
            lower_cut_off_flags=list(self.flags.cut_off_below),
            upper_cut_off_flags=list(self.flags.cut_off_above),
            locale=self.locale,
            first_names=self.employee_first_names,
            last_names=self.employee_last_names,
            apply_cutoffs=True,
        )
        return anonymized_text

    def pdf_hash(self, pdf_binary: bytes) -> str:
        return hashlib.sha256(pdf_binary).hexdigest()

    def pdf_hash_file(
        self,
        pdf_path: str | os.PathLike[str] | Path | None,
        chunk_size: int = 1024 * 1024,
    ) -> str | None:
        """Calculate PDF SHA-256 hash with chunked IO to avoid loading full files into memory."""
        if not pdf_path:
            return None
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            return None

        try:
            hasher = hashlib.sha256()
            with open(pdf_file, "rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Could not stream-hash PDF {pdf_file}: {e}")
            return None

    def process_report_with_cropping(
        self,
        pdf_path: str | os.PathLike[str] | Path | None = None,
        image_path: str | os.PathLike[str] | Path | None = None,
        use_ensemble: bool = False,
        verbose: bool = True,
        use_llm: bool | None = None,
        text: str | None = None,
        crop_output_dir: str | os.PathLike[str] | Path | None = None,
        crop_sensitive_regions: bool = True,
        anonymization_output_dir: str | os.PathLike[str] | Path | None = None,
    ) -> tuple[str, str, dict[str, object], dict[str, list[str]], Path | None]:
        """Extended version of process_report with optional cropping."""
        pdf_path = Path(pdf_path) if pdf_path is not None else None
        image_path = Path(image_path) if image_path is not None else None
        crop_output_dir = Path(crop_output_dir) if crop_output_dir is not None else None
        anonymization_output_dir = (
            Path(anonymization_output_dir)
            if anonymization_output_dir is not None
            else None
        )
        request = ReportCroppingRequest(
            pdf_path=pdf_path,
            image_path=image_path,
            use_ensemble=use_ensemble,
            verbose=verbose,
            use_llm=use_llm,
            text=text,
            crop_output_dir=crop_output_dir,
            crop_sensitive_regions=crop_sensitive_regions,
            anonymization_output_dir=anonymization_output_dir,
        )
        process_request = request.process_request()
        original_text, anonymized_text, report_meta, _ = self.process_report(
            pdf_path=process_request.pdf_path,
            image_path=process_request.image_path,
            use_ensemble=process_request.use_ensemble,
            verbose=process_request.verbose,
            use_llm=process_request.use_llm,
            text=process_request.text,
            create_anonymized_pdf=process_request.create_anonymized_pdf,
            anonymized_pdf_output_path=process_request.anonymized_pdf_output_path,
        )
        cropped_regions_info, anonymized_pdf_path, report_meta = (
            self._apply_cropping_to_report_meta(request, report_meta)
        )

        report_meta = self._validated_report_meta(report_meta)
        return (
            original_text,
            anonymized_text,
            report_meta,
            cropped_regions_info,
            anonymized_pdf_path,
        )

    def _apply_cropping_to_report_meta(
        self, request: ReportCroppingRequest, report_meta: Mapping[str, object]
    ) -> tuple[dict[str, list[str]], Path | None, dict[str, object]]:
        if not request.crop_sensitive_regions or not request.pdf_path:
            return (
                {},
                None,
                self._merge_report_meta(
                    report_meta,
                    self._partial_report_meta(
                        ReportCropInfo(cropping_enabled=False).model_dump(
                            exclude_none=True,
                            exclude={"cropped_regions", "total_cropped_regions"},
                        )
                    ),
                ),
            )

        crop_output_dir = (
            request.crop_output_dir
            or Path(os.getcwd()).parent / "pdfs" / "cropped_regions"
        )
        anonymization_output_dir = (
            request.anonymization_output_dir
            or Path(os.getcwd()).parent / "pdfs" / "anonymized"
        )
        cropped_regions = self._crop_sensitive_regions(
            pdf_path=request.pdf_path,
            crop_output_dir=crop_output_dir,
        )
        anonymized_pdf_path, meta = self._create_pdf_from_crops(
            pdf_path=request.pdf_path,
            crop_output_dir=crop_output_dir,
            anonymization_output_dir=anonymization_output_dir,
            report_meta=report_meta,
            cropped_regions=cropped_regions,
        )
        crop_info = ReportCropInfo(
            cropped_regions=cropped_regions,
            cropping_enabled=True,
            total_cropped_regions=sum(len(crops) for crops in cropped_regions.values()),
        )
        return (
            cropped_regions,
            anonymized_pdf_path,
            self._merge_report_meta(
                meta, self._partial_report_meta(crop_info.model_dump(exclude_none=True))
            ),
        )

    def _crop_sensitive_regions(
        self, *, pdf_path: Path, crop_output_dir: Path
    ) -> dict[str, list[str]]:
        logger.info("Beginne Cropping sensitiver Regionen...")
        try:
            return self.sensitive_cropper.crop_sensitive_regions(
                pdf_path=str(pdf_path), output_dir=str(crop_output_dir)
            )
        except Exception as exc:
            logger.error(
                "Fehler beim initialien Aufruf der Funktion zum Cropping: %s",
                exc,
            )
            return {}

    def _create_pdf_from_crops(
        self,
        *,
        pdf_path: Path,
        crop_output_dir: Path,
        anonymization_output_dir: Path,
        report_meta: Mapping[str, object],
        cropped_regions: Mapping[str, list[str]],
    ) -> tuple[Path | None, dict[str, object]]:
        if not cropped_regions:
            return None, dict(report_meta)
        anonymization_output_dir.mkdir(parents=True, exist_ok=True)
        anonymized_pdf_path = anonymization_output_dir / f"{pdf_path.stem}.pdf"
        try:
            self.sensitive_cropper.create_anonymized_pdf_with_crops(
                pdf_path=str(pdf_path),
                crop_output_dir=str(crop_output_dir),
                anonymized_pdf_path=str(anonymized_pdf_path),
            )
        except Exception as exc:
            logger.warning("Konnte anonymisiertes PDF nicht erstellen: %s", exc)
            crop_error = ReportCropInfo(anonymized_pdf_error=str(exc))
            return None, self._merge_report_meta(
                report_meta,
                self._partial_report_meta(
                    {"anonymized_pdf_error": crop_error.anonymized_pdf_error}
                ),
            )
        return anonymized_pdf_path, self._merge_report_meta(
            report_meta,
            self._partial_report_meta(
                {"anonymized_pdf_path": str(anonymized_pdf_path)}
            ),
        )

    def create_visualization_report(
        self,
        pdf_path: str | os.PathLike[str] | Path,
        output_dir: str | os.PathLike[str] | Path,
        visualize_all_pages: bool = False,
    ) -> list[str]:
        """Create visualization report for sensitive regions."""
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_name = Path(pdf_path).stem
        images = convert_pdf_to_images(pdf_path)
        visualization_files: list[str] = []

        pages_to_process = range(len(images)) if visualize_all_pages else [0]

        for page_num in pages_to_process:
            if page_num >= len(images):
                continue

            image = images[page_num]
            _full_text, word_boxes = tesseract_full_image_ocr(image)

            vis_filename = f"{pdf_name}_page_{page_num + 1}_analysis.png"
            vis_path = output_dir / vis_filename

            self.sensitive_cropper.visualize_sensitive_regions(
                image, word_boxes, str(vis_path)
            )
            visualization_files.append(str(vis_path))

        return visualization_files

    @classmethod
    def _merge_report_meta(
        cls,
        report_meta: Mapping[str, object],
        update: ReportMeta | Mapping[str, object],
    ) -> dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def _validated_report_meta(data: Mapping[str, object]) -> dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def _partial_report_meta(data: Mapping[str, object]) -> dict[str, object]:
        raise NotImplementedError

    def process_report(
        self,
        pdf_path: str | os.PathLike[str] | Path | None = None,
        image_path: str | os.PathLike[str] | Path | None = None,
        use_ensemble: bool = False,
        verbose: bool = True,
        use_llm: bool | None = None,
        text: str | None = None,
        create_anonymized_pdf: bool = False,
        anonymized_pdf_output_path: str | os.PathLike[str] | Path | None = None,
    ) -> tuple[str, str, dict[str, object], Path | None]:
        raise NotImplementedError

    def _load_report_text(self, request: ReportProcessRequest) -> str | None:
        raise NotImplementedError

    def _apply_ocr_fallback_if_needed(
        self, text: str, request: ReportProcessRequest
    ) -> str:
        raise NotImplementedError

    def _should_use_provider_llm(self, request: ReportProcessRequest) -> bool:
        raise NotImplementedError

    def _extract_or_default_report_meta(
        self, *, text: str, pdf_path: Path | None, use_provider_llm: bool
    ) -> dict[str, object]:
        raise NotImplementedError

    def _maybe_create_anonymized_pdf(
        self, *, request: ReportProcessRequest, report_meta: Mapping[str, object]
    ) -> tuple[Path | None, dict[str, object]]:
        raise NotImplementedError

    def _finalize_report_meta(
        self,
        *,
        report_meta: Mapping[str, object],
        text: str,
        anonymized_text: str,
        pdf_path: Path | None,
        use_provider_llm: bool,
    ) -> dict[str, object]:
        raise NotImplementedError

    @staticmethod
    def _metadata_field_count(meta: Mapping[str, object]) -> int:
        raise NotImplementedError

    @staticmethod
    def _report_metadata_has_signal(meta: Mapping[str, object]) -> bool:
        raise NotImplementedError

    @staticmethod
    def _should_attempt_report_llm(text: str) -> bool:
        raise NotImplementedError

    def _flag_strings(self, flag_name: str) -> Sequence[str]:
        value = getattr(self.flags, flag_name, None)
        if value is None:
            return ()
        if isinstance(value, str):
            stripped = value.strip()
            return (stripped,) if stripped else ()
        if isinstance(value, Sequence):
            items = cast(Sequence[object], value)
            result: list[str] = []
            for item in items:
                text = str(item).strip()
                if text:
                    result.append(text)
            return tuple(result)
        stripped = str(value).strip()
        return (stripped,) if stripped else ()
