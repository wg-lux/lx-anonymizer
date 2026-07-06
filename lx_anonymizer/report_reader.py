from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, cast
import os
import gender_guesser.detector as gender_detector  # type: ignore[import-untyped]
from faker import Faker
from PIL import Image
from lx_dtypes.models.meta.ReportMeta import (
    ReportAnonymizerProvenance,
    ReportMeta,
    ReportProcessRequest,
    ReportProcessResult,
    ReportReaderFlags,
    ReportRedactionSummary,
)

from lx_anonymizer.anonymization.anonymizer import Anonymizer
from lx_anonymizer.anonymization.sensitive_region_cropper import (
    SensitiveRegionCropper,
)
from lx_anonymizer.config import settings
from lx_anonymizer.image_processing.pdf_operations import convert_pdf_to_images
from lx_anonymizer.llm.factory import LLMFactory
from lx_anonymizer.llm.llm_service import LLMService
from lx_anonymizer.metrics_provenance import build_anonymizer_provenance
from lx_anonymizer.ner.spacy_extractor import (
    EndoscopeDataExtractor,
    ExaminationDataExtractor,
    ExaminerDataExtractor,
    PatientDataExtractor,
)
from lx_anonymizer.ocr.ocr import (
    tesseract_full_image_ocr,
)  # , trocr_full_image_ocr_on_boxes # Import OCR fallback
from lx_anonymizer.ocr.ocr_ensemble import ensemble_ocr  # Import the new ensemble OCR
from lx_anonymizer.report_reader_extraction import (
    LLMExtractorProtocol,
    ReportReaderExtractionMixin,
)
from lx_anonymizer.report_reader_settings import get_report_reader_settings
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta
from lx_anonymizer.setup.custom_logger import logger


class ReportReader(ReportReaderExtractionMixin):
    _SUPPORTED_FLAG_KEYS = {
        "patient_info_line",
        "endoscope_info_line",
        "examiner_info_line",
        "cut_off_below",
        "cut_off_above",
    }

    def __init__(
        self,
        report_root_path: Optional[str] = None,
        locale: Optional[str] = None,
        employee_first_names: Sequence[str] | None = None,
        employee_last_names: Sequence[str] | None = None,
        flags: Mapping[str, object] | None = None,
        text_date_format: Optional[str] = None,
    ):
        """
        Initialize the report reader.

        Args:
            report_root_path: Optional root path for report files.
            locale: Faker locale used for anonymization replacements.
            employee_first_names: Optional override for replacement first names.
            employee_last_names: Optional override for replacement last names.
            flags: Optional report parsing/anonymization flags.
                Supported keys (all optional; missing keys fall back to defaults):
                - ``patient_info_line``: str marker for patient info line detection
                - ``endoscope_info_line``: str marker for endoscope info detection
                - ``examiner_info_line``: str marker for examiner info detection
                - ``cut_off_below``: list[str] (or str) markers for lower text cut-off
                - ``cut_off_above``: list[str] (or str) markers for upper text cut-off
                Behavior:
                - If ``flags`` is passed, it is merged with report-reader settings.
                - Unknown keys are preserved for compatibility, but logged.
            text_date_format: Date format string used during text anonymization.
        """
        self.report_root_path = report_root_path
        report_settings = get_report_reader_settings()

        self.locale = locale if locale is not None else report_settings.locale
        self.text_date_format = (
            text_date_format
            if text_date_format is not None
            else report_settings.text_date_format
        )
        self.employee_first_names = (
            employee_first_names
            if employee_first_names is not None
            else tuple(report_settings.first_names)
        )
        self.employee_last_names = (
            employee_last_names
            if employee_last_names is not None
            else tuple(report_settings.last_names)
        )
        self.flags = self._resolve_flags(flags)
        self.fake = Faker(locale=self.locale)
        self.gender_detector = gender_detector.Detector(case_sensitive=True)

        # Initialize extractors
        self.patient_extractor = PatientDataExtractor()
        self.examiner_extractor = ExaminerDataExtractor()
        self.endoscope_extractor = EndoscopeDataExtractor()
        self.examination_extractor = ExaminationDataExtractor()

        # Initialize sensitive region cropper
        self.sensitive_cropper = SensitiveRegionCropper()

        # Initialize Anonymizer
        self.anonymizer = Anonymizer()

        # Initialize provider-backed extractor
        self.llm_extractor: LLMExtractorProtocol | None = None
        self.llm_available = False
        self.ollama_available = False

        # initialize global sensitive meta
        self.sensitive_meta = SensitiveMeta()

        try:
            self.llm_extractor = LLMFactory.create_metadata_extractor()

            # Check if models are available
            if self.llm_extractor and self.llm_extractor.current_model:
                self.llm_available = True
                self.ollama_available = settings.LLM_PROVIDER == "ollama"
                logger.info(
                    "LLM features enabled for ReportReader via provider %s",
                    settings.LLM_PROVIDER,
                )
            else:
                logger.warning(
                    "Provider %s has no available models, LLM extraction disabled for ReportReader",
                    settings.LLM_PROVIDER,
                )
                self.llm_available = False
                self.ollama_available = False
                self.llm_extractor = None

        except Exception as e:
            logger.warning(
                "Provider %s unavailable for ReportReader, will use SpaCy/regex fallback: %s",
                settings.LLM_PROVIDER,
                e,
            )
            self.llm_available = False
            self.ollama_available = False
            self.llm_extractor = None

    @classmethod
    def _resolve_flags(cls, flags: object | None) -> ReportReaderFlags:
        """
        Merge caller-provided flags with defaults and normalize expected types.

        This keeps initialization robust when callers provide only a subset of flags.
        """
        if flags is not None and not isinstance(flags, Mapping):
            raise TypeError("flags must be a mapping")

        typed_flags = cast(Mapping[object, object], flags)
        default_flags = get_report_reader_settings().flags
        merged: dict[str, object] = default_flags.model_dump()
        provided = (
            {str(key): value for key, value in typed_flags.items()}
            if flags is not None
            else {}
        )

        unknown_keys = [k for k in provided.keys() if k not in cls._SUPPORTED_FLAG_KEYS]
        if unknown_keys:
            logger.warning(
                "ReportReader received unknown flags keys (preserved for compatibility): %s",
                sorted(map(str, unknown_keys)),
            )

        merged.update(provided)
        resolved = ReportReaderFlags.model_validate(merged)
        return resolved

    @staticmethod
    def _partial_report_meta(data: Mapping[str, object]) -> dict[str, object]:
        """Build a ReportMeta update from fields already validated at the boundary."""
        return dict(data)

    @staticmethod
    def _validated_report_meta(data: Mapping[str, object]) -> dict[str, object]:
        """Validate ReportReader-owned keys while ignoring SensitiveMeta extras."""
        aliases = getattr(ReportMeta, "_LEGACY_FIELD_ALIASES", {})
        report_meta_keys = set(ReportMeta.model_fields) | set(aliases)
        crop_default_marker = "__report_reader_default__"
        payload = {
            str(key): value
            for key, value in data.items()
            if str(key) in report_meta_keys
            and not (str(key) == "cropped_regions" and value is None)
            and str(key) != "sensitive_meta_state"
        }
        if not payload.get("cropped_regions"):
            payload["cropped_regions"] = {crop_default_marker: []}
        report_meta = ReportMeta.model_validate(payload).to_report_reader_dict()
        if report_meta.get("cropped_regions") == {crop_default_marker: []}:
            report_meta["cropped_regions"] = {}
        return report_meta

    @classmethod
    def _merge_report_meta(
        cls,
        report_meta: Mapping[str, object],
        update: ReportMeta | Mapping[str, object],
    ) -> dict[str, object]:
        """Merge a typed report metadata update into an existing metadata dict."""
        merged = dict(report_meta)
        if isinstance(update, ReportMeta):
            update_payload: dict[str, object] = update.model_dump(
                mode="json", exclude_none=True, exclude_unset=True
            )
            explicit_nulls = {
                field_name: None
                for field_name in update.__pydantic_fields_set__
                if getattr(update, field_name) is None
            }
            merged.update(update_payload | explicit_nulls)
        else:
            merged.update(update)
        return cls._validated_report_meta(merged)

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
        """
        Process a report by extracting text, metadata, and creating an anonymized version.
        """
        pdf_path = Path(pdf_path) if pdf_path is not None else None
        image_path = Path(image_path) if image_path is not None else None
        anonymized_pdf_output_path = (
            Path(anonymized_pdf_output_path)
            if anonymized_pdf_output_path is not None
            else None
        )
        request = ReportProcessRequest(
            pdf_path=pdf_path,
            image_path=image_path,
            use_ensemble=use_ensemble,
            verbose=verbose,
            use_llm=use_llm,
            text=text,
            create_anonymized_pdf=create_anonymized_pdf,
            anonymized_pdf_output_path=anonymized_pdf_output_path,
        )
        return self._process_report_request(request).as_tuple()

    def _process_report_request(
        self, request: ReportProcessRequest
    ) -> ReportProcessResult:
        self.sensitive_meta = SensitiveMeta()
        source_text = self._load_report_text(request)
        if source_text is None:
            return self._empty_process_result(request)

        text = self._apply_ocr_fallback_if_needed(source_text, request)
        if len(text.strip()) < 10 and not self._is_text_only_request(request):
            logger.error(
                "OCR fallback produced very short/no text, cannot proceed with "
                "metadata extraction."
            )
            original = self.read_pdf(request.pdf_path) if request.pdf_path else text
            return ReportProcessResult(
                text=original,
                anonymized_text=original,
                report_meta={},
                anonymized_pdf_path=request.pdf_path,
            )

        use_provider_llm = self._should_use_provider_llm(request)
        report_meta = self._extract_or_default_report_meta(
            text=text,
            pdf_path=request.pdf_path,
            use_provider_llm=use_provider_llm,
        )
        self.sensitive_meta.safe_update(report_meta)

        anonymized_text = self.anonymize_report(text=text, report_meta=report_meta)
        anonymized_pdf_path, report_meta = self._maybe_create_anonymized_pdf(
            request=request,
            report_meta=report_meta,
        )
        report_meta = self._finalize_report_meta(
            report_meta=report_meta,
            text=text,
            anonymized_text=anonymized_text,
            pdf_path=request.pdf_path,
            use_provider_llm=use_provider_llm,
        )

        self.sensitive_meta.safe_update(report_meta)
        return ReportProcessResult(
            text=text,
            anonymized_text=anonymized_text,
            report_meta=self.sensitive_meta.to_dict(),
            anonymized_pdf_path=anonymized_pdf_path,
        )

    @staticmethod
    def _is_text_only_request(request: ReportProcessRequest) -> bool:
        return (
            request.text is not None and not request.pdf_path and not request.image_path
        )

    @staticmethod
    def _empty_process_result(
        request: ReportProcessRequest,
    ) -> ReportProcessResult:
        fallback_path = request.pdf_path or request.image_path or Path("")
        return ReportProcessResult(
            text="",
            anonymized_text="",
            report_meta={},
            anonymized_pdf_path=fallback_path,
        )

    def _load_report_text(self, request: ReportProcessRequest) -> str | None:
        if request.text is not None:
            return request.text
        if request.pdf_path:
            if not request.pdf_path.exists():
                logger.error("PDF file not found: %s", request.pdf_path)
                return None
            return self.read_pdf(request.pdf_path)
        if request.image_path:
            if not request.image_path.exists():
                logger.error("Image file not found: %s", request.image_path)
                return None
            logger.info("Reading text from image file: %s", request.image_path)
            return self._ocr_single_image(request.image_path)
        return None

    def _ocr_single_image(self, image_path: Path) -> str | None:
        try:
            pil_image = Image.open(image_path)
        except OSError as exc:
            logger.error("Error reading image %s: %s", image_path, exc)
            return None
        text, _ = tesseract_full_image_ocr(pil_image)
        return text

    def _apply_ocr_fallback_if_needed(
        self, text: str, request: ReportProcessRequest
    ) -> str:
        if self._is_text_only_request(request):
            logger.debug(
                "Skipping OCR fallback for text-only input "
                "(no pdf_path/image_path provided)."
            )
            return text
        if len(text.strip()) >= 50:
            return text

        logger.info(
            "Short/No text detected by pdfplumber (%d chars), applying OCR fallback.",
            len(text.strip()),
        )
        images = self._load_images_for_ocr(request)
        if not images:
            return text

        ocr_text = self._ocr_images(images, use_ensemble=request.use_ensemble)
        logger.info(
            "OCR fallback finished. Total text length: %d. Preview: %s...",
            len(ocr_text),
            ocr_text[:200],
        )
        return self._correct_ocr_text_if_enabled(ocr_text)

    def _load_images_for_ocr(self, request: ReportProcessRequest) -> list[Image.Image]:
        if request.pdf_path:
            logger.info("Converting PDF to images for OCR: %s", request.pdf_path)
            return convert_pdf_to_images(request.pdf_path)
        if request.image_path:
            try:
                return [Image.open(request.image_path)]
            except OSError as exc:
                logger.error("Failed to open image file: %s", exc)
        return []

    def _ocr_images(self, images: list[Image.Image], *, use_ensemble: bool) -> str:
        parts: list[str] = []
        for idx, pil_image in enumerate(images, start=1):
            logger.info("Processing page %d with OCR...", idx)
            ocr_part = self._ocr_image(
                pil_image, page_num=idx, use_ensemble=use_ensemble
            )
            if ocr_part:
                parts.append(ocr_part)
        return " ".join(parts).strip()

    def _ocr_image(
        self, pil_image: Image.Image, *, page_num: int, use_ensemble: bool
    ) -> str:
        if use_ensemble:
            try:
                return ensemble_ocr(pil_image)
            except Exception as exc:
                logger.error("Ensemble OCR failed on page %d: %s", page_num, exc)
        try:
            text_part, _ = tesseract_full_image_ocr(pil_image)
            logger.info("Tesseract OCR successful for page %d", page_num)
            return text_part
        except Exception as exc:
            logger.error("Tesseract OCR failed on page %d: %s", page_num, exc)
            return ""

    def _correct_ocr_text_if_enabled(self, text: str) -> str:
        if not text:
            return text
        if len(text.strip()) < int(settings.REPORT_OCR_CORRECTION_MIN_TEXT_LENGTH):
            logger.info(
                "Skipping LLM OCR correction for short OCR text (%d chars).",
                len(text.strip()),
            )
            return text
        if not getattr(self, "llm_available", False):
            logger.info("Skipping LLM correction for OCR text: provider unavailable")
            return text

        logger.info(
            "Applying LLM correction to OCR text via provider %s",
            settings.LLM_PROVIDER,
        )
        try:
            llm_client = LLMService(
                provider=settings.LLM_PROVIDER,
                base_url=settings.resolved_llm_base_url,
            )
            corrected = llm_client.correct_ocr_text_in_chunks(text)
        except Exception as exc:
            logger.warning("Error using LLM for correction: %s", exc)
            return text

        if corrected and corrected != text and len(corrected) > 0.5 * len(text):
            logger.info("OCR text successfully corrected by LLM.")
            return corrected
        if corrected == text:
            logger.info("LLM correction resulted in the same text.")
        else:
            logger.warning(
                "LLM OCR correction failed or produced poor result, "
                "using original OCR text."
            )
        return text

    def _should_use_provider_llm(self, request: ReportProcessRequest) -> bool:
        return self.llm_available if request.use_llm is None else request.use_llm

    def _extract_or_default_report_meta(
        self, *, text: str, pdf_path: Path | None, use_provider_llm: bool
    ) -> dict[str, object]:
        if len(text.strip()) < 10:
            logger.warning(
                "Skipping metadata extraction due to insufficient text content."
            )
            return self._validated_report_meta(
                self._partial_report_meta(
                    {
                        "pdf_hash": self.pdf_hash_file(pdf_path)
                        if pdf_path and pdf_path.exists()
                        else None
                    }
                )
            )

        if not use_provider_llm:
            logger.info("Using default SpaCy/Regex metadata extraction.")
            report_meta = self.extract_report_meta(text, pdf_path)
            self.sensitive_meta.safe_update(report_meta)
            return self.sensitive_meta.to_dict()

        logger.info("Using provider-backed LLM metadata extraction.")
        report_meta = self.extract_report_meta(text, pdf_path)
        if self._report_metadata_has_signal(report_meta):
            logger.info(
                "Skipping LLM report metadata extraction because local extraction "
                "already found signal."
            )
            self.sensitive_meta.safe_update(report_meta)
            return self.sensitive_meta.to_dict()
        if not self._should_attempt_report_llm(text):
            logger.info(
                "Skipping LLM report metadata extraction for short/low-signal text "
                "(%d chars).",
                len(text.strip()),
            )
            self.sensitive_meta.safe_update(report_meta)
            return self.sensitive_meta.to_dict()

        llm_meta = self.extract_report_meta_with_llm(text)
        if llm_meta:
            self.sensitive_meta.safe_update(llm_meta)
            return self.sensitive_meta.to_dict()
        logger.warning(
            "Provider-backed LLM extraction failed. Falling back to default "
            "SpaCy/Regex extraction."
        )
        return self.extract_report_meta(text, pdf_path)

    def _maybe_create_anonymized_pdf(
        self, *, request: ReportProcessRequest, report_meta: Mapping[str, object]
    ) -> tuple[Path | None, dict[str, object]]:
        if not request.create_anonymized_pdf or not request.pdf_path:
            return None, dict(report_meta)

        logger.info("Creating anonymized PDF with blackened sensitive regions...")
        output_path = self.anonymizer.create_anonymized_pdf(
            pdf_path=str(request.pdf_path),
            output_path=(
                str(request.anonymized_pdf_output_path)
                if request.anonymized_pdf_output_path is not None
                else None
            ),
            report_meta=dict(report_meta),
        )
        if not output_path:
            raise RuntimeError(
                "create_anonymized_pdf=True but no anonymized PDF path was produced."
            )

        update = self._partial_report_meta({"anonymized_pdf_path": output_path})
        redaction_summary = self.anonymizer.last_redaction_summary
        if isinstance(redaction_summary, dict):
            update = self._partial_report_meta(
                update
                | {
                    "redaction_summary": ReportRedactionSummary.model_validate(
                        redaction_summary
                    )
                }
            )
        logger.info("Anonymized PDF created: %s", output_path)
        return Path(str(output_path)), self._merge_report_meta(report_meta, update)

    def _finalize_report_meta(
        self,
        *,
        report_meta: Mapping[str, object],
        text: str,
        anonymized_text: str,
        pdf_path: Path | None,
        use_provider_llm: bool,
    ) -> dict[str, object]:
        meta = self._merge_report_meta(
            report_meta,
            self._partial_report_meta(
                {
                    "file_path": str(pdf_path) if pdf_path else None,
                    "text": text,
                    "anonymized_text": anonymized_text,
                }
            ),
        )
        detector_sources = ["regex", "spacy"]
        if use_provider_llm:
            detector_sources.append("llm")
        redaction_summary_value = meta.get("redaction_summary")
        redaction_count = 0
        if isinstance(redaction_summary_value, dict):
            redaction_summary = ReportRedactionSummary.model_validate(
                redaction_summary_value
            )
            redaction_count = redaction_summary.redaction_region_count
            detector_sources.append("pdf_redaction")
        return self._merge_report_meta(
            meta,
            self._partial_report_meta(
                {
                    "anonymizer_provenance": ReportAnonymizerProvenance.model_validate(
                        build_anonymizer_provenance(
                            detector_sources=detector_sources,
                            proposal_counts={
                                "report_metadata_fields": self._metadata_field_count(
                                    meta
                                ),
                                "redaction_regions": redaction_count,
                            },
                        ).model_dump()
                    )
                }
            ),
        )

    @staticmethod
    def _metadata_field_count(meta: Mapping[str, object]) -> int:
        keys = (
            "first_name",
            "last_name",
            "dob",
            "casenumber",
            "gender",
            "examination_date",
            "examination_time",
            "examiner_first_name",
            "examiner_last_name",
            "center",
        )
        return sum(1 for key in keys if meta.get(key))

    @staticmethod
    def _report_metadata_has_signal(meta: Mapping[str, object]) -> bool:
        signal_keys = (
            "first_name",
            "last_name",
            "dob",
            "casenumber",
            "gender",
            "examination_date",
            "examination_time",
            "examiner_first_name",
            "examiner_last_name",
            "center",
        )
        return any(
            (
                (value := meta.get(key)) is not None
                and (not isinstance(value, str) or bool(value.strip()))
            )
            for key in signal_keys
        )

    @staticmethod
    def _should_attempt_report_llm(text: str) -> bool:
        return bool(
            text and len(text.strip()) >= int(settings.REPORT_LLM_MIN_TEXT_LENGTH)
        )
