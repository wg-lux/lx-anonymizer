"""Spawn-compatible model factories; heavyweight imports stay inside workers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lx_anonymizer.config import Settings
from lx_anonymizer.model_loading import ModelBlock, ModelChain, ModelUnavailableError

if TYPE_CHECKING:
    from lx_dtypes.models.meta.ReportMeta import ReportReaderFlags
    from PIL import Image

    from lx_anonymizer.ocr.ocr_frame import RoiInput
    from lx_anonymizer.runtime_types import ImageArray


def _configure_worker(config: Settings) -> None:
    """Install the already validated parent snapshot inside this private worker."""
    from lx_anonymizer.config import settings

    for name, value in config.model_dump().items():
        setattr(settings, name, value)


@dataclass(frozen=True)
class OCRRequest:
    frame: ImageArray
    roi: RoiInput = None
    high_quality: bool = False


OCRResult = tuple[str, float, dict[str, object]]


@dataclass(frozen=True)
class OCRFactory:
    backend: str
    config: Settings
    inference_threads: int = 2

    def __call__(self) -> Callable[[OCRRequest], OCRResult]:
        from lx_anonymizer.ocr.ocr_frame import FrameOCR

        _configure_worker(self.config)
        engine = FrameOCR(inference_threads=self.inference_threads)

        def recognize(request: OCRRequest) -> OCRResult:
            return engine.extract_text_with_backend(
                self.backend, request.frame, request.roi, request.high_quality
            )

        return recognize


@dataclass(frozen=True)
class MetadataFactory:
    backend: str
    config: Settings

    def __call__(self) -> Callable[[str], dict[str, object]]:
        _configure_worker(self.config)
        if self.backend == "llm":
            from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor

            extractor = LLMMetadataExtractor(
                provider=self.config.LLM_PROVIDER,
                base_url=self.config.resolved_llm_base_url,
                preferred_model=self.config.LLM_MODEL,
                model_timeout=self.config.LLM_TIMEOUT,
            )
            if extractor.current_model is None:
                raise ModelUnavailableError("Configured LLM is unavailable")

            def extract(text: str) -> dict[str, object]:
                result = extractor.extract_metadata(text)
                return result.to_dict() if result is not None else {}

            return extract
        if self.backend == "spacy":
            from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor

            extractor = PatientDataExtractor()

            def extract_local(text: str) -> dict[str, object]:
                from lx_anonymizer.sensitive_meta_interface import SensitiveMeta

                extractor.meta = SensitiveMeta()
                return dict(extractor(text))

            return extract_local
        raise ValueError(f"Unknown metadata model block: {self.backend}")


def ocr_chain(
    config: Settings, *, inference_threads: int = 2
) -> ModelChain[OCRRequest, OCRResult]:
    return ModelChain(
        [
            ModelBlock(
                name,
                OCRFactory(name, config, inference_threads),
                config.MODEL_BLOCK_TIMEOUT,
            )
            for name in config.OCR_MODEL_ORDER
        ]
    )


def metadata_chain(
    config: Settings, *, use_llm: bool
) -> ModelChain[str, dict[str, object]]:
    return ModelChain(
        [
            ModelBlock(
                f"llm:{config.LLM_PROVIDER}:{config.LLM_MODEL}"
                if name == "llm"
                else f"spacy:{config.SPACY_MODEL}",
                MetadataFactory(name, config),
                config.MODEL_BLOCK_TIMEOUT,
                metadata_has_signal if name == "llm" else None,
            )
            for name in config.METADATA_MODEL_ORDER
            if name != "llm" or use_llm
        ]
    )


def metadata_has_signal(result: dict[str, object]) -> bool:
    return any(
        result.get(key)
        for key in ("first_name", "last_name", "dob", "casenumber", "gender")
    )


class BoundedFrameOCR:
    """Attempt-owned OCR facade; models are loaded only in disposable workers."""

    def __init__(self, *, inference_threads: int | None = None) -> None:
        from lx_anonymizer.config import settings

        if inference_threads is not None and inference_threads <= 0:
            raise ValueError("OCR inference thread count must be positive")
        self.chain = ocr_chain(
            settings.model_copy(deep=True), inference_threads=inference_threads or 2
        )

    def extract_text_from_frame(
        self,
        frame: ImageArray,
        roi: RoiInput,
        high_quality: bool = True,
    ) -> OCRResult:
        if frame.size == 0 or frame.ndim not in (2, 3):
            raise ValueError("OCR requires a nonempty image")
        result = self.chain.run(OCRRequest(frame, roi, high_quality))
        return result

    def extract_text_with_rapidocr(self, frame: ImageArray) -> OCRResult:
        """Compatibility entry point for exhaustive full-frame analysis."""
        return self.extract_text_from_frame(frame, None, high_quality=False)

    def close(self) -> None:
        self.chain.close()


class BoundedMetadataExtractor:
    def __init__(self, *, use_llm: bool) -> None:
        from lx_anonymizer.config import settings

        self.chain = metadata_chain(settings.model_copy(deep=True), use_llm=use_llm)

    def __call__(self, text: str) -> dict[str, object]:
        return self.chain.run(text)

    def close(self) -> None:
        self.chain.close()


@dataclass(frozen=True)
class ReportMetadataFactory:
    config: Settings
    flags: ReportReaderFlags

    def __call__(self) -> Callable[[str], dict[str, object]]:
        from lx_anonymizer.ner.spacy_extractor import (
            EndoscopeDataExtractor,
            ExaminationDataExtractor,
            ExaminerDataExtractor,
            PatientDataExtractor,
            SpacyModelManager,
        )
        from lx_anonymizer.report_reader import ReportReader
        from lx_anonymizer.report_reader.extraction import ReportReaderExtractionMixin
        from lx_anonymizer.sensitive_meta_interface import SensitiveMeta

        _configure_worker(self.config)
        SpacyModelManager.get_model()

        def extract(text: str) -> dict[str, object]:
            # Model weights are cached; mutable patient state is request-local.
            context = ReportReader.__new__(ReportReader)
            context.flags = self.flags
            context.sensitive_meta = SensitiveMeta()
            context.patient_extractor = PatientDataExtractor()
            context.examiner_extractor = ExaminerDataExtractor()
            context.examination_extractor = ExaminationDataExtractor()
            context.endoscope_extractor = EndoscopeDataExtractor()
            return ReportReaderExtractionMixin.extract_report_meta(context, text, None)

        return extract


@dataclass(frozen=True)
class ReportOCRFactory:
    backend: str
    config: Settings

    def __call__(self) -> Callable[[Image.Image], str]:
        _configure_worker(self.config)
        if self.backend == "ensemble":
            from lx_anonymizer.ocr.ocr_ensemble import ensemble_ocr

            return ensemble_ocr
        if self.backend == "tesseract":
            from lx_anonymizer.ocr.ocr import tesseract_full_image_ocr

            def recognize(image: Image.Image) -> str:
                return tesseract_full_image_ocr(image)[0]

            return recognize
        if self.backend == "vision":
            from lx_anonymizer.llm.llm_service import LLMService

            service = LLMService(
                provider=self.config.LLM_PROVIDER,
                base_url=self.config.resolved_llm_base_url,
                timeout=self.config.LLM_TIMEOUT,
            )
            return service.recognize_image
        raise ValueError(f"Unknown report OCR model block: {self.backend}")


def text_has_signal(text: str) -> bool:
    return bool(text.strip())


@dataclass(frozen=True)
class OCRCorrectionFactory:
    config: Settings

    def __call__(self) -> Callable[[str], str]:
        from lx_anonymizer.llm.llm_service import LLMService

        _configure_worker(self.config)
        service = LLMService(
            provider=self.config.LLM_PROVIDER,
            base_url=self.config.resolved_llm_base_url,
            timeout=self.config.LLM_TIMEOUT,
        )
        return service.correct_ocr_text_in_chunks
