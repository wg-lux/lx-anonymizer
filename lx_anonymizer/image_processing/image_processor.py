import uuid
from pathlib import Path
from typing import Callable, TypeAlias, cast

import cv2
import pytesseract  # type: ignore
import numpy as np
from PIL import Image

from lx_anonymizer.config import settings
from lx_anonymizer.sensitive_meta_interface import (
    SensitiveMeta,
    sensitive_meta_to_dict,
)
from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor
from lx_anonymizer.pipeline_manager import process_images_with_OCR_and_NER
from lx_anonymizer.setup.custom_logger import get_logger
from lx_dtypes.models.contracts.image_processing import ImageProcessingResultPayload

logger = get_logger(__name__)

sensitive_meta = SensitiveMeta()

OcrTextOutput: TypeAlias = bytes | str | dict[str, bytes | str]
ModifiedImageMap: TypeAlias = dict[tuple[str, str], str]
ProcessPipelineResult: TypeAlias = tuple[ModifiedImageMap, dict[str, object]]
ProcessImagesCallable: TypeAlias = Callable[
    [Path, str, str, float, int, int, bool, bool], ProcessPipelineResult
]
OcrToStringCallable: TypeAlias = Callable[[Image.Image], OcrTextOutput]


def _coerce_tesseract_output(output: OcrTextOutput) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="ignore")
    return "\n".join(
        entry.decode("utf-8", errors="ignore")
        if isinstance(entry, bytes)
        else str(entry)
        for entry in output.values()
    )


def _run_ocr_text_extraction(image: Image.Image) -> str:
    image_to_string = cast(
        OcrToStringCallable,
        getattr(pytesseract, "image_to_string", None),
    )
    raw_output = image_to_string(image)
    conventional_text = _coerce_tesseract_output(raw_output)
    if not (
        settings.LLM_ENABLED
        and settings.OLLAMA_OCR_ENABLED
        and settings.LLM_PROVIDER == "ollama"
    ):
        return conventional_text

    try:
        from lx_anonymizer.llm.llm_service import LLMService

        recognized_text = LLMService(
            provider="ollama",
            base_url=settings.resolved_llm_base_url,
            model_name=settings.LLM_MODEL,
            timeout=settings.LLM_TIMEOUT,
        ).recognize_image(image, candidate_text=conventional_text)
        return recognized_text or conventional_text
    except Exception as exc:
        logger.warning("Gemma 4 vision OCR failed; using Tesseract output: %s", exc)
        return conventional_text


_typed_process_images_with_OCR_and_NER: ProcessImagesCallable = cast(
    ProcessImagesCallable, process_images_with_OCR_and_NER
)


def process_image(
    img_path: Path,
    east_path: Path,
    device: str,
    min_confidence: float,
    width: int,
    height: int,
    results_dir: Path,
    temp_dir: Path,
    text_extracted: bool = False,
    skip_blur: bool = False,
    skip_reassembly: bool = False,
    disable_llm: bool = False,
) -> tuple[Path, dict[str, object]]:
    """
    Process a single image or PDF page

    Parameters:
    - img_path: Path to the image
    - east_path: Path to the EAST model
    - device: Device configuration name
    - min_confidence: Minimum confidence for text detection
    - width: Width for resizing
    - height: Height for resizing
    - results_dir: Directory to save results
    - temp_dir: Temporary directory for processing
    - text_extracted: Optional extracted text from PDF
    - skip_blur: Whether to skip blurring operations
    - skip_reassembly: Whether to skip PDF reassembly
    - disable_llm: Whether to disable LLM analysis

    Returns:
    - Path to the processed image
    - Anonymization data dictionary
    """
    logger.info(f"Processing image: {img_path}")

    # If we're skipping the blur operations but want analysis, use LLM
    if skip_blur and not disable_llm:
        logger.info("Skipping blur operations, performing analysis only")

        extractor = LLMMetadataExtractor(
            base_url=settings.LLM_BASE_URL,
            preferred_model=settings.LLM_MODEL,
            model_timeout=settings.LLM_TIMEOUT,
        )

        # Extract text from image for LLM analysis
        try:
            image = Image.open(img_path).convert("RGB")
            ocr_text = _run_ocr_text_extraction(image)
        except Exception as e:
            logger.warning(f"Failed to extract OCR text: {e}")
            ocr_text = ""

        # Use LLM to analyze the extracted text
        llm_metadata = extractor.extract_metadata(ocr_text) if ocr_text else None
        if llm_metadata is None:
            llm_results: dict[str, object] = {}
        else:
            sensitive_meta.safe_update(llm_metadata)
            llm_results = sensitive_meta_to_dict(sensitive_meta)

        # Combine analysis results if we have text
        if text_extracted and not disable_llm:
            text_metadata = extractor.extract_metadata(ocr_text)
            if text_metadata:
                sensitive_meta.safe_update(text_metadata)
                text_analysis = sensitive_meta_to_dict(sensitive_meta)
            else:
                text_analysis = {}

            combined_results: dict[str, object] = {
                "image_analysis": llm_results,
                "text_analysis": text_analysis,
            }
        else:
            combined_results: dict[str, object] = {"image_analysis": llm_results}

        # Return the original image path and the analysis results
        return img_path, combined_results

    # Normal processing path with OCR, NER, and optional blurring
    pipeline_result_tuple = _typed_process_images_with_OCR_and_NER(
        Path(img_path),
        str(east_path),
        device,
        min_confidence,
        width,
        height,
        skip_blur,
        skip_reassembly,
    )
    modified_images_map = pipeline_result_tuple[0]
    pipeline_result = pipeline_result_tuple[1]

    # Get the processed image path
    if modified_images_map:
        # Get the last modified image (the final result)
        last_key = list(modified_images_map.keys())[-1]
        processed_image_path = Path(modified_images_map[last_key])
    else:
        # If no modification was done, return the original
        processed_image_path = img_path

    # If we have a processed path, copy it to the results directory
    if processed_image_path != img_path:
        file_extension = processed_image_path.suffix
        result_path = results_dir / f"processed_{uuid.uuid4()}{file_extension}"
        try:
            img_to_save = cast(np.ndarray | None, cv2.imread(str(processed_image_path)))
            if img_to_save is None:
                logger.error(f"Failed to read image: {processed_image_path}")
            else:
                # Copy the image to the results directory
                success = cv2.imwrite(str(result_path), img_to_save)
                if not success:
                    logger.error(f"Failed to write image to: {result_path}")
                else:
                    logger.debug(f"Image copied successfully to: {result_path}")
        except Exception as e:
            logger.error(f"Failed to save image to results with error {e}")

        processed_image_path = result_path

    typed_result = ImageProcessingResultPayload.model_validate(pipeline_result)
    return processed_image_path, typed_result.model_dump()


def resize_image(image_path: Path, max_width: int, max_height: int) -> None:
    try:
        image = cast(np.ndarray | None, cv2.imread(str(image_path)))
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return
    if image is None:
        logger.error(f"Unable to read image for resizing: {image_path}")
        return
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        try:
            success = cv2.imwrite(str(image_path), resized_image)  # Saving image
            if not success:
                logger.error("Failed to write image")
            else:
                logger.debug(f"Image saved successfully: {image_path}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")

        logger.debug(f"Image resized to {new_size}")
        logger.debug(f"Image resized to {new_size}")
        logger.debug(f"Image resized to {new_size}")
