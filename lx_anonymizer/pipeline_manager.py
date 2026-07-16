import csv
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, TypeAlias, cast

import cv2
import numpy as np
import numpy.typing as npt
import pymupdf  # type: ignore[import-untyped]
from PIL import Image
from lx_anonymizer.anonymization.blur import blur_function
from lx_anonymizer.region_processing.box_operations import (
    close_to_box,
    filter_empty_boxes,
    find_or_create_close_box,
    get_dominant_color,
)

from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.setup.device_reader import read_background_color, read_name_boxes
from lx_anonymizer.setup.directory_setup import (
    create_blur_directory,
    create_temp_directory,
)
from lx_anonymizer.text_detection.east_text_detection import east_text_detection
from lx_anonymizer.ner.flair_NER import flair_NER_German
from lx_anonymizer.nlp.fuzzy_matching import (
    correct_box_for_new_text,
    fuzzy_match_snippet,
)
from lx_anonymizer.pseudonymization.names_generator import (
    gender_and_handle_device_names,
    gender_and_handle_separate_names,
)
from lx_anonymizer.ocr.ocr import (
    tesseract_full_image_ocr,
    tesseract_on_boxes,
    trocr_on_boxes,
)
from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor
from lx_anonymizer.llm.llm_service import LLMService
from lx_anonymizer.config import settings
from lx_anonymizer.ner.spacy_NER import spacy_NER_German
from lx_anonymizer.text_detection.tesseract_text_detection import (
    tesseract_text_detection,
)
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta

# Import CRAFT text detection if available (requires hezar)
try:
    from lx_anonymizer.text_detection.craft_text_detection import craft_text_detection

    _craft_available = True
except ImportError:
    _craft_available = False

    def craft_text_detection(
        image_input: Any,
        min_confidence: Any = None,
        width: Any = None,
        height: Any = None,
    ) -> Any:
        raise ImportError(
            "CRAFT text detection requires 'hezar' package. Install with: pip install lx-anonymizer[llm]"
        )


CRAFT_AVAILABLE: bool = _craft_available

BoundingBox: TypeAlias = Tuple[int, int, int, int]
OcrResult: TypeAlias = Tuple[str, BoundingBox]
DetectedEntity: TypeAlias = Tuple[str, str]
CombinedResult: TypeAlias = Tuple[str, BoundingBox, float, List[DetectedEntity]]
ModifiedImageMap: TypeAlias = Dict[Tuple[str, str], str]
ProcessingResult: TypeAlias = Dict[str, object]


class _FlairEntity(Protocol):
    text: str
    tag: str


# Configure logging
logger = get_logger(__name__)

sensitive_meta = SensitiveMeta()
_MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
    "pdf": "application/pdf",
}
_SUPPORTED_FILE_TYPES = {"jpg", "jpeg", "png", "tiff", "pdf"}


def _resolve_file_type(file_path: Path) -> str:
    file_extension = file_path.suffix.lower().lstrip(".")
    file_type = _MIME_TYPES.get(file_extension, "application/octet-stream").split("/")[
        -1
    ]
    if file_type not in _SUPPORTED_FILE_TYPES:
        raise ValueError("Invalid file type.")
    return file_type


def _prepare_image_paths(
    file_path: Path, file_type: str, temp_dir: Path
) -> Tuple[List[Path], str]:
    image_paths: List[Path] = []
    extracted_text = ""

    if file_type == "pdf":
        doc = pymupdf.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = cast(object, page.get_text())  # pyright: ignore[reportUnknownMemberType]
            if isinstance(text, str) and text:
                extracted_text += text
            pix = page.get_pixmap()  # pyright: ignore[reportUnknownMemberType]
            image_path = temp_dir / f"{uuid.uuid4()}_page_{page_num}.png"
            pix.save(image_path)  # pyright: ignore[reportUnknownMemberType]
            image_paths.append(image_path)
    else:
        image_paths = [file_path]

    if not image_paths:
        error_message = "No images to process."
        logger.error(error_message)
        raise RuntimeError(error_message)

    return image_paths, extracted_text


def _load_device_defaults(
    device: str, img_path: Path
) -> Tuple[
    Optional[BoundingBox],
    Optional[BoundingBox],
    Tuple[int, int, int],
]:
    try:
        first_name_box, last_name_box = read_name_boxes(device)
        background_color = cast(
            Tuple[int, int, int],
            read_background_color(device),  # pyright: ignore[reportUnknownArgumentType]
        )
    except Exception:
        logger.warning("Using default values for name replacement.")
        first_name_box, last_name_box = None, None
        image_obj = cast(object, cv2.imread(str(img_path)))
        if image_obj is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cast(npt.NDArray[np.uint8], image_obj)
        image = image.astype(np.uint8, copy=False)
        background_color = get_dominant_color(image)
    return first_name_box, last_name_box, background_color


def _run_llm_image_analysis(img_path: Path) -> Dict[str, object]:
    logger.info("Skipping blur, running LLM analysis on image")
    extractor = LLMMetadataExtractor(
        base_url=settings.LLM_BASE_URL,
        preferred_model=settings.LLM_MODEL,
        model_timeout=settings.LLM_TIMEOUT,
    )

    try:
        ocr_text, _ = tesseract_full_image_ocr(img_path)
    except Exception as e:
        logger.warning(f"Failed to extract OCR text: {e}")
        ocr_text = ""

    if (
        settings.LLM_ENABLED
        and settings.OLLAMA_OCR_ENABLED
        and settings.LLM_PROVIDER == "ollama"
    ):
        try:
            with Image.open(img_path) as image:
                recognized_text = LLMService(
                    provider="ollama",
                    base_url=settings.resolved_llm_base_url,
                    model_name=settings.LLM_MODEL,
                    timeout=settings.LLM_TIMEOUT,
                ).recognize_image(image, candidate_text=ocr_text)
            ocr_text = recognized_text or ocr_text
        except Exception as exc:
            logger.warning(
                "Gemma 4 vision OCR failed during image analysis; "
                "using Tesseract output: %s",
                exc,
            )

    if not ocr_text:
        return {}

    llm_metadata = extractor.extract_metadata(ocr_text)
    if not llm_metadata:
        return {}

    sensitive_meta.safe_update(llm_metadata)
    return sensitive_meta.to_dict()


def detect_combined_text_boxes(
    img_path: Path,
    east_path: str,
    min_confidence: float,
    width: int,
    height: int,
) -> List[BoundingBox]:
    east_boxes, _ = east_text_detection(
        img_path, east_path, min_confidence, width, height
    )
    tesseract_boxes, _ = tesseract_text_detection(
        img_path, min_confidence, width, height
    )
    craft_boxes: List[BoundingBox] = []
    if CRAFT_AVAILABLE:
        craft_boxes, _ = craft_text_detection(img_path, min_confidence, width, height)
    else:
        logger.info("CRAFT is unavailable; continuing with EAST and Tesseract")
    return east_boxes + tesseract_boxes + craft_boxes


_detect_combined_text_boxes = detect_combined_text_boxes


def _run_ocr_for_boxes(
    img_path: Path, combined_boxes: List[BoundingBox]
) -> Tuple[List[OcrResult], List[float]]:
    logger.info("Running OCR on boxes")
    trocr_results, trocr_confidences = trocr_on_boxes(img_path, combined_boxes)
    tesseract_results, tess_confidences = tesseract_on_boxes(img_path, combined_boxes)

    all_ocr_results = trocr_results + tesseract_results
    all_ocr_confidences = trocr_confidences + tess_confidences
    all_ocr_results = filter_empty_boxes(all_ocr_results)
    return all_ocr_results, all_ocr_confidences


def _write_ner_csv(
    csv_dir: Path,
    file_path: Path,
    combined_results: List[CombinedResult],
) -> Path:
    csv_path = (
        csv_dir / f"name_anonymization_data_i{Path(file_path).stem}{uuid.uuid4()}.csv"
    )
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "filename",
            "startX",
            "startY",
            "endX",
            "endY",
            "text",
            "phrase_box",
            "ocr_confidence",
            "entity_text",
            "entity_tag",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for combined_result in combined_results:
            phrase, phrase_box, ocr_confidence, entities = combined_result
            startX, startY, endX, endY = phrase_box
            for entity_text, entity_tag in entities:
                writer.writerow(
                    {
                        "filename": file_path,
                        "startX": startX,
                        "startY": startY,
                        "endX": endX,
                        "endY": endY,
                        "text": phrase,
                        "phrase_box": phrase_box,
                        "ocr_confidence": ocr_confidence,
                        "entity_text": entity_text,
                        "entity_tag": entity_tag,
                    }
                )
    return csv_path


def _save_final_blurred_image(blurred_image_path: Optional[Path]) -> None:
    if blurred_image_path is None:
        return
    output_filename = f"blurred_image_{uuid.uuid4()}.jpg"
    blur_dir = create_blur_directory(default_main_directory=None)
    output_path = Path(blur_dir) / output_filename
    final_image_obj = cast(object, cv2.imread(str(blurred_image_path)))
    if final_image_obj is None:
        raise ValueError(f"Unable to read blurred image: {blurred_image_path}")
    final_image = cast(npt.NDArray[np.uint8], final_image_obj)
    cv2.imwrite(str(output_path), final_image)
    logger.info(f"Final blurred image saved to: {output_path}")


def process_images_with_OCR_and_NER(
    file_path: Path | str,
    east_path: str = "frozen_east_text_detection.pb",
    device: str = "default",
    min_confidence: float = 0.5,
    width: int = 320,
    height: int = 320,
    skip_blur: bool = False,
    skip_reassembly: bool = False,
) -> Tuple[ModifiedImageMap, ProcessingResult]:
    """
    Process images with OCR and NER.

    Parameters:
    - file_path: str
        The path to the image or PDF file to process.
    - east_path: str
        The path to the EAST text detector model. Default is 'frozen_east_text_detection.pb'.
    - device: str
        The device name to set the correct text settings. Default is 'default'.
    - min_confidence: float
        The minimum probability required to inspect a region. Default is 0.5.
    - width: int
        The resized image width (should be a multiple of 32). Default is 320.
    - height: int
        The resized image height (should be a multiple of 32). Default is 320.
    - skip_blur: bool
        A boolean value representing if the blur function should be skipped. Default is False.
    - skip_reassembly: bool
        Skip reassembling the PDF after processing. Default is False.
    """
    file_path = Path(file_path)
    temp_dir, _base_dir, csv_dir = create_temp_directory()
    temp_dir = Path(temp_dir)
    csv_dir = Path(csv_dir)
    logger.info(f"Processing file: {file_path}")
    modified_images_map: ModifiedImageMap = {}
    combined_results: List[CombinedResult] = []
    names_detected: List[str] = []
    gender_pars: List[str] = []
    llm_results: Dict[str, object] = {}

    try:
        file_type = _resolve_file_type(file_path)
        image_paths, extracted_text = _prepare_image_paths(
            file_path, file_type, temp_dir
        )

        blurred_image_path = image_paths[0]
        for img_path in image_paths:
            logger.info(f"Processing image: {img_path}")
            _full_text, _word_boxes = tesseract_full_image_ocr(img_path)
            first_name_box, last_name_box, background_color = _load_device_defaults(
                device, img_path
            )

            # Handle LLM analysis of the image
            if skip_blur:
                llm_results = _run_llm_image_analysis(img_path)

            logger.info(
                f"LLM image analysis results: {len(llm_results) if llm_results else 0} entities"
            )

            # Regular blur processing (only if not skipping blur)
            if not skip_blur:
                if first_name_box and last_name_box:
                    blurred_image_path = blur_function(
                        blurred_image_path, first_name_box, background_color
                    )
                    blurred_image_path = blur_function(
                        blurred_image_path, last_name_box, background_color
                    )

            combined_boxes = detect_combined_text_boxes(
                img_path, east_path, min_confidence, width, height
            )
            all_ocr_results, all_ocr_confidences = _run_ocr_for_boxes(
                img_path, combined_boxes
            )

            if skip_blur:
                combined_results.extend(
                    (phrase, phrase_box, ocr_confidence, [])
                    for (phrase, phrase_box), ocr_confidence in zip(
                        all_ocr_results, all_ocr_confidences
                    )
                )
            else:
                full_text_candidates = [phrase for phrase, _box in all_ocr_results]
                (
                    blurred_image_path,
                    modified_images_map,
                    combined_results,
                    updated_genders,
                ) = do_ocr_with_fuzzy_correction(
                    all_ocr_results=all_ocr_results,
                    all_ocr_confidences=all_ocr_confidences,
                    blurred_image_path=blurred_image_path,
                    combined_results=combined_results,
                    names_detected=names_detected,
                    device=device,
                    modified_images_map=modified_images_map,
                    combined_boxes=combined_boxes,
                    first_name_box=first_name_box,
                    last_name_box=last_name_box,
                    full_text_candidates=full_text_candidates,
                )
                gender_pars.extend(updated_genders)

        csv_path = _write_ner_csv(csv_dir, file_path, combined_results)

        logger.info(f"NER results saved to {csv_path}")

        result: ProcessingResult = {
            "filename": file_path,
            "file_type": file_type,
            "extracted_text": extracted_text,
            "names_detected": names_detected,
            "combined_results": combined_results,
            "modified_images_map": modified_images_map,
            "gender_pars": gender_pars,
        }

        # If not skipping blur and we have a blurred image path
        blurred_image_path_for_save = cast(Optional[str | Path], blurred_image_path)
        if not skip_blur and blurred_image_path_for_save is not None:
            _save_final_blurred_image(Path(blurred_image_path_for_save))

        # Always run LLM analysis and add results
        result["llm_results"] = llm_results

        logger.info(
            f"Processing completed: {len(combined_results)} results, csv: {csv_path}"
        )
        return modified_images_map, result
    except Exception as e:
        error_message = (
            f"Error in process_images_with_OCR_and_NER: {e}, File Path: {file_path}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message)


def process_text(extracted_text: str) -> str:
    cleaned_text = re.sub(r"\n{2,}", "\n", extracted_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    return cleaned_text


def process_ocr_results(
    image_path: str,
    phrase: str,
    phrase_box: BoundingBox,
    ocr_confidence: float,
    combined_results: List[CombinedResult],
    names_detected: List[str],
    device: str,
    modified_images_map: ModifiedImageMap,
    combined_boxes: List[BoundingBox],
    first_name_box: Optional[BoundingBox] = None,  # Changed here
    last_name_box: Optional[BoundingBox] = None,  # Changed here
) -> Tuple[
    str,
    ModifiedImageMap,
    List[CombinedResult],
    List[str],
]:
    processed_text = process_text(phrase)
    entities = split_and_check(processed_text)
    logger.info(f"Entities detected: {entities}")

    box_to_image_map: Dict[Tuple[str, str], Path] = {}
    gender_pars: List[str] = []  # Changed from {} to []
    new_image_path = (
        image_path  # Keep track of the current image path being manipulated
    )

    for entity in entities:
        name = entity[0]
        if first_name_box and last_name_box:
            if close_to_box(first_name_box, phrase_box) or close_to_box(
                last_name_box, phrase_box
            ):
                box_to_image_map, gender_par = gender_and_handle_device_names(
                    name, phrase_box, cast(str, new_image_path), device
                )
            else:
                new_image_path, last_name_box = modify_image_for_name(
                    new_image_path, phrase_box, combined_boxes
                )
                box_to_image_map, gender_par = gender_and_handle_separate_names(
                    name, phrase_box, last_name_box, cast(str, new_image_path), device
                )
        else:
            new_image_path, last_name_box = modify_image_for_name(
                new_image_path, phrase_box, combined_boxes
            )
            box_to_image_map, gender_par = gender_and_handle_separate_names(
                name, phrase_box, last_name_box, cast(str, new_image_path), device
            )

        names_detected.append(name)
        gender_pars.append(str(gender_par))  # Now valid since gender_pars is a list
        for box_key, modified_image_path in box_to_image_map.items():
            modified_images_map[(cast(str, box_key), str(new_image_path))] = cast(
                str, modified_image_path
            )

    combined_results.append((phrase, phrase_box, ocr_confidence, entities))
    return (
        str(new_image_path),
        modified_images_map,
        combined_results,
        gender_pars,
    )  # Always return the last modified path


def do_ocr_with_fuzzy_correction(
    all_ocr_results: List[OcrResult],
    all_ocr_confidences: List[float],
    blurred_image_path: str | Path,
    combined_results: List[CombinedResult],
    names_detected: List[str],
    device: str,
    modified_images_map: ModifiedImageMap,
    combined_boxes: List[BoundingBox],
    first_name_box: Optional[BoundingBox] = None,
    last_name_box: Optional[BoundingBox] = None,
    full_text_candidates: Optional[Sequence[str]] = None,
) -> Tuple[str, ModifiedImageMap, List[CombinedResult], List[str]]:
    """
    Demonstrates how to integrate fuzzy matching and box correction
    into your existing loop before calling 'process_ocr_results'.
    """
    # If you don't have a big list of words, you can pass in your entire recognized text
    # as e.g. `full_text_candidates = full_text.split()`.
    if full_text_candidates is None:
        full_text_candidates = []

    updated_genders: List[str] = []

    for (snippet_text, snippet_box), snippet_conf in zip(
        all_ocr_results, all_ocr_confidences
    ):
        # 1) Fuzzy match the snippet_text against a global list or full-image words
        best_match, ratio = fuzzy_match_snippet(
            snippet_text, full_text_candidates, threshold=0.7
        )

        corrected_text = snippet_text  # default is the original
        corrected_box = snippet_box

        # 2) If the best_match is significantly better, use it
        #    Let's say ratio > 0.85 is considered a "strong" match
        if best_match and ratio > 0.85 and len(best_match) > len(snippet_text):
            logger.info(
                f"Fuzzy corrected '{snippet_text}' -> '{best_match}' (ratio={ratio:.2f})"
            )
            corrected_text = best_match

            # 3) Adjust bounding box if the new text is longer
            corrected_box = correct_box_for_new_text(
                blurred_image_path,
                snippet_box,
                old_text=snippet_text,
                new_text=corrected_text,
                extension_margin=15,
            )

        # 4) Now feed the (corrected) text + box to your NER and anonymization logic
        blurred_image_path, modified_images_map, combined_results, genders = (
            process_ocr_results(
                cast(str, blurred_image_path),
                corrected_text,
                corrected_box,
                snippet_conf,
                combined_results,
                names_detected,
                device,
                modified_images_map,
                combined_boxes,
                first_name_box,
                last_name_box,
            )
        )
        updated_genders.extend(genders)

    return (
        cast(str, blurred_image_path),
        modified_images_map,
        combined_results,
        updated_genders,
    )


def modify_image_for_name(
    image_path: str | Path, phrase_box: BoundingBox, combined_boxes: List[BoundingBox]
) -> Tuple[Path, BoundingBox]:
    image_obj = cast(object, cv2.imread(str(image_path)))
    image = cast(npt.NDArray[np.uint8], image_obj)
    _image_height, image_width, _ = image.shape
    last_name_box = find_or_create_close_box(phrase_box, combined_boxes, image_width)

    temp_dir, _base_dir, _csv_dir = create_temp_directory()
    temp_image_path = Path(temp_dir) / f"{uuid.uuid4()}.jpg"
    cv2.imwrite(str(temp_image_path), image)

    return blur_function(temp_image_path, phrase_box), last_name_box


def split_and_check(phrase: str) -> List[DetectedEntity]:
    # Initialize an empty list to store entities
    all_entities: List[DetectedEntity] = []

    # Get entities from spaCy
    spacy_entities = spacy_NER_German(phrase)
    if spacy_entities:
        all_entities.extend(
            [(entity[0], "PER") for entity in spacy_entities if entity[3] == "PER"]
        )

    # Get entities from Flair
    flair_entities = flair_NER_German(phrase)
    if flair_entities:
        flair_entities_typed = cast(Sequence[_FlairEntity], flair_entities)
        all_entities.extend(
            [
                (entity.text, "PER")
                for entity in flair_entities_typed
                if entity.tag == "PER"
            ]
        )

    if all_entities:
        return all_entities

    # If no entities found, try with parts of the phrase
    parts = [
        phrase[:3],
        phrase[-3:],
        phrase[:4] + phrase[-4:],
        phrase[:5] + phrase[-5:],
        phrase[:6] + phrase[-6:],
    ]

    for part in parts:
        spacy_entities = spacy_NER_German(part)
        if spacy_entities:
            entities = [
                (entity[0], "PER") for entity in spacy_entities if entity[3] == "PER"
            ]
            if entities:
                return entities

        flair_entities = flair_NER_German(part)
        if flair_entities:
            flair_entities_typed = cast(Sequence[_FlairEntity], flair_entities)
            entities = [
                (entity.text, "PER")
                for entity in flair_entities_typed
                if entity.tag == "PER"
            ]
            if entities:
                return entities

    return []
