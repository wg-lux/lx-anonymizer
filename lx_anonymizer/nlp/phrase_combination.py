from lx_anonymizer.runtime_types import Box as Box
from lx_anonymizer.runtime_types import OcrResult
from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)

OcrTextWithBox = tuple[str, Box | list[int]]


def _normalize_box(box: Box | list[int]) -> Box:
    if len(box) != 4:
        logger.error("Box must be a tuple or list of four elements: %s", box)
        raise ValueError("Box must be a tuple or list of four elements")
    start_x, start_y, end_x, end_y = box
    return (int(start_x), int(start_y), int(end_x), int(end_y))


def create_combined_phrases(
    ocr_texts_with_boxes: list[OcrTextWithBox],
) -> list[OcrResult]:
    combined_phrases: list[OcrResult] = []
    combined_box: Box | None = None
    phrase = ""

    for text, box in ocr_texts_with_boxes:
        current_box = _normalize_box(box)

        if not phrase:
            # Start a new phrase
            phrase = text
            combined_box = current_box
        else:
            # Add to existing phrase and update the box
            phrase += " " + text

            # Ensure combined_box is valid before unpacking
            if combined_box is None:
                logger.error(
                    f"Combined box is not in the correct format: {combined_box}"
                )
                raise ValueError("Combined box is not in the correct format")

            startX, startY, endX, endY = combined_box
            new_startX, new_startY, new_endX, new_endY = current_box
            combined_box = (
                min(startX, new_startX),
                min(startY, new_startY),
                max(endX, new_endX),
                max(endY, new_endY),
            )

    # Append the final phrase and its box after the loop
    if phrase and combined_box is not None:
        combined_phrases.append((phrase, combined_box))

    logger.debug(
        f"Combined {len(ocr_texts_with_boxes)} phrases into {len(combined_phrases)}"
    )

    return combined_phrases
