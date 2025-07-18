from typing import List, Tuple
from .custom_logger import get_logger

logger = get_logger(__name__)

def create_combined_phrases(ocr_texts_with_boxes):
    combined_phrases = []
    combined_box = None
    phrase = ""

    for text, box in ocr_texts_with_boxes:
        # Ensure box is a tuple or list of four elements
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            logger.error(f"Box must be a tuple or list of four elements: {box}")
            raise ValueError("Box must be a tuple or list of four elements")

        if not phrase:
            # Start a new phrase
            phrase = text
            combined_box = box
        else:
            # Add to existing phrase and update the box
            phrase += " " + text

            # Ensure combined_box is valid before unpacking
            if not isinstance(combined_box, (tuple, list)) or len(combined_box) != 4:
                logger.error(f"Combined box is not in the correct format: {combined_box}")
                raise ValueError("Combined box is not in the correct format")

            startX, startY, endX, endY = combined_box
            new_startX, new_startY, new_endX, new_endY = box
            combined_box = (min(startX, new_startX), min(startY, new_startY),
                            max(endX, new_endX), max(endY, new_endY))

    # Append the final phrase and its box after the loop
    if phrase:
        combined_phrases.append((phrase, combined_box))
    
    logger.debug(f"Combined {len(ocr_texts_with_boxes)} phrases into {len(combined_phrases)}")

    return combined_phrases
