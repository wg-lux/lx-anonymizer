from pathlib import Path
import cv2
import uuid
from custom_logger import get_logger
from image_reassembly import reassemble_image
from pipeline_manager import process_images_with_OCR_and_NER
from gpu_management import clear_gpu_memory

logger = get_logger(__name__)

def process_image(image_path: Path, east_path, device, min_confidence, width, height, results_dir: Path, temp_dir: Path):
    logger.info(f"Processing file: {image_path}")
    unique_id = str(uuid.uuid4())[:8]
    id = f"image_{unique_id}"

    original_image = cv2.imread(str(image_path))  # OpenCV requires a string path
    if original_image is None:
        error_msg = f"Could not load image at {image_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        resize_image(image_path)  # Resize to manage memory
        modified_images_map, result = process_images_with_OCR_and_NER(
            image_path, east_path, device, min_confidence, width, height
        )
        logger.info("Images processed")
        logger.debug(f"Modified Images Map: {modified_images_map}")

        reassembled_image_path = reassemble_image(modified_images_map, results_dir, id, image_path)
        return reassembled_image_path, result
    except Exception as e:
        error_message = f"Error in process_image: {e}, Image Path: {image_path}"
        logger.error(error_message)
        raise RuntimeError(error_message)
    finally:
        clear_gpu_memory()
        
def resize_image(image_path: Path, max_width=1024, max_height=1024):
    image = cv2.imread(str(image_path))  # OpenCV expects a string
    if image is None:
        logger.error(f"Unable to read image for resizing: {image_path}")
        return
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(image_path), resized_image)  # Saving image
        logger.debug(f"Image resized to {new_size}")