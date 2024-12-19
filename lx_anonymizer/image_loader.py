from pathlib import Path
import fitz
import cv2
import uuid
from custom_logger import get_logger
from image_reassembly import reassemble_image
from pipeline_manager import process_images_with_OCR_and_NER
from gpu_management import clear_gpu_memory


logger = get_logger(__name__)


def get_image_paths(image_or_pdf_path: Path, temp_dir: Path):
    image_paths = []

    if not temp_dir.exists() or not temp_dir.is_dir():
        raise ValueError(f"Temporary directory {temp_dir} does not exist or is not a directory.")

    if image_or_pdf_path.suffix.lower() == '.pdf':
        try:
            doc = fitz.open(str(image_or_pdf_path))  # fitz expects a string
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                temp_img_path = Path(temp_dir) / f"page_{page_num}.png"
                pix.save(str(temp_img_path))  # fitz requires a string path
                image_paths.append(temp_img_path)
        except Exception as e:
            raise RuntimeError(f"Error processing PDF {image_or_pdf_path}: {e}")
    else:
        if not image_or_pdf_path.exists():
            raise FileNotFoundError(f"The file {image_or_pdf_path} does not exist.")
        image_paths.append(image_or_pdf_path)

    return image_paths

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