import cv2
from pathlib import Path
import uuid
from .custom_logger import get_logger



# Configure logger
logger = get_logger(__name__)


def reassemble_image(modified_images_map, output_dir, id, original_image_path=None):
    """
    Reassembles an image by overlaying modified image regions (patches) on the original image.

    Args:
        modified_images_map (dict): A dictionary mapping tuple keys representing bounding box coordinates 
                                    and original image path to the corresponding modified image paths.
        output_dir (str or Path): The directory where the reassembled image will be saved.
        id (str): An identifier used in the naming of the reassembled image file.
        original_image_path (str or Path): The path to the original image that will be reassembled.

    Returns:
        Path: The path to the saved reassembled image, or None if there was an error.
    """
    
    # Convert paths to Path objects if they aren't already
    output_dir = Path(output_dir)
    original_image_path = Path(original_image_path)
    
    # Load the original image only once
    logger.info(f"Loading original image from {str(original_image_path)}.")
    curr_image = cv2.imread(str(original_image_path))  # cv2.imread expects a string path
    if curr_image is None:
        logger.error(f"Could not load original image from {str(original_image_path)}.")
        return None

    # Iterate through the modified images and overlay them onto the original image
    for ((box_key, original_image_path), modified_image_path) in modified_images_map.items():
        logger.info(f"Processing box {box_key} with modified image {str(modified_image_path)}.")
        modified_image = cv2.imread(str(modified_image_path))
        if modified_image is None:
            logger.warning(f"Could not load modified image from {str(modified_image_path)}. Skipping this modification.")
            continue

        # Extract bounding box coordinates
        startX, startY, endX, endY = map(int, box_key[0].split(','))

        # Ensure the bounding box fits within the original image dimensions
        startX = max(min(startX, curr_image.shape[1] - modified_image.shape[1]), 0)
        startY = max(min(startY, curr_image.shape[0] - modified_image.shape[0]), 0)
        endX = min(startX + modified_image.shape[1], curr_image.shape[1])
        endY = min(startY + modified_image.shape[0], curr_image.shape[0])

        overlay_width = endX - startX
        overlay_height = endY - startY

        # Check if calculated dimensions are valid
        if overlay_width <= 0 or overlay_height <= 0:
            logger.warning(f"Invalid overlay dimensions for box {box_key}, skipping this modification.")
            continue

        # Overlay the modified image onto the current image within the effective dimensions
        logger.debug(f"Overlaying modified image onto original image at coordinates: ({startX}, {startY}, {endX}, {endY}).")
        curr_image[startY:endY, startX:endX] = modified_image[0:overlay_height, 0:overlay_width]

    # Create output directory if it doesn't exist
    if not output_dir.exists():
        logger.info(f"Output directory {output_dir} does not exist. Creating it.")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save the final reassembled image
    final_image_path = Path(output_dir) / f"reassembled_image_{id}_{uuid.uuid4()}.jpg"
    success = cv2.imwrite(str(final_image_path), curr_image)  # cv2.imwrite expects a string path

    if success:
        logger.info(f"Reassembled image successfully saved to: {final_image_path}")
    else:
        logger.error(f"Failed to save reassembled image to {final_image_path}.")
        return None

    return final_image_path

# Example usage
if __name__ == "__main__":
    modified_images_map = {
        # Example structure, populate with actual data
        (('100,100,200,200', 'original_image_path.jpg'), 'modified_image_path1.jpg'): 'modified_image_path1.jpg',
        (('150,150,250,250', 'original_image_path.jpg'), 'modified_image_path2.jpg'): 'modified_image_path2.jpg'
    }
    output_dir = 'output_directory'
    id = 'example_id'
    original_image_path = 'original_image_path.jpg'

    reassemble_image(modified_images_map, output_dir, id, original_image_path)
