import uuid
from custom_logger import get_logger, configure_global_logger
from pdf_operations import merge_pdfs, convert_image_to_pdf
from directory_setup import create_temp_directory, create_results_directory
from pathlib import Path
from image_loader import get_image_paths
from image_processor import process_image
from gpu_management import clear_gpu_memory



'''
Main Function

Run by calling main(image_or_pdf_path, east_path=None, device="olympus_cv_1500", validation=False, min_confidence=0.5, width=320, height=320)

Or in the directory python main.py -i /path/to/image.png -east /path/to/east_detector.pb -d olympus_cv_1500 -c 0.5 -w 320 -e 320

Disclaimer: The Pipeline is easier to run with the Agl-Anonymizer-Flake API.

Parameters:
- image_or_pdf_path: str
    The path to the image or PDF file to process.
- east_path: str
    The path to the EAST text detector model. Default is None.
- device: str
    The device name to set the correct text settings. Default is "olympus_cv_1500".
- validation: bool
    A boolean value representing if validation through the AGL-Validator is required. Default is False.
- min_confidence: float
    The minimum probability required to inspect a region. Default is 0.5.
- width: int
    The resized image width (should be a multiple of 32). Default is 320.
- height: int
    The resized image height (should be a multiple of 32). Default is 320.

Returns:
- str
    The path to the output image or PDF file.
- tuple
    A tuple containing the path to the output image or PDF file, the result, and the original image or PDF file path.

Directory Setup:

You can change the default path for the base directory as well as the temporary directory 
by changing the values specified in the directory_setup.py file.
'''
from contextlib import contextmanager



@contextmanager
def temp_directory_manager():
    logger = get_logger(__name__)  # Use global logger
    temp_dir, base_dir, csv_dir = create_temp_directory()
    try:
        yield temp_dir, base_dir, csv_dir
    finally:
        temp_dir_path = Path(temp_dir)
        if temp_dir_path.exists() and temp_dir_path.is_dir():
            for file in temp_dir_path.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file}: {e}")

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


def main(image_or_pdf_path, east_path=None, device="olympus_cv_1500", validation=False, min_confidence=0.5, width=320, height=320):
    logger = get_logger(__name__)  # Use global logger
    try:
        clear_gpu_memory()
        with temp_directory_manager() as (temp_dir, base_dir, csv_dir):
            results_dir = create_results_directory()
            image_paths = get_image_paths(Path(image_or_pdf_path), Path(temp_dir))
            
            processed_pdf_paths = []
            anonymization_data = None
            
            for img_path in image_paths:
                success = False
                try:
                    # Try with original path
                    if not Path(img_path).exists():
                        raise ImageProcessingError(f"Image path does not exist: {img_path}")
                    
                    processed_image_path, result = process_image(
                        img_path, east_path, device, min_confidence, width, height, 
                        Path(results_dir), Path(temp_dir)
                    )
                    success = True
                except (ImageProcessingError, RuntimeError, ValueError) as e:
                    # Try with local path
                    logger.info(f"Error processing with original path: {e}, trying local path")
                    try:
                        root_dir = Path(__file__).resolve().parent
                        local_img_path = root_dir / img_path
                        logger.info(f"Trying local path: {local_img_path}")
                        
                        if not local_img_path.exists():
                            raise ImageProcessingError(f"Local image path does not exist: {local_img_path}")
                        
                        processed_image_path, anonymization_data = process_image(
                            local_img_path, east_path, device, min_confidence, width, height,
                            Path(results_dir), Path(temp_dir)
                        )
                        success = True
                    except Exception as local_err:
                        logger.error(f"Error processing with local path: {local_err}")
                        raise ImageProcessingError(f"Failed to process image with both paths: {e}, local error: {local_err}")
                
                if success:
                    if str(image_or_pdf_path).lower().endswith('.pdf'):
                        temp_pdf_path = Path(temp_dir) / f"temporary_pdf_{uuid.uuid4()}.pdf"
                        convert_image_to_pdf(processed_image_path, temp_pdf_path)
                        processed_pdf_paths.append(temp_pdf_path)
                    else:
                        processed_pdf_paths.append(processed_image_path)

            if not processed_pdf_paths:
                raise ImageProcessingError("No processed images were generated.")

            if str(image_or_pdf_path).lower().endswith('.pdf'):
                final_pdf_path = Path(results_dir) / f"final_document_{uuid.uuid4()}.pdf"
                merge_pdfs(processed_pdf_paths, final_pdf_path)
                output_path = final_pdf_path
            else:
                output_path = processed_pdf_paths[0]

            logger.info(f"Output Path: {output_path}")
            if not validation:
                return output_path
            else:
                return output_path, anonymization_data, image_or_pdf_path

    except Exception as e:
        raise ImageProcessingError(f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    ap.add_argument("-east", "--east", type=str, required=False, help="Path to input EAST text detector")
    ap.add_argument("-d", "--device", type=str, default="olympus_cv_1500", help="Device name to set the correct text settings")
    ap.add_argument("-V", "--validation", type=bool, default=False, help="Boolean value representing if validation through the AGL-Validator is required.")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5, help="Minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320, help="Resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320, help="Resized image height (should be multiple of 32)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = vars(ap.parse_args())
    configure_global_logger(verbose=args["verbose"])

    main(
        args["image"], 
        args["east"], 
        args["device"], 
        args["validation"], 
        args["min_confidence"], 
        args["width"], 
        args["height"], 
    )
