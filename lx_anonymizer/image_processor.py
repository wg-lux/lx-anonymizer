from pathlib import Path
import uuid
from .custom_logger import get_logger
from .image_reassembly import reassemble_image
from .pipeline_manager import process_images_with_OCR_and_NER
from .gpu_management import clear_gpu_memory
import cv2

logger = get_logger(__name__)

def process_image(
    img_path, 
    east_path, 
    device, 
    min_confidence, 
    width, 
    height, 
    results_dir, 
    temp_dir, 
    text_extracted=None,
    skip_blur=False, 
    skip_reassembly=False,
    disable_llm=False
):
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
    
    # If we have pre-extracted text and LLM is enabled, analyze it
    if text_extracted and not disable_llm:
        logger.info("Analyzing extracted text with LLM")
        csv_path = temp_dir / f"text_analysis_{uuid.uuid4()}.csv"
        text_results, _ = analyze_text_with_phi4(
            text_extracted, 
            csv_path=csv_path,
            image_path=img_path
        )
    
    # If we're skipping the blur operations but want analysis, use LLM
    if skip_blur and not disable_llm:
        logger.info("Skipping blur operations, performing analysis only")
        
        # Use LLM to analyze the image directly
        csv_path = temp_dir / f"image_analysis_{uuid.uuid4()}.csv"
        llm_results, llm_csv_path = analyze_full_image_with_context(img_path, csv_path, temp_dir)
        
        # Combine analysis results if we have text
        if text_extracted and not disable_llm:
            combined_results = {
                "image_analysis": llm_results,
                "text_analysis": text_results
            }
        else:
            combined_results = {"image_analysis": llm_results}
            
        # Return the original image path and the analysis results
        return img_path, combined_results
    
    # Normal processing path with OCR, NER, and optional blurring
    modified_images_map, result = process_images_with_OCR_and_NER(
        Path(img_path), 
        east_path=east_path,
        device=device,
        min_confidence=min_confidence,
        width=width,
        height=height,
        skip_blur=skip_blur,
        skip_reassembly=skip_reassembly
    )
    
    # Get the processed image path
    if modified_images_map:
        # Get the last modified image (the final result)
        last_key = list(modified_images_map.keys())[-1]
        processed_image_path = modified_images_map[last_key]
    else:
        # If no modification was done, return the original
        processed_image_path = img_path
        
    # If we have a processed path, copy it to the results directory
    if processed_image_path != img_path:
        file_extension = os.path.splitext(processed_image_path)[1]
        result_path = results_dir / f"processed_{uuid.uuid4()}{file_extension}"
        try:
            if cv2.imread(str(processed_image_path)) is None:
                logger.error(f"Failed to read image: {processed_image_path}")
            else:
                # Copy the image to the results directory
                cv2.imwrite(
                    str(result_path), 
                    cv2.imread(str(processed_image_path))
                )
                logger.debug(f"Image copied successfully to: {result_path}")
        except Exception as e:
            logger.error(f"Failed to save image to results with error {e}")
                
        processed_image_path = result_path
    
    return processed_image_path, result

def resize_image(image_path: Path, max_width=1024, max_height=1024):
    try:
        if cv2.imread(str(image_path)) is None:
            logger.error(f"Failed to read image: {image_path}")
            # Handle the error or return gracefully
        else:
            image=cv2.imread(str(image_path))
    except Exception as e:
        logger.error(f"Error copying processed image: {e}")
        # Handle the exception or return gracefully    if image is None:
        logger.error(f"Unable to read image for resizing: {image_path}")
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
            if cv2.imwrite(str(image_path)) is None:
                logger.error("Failed to write image")
            else:
                cv2.imwrite(str(image_path), resized_image)  # Saving image
                logger.debug(f"Image saved successfully: {image_path}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")

        logger.debug(f"Image resized to {new_size}")