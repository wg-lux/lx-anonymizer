from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytesseract
from PIL import Image

from .custom_logger import logger
from .ocr_preprocessing import preprocess_image


def pyramid_ocr(
    image: Image.Image,
    scales: Optional[List[float]] = None,
    preprocessing_methods: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    """
    Process an image at multiple scales and combine results for best OCR output.

    Parameters:
        image: PIL.Image - Input image
        scales: List[float] - List of scaling factors (e.g., [0.5, 1.0, 1.5, 2.0])
        preprocessing_methods: List[str] - List of preprocessing methods to apply

    Returns:
        str - Combined OCR text
        Dict - Dictionary with metadata about OCR process
    """
    if scales is None:
        scales = [0.8, 1.0, 1.5, 2.0]

    if preprocessing_methods is None:
        preprocessing_methods = ["grayscale", "contrast", "sharpen"]

    results = []
    confidences = []
    metadata: Dict[str, Any] = {"scale_results": {}}

    for scale in scales:
        logger.info(f"Processing image at scale {scale}")
        width, height = image.size
        new_size = (int(width * scale), int(height * scale))

        # Resize image to current scale
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Apply preprocessing
        processed_image = preprocess_image(resized_image, preprocessing_methods)

        # Apply Tesseract OCR with optimized parameters
        try:
            # Use a page segmentation mode appropriate for the scale
            # PSM 6: Assume a single uniform block of text
            # PSM 3: Fully automatic page segmentation, but no OSD
            psm = 6 if scale <= 1.0 else 3
            config = f"--oem 1 --psm {psm}"

            # Extract text and data for confidence calculation
            text = pytesseract.image_to_string(processed_image, config=config)
            data = pytesseract.image_to_data(
                processed_image, output_type=pytesseract.Output.DICT, config=config
            )

            # Calculate average confidence, excluding negative values
            confidences_list = [
                float(conf) for conf in data["conf"] if conf != "-1" and conf.strip()
            ]
            avg_confidence = (
                sum(confidences_list) / len(confidences_list) if confidences_list else 0
            )

            # Store result for this scale
            results.append(text)
            confidences.append(avg_confidence)

            # Add to metadata
            metadata["scale_results"][str(scale)] = {
                "confidence": avg_confidence,
                "text_length": len(text),
                "word_count": len(text.split()),
            }

            logger.debug(
                f"Scale {scale} - Confidence: {avg_confidence:.2f}, Text length: {len(text)}"
            )

        except Exception as e:
            logger.error(f"OCR processing failed at scale {scale}: {str(e)}")
            results.append("")
            confidences.append(0)
            metadata["scale_results"][str(scale)] = {"error": str(e)}

    # Choose best scale based on confidence
    best_index = confidences.index(max(confidences)) if any(confidences) else 0
    best_scale = scales[best_index]
    best_text = results[best_index]

    logger.info(
        f"Best scale: {best_scale} with confidence {confidences[best_index]:.2f}"
    )

    # Add summary to metadata
    metadata["selected_scale"] = best_scale
    metadata["selected_confidence"] = confidences[best_index]
    metadata["all_confidences"] = dict(zip(map(str, scales), confidences))

    return best_text, metadata


def segment_and_ocr(
    image: Image.Image,
    segmentation_mode: str = "blocks",
    preprocessing_methods: Optional[List[str]] = None,
) -> str:
    """
    Segment the image into logical blocks and apply OCR to each segment.

    Different segmentation modes:
    - 'blocks': Use Tesseract to find text blocks
    - 'table': Detect and handle tables differently
    - 'layout': Use more advanced layout analysis

    Parameters:
        image: PIL.Image - Input image
        segmentation_mode: str - How to segment the image
        preprocessing_methods: List[str] - List of preprocessing methods

    Returns:
        str - Combined OCR text from all segments
    """
    if preprocessing_methods is None:
        preprocessing_methods = ["grayscale", "contrast"]

    # Apply basic preprocessing
    processed_image = preprocess_image(image, preprocessing_methods)

    # Get image dimensions
    width, height = processed_image.size

    if segmentation_mode == "blocks":
        # Use Tesseract to find text blocks
        data = pytesseract.image_to_data(
            processed_image, output_type=pytesseract.Output.DICT
        )

        # Group words into blocks based on block_num
        blocks: Dict[int, List[Dict[str, Any]]] = {}
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                block_num = data["block_num"][i]
                if block_num not in blocks:
                    blocks[block_num] = []
                blocks[block_num].append(
                    {
                        "text": data["text"][i],
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    }
                )

        # Sort blocks by vertical position (top to bottom)
        sorted_blocks = sorted(
            blocks.values(), key=lambda block: min(word["top"] for word in block)
        )

        # Extract text from each block
        text_blocks = []
        for block in sorted_blocks:
            block_text = " ".join(word["text"] for word in block)
            text_blocks.append(block_text)

        return "\n".join(text_blocks)

    elif segmentation_mode == "table":
        # Convert to numpy array for OpenCV operations
        import cv2

        img_np = np.array(processed_image)

        # Convert to grayscale if needed
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Detect lines for table structure
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(img_np, cv2.MORPH_OPEN, vertical_kernel)

        # Combine lines to get table structure
        table_structure = cv2.add(horizontal_lines, vertical_lines)

        # Find contours of cells
        contours, _ = cv2.findContours(
            table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Process each cell
        cell_texts = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cell_image = processed_image.crop((x, y, x + w, y + h))
            cell_text = pytesseract.image_to_string(cell_image)
            cell_texts.append(cell_text.strip())

        return "\n".join(cell_texts)

    elif segmentation_mode == "layout":
        # Simplified layout analysis - divide into rows
        row_height = height // 10  # Divide image into 10 rows

        row_texts = []
        for i in range(10):
            y_start = i * row_height
            y_end = (i + 1) * row_height

            # Skip empty rows
            if y_end > height:
                continue

            row_image = processed_image.crop((0, y_start, width, y_end))
            row_text = pytesseract.image_to_string(row_image)

            if row_text.strip():
                row_texts.append(row_text.strip())

        return "\n".join(row_texts)

    else:
        # Fallback - process entire image
        return pytesseract.image_to_string(processed_image)


def adaptive_multi_engine_ocr(image: Image.Image) -> str:
    """
    Adaptively choose and apply the best OCR engine based on image characteristics.

    Parameters:
        image: PIL.Image - Input image

    Returns:
        str - OCR result from the best engine
    """
    # Analyze image to determine characteristics
    width, height = image.size
    is_high_res = width > 2000 or height > 2000

    # Convert to grayscale for analysis
    gray_img = image.convert("L")
    img_array = np.array(gray_img)

    # Check if image has low contrast
    min_val, max_val = np.min(img_array), np.max(img_array)
    contrast_ratio = (max_val - min_val) / 255
    is_low_contrast = contrast_ratio < 0.3

    # Check noise level (standard deviation in small patches)
    noise_level = np.std(img_array)
    is_noisy = noise_level > 20

    logger.info(
        f"Image characteristics - High res: {is_high_res}, "
        f"Low contrast: {is_low_contrast}, Noisy: {is_noisy}"
    )

    # Choose best OCR approach based on characteristics
    if is_high_res:
        logger.info("High resolution image detected, using pyramid OCR")
        text, _ = pyramid_ocr(image, scales=[0.5, 0.75, 1.0])
        return text

    if is_low_contrast:
        logger.info("Low contrast image detected, applying contrast enhancement")
        from .ocr_preprocessing import preprocess_image

        image = preprocess_image(image, methods=["grayscale", "contrast", "threshold"])

    if is_noisy:
        logger.info("Noisy image detected, applying denoising")
        from .ocr_preprocessing import preprocess_image

        image = preprocess_image(image, methods=["grayscale", "denoise"])

    # Default: try both Tesseract and TrOCR and pick the best result
    try:
        tesseract_text = pytesseract.image_to_string(image)
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
        tesseract_text = ""

    try:
        from .ocr import trocr_full_image_ocr

        trocr_text = trocr_full_image_ocr(image)
    except Exception as e:
        logger.error(f"TrOCR failed: {str(e)}")
        trocr_text = ""

    # Choose result with more content (basic heuristic)
    if len(trocr_text.strip()) > len(tesseract_text.strip()):
        logger.info("TrOCR produced better results (more content)")
        return trocr_text
    else:
        logger.info("Tesseract produced better results (more content)")
        return tesseract_text
        logger.info("Tesseract produced better results (more content)")
        return tesseract_text
