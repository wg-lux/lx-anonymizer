import cv2
import pytesseract
from pytesseract import Output
import json
from .box_operations import extend_boxes_if_needed
from .custom_logger import get_logger

logger=get_logger(__name__)

def tesseract_text_detection(image_path, min_confidence=0.5, width=320, height=320):
    """
    Detects text at word level using Tesseract OCR.
    """
    # Load the input image
    image = cv2.imread(str(image_path))
    logger.debug("Loading tesseract text detection")
    if image is None:
        raise ValueError("Could not open or find the image.")
    
    orig = image.copy()
    (H, W) = image.shape[:2]
    
    # Resize the image and grab the new image dimensions
    image = cv2.resize(image, (width, height))
    (rH, rW) = H / float(height), W / float(width)
    
    # Configure Tesseract for word-level detection
    custom_config = r'--oem 3 --psm 11'  # PSM 11 for sparse text with OEM 3 (default)
    
    # Detecting text using Tesseract with word-level configuration
    results = pytesseract.image_to_data(
        image, 
        output_type=Output.DICT,
        config=custom_config
    )
    
    output_boxes = []
    output_confidences = []
    
    # Add size filtering parameters
    min_width = 15  # Minimum width for a word
    max_width = int(W * 0.2)  # Maximum width (20% of image width)
    min_height = 8  # Minimum height for a word
    max_height = int(H * 0.1)  # Maximum height (10% of image height)
    aspect_ratio_threshold = 10.0  # Maximum width/height ratio
    
    # Loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
        # Extract the bounding box coordinates and confidence
        x, y, w, h = (results["left"][i], results["top"][i],
                      results["width"][i], results["height"][i])
        conf = float(results["conf"][i])
        text = results["text"][i].strip()
        
        # Filter out weak detections and empty text
        if (conf > min_confidence and 
            text and  # Check if text is not empty
            not text.isspace() and  # Check if text is not just whitespace
            w >= min_width and w <= max_width and  # Width constraints
            h >= min_height and h <= max_height and  # Height constraints
            w/h <= aspect_ratio_threshold):  # Aspect ratio check
            
            # Scale the bounding box coordinates back to the original image size
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)
            
            # Additional check for minimum box area
            if (endX - startX) * (endY - startY) >= 100:  # Minimum area of 100 pixels
                output_boxes.append((startX, startY, endX, endY))
                output_confidences.append({
                    "startX": startX,
                    "startY": startY,
                    "endX": endX,
                    "endY": endY,
                    "confidence": conf,
                    "text": text
                })
    
    # Merge very close boxes that might be parts of the same word
    output_boxes = merge_close_boxes(output_boxes)
    
    # Sort boxes with a smaller vertical threshold
    output_boxes = sort_boxes(output_boxes, vertical_threshold=5)
    
    # Extend boxes with minimal margins
    output_boxes = extend_boxes_if_needed(orig, output_boxes, extension_margin=2)
    
    logger.info(f"Tesseract text detection complete. Found {len(output_boxes)} words.")
    return output_boxes, json.dumps(output_confidences)

def merge_close_boxes(boxes, horizontal_threshold=8):
    """Merge boxes that are horizontally very close and likely part of the same word."""
    if not boxes:
        return boxes
    
    merged = []
    current_box = list(boxes[0])
    
    for box in boxes[1:]:
        # Check if boxes are close horizontally and on the same line
        if (abs(box[0] - current_box[2]) < horizontal_threshold and 
            abs(box[1] - current_box[1]) < 5):  # Vertical alignment threshold
            # Merge boxes
            current_box[2] = box[2]  # Extend to end of next box
            current_box[3] = max(current_box[3], box[3])  # Take max height
        else:
            merged.append(tuple(current_box))
            current_box = list(box)
    
    merged.append(tuple(current_box))
    return merged

def sort_boxes(boxes, vertical_threshold=5):
    """Sort boxes by vertical position then horizontal position."""
    boxes.sort(key=lambda b: (round(b[1] / vertical_threshold), b[0]))
    return boxes