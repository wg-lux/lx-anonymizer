from hezar.models import Model
from hezar.utils import load_image, draw_boxes
from custom_logger import get_logger
from pathlib import Path
import cv2
import json
import numpy as np
from box_operations import extend_boxes_if_needed
import torch

logger = get_logger(__name__)

def craft_text_detection(image_path, min_confidence=0.5, width=320, height=320):
    try:
        # Load the original image for scaling calculations
        orig = cv2.imread(str(image_path))
        if orig is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
            
        (origH, origW) = orig.shape[:2]
        rW = origW / float(width)
        rH = origH / float(height)

        logger.info("Loading CRAFT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model.load("hezarai/CRAFT", device=device)
        image = load_image(str(image_path))
        
        logger.info("Running CRAFT text detection...")
        # Add parameters to encourage word-level detection
        outputs = model.predict(
            image,
            text_threshold=0.7,    # Higher threshold for more confident word detection
            link_threshold=0.3,    # Lower threshold to better separate words
            low_text=0.4,         # Balance between detection and separation
            poly=False,           # Use rectangles for simpler processing
            word_level=True       # Explicit word-level detection
        )
        logger.info("Detection complete.")

        # Add size filtering parameters
        min_width = 15  # Minimum width for a word
        max_width = int(origW * 0.2)  # Maximum 20% of image width
        min_height = 8  # Minimum height for a word
        max_height = int(origH * 0.1)  # Maximum 10% of image height
        aspect_ratio_threshold = 10.0  # Maximum width/height ratio

        output_boxes = []
        output_confidences = []

        if outputs and isinstance(outputs, list) and len(outputs) > 0 and "boxes" in outputs[0]:
            boxes = outputs[0]["boxes"]
            for box in boxes:
                points = np.array(box, dtype=np.int32)
                logger.debug(f"Box shape: {points.shape}")
                
                if len(points.shape) == 1 and points.shape[0] == 4:
                    # Handle flat array [x1, y1, x2, y2]
                    x_coords = points[::2]
                    y_coords = points[1::2]
                    
                    # Calculate bounding box coordinates
                    startX = int(min(x_coords) * rW)
                    startY = int(min(y_coords) * rH)
                    endX = int(max(x_coords) * rW)
                    endY = int(max(y_coords) * rH)
                    
                    # Calculate box dimensions
                    width = endX - startX
                    height = endY - startY
                    
                    # Apply size and aspect ratio filters
                    if (min_width <= width <= max_width and 
                        min_height <= height <= max_height and 
                        width/height <= aspect_ratio_threshold):
                        
                        # Ensure coordinates are within image bounds
                        startX = max(0, min(startX, origW - 1))
                        startY = max(0, min(startY, origH - 1))
                        endX = max(0, min(endX, origW))
                        endY = max(0, min(endY, origH))
                        
                        # Only add valid boxes with sufficient area
                        if startX < endX and startY < endY and (endX - startX) * (endY - startY) >= 100:
                            output_boxes.append((startX, startY, endX, endY))
                            output_confidences.append({
                                "startX": startX,
                                "startY": startY,
                                "endX": endX,
                                "endY": endY,
                                "confidence": min_confidence
                            })

        if output_boxes:
            logger.info(f"Detected {len(output_boxes)} text regions")
            # Merge very close boxes that might be parts of the same word
            output_boxes = merge_close_boxes(output_boxes)
            # Sort with reduced vertical threshold
            output_boxes = sort_boxes(output_boxes, vertical_threshold=5)
            # Extend boxes with minimal margins
            output_boxes = extend_boxes_if_needed(orig, output_boxes, extension_margin=2)
            
        return output_boxes, json.dumps(output_confidences)
        
    except Exception as e:
        logger.error(f"Error in CRAFT text detection: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

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
    """Sort boxes by vertical position then horizontal position with tighter threshold."""
    boxes.sort(key=lambda b: (round(b[1] / vertical_threshold), b[0]))
    return boxes