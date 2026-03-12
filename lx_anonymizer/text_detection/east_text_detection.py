# import the necessary packages
import json
import ssl
import urllib.request
from pathlib import Path

import certifi
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression  # type: ignore[import-untyped]

from lx_anonymizer.region_processing.box_operations import extend_boxes_if_needed
from lx_anonymizer.setup.custom_logger import get_logger
from lx_anonymizer.setup.directory_setup import create_model_directory

logger = get_logger(__name__)

"""
This module implements argman's EAST Text Detection in a function. 
The model that's being used is specified by the east_path variable. 
The text region detection provides the starting point for the anonymization pipeline.

TO-DO

- EAST Re-Implementation for training purposes
"""

# Define the URL to download the frozen EAST model from GitHub
MODEL_URL = "https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/raw/master/frozen_east_text_detection.pb"


def _default_east_model_path() -> Path:
    return Path(create_model_directory()) / "frozen_east_text_detection.pb"


def _ensure_east_model(east_path: str | Path | None = None) -> Path:
    resolved_path = (
        Path(east_path) if east_path is not None else _default_east_model_path()
    )
    if resolved_path.exists():
        return resolved_path

    logger.info("Model not found. Downloading EAST model to %s...", resolved_path)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(MODEL_URL, context=ssl_context) as response:
            resolved_path.write_bytes(response.read())
        logger.debug("Download complete.")
        return resolved_path
    except Exception as exc:
        logger.error("Error downloading the model: %s", exc)
        raise


def east_text_detection(
    image_path, east_path=None, min_confidence=0.6, width=320, height=320
):
    east_model_path = _ensure_east_model(east_path)
    east_path = str(east_model_path)

    logger.debug(
        f"Using EAST model at: {east_path} (size: {Path(east_path).stat().st_size} bytes)"
    )

    # Load the input image and grab the image dimensions
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    (origH, origW) = orig.shape[:2]

    # Set the new width and height and then determine the ratio in change for both the width and height
    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)

    # Resize the image and grab the new image dimensions
    image = cv2.resize(orig, (newW, newH))
    (H, W) = image.shape[:2]

    # Define the two output layer names for the EAST detector model that we are interested in
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Load the pre-trained EAST text detector
    logger.debug("[INFO] Loading EAST text detector...")
    if east_model_path.exists():
        logger.debug(f"EAST model size: {east_model_path.stat().st_size} bytes")
    else:
        logger.debug(f"EAST model not found at {east_model_path}")
        raise FileNotFoundError(f"EAST model not found at {east_model_path}")
    net = cv2.dnn.readNet(east_path)

    blob = cv2.dnn.blobFromImage(
        image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )

    # start = time.time()

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # end = time.time()

    # Grab the number of rows and columns from the scores volume
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the rows
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the columns
        for x in range(0, numCols):
            # Check if the score is above the minimum confidence
            if scoresData[x] < min_confidence:
                continue

            # Compute the offset
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Extract the rotation angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Use the geometry to derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute the starting and ending (x, y)-coordinates
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability score to the lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # Modify non-maxima suppression parameters
    boxes = non_max_suppression(
        np.array(rects),
        probs=confidences,
        overlapThresh=0.2,  # Decrease overlap threshold (default is 0.3)
    )
    output_boxes = []
    output_confidences = []
    min_width = 15
    max_width = int(origW * 0.3)  # Maximum width (30% of image width)
    min_height = 8  # Minimum height for a word box
    max_height = int(origH * 0.1)  # Maximum height (10% of image height)

    # Loop over the bounding boxes and apply size filtering
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        # Scale the bounding box coordinates
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Calculate box dimensions
        width = endX - startX
        height = endY - startY

        # Filter out boxes that are too small or too large
        if (
            min_width <= width <= max_width
            and min_height <= height <= max_height
            and width / height < 15
        ):  # Aspect ratio check to avoid long text lines
            output_boxes.append((startX, startY, endX, endY))
            output_confidences.append(
                {
                    "startX": startX,
                    "startY": startY,
                    "endX": endX,
                    "endY": endY,
                    "confidence": float(confidences[i]),
                }
            )

    vertical_threshold = max(2, int(0.03 * H))
    horizontal_threshold = max(2, int(0.03 * W))
    output_boxes = sort_boxes(output_boxes, vertical_threshold)

    # Optional: Merge very close boxes that might be parts of the same word
    output_boxes = merge_close_boxes(output_boxes, horizontal_threshold)

    # Extend boxes if needed with smaller margins
    output_boxes = extend_boxes_if_needed(orig, output_boxes, extension_margin=2)

    return output_boxes, json.dumps(output_confidences)


def merge_close_boxes(boxes, horizontal_threshold=10):
    """Merge boxes that are horizontally very close and likely part of the same word."""
    if not boxes:
        return boxes

    merged = []
    current_box = list(boxes[0])

    for box in boxes[1:]:
        if (
            abs(box[0] - current_box[2]) < horizontal_threshold  # Close horizontally
            and abs(box[1] - current_box[1]) < 5
        ):  # On same line
            # Merge boxes
            current_box[2] = box[2]  # Extend to end of next box
            current_box[3] = max(current_box[3], box[3])  # Take max height
        else:
            merged.append(tuple(current_box))
            current_box = list(box)

    merged.append(tuple(current_box))
    return merged


def sort_boxes(boxes, vertical_threshold=10):
    # Define a threshold to consider boxes on the same line
    vertical_threshold = 10

    # Sort boxes by y-coordinate, then by x-coordinate if y-coordinates are similar
    boxes.sort(key=lambda b: (round(b[1] / vertical_threshold), b[0]))

    return boxes
