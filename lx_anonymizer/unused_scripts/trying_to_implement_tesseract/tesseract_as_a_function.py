import cv2
import pytesseract
from pytesseract import Output

def tesseract_text_detection(image_path, min_confidence=0.5, width=320, height=320):
    """
    Detects text from an image using Tesseract OCR and EAST detector.
    Draws bounding boxes around detected text regions and displays the image.
    
    :param image_path: Path to the input image.
    :param min_confidence: Minimum confidence value to consider a detection valid.
    :param width: Resized image width for the detection model.
    :param height: Resized image height for the detection model.
    :return: List of bounding boxes (x, y, w, h) of detected text regions.
    """
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open or find the image.")
    
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Resize the image and grab the new image dimensions
    image = cv2.resize(image, (width, height))
    (rH, rW) = H / float(height), W / float(width)

    # Detecting text using Tesseract
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    output_boxes = []

    # Loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
        # Extract the bounding box coordinates and confidence
        x, y, w, h = (results["left"][i], results["top"][i], 
                    results["width"][i], results["height"][i])
        conf = float(results["conf"][i])

        # Filter out weak detections
        if conf > min_confidence:
            # Scale the bounding box coordinates back to the original image size
            (startX, startY, endX, endY) = (int(x * rW), int(y * rH), 
                                            int((x + w) * rW), int((y + h) * rH))
            output_boxes.append((startX, startY, endX - startX, endY - startY))

            # Draw the bounding box on the original image
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the output image
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return output_boxes

# if this script is the main script being run, parse command line args and run
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to input image")
    ap.add_argument("-east", "--east", type=str, required=True,
                    help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                    help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
                    help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
                    help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

