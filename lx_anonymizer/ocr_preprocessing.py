from PIL import ImageOps, Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from typing import List, Tuple, Union, Optional


def preprocess_image(image: Image.Image, methods: Optional[List[str]] = None) -> Image.Image:
    """
    Preprocess the image for OCR with multiple enhancement techniques.
    
    Parameters:
        image: PIL.Image - The image to preprocess
        methods: list - List of preprocessing methods to apply
                Options: 'grayscale', 'denoise', 'contrast', 
                        'threshold', 'sharpen', 'resize',
                        'autocontrast', 'deskew'
                        
    Returns:
        PIL.Image - Processed image
    """
    if methods is None:
        # Default preprocessing pipeline
        methods = ['grayscale', 'denoise', 'contrast', 'sharpen']
    
    # Apply selected preprocessing methods
    for method in methods:
        if method == 'grayscale':
            # Convert to grayscale
            if image.mode != "L":
                image = image.convert("L")
                
        elif method == 'denoise':
            # Apply denoising
            image_cv = np.array(image)
            if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
                # Color image
                image_cv = cv2.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)
            else:
                # Grayscale image
                image_cv = cv2.fastNlMeansDenoising(image_cv, None, 30, 7, 21)
            image = Image.fromarray(image_cv)
            
        elif method == 'contrast':
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(image)
            image = contrast_enhancer.enhance(1.5)  # Increase contrast by 50%

            # Enhance brightness
            brightness_enhancer = ImageEnhance.Brightness(image)
            image = brightness_enhancer.enhance(1.2)  # Increase brightness by 20%
            
        elif method == 'threshold':
            # Apply adaptive thresholding
            image_cv = np.array(image.convert('L'))
            image_cv = cv2.adaptiveThreshold(
                image_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            image = Image.fromarray(image_cv)
            
        elif method == 'sharpen':
            # Sharpen the image
            image = image.filter(ImageFilter.SHARPEN)
            
        elif method == 'resize':
            # Resize for optimal OCR (300-400 DPI)
            width, height = image.size
            scale_factor = 2.0  # Adjust based on your image resolution needs
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)  # Changed here
            
        elif method == 'autocontrast':
            # Auto-adjust contrast
            image = ImageOps.autocontrast(image, cutoff=0.5)
            
        elif method == 'deskew':
            # Deskew image for better OCR accuracy
            image_cv = np.array(image.convert('L'))
            coords = np.column_stack(np.where(image_cv > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            # Correct the angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Rotate the image to deskew it
            (h, w) = image_cv.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            image = Image.fromarray(rotated)
    
    return image


def expand_roi(startX: int, startY: int, endX: int, endY: int, margin: int, 
              image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Expand region of interest by a margin, ensuring it stays within image boundaries.
    
    Parameters:
        startX, startY, endX, endY: int - Original ROI coordinates
        margin: int - Expansion margin in pixels
        image_shape: tuple - (height, width) of the image
        
    Returns:
        tuple - (startX, startY, endX, endY) expanded coordinates
    """
    height, width = image_shape[:2]
    return (
        max(startX - margin, 0),
        max(startY - margin, 0),
        min(endX + margin, width),
        min(endY + margin, height)
    )


def optimize_image_for_medical_text(image: Image.Image) -> Image.Image:
    """
    Apply specific optimizations for medical document OCR.
    
    Medical documents often have standard formats with specific challenges:
    - Small font sizes for patient details
    - Mixed fonts (serif for headers, sans-serif for content)
    - Stamps, signatures and handwritten notes
    - Table structures with thin lines
    
    Parameters:
        image: PIL.Image - Input image
        
    Returns:
        PIL.Image - Optimized image
    """
    # Initial grayscale conversion
    image = image.convert('L')
    
    # Enhance contrast to make text more visible
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)  # Higher contrast for medical documents
    
    # Apply sharpening multiple times to enhance text edges
    for _ in range(2):
        image = image.filter(ImageFilter.SHARPEN)
    
    # Convert to numpy for OpenCV operations
    image_cv = np.array(image)
    
    # Apply Otsu's thresholding to separate text from background
    _, binary = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small noise using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Enhance text edges
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return Image.fromarray(binary)


def detect_table_structure(image: Image.Image) -> List[Tuple[int, int, int, int]]:
    """
    Detect table structures in the image and return bounding boxes.
    
    Medical reports often contain tabular data that needs special handling.
    
    Parameters:
        image: PIL.Image - Input image
        
    Returns:
        List of (x1, y1, x2, y2) tuples representing table boundaries
    """
    # Convert to grayscale
    gray = image.convert('L')
    img_np = np.array(gray)
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    
    # Detect horizontal lines
    horizontal = cv2.erode(img_np, horizontal_kernel, iterations=3)
    horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=3)
    
    # Detect vertical lines
    vertical = cv2.erode(img_np, vertical_kernel, iterations=3)
    vertical = cv2.dilate(vertical, vertical_kernel, iterations=3)
    
    # Combine horizontal and vertical lines
    table_mask = cv2.bitwise_or(horizontal, vertical)
    
    # Find contours of table structures
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes
    table_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small boxes
        if w > 50 and h > 50:
            table_boxes.append((x, y, x+w, y+h))
    
    return table_boxes


def preprocess_for_handwriting(image: Image.Image) -> Image.Image:
    """
    Special preprocessing for images that may contain handwritten text.
    
    Parameters:
        image: PIL.Image - Input image
        
    Returns:
        PIL.Image - Processed image optimized for handwriting recognition
    """
    # Convert to grayscale
    gray = image.convert('L')
    
    # Apply bilateral filter to preserve edges while reducing noise
    img_np = np.array(gray)
    bilateral = cv2.bilateralFilter(img_np, 9, 75, 75)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(bilateral)
    
    # Apply slight sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return Image.fromarray(sharpened)
