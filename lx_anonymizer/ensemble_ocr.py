from PIL import Image
from .custom_logger import get_logger
from .ocr import tesseract_full_image_ocr, trocr_full_image_ocr
from .ocr_GOT import ocr as got_ocr
from .donut_ocr import donut_full_image_ocr

logger = get_logger(__name__)

def ensemble_ocr(image_input):
    """
    Perform OCR using multiple methods (Tesseract, TrOCR, GOT, and Donut) and choose the best result.
    
    Args:
        image_input: a PIL Image or file path.
        
    Returns:
        A string of the ensemble OCR result.
    """
    # Run Tesseract OCR
    tesseract_text, _ = tesseract_full_image_ocr(image_input)
    
    # Run TrOCR OCR
    trocr_text = trocr_full_image_ocr(image_input)
    
    # Run GOT OCR
    got_text = got_ocr(image_input)
    
    # Run Donut OCR
    donut_text = donut_full_image_ocr(image_input)
    
    # Log outputs
    logger.info(f"Tesseract output length: {len(tesseract_text)}")
    logger.info(f"TrOCR output length: {len(trocr_text)}")
    logger.info(f"GOT output length: {len(got_text)}")
    logger.info(f"Donut output length: {len(donut_text)}")
    
    # Simple ensemble selection: Choose the output with the maximum length.
    # (You might also use a re-ranking model or similarity score for better robustness.)
    outputs = [tesseract_text, trocr_text, got_text, donut_text]
    best_output = max(outputs, key=len)
    
    # Identify which model produced the best result
    models = ["Tesseract", "TrOCR", "GOT", "Donut"]
    best_model_index = outputs.index(best_output)
    logger.info(f"Selected output from {models[best_model_index]} (length: {len(best_output)})")
    
    return best_output
