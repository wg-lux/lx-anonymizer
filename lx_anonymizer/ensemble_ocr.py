from PIL import Image
import pytesseract
import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from spellchecker import SpellChecker
import re
from .ocr import trocr_full_image_ocr
from .donut_ocr import donut_full_image_ocr
from .ocr_preprocessing import preprocess_image, optimize_image_for_medical_text
from .custom_logger import logger

# Initialize spellchecker with medical dictionary if available
try:
    spell = SpellChecker(language='en', distance=1)
    # Add medical terms
    medical_terms = ["endoscopy", "colonoscopy", "gastroscopy", "biopsy", 
                    "polyp", "lesion", "mucosa", "duodenum", "esophagus",
                    "stomach", "colon", "rectum", "ileum", "jejunum"]
    spell.word_frequency.load_words(medical_terms)
except ImportError:
    spell = None
    logger.warning("SpellChecker not available. Install with 'pip install pyspellchecker'")

def ensemble_ocr(image_path_or_object: Union[str, Image.Image], 
                optimize_for_medical: bool = True) -> str:
    """
    Apply multiple OCR engines and combine their results for best accuracy.
    
    Parameters:
        image_path_or_object: str or PIL.Image - Input image or path to image
        optimize_for_medical: bool - Whether to apply medical document optimizations
        
    Returns:
        str - Combined OCR text result
    """
    # Load image if path is provided
    if isinstance(image_path_or_object, str):
        image = Image.open(image_path_or_object).convert('RGB')
    else:
        image = image_path_or_object
    
    # Make a copy for each OCR engine to prevent interference
    image_tesseract = image.copy()
    image_trocr = image.copy()
    image_donut = image.copy()
    
    # Apply appropriate preprocessing for each engine
    if optimize_for_medical:
        image_tesseract = optimize_image_for_medical_text(image_tesseract)
        # For TrOCR and Donut, we'll use different preprocessing
        image_trocr = preprocess_image(image_trocr, methods=['grayscale', 'denoise', 'contrast'])
        image_donut = preprocess_image(image_donut, methods=['grayscale', 'autocontrast', 'sharpen'])
    else:
        image_tesseract = preprocess_image(image_tesseract, methods=['grayscale', 'threshold', 'sharpen'])
        image_trocr = preprocess_image(image_trocr, methods=['grayscale', 'denoise', 'contrast'])
        image_donut = preprocess_image(image_donut, methods=['grayscale', 'autocontrast'])
    
    # Get results from each OCR engine
    results = {}
    confidence_scores = {}
    
    # Tesseract OCR with optimized parameters
    try:
        config = '--oem 1 --psm 6'  # LSTM only + assume single uniform text block
        text_tesseract = pytesseract.image_to_string(image_tesseract, config=config)
        
        # Calculate confidence score
        data = pytesseract.image_to_data(image_tesseract, output_type=pytesseract.Output.DICT, config=config)
        confidences = [float(c) for c in data['conf'] if c != '-1' and c != '']
        confidence_tesseract = sum(confidences) / len(confidences) if confidences else 0
        
        results['tesseract'] = text_tesseract
        confidence_scores['tesseract'] = confidence_tesseract
        logger.debug(f"Tesseract OCR confidence: {confidence_tesseract:.2f}")
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
        results['tesseract'] = ""
        confidence_scores['tesseract'] = 0
    
    # TrOCR using imported function
    try:
        from .ocr import trocr_full_image_ocr
        text_trocr = trocr_full_image_ocr(image_trocr)
        # TrOCR doesn't provide confidence scores directly, use length as proxy
        confidence_trocr = min(1.0, len(text_trocr) / 500)  # Normalize to 0-1
        
        results['trocr'] = text_trocr
        confidence_scores['trocr'] = confidence_trocr
        logger.debug(f"TrOCR confidence proxy: {confidence_trocr:.2f}")
    except Exception as e:
        logger.error(f"TrOCR failed: {str(e)}")
        results['trocr'] = ""
        confidence_scores['trocr'] = 0
    
    # Donut OCR using imported function
    try:
        from .donut_ocr import donut_full_image_ocr
        logger.info("Running Donut OCR with improved parameters")
        text_donut = donut_full_image_ocr(image_donut)
        
        # Improved evaluation: check actual content rather than just length
        # Count number of sentences and meaningful words
        sentences = re.split(r'[.!?]', text_donut)
        sentences = [s.strip() for s in sentences if s.strip()]
        word_count = len(re.findall(r'\b\w{3,}\b', text_donut))  # Words with at least 3 chars
        
        # Base confidence on content structure rather than just length
        confidence_donut = min(1.0, (len(sentences) * 0.1 + word_count * 0.01))
        
        results['donut'] = text_donut
        confidence_scores['donut'] = confidence_donut
        logger.debug(f"Donut OCR confidence: {confidence_donut:.2f} (based on {len(sentences)} sentences, {word_count} words)")
    except Exception as e:
        logger.error(f"Donut OCR failed: {str(e)}")
        results['donut'] = ""
        confidence_scores['donut'] = 0
    
    # Determine which result to use
    best_engine = max(confidence_scores.items(), key=lambda x: x[1])[0]
    logger.info(f"Selected OCR engine: {best_engine} with confidence score {confidence_scores[best_engine]:.2f}")
    
    # Apply post-processing to the selected result
    selected_text = results[best_engine]
    
    # Check if the selected text is empty or very short
    if len(selected_text.strip()) < 20:
        # Try to combine results from all engines
        logger.info("Selected OCR result is too short, attempting to combine results")
        combined_text = combine_ocr_results(results)
        if len(combined_text.strip()) > len(selected_text.strip()):
            selected_text = combined_text
    
    # Apply spelling correction if enabled and library is available
    if spell is not None:
        logger.debug("Applying spell checking")
        selected_text = apply_spelling_correction(selected_text)
    
    return selected_text


def combine_ocr_results(results: Dict[str, str]) -> str:
    """
    Combine results from multiple OCR engines using a voting approach.
    
    Parameters:
        results: Dict - Dictionary of OCR engine names and their outputs
        
    Returns:
        str - Combined OCR text
    """
    # If any result is empty, remove it
    results = {k: v for k, v in results.items() if v.strip()}
    
    # If no valid results, return empty string
    if not results:
        return ""
    
    # If only one result, return it
    if len(results) == 1:
        return list(results.values())[0]
    
    # Split each result into lines
    line_sets = {engine: text.split('\n') for engine, text in results.items()}
    
    # Find the result with the most lines as a base
    base_engine = max(line_sets.items(), key=lambda x: len(x[1]))[0]
    combined_lines = []
    
    # For each line in the base result
    for i, base_line in enumerate(line_sets[base_engine]):
        if not base_line.strip():
            combined_lines.append("")
            continue
        
        line_candidates = [base_line]
        
        # Look for corresponding lines in other results
        for engine, lines in line_sets.items():
            if engine == base_engine:
                continue
                
            # Find the most similar line
            if i < len(lines):
                line_candidates.append(lines[i])
        
        # Choose the longest non-empty line as the best candidate
        best_line = max(line_candidates, key=lambda x: len(x.strip()) if x.strip() else 0)
        combined_lines.append(best_line)
    
    return '\n'.join(combined_lines)


def apply_spelling_correction(text: str) -> str:
    """
    Apply spelling correction to OCR text.
    
    Parameters:
        text: str - Input text
        
    Returns:
        str - Corrected text
    """
    if not spell:
        return text
        
    lines = text.split('\n')
    corrected_lines = []
    
    for line in lines:
        # Don't correct very short lines or lines that might be measurements/values
        if len(line) < 5 or any(c.isdigit() for c in line):
            corrected_lines.append(line)
            continue
            
        words = line.split()
        corrected_words = []
        
        for word in words:
            # Skip correction for special cases
            if len(word) < 3 or word.isdigit() or word[0].isdigit():
                corrected_words.append(word)
                continue
                
            # Remove punctuation for spell checking
            clean_word = ''.join(c for c in word if c.isalpha())
            
            # Only check/correct alphabetical words
            if clean_word and clean_word.lower() != "nan" and not spell.known([clean_word]):
                correction = spell.correction(clean_word)
                
                # Replace original word with correction but preserve capitalization and punctuation
                if correction and correction != clean_word:
                    # Preserve capitalization
                    if clean_word[0].isupper():
                        correction = correction.capitalize()
                    
                    # Replace the clean part in the original word
                    corrected_word = word.replace(clean_word, correction)
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
                
        corrected_lines.append(' '.join(corrected_words))
    
    return '\n'.join(corrected_lines)


def multi_scale_ocr(image: Image.Image, ocr_function, scales=None) -> Tuple[str, float]:
    """
    Process image at multiple scales and select best OCR result.
    
    Parameters:
        image: PIL.Image - Input image
        ocr_function: function - OCR function to apply
        scales: list - List of scale factors
        
    Returns:
        str - Best OCR result
        float - Confidence score
    """
    if scales is None:
        scales = [0.8, 1.0, 1.2, 1.5]
    
    results = []
    confidences = []
    logger.info(f"Running multi-scale OCR with scales {scales}")
    
    for scale in scales:
        width, height = image.size
        resized_image = image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        
        try:
            # Assume ocr_function returns (text, confidence)
            text, confidence = ocr_function(resized_image)
            results.append(text)
            confidences.append(confidence if confidence else len(text)/100)  # Default confidence based on length
            logger.debug(f"Scale {scale}: confidence {confidence if confidence else 'N/A'}, text length {len(text)}")
        except ValueError:
            # If only text is returned
            text = ocr_function(resized_image)
            results.append(text)
            confidences.append(len(text)/100)  # Use length as confidence proxy
            logger.debug(f"Scale {scale}: text length {len(text)}")
    
    # Find best result based on confidence
    best_index = confidences.index(max(confidences))
    logger.info(f"Selected scale {scales[best_index]} with confidence {confidences[best_index]}")
    
    return results[best_index], confidences[best_index]
