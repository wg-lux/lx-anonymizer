from .ocr import trocr_on_boxes, tesseract_on_boxes, tesseract_full_image_ocr
from .spacy_NER import spacy_NER_German
from .flair_NER import flair_NER_German
from .names_generator import gender_and_handle_full_names, gender_and_handle_separate_names, gender_and_handle_device_names
import re
from .blur import blur_function
from .device_reader import read_name_boxes, read_background_color
from .east_text_detection import east_text_detection
from .tesseract_text_detection import tesseract_text_detection
from .craft_text_detection import craft_text_detection
import cv2
import json
from pathlib import Path
import uuid
from .directory_setup import create_temp_directory, create_blur_directory
import csv
from .custom_logger import get_logger
import torch
import pymupdf
from .lx_anonymizer.lx_anonymizer.llm_phi4 import analyze_full_image_with_context, analyze_text_with_phi4
from .box_operations import find_or_create_close_box, combine_boxes, close_to_box, filter_empty_boxes, get_dominant_color
from .fuzzy_matching import fuzzy_match_snippet, correct_box_for_new_text

# Configure logging
logger = get_logger(__name__)


def process_images_with_OCR_and_NER(file_path, east_path='frozen_east_text_detection.pb', device="default", min_confidence=0.5, width=320, height=320, skip_blur=False, skip_reassembly=False):
    """
    Process images with OCR and NER.
    
    Parameters:
    - file_path: str
        The path to the image or PDF file to process.
    - east_path: str
        The path to the EAST text detector model. Default is 'frozen_east_text_detection.pb'.
    - device: str
        The device name to set the correct text settings. Default is 'default'.
    - min_confidence: float
        The minimum probability required to inspect a region. Default is 0.5.
    - width: int
        The resized image width (should be a multiple of 32). Default is 320.
    - height: int
        The resized image height (should be a multiple of 32). Default is 320.
    - skip_blur: bool
        A boolean value representing if the blur function should be skipped. Default is False.
    - skip_reassembly: bool
        Skip reassembling the PDF after processing. Default is False.
    """
    temp_dir, base_dir, csv_dir = create_temp_directory()
    logger.info(f"Processing file: {file_path}")
    modified_images_map = {}
    combined_results = []
    names_detected = []
    gender_pars = []

    try:
        file_extension = file_path.suffix.lower().lstrip('.')  # lstrip removes the leading '.'        
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'pdf': 'application/pdf',
        }
        file_type = mime_types.get(file_extension, 'application/octet-stream').split('/')[-1]

        if file_type not in ['jpg', 'jpeg', 'png', 'tiff', 'pdf']:
            raise ValueError('Invalid file type.')

        image_paths = []
        extracted_text = ''

        if file_type == 'pdf':
            # Open PDF using pymupdf
            doc = pymupdf.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text:
                    extracted_text += text
                pix = page.get_pixmap()
                image_path = Path(temp_dir) / f"{uuid.uuid4()}_page_{page_num}.png"
                pix.save(image_path)
                image_paths.append(image_path)
        else:
            image_paths = [file_path]

        if not image_paths:
            error_message = "No images to process."
            logger.error(error_message)
            raise RuntimeError(error_message)
            
        # If we have extracted text and we're skipping blur, analyze with LLM first
        if extracted_text and skip_blur:
            logger.info("Analyzing extracted text with LLM")
            text_csv_path = csv_dir / f"text_analysis_{uuid.uuid4()}.csv"
            text_results, _ = analyze_text_with_phi4(extracted_text, text_csv_path, file_path)
            logger.info(f"LLM text analysis results: {text_results}")

        blurred_image_path = image_paths[0]
        for img_path in image_paths:
            logger.info(f"Processing image: {img_path}")
            full_text, word_boxes = tesseract_full_image_ocr(img_path)
            try:
                first_name_box, last_name_box = read_name_boxes(device)
                background_color = read_background_color(device)
            except Exception as e:
                logger.warning("Using default values for name replacement.")
                first_name_box, last_name_box = None, None
                background_color = get_dominant_color(cv2.imread(str(img_path)))

            # Handle LLM analysis of the image
            if skip_blur:
                logger.info("Skipping blur, running LLM analysis on image")
                llm_csv_path = csv_dir / f"image_analysis_{uuid.uuid4()}.csv"
                image_results, _ = analyze_full_image_with_context(img_path, llm_csv_path, csv_dir)
                logger.info(f"LLM image analysis results: {len(image_results) if image_results else 0} entities")
            else:
                # Regular blur processing 
                if first_name_box and last_name_box:
                    blurred_image_path = blur_function(blurred_image_path, first_name_box, background_color)
                    blurred_image_path = blur_function(blurred_image_path, last_name_box, background_color)

            east_boxes, east_confidences_json = east_text_detection(img_path, east_path, min_confidence, width, height)
            tesseract_boxes, tesseract_confidences_json = tesseract_text_detection(img_path, min_confidence, width, height)
            craft_boxes, craft_confidences_json = craft_text_detection(img_path, min_confidence, width, height)  
            combined_boxes = east_boxes + tesseract_boxes + craft_boxes

            logger.info("Running OCR on boxes")
            trocr_results, trocr_confidences = trocr_on_boxes(img_path, combined_boxes)
            tesseract_results, tess_confidences = tesseract_on_boxes(img_path, combined_boxes)

            all_ocr_results = trocr_results + tesseract_results
            all_ocr_confidences = trocr_confidences + tess_confidences
            east_confidences = json.loads(east_confidences_json)
            tesseract_confidences = json.loads(tesseract_confidences_json)
            craft_confidences = json.loads(craft_confidences_json)
            
            all_text_detection_confidences = (
                east_confidences +
                tesseract_confidences +
                craft_confidences
            )
            all_ocr_results = filter_empty_boxes(all_ocr_results)

            # If you need it as JSON later:
            all_text_detection_confidences_json = json.dumps(all_text_detection_confidences)
            
            for (phrase, phrase_box), ocr_confidence in zip(all_ocr_results, all_ocr_confidences):
                # Skip blur operations if requested
                if skip_blur:
                    # Just collect OCR results without blurring
                    combined_results.append((phrase, phrase_box, ocr_confidence, []))
                    continue
                
                # Normal processing path with fuzzy correction and blurring
                blurred_image_path, modified_images_map, combined_results, updated_genders = do_ocr_with_fuzzy_correction(
                    all_ocr_results=all_ocr_results,
                    all_ocr_confidences=all_ocr_confidences,
                    blurred_image_path=blurred_image_path,
                    combined_results=combined_results,
                    names_detected=names_detected,
                    device=device,
                    modified_images_map=modified_images_map,
                    combined_boxes=combined_boxes,
                    first_name_box=first_name_box,
                    last_name_box=last_name_box,
                    full_text_candidates=phrase
                )
                gender_pars.extend(updated_genders)

        # Prepare CSV writing
        csv_path = csv_dir / f"name_anonymization_data_i{Path(file_path).stem}{uuid.uuid4()}.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = [
                'filename', 'startX', 'startY', 'endX', 'endY', 
                'text', 'phrase_box', 'ocr_confidence', 
                'entity_text', 'entity_tag'
            ]            
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()

            for combined_result in combined_results:
                phrase, phrase_box, ocr_confidence, entities = combined_result
                startX, startY, endX, endY = phrase_box
                for entity_text, entity_tag in entities:
                    writer.writerow({
                        'filename': file_path,
                        'startX': startX,
                        'startY': startY,
                        'endX': endX,
                        'endY': endY,
                        'text': phrase,
                        'phrase_box': phrase_box,
                        'ocr_confidence': ocr_confidence,
                        'entity_text': entity_text,
                        'entity_tag': entity_tag
                    })

        logger.info(f"NER results saved to {csv_path}")

        result = {
            'filename': file_path,
            'file_type': file_type,
            'extracted_text': extracted_text,
            'names_detected': names_detected,
            'combined_results': combined_results,
            'modified_images_map': modified_images_map,
            'gender_pars': gender_pars
        }

        # If not skipping blur and we have a blurred image path
        if not skip_blur and blurred_image_path is not None:
            output_filename = f"blurred_image_{uuid.uuid4()}.jpg"
            blur_dir = create_blur_directory()
            output_path = Path(blur_dir) / output_filename
            final_image = cv2.imread(str(blurred_image_path))
            cv2.imwrite(str(output_path), final_image)
            logger.info(f"Final blurred image saved to: {output_path}")
        
        # Always run LLM analysis and add results 
        llm_results, llm_csv_path = analyze_full_image_with_context(file_path, csv_path, csv_dir)
        result['llm_results'] = llm_results
        result['llm_csv_path'] = llm_csv_path

        logger.info(f"Processing completed: {len(combined_results)} results, csv: {csv_path}, llm results: {llm_csv_path}")
        return modified_images_map, result
    except Exception as e:
        error_message = f"Error in process_images_with_OCR_and_NER: {e}, File Path: {file_path}"
        logger.error(error_message)
        raise RuntimeError(error_message)



def process_text(extracted_text):
    cleaned_text = re.sub(r'\n{2,}', '\n', extracted_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    return cleaned_text

from typing import List, Tuple, Dict, Optional # Added Optional

def process_ocr_results(
    image_path: str,
    phrase: str,
    phrase_box: Tuple[int, int, int, int],
    ocr_confidence: float,
    combined_results: List[Tuple[str, Tuple[int, int, int, int], float, List[Tuple[str, str]]]],
    names_detected: List[str],
    device: str,
    modified_images_map: Dict[Tuple[str, str], str],
    combined_boxes: List[Tuple[int, int, int, int]],
    first_name_box: Optional[Tuple[int, int, int, int]] = None, # Changed here
    last_name_box: Optional[Tuple[int, int, int, int]] = None # Changed here
) -> Tuple[str, Dict[Tuple[str, str], str], List[Tuple[str, Tuple[int, int, int, int], float, List[Tuple[str, str]]]], List[str]]:
    processed_text = process_text(phrase)
    entities = split_and_check(processed_text)
    logger.info(f"Entities detected: {entities}")
    
    box_to_image_map = {}
    gender_pars = []  # Changed from {} to []
    new_image_path = image_path  # Keep track of the current image path being manipulated

    for entity in entities:
        name = entity[0]
        if first_name_box and last_name_box:
            if close_to_box(first_name_box, phrase_box) or close_to_box(last_name_box, phrase_box):
                box_to_image_map, gender_par = gender_and_handle_device_names(name, phrase_box, new_image_path, device)
            else:
                new_image_path, last_name_box = modify_image_for_name(new_image_path, phrase_box, combined_boxes)
                box_to_image_map, gender_par = gender_and_handle_separate_names(name, phrase_box, last_name_box, new_image_path, device)
        else:
            new_image_path, last_name_box = modify_image_for_name(new_image_path, phrase_box, combined_boxes)
            box_to_image_map, gender_par = gender_and_handle_separate_names(name, phrase_box, last_name_box, new_image_path, device)

        names_detected.append(name)
        gender_pars.append(gender_par)  # Now valid since gender_pars is a list
        for box_key, modified_image_path in box_to_image_map.items():
            modified_images_map[(box_key, new_image_path)] = modified_image_path

    combined_results.append((phrase, phrase_box, ocr_confidence, entities))
    return new_image_path, modified_images_map, combined_results, gender_pars  # Always return the last modified path

def do_ocr_with_fuzzy_correction(
    all_ocr_results,
    all_ocr_confidences,
    blurred_image_path,
    combined_results,
    names_detected,
    device,
    modified_images_map,
    combined_boxes,
    first_name_box=None,
    last_name_box=None,
    full_text_candidates=None
):
    """
    Demonstrates how to integrate fuzzy matching and box correction
    into your existing loop before calling 'process_ocr_results'.
    """
    # If you don't have a big list of words, you can pass in your entire recognized text 
    # as e.g. `full_text_candidates = full_text.split()`.
    if full_text_candidates is None:
        full_text_candidates = []

    updated_genders = []

    for ((snippet_text, snippet_box), snippet_conf) in zip(all_ocr_results, all_ocr_confidences):

        # 1) Fuzzy match the snippet_text against a global list or full-image words
        best_match, ratio = fuzzy_match_snippet(snippet_text, full_text_candidates, threshold=0.7)

        corrected_text = snippet_text  # default is the original
        corrected_box = snippet_box

        # 2) If the best_match is significantly better, use it
        #    Let's say ratio > 0.85 is considered a "strong" match
        if best_match and ratio > 0.85 and len(best_match) > len(snippet_text):
            logger.info(f"Fuzzy corrected '{snippet_text}' -> '{best_match}' (ratio={ratio:.2f})")
            corrected_text = best_match

            # 3) Adjust bounding box if the new text is longer
            corrected_box = correct_box_for_new_text(
                blurred_image_path,
                snippet_box,
                old_text=snippet_text,
                new_text=corrected_text,
                extension_margin=15
            )

        # 4) Now feed the (corrected) text + box to your NER and anonymization logic
        blurred_image_path, modified_images_map, combined_results, genders = process_ocr_results(
            blurred_image_path,
            corrected_text,
            corrected_box,
            snippet_conf,
            combined_results,
            names_detected,
            device,
            modified_images_map,
            combined_boxes,
            first_name_box,
            last_name_box
        )
        updated_genders.extend(genders)

    return blurred_image_path, modified_images_map, combined_results, updated_genders


def modify_image_for_name(image_path, phrase_box, combined_boxes):
    image = cv2.imread(str(image_path))
    image_height, image_width, _ = image.shape  
    last_name_box = find_or_create_close_box(phrase_box, combined_boxes, image_width)
    
    temp_dir, base_dir, csv_dir = create_temp_directory()
    temp_image_path = Path(temp_dir) / f"{uuid.uuid4()}.jpg"
    cv2.imwrite(str(temp_image_path), image)
    
    return blur_function(temp_image_path, phrase_box), last_name_box

def split_and_check(phrase):
    # Initialize an empty list to store entities
    all_entities = []
    
    # Get entities from spaCy
    spacy_entities = spacy_NER_German(phrase)
    if spacy_entities:
        all_entities.extend([(entity[0], 'PER') for entity in spacy_entities if entity[3] == 'PER'])
    
    # Get entities from Flair
    flair_entities = flair_NER_German(phrase)
    if flair_entities:
        all_entities.extend([(entity.text, 'PER') for entity in flair_entities if entity.tag == 'PER'])
    
    if all_entities:
        return all_entities
        
    # If no entities found, try with parts of the phrase
    parts = [phrase[:3], phrase[-3:], phrase[:4] + phrase[-4:], 
            phrase[:5] + phrase[-5:], phrase[:6] + phrase[-6:]]
    
    for part in parts:
        spacy_entities = spacy_NER_German(part)
        if spacy_entities:
            entities = [(entity[0], 'PER') for entity in spacy_entities if entity[3] == 'PER']
            if entities:
                return entities
                
        flair_entities = flair_NER_German(part)
        if flair_entities:
            entities = [(entity.text, 'PER') for entity in flair_entities if entity.tag == 'PER']
            if entities:
                return entities
    
    return []

# Example usage
if __name__ == "__main__":
    file_path = "your_file_path.jpg"
    modified_images_map, result = process_images_with_OCR_and_NER(file_path)
    for res in result['combined_results']:
        logger.info(res)
