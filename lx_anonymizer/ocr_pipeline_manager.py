from ocr import trocr_on_boxes, tesseract_on_boxes
from spacy_NER import NER_German
from names_generator import gender_and_handle_full_names, gender_and_handle_separate_names, gender_and_handle_device_names
from east_text_detection import east_text_detection
import re
from pdf_operations import convert_pdf_to_images
from blur import blur_function
from device_reader import read_name_boxes, read_background_color
from tesseract_text_detection import tesseract_text_detection
import cv2
import json
from pathlib import Path
import uuid
from directory_setup import create_temp_directory, create_blur_directory
import csv
from custom_logger import get_logger
import torch
import fitz

# Configure logging
logger = get_logger(__name__)



def find_or_create_close_box(phrase_box, boxes, image_width, offset=60):
    (startX, startY, endX, endY) = phrase_box
    same_line_boxes = [box for box in boxes if abs(box[1] - startY) <= 10]

    if same_line_boxes:
        same_line_boxes.sort(key=lambda box: box[0])
        for box in same_line_boxes:
            if box[0] > endX:
                return box

    new_startX = min(endX + offset, image_width)
    new_endX = new_startX + (endX - startX)
    new_box = (new_startX, startY, new_endX, endY)
    return new_box

def combine_boxes(text_with_boxes):
    if not text_with_boxes:
        return text_with_boxes

    text_with_boxes = sorted(text_with_boxes, key=lambda x: (x[1][1], x[1][0]))

    merged_text_with_boxes = [text_with_boxes[0]]

    for current in text_with_boxes[1:]:
        last = merged_text_with_boxes[-1]

        current_text, current_box = current
        last_text, last_box = last

        (last_startX, last_startY, last_endX, last_endY) = last_box
        (current_startX, current_startY, current_endX, current_endY) = current_box

        if last_startY == current_startY and (current_startX - last_endX) <= 10:
            merged_box = (min(last_startX, current_startX), last_startY, max(last_endX, current_endX), last_endY)
            merged_text = last_text + ' ' + current_text
            merged_text_with_boxes[-1] = (merged_text, merged_box)
        else:
            merged_text_with_boxes.append(current)

    return merged_text_with_boxes

def process_images_with_OCR_and_NER(file_path, east_path='frozen_east_text_detection.pb', device="default", min_confidence=0.5, width=320, height=320):
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
            # Open PDF using fitz
            doc = fitz.open(file_path)
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

        blurred_image_path = image_paths[0]
        for img_path in image_paths:
            logger.info(f"Processing image: {img_path}")
            try:
                first_name_box, last_name_box = read_name_boxes(device)
                background_color = read_background_color(device)
            except Exception as e:
                logger.warning("Using default values for name replacement.")
                first_name_box, last_name_box = None, None
                background_color = (0, 0, 0)

            if first_name_box and last_name_box:
                blurred_image_path = blur_function(blurred_image_path, first_name_box, background_color)
                blurred_image_path = blur_function(blurred_image_path, last_name_box, background_color)

            east_boxes, east_confidences_json = east_text_detection(img_path, east_path, min_confidence, width, height)
            tesseract_boxes, tesseract_confidences = tesseract_text_detection(img_path, min_confidence, width, height)
            combined_boxes = east_boxes + tesseract_boxes

            logger.info("Running OCR on boxes")
            trocr_results, trocr_confidences = trocr_on_boxes(img_path, combined_boxes)
            tesseract_results, tess_confidences = tesseract_on_boxes(img_path, combined_boxes)

            all_ocr_results = trocr_results + tesseract_results
            all_ocr_confidences = trocr_confidences + tess_confidences

            for (phrase, phrase_box), ocr_confidence in zip(all_ocr_results, all_ocr_confidences):
                blurred_image_path, modified_images_map, combined_results, genders = process_ocr_results(
                    blurred_image_path, phrase, phrase_box, ocr_confidence,
                    combined_results, names_detected, device,
                    modified_images_map, combined_boxes,
                    first_name_box, last_name_box
                )
                gender_pars.extend(genders)  # Assuming 'genders' is a list

        # Prepare CSV writing
        csv_path = csv_dir / f"name_anonymization_data_i{Path(file_path).stem}{uuid.uuid4()}.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['filename', 'startX', 'startY', 'endX', 'endY', 'ocr_confidence', 'entity_text', 'entity_tag']
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
            'gender_pars': gender_pars  # Consistent key name
        }

        if blurred_image_path is not None:
            output_filename = f"blurred_image_{uuid.uuid4()}.jpg"
            blur_dir = create_blur_directory()
            output_path = Path(blur_dir) / output_filename
            final_image = cv2.imread(str(blurred_image_path))
            cv2.imwrite(str(output_path), final_image)
            logger.info(f"Final blurred image saved to: {output_path}")

        logger.info(f"Processing completed: {combined_results}")
        return modified_images_map, result
    except Exception as e:
        error_message = f"Error in process_images_with_OCR_and_NER: {e}, File Path: {file_path}"
        logger.error(error_message)
        raise RuntimeError(error_message)



def process_text(extracted_text):
    cleaned_text = re.sub(r'\n{2,}', '\n', extracted_text)
    cleaned_text = cleaned_text.replace("\n", " ")
    return cleaned_text

from typing import List, Tuple, Dict

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
    first_name_box: Tuple[int, int, int, int] = None,
    last_name_box: Tuple[int, int, int, int] = None
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

def close_to_box(name_box, phrase_box):
    (startX, startY, _, _) = phrase_box
    return abs(name_box[0] - startX) <= 10 and abs(name_box[1] - startY) <= 10

def modify_image_for_name(image_path, phrase_box, combined_boxes):
    image = cv2.imread(str(image_path))
    image_height, image_width, _ = image.shape  
    last_name_box = find_or_create_close_box(phrase_box, combined_boxes, image_width)
    
    temp_dir, base_dir, csv_dir = create_temp_directory()
    temp_image_path = Path(temp_dir) / f"{uuid.uuid4()}.jpg"
    cv2.imwrite(str(temp_image_path), image)
    
    return blur_function(temp_image_path, phrase_box), last_name_box

def split_and_check(phrase):
    entities = NER_German(phrase)
    if entities:
        return [(entity.text, entity.tag) for entity in entities if entity.tag == 'PER']
    parts = [phrase[:3], phrase[-3:], phrase[:4] + phrase[-4:], phrase[:5] + phrase[-5:], phrase[:6] + phrase[-6:]]
    for part in parts:
        entities = NER_German(part)
        if entities:
            return [(entity.text, entity.tag) for entity in entities if entity.tag == 'PER']
    return []

# Example usage
if __name__ == "__main__":
    file_path = "your_file_path.jpg"
    modified_images_map, result = process_images_with_OCR_and_NER(file_path)
    for res in result['combined_results']:
        logger.info(res)
