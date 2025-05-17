from django.template.loader_tags import do_block
from torch.cuda import temperature
from .region_detector import expand_roi  # Ensure this module is correctly referenced
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline
)
import torch
import pytesseract
import numpy as np
from .custom_logger import get_logger
from .craft_text_detection import craft_text_detection  # Neuer Import für CRAFT
from .model_service import model_service
from typing import List, Tuple, Any # Added List, Tuple, Any

logger = get_logger(__name__)
# At the start of your script
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")

def preload_models():
    global processor, model, device

    logger.info("Preloading models...")

    # More explicit CUDA availability check
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # Enable CUDA optimization
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")

    # Load models with CUDA memory optimization
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')

    # Move model to GPU and enable CUDA optimizations
    if torch.cuda.is_available():
        model = model.cuda()

    model.to(device)

    logger.info("Models preloaded successfully.")
    return processor, model, device

def cleanup_gpu():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
def print_gpu_info():
    if torch.cuda.is_available():
        logger.info(f"GPU Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        cudasupport = True
        return cudasupport
    else:
        cudasupport = False
        return cudasupport

def tesseract_full_image_ocr(image_path):
    """
    Perform OCR on the entire image using Tesseract.
    Returns:
      - A single string with all recognized text.
      - A list of (word, (left, top, width, height)) for each recognized word.
    """

    if hasattr(image_path, "convert"):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")    # 1. Full text in a single chunk
    full_text = pytesseract.image_to_string(image, config='--psm 6')

    # 2. Word-level bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    word_boxes = []
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if word != "":
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            word_boxes.append((word, (x, y, w, h)))

    return full_text.strip(), word_boxes

def trocr_full_image_ocr(image_path):
    if hasattr(image_path, "convert"):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")    # 1. Full text in a single chunk
    processor, model, _, device = model_service.load_trocr_model()

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values, 
        output_scores=True, 
        do_sample=False,            # Use beam search for more deterministic results
        num_beams=5,                # Example beam count for better exploration
        return_dict_in_generate=True, 
        max_new_tokens=30000          # Increase max tokens to allow for longer text
    )
    return processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()


def trocr_full_image_ocr_on_boxes(image_path):
    """
    Perform OCR on the entire image using TrOCR.
    """
    # Lade Modelle vom Service anstatt von preload_models
    processor, model, tokenizer, device = model_service.load_trocr_model()
    
    # Behandle Fehler, wenn Modelle nicht geladen werden konnten
    if processor is None or model is None or tokenizer is None:
        logger.warning("TrOCR model not available, falling back to tesseract")
        # Fallback zu Tesseract implementieren
        image = Image.open(image_path).convert("RGB")
        full_text = pytesseract.image_to_string(image, config='--psm 6')
        return full_text.strip()
    
    if hasattr(image_path, "convert"):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")    # 1. Full text in a single chunk
            
    # Detect text regions using CRAFT
    boxes, _ = craft_text_detection(image_path)
    
    ocr_results = []
    
    if boxes:
        logger.info(f"CRAFT detected {len(boxes)} regions. Processing each region with TrOCR.")
        for box in boxes:
            (startX, startY, endX, endY) = box
            cropped_image = image.crop((startX, startY, endX, endY))
            pixel_values = processor(cropped_image, return_tensors="pt").pixel_values.to(device)
            outputs = model.generate(
                pixel_values, 
                output_scores=True, 
                do_sample=True,
                temperature=0.6,
                return_dict_in_generate=True, 
                max_new_tokens=50
            )
            recognized_text = tokenizer.batch_decode(
                outputs.sequences, 
                skip_special_tokens=True
            )[0]
            logger.debug(f"Box {box} yielded text: '{recognized_text}'")
            if recognized_text.strip():
                ocr_results.append(recognized_text.strip())
        final_text = "\n".join(ocr_results)
        if not final_text.strip():
            logger.warning("No text recognized in regions, falling back to full image OCR.")
            final_text = trocr_full_image_ocr(image)
    else:
        logger.info("No regions detected by CRAFT. Falling back to full image OCR.")
        final_text = trocr_full_image_ocr(image)
    
    return final_text

def trocr_on_boxes(image_path, boxes) -> Tuple[List[Tuple[str, Tuple[int, int, int, int]]], List[float]]: # Corrected return type hint
    try:
        if hasattr(image_path, "convert"):
            image = image_path.convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")    # 1. Full text in a single chunk
        extracted_text_with_boxes = []
        confidences = []
        # Ensure models are loaded
        processor, model, device = preload_models()

        logger.debug("Processing image with TrOCR")

        for idx, box in enumerate(boxes):
            cudasupport = print_gpu_info()
            try:
                (startX, startY, endX, endY) = box

                # Expand the region of interest
                image_np = np.asarray(image)
                image_shape = image_np.shape
                expanded_box = expand_roi(startX, startY, endX, endY, 5, image_shape)
                (startX_exp, startY_exp, endX_exp, endY_exp) = expanded_box

                # Crop the image to the expanded box
                cropped_image = image.crop((startX_exp, startY_exp, endX_exp, endY_exp))

                # Process the cropped image using the processor
                if cudasupport:
                    # Use CUDA with automatic mixed precision
                    with torch.amp.autocast(device_type='cuda'):
                        pixel_values = processor(
                            images=cropped_image, 
                            return_tensors="pt"
                        ).pixel_values.to(device)

                        # Generate text with CUDA optimizations
                        outputs = model.generate(
                            pixel_values, 
                            output_scores=True, 
                            return_dict_in_generate=True, 
                            max_new_tokens=50
                        )
                else:
                    # CPU fallback
                    pixel_values = processor(
                        images=cropped_image, 
                        return_tensors="pt"
                    ).pixel_values
                    
                    # Generate text without CUDA optimizations
                    outputs = model.generate(
                        pixel_values, 
                        output_scores=True, 
                        return_dict_in_generate=True,
                        max_new_tokens=50
                    )

                # Decode the output tokens into readable text
                generated_text = processor.batch_decode(
                    outputs.sequences, 
                    skip_special_tokens=True
                )[0]

                # Calculate confidence score from the last token's scores
                scores = outputs.scores  # List of logits for each generation step
                if scores:
                    # Take the scores from the last generation step
                    last_scores = scores[-1]
                    confidence_score = torch.nn.functional.softmax(last_scores, dim=-1).max().item()
                else:
                    confidence_score = 0.0  # Default confidence if scores are unavailable

                # Append results to the lists
                extracted_text_with_boxes.append((generated_text.strip(), expanded_box))
                confidences.append(confidence_score)

                logger.info(f"Processed box {idx + 1}/{len(boxes)}: '{generated_text.strip()}' with confidence {confidence_score:.4f}")

            except Exception as e:
                logger.info(f"Error processing box {idx + 1}/{len(boxes)}: {e}")
                extracted_text_with_boxes.append(("", box))
                confidences.append(0.0)

        logger.debug("TrOCR processing complete")
        return extracted_text_with_boxes, confidences
    except Exception as e:
        logger.error(f"Error in TrOCR processing: {e}")
        cleanup_gpu()
        return [], [] # Added return for this exception path
    finally:
        cleanup_gpu()

def fallback_full_ocr(image, processor, model, device):
    """
    Fallback OCR on the entire image when region detection fails
    """
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")
    try:
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        outputs = model.generate(
            pixel_values, 
            max_new_tokens=150
        )
        return processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"Fallback OCR failed: {e}")
        return ""

def tesseract_on_boxes(image_path, boxes):
    if hasattr(image_path, "convert"):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")    # 1. Full text in a single chunk    extracted_text_with_boxes = []
    confidences = []
    extracted_text_with_boxes = []

    logger.debug("Processing image with Tesseract OCR")

    for idx, box in enumerate(boxes):
        try:
            (startX, startY, endX, endY) = box

            # Expand the region of interest
            image_np = np.asarray(image)
            image_shape = image_np.shape
            #expanded_box = expand_roi(startX, startY, endX, endY, 5, image_shape)
            (startX_exp, startY_exp, endX_exp, endY_exp) = box

            # Crop the image to the expanded box
            cropped_image = image.crop((startX_exp, startY_exp, endX_exp, endY_exp))

            # Use pytesseract to perform OCR on the cropped image
            ocr_result = pytesseract.image_to_string(cropped_image, config='--psm 6')

            # Get confidence scores from pytesseract
            details = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT)
            text_confidences = [
                int(conf) for conf in details['conf'] 
                if isinstance(conf, (int, str)) and str(conf).isdigit()
            ]

            # Calculate the average confidence score
            confidence_score = sum(text_confidences) / len(text_confidences) if text_confidences else 0.0

            # Append results to the lists
            extracted_text_with_boxes.append((ocr_result.strip(), box))
            confidences.append(confidence_score)

            logger.debug(f"Processed box {idx + 1}/{len(boxes)}: '{ocr_result.strip()}' with confidence {confidence_score:.2f}")

        except Exception as e:
            logger.info(f"Error processing box {idx + 1}/{len(boxes)}: {e}")
            extracted_text_with_boxes.append(("", box))
            confidences.append(0.0)

    logger.info("Tesseract OCR processing complete")
    return extracted_text_with_boxes, confidences

if __name__ == "__main__":
    # Preload models once when the script runs
    processor, model, device = preload_models()
    
    # Example usage:
    # Define your image path and bounding boxes
    image_path = "path/to/your/image.jpg"
    boxes = [
        (50, 50, 200, 150),
        (250, 80, 400, 180),
        # Add more boxes as needed
    ]
    
    # Perform OCR using TrOCR
    trocr_results, trocr_confidences = trocr_on_boxes(image_path, boxes)
    logger.debug("TrOCR Results:", trocr_results)
    logger.debug("TrOCR Confidences:", trocr_confidences)
    
    # Perform OCR using Tesseract
    tesseract_results, tesseract_confidences = tesseract_on_boxes(image_path, boxes)
    logger.debug("Tesseract Results:", tesseract_results)
    logger.debug("Tesseract Confidences:", tesseract_confidences)
