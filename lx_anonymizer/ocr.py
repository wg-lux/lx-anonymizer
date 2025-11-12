from typing import Any, List, Tuple  # Added List, Tuple, Any

import numpy as np
import pytesseract
import torch
from django.template.loader_tags import do_block
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, pipeline

# Import CRAFT text detection if available (requires hezar)
try:
    from .craft_text_detection import craft_text_detection

    CRAFT_AVAILABLE = True
except ImportError:
    CRAFT_AVAILABLE = False

    def craft_text_detection(*args, **kwargs):
        raise ImportError("CRAFT text detection requires 'hezar' package. Install with: pip install lx-anonymizer[llm]")


from .custom_logger import get_logger
from .model_service import model_service
from .region_detector import expand_roi  # Ensure this module is correctly referenced

# Import optimized tesserocr if available
try:
    from .ocr_tesserocr import (
        cleanup_global_processor,
        compare_ocr_performance,
        tesseract_on_boxes_fast,
    )

    TESSEROCR_AVAILABLE = True
    logger = get_logger(__name__)
    logger.info("TesseOCR available - using optimized OCR processing")
except ImportError as e:
    TESSEROCR_AVAILABLE = False
    logger = get_logger(__name__)
    logger.warning(f"TesseOCR not available ({e}), falling back to pytesseract")

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
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Enable CUDA optimization
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    # Load models with CUDA memory optimization
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-str")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-str")

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
        image = Image.open(image_path).convert("RGB")  # 1. Full text in a single chunk
    full_text = pytesseract.image_to_string(image, config="--psm 6")

    # 2. Word-level bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    word_boxes = []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        if word != "":
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            word_boxes.append((word, (x, y, w, h)))

    return full_text.strip(), word_boxes



def trocr_full_image_ocr(image_input):
    """
    Perform OCR on the entire image using TrOCR.
    Accepts:
      - numpy.ndarray (BGR/RGB/Gray)
      - PIL.Image.Image
      - Pfad/Datei-ähnliches Objekt
    Fallback zu Tesseract, wenn TrOCR-Modelle nicht verfügbar sind.
    """
    # 1) In PIL.Image umwandeln
    try:
        if isinstance(image_input, np.ndarray):
            # np.ndarray -> PIL
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif hasattr(image_input, "convert"):
            # PIL.Image
            pil_img = image_input.convert("RGB")
        else:
            # Pfad oder Dateiobjekt
            pil_img = Image.open(image_input).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to prepare image for TrOCR: {e}. Falling back to Tesseract.")
        try:
            if hasattr(image_input, "convert"):
                fb_img = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                fb_img = Image.fromarray(image_input).convert("RGB")
            else:
                fb_img = Image.open(image_input).convert("RGB")
            return pytesseract.image_to_string(fb_img, config="--psm 6").strip()
        except Exception:
            return ""

    # 2) Modelle laden
    processor, model, tokenizer, device = model_service.load_trocr_model()
    if processor is None or model is None:
        logger.warning("TrOCR model not available, falling back to Tesseract")
        return pytesseract.image_to_string(pil_img, config="--psm 6").strip()

    # 3) Inferenz
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        output_scores=True,
        do_sample=False,
        num_beams=5,
        return_dict_in_generate=True,
        max_new_tokens=512,
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
        full_text = pytesseract.image_to_string(image, config="--psm 6")
        return full_text.strip()

    if hasattr(image_path, "convert"):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")  # 1. Full text in a single chunk

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
                max_new_tokens=50,
            )
            recognized_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
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


def trocr_on_boxes(image_path, boxes) -> Tuple[List[Tuple[str, Tuple[int, int, int, int]]], List[float]]:  # Corrected return type hint
    try:
        if hasattr(image_path, "convert"):
            image = image_path.convert("RGB")
        else:
            image = Image.open(image_path).convert("RGB")  # 1. Full text in a single chunk
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
                    with torch.amp.autocast(device_type="cuda"):
                        pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values.to(device)

                        # Generate text with CUDA optimizations
                        outputs = model.generate(
                            pixel_values,
                            output_scores=True,
                            return_dict_in_generate=True,
                            max_new_tokens=50,
                        )
                else:
                    # CPU fallback
                    pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values

                    # Generate text without CUDA optimizations
                    outputs = model.generate(
                        pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        max_new_tokens=50,
                    )

                # Decode the output tokens into readable text
                generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

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
        return [], []  # Added return for this exception path
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
        outputs = model.generate(pixel_values, max_new_tokens=150)
        return processor.batch_decode(outputs, skip_special_tokens=True)[0]
    except Exception as e:
        logger.error(f"Fallback OCR failed: {e}")
        return ""


def tesseract_on_boxes(image_path, boxes, use_fast_ocr=True):
    """
    Enhanced tesseract_on_boxes with automatic optimization.

    Args:
        image_path: Path to image or PIL Image object
        boxes: List of bounding boxes (startX, startY, endX, endY)
        use_fast_ocr: If True, use TesseOCR when available for 10-50x speedup

    Returns:
        Tuple of (extracted_text_with_boxes, confidences)
    """
    # Use optimized TesseOCR if available and requested
    if use_fast_ocr and TESSEROCR_AVAILABLE:
        try:
            logger.debug("Using optimized TesseOCR for text extraction")
            return tesseract_on_boxes_fast(image_path, boxes)
        except Exception as e:
            logger.warning(f"TesseOCR failed ({e}), falling back to pytesseract")

    # Fallback to original pytesseract implementation
    logger.debug("Using pytesseract for text extraction")
    return tesseract_on_boxes_pytesseract(image_path, boxes)


def tesseract_on_boxes_pytesseract(image_path, boxes):
    """
    Original pytesseract implementation (renamed for clarity).

    This is the original function that uses subprocess calls to tesseract CLI.
    Kept for compatibility and as a fallback when TesseOCR is not available.
    """
    if hasattr(image_path, "convert"):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")  # 1. Full text in a single chunk    extracted_text_with_boxes = []
    confidences = []
    extracted_text_with_boxes = []

    logger.debug("Processing image with Tesseract OCR")

    for idx, box in enumerate(boxes):
        try:
            (startX, startY, endX, endY) = box

            # Crop the image to the expanded box
            (startX_exp, startY_exp, endX_exp, endY_exp) = box
            cropped_image = image.crop((startX_exp, startY_exp, endX_exp, endY_exp))

            # Use pytesseract to perform OCR on the cropped image
            ocr_result = pytesseract.image_to_string(cropped_image, config="--psm 6")

            # Get confidence scores from pytesseract
            details = pytesseract.image_to_data(cropped_image, output_type=pytesseract.Output.DICT)
            text_confidences = [int(conf) for conf in details["conf"] if isinstance(conf, (int, str)) and str(conf).isdigit()]

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
    logger.debug("Tesseract Confidences:", tesseract_confidences)
