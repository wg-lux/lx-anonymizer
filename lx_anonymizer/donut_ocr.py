import os
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from .custom_logger import get_logger

# Set the environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger(__name__)

def get_decoder_start_token_id(tokenizer):
    """
    Helper function to obtain the decoder start token ID.
    Checks for the method and falls back to known attributes.
    """
    if hasattr(tokenizer, "get_decoder_start_token_id"):
        return tokenizer.get_decoder_start_token_id()
    elif hasattr(tokenizer, "decoder_start_token_id"):
        return tokenizer.decoder_start_token_id
    elif tokenizer.bos_token:
        return tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    else:
        raise AttributeError("Tokenizer does not have a method or attribute for decoder start token ID.")

def split_image_into_chunks(image, max_height=1000, max_width=1000, overlap=50):
    """
    Split a large image into smaller chunks that can be processed separately.
    
    Args:
        image: PIL Image to split
        max_height: Maximum height of each chunk
        max_width: Maximum width of each chunk
        overlap: Overlap between chunks to avoid cutting text
        
    Returns:
        List of PIL Image chunks
    """
    width, height = image.size
    chunks = []
    
    # Calculate number of chunks needed in each dimension
    num_chunks_h = max(1, (height - overlap) // (max_height - overlap))
    num_chunks_w = max(1, (width - overlap) // (max_width - overlap))
    
    logger.info(f"Splitting image ({width}x{height}) into {num_chunks_h}x{num_chunks_w} chunks")
    
    for i in range(num_chunks_h):
        for j in range(num_chunks_w):
            left = min(j * (max_width - overlap), width - max_width)
            upper = min(i * (max_height - overlap), height - max_height)
            right = min(left + max_width, width)
            lower = min(upper + max_height, height)
            left = max(0, left)
            upper = max(0, upper)
            chunk = image.crop((left, upper, right, lower))
            chunks.append(chunk)
    
    return chunks

# Move the definition of process_chunk to an outer scope.
def process_chunk(chunk, max_new_tokens, num_beams, processor, model, device):
    """
    Process a single image chunk using the provided processor and model.
    """
    pixel_values = processor(chunk, return_tensors="pt").pixel_values.to(device)
    decoder_start_token_id = get_decoder_start_token_id(processor.tokenizer)
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        decoder_start_token_id=decoder_start_token_id
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def donut_full_image_ocr(image_input, token=None):
    """
    Perform OCR on an image using a Donut model.
    For large images, splits the image into manageable chunks to avoid CUDA OOM errors.
    
    Args:
        image_input: a PIL Image or file path.
        token (str, optional): Hugging Face auth token if the model repository is private.
        
    Returns:
        A string of the recognized text.
    """
    # Define the model identifier at the top
    model_id = "naver-clova-ix/donut-base-finetuned-rvlcdip"
    
    try:
        # Ensure we have a PIL Image
        if hasattr(image_input, "convert"):
            image = image_input.convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
        
        width, height = image.size
        large_image = width > 1000 or height > 1000
        
        # Load processor and model (with token if required)
        processor = DonutProcessor.from_pretrained(model_id, use_fast=True, use_auth_token=token)
        model = VisionEncoderDecoderModel.from_pretrained(model_id, use_auth_token=token)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        if large_image and device.type == 'cuda':
            logger.info(f"Large image detected ({width}x{height}), processing in chunks")
            chunks = split_image_into_chunks(image)
            all_outputs = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    output = process_chunk(chunk, max_new_tokens=1024, num_beams=5, processor=processor, model=model, device=device)
                    all_outputs.append(output)
                    
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"CUDA OOM on chunk {i+1}, skipping")
                    continue
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    continue
            combined_output = " ".join(all_outputs)
            logger.info("Donut OCR completed successfully (chunked mode)")
            return combined_output
        else:
            output = process_chunk(image, max_new_tokens=2048, num_beams=10, processor=processor, model=model, device=device)
            logger.info("Donut OCR completed successfully")
            return output
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory in Donut OCR, falling back to CPU")
        try:
            processor = DonutProcessor.from_pretrained(model_id, use_fast=True, use_auth_token=token)
            model = VisionEncoderDecoderModel.from_pretrained(model_id, use_auth_token=token)
            
            if hasattr(image_input, "convert"):
                image = image_input.convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
            
            chunks = split_image_into_chunks(image)
            all_outputs = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} on CPU")
                output = process_chunk(chunk, max_new_tokens=1024, num_beams=5, processor=processor, model=model, device=torch.device("cpu"))
                all_outputs.append(output)
            combined_output = " ".join(all_outputs)
            logger.info("Donut OCR completed successfully on CPU")
            return combined_output
        except Exception as cpu_error:
            logger.error(f"Error during CPU fallback: {cpu_error}")
            return ""
    except Exception as e:
        logger.error(f"Error in Donut OCR: {e}")
        return ""
