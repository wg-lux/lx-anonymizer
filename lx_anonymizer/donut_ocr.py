import os
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from .custom_logger import get_logger
import re
from typing import List, Tuple, Dict, Any, Union, Optional

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

def split_image_into_chunks(image, max_height=800, max_width=800, overlap=100):
    """
    Split a large image into smaller chunks with intelligent boundaries.
    
    This improved version tries to split at whitespace areas when possible
    to avoid cutting through text.
    
    Args:
        image: PIL Image to split
        max_height: Maximum height of each chunk
        max_width: Maximum width of each chunk
        overlap: Overlap between chunks to avoid cutting text
        
    Returns:
        List of PIL Image chunks and their positions (for merging context)
    """
    width, height = image.size
    chunks = []
    positions = []
    
    # Convert to numpy for analysis
    import numpy as np
    import cv2
    img_array = np.array(image.convert('L'))
    
    # Calculate number of chunks needed in each dimension
    num_chunks_h = max(1, (height - overlap) // (max_height - overlap))
    num_chunks_w = max(1, (width - overlap) // (max_width - overlap))
    
    logger.info(f"Splitting image ({width}x{height}) into {num_chunks_h}x{num_chunks_w} chunks")
    
    # Helper function to find better split position near whitespace
    def find_better_split(start_pos, end_pos, is_vertical):
        # Define a window to look for better split positions
        window = 100  # Pixels to look around for better splits
        
        # If we're too close to the edge, just use the calculated position
        if end_pos >= (height if is_vertical else width) - window:
            return end_pos
        
        search_start = max(start_pos + (max_height if is_vertical else max_width) // 2, end_pos - window)
        search_end = min(end_pos + window, height if is_vertical else width)
        
        # Get the relevant slice of the image for analysis
        if is_vertical:
            # Looking for horizontal lines (rows with high avg whitespace)
            slice_values = [np.mean(img_array[y, :]) for y in range(search_start, search_end)]
        else:
            # Looking for vertical lines (columns with high avg whitespace)
            slice_values = [np.mean(img_array[:, x]) for x in range(search_start, search_end)]
        
        # Higher values indicate more whitespace in grayscale
        whitespace_threshold = np.mean(slice_values) + np.std(slice_values) * 0.5
        whitespace_positions = [i + search_start for i, v in enumerate(slice_values) if v > whitespace_threshold]
        
        # If we found good whitespace positions, use the one closest to our target
        if whitespace_positions:
            return min(whitespace_positions, key=lambda x: abs(x - end_pos))
        
        # Fall back to the original position if no good whitespace found
        return end_pos
    
    for i in range(num_chunks_h):
        for j in range(num_chunks_w):
            # Calculate basic positions
            left = j * (max_width - overlap)
            upper = i * (max_height - overlap)
            right = min(left + max_width, width)
            lower = min(upper + max_height, height)
            
            # Try to find better split positions if not at the edges
            if j > 0:
                left = find_better_split(left - max_width, left, False)
            if i > 0:
                upper = find_better_split(upper - max_height, upper, True)
            
            # Ensure we don't go out of bounds
            left = max(0, left)
            upper = max(0, upper)
            
            chunk = image.crop((left, upper, right, lower))
            chunks.append(chunk)
            positions.append((left, upper, right, lower))
    
    return chunks, positions

def process_chunk(chunk, processor, model, device, 
                  max_new_tokens=1024, min_new_tokens=50, num_beams=5,
                  do_sample=True, temperature=0.7, top_k=50, top_p=0.95):
    """
    Process a single image chunk using the provided processor and model.
    
    Args:
        chunk: PIL Image chunk to process
        processor: DonutProcessor instance
        model: VisionEncoderDecoderModel instance
        device: torch device to use
        max_new_tokens: Maximum number of tokens to generate
        min_new_tokens: Minimum number of tokens to generate
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        
    Returns:
        Extracted text string
    """
    pixel_values = processor(chunk, return_tensors="pt").pixel_values.to(device)
    decoder_start_token_id = get_decoder_start_token_id(processor.tokenizer)
    
    # Generate text with enhanced parameters
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        decoder_start_token_id=decoder_start_token_id,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # Get text and confidence
    decoded_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0].strip()
    
    return decoded_text

def post_process_chunks(chunk_texts: List[str], positions: List[Tuple[int, int, int, int]]) -> str:
    """
    Post-process and merge text from multiple chunks intelligently.
    
    This function handles:
    1. Removing duplicate text at chunk boundaries
    2. Preserving paragraph structure
    3. Sorting text blocks by their position in the document
    
    Args:
        chunk_texts: List of text strings from each chunk
        positions: List of (left, upper, right, lower) positions for each chunk
        
    Returns:
        Merged and cleaned text
    """
    if not chunk_texts:
        return ""
    if len(chunk_texts) == 1:
        return chunk_texts[0]
    
    # Sort chunks by position (top to bottom, left to right)
    positioned_texts = list(zip(chunk_texts, positions))
    positioned_texts.sort(key=lambda x: (x[1][1], x[1][0]))  # Sort by upper then left
    
    # Find and remove duplicate text segments
    processed_texts = []
    for i, (text, _) in enumerate(positioned_texts):
        if not text.strip():
            continue
            
        # Check for significant overlap with previous chunks
        current_text = text
        for prev_text in processed_texts:
            # Look for the longest common substring
            current_words = current_text.split()
            prev_words = prev_text.split()
            
            # Skip if either text is too short
            if len(current_words) < 5 or len(prev_words) < 5:
                continue
                
            # Check for overlap at beginning of current text
            overlap_len = 0
            for j in range(min(15, len(current_words))):
                if j >= len(prev_words):
                    break
                if " ".join(current_words[:j+1]).lower() in prev_text.lower():
                    overlap_len = j + 1
            
            # Remove overlapping part if significant
            if overlap_len > 3:  # At least 3 words overlap
                current_text = " ".join(current_words[overlap_len:])
        
        if current_text.strip():
            processed_texts.append(current_text)
    
    # Join with appropriate spacing
    result = "\n\n".join(text for text in processed_texts if text.strip())
    
    # Final cleanup
    # Remove redundant newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    # Remove redundant spaces
    result = re.sub(r' {2,}', ' ', result)
    
    return result

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
    # Use document understanding model instead of document classification
    model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
    
    try:
        # Ensure we have a PIL Image
        if hasattr(image_input, "convert"):
            image = image_input.convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")
        
        width, height = image.size
        large_image = width > 800 or height > 800
        
        # Load processor and model (with token if required)
        processor = DonutProcessor.from_pretrained(model_id, use_fast=True, use_auth_token=token)
        model = VisionEncoderDecoderModel.from_pretrained(model_id, use_auth_token=token)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        if large_image:
            logger.info(f"Large image detected ({width}x{height}), processing in chunks")
            chunks, positions = split_image_into_chunks(image)
            all_outputs = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    output = process_chunk(
                        chunk, 
                        processor=processor, 
                        model=model, 
                        device=device,
                        max_new_tokens=1024,
                        min_new_tokens=50,
                        num_beams=5,
                        do_sample=True,
                        temperature=0.7
                    )
                    all_outputs.append(output)
                    
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"CUDA OOM on chunk {i+1}, reducing chunk size and retrying")
                    try:
                        # Try with a smaller chunk
                        smaller_chunks, _ = split_image_into_chunks(chunk, 
                                                                  max_height=400, 
                                                                  max_width=400,
                                                                  overlap=50)
                        sub_outputs = []
                        for j, smaller_chunk in enumerate(smaller_chunks):
                            sub_output = process_chunk(
                                smaller_chunk, 
                                processor=processor, 
                                model=model, 
                                device=device,
                                max_new_tokens=512,
                                min_new_tokens=20
                            )
                            sub_outputs.append(sub_output)
                            torch.cuda.empty_cache()
                        all_outputs.append(" ".join(sub_outputs))
                    except Exception as sub_e:
                        logger.error(f"Error processing sub-chunks: {sub_e}")
                        continue
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    continue
            
            # Process and merge the outputs
            combined_output = post_process_chunks(all_outputs, positions)
            logger.info("Donut OCR completed successfully (chunked mode)")
            return combined_output
        else:
            output = process_chunk(
                image, 
                processor=processor, 
                model=model, 
                device=device,
                max_new_tokens=2048,
                min_new_tokens=100,
                num_beams=10,
                do_sample=True,
                temperature=0.7
            )
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
            
            chunks, positions = split_image_into_chunks(image, max_height=600, max_width=600)
            all_outputs = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} on CPU")
                output = process_chunk(
                    chunk, 
                    processor=processor, 
                    model=model, 
                    device=torch.device("cpu"),
                    max_new_tokens=512,
                    min_new_tokens=20,
                    num_beams=3  # Lower for CPU to improve speed
                )
                all_outputs.append(output)
            
            # Process and merge the outputs
            combined_output = post_process_chunks(all_outputs, positions)
            logger.info("Donut OCR completed successfully on CPU")
            return combined_output
        except Exception as cpu_error:
            logger.error(f"Error during CPU fallback: {cpu_error}")
            return ""
    except Exception as e:
        logger.error(f"Error in Donut OCR: {e}")
        return ""

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        text = donut_full_image_ocr(image_path)
        print(f"Extracted text:\n{text}")
    else:
        print("Usage: python donut_ocr.py <image_path>")
