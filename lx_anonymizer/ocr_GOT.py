from transformers import AutoModel, AutoTokenizer
import cv2
import numpy as np
from PIL import Image
import io
import torch

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'ucaslcl/GOT-OCR2_0', 
        trust_remote_code=True, 
        use_safetensors=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    model = model.eval()
    if device == "cuda":
        model = model.cuda()
except Exception as e:
    print(f"Error loading GOT-OCR model: {e}")
    raise

def ocr(image):
    """
    Perform OCR on an image.
    
    Args:
        image: A PIL Image object or a path to an image file.
    
    Returns:
        str: The extracted text from the image.
    """
    if isinstance(image, Image.Image):
        # Convert PIL Image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # You can use other formats like JPEG
        image_bytes = buffered.getvalue()
        
        res = model.chat(tokenizer, image_bytes, ocr_type='ocr')
    else:
        # Assume it's a file path
        with open(str(image), 'rb') as f:
            image_bytes = f.read()
        res = model.chat(tokenizer, image_bytes, ocr_type='ocr')

    return res
