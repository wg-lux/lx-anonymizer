from transformers import AutoModel, AutoTokenizer
import cv2
import numpy as np
from PIL import Image

import torch

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'ucaslcl/GOT-OCR2_0', 
        trust_remote_code=True, 
        low_cpu_mem_usage=True, 
        device_map=device, 
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

    res = model.chat(tokenizer, image, ocr_type='ocr')

    return res
