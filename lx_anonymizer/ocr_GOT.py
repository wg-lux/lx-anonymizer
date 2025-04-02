from transformers import AutoModel, AutoTokenizer
import cv2
import numpy as np
from PIL import Image

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
model = model.eval().cuda()

def ocr(image):

    res = model.chat(tokenizer, image, ocr_type='ocr')

    return res
