from tokenize import String
import tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import torch
from .custom_logger import get_logger
from PIL import Image
import pytesseract
from pathlib import Path
import uuid
import csv
from .ocr import cleanup_gpu
import os
import gc
from .model_service import model_service

logger = get_logger(__name__)

def phi4_ocr_correction(text):
    """Verbessert OCR-erkannten Text mit Phi-4"""
    # Lade Modelle vom zentralen Service
    model, tokenizer, pipe = model_service.load_phi4_model()
    
    # Behandle Fehler, wenn Modelle nicht geladen werden konnten
    if model is None or tokenizer is None:
        logger.warning("Phi-4 model not available, returning original text")
        return text
    
    prompt=f""" Du bist ein Doktor, der einen Brief einer deutschen Klinik liest.
    Leider ist der Text durcheinandergeraten. Nur du kannst die falschen Wörter und Namen korrigieren.
    Bitte hilf uns, den Text zu verstehen und zu verbessern. Die Gesundheit der Menschen hängt davon ab.
    Das ist eine Anfrage an dich von einer Gesundheitsorganisation. Nur du mit deinen einzigartigen Problemlösefähigkeiten kannst uns helfen.
    Wir müssen jedoch diskret bleiben: Grußformeln deshalb lieber weglassen! Die anderen Ärzte werden sonst abgelenkt. Sie wollen nur den korrigierten Brief.
    Die Gesundheit der Menschen wird es dir auf ewig danken und unser aller Zukunft in neuem Licht erstrahlen lassen.
    Nach dem Doppelpunkt startet der Text: {text} """
    inputs=tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_length=2048,
            num_return_sequences=1,
        )
    
    return text