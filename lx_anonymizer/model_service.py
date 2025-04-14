import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import gc
import os
from pathlib import Path
from .custom_logger import get_logger
from .ollama_service import ollama_service

logger = get_logger(__name__)

class ModelService:
    """
    Singleton-Pattern-Implementierung für die Modellverwaltung.
    Stellt sicher, dass Modelle nur einmal geladen werden und zwischen Aufrufen wiederverwendet werden können.
    """
    _instance = None
    
    # Phi-4 LLM Modell
    phi4_model = None
    phi4_tokenizer = None
    phi4_pipe = None
    
    # TrOCR Modell
    trocr_processor = None
    trocr_model = None
    trocr_tokenizer = None
    trocr_device = None
    
    def __new__(cls):
        """Implementierung des Singleton-Patterns"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            # Setze CUDA-Konfiguration
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.random.manual_seed(0)
        return cls._instance
    
    def get_device(self):
        """Ermittelt das optimale Gerät für die Modellausführung"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
        return device
    
    def load_phi4_model(self, force_reload=False):
        """Lädt das Phi-4 LLM Modell, wenn es noch nicht geladen ist oder wenn force_reload=True"""
        if self.phi4_model is not None and self.phi4_tokenizer is not None and self.phi4_pipe is not None and not force_reload:
            logger.debug("Using cached Phi-4 model")
            return self.phi4_model, self.phi4_tokenizer, self.phi4_pipe
        
        logger.info("Loading Phi-4 model...")
        # Speicher bereinigen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device = self.get_device()
        
        try:
            # Verwende ein kleineres Modell für bessere Speichereffizienz
            model_id = "microsoft/Phi-3-mini-4k-instruct"
            
            try:
                # Versuche auf dem ermittelten Gerät zu laden
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device,
                    torch_dtype="auto",
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                
                logger.info(f"Successfully loaded Phi-4 model on {device}")
                
                # Cache-Modelle speichern
                self.phi4_model = model
                self.phi4_tokenizer = tokenizer
                self.phi4_pipe = pipe
                
                return model, tokenizer, pipe
                
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    logger.warning(f"CUDA out of memory, falling back to CPU: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Versuche auf CPU zu laden
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=False,
                        low_cpu_mem_usage=True,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                    )
                    logger.info("Successfully loaded Phi-4 model on CPU")
                    
                    # Cache-Modelle speichern
                    self.phi4_model = model
                    self.phi4_tokenizer = tokenizer
                    self.phi4_pipe = pipe
                    
                    return model, tokenizer, pipe
                else:
                    logger.error(f"Error initializing Phi-4 model: {e}")
                    return None, None, None
        except Exception as e:
            logger.error(f"Error initializing Phi-4 model: {e}")
            return None, None, None
    
    def load_trocr_model(self, force_reload=False):
        """Lädt das TrOCR-Modell, wenn es noch nicht geladen ist oder wenn force_reload=True"""
        if self.trocr_processor is not None and self.trocr_model is not None and not force_reload:
            logger.debug("Using cached TrOCR model")
            return self.trocr_processor, self.trocr_model, self.trocr_tokenizer, self.trocr_device
        
        logger.info("Loading TrOCR model...")
        # Speicher bereinigen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        device = self.get_device()
        
        try:
            # TrOCRProcessor verwenden (kombiniert Feature Extractor und Tokenizer)
            processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-str')
            model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
            
            # Auf Gerät verschieben mit zusätzlichen CUDA-Optimierungen
            if device.type == 'cuda':
                try:
                    # CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                    model = model.cuda()
                    
                    # Optional: Mixed precision für bessere Performance
                    # model = model.half()  # Nur aktivieren, wenn gemischte Präzision gewünscht
                    
                    logger.info(f"TrOCR model loaded on CUDA device: {torch.cuda.get_device_name(0)}")
                    logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"CUDA out of memory when loading TrOCR model: {e}")
                        # Auf CPU zurückfallen
                        device = torch.device('cpu')
                        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-str')
                        logger.info("Fallback: TrOCR model loaded on CPU")
                    else:
                        raise
            
            model.to(device)
            
            # Cache-Modelle speichern
            self.trocr_processor = processor
            self.trocr_model = model
            self.trocr_tokenizer = processor.tokenizer  # TrOCRProcessor enthält bereits den Tokenizer
            self.trocr_device = device
            
            logger.info(f"Successfully loaded TrOCR model on {device}")
            return processor, model, processor.tokenizer, device
            
        except Exception as e:
            logger.error(f"Error initializing TrOCR model: {e}")
            return None, None, None, None
    
    def correct_text_with_ollama(self, text, model_name="deepseek-r1:1.5b"):
        """Use Ollama with DeepSeek model to correct text"""
        # Use the ollama service to correct the text
        return ollama_service.correct_ocr_text(text, model_name)
    
    def cleanup_models(self):
        """Bereinigt alle geladenen Modelle und gibt Speicher frei"""
        self.phi4_model = None
        self.phi4_tokenizer = None
        self.phi4_pipe = None
        
        self.trocr_processor = None
        self.trocr_model = None
        self.trocr_tokenizer = None
        self.trocr_device = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("All models have been cleaned up")

# Erzeuge eine globale Instanz des ModelService
model_service = ModelService()
