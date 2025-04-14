import json
from ollama_python.endpoints import GenerateAPI, ModelManagementAPI
from requests import api
from .custom_logger import get_logger
import subprocess

logger = get_logger(__name__)

class OllamaService:
    """Service to interact with a locally running Ollama instance via the ollama-python library."""
    
    def __init__(self, base_url="http://127.0.0.1:11434"):  # Korrigierter Standard-Port für Ollama
        self.base_url = base_url
        # The ModelManagementAPI is used to manage model availability.
        self.model_api = ModelManagementAPI(base_url=self.base_url)
        self.generate_api = GenerateAPI(base_url=self.base_url, model="deepseek-r1:1.5b")  # Remove default model here
        logger.info(f"Initialized OllamaService with base_url: {self.base_url}")

    def is_server_running(self):
        """Check if the Ollama server is accessible by attempting to list local models."""
        try:
            _ = self.list_models()
            logger.info("Ollama server is running")
            return True
        except Exception as e:
            logger.error(f"Ollama server is not running: {e}")
            return False

    def list_models(self):
        """List all models available locally from the Ollama instance."""
        try:
            result = self.model_api.list_local_models()
            models = [model.name for model in result.models]  # Extract model names from the list of Model objects
            logger.info("Retrieved models: " + ", ".join(models))
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def ensure_model_loaded(self, model_name="mistral"):
        """
        Ensure the specified model is available. If not, pull it using the ModelManagementAPI.
        
        Returns:
            bool: True if the model is available after this call, False otherwise.
        """
        models = self.list_models()
        if model_name in models:
            logger.info(f"Model {model_name} is already loaded")
            return True
        
        logger.info(f"Model {model_name} not found. Pulling model (this may take a while)...")
        try:
            # Use the ModelManagementAPI to pull the model
            result = self.model_api.pull(name=model_name, stream=False)  # Disable streaming for simplicity
            logger.info(f"Pull result: {result}")
            return True  # Assume success if no exception
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False

    def generate_text(self, prompt, model_name="deepseek-r1:1.5b", max_tokens=1000, temperature=0.3):
        """
        Generate text using the specified model.
        
        This method automatically ensures that the model is loaded.
        
        Returns:
            str or None: The generated text response or None if an error occurred.
        """
        if not self.ensure_model_loaded(model_name):
            logger.error(f"Model {model_name} is not available")
            return None
        try:
            # Use the GenerateAPI to generate text
            options = {"num_predict": max_tokens, "temperature": temperature}  # Use num_predict instead of num_tokens
            result = self.generate_api.generate(prompt=prompt, stream=False, options=options)  # Disable streaming
            logger.info(f"Generated text using model {model_name}")
            return result.response
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None

    def correct_ocr_text(self, text, model_name="deepseek-r1:1.5b"):
        """
        Correct OCR text using an Ollama model.
        
        Constructs a prompt instructing the model to fix common OCR errors and return just the corrected text.
        
        Returns:
            str: The corrected OCR text or the original text if an error occurred.
        """
        if not self.ensure_model_loaded(model_name):
            logger.error(f"Model {model_name} is not available")
            return text

        prompt = f"""Korrigiere den folgenden OCR-Text und behebe Fehler, die bei der Texterkennung entstanden sein könnten.
Liefere nur den korrigierten Text zurück, ohne weitere Erklärungen.

OCR-Text:
{text}

Korrigierter Text:"""
        try:
            try:
                result = self.generate_api.generate(prompt=prompt, stream=False, format="json")

                logger.info("OCR text correction completed.")
                return result.response
            except ConnectionError as conn_err:
                logger.error(f"Ollama server connection failed: {conn_err}")
                logger.warning(f"Please check if Ollama server is running at {self.base_url}")
                logger.warning("OCR text correction skipped due to Ollama server connection failure.")
                return text  # Gebe den ursprünglichen Text zurück, wenn keine Verbindung möglich ist
                
        except Exception as e:
            logger.error(f"Error correcting OCR text: {e}")
            logger.warning("Returning original text due to correction error.")
            return text  # Gebe den ursprünglichen Text zurück statt einer Fehlermeldung

# Global instance of the service for convenience.
ollama_service = OllamaService()

def correct_ocr_with_ollama(text):
    """Convenience function to correct OCR text using the DeepSeek model."""
    return ollama_service.correct_ocr_text(text)
