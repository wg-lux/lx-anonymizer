import json
from logging import root
from pathlib import Path
import requests
from .custom_logger import get_logger
import subprocess
import time
import subprocess
import os
logger = get_logger(__name__)

class OllamaService:
    """Service to interact with a locally running Ollama instance."""
    
    def __init__(self, base_url="http://127.0.0.1:11434", model_name="deepseek-r1:1.5b"):

        self.base_url = base_url
        self.model_name = model_name
        self.ensure_model_is_running()  # Ensure the model is running during initialization
        logger.info(f"Initialized OllamaService with base_url: {self.base_url}, model: {self.model_name}")
        
    def correct_ocr_with_ollama(self, text):
        """Convenience function to correct OCR text using the DeepSeek model.
        
        It first checks if the Ollama server is running and, if not, starts it using
        the 'devenv up -d' command. The function then proceeds to perform OCR text correction.
        """
        # First check if the server is running
        if not self.is_server_running():
            logger.info("Ollama service is not running. Starting with 'devenv up -d'...")
            try:
                anon_env = os.environ.copy()
                anon_env["DEVENV_RUNTIME"] = "./lx-anonymizer"          
                subprocess.run(["devenv", "up", "-d"], env=anon_env, check=True)
                # Wait for the service to properly start (adjust sleep time as needed)
                time.sleep(10)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start devenv: {e}")
                # Optionally, return the original text if the service fails to start
                return text

        return self.correct_ocr_text(text)

    def is_server_running(self):
        """Check if the Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/version")
            if response.status_code == 200:
                logger.info("Ollama server is running")
                return True
            else:
                logger.error(f"Ollama server is not running. Status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Ollama server is not running: {e}")
            return False

    def ensure_model_is_running(self):
        """Ensure the specified model is running using subprocess."""
        try:
            # Check if the model is already running
            process = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
            if self.model_name in process.stdout:
                logger.info(f"Model {self.model_name} is already running.")
                return True

            # Run the model using subprocess
            logger.info(f"Starting model {self.model_name} in background...")
            subprocess.Popen(["ollama", "run", self.model_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Wait for a short time to allow the model to start
            time.sleep(5)  # Adjust the waiting time as needed

            logger.info(f"Model {self.model_name} started in background.")
            return True
        except Exception as e:
            logger.error(f"Error starting model {self.model_name}: {e}")
            return False

    def generate_text(self, prompt, max_tokens=1000, temperature=0.3):
        """Generate text using the specified model."""
        if not self.is_server_running():
            logger.error("Ollama server is not running")
            return None

        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            response = requests.post(f"{self.base_url}/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Generated text using model {self.model_name}")
                return result["response"]
            else:
                logger.error(f"Error generating text: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None

    def split_text_into_chunks(self, text, max_chunk_size=2048):
        """Split text into smaller chunks."""
        words = text.split()
        chunks = []
        current_chunk = []

        current_size = 0
        for word in words:
            word_size = len(word) + 1  # accounting for space
            if current_size + word_size > max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def correct_ocr_text(self, text):
        """Correct OCR text using an Ollama model."""
        if not self.is_server_running():
            logger.error("Ollama server is not running")
            return text

        # Ensure text is a string
        if not isinstance(text, str):
            logger.error(f"Expected string for text correction but got {type(text)}")
            # Try to convert to string if possible
            try:
                text = str(text)
            except Exception as e:
                logger.error(f"Could not convert input to string: {e}")
                return ""

        prompt = f"""Bitte korrigiere den folgenden OCR-Text. Gib als Antwort **nur** den vollständig korrigierten Text (ohne Erklärungen oder Kommentare) zurück, der grammatikalisch korrekt, flüssig und konsistent formatiert ist.
Hier ist der OCR-Text:
{text}
Korrigierter Text:"""
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
            response = requests.post(f"{self.base_url}/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                corrected_text = result["response"]
                return corrected_text
            else:
                logger.error(f"Error correcting OCR text: {response.status_code} - {response.text}")
                return text  # Return original text on error
        except Exception as e:
            logger.error(f"Error correcting OCR text: {e}")
            return text  # Return original text on exception

    def correct_ocr_text_in_chunks(self, text):# -> Any | Any:
        """Correct OCR text using an Ollama model."""
        if not self.is_server_running():
            logger.error("Ollama server is not running")
            return text

        # Split the text into chunks
        text_chunks = self.split_text_into_chunks(text)
        corrected_chunks = []

        # Process each chunk independently
        for chunk in text_chunks:
            prompt = f"""Bitte korrigiere den folgenden OCR-Text. Gib als Antwort **nur** den vollständig korrigierten Text (ohne Erklärungen oder Kommentare) zurück, der grammatikalisch korrekt, flüssig und konsistent formatiert ist.
Hier ist der OCR-Text:
{chunk}
Korrigierter Text:"""
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                }
                response = requests.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
                    result = response.json()
                    corrected_chunks.append(result["response"])
                else:
                    logger.error(f"Error correcting OCR text: {response.status_code} - {response.text}")
                    corrected_chunks.append(chunk)  # Return original chunk on error
            except Exception as e:
                logger.error(f"Error correcting OCR text: {e}")
                corrected_chunks.append(chunk)  # Return original chunk on error

        # Combine corrected chunks back into a single string
        corrected_text = " ".join(corrected_chunks)
        logger.info("OCR text correction completed.")
        return corrected_text

# Global instance of the service for convenience.
ollama_service = OllamaService()

def correct_ocr_with_ollama(text):
    """Convenience function to correct OCR text using the DeepSeek model."""
    return ollama_service.correct_ocr_text(text)
