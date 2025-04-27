import json
from logging import root
from pathlib import Path
import requests
from .custom_logger import get_logger
import subprocess, time, logging, shutil, sys
import os
logger = get_logger(__name__)

class OllamaService:
    """Service to interact with a locally running Ollama instance."""
    
    OLLAMA_BIN = shutil.which("ollama") or "ollama"

    def __init__(self,
                 base_url: str = "http://127.0.0.1:11434",
                 model_name: str = "deepseek-r1:1.5b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._logger = get_logger(__name__)
        #self.start_server_with_devenv()


        if not self.probe_server():
            self.wait_until_ready()           # <- BLOCKS until the API replies

        self.ensure_model_is_running()        # <- lightweight, non-blocking
        self._logger.info("Ollama ready on %s", self.base_url)

    # --------------------------------------------------------------------- #
    # 1) Probe   — return True/False,   NO side effects
    # --------------------------------------------------------------------- #
    def probe_server(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/version", timeout=timeout)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    # --------------------------------------------------------------------- #
    # 2) Start   — only start,          NO polling / waiting
    # --------------------------------------------------------------------- #
    def start_server_with_devenv(self) -> None:
        self._logger.info("Starting Ollama daemon via `devenv up -d` …")
        env = os.environ.copy()
        env["DEVENV_RUNTIME"] = "./lx-anonymizer"
        subprocess.run(["devenv", "up", "-d"], env=env, check=True)  # raises on failure

    # --------------------------------------------------------------------- #
    # 3) Wait    — poll until probe succeeds (max_wait seconds)
    # --------------------------------------------------------------------- #
    def wait_until_ready(self, interval: int = 2, max_wait: int = 60) -> None:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self.probe_server():
                return
            self._logger.info("Waiting for Ollama …")
            time.sleep(interval)
        raise TimeoutError("Ollama did not start within %ss" % max_wait)

    # --------------------------------------------------------------------- #
    # 4) Ensure model is loaded (unchanged logic, but idempotent)
    # --------------------------------------------------------------------- #
    def ensure_model_is_running(self) -> None:
        out = subprocess.run([self.OLLAMA_BIN, "ps"],
                             capture_output=True, text=True, check=True).stdout
        if self.model_name in out:
            return                                  # already there

        self._logger.info("Launching model %s …", self.model_name)
        subprocess.Popen([self.OLLAMA_BIN, "run", self.model_name],
                         stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def generate_text(self, prompt, max_tokens=1000, temperature=0.3):
        """Generate text using the specified model."""
        if not self.probe_server():
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
        if not self.probe_server():
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
        if not self.probe_server():
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
    
    def extract_report_meta(self, text):
        """Extract metadata from the report text."""
        prompt = f"""Bitte extrahiere die Patientendaten aus dem folgenden Text. Gib als Antwort **nur** die extrahierten Daten zurück, ohne Erklärungen oder Kommentare.
Hier ist der Text:
{text}
Extrahierte Daten:"""
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
                return result["response"]
            else:
                logger.error(f"Error extracting report metadata: {response.status_code} - {response.text}")
                return None  # Return None on error
        except Exception as e:
            logger.error(f"Error extracting report metadata: {e}")
            return None

# Global instance of the service for convenience.
ollama_service = OllamaService()


