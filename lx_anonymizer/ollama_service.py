import json
import os
import signal
import atexit
import threading
import requests
import time
import logging
import shutil
import subprocess
from pathlib import Path
from functools import lru_cache
from .custom_logger import get_logger

logger = get_logger(__name__)
_lock = threading.Lock()

class OllamaService:
    """Service to interact with a locally running Ollama instance."""
    
    # Custom exception classes for better error handling
    class OllamaError(Exception):
        """Base class for Ollama-related exceptions."""
        pass
    
    class Unavailable(OllamaError):
        """Raised when Ollama server is unavailable or cannot be started."""
        pass
    
    class ModelError(OllamaError):
        """Raised when there's an issue with the Ollama model."""
        pass
    
    class RequestError(OllamaError):
        """Raised when a request to Ollama API fails."""
        pass
    
    # Get configurations from environment variables with sensible defaults
    OLLAMA_BIN = os.environ.get("OLLAMA_BIN") or shutil.which("ollama") or "ollama"
    DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
    DEFAULT_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))
    
    def __init__(self,
                 base_url: str = None,
                 model_name: str = None):
        """
        Initialize the OllamaService.
        
        Args:
            base_url: Override the default base URL (normally read from OLLAMA_PORT)
            model_name: Override the default model name (normally read from OLLAMA_MODEL)
        """
        # Use environment variables if not explicitly provided
        if base_url is None:
            self.base_url = f"http://127.0.0.1:{self.DEFAULT_PORT}"
        else:
            self.base_url = base_url.rstrip("/")
            
        self.model_name = model_name or self.DEFAULT_MODEL
        self._logger = get_logger(__name__)
        self._ollama_process = None
        self._http = requests.Session()  # Reuse HTTP connections
        self._model_checked = False
        
        # First check if server is already running
        if not self.probe_server():
            self._logger.info("Ollama server not running, starting it...")
            self.start_server()

        # Wait for it to be ready
        if not self.probe_server():
            self.wait_until_ready()
        
        # Register shutdown handler
        atexit.register(self.stop)

    # --------------------------------------------------------------------- #
    # 1) Lifecycle Management                                                #
    # --------------------------------------------------------------------- #
    def stop(self):
        """
        Explicitly stop the Ollama process and any child processes.
        """
        if self._ollama_process:
            try:
                self._logger.info("Stopping Ollama process group...")
                pgid = os.getpgid(self._ollama_process.pid)
                os.killpg(pgid, signal.SIGTERM)
                
                # Give it some time to terminate gracefully
                time.sleep(2)
                
                # Check if it's still running and force kill if needed
                if self._ollama_process.poll() is None:
                    self._logger.warning("Ollama process didn't terminate gracefully, forcing kill...")
                    os.killpg(pgid, signal.SIGKILL)
                    
                self._ollama_process = None
                self._logger.info("Ollama process group stopped.")
            except Exception as e:
                self._logger.error(f"Error shutting down Ollama process: {e}")
    
    def __del__(self):
        """
        Cleanup resources when object is destroyed.
        Note: This is not guaranteed to run, use stop() explicitly.
        """
        try:
            self.stop()
        except Exception as e:
            # Avoid raising exceptions in destructor
            self._logger.error(f"Error in __del__: {e}")

    # --------------------------------------------------------------------- #
    # 2) Probe   — return True/False,   NO side effects                     #
    # --------------------------------------------------------------------- #
    def probe_server(self, timeout: float = 2.0) -> bool:
        """
        Check if Ollama server is responsive.
        
        Args:
            timeout: Seconds to wait for response
            
        Returns:
            True if server is responding, False otherwise
        """
        try:
            r = self._http.get(f"{self.base_url}/api/version", timeout=timeout)
            return r.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False

    # --------------------------------------------------------------------- #
    # 3) Start   — only start,          NO polling / waiting                #
    # --------------------------------------------------------------------- #
    def start_server(self) -> None:
        """
        Start Ollama server as a background process.
        
        Raises:
            Unavailable: If Ollama server fails to start.
        """
        self._logger.info("Starting Ollama daemon...")
        
        try:
            # Run ollama serve as a background process
            self._logger.info(f"Running: {self.OLLAMA_BIN} serve")
            
            # Create a detached process that won't be killed when the parent exits
            self._ollama_process = subprocess.Popen(
                [self.OLLAMA_BIN, "serve"],
                stdout=subprocess.DEVNULL,  # Avoid pipe buffer blockage
                stderr=subprocess.DEVNULL,  # Avoid pipe buffer blockage
                start_new_session=True      # Create a new process group
            )
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if process is still running
            if self._ollama_process.poll() is not None:
                error_message = f"Failed to start Ollama. Exit code: {self._ollama_process.returncode}"
                self._logger.error(error_message)
                raise self.Unavailable(error_message)
                
            self._logger.info(f"Ollama server process started with PID {self._ollama_process.pid}")
            
        except Exception as e:
            self._logger.error(f"Error starting Ollama: {e}")
            raise self.Unavailable(f"Failed to start Ollama: {e}")

    # --------------------------------------------------------------------- #
    # 4) Wait    — poll until probe succeeds (max_wait seconds)             #
    # --------------------------------------------------------------------- #
    def wait_until_ready(self, interval: int = 2, max_wait: int = 60) -> None:
        """
        Wait until the Ollama server is responsive.
        
        Args:
            interval: Seconds between probe attempts
            max_wait: Maximum seconds to wait
            
        Raises:
            Unavailable: If Ollama doesn't start within max_wait seconds
        """
        deadline = time.time() + max_wait
        while time.time() < deadline:
            if self.probe_server():
                self._logger.info("Ollama server is now responding")
                return
            self._logger.info("Waiting for Ollama...")
            time.sleep(interval)
        
        # If we get here, it timed out
        # If we started the process, kill it before raising timeout
        if self._ollama_process:
            self._logger.error("Ollama server failed to start in time, terminating process")
            self.stop()
            
        raise self.Unavailable(f"Ollama did not start within {max_wait}s")

    # --------------------------------------------------------------------- #
    # 5) Ensure model is loaded and ready                                   #
    # --------------------------------------------------------------------- #
    def ensure_model_is_running(self, force_check=False) -> None:
        """
        Ensure the model is downloaded and running.
        
        Args:
            force_check: Skip cache and check model availability
            
        Raises:
            ModelError: If the model cannot be loaded
        """
        # Skip the check if we've already confirmed model is available
        if self._model_checked and not force_check:
            return
            
        # Verify server is responding first
        if not self.probe_server():
            raise self.Unavailable("Ollama server is not running")

        try:
            # Check if model is already loaded
            out = subprocess.run(
                [self.OLLAMA_BIN, "ps"],
                capture_output=True, text=True, check=True
            ).stdout
            
            if self.model_name in out:
                self._logger.info(f"Model {self.model_name} is already running")
                self._model_checked = True
                return
                
            # Model not loaded, need to download
            self._logger.info(f"Ensuring model {self.model_name} is available...")
            
            # Pull the model first
            pull_process = subprocess.run(
                [self.OLLAMA_BIN, "pull", self.model_name],
                capture_output=True, text=True
            )
            
            # Check for errors, but allow if model already exists (code 7)
            if pull_process.returncode != 0 and pull_process.returncode != 7:
                error_msg = f"Failed to pull model '{self.model_name}': {pull_process.stderr.strip()}"
                self._logger.error(error_msg)
                raise self.ModelError(error_msg)
                
            self._logger.info(f"Running model {self.model_name}...")
            # Now run the model
            subprocess.Popen(
                [self.OLLAMA_BIN, "run", self.model_name],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                start_new_session=True  # Create a new process group
            )
            
            # Mark as checked
            self._model_checked = True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Error checking model status: {e}"
            self._logger.error(error_msg)
            raise self.ModelError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error ensuring model: {e}"
            self._logger.error(error_msg)
            raise self.ModelError(error_msg)

    # --------------------------------------------------------------------- #
    # 6) Text generation helpers                                            #
    # --------------------------------------------------------------------- #
    def generate_text(self, prompt, max_tokens=1000, temperature=0.3):
        """
        Generate text using the specified model.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Controls randomness (0.0-1.0)
            
        Returns:
            Generated text or None if error
            
        Raises:
            Unavailable: If Ollama server is not running
            RequestError: If the API request fails
        """
        self.ensure_model_is_running()

        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            response = self._http.post(f"{self.base_url}/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Generated text using model {self.model_name}")
                return result["response"]
            else:
                error_msg = f"Error generating text: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise self.RequestError(error_msg)
                
        except self.OllamaError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Error generating text: {e}"
            logger.error(error_msg)
            raise self.RequestError(error_msg)

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
        """
        Correct OCR text using an Ollama model.
        
        Args:
            text: The OCR text to correct
            
        Returns:
            Corrected text or original text on error
            
        Raises:
            Unavailable: If Ollama server is not running
            RequestError: If the API request fails
        """
        self.ensure_model_is_running()

        # Ensure text is a string
        if not isinstance(text, str):
            logger.error(f"Expected string for text correction but got {type(text)}")
            # Try to convert to string if possible
            try:
                text = str(text)
            except Exception as e:
                logger.error(f"Could not convert input to string: {e}")
                return ""

        # Use safe string formatting with braces escaped
        prompt = (
            "Bitte korrigiere den folgenden OCR-Text. Gib als Antwort **nur** den vollständig korrigierten Text "
            "(ohne Erklärungen oder Kommentare) zurück, der grammatikalisch korrekt, flüssig und konsistent formatiert ist.\n"
            "Hier ist der OCR-Text:\n"
            f"{text!s}\n"  # !s ensures string conversion and escapes braces
            "Korrigierter Text:"
        )

        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                # Remove format:json to avoid double JSON encoding
            }
            response = self._http.post(f"{self.base_url}/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                corrected_text = result["response"]
                
                # Strip any potential code fences
                corrected_text = self._strip_code_fences(corrected_text)
                return corrected_text
            else:
                error_msg = f"Error correcting OCR text: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise self.RequestError(error_msg)
                
        except self.OllamaError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Error correcting OCR text: {e}"
            logger.error(error_msg)
            raise self.RequestError(error_msg)

    def correct_ocr_text_in_chunks(self, text):
        """
        Correct OCR text in chunks, for handling long texts.
        
        Args:
            text: The OCR text to correct
            
        Returns:
            Corrected text or original text on error
            
        Raises:
            Unavailable: If Ollama server is not running
            RequestError: If the API request fails
        """
        self.ensure_model_is_running()

        # Split the text into chunks
        text_chunks = self.split_text_into_chunks(text)
        corrected_chunks = []

        # Process each chunk independently
        for chunk in text_chunks:
            # Use safe string formatting
            prompt = (
                "Bitte korrigiere den folgenden OCR-Text. Gib als Antwort **nur** den vollständig korrigierten Text "
                "(ohne Erklärungen oder Kommentare) zurück, der grammatikalisch korrekt, flüssig und konsistent formatiert ist.\n"
                "Hier ist der OCR-Text:\n"
                f"{chunk!s}\n"
                "Korrigierter Text:"
            )

            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    # Remove format:json
                }
                response = self._http.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
                    result = response.json()
                    corrected_text = self._strip_code_fences(result["response"])
                    corrected_chunks.append(corrected_text)
                else:
                    error_msg = f"Error correcting OCR text chunk: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    corrected_chunks.append(chunk)  # Use original chunk on error
            except Exception as e:
                logger.error(f"Error correcting OCR text chunk: {e}")
                corrected_chunks.append(chunk)  # Use original chunk on error

        # Combine corrected chunks back into a single string
        corrected_text = " ".join(corrected_chunks)
        logger.info("OCR text correction completed.")
        return corrected_text
    
    def extract_report_meta(self, text):
        """
        Extract metadata from the report text.
        
        Args:
            text: The report text to analyze
            
        Returns:
            Extracted metadata or None on error
            
        Raises:
            Unavailable: If Ollama server is not running
            RequestError: If the API request fails
        """
        self.ensure_model_is_running()

        # Safe string formatting
        prompt = (
            "Bitte extrahiere die Patientendaten aus dem folgenden Text. "
            "Gib als Antwort **nur** die extrahierten Daten zurück, ohne Erklärungen oder Kommentare.\n"
            "Hier ist der Text:\n"
            f"{text!s}\n"
            "Extrahierte Daten:"
        )

        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                # Remove format:json
            }
            response = self._http.post(f"{self.base_url}/api/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                return self._strip_code_fences(result["response"])
            else:
                error_msg = f"Error extracting report metadata: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise self.RequestError(error_msg)
                
        except self.OllamaError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Error extracting report metadata: {e}"
            logger.error(error_msg)
            raise self.RequestError(error_msg)

    # --------------------------------------------------------------------- #
    # 7) Helper methods                                                     #
    # --------------------------------------------------------------------- #
    def _strip_code_fences(self, text):
        """Strip markdown code fences from the text if present."""
        if text.startswith("```") and "```" in text[3:]:
            # Extract content between code fences
            start_idx = text.find("\n", 3)
            end_idx = text.rfind("```")
            if start_idx != -1 and end_idx > start_idx:
                return text[start_idx+1:end_idx].strip()
        # Return original if no code fence pattern found
        return text


# Thread-safe lazy initialization of a global singleton
@lru_cache(maxsize=1)
def get_ollama_service():
    """
    Get or create the OllamaService instance in a thread-safe manner.
    """
    with _lock:
        return OllamaService()

# For backward compatibility
try:
    ollama_service = get_ollama_service()
except Exception as e:
    logger.error(f"Failed to initialize OllamaService: {e}")
    # Create a dummy service that will raise Unavailable when methods are called
    class DummyOllamaService:
        def __getattr__(self, name):
            def _unavailable(*args, **kwargs):
                raise OllamaService.Unavailable("OllamaService failed to initialize")
            return _unavailable
    
    ollama_service = DummyOllamaService()


