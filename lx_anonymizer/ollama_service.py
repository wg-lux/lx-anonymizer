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
import urllib3
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
    # Get log level from environment, default to INFO
    LOG_LEVEL = getattr(logging, os.environ.get("OLLAMA_LOG_LEVEL", "INFO"))
    # Log file path
    OLLAMA_LOG_FILE = os.environ.get("OLLAMA_LOG_FILE", "ollama_service.log")
    
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
        
        # Setup enhanced logging
        self._setup_logging()
        
        self._ollama_process = None
        self._log_file = None
        
        # Configure HTTP session with retries
        self._setup_http_session()
        
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

    def _setup_logging(self):
        """Configure enhanced logging for the Ollama service."""
        # Add a stream handler with a nice format if not already present
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) 
                  for h in self._logger.handlers):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        
        # Set log level from environment variable
        self._logger.setLevel(self.LOG_LEVEL)
        self._logger.debug("Ollama service logging initialized at level %s", 
                          logging.getLevelName(self._logger.level))

    def _setup_http_session(self):
        """Configure HTTP session with retries and timeouts."""
        self._http = requests.Session()
        
        # Configure retry strategy
        retry_strategy = urllib3.Retry(
            total=3,                   # Maximum number of retries
            backoff_factor=1,          # Factor by which to multiply delay
            status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
            allowed_methods=["GET", "POST"]  # HTTP methods to retry
        )
        
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self._http.mount("http://", adapter)
        self._http.mount("https://", adapter)

    def _stream_to_logger(self, pipe, log_level=logging.INFO):
        """Stream subprocess output to logger."""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    line = line.strip()
                    if line:
                        self._logger.log(log_level, "Ollama: %s", line)
            self._logger.debug("Ollama process stream closed")
        except Exception as e:
            self._logger.error("Error reading from Ollama process pipe: %s", e)

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
        
        # Close log file if open
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
            self._log_file = None
    
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
            # Setup log file for Ollama process output
            self._log_file = open(self.OLLAMA_LOG_FILE, "ab", 0)  # Open in unbuffered append mode
            
            # Run ollama serve as a background process
            self._logger.info(f"Running: {self.OLLAMA_BIN} serve")
            
            # Create a detached process that won't be killed when the parent exits
            self._ollama_process = subprocess.Popen(
                [self.OLLAMA_BIN, "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True      # Create a new process group
            )
            
            # Start a background thread to read and log the output
            threading.Thread(
                target=self._stream_to_logger,
                args=(self._ollama_process.stdout,),
                daemon=True,
            ).start()
            
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
            # First, check if the model is already running using the API
            self._logger.debug(f"Checking if model {self.model_name} is available...")
            response = self._http.get(f"{self.base_url}/api/tags", timeout=(3.05, 10))
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_available = any(m.get("name") == self.model_name for m in models)
                
                if model_available:
                    self._logger.info(f"Model {self.model_name} is available")
                    
                    # Warm up the model with a simple generate request
                    self._warm_up_model()
                    self._model_checked = True
                    return
            
            # Model not loaded, need to download
            self._logger.info(f"Pulling model {self.model_name}...")
            
            # Use the API to pull the model
            response = self._http.post(
                f"{self.base_url}/api/pull", 
                json={"name": self.model_name},
                timeout=(3.05, 600)  # Allow up to 10 minutes for a large model download
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to pull model '{self.model_name}': {response.text}"
                self._logger.error(error_msg)
                raise self.ModelError(error_msg)
            
            self._logger.info(f"Model {self.model_name} successfully pulled")
            
            # Warm up the model with a simple generate request
            self._warm_up_model()
            self._model_checked = True
            
        except self.OllamaError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            error_msg = f"Unexpected error ensuring model: {e}"
            self._logger.error(error_msg)
            raise self.ModelError(error_msg)

    def _warm_up_model(self):
        """Send a simple request to warm up the model."""
        try:
            self._logger.debug(f"Warming up model {self.model_name}...")
            response = self._http.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1}  # Minimal token generation
                },
                timeout=(3.05, 30)  # 30 second timeout for first load
            )
            
            if response.status_code == 200:
                self._logger.info(f"Model {self.model_name} is ready")
            else:
                self._logger.warning(f"Model warm-up returned status {response.status_code}: {response.text}")
        except Exception as e:
            self._logger.warning(f"Error during model warm-up: {e}")
            # Don't raise here, just log the warning

    # --------------------------------------------------------------------- #
    # 6) Health and diagnostics                                             #
    # --------------------------------------------------------------------- #
    def health(self) -> dict:
        """
        Get health status of the Ollama service and models.
        
        Returns:
            Dictionary with health information or raises Unavailable exception
        
        Raises:
            Unavailable: If Ollama server is not running
        """
        if not self.probe_server():
            raise self.Unavailable("Ollama server is not running")
            
        health_info = {
            "status": "healthy",
            "server_url": self.base_url,
            "default_model": self.model_name,
            "version": None,
            "models": [],
            "gpu": False
        }
        
        try:
            # Get version
            response = self._http.get(f"{self.base_url}/api/version", timeout=(3.05, 5))
            if response.status_code == 200:
                health_info["version"] = response.json().get("version")
                
            # Get models
            response = self._http.get(f"{self.base_url}/api/tags", timeout=(3.05, 5))
            if response.status_code == 200:
                models = []
                for model in response.json().get("models", []):
                    models.append({
                        "name": model.get("name"),
                        "size": model.get("size"),
                        "modified_at": model.get("modified_at"),
                        "family": model.get("model"),
                    })
                health_info["models"] = models
                
            return health_info
        except Exception as e:
            self._logger.error(f"Error getting health status: {e}")
            raise self.Unavailable(f"Failed to get health status: {e}")

    # --------------------------------------------------------------------- #
    # 7) Text generation helpers                                            #
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
            response = self._http.post(
                f"{self.base_url}/api/generate", 
                json=data,
                timeout=(3.05, 120)  # 2-minute timeout for generation
            )

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
        """
        Split text into smaller chunks, improved to handle large words and UTF-8.
        
        Args:
            text: The text to split
            max_chunk_size: Maximum chunk size in bytes
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Convert max_chunk_size from characters to bytes if needed
        if isinstance(text, str):
            encoding = 'utf-8'
            text_bytes = text.encode(encoding)
        else:
            text_bytes = text
            encoding = 'utf-8'  # Default encoding for decoding
        
        chunks = []
        start = 0
        text_len = len(text_bytes)
        
        while start < text_len:
            end = min(start + max_chunk_size, text_len)
            
            # Ensure we don't cut in the middle of a UTF-8 character
            if end < text_len:
                # Move back until we find a valid character boundary
                while end > start and (text_bytes[end] & 0xC0) == 0x80:
                    end -= 1
                    
            # Ensure we don't cut in the middle of a word if possible
            if end < text_len:
                # Try to find a space or newline to break at
                space_pos = text_bytes.rfind(b' ', start, end)
                newline_pos = text_bytes.rfind(b'\n', start, end)
                break_pos = max(space_pos, newline_pos)
                
                if break_pos > start:
                    end = break_pos + 1  # Include the space/newline in the chunk
            
            # Extract the chunk and decode it back to string
            chunk_bytes = text_bytes[start:end]
            chunk = chunk_bytes.decode(encoding, errors='replace')
            chunks.append(chunk)
            
            start = end
            
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
            }
            response = self._http.post(
                f"{self.base_url}/api/generate", 
                json=data,
                timeout=(3.05, 120)  # 2-minute timeout
            )

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
                }
                response = self._http.post(
                    f"{self.base_url}/api/generate", 
                    json=data,
                    timeout=(3.05, 120)
                )

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
            }
            response = self._http.post(
                f"{self.base_url}/api/generate", 
                json=data,
                timeout=(3.05, 120)
            )

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
    # 8) Helper methods                                                     #
    # --------------------------------------------------------------------- #
    def _strip_code_fences(self, text):
        """Strip markdown code fences from the text if present."""
        if not text:
            return text
            
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


