import requests
import json
import time
import subprocess
from pathlib import Path
from .custom_logger import get_logger

logger = get_logger(__name__)

class OllamaService:
    """Service to interact with a locally running Ollama instance"""
    
    def __init__(self, server_url="http://127.0.0.1:11434"):
        self.server_url = server_url
        self.api_endpoint = f"{server_url}/api/generate"
        self.models_endpoint = f"{server_url}/api/tags"
        self.initialized = False
        
    def ensure_server_running(self):
        """Check if Ollama server is running, attempt to start if not"""
        try:
            response = requests.get(f"{self.server_url}/api/tags")
            if response.status_code == 200:
                logger.info("Ollama server is running")
                self.initialized = True
                return True
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama server not detected, attempting to start...")
            
        # Try to start Ollama server
        try:
            # Start Ollama as a subprocess
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it time to start
            time.sleep(3)
            
            # Check if it's running
            for attempt in range(5):
                try:
                    response = requests.get(f"{self.server_url}/api/tags")
                    if response.status_code == 200:
                        logger.info("Ollama server started successfully")
                        self.initialized = True
                        return True
                except requests.exceptions.ConnectionError:
                    logger.info(f"Waiting for Ollama server to start (attempt {attempt+1}/5)")
                    time.sleep(2)
            
            logger.error("Failed to start Ollama server automatically")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ollama server: {e}")
            return False
    
    def list_models(self):
        """List all models available in the local Ollama instance"""
        if not self.ensure_server_running():
            return []
        
        try:
            response = requests.get(self.models_endpoint)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                logger.error(f"Failed to list models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def ensure_model_loaded(self, model_name="deepseek-coder:1.3b"):
        """Ensure the specified model is loaded in Ollama"""
        if not self.ensure_server_running():
            return False
        
        # Check if model is already available
        available_models = self.list_models()
        if model_name in available_models:
            logger.info(f"Model {model_name} is already loaded")
            return True
        
        # Try to pull the model
        logger.info(f"Pulling model {model_name} (this may take a while)...")
        try:
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for the pull to complete
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Successfully pulled model {model_name}")
                return True
            else:
                logger.error(f"Failed to pull model {model_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def generate_text(self, prompt, model_name="deepseek-coder:1.3b", max_tokens=1000, temperature=0.3):
        """Generate text using the specified model"""
        if not self.ensure_model_loaded(model_name):
            logger.error(f"Model {model_name} is not available")
            return None
            
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(
                self.api_endpoint, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=60  # 60 seconds timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Error generating text: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return None
    
    def correct_ocr_text(self, text, model_name="deepseek-r1:1.5b"):
        """
        Korrigiert erkannten OCR-Text mit einem Ollama-Modell.
        
        Args:
            text (str): Der zu korrigierende OCR-Text
            model_name (str): Name des zu verwendenden Ollama-Modells
            
        Returns:
            str: Der korrigierte Text
        """
        prompt = f"""Korrigiere den folgenden OCR-Text und behebe Fehler, die bei der Texterkennung entstanden sein könnten.
Liefere nur den korrigierten Text zurück, ohne weitere Erklärungen.

OCR-Text:
{text}

Korrigierter Text:"""
        
        options = {
            "temperature": 0.1,  # Niedrige Temperatur für deterministische Ergebnisse
            "top_p": 0.9,
            "num_predict": 2048,  # Maximale Anzahl zu generierender Token
        }
        
        result = self.run_ollama(prompt, model_name, options)
        
        # Wenn nötig, zusätzliche Nachbearbeitung des Ergebnisses hier einfügen
        
        return result
    
    def run_ollama(self, prompt, model_name="deepseek-r1:1.5b", options=None):
        """
        Führt einen Ollama-Befehl mit einem gegebenen Prompt und Modell aus.
        
        Args:
            prompt (str): Der Prompt-Text, der an das Modell gesendet wird
            model_name (str): Name des zu verwendenden Ollama-Modells
            options (dict): Optionale Parameter für den Ollama-Aufruf
            
        Returns:
            str: Die Antwort des Modells
        """
        try:
            cmd = ["ollama", "run", model_name]
            
            # Wenn Optionen angegeben wurden, diese als JSON formatieren
            if options:
                cmd.extend(["--options", json.dumps(options)])
            
            # Ollama-Prozess starten und Prompt übergeben
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Prompt senden und Ausgabe erfassen
            stdout, stderr = process.communicate(input=prompt)
            
            if process.returncode != 0:
                logger.error(f"Ollama process failed with code {process.returncode}: {stderr}")
                return f"Error: {stderr}"
            
            return stdout.strip()
            
        except Exception as e:
            logger.error(f"Error running Ollama: {e}")
            return f"Error: {str(e)}"
    
    def ensure_model_available(self, model_name="deepseek-r1:1.5b"):
        """
        Stellt sicher, dass das angegebene Modell lokal verfügbar ist.
        
        Args:
            model_name (str): Name des zu prüfenden Ollama-Modells
            
        Returns:
            bool: True wenn erfolgreich, False wenn ein Fehler auftritt
        """
        try:
            logger.info(f"Ensuring Ollama model {model_name} is available")
            process = subprocess.run(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            logger.info(f"Ollama model check complete: {process.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama model: {e}")
            return False

# Initialize global instance
ollama_service = OllamaService()

# Convenience function
def correct_ocr_with_ollama(text):
    """Correct OCR text using locally running Ollama with DeepSeek model"""
    return ollama_service.correct_ocr_text(text)
