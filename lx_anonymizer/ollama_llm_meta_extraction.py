"""
Optimierte Ollama LLM Metadaten-Extraktion mit leichtgewichtigen Modellen und REST API.

Diese Implementierung basiert auf den Best Practices:
1. Verwendung von instruction-tuned, quantisierten Modellen für bessere Performance
2. Direkte REST API Verwendung statt Python Client für bessere Kontrolle
3. Fail-safe Model Factory mit automatischem Fallback
4. Strukturierte Ausgabe mit JSON Schema Validation
"""

import json
import logging
import requests
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_fixed
import time

from .schema import PatientMeta
from .spacy_regex import PatientDataExtractorLg

# Konfiguriere Logging
logger = logging.getLogger(__name__)

DEFAULT_EXTRACTION_MODEL = "qwen2.5:1.5b-instruct"
_REGEX_FALLBACK_LOG_MARKER = "metadata_extraction_path=regex_fallback"
_PYDANTIC_SUCCESS_LOG_MARKER = "metadata_extraction_path=pydantic_success"

class PatientMetadata(BaseModel):
    """Pydantic-Modell für Patientenmetadaten mit Validierung."""
    patient_name: str
    patient_age: int
    examination_date: str
    gender: str
    additional_info: Optional[str] = None

class ModelConfig:
    """Konfiguration für verfügbare Modelle mit Prioritäten."""
    
    # Priorisierte Modelliste: leichte, instruction-tuned Modelle zuerst
    MODELS = [
                {
            "name": "deepseek-r1:1.5b",
            "priority": 1,
            "timeout": 60,
            "description": "Deepseek R1 - nur als letzter Fallback (reasoning model, langsam)"
        },
        {
            "name": "llama3.2:1b", 
            "priority": 2,
            "timeout": 20,
            "description": "Llama 3.2 1B - kompakt und effizient"
        },
        {
            "name": "qwen2.5:1.5b-instruct",
            "priority": 3,
            "timeout": 15,
            "description": "Qwen 2.5 1.5B instruction-tuned - optimal für strukturierte Extraktion"
        },
        {
            "name": "phi3.5:3.8b-mini-instruct-q4_K_M",
            "priority": 4,
            "timeout": 25,
            "description": "Phi 3.5 Mini quantisiert - gute Balance aus Größe und Qualität"
        }
    ]
    
    @classmethod
    def get_models_by_priority(cls) -> List[Dict[str, Any]]:
        """Gibt Modelle sortiert nach Priorität zurück."""
        return sorted(cls.MODELS, key=lambda x: x["priority"])

class OllamaOptimizedExtractor:
    """
    Optimierte Ollama-Integration für Metadaten-Extraktion.
    
    Verwendet REST API direkt und implement ein fail-safe Modell-System.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/api/chat"
        self.available_models = self._check_available_models()
        self.current_model = None
        self._initialize_best_model()
    
    def _check_available_models(self) -> List[str]:
        """Überprüft, welche Modelle verfügbar sind."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            return []
        except Exception as e:
            logger.warning(f"Konnte verfügbare Modelle nicht abfragen: {e}")
            return []
    
    def _initialize_best_model(self):
        """Initialisiert das beste verfügbare Modell."""
        for model_config in ModelConfig.get_models_by_priority():
            if model_config["name"] in self.available_models:
                self.current_model = model_config
                logger.info(f"Verwende Modell: {model_config['name']} - {model_config['description']}")
                return
        
        # Fallback falls kein konfiguriertes Modell verfügbar ist
        if self.available_models:
            fallback_model = {
                "name": self.available_models[0],
                "priority": 999,
                "timeout": 30,
                "description": "Fallback-Modell"
            }
            self.current_model = fallback_model
            logger.warning(f"Verwende Fallback-Modell: {fallback_model['name']}")
        else:
            raise RuntimeError("Keine Ollama-Modelle verfügbar!")
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Erstellt einen optimierten Prompt für Metadaten-Extraktion."""
        return f"""Extrahiere die Patientenmetadaten aus dem folgenden Text und gib sie im exakten JSON-Format zurück:

Text: {text}

Antworte nur mit dem JSON-Objekt, ohne zusätzliche Erklärungen oder Markdown-Formatierung.
Verwende für gender: "male", "female" oder "unknown".
Verwende für examination_date das Format aus dem Text (DD.MM.YYYY wenn möglich).
"""
    
    def _create_json_schema(self) -> Dict[str, Any]:
        """Erstellt das JSON-Schema für strukturierte Ausgabe."""
        return {
            "type": "object",
            "properties": {
                "patient_name": {"type": "string"},
                "patient_age": {"type": "integer"},
                "examination_date": {"type": "string"},
                "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                "additional_info": {"type": "string"}
            },
            "required": ["patient_name", "patient_age", "examination_date", "gender"]
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Macht API-Request mit Retry-Logik."""
        try:
            timeout = self.current_model.get("timeout", 30)
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API-Fehler {response.status_code}: {response.text}")
                raise requests.RequestException(f"HTTP {response.status_code}")
                
        except requests.Timeout:
            logger.error(f"Timeout bei Modell {self.current_model['name']}")
            raise
        except Exception as e:
            logger.error(f"API-Request fehlgeschlagen: {e}")
            raise
    
    def _try_next_model(self) -> bool:
        """Wechselt zum nächsten verfügbaren Modell."""
        current_priority = self.current_model.get("priority", 0)
        
        for model_config in ModelConfig.get_models_by_priority():
            if (model_config["priority"] > current_priority and 
                model_config["name"] in self.available_models):
                
                old_model = self.current_model["name"]
                self.current_model = model_config
                logger.info(f"Wechsle von {old_model} zu {model_config['name']}")
                return True
        
        logger.error("Keine weiteren Modelle verfügbar")
        return False
    
    def extract_metadata(self, text: str) -> Optional[PatientMetadata]:
        """
        Extrahiert Patientenmetadaten aus Text mit fail-safe Modell-System.
        
        Args:
            text: Input-Text zur Extraktion
            
        Returns:
            PatientMetadata Objekt oder None bei Fehler
        """
        max_model_attempts = len([m for m in ModelConfig.MODELS if m["name"] in self.available_models])
        
        for attempt in range(max_model_attempts):
            try:
                start_time = time.time()
                
                payload = {
                    "model": self.current_model["name"],
                    "messages": [
                        {
                            "role": "user",
                            "content": self._create_extraction_prompt(text)
                        }
                    ],
                    "stream": False,
                    "format": self._create_json_schema()
                }
                
                logger.info(f"Versuche Extraktion mit {self.current_model['name']}")
                
                response = self._make_api_request(payload)
                content = response["message"]["content"]
                
                # Performance-Metriken loggen
                duration = time.time() - start_time
                token_count = response.get("eval_count", 0)
                logger.info(f"Erfolg in {duration:.2f}s, {token_count} Tokens")
                
                # JSON parsen und validieren
                try:
                    metadata_dict = json.loads(content)
                    metadata = PatientMetadata(**metadata_dict)
                    
                    logger.info(f"Erfolgreich extrahiert: {metadata.patient_name}, "
                              f"Alter: {metadata.patient_age}, Datum: {metadata.examination_date}")
                    
                    return metadata
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"JSON/Validierung fehlgeschlagen: {e}")
                    logger.debug(f"Rohe Antwort: {content}")
                    raise ValueError(f"Ungültige Antwort: {e}")
                
            except Exception as e:
                logger.error(f"Fehler mit Modell {self.current_model['name']}: {e}")
                
                # Versuche nächstes Modell
                if attempt < max_model_attempts - 1:
                    if self._try_next_model():
                        logger.info("Versuche es mit dem nächsten Modell...")
                        continue
                    else:
                        break
        
        logger.error("Alle Modelle fehlgeschlagen")
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Informationen über das aktuelle Modell zurück."""
        return {
            "current_model": self.current_model,
            "available_models": self.available_models,
            "total_models": len(self.available_models)
        }


def _safe_keys_view(data: Optional[Dict[str, Any]]) -> List[str]:
    """Return a deterministic, PHI-safe view of dict keys."""
    if not isinstance(data, dict):
        return []

    try:
        return sorted(data.keys())
    except Exception:
        return []


def _extract_json_block(text: Optional[str]) -> Optional[str]:
    """Extract the first valid JSON object embedded within text."""
    if not text:
        return None

    candidate_start = text.find("{")
    while candidate_start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(candidate_start, len(text)):
            char = text[idx]

            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char == '{':
                depth += 1
            elif char == '}' and depth > 0:
                depth -= 1
                if depth == 0:
                    segment = text[candidate_start: idx + 1]
                    try:
                        json.loads(segment)
                        return segment
                    except json.JSONDecodeError:
                        break

        candidate_start = text.find("{", candidate_start + 1)

    return None


def chat(**payload: Any) -> Dict[str, Any]:
    """Call Ollama's chat endpoint (wrapper for patchability in tests)."""
    import ollama

    return ollama.chat(**payload)


try:
    extractor_instance = PatientDataExtractorLg()
except Exception as exc:  # pragma: no cover - spaCy model might be unavailable in CI
    logger.warning("Failed to initialize PatientDataExtractorLg: %s", exc)

    class _FallbackExtractor:
        def regex_extract_llm_meta(self, text: str) -> Dict[str, Any]:
            return {}

    extractor_instance = _FallbackExtractor()


def _regex_fallback(
    extractor: Any,
    cleaned_json_str: Optional[str],
    raw_response_content: Optional[str],
    model: str,
) -> Dict[str, Any]:
    """Execute the regex fallback extraction path."""
    try:
        source = raw_response_content or cleaned_json_str or ""
        result = extractor.regex_extract_llm_meta(source)
        if not isinstance(result, dict):
            result = {}

        logger.info("%s model=%s", _REGEX_FALLBACK_LOG_MARKER, model)
        logger.debug(
            "Content used for regex extraction available=%s raw_available=%s",
            bool(cleaned_json_str),
            bool(raw_response_content),
        )
        logger.debug("metadata_keys_regex=%s", _safe_keys_view(result))
        return result
    except Exception as exc:
        logger.error(
            "Regex extraction failed: %s; returning empty metadata. model=%s",
            exc,
            model,
        )
        return {}


def extract_meta_ollama(
    text: str,
    *,
    model: str = DEFAULT_EXTRACTION_MODEL,
    extractor: Optional[Any] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Extract patient metadata using Ollama with Pydantic + regex fallback."""

    extractor = extractor or extractor_instance

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical report extraction assistant. "
                    "Return only JSON with patient metadata fields."
                ),
            },
            {"role": "user", "content": text},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = chat(**payload)
    except Exception as exc:
        logger.error("Ollama chat failed: %s", exc)
        return {}

    message = response.get("message") or {}
    raw_content = message.get("content") or ""
    cleaned_json_str = _extract_json_block(raw_content)

    try:
        if not cleaned_json_str:
            raise ValueError("No JSON content returned")

        validated_model = PatientMeta.model_validate_json(cleaned_json_str)
        result = validated_model.model_dump(mode="json")

        logger.info("%s model=%s", _PYDANTIC_SUCCESS_LOG_MARKER, model)
        logger.debug("metadata_keys_pydantic=%s", _safe_keys_view(result))
        return result

    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Pydantic/JSON validation failed: %s; falling back to regex. model=%s",
            exc,
            model,
        )

    return _regex_fallback(extractor, cleaned_json_str, raw_content, model)

# Factory-Funktion für einfache Verwendung
def create_ollama_extractor() -> OllamaOptimizedExtractor:
    """Erstellt eine optimierte Ollama-Extractor-Instanz."""
    return OllamaOptimizedExtractor()

# Beispiel-Verwendung
if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(level=logging.INFO)
    
    # Extractor erstellen
    extractor = create_ollama_extractor()
    
    # Test-Texte
    test_texts = [
        "Herr Müller, 45 Jahre alt, wurde am 15.01.2024 untersucht.",
        "Frau Schmidt, 32 Jahre alt, Untersuchung am 20.02.2024",
        "Patient Klaus Weber (m), 58 Jahre, Termin: 10.03.2024"
    ]
    
    print(f"Modell-Info: {extractor.get_model_info()}")
    print("-" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        
        start_time = time.time()
        metadata = extractor.extract_metadata(text)
        duration = time.time() - start_time
        
        if metadata:
            print(f"✅ Erfolgreich in {duration:.2f}s:")
            print(f"   Name: {metadata.patient_name}")
            print(f"   Alter: {metadata.patient_age}")
            print(f"   Datum: {metadata.examination_date}")
            print(f"   Geschlecht: {metadata.gender}")
        else:
            print(f"❌ Fehlgeschlagen nach {duration:.2f}s")
