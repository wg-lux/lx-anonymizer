"""
Optimierte LLM-Metadaten-Extraktion mit leichtgewichtigen Modellen und REST API.

Diese Implementierung basiert auf den Best Practices:
1. Verwendung von instruction-tuned, quantisierten Modellen für bessere Performance
2. Direkte REST API Verwendung statt Python Client für bessere Kontrolle
3. Fail-safe Model Factory mit automatischem Fallback
4. Strukturierte Ausgabe mit JSON Schema Validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Optional, Self, Sequence, cast

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tenacity import retry, stop_after_attempt, wait_fixed
from lx_dtypes.models.contracts.llm_service import (
    LLMChatMessagePayload,
    LLMChatOllamaPayload,
    LLMChatOllamaOptionsPayload,
    LLMChatOpenAIPayload,
    LLMChatResponsePayload,
)
from lx_dtypes.models.contracts.llm_extractor import (
    LLMEnrichedMetadataPayload,
    LLMFrameContextPayload,
    LLMFrameDataPayload,
    LLMMetadataCacheStatsPayload,
    LLMModelInfoPayload,
    LLMTextTimelineEntryPayload,
    LLMTemporalAnalysisPayload,
    LLMEvaluationResultPayload,
    LLMVllmModelsPayload,
)
from lx_dtypes.models.contracts.text_anonymization import (
    LLMMetadataPayload,
)
from lx_dtypes.models.meta.VideoMeta import FrameCollectionItem
from lx_anonymizer.sensitive_meta_interface import SensitiveMeta

# Konfiguriere Logging
logger = logging.getLogger(__name__)

OLLAMA_GENERATION_OPTIONS = {"temperature": 0, "num_ctx": 8192}

STRICT_METADATA_KEYS = (
    "first_name",
    "last_name",
    "dob",
    "casenumber",
    "examination_date",
)

STRICT_METADATA_TEMPLATE = (
    '{"first_name":null,"last_name":null,'
    '"dob":null,"casenumber":null,"examination_date":null}'
)

LLMChatRequestPayload = LLMChatOllamaPayload | LLMChatOpenAIPayload


class _OllamaModelDetailsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    parent_model: str | None = None
    format: str | None = None
    family: str | None = None
    families: list[str] | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None
    embedding_length: int | None = None


class _OllamaModelTagPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str | None = None
    model: str | None = None
    modified_at: str | None = None
    size: int | None = None
    digest: str | None = None
    details: _OllamaModelDetailsPayload | None = None
    capabilities: list[str] | None = None

    @model_validator(mode="after")
    def _require_identifier(self) -> Self:
        if not self.name and not self.model:
            raise ValueError("Ollama model entry must include name or model")
        return self

    @property
    def identifier(self) -> str:
        if self.name:
            return self.name
        if self.model:
            return self.model
        raise RuntimeError("validated Ollama model entry missing identifier")


def _empty_ollama_model_tags() -> list[_OllamaModelTagPayload]:
    return []


class _OllamaTagsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    models: list[_OllamaModelTagPayload] = Field(
        default_factory=_empty_ollama_model_tags
    )


def _parse_ollama_model_names(raw_models_json: object) -> list[str]:
    models_payload = _OllamaTagsPayload.model_validate(raw_models_json)
    return [model.identifier for model in models_payload.models]


def _coerce_str_object_map(value: object) -> dict[str, object] | None:
    """Konvertiert Mapping-ähnliche Objekte in str->object-Maps."""
    if not isinstance(value, Mapping):
        return None

    mapped = cast(Mapping[str, object], value)
    casted: dict[str, object] = {}
    for key in mapped:
        casted[key] = mapped[key]
    return casted


LLMFrameProcessorInput = (
    Mapping[str, object] | FrameCollectionItem | tuple[object, ...] | list[object]
)


class _LLMModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    priority: int = 0
    timeout: int
    description: str = "Runtime model"

    def with_timeout(self, timeout: int) -> "_LLMModelConfig":
        return self.model_copy(update={"timeout": timeout})


class ModelConfig:
    """Konfiguration für verfügbare Modelle mit Prioritäten."""

    # Prioritized for local RTX 3050 / 4 GB VRAM usage via Ollama.
    MODELS: list[_LLMModelConfig] = [
        _LLMModelConfig(
            name="lx-gemma4-e2b-json",
            priority=1,
            timeout=120,
            description="Gemma 4 E2B JSON profile",
        ),
        _LLMModelConfig(
            name="gemma4:e2b",
            priority=2,
            timeout=120,
            description="Gemma 4 E2B base Ollama model",
        ),
        _LLMModelConfig(
            name="llama3.2:1b",
            priority=10,
            timeout=45,
            description="Tiny fallback model",
        ),
    ]

    @classmethod
    def get_models_by_priority(cls) -> list[_LLMModelConfig]:
        """Gibt Modelle sortiert nach Priorität zurück."""
        return sorted(cls.MODELS, key=lambda model: model.priority)


class MetadataCache:
    """
    Cache für Metadaten-Extraktionsergebnisse um wiederholte LLM-Aufrufe zu vermeiden.
    """

    def __init__(self, max_size: int = 100):
        self.cache: dict[str, SensitiveMeta] = {}
        self.max_size = max_size
        self.hit_count = 0
        self.miss_count = 0
        self.sensitive_meta = SensitiveMeta()

    def _generate_key(self, text: str) -> str:
        """Generiert einen Cache-Key basierend auf Text-Inhalt."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def get(self, text: str) -> Optional[SensitiveMeta]:
        """Holt Metadaten aus dem Cache."""
        key = self._generate_key(text)
        if key in self.cache:
            self.hit_count += 1
            logger.debug(f"Cache HIT für Key {key}")
            return self.cache[key]
        else:
            self.miss_count += 1
            return None

    def put(self, text: str, metadata: SensitiveMeta) -> None:
        """Speichert Metadaten im Cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        key = self._generate_key(text)
        self.cache[key] = metadata
        logger.debug(f"Cache PUT für Key {key}")

    def get_stats(self) -> LLMMetadataCacheStatsPayload:
        """Gibt Cache-Statistiken zurück."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        return LLMMetadataCacheStatsPayload(
            hit_count=self.hit_count,
            miss_count=self.miss_count,
            hit_rate=hit_rate,
            cache_size=len(self.cache),
            max_size=self.max_size,
        )


class LLMMetadataExtractor:
    """
    Optimierte LLM-Integration fuer Metadaten-Extraktion.

    Verwendet entweder eine OpenAI-kompatible REST API
    oder native Ollama-Endpunkte und implementiert ein fail-safe Modell-System.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        enable_cache: bool = True,
        provider: str = "ollama",
        preferred_model: Optional[str] = None,
        model_timeout: Optional[int] = None,
    ):
        self.provider = (provider or "ollama").strip().lower()
        if self.provider not in {"ollama", "vllm"}:
            logger.warning(
                "Unknown LLM provider %s, falling back to ollama.",
                self.provider,
            )
            self.provider = "ollama"
        self.base_url = (base_url or self._default_base_url()).rstrip("/")
        self.chat_endpoint = self._build_chat_endpoint()
        self.available_models_retry = False
        self.available_models = self._check_available_models()
        self.current_model: _LLMModelConfig | None = None
        self.cache = MetadataCache() if enable_cache else None
        self.sensitive_meta = SensitiveMeta()
        self.preferred_model = preferred_model
        self.preferred_timeout = model_timeout

        self._initialize_best_model()

    def _require_current_model(self) -> _LLMModelConfig:
        current_model = self.current_model
        if current_model is None:
            raise RuntimeError("Kein aktives LLM-Modell verfügbar.")
        return current_model

    def _default_base_url(self) -> str:
        if self.provider == "ollama":
            return "http://127.0.0.1:11434"
        return "http://127.0.0.1:8000"

    def _build_chat_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        provider = getattr(self, "provider", "ollama")
        if provider == "ollama":
            return f"{base}/api/chat"
        return f"{base}/v1/chat/completions"

    def _check_available_models(self) -> list[str]:
        """Überprüft, welche Modelle verfügbar sind."""
        if self.available_models_retry:
            return []

        self.available_models_retry = False
        try:
            model_endpoint = (
                f"{self.base_url.rstrip('/')}/api/tags"
                if self.provider == "ollama"
                else f"{self.base_url.rstrip('/')}/v1/models"
            )
            response = requests.get(model_endpoint, timeout=5)
            if response.status_code == 200:
                raw_models_json = response.json()

                if self.provider == "ollama":
                    return _parse_ollama_model_names(raw_models_json)

                models_payload = LLMVllmModelsPayload.model_validate(raw_models_json)
                return [model.id for model in models_payload.data]

            return []
        except Exception as e:
            logger.warning(f"could not check available models: {e}")
            try:
                # Short one-shot backoff to avoid slowing tests/pipeline startup by ~100s.
                self.available_models_retry = True
                time.sleep(1.0)
                return self._check_available_models()
            except Exception:
                return []

    def _initialize_best_model(self):
        """Initialisiert das beste verfügbare Modell."""
        if self.preferred_model:
            if self.preferred_model in self.available_models:
                model_config = next(
                    (
                        m
                        for m in ModelConfig.get_models_by_priority()
                        if m.name == self.preferred_model
                    ),
                    None,
                )
                if model_config:
                    pass
                else:
                    model_config = _LLMModelConfig(
                        name=self.preferred_model,
                        priority=0,
                        timeout=self.preferred_timeout or 30,
                        description="Preferred model",
                    )
                if self.preferred_timeout:
                    model_config = model_config.with_timeout(self.preferred_timeout)
                self.current_model = model_config
                logger.info(
                    "Verwende bevorzugtes Modell: %s",
                    model_config.name,
                )
                return
            if not self.available_models:
                logger.warning(
                    "Modellliste nicht verfügbar, aktiviere LLM nicht fuer konfiguriertes Modell: %s",
                    self.preferred_model,
                )
                return
            logger.warning(
                "Bevorzugtes Modell nicht verfuegbar: %s",
                self.preferred_model,
            )

        for model_config in ModelConfig.get_models_by_priority():
            if model_config.name in self.available_models:
                self.current_model = model_config
                logger.info(
                    f"Verwende Modell: {model_config.name} - {model_config.description}"
                )
                return

        logger.warning(
            "Keine kompatiblen LLM-Modelle verfügbar. LLM-Features werden deaktiviert, OCR-basierte Verarbeitung wird fortgesetzt."
        )
        self.current_model = None

    def _create_extraction_prompt(self, text: str) -> str:
        """
        Erstellt einen optimierten Prompt für medizinische Metadaten-Extraktion.

        Args:
            text: Input-Text zur Metadaten-Extraktion

        Returns:
            Optimierter Prompt-String für medizinische Dokumente
        """
        text_window = text[:6000]
        return f"""OUTPUT_JSON_ONLY
NO_MARKDOWN
NO_PROSE
NO_COMMENTS
NO_THINK
UNKNOWN_IS_NULL
DO_NOT_INVENT_VALUES
DATE_FORMAT=YYYY-MM-DD
PATIENT_DOB_IS_BIRTH_DATE_ONLY
EXAMINATION_DATE_IS_REPORT_OR_EXAM_DATE_ONLY
KEEP_CASENUMBER_AS_SEEN
REQUIRED_JSON={STRICT_METADATA_TEMPLATE}
OCR_TEXT_BEGIN
{text_window}
OCR_TEXT_END"""

    def _create_json_schema(self) -> dict[str, object]:
        """Erstellt das erweiterte JSON-Schema für medizinische Metadaten-Extraktion."""
        return {
            "type": "object",
            "properties": {
                # Patientendaten
                "first_name": {"type": ["string", "null"]},
                "last_name": {"type": ["string", "null"]},
                "dob": {"type": ["string", "null"]},
                "gender": {
                    "type": ["string", "null"],
                    "enum": ["male", "female", "unknown", None],
                },
                # Untersuchungsdaten
                "examination_date": {"type": ["string", "null"]},
                "examination_time": {"type": ["string", "null"]},
                "examiner_first_name": {"type": ["string", "null"]},
                "examiner_last_name": {"type": ["string", "null"]},
                # Administrative Daten
                "casenumber": {"type": ["string", "null"]},
                # Zusätzliche Informationen
            },
            "required": [],  # Keine Felder sind zwingend erforderlich
        }

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
    def _make_api_request(
        self, payload: LLMChatRequestPayload
    ) -> LLMChatResponsePayload:
        """
        Macht API-Request mit Retry-Logik und robuster Fehlerbehandlung.

        Args:
            payload: Request-Payload für die LLM/OpenAI-kompatible API

        Returns:
            Response-Dictionary von der API

        Raises:
            requests.RequestException: Bei API-Fehlern
            requests.Timeout: Bei Timeouts
        """
        timeout = 10
        try:
            assert self.current_model is not None
            timeout = self.current_model.timeout

            model_name = payload.model
            logger.debug(
                f"🔗 API-Request an {self.chat_endpoint} mit Modell {model_name}"
            )

            response = requests.post(
                self.chat_endpoint,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )

            if response.status_code == 200:
                raw_response = response.json()
                result = LLMChatResponsePayload.model_validate(raw_response)
                content = self._extract_response_content(result)

                # Validiere Antwort
                if not content:
                    logger.warning(f"⚠️ Leere Antwort vom Modell: {result}")

                content_length = len(content)
                logger.debug(f"✅ API-Response erhalten: {content_length} Zeichen")

                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.error(f"❌ API-Fehler: {error_msg}")
                raise requests.RequestException(error_msg)

        except requests.Timeout:
            assert self.current_model is not None
            logger.error(
                f"⏰ Timeout ({timeout}s) bei Modell {self.current_model.name}"
            )
            raise
        except requests.ConnectionError as e:
            logger.error("🔌 Verbindungsfehler zu %s: %s", self.provider, e)
            raise requests.RequestException(f"{self.provider} nicht erreichbar: {e}")
        except Exception as e:
            logger.error(f"💥 Unerwarteter API-Fehler: {e}")
            raise

    def _extract_response_content(self, result: LLMChatResponsePayload) -> str:
        for choice in result.choices:
            content = choice.message.content
            if content:
                return content

        if result.message and result.message.content:
            return result.message.content

        return ""

    def _build_chat_payload(
        self,
        text: str,
        fast_mode: bool = False,
        prompt_type: str = "full",
    ) -> LLMChatRequestPayload:
        messages = [
            LLMChatMessagePayload(
                role="user",
                content=(
                    self._create_fast_extraction_prompt(text)
                    if prompt_type == "fast"
                    else self._create_extraction_prompt(text)
                ),
            )
        ]

        model_name = self._require_current_model().name

        if self.provider == "ollama":
            return LLMChatOllamaPayload(
                model=model_name,
                messages=messages,
                stream=False,
                format="json",
                options=LLMChatOllamaOptionsPayload.model_validate(
                    OLLAMA_GENERATION_OPTIONS
                ),
            )

        if fast_mode or prompt_type == "fast":
            return LLMChatOpenAIPayload(
                model=model_name,
                messages=messages,
                max_tokens=150,
                response_format={"type": "json_object"},
                top_p=0.9,
                temperature=0.0,
            )

        return LLMChatOpenAIPayload(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            top_p=1.0,
            stream=False,
            temperature=0.0,
        )

    def _try_next_model(self) -> bool:
        """
        Wechselt zum nächsten verfügbaren Modell basierend auf Priorität.

        Returns:
            True wenn ein nächstes Modell verfügbar ist, False sonst
        """
        if not self.current_model:
            return False

        current_priority = self.current_model.priority

        # Finde nächstes Modell mit höherer Priorität
        for model_config in ModelConfig.get_models_by_priority():
            if (
                model_config.priority > current_priority
                and model_config.name in self.available_models
            ):
                old_model = self.current_model.name
                self.current_model = model_config
                logger.info(
                    f"🔄 Modell-Wechsel: {old_model} → {model_config.name} (Priorität {current_priority} → {model_config.priority})"
                )
                return True

        logger.warning("⚠️ Keine weiteren Modelle für Fallback verfügbar")
        return False

    def extract_metadata(self, text: str) -> Optional[SensitiveMeta]:
        """
        Extrahiert Patientenmetadaten aus Text mit fail-safe Modell-System.

        Args:
            text: Input-Text zur Extraktion

        Returns:
            SensitiveMeta Objekt oder None bei Fehler
        """
        # Early return if no models are available
        if not self.current_model and not self.available_models:
            logger.warning(
                "Keine Modelle fuer Provider %s verfuegbar. Ueberspringe LLM-Extraktion.",
                self.provider,
            )
            return None

        # Überprüfe zuerst den Cache
        cached_metadata = self.cache.get(text) if self.cache is not None else None
        if cached_metadata:
            logger.info("✅ Metadaten aus Cache geladen")
            self.sensitive_meta.safe_update(cached_metadata)
            return self.sensitive_meta

        if not self.current_model:
            logger.error("Kein Modell verfügbar für Extraktion")
            return None

        # Bestimme verfügbare Modelle für Fallback
        available_model_configs: list[_LLMModelConfig] = []
        seen_model_names: set[str] = set()

        if self.current_model:
            available_model_configs.append(self.current_model)
            seen_model_names.add(self.current_model.name)

        for model_config in ModelConfig.get_models_by_priority():
            if (
                model_config.name in self.available_models
                and model_config.name not in seen_model_names
            ):
                available_model_configs.append(model_config)
                seen_model_names.add(model_config.name)

        if not available_model_configs:
            logger.error("Keine konfigurierten Modelle verfügbar")
            return None

        # Versuche alle verfügbaren Modelle der Reihe nach
        for model_attempt, model_config in enumerate(available_model_configs):
            # Setze aktuelles Modell für diesen Versuch
            self.current_model = model_config

            try:
                start_time = time.time()

                payload = self._build_chat_payload(text, prompt_type="full")

                logger.info(
                    f"Versuch {model_attempt + 1}/{len(available_model_configs)}: Extraktion mit {self.current_model.name}"
                )

                # API-Request mit Retry-Logik
                response = self._make_api_request(payload)
                content = self._extract_response_content(response)

                # Performance-Metriken loggen
                duration = time.time() - start_time
                token_count = 0
                logger.info(f"API-Erfolg in {duration:.2f}s, {token_count} Tokens")

                # JSON parsen und validieren
                try:
                    # Bereinige Antwort falls nötig (entferne Markdown-Blöcke etc.)
                    cleaned_content = self._clean_json_response(content)
                    metadata_payload = LLMMetadataPayload.model_validate_json(
                        cleaned_content
                    )
                    self.sensitive_meta.safe_update(metadata_payload)
                    metadata = self.sensitive_meta

                    # Speichere im Cache
                    if self.cache is not None:
                        self.cache.put(text, metadata)

                    logger.info(
                        f"✅ Erfolgreich extrahiert mit {self.current_model.name}: "
                        f"Datum: {metadata.examination_date}"
                    )

                    self.sensitive_meta.safe_update(metadata)
                    return self.sensitive_meta

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(
                        f"JSON/Validierung fehlgeschlagen für {self.current_model.name}: {e}"
                    )
                    logger.debug(f"Rohe Antwort: {content}")

                    # Bei JSON-Fehlern versuche das nächste Modell
                    if model_attempt < len(available_model_configs) - 1:
                        logger.info("Versuche nächstes Modell wegen JSON-Fehler...")
                        continue
                    else:
                        raise ValueError(
                            "Alle Modelle lieferten ungültige JSON-Antworten"
                        )

            except requests.Timeout:
                logger.warning(
                    f"Timeout bei {self.current_model.name} nach {self.current_model.timeout}s"
                )
                # Bei Timeout versuche das nächste Modell
                if model_attempt < len(available_model_configs) - 1:
                    logger.info("Versuche nächstes Modell wegen Timeout...")
                    continue
                else:
                    logger.error("Alle Modelle liefen in Timeout")
                    break

            except Exception as e:
                logger.error(f"Fehler mit Modell {self.current_model.name}: {e}")

                # Bei anderen Fehlern versuche das nächste Modell
                if model_attempt < len(available_model_configs) - 1:
                    logger.info("Versuche nächstes Modell wegen Fehler...")
                    continue
                else:
                    logger.error("Alle Modelle fehlgeschlagen")
                    break

        logger.error("❌ Alle verfügbaren Modelle fehlgeschlagen")
        return None

    def _clean_json_response(self, content: str) -> str:
        """
        Bereinigt die Modell-Antwort um gültiges JSON zu extrahieren.

        Args:
            content: Rohe Antwort vom Modell

        Returns:
            Bereinigter JSON-String
        """
        # Entferne reasoning blocks, including malformed/unclosed variants.
        content = re.sub(
            r"<think\b[^>]*>.*?</think\s*>",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        unclosed_think = re.search(r"<think\b[^>]*>", content, flags=re.IGNORECASE)
        if unclosed_think:
            prefix = content[: unclosed_think.start()]
            tail = content[unclosed_think.end() :]
            json_start = tail.find("{")
            content = prefix + (tail[json_start:] if json_start != -1 else "")

        # Entferne Markdown-Code-Blöcke falls vorhanden
        content = content.strip()
        if "```" in content:
            match = re.search(
                r"```(?:json)?\s*(.*?)\s*```",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if match:
                content = match.group(1)

        start = content.find("{")
        if start == -1:
            return content.strip()

        depth = 0
        in_string = False
        escaped = False
        for idx, char in enumerate(content[start:], start=start):
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
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start : idx + 1].strip()

        return content[start:].strip()

    def get_model_info(self) -> LLMModelInfoPayload:
        """Gibt Informationen über das aktuelle Modell und Cache-Statistiken zurück."""
        current_model_name = self.current_model.name if self.current_model else None
        cache_stats = self.cache.get_stats() if self.cache else None

        return LLMModelInfoPayload(
            current_model=current_model_name,
            available_models=self.available_models,
            total_models=len(self.available_models),
            cache_stats=cache_stats,
        )

    def extract_metadata_smart_sampling(
        self, text: str, confidence_threshold: float = 0.7
    ) -> Optional[SensitiveMeta]:
        """
        Extrahiert Metadaten mit Smart-Sampling für bessere Performance.
        Stoppt früh wenn Konfidenz erreicht wird.

        Args:
            text: Input-Text zur Extraktion
            confidence_threshold: Schwellwert für frühen Stopp

        Returns:
            SensitiveMeta Objekt oder None bei Fehler
        """
        # Pre-Check: Enthält der Text überhaupt relevante Informationen?
        if not self._contains_patient_data(text):
            logger.debug(
                "Text enthält keine erkennbaren Patientendaten, überspringe LLM-Extraktion"
            )
            return None

        # Verwende nur das schnellste verfügbare Modell für Smart Sampling
        fastest_model = self._get_fastest_available_model()
        if not fastest_model:
            return self.extract_metadata(text)  # Fallback zur normalen Extraktion
        if self.current_model and fastest_model.name == self.current_model.name:
            logger.info(
                "Fastest model is same as main model. Skipping sampling to avoid double-execution."
            )
            return self.extract_metadata(text)

        original_model = self.current_model
        self.current_model = fastest_model

        try:
            # Verwende sehr kurzen Text für bessere Performance
            truncated_text = text[:200] if len(text) > 200 else text

            start_time = time.time()

            payload = self._build_chat_payload(
                truncated_text,
                fast_mode=True,
                prompt_type="fast",
            )

            logger.info(f"Smart-Sampling mit {self.current_model.name}")
            response = self._make_api_request(payload)
            content = self._extract_response_content(response)

            duration = time.time() - start_time
            logger.info(f"Smart-Sampling in {duration:.2f}s abgeschlossen")

            # Schnelle JSON-Parsing ohne komplexe Validierung
            try:
                cleaned_content = self._clean_json_response(content)
                metadata_payload = LLMMetadataPayload.model_validate_json(
                    cleaned_content
                )

                # Einfache Konfidenz-Bewertung basierend auf gefundenen Daten
                confidence = self._calculate_confidence(metadata_payload)

                if confidence >= confidence_threshold:
                    self.sensitive_meta.safe_update(metadata_payload)
                    metadata = self.sensitive_meta
                    logger.info(
                        f"✅ Smart-Sampling erfolgreich (Konfidenz: {confidence:.2f}): {metadata.first_name} {metadata.last_name}, DOB: {metadata.dob}"
                    )
                    return metadata
                else:
                    logger.info(
                        f"Smart-Sampling Konfidenz zu niedrig ({confidence:.2f}), verwende Vollextraktion"
                    )
                    return self.extract_metadata(
                        text
                    )  # Fallback zur vollständigen Extraktion

            except (json.JSONDecodeError, ValidationError) as e:
                logger.debug(
                    f"Smart-Sampling JSON-Fehler, Fallback zur Vollextraktion: {e}"
                )
                return self.extract_metadata(text)

        except Exception as e:
            logger.warning(
                f"Smart-Sampling fehlgeschlagen, Fallback zur Vollextraktion: {e}"
            )
            return self.extract_metadata(text)

        finally:
            # Restore original model
            self.current_model = original_model

    def _contains_patient_data(self, text: str) -> bool:
        """Erweiterte Prüfung ob Text potentielle Patientendaten enthält."""
        if not text or len(text.strip()) < 10:
            return False

        # Erweiterte Keyword-Liste für medizinische Dokumente
        keywords = [
            # Patienteninformationen
            "patient",
            "name",
            "alter",
            "geburt",
            "datum",
            "untersuchung",
            "herr",
            "frau",
            "jahre",
            "jahr",
            "männlich",
            "weiblich",
            "mr",
            "mrs",
            "dr",
            "prof",
            "geb.",
            "geboren",
            # Administrative Begriffe
            "fall",
            "case",
            "nummer",
            "nr.",
            "id",
            "pat-id",
            "patient-id",
            "fallnummer",
            "case-id",
            "kasenummer",
            "akte",
            # Medizinische Begriffe
            "untersucher",
            "arzt",
            "doktor",
            "examination",
            "diagnose",
            "befund",
            "termin",
            "aufnahme",
            "entlassung",
            # Datumsformate und Zeitangaben
            "20",
            "19",
            ".",
            "/",
            "-",
            "uhr",
            "zeit",
            "time",
            # Spezifische Endoskopie-Begriffe
            "endoskopie",
            "koloskopie",
            "gastroskopie",
            "scope",
            "bildgebung",
            "aufzeichnung",
        ]

        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)

        # Erhöhte Schwelle: mindestens 2 Keywords für bessere Präzision
        return keyword_count >= 2

    def _get_fastest_available_model(self) -> _LLMModelConfig | None:
        """Gibt das schnellste verfügbare Modell zurück."""
        if self.preferred_model and (
            self.preferred_model in self.available_models or not self.available_models
        ):
            model_config = next(
                (
                    m
                    for m in ModelConfig.get_models_by_priority()
                    if m.name == self.preferred_model
                ),
                None,
            )
            if model_config:
                if self.preferred_timeout:
                    model_config = model_config.with_timeout(self.preferred_timeout)
                return model_config
            return _LLMModelConfig(
                name=self.preferred_model,
                priority=0,
                timeout=self.preferred_timeout or 30,
                description="Preferred model",
            )

        for model_config in ModelConfig.get_models_by_priority():
            if model_config.name in self.available_models:
                return model_config
        return None

    def _create_fast_extraction_prompt(self, text: str) -> str:
        """Erstellt einen optimierten Fast-Prompt für Smart-Sampling."""
        return f"""OUTPUT_JSON_ONLY
NO_MARKDOWN
NO_PROSE
NO_THINK
UNKNOWN_IS_NULL
DATE_FORMAT=YYYY-MM-DD
REQUIRED_JSON={STRICT_METADATA_TEMPLATE}
OCR_TEXT={text}"""

    def _is_small_model(self, model_name: str) -> bool:
        """Heuristik für kleine Modelle, die stärker von knappen Prompts profitieren."""
        if not model_name:
            return False
        return any(
            token in model_name
            for token in (
                "1b",
                "1.5b",
                "2b",
                "3b",
                "e2b",
                "lx-gemma4-e2b-json",
                "gemma4:e2b",
            )
        )

    def calculate_confidence(self, metadata: LLMMetadataPayload) -> float:
        """Berechnet den Konfidenz-Score für die extrahierten Metadaten."""
        return self._calculate_confidence(metadata)

    def _calculate_confidence(self, metadata: LLMMetadataPayload) -> float:
        """
        Erweiterte Konfidenz-Berechnung basierend auf gefundenen medizinischen Daten.

        Returns:
            Konfidenz-Score zwischen 0.0 und 1.0
        """
        score = 0.0

        # Patient Name (höchste Priorität)
        first_name = metadata.first_name
        last_name = metadata.last_name

        if (
            first_name
            and last_name
            and f"{first_name} {last_name}".strip().lower()
            not in ["unknown", "patient", "null", "nix"]
        ):
            score += 0.30
        elif (first_name and first_name.lower() not in ["unknown", "", "null"]) or (
            last_name and last_name.lower() not in ["unknown", "", "null"]
        ):
            score += 0.25

        # Examination Date (wichtig für medizinische Aufzeichnungen)
        exam_date = metadata.examination_date
        if exam_date and exam_date.lower() not in ["unknown", "", "null"]:
            score += 0.20

        # Fallnummer/Case Number (sehr wichtig für medizinische Identifikation)
        case_num = metadata.casenumber
        if case_num and case_num.lower() not in ["unknown", "", "null"]:
            score += 0.20

        # Geburtsdatum (wichtig für Patientenidentifikation)
        dob = metadata.dob
        if dob and dob.lower() not in ["unknown", "", "null"]:
            score += 0.15

        # Gender (weniger wichtig, aber hilfreich)
        gender = metadata.gender
        if gender and gender.lower() in ["male", "female"]:
            score += 0.05

        # Examiner (zusätzlicher Kontext)
        examiner_first_name = metadata.examiner_first_name
        if examiner_first_name and examiner_first_name.lower() not in [
            "unknown",
            "",
            "null",
        ]:
            score += 0.05

        examiner_last_name = metadata.examiner_last_name
        if examiner_last_name and examiner_last_name.lower() not in [
            "unknown",
            "",
            "null",
        ]:
            score += 0.05

        return min(score, 1.0)


class FrameSamplingOptimizer:
    """
    Optimiert Frame-Sampling für bessere Performance.
    Reduziert unnötige OCR-Operationen durch intelligente Frame-Auswahl.
    """

    def __init__(self, max_frames: int = 50, skip_similar_threshold: float = 0.85):
        self.max_frames = max_frames
        self.skip_similar_threshold = skip_similar_threshold
        self.processed_hashes: set[str] = set()
        self.last_metadata: LLMFrameDataPayload | None = None

    def should_process_frame(
        self, frame_idx: int, total_frames: int, frame_hash: str
    ) -> bool:
        """
        Entscheidet ob ein Frame verarbeitet werden soll.

        Args:
            frame_idx: Index des aktuellen Frames
            total_frames: Gesamtanzahl der Frames
            frame_hash: Optional - Hash des Frame-Inhalts für Duplikat-Erkennung

        Returns:
            True wenn Frame verarbeitet werden soll
        """
        # Erste Frames immer verarbeiten
        if frame_idx < 5:
            return True

        # Letzte Frames immer verarbeiten
        if frame_idx >= total_frames - 5:
            return True

        # Frame-Hash Duplikat-Check
        if frame_hash and frame_hash in self.processed_hashes:
            return False

        # Adaptive Sampling basierend auf Videolänge
        if total_frames <= 100:
            # Kurze Videos: jeden 2. Frame
            return frame_idx % 2 == 0
        elif total_frames <= 500:
            # Mittlere Videos: jeden 5. Frame
            return frame_idx % 5 == 0
        else:
            # Lange Videos: jeden 10. Frame
            return frame_idx % 10 == 0

    def register_processed_frame(
        self, frame_hash: str, metadata: LLMFrameDataPayload | None = None
    ) -> None:
        """Registriert einen verarbeiteten Frame."""
        if frame_hash:
            self.processed_hashes.add(frame_hash)
        if metadata is not None:
            self.last_metadata = metadata

    def get_sampling_strategy(self, total_frames: int) -> dict[str, object]:
        """
        Gibt die optimale Sampling-Strategie für die gegebene Frame-Anzahl zurück.

        Returns:
            Dictionary mit Sampling-Parametern
        """
        if total_frames <= 50:
            return {
                "strategy": "dense",
                "skip_factor": 1,
                "max_samples": total_frames,
                "description": "Alle Frames verarbeiten (kurzes Video)",
            }
        elif total_frames <= 200:
            return {
                "strategy": "moderate",
                "skip_factor": 2,
                "max_samples": total_frames // 2,
                "description": "Jeden 2. Frame verarbeiten",
            }
        elif total_frames <= 1000:
            return {
                "strategy": "sparse",
                "skip_factor": 5,
                "max_samples": total_frames // 5,
                "description": "Jeden 5. Frame verarbeiten",
            }
        else:
            return {
                "strategy": "minimal",
                "skip_factor": 10,
                "max_samples": min(self.max_frames, total_frames // 10),
                "description": "Jeden 10. Frame verarbeiten (langes Video)",
            }


# Factory-Funktion für einfache Verwendung
def create_llm_extractor(
    enable_cache: bool = True, enable_smart_sampling: bool = True
) -> LLMMetadataExtractor:
    """
    Erstellt eine optimierte LLM-Extractor-Instanz.

    Args:
        enable_cache: Aktiviert Metadaten-Caching
        enable_smart_sampling: Aktiviert Smart-Sampling für bessere Performance
    """
    extractor = LLMMetadataExtractor(enable_cache=enable_cache)

    # Setze optimale Einstellungen für medizinische Datenextraktion
    if enable_smart_sampling:
        logger.info("Smart-Sampling aktiviert für optimale Performance")

    return extractor


# Convenience-Funktion für maximale Performance
def create_fast_extractor() -> LLMMetadataExtractor:
    """Erstellt Extractor mit maximaler Performance-Optimierung."""
    return create_llm_extractor(enable_cache=True, enable_smart_sampling=True)


# Erweiterte Factory-Funktion
def create_optimized_extractor_with_sampling() -> tuple[
    LLMMetadataExtractor, FrameSamplingOptimizer
]:
    """Erstellt optimierte Extractor- und Sampling-Instanzen."""
    extractor = LLMMetadataExtractor()
    optimizer = FrameSamplingOptimizer()
    return extractor, optimizer


# Beispiel-Verwendung
if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(level=logging.INFO)

    # Extractor erstellen
    extractor = create_llm_extractor()

    # Test-Texte
    test_texts = [
        "Herr Müller, 45 Jahre alt, wurde am 15.01.2024 untersucht.",
        "Frau Schmidt, 32 Jahre alt, Untersuchung am 20.02.2024",
        "Patient Klaus Weber (m), 58 Jahre, Termin: 10.03.2024",
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

        else:
            print(f"❌ Fehlgeschlagen nach {duration:.2f}s")


class EnrichedMetadataExtractor:
    """
    Erweiterte Metadaten-Extraktion mit Frame-Sampling-Integration.
    Kombiniert LLM-Extraktion mit visuellen Frame-Daten.
    """

    def __init__(
        self,
        llm_extractor: LLMMetadataExtractor,
        frame_optimizer: FrameSamplingOptimizer,
    ):
        self.llm_extractor = llm_extractor
        self.frame_optimizer = frame_optimizer
        if not self.llm_extractor:
            self.llm_extractor = create_llm_extractor()
        if not self.frame_optimizer:
            self.frame_optimizer = FrameSamplingOptimizer()
        self.frame_context: LLMFrameContextPayload = LLMFrameContextPayload()
        self.temporal_metadata: list[LLMTextTimelineEntryPayload] = []
        self.sensitive_meta = SensitiveMeta()

    def extract_from_frame_sequence(
        self, frames_data: Sequence[LLMFrameDataPayload], ocr_texts: list[str]
    ) -> LLMEnrichedMetadataPayload:
        """
        Extrahiert angereicherte Metadaten aus einer Frame-Sequenz.

        Args:
            frames_data: Liste von Frame-Daten aus sample_frames_coroutine
            ocr_texts: Optional - bereits extrahiierte OCR-Texte

        Returns:
            Angereicherte Metadaten-Dictionary
        """
        enriched_metadata = LLMEnrichedMetadataPayload()

        # 1. Sammle Frame-Kontext
        self._analyze_frame_context(frames_data, enriched_metadata)

        # 2. OCR-Text-Aggregation
        combined_text = self.aggregate_ocr_texts(frames_data, ocr_texts)

        # 3. LLM-Extraktion auf aggregiertem Text
        if combined_text:
            llm_metadata = self.llm_extractor.extract_metadata_smart_sampling(
                combined_text
            )
            if llm_metadata:
                self.sensitive_meta.safe_update(llm_metadata)

        # 4. Temporale Analyse der Frames
        enriched_metadata.temporal_analysis = self._perform_temporal_analysis(
            frames_data, enriched_metadata
        )

        # 5. Konfidenz-Bewertung
        enriched_metadata.confidence_scores = self._calculate_enriched_confidence(
            enriched_metadata
        )

        enriched_metadata.llm_extracted = LLMMetadataPayload.model_validate(
            self.sensitive_meta.to_dict()
        )
        enriched_metadata.source_frames = list(frames_data)
        return enriched_metadata

    def _analyze_frame_context(
        self,
        frames_data: Sequence[LLMFrameDataPayload],
        enriched_metadata: LLMEnrichedMetadataPayload,
    ) -> None:
        """Analysiert visuellen Kontext der Frames."""
        quality_scores: list[float] = []
        timestamps: list[int | float] = []
        frame_types: dict[str, int] = {}
        frame_stats = LLMFrameContextPayload(
            total_frames=len(frames_data),
            frame_types=frame_types,
            quality_scores=quality_scores,
            timestamps=timestamps,
        )
        text_frames = 0

        for frame_data in frames_data:
            # Frame-Qualität bewerten
            quality_score = frame_data.ocr_confidence
            quality_scores.append(float(quality_score))

            # Timestamp-Information
            timestamps.append(float(frame_data.timestamp))

            # Text-Frames zählen
            has_text = bool(frame_data.has_text)
            ocr_confidence = frame_data.ocr_confidence
            has_confidence = ocr_confidence > 0.5
            if has_text or has_confidence:
                text_frames += 1

            # Frame-Typ klassifizieren
            frame_type = self._classify_frame_type(frame_data)
            frame_types[frame_type] = frame_types.get(frame_type, 0) + 1

        frame_stats.text_frames = text_frames

        enriched_metadata.frame_context = frame_stats

    def aggregate_ocr_texts(
        self, frames_data: Sequence[LLMFrameDataPayload], ocr_texts: list[str]
    ) -> str:
        """Aggregiert OCR-Texte intelligent und reduziert Input-Groesse."""
        return self._aggregate_ocr_texts(frames_data, ocr_texts)

    def _aggregate_ocr_texts(
        self, frames_data: Sequence[LLMFrameDataPayload], ocr_texts: list[str]
    ) -> str:
        """Aggregiert OCR-Texte intelligent und reduziert Input-Groesse."""
        raw_texts: list[str] = []

        # Verwende bereitgestellte OCR-Texte oder extrahiere aus Frame-Daten
        if ocr_texts:
            raw_texts.extend(t for t in ocr_texts if t)

        for frame_data in frames_data:
            ocr_text = frame_data.ocr_text
            if ocr_text:
                raw_texts.append(ocr_text)

        if not raw_texts:
            return ""

        # Smart Deduplication (Zeilen-basiert, normalisiert)
        unique_lines: set[str] = set()
        cleaned_text_parts: list[str] = []

        for text in raw_texts:
            for line in text.split("\n"):
                line_clean = line.strip()
                if len(line_clean) < 4:
                    continue
                comp_key = "".join(ch for ch in line_clean.lower() if ch.isalnum())
                if not comp_key:
                    continue
                if comp_key not in unique_lines:
                    unique_lines.add(comp_key)
                    cleaned_text_parts.append(line_clean)

        full_text = " | ".join(cleaned_text_parts)
        return full_text[:1500]

    def _deduplicate_texts(self, texts: list[str]) -> list[str]:
        """Entfernt doppelte und sehr ähnliche Texte."""
        unique_texts: list[str] = []
        seen_hashes: set[str] = set()

        for text in texts:
            if not text or len(text.strip()) < 3:
                continue

            # Normalisiere Text für Vergleich
            normalized = text.lower().strip()
            text_hash = hashlib.md5(normalized.encode()).hexdigest()[:8]

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)

        return unique_texts

    def _prioritize_by_confidence(
        self, texts: list[str], frames_data: Sequence[LLMFrameDataPayload]
    ) -> list[str]:
        """Priorisiert Texte basierend auf OCR-Konfidenz."""
        text_confidence_pairs: list[tuple[str, float]] = []

        for i, text in enumerate(texts):
            confidence = 0.5  # Default-Konfidenz

            # Suche entsprechende Frame-Daten
            if i < len(frames_data):
                confidence = frames_data[i].ocr_confidence

            text_confidence_pairs.append((text, confidence))

        # Sortiere nach Konfidenz (absteigend)
        text_confidence_pairs.sort(key=lambda x: x[1], reverse=True)

        return [text for text, _ in text_confidence_pairs]

    def _classify_frame_type(self, frame_data: LLMFrameDataPayload) -> str:
        """Klassifiziert Frame-Typ basierend auf Eigenschaften."""
        if frame_data.has_patient_info:
            return "patient_info"
        elif frame_data.has_ui_elements:
            return "ui_frame"
        if frame_data.ocr_confidence > 0.7:
            return "text_frame"
        if frame_data.is_endoscopy_view:
            return "endoscopy"
        else:
            return "unknown"

    def _perform_temporal_analysis(
        self,
        frames_data: Sequence[LLMFrameDataPayload],
        enriched_metadata: LLMEnrichedMetadataPayload,
    ) -> LLMTemporalAnalysisPayload:
        """Führt temporale Analyse der Frame-Sequenz durch."""
        timeline: list[LLMTextTimelineEntryPayload] = []
        temporal_info = LLMTemporalAnalysisPayload()

        # Analysiere Text-Erscheinungen über Zeit
        for i, frame_data in enumerate(frames_data):
            ocr_text = frame_data.ocr_text
            if not ocr_text:
                continue

            timestamp = frame_data.timestamp

            timeline.append(
                LLMTextTimelineEntryPayload(
                    frame_index=i,
                    timestamp=float(timestamp),
                    text_snippet=ocr_text[:50] + "..."
                    if len(ocr_text) > 50
                    else ocr_text,
                    confidence=frame_data.ocr_confidence,
                )
            )

            # Erkenne Änderungspunkte in der Text-Stabilität
            if len(timeline) > 1:
                temporal_info.change_points = self._detect_text_change_points(timeline)

        temporal_info.text_appearance_timeline = timeline
        temporal_info.stability_scores["temporal_density"] = (
            len(timeline) / len(list(frames_data)) if frames_data else 0.0
        )

        if timeline:
            temporal_info.duration_analysis["covered_duration"] = (
                timeline[-1].timestamp - timeline[0].timestamp
            )

        return temporal_info

    def _detect_text_change_points(
        self, timeline: list[LLMTextTimelineEntryPayload]
    ) -> list[int]:
        """Erkennt Punkte wo sich Text-Inhalt signifikant ändert."""
        change_points: list[int] = []

        for i in range(1, len(timeline)):
            current_text = timeline[i].text_snippet
            previous_text = timeline[i - 1].text_snippet
            # Einfache Ähnlichkeitsberechnung
            similarity = self._calculate_text_similarity(current_text, previous_text)

            if similarity < 0.3:  # Signifikante Änderung
                change_points.append(timeline[i].frame_index)

        return change_points

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Berechnet einfache Text-Ähnlichkeit."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_enriched_confidence(
        self, enriched_metadata: LLMEnrichedMetadataPayload
    ) -> dict[str, float]:
        """Berechnet erweiterte Konfidenz-Scores."""
        confidence_scores: dict[str, float] = {
            "llm_confidence": 0.0,
            "frame_quality_confidence": 0.0,
            "temporal_stability_confidence": 0.0,
            "overall_confidence": 0.0,
        }

        # LLM-Konfidenz
        if enriched_metadata.llm_extracted:
            confidence_scores["llm_confidence"] = (
                self.llm_extractor.calculate_confidence(enriched_metadata.llm_extracted)
            )

        # Frame-Qualität-Konfidenz
        if enriched_metadata.frame_context.quality_scores:
            frame_quality_scores = enriched_metadata.frame_context.quality_scores
            confidence_scores["frame_quality_confidence"] = sum(
                frame_quality_scores
            ) / len(frame_quality_scores)

        # Temporale Stabilität
        if enriched_metadata.temporal_analysis.text_appearance_timeline:
            timeline_confidences = [
                text_frame.confidence
                for text_frame in enriched_metadata.temporal_analysis.text_appearance_timeline
            ]
            if timeline_confidences:
                confidence_scores["temporal_stability_confidence"] = sum(
                    timeline_confidences
                ) / len(timeline_confidences)

        # Gesamt-Konfidenz (gewichteter Durchschnitt)
        weights = {
            "llm_confidence": 0.5,
            "frame_quality_confidence": 0.2,
            "temporal_stability_confidence": 0.3,
        }
        overall = sum(confidence_scores[key] * weights[key] for key in weights.keys())
        confidence_scores["overall_confidence"] = overall

        # Keep deterministic decimal outputs for comparisons and UI display.
        for key in confidence_scores:
            confidence_scores[key] = round(float(confidence_scores[key]), 6)

        return confidence_scores


class FrameDataProcessor:
    """
    Verarbeitet Frame-Daten aus sample_frames_coroutine für Metadaten-Anreicherung.
    """

    @staticmethod
    def process_coroutine_output(
        coroutine_result: Iterable[LLMFrameProcessorInput],
    ) -> list[LLMFrameDataPayload]:
        """
        Verarbeitet die Ausgabe von sample_frames_coroutine zu standardisiertem Format.

        Args:
            coroutine_result: Ergebnis der sample_frames_coroutine

        Returns:
            Liste von standardisierten Frame-Daten
        """
        processed_frames: list[LLMFrameDataPayload] = []

        # Anpassung je nach Format der Coroutine-Ausgabe
        for i, frame_item in enumerate(coroutine_result):
            processed_frame = FrameDataProcessor._normalize_frame_data(frame_item, i)
            processed_frames.append(processed_frame)

        return processed_frames

    @staticmethod
    def _to_float(value: object, default: float = 0.0) -> float:
        """Konvertiert numerische Werte sicher nach float."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        return default

    @staticmethod
    def _to_int(value: object, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        return default

    @staticmethod
    def _to_bool(value: object, default: bool = False) -> bool:
        return value is True if isinstance(value, bool) else default

    @staticmethod
    def _normalize_frame_data(
        frame_item: LLMFrameProcessorInput, frame_index: int
    ) -> LLMFrameDataPayload:
        """Normalisiert Frame-Daten zu einheitlichem Format."""
        normalized: dict[str, object] = {
            "frame_index": frame_index,
            "timestamp": float(frame_index),
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "has_text": False,
            "has_patient_info": False,
            "has_ui_elements": False,
            "is_endoscopy_view": False,
            "quality_score": 0.5,
        }

        # Verschiedene Input-Formate handhaben
        if isinstance(frame_item, Mapping):
            frame_payload = _coerce_str_object_map(frame_item)
            if frame_payload is None:
                return LLMFrameDataPayload.model_validate(normalized)

            if "frame_index" in frame_payload:
                normalized["frame_index"] = FrameDataProcessor._to_int(
                    frame_payload.get("frame_index"), frame_index
                )
            normalized["timestamp"] = FrameDataProcessor._to_float(
                frame_payload.get("timestamp", frame_index), float(frame_index)
            )
            ocr_text = frame_payload.get("ocr_text")
            if isinstance(ocr_text, str):
                normalized["ocr_text"] = ocr_text
            else:
                normalized["ocr_text"] = ""
            normalized["ocr_confidence"] = FrameDataProcessor._to_float(
                frame_payload.get("ocr_confidence"), 0.0
            )
            normalized["has_text"] = FrameDataProcessor._to_bool(
                frame_payload.get("has_text", False), False
            )
            normalized["has_patient_info"] = FrameDataProcessor._to_bool(
                frame_payload.get("has_patient_info", False), False
            )
            normalized["has_ui_elements"] = FrameDataProcessor._to_bool(
                frame_payload.get("has_ui_elements", False), False
            )
            normalized["is_endoscopy_view"] = FrameDataProcessor._to_bool(
                frame_payload.get("is_endoscopy_view", False), False
            )
            normalized["quality_score"] = FrameDataProcessor._to_float(
                frame_payload.get("quality_score"), 0.5
            )
            frame_id_value = frame_payload.get("frame_id")
            if isinstance(frame_id_value, int):
                normalized["frame_id"] = frame_id_value
            frame_number_value = frame_payload.get("frame_number")
            if isinstance(frame_number_value, int):
                normalized["frame_number"] = frame_number_value
        elif isinstance(frame_item, FrameCollectionItem):
            item_payload = frame_item.model_dump()
            frame_index_payload = item_payload.get("frame_id")
            frame_number_payload = item_payload.get("frame_number")
            if isinstance(frame_index_payload, int):
                normalized["frame_index"] = frame_index_payload
            elif isinstance(frame_number_payload, int):
                normalized["frame_index"] = frame_number_payload
            else:
                normalized["frame_index"] = frame_index
            normalized.update(
                {
                    "frame_id": item_payload.get("frame_id"),
                    "frame_number": item_payload.get("frame_number"),
                    "ocr_text": item_payload.get("ocr_text", ""),
                    "ocr_confidence": FrameDataProcessor._to_float(
                        item_payload.get("ocr_confidence"), 0.0
                    ),
                    "has_text": bool(item_payload.get("ocr_text", "")),
                    "timestamp": FrameDataProcessor._to_float(
                        frame_number_payload, float(frame_index)
                    ),
                }
            )
            meta = item_payload.get("meta")
            meta_payload = (
                _coerce_str_object_map(cast(Mapping[str, object], meta))
                if isinstance(meta, Mapping)
                else None
            )
            if meta_payload is not None:
                normalized["has_patient_info"] = FrameDataProcessor._to_bool(
                    meta_payload.get("has_patient_info", False), False
                )
                normalized["has_ui_elements"] = FrameDataProcessor._to_bool(
                    meta_payload.get("has_ui_elements", False), False
                )
                normalized["is_endoscopy_view"] = FrameDataProcessor._to_bool(
                    meta_payload.get("is_endoscopy_view", False), False
                )
                normalized["quality_score"] = FrameDataProcessor._to_float(
                    meta_payload.get("quality_score", normalized["quality_score"]),
                    0.5,
                )
        else:
            frame_payload = list(frame_item)
            if len(frame_payload) < 2:
                return LLMFrameDataPayload.model_validate(normalized)
            # Annahme: (frame_data, ocr_text) oder ähnlich
            ocr_payload = frame_payload[1]
            if isinstance(ocr_payload, str):
                normalized["ocr_text"] = ocr_payload
                normalized["has_text"] = bool(ocr_payload)

            if len(frame_payload) >= 3:
                raw_ocr_confidence = frame_payload[2]
                normalized["ocr_confidence"] = FrameDataProcessor._to_float(
                    raw_ocr_confidence, 0.0
                )

            if len(frame_payload) >= 4:
                raw_meta = frame_payload[3]
                if isinstance(raw_meta, Mapping):
                    meta_payload = _coerce_str_object_map(
                        cast(Mapping[str, object], raw_meta)
                    )
                    if meta_payload is not None:
                        normalized["has_patient_info"] = FrameDataProcessor._to_bool(
                            meta_payload.get("has_patient_info", False), False
                        )
                        normalized["has_ui_elements"] = FrameDataProcessor._to_bool(
                            meta_payload.get("has_ui_elements", False), False
                        )
                        normalized["is_endoscopy_view"] = FrameDataProcessor._to_bool(
                            meta_payload.get("is_endoscopy_view", False), False
                        )
                        normalized["quality_score"] = FrameDataProcessor._to_float(
                            meta_payload.get(
                                "quality_score", normalized["quality_score"]
                            ),
                            0.5,
                        )

        return LLMFrameDataPayload.model_validate(normalized)


# Erweiterte Factory-Funktion für angereicherte Extraktion
def create_enriched_extractor() -> EnrichedMetadataExtractor:
    """Erstellt einen angereicherten Metadaten-Extractor."""
    llm_extractor = create_fast_extractor()
    frame_optimizer = FrameSamplingOptimizer()
    return EnrichedMetadataExtractor(llm_extractor, frame_optimizer)


class VideoMetadataEnricher:
    """
    Haupt-Klasse für Video-Metadaten-Anreicherung mit verschiedenen Datenquellen.
    """

    def __init__(self):
        self.enriched_extractor = create_enriched_extractor()
        self.frame_processor = FrameDataProcessor()

    def enrich_from_multiple_sources(
        self,
        video_path: str,
        frame_samples: list[LLMFrameProcessorInput],
        ocr_texts: list[str],
        existing_metadata: dict[str, object],
    ) -> dict[str, object]:
        """
        Reichert Metadaten aus verschiedenen Quellen an.

        Args:
            video_path: Pfad zum Video
            frame_samples: Ausgabe von sample_frames_coroutine
            ocr_texts: Bereits extrahierte OCR-Texte
            existing_metadata: Bestehende Metadaten (z.B. aus FrameCleaner)

        Returns:
            Vollständig angereicherte Metadaten
        """

        final_metadata: dict[str, object] = {
            "source_info": {
                "video_path": video_path,
                "processing_method": "enriched_extraction",
                "timestamp": time.time(),
            },
            "enriched_data": {},
            "legacy_data": existing_metadata or {},
            "integration_stats": {},
        }

        # 1. Verarbeite Frame-Samples falls vorhanden
        if frame_samples:
            processed_frames = self.frame_processor.process_coroutine_output(
                frame_samples
            )
            enriched_data = self.enriched_extractor.extract_from_frame_sequence(
                processed_frames, ocr_texts
            )
            final_metadata["enriched_data"] = enriched_data.model_dump()

        # 2. Merge mit bestehenden Metadaten
        if existing_metadata:
            final_metadata = self._merge_metadata_sources(
                final_metadata, existing_metadata
            )

        # 3. Berechne Integrations-Statistiken
        final_metadata["integration_stats"] = self._calculate_integration_stats(
            final_metadata
        )

        return final_metadata

    def _merge_metadata_sources(
        self,
        enriched_metadata: dict[str, object],
        existing_metadata: dict[str, object],
    ) -> dict[str, object]:
        """Merged angereicherte Metadaten mit bestehenden Daten."""

        # Prioritäts-basiertes Merging
        merged = enriched_metadata.copy()

        # LLM-Daten haben Priorität über legacy OCR-Extraktion
        enriched_data = enriched_metadata.get("enriched_data")
        if not isinstance(enriched_data, dict):
            return merged
        try:
            enriched_data_payload = LLMEnrichedMetadataPayload.model_validate(
                enriched_data
            )
        except ValidationError:
            return merged

        llm_data = enriched_data_payload.llm_extracted
        existing_fallback = merged.get("fallback_data")
        merged_fallback_data: dict[str, object] = {}
        if isinstance(existing_fallback, Mapping):
            merged_fallback_data = (
                _coerce_str_object_map(cast(Mapping[str, object], existing_fallback))
                or {}
            )

        merged["fallback_data"] = merged_fallback_data

        fallback_candidates = [
            "first_name",
            "last_name",
            "examination_date",
            "gender",
            "dob",
        ]
        for key in fallback_candidates:
            existing_value = existing_metadata.get(key)
            llm_value = getattr(llm_data, key)
            if existing_value is not None and llm_value in {"", "unknown"}:
                merged_fallback_data[key] = existing_value

        return merged

    @staticmethod
    def _coerce_metadata_payload(
        source_data: object,
    ) -> LLMMetadataPayload | None:
        if not isinstance(source_data, Mapping):
            return None

        source_payload = _coerce_str_object_map(cast(Mapping[str, object], source_data))
        if source_payload is None:
            return None

        nested_payload = source_payload.get("llm_extracted")
        if isinstance(nested_payload, Mapping):
            source_payload = _coerce_str_object_map(
                cast(Mapping[str, object], nested_payload)
            )
            if source_payload is None:
                return None

        try:
            return LLMMetadataPayload.model_validate(source_payload)
        except ValidationError:
            return None

    def _calculate_integration_stats(
        self, metadata: dict[str, object]
    ) -> LLMEvaluationResultPayload:
        """Berechnet Statistiken über die Metadaten-Integration."""

        data_sources_used: list[str] = []
        required_fields = (
            "first_name",
            "last_name",
            "dob",
            "examination_date",
            "gender",
        )
        filled_fields = 0

        # Erkenne verwendete Datenquellen
        if _coerce_str_object_map(metadata.get("enriched_data")) is not None:
            data_sources_used.append("enriched_llm")
        if _coerce_str_object_map(metadata.get("legacy_data")) is not None:
            data_sources_used.append("legacy_ocr")
        if _coerce_str_object_map(metadata.get("fallback_data")) is not None:
            data_sources_used.append("fallback_data")

        for source in ("enriched_data", "legacy_data", "fallback_data"):
            source_payload = self._coerce_metadata_payload(metadata.get(source))
            if source_payload is None:
                continue

            for field in required_fields:
                field_value = getattr(source_payload, field)
                if isinstance(field_value, str) and field_value not in [
                    None,
                    "",
                    "unknown",
                ]:
                    filled_fields += 1
                    break  # Feld ist gefüllt, nächstes Feld

        source_count = len(required_fields)
        return LLMEvaluationResultPayload(
            data_sources_used=data_sources_used,
            confidence_comparison={},
            data_completeness=filled_fields / source_count,
        )


class AsyncMetadataWorker:
    """Single-threaded wrapper for isolating blocking Ollama metadata calls."""

    def __init__(
        self,
        extractor: Optional[LLMMetadataExtractor] = None,
        base_url: str | None = None,
        enable_cache: bool = True,
        provider: str = "ollama",
        preferred_model: str | None = None,
        model_timeout: int | None = None,
    ) -> None:
        if extractor is None:
            self.extractor = LLMMetadataExtractor(
                base_url=base_url,
                enable_cache=enable_cache,
                provider=provider,
                preferred_model=preferred_model,
                model_timeout=model_timeout,
            )
        else:
            self.extractor = extractor
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lx-llm-metadata",
        )

    def submit(self, text: str) -> Future[Optional[SensitiveMeta]]:
        """Submit metadata extraction without blocking the caller."""
        return self._executor.submit(self._safe_extract_metadata, text)

    def extract_metadata(
        self,
        text: str,
        timeout: Optional[float] = None,
    ) -> Optional[SensitiveMeta]:
        """Blocking convenience wrapper that returns None on timeout/failure."""
        future = self.submit(text)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(
                "Async metadata extraction timed out after %s seconds",
                timeout,
            )
            return None

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> "AsyncMetadataWorker":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()

    def _safe_extract_metadata(self, text: str) -> Optional[SensitiveMeta]:
        try:
            return self.extractor.extract_metadata(text)
        except Exception as exc:
            logger.warning("Async metadata extraction failed: %s", exc)
            return None
