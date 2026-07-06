import gc
import os
from collections.abc import Callable
from typing import Optional, Protocol, Self, TypeAlias, TypeVar, cast

import torch  # type: ignore[import-untyped]
from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    pipeline as transformers_pipeline,  # type: ignore[reportUnknownVariableType]
)

from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)


class Phi4ModelProtocol(Protocol):
    """Boundary protocol for the dynamically loaded causal LM instance."""


class Phi4TokenizerProtocol(Protocol):
    """Boundary protocol for the dynamically loaded tokenizer instance."""


class TextGenerationPipelineProtocol(Protocol):
    """Boundary protocol for the transformers text-generation pipeline."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class TrocrTokenizerProtocol(Protocol):
    """Boundary protocol for the tokenizer exposed by TrOCRProcessor."""


class TrocrProcessorProtocol(Protocol):
    tokenizer: TrocrTokenizerProtocol


class TrocrModelProtocol(Protocol):
    def cuda(self) -> Self: ...

    def to(self, device: torch.device) -> Self: ...


_LoadedFactoryResult = TypeVar("_LoadedFactoryResult", covariant=True)


class _PretrainedFactory(Protocol[_LoadedFactoryResult]):
    def from_pretrained(
        self, pretrained_model_name_or_path: str, **kwargs: object
    ) -> _LoadedFactoryResult: ...


class _TextGenerationPipelineFactory(Protocol):
    def __call__(
        self,
        task: str,
        *,
        model: Phi4ModelProtocol,
        tokenizer: Phi4TokenizerProtocol,
    ) -> TextGenerationPipelineProtocol: ...


Phi4LoadResult: TypeAlias = (
    tuple[Phi4ModelProtocol, Phi4TokenizerProtocol, TextGenerationPipelineProtocol]
    | tuple[None, None, None]
)
TrocrLoadResult: TypeAlias = (
    tuple[
        TrocrProcessorProtocol,
        TrocrModelProtocol,
        TrocrTokenizerProtocol | None,
        torch.device | None,
    ]
    | tuple[None, None, None, None]
)

_manual_seed = cast(
    Callable[[int], object],
    torch.random.manual_seed,  # type: ignore[reportUnknownMemberType]
)
_load_phi4_model = cast(
    _PretrainedFactory[Phi4ModelProtocol], AutoModelForCausalLM
).from_pretrained
_load_phi4_tokenizer = cast(
    _PretrainedFactory[Phi4TokenizerProtocol], AutoTokenizer
).from_pretrained
_load_trocr_processor = cast(
    _PretrainedFactory[TrocrProcessorProtocol], TrOCRProcessor
).from_pretrained
_load_trocr_model = cast(
    _PretrainedFactory[TrocrModelProtocol], VisionEncoderDecoderModel
).from_pretrained
_text_generation_pipeline = cast(_TextGenerationPipelineFactory, transformers_pipeline)


class ModelService:
    """
    Singleton-Pattern-Implementierung für die Modellverwaltung.
    Stellt sicher, dass Modelle nur einmal geladen werden und zwischen Aufrufen wiederverwendet werden können.
    """

    _instance = None

    # Phi-4 LLM Modell
    phi4_model: Optional[Phi4ModelProtocol] = None
    phi4_tokenizer: Optional[Phi4TokenizerProtocol] = None
    phi4_pipe: Optional[TextGenerationPipelineProtocol] = None

    # TrOCR Modell
    trocr_processor: Optional[TrocrProcessorProtocol] = None
    trocr_model: Optional[TrocrModelProtocol] = None
    trocr_tokenizer: Optional[TrocrTokenizerProtocol] = None
    trocr_device: Optional[torch.device] = None

    def __new__(cls) -> Self:
        """Implementierung des Singleton-Patterns"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            # Setze CUDA-Konfiguration
            os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
            _manual_seed(0)
        return cls._instance

    def get_device(self) -> torch.device:
        """Ermittelt das optimale Gerät für die Modellausführung"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU")
        return device

    def load_phi4_model(self, force_reload: bool = False) -> Phi4LoadResult:
        """Lädt das Phi-4 LLM Modell, wenn es noch nicht geladen ist oder wenn force_reload=True"""
        if (
            self.phi4_model is not None
            and self.phi4_tokenizer is not None
            and self.phi4_pipe is not None
            and not force_reload
        ):
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
                model = _load_phi4_model(
                    model_id,
                    device_map="auto",  # or "cpu"/"cuda", or a dictionary mapping module names to devices
                    torch_dtype="auto",
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                )
                tokenizer = _load_phi4_tokenizer(model_id)
                pipe = _text_generation_pipeline(
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
                    model = _load_phi4_model(
                        model_id,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=False,
                        low_cpu_mem_usage=True,
                    )
                    tokenizer = _load_phi4_tokenizer(model_id)
                    pipe = _text_generation_pipeline(
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

    def load_trocr_model(self, force_reload: bool = False) -> TrocrLoadResult:
        """Lädt das TrOCR-Modell, wenn es noch nicht geladen ist oder wenn force_reload=True"""
        if (
            self.trocr_processor is not None
            and self.trocr_model is not None
            and not force_reload
        ):
            logger.debug("Using cached TrOCR model")
            return (
                self.trocr_processor,
                self.trocr_model,
                self.trocr_tokenizer,
                self.trocr_device,
            )

        logger.info("Loading TrOCR model...")
        # Speicher bereinigen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        device = self.get_device()

        try:
            # TrOCRProcessor verwenden (kombiniert Feature Extractor und Tokenizer)
            processor = _load_trocr_processor("microsoft/trocr-base-str")
            model = _load_trocr_model("microsoft/trocr-base-str")

            # Auf Gerät verschieben mit zusätzlichen CUDA-Optimierungen
            if device.type == "cuda":
                try:
                    # CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                    model = model.cuda()

                    # Optional: Mixed precision für bessere Performance
                    # model = model.half()  # Nur aktivieren, wenn gemischte Präzision gewünscht

                    logger.info(
                        f"TrOCR model loaded on CUDA device: {torch.cuda.get_device_name(0)}"
                    )
                    logger.info(
                        f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                    )
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(
                            f"CUDA out of memory when loading TrOCR model: {e}"
                        )
                        # Auf CPU zurückfallen
                        device = torch.device("cpu")
                        model = _load_trocr_model("microsoft/trocr-base-str")
                        logger.info("Fallback: TrOCR model loaded on CPU")
                    else:
                        raise

            model.to(device)

            # Cache-Modelle speichern
            self.trocr_processor = processor
            self.trocr_model = model
            self.trocr_tokenizer = (
                processor.tokenizer
            )  # TrOCRProcessor enthält bereits den Tokenizer
            self.trocr_device = device

            logger.info(f"Successfully loaded TrOCR model on {device}")
            return processor, model, processor.tokenizer, device

        except Exception as e:
            logger.error(f"Error initializing TrOCR model: {e}")
            return None, None, None, None

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


model_service = ModelService()
