from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- General Toggles ---
    MODE: str = "production"
    DEBUG_SAVE_FRAMES: bool = False

    # --- spaCy / Clinical NER ---
    SPACY_MODEL: str = "de_core_news_sm"
    SPACY_AUTO_DOWNLOAD: bool = False
    SPACY_STRICT: bool = False

    # --- LLM Configuration ---
    # Conservative library default: require explicit opt-in before doing network/model work.
    LLM_ENABLED: bool = False
    # When enabled, prefer a laptop-friendly local backend by default.
    LLM_PROVIDER: Literal["vllm", "ollama"] = "ollama"
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "lx-gemma4-e2b-json"
    LLM_TIMEOUT: int = 120
    LLM_MAX_CALLS_PER_VIDEO: int = 1
    LLM_MIN_TEXT_LENGTH: int = 32
    REPORT_LLM_MIN_TEXT_LENGTH: int = 64
    REPORT_OCR_CORRECTION_MIN_TEXT_LENGTH: int = 120

    # --- Performance & Sampling ---
    MAX_FRAMES_TO_SAMPLE: int = 24
    SMART_EARLY_STOPPING: bool = True
    FRAME_CLEANER_QUALITY_PROFILE: Literal["fast", "balanced", "quality"] = "balanced"

    # --- OCR / Detection ---
    OCR_CONFIDENCE_THRESHOLD: float = 0.6
    RAPIDOCR_ACCELERATION: Literal["auto", "cpu", "cuda"] = "auto"
    PHI_REGION_DETECTOR_MODEL_PATH: str = ""
    PHI_REGION_DETECTOR_MODEL_SHA256: str = ""
    PHI_REGION_DETECTOR_REQUIRED: bool = False
    PHI_REGION_DETECTOR_CONFIDENCE: float = 0.35
    PHI_REGION_DETECTOR_NMS_THRESHOLD: float = 0.45
    PHI_REGION_DETECTOR_INPUT_SIZE: int = 640
    PHI_REGION_DETECTOR_BOX_FORMAT: Literal["yolo_xywh", "xyxy"] = "yolo_xywh"
    PHI_REGION_DETECTOR_SCORE_FORMAT: Literal["class_scores", "objectness"] = (
        "class_scores"
    )
    PHI_REGION_DETECTOR_CLASS_IDS: str = ""

    # --- Masking / Encoding ---
    MASKING_STRATEGY: str = "mask_overlay"
    VIDEO_ENCODER: str = "auto"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def resolved_llm_base_url(self) -> str:
        base_url = self.LLM_BASE_URL.strip()
        if base_url:
            return base_url
        if self.LLM_PROVIDER.lower() == "ollama":
            return "http://127.0.0.1:11434"
        return "http://127.0.0.1:8000"


settings = Settings()
