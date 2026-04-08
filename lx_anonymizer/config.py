from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- General Toggles ---
    MODE: str = "production"
    DEBUG_SAVE_FRAMES: bool = False

    # --- LLM Configuration ---
    # Conservative library default: require explicit opt-in before doing network/model work.
    LLM_ENABLED: bool = False
    # When enabled, prefer a laptop-friendly local backend by default.
    LLM_PROVIDER: Literal["vllm", "ollama"] = "ollama"
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "qwen2.5:7b-instruct"
    LLM_TIMEOUT: int = 45
    LLM_MAX_CALLS_PER_VIDEO: int = 1
    LLM_MIN_TEXT_LENGTH: int = 32
    REPORT_LLM_MIN_TEXT_LENGTH: int = 64
    REPORT_OCR_CORRECTION_MIN_TEXT_LENGTH: int = 120

    # --- Performance & Sampling ---
    MAX_FRAMES_TO_SAMPLE: int = 24
    SMART_EARLY_STOPPING: bool = True

    # --- OCR / Detection ---
    OCR_CONFIDENCE_THRESHOLD: float = 0.6

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
