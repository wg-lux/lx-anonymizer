from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- General Toggles ---
    MODE: str = "production"
    DEBUG_SAVE_FRAMES: bool = False

    # --- LLM / Ollama Configuration ---
    LLM_ENABLED: bool = True
    LLM_MODEL: str = "llama3.2:1b"
    LLM_TIMEOUT: int = 30

    # --- Performance & Sampling ---
    MAX_FRAMES_TO_SAMPLE: int = 50
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


settings = Settings()
