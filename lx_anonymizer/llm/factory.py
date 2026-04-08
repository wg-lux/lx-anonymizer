from lx_anonymizer.config import settings
from lx_anonymizer.llm.llm_extractor import LLMMetadataExtractor


class LLMFactory:
    """Central place for building provider-specific LLM helpers."""

    @staticmethod
    def create_metadata_extractor() -> LLMMetadataExtractor:
        return LLMMetadataExtractor(
            provider=settings.LLM_PROVIDER,
            base_url=settings.resolved_llm_base_url,
            preferred_model=settings.LLM_MODEL,
            model_timeout=settings.LLM_TIMEOUT,
        )
