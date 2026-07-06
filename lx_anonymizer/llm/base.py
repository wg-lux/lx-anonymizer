from typing import Optional, Protocol

from lx_anonymizer.sensitive_meta_interface import SensitiveMeta


class BaseLLMExtractor(Protocol):
    """Minimal provider-agnostic interface for metadata extraction."""

    current_model: Optional[dict[str, object]]

    def extract_metadata(self, text: str) -> Optional[SensitiveMeta]: ...
