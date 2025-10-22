"""Compatibility layer exposing the Ollama LLM processor under a stable module name.

The test-suite and some downstream consumers expect ``lx_anonymizer.ollama_llm`` to
provide both the high level helper functions as well as the third party modules
used internally (``ollama``, ``pytesseract`` and ``PIL.Image``).  The original
implementation lived in :mod:`lx_anonymizer.ollama_llm_processor`, so this module
simply re-exports those symbols while keeping the public surface predictable.

Optional dependencies: The Ollama integration is an optional feature.  To keep the
package importable even when ``ollama`` or ``pytesseract`` are missing, we expose
light-weight placeholders that raise clear import errors upon use while still
allowing the test-suite to patch the attributes.
"""

from __future__ import annotations

from types import SimpleNamespace

from .ollama_llm_processor import (
    OllamaLLMProcessor,
    analyze_full_image_with_ollama,
    analyze_text_with_ollama,
    initialize_ollama_processor,
    replace_phi4_with_ollama,
)

try:  # pillow is an optional dependency
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - exercised only without Pillow
    def _missing_pillow(*_: object, **__: object) -> None:  # noqa: D401 - tiny helper
        """Raise an informative error when Pillow is unavailable."""

        raise ImportError(
            "Pillow is required for image based Ollama anonymization. "
            "Install the 'models' extra or add 'pillow' manually."
        )

    Image = SimpleNamespace(open=_missing_pillow)  # type: ignore[assignment]


def _missing_dependency(name: str):
    """Return a callable that raises an informative ImportError for ``name``."""

    def _raiser(*_: object, **__: object) -> None:
        raise ImportError(
            f"The optional dependency '{name}' is required for Ollama integration. "
            "Install the 'llm' extra or add it to your environment."
        )

    return _raiser


try:
    import ollama  # type: ignore
except Exception:  # pragma: no cover - exercised only without ollama
    ollama = SimpleNamespace(Client=_missing_dependency("ollama"))  # type: ignore[assignment]


try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - exercised only without pytesseract
    pytesseract = SimpleNamespace(image_to_string=_missing_dependency("pytesseract"))  # type: ignore[assignment]



__all__ = [
    "OllamaLLMProcessor",
    "analyze_full_image_with_ollama",
    "analyze_text_with_ollama",
    "initialize_ollama_processor",
    "replace_phi4_with_ollama",
    "ollama",
    "pytesseract",
    "Image",
]
