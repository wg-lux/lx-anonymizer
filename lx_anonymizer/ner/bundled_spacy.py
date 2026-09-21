"""The default trained German model is an immutable package resource."""

from pathlib import Path

import spacy
from spacy.language import Language

MODEL_NAME = "de_core_news_sm"
MODEL_VERSION = "3.8.0"


def bundled_german_model_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "spacy"
        / f"{MODEL_NAME}-{MODEL_VERSION}"
    )


def load_bundled_german_model() -> Language:
    path = bundled_german_model_path()
    try:
        model = spacy.load(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "The bundled German spaCy model is missing or corrupt. Reinstall lx-anonymizer; "
            "the default model does not require a separate download."
        ) from exc
    if (
        model.lang != "de"
        or not model.has_pipe("ner")
        or model.meta.get("version") != MODEL_VERSION
    ):
        raise RuntimeError(
            "The bundled German spaCy model does not match its trained model contract."
        )
    return model
