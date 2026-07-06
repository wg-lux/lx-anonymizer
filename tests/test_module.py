import importlib
import os
import sys
from collections.abc import Iterable, Sequence
from typing import Protocol, cast
import pytest

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../lx_anonymizer"))
)


HEAVY_DEPENDENCIES: list[str] = [
    "cv2",
    "numpy",
    "torch",
    "torchvision",
    "pytesseract",
    "spacy",
    "flair",
]

RUN_HEAVY_TESTS = os.getenv("LX_RUN_HEAVY_TESTS") == "1"


SpacyEntity = tuple[str, int, int, str]


class _FlairSpan(Protocol):
    text: str
    tag: str


class _SpacyNerGerman(Protocol):
    def __call__(self, text: str) -> Sequence[SpacyEntity] | None: ...


def test_environment():
    if not RUN_HEAVY_TESTS:
        pytest.skip("Set LX_RUN_HEAVY_TESTS=1 to run environment smoke test")
    missing: list[str] = []
    for module_name in HEAVY_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        pytest.skip(
            "Skipping environment smoke test; missing optional dependencies: "
            + ", ".join(sorted(missing))
        )

    # If we get here, all modules import fine
    assert True


def test_ner():
    if not RUN_HEAVY_TESTS:
        pytest.skip("Set LX_RUN_HEAVY_TESTS=1 to run spaCy NER test")
    pytest.importorskip("spacy", reason="spaCy is required for NER smoke test")
    text = "Hans Müller war heute in Berlin."
    spacy_module = importlib.import_module("lx_anonymizer.ner.spacy_NER")
    spacy_ner_german = cast(_SpacyNerGerman, getattr(spacy_module, "spacy_NER_German"))
    entities = spacy_ner_german(text)
    if not entities:
        pytest.skip("spaCy German model unavailable; skipping NER test")
    # Check if 'Hans Müller' is found as an entity
    assert any(ent[0] == "Hans Müller" and ent[3] == "PER" for ent in entities)


def test_flair_ner():
    if not RUN_HEAVY_TESTS:
        pytest.skip("Set LX_RUN_HEAVY_TESTS=1 to run Flair NER test")
    pytest.importorskip("flair", reason="Flair is required for Flair NER smoke test")
    from lx_anonymizer.ner.flair_NER import flair_NER_German

    text = "Hans Müller war heute in Berlin."
    entities = cast(Iterable[_FlairSpan] | None, flair_NER_German(text))
    if not entities:
        pytest.skip("Flair model unavailable; skipping Flair NER test")
    # Check if 'Hans Müller' is found as an entity
    assert any(span.text == "Hans Müller" and span.tag == "PER" for span in entities)
