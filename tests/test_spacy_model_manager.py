import pytest
from spacy.language import Language

from lx_anonymizer.ner import spacy_extractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor, SpacyModelManager


@pytest.fixture(autouse=True)
def reset_spacy_model_manager():
    previous_instance = SpacyModelManager._instance
    SpacyModelManager._instance = None
    yield
    SpacyModelManager._instance = previous_instance


def _raise_missing_model(_model_name: str) -> Language:
    raise OSError("missing model")


def test_missing_model_uses_blank_fallback_without_download(monkeypatch):
    calls: list[str] = []

    def fake_load(model_name: str) -> Language:
        calls.append(model_name)
        raise OSError("missing model")

    monkeypatch.setattr(spacy_extractor.spacy, "load", fake_load)
    monkeypatch.delenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.STRICT_MODEL_ENV, raising=False)

    nlp = SpacyModelManager.get_model("de_core_news_sm")

    assert isinstance(nlp, Language)
    assert nlp.lang == "de"
    assert "sentencizer" in nlp.pipe_names
    assert calls == ["de_core_news_sm"]


def test_missing_model_raises_in_strict_mode(monkeypatch):
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.delenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, raising=False)
    monkeypatch.setenv(SpacyModelManager.STRICT_MODEL_ENV, "1")

    with pytest.raises(RuntimeError, match="de_core_news_sm"):
        SpacyModelManager.get_model("de_core_news_sm")


def test_patient_extractor_matches_with_blank_fallback(monkeypatch):
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.delenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.STRICT_MODEL_ENV, raising=False)

    extractor = PatientDataExtractor()
    patient_info = extractor("Patient: Lux, Thomas geb. 15.02.2024 Fallnr.: A123")

    assert patient_info["first_name"] == "Thomas"
    assert patient_info["last_name"] == "Lux"
    assert patient_info["dob"] == "2024-02-15"
    assert patient_info["casenumber"] == "A123"
