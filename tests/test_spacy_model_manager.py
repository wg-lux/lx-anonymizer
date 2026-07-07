from typing import Iterator

import pytest
from pytest import MonkeyPatch
from spacy.language import Language

from lx_anonymizer.ner import spacy_extractor
from lx_anonymizer.ner.spacy_extractor import PatientDataExtractor, SpacyModelManager


def _configure_nonclinical_spacy_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(SpacyModelManager.MODEL_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.SETTINGS_MODEL_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.SETTINGS_AUTO_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.STRICT_MODEL_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.SETTINGS_STRICT_MODEL_ENV, raising=False)
    monkeypatch.delenv(SpacyModelManager.PROFILE_ENV, raising=False)
    monkeypatch.setenv("MODE", "production")


@pytest.fixture(autouse=True)
def reset_spacy_model_manager() -> Iterator[None]:
    previous_instance = SpacyModelManager._instance  # pyright: ignore[reportPrivateUsage]
    SpacyModelManager._instance = None  # pyright: ignore[reportPrivateUsage]
    yield
    SpacyModelManager._instance = previous_instance  # pyright: ignore[reportPrivateUsage]


def _raise_missing_model(_model_name: str) -> Language:
    raise OSError("missing model")


def test_missing_model_uses_blank_fallback_without_download(
    monkeypatch: MonkeyPatch,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    calls: list[str] = []

    def fake_load(model_name: str) -> Language:
        calls.append(model_name)
        raise OSError("missing model")

    monkeypatch.setattr(spacy_extractor.spacy, "load", fake_load)

    nlp = SpacyModelManager.get_model("de_core_news_sm")

    assert isinstance(nlp, Language)
    assert nlp.lang == "de"
    assert "sentencizer" in nlp.pipe_names
    assert calls == ["de_core_news_sm"]


@pytest.mark.parametrize(
    "auto_download_env",
    [
        SpacyModelManager.AUTO_DOWNLOAD_ENV,
        SpacyModelManager.SETTINGS_AUTO_DOWNLOAD_ENV,
    ],
)
def test_missing_model_attempts_download_when_auto_download_enabled(
    monkeypatch: MonkeyPatch,
    auto_download_env: str,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setenv(auto_download_env, "1")
    calls: list[str] = []
    downloads: list[str] = []

    def fake_load(model_name: str) -> Language:
        calls.append(model_name)
        if len(calls) == 1:
            raise OSError("missing model")
        return spacy_extractor.spacy.blank("de")

    def fake_download(
        _manager: type[SpacyModelManager],
        model_name: str,
    ) -> None:
        downloads.append(model_name)

    monkeypatch.setattr(spacy_extractor.spacy, "load", fake_load)
    monkeypatch.setattr(
        SpacyModelManager,
        "_download_model",
        classmethod(fake_download),
    )

    nlp = SpacyModelManager.get_model("de_core_news_sm")

    assert isinstance(nlp, Language)
    assert nlp.lang == "de"
    assert calls == ["de_core_news_sm", "de_core_news_sm"]
    assert downloads == ["de_core_news_sm"]


def test_auto_download_reports_model_still_missing_after_download(
    monkeypatch: MonkeyPatch,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, "1")
    downloads: list[str] = []

    def fake_download(
        _manager: type[SpacyModelManager],
        model_name: str,
    ) -> None:
        downloads.append(model_name)

    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.setattr(
        SpacyModelManager,
        "_download_model",
        classmethod(fake_download),
    )

    with pytest.raises(RuntimeError, match="still not loadable"):
        SpacyModelManager.get_model("de_core_news_sm")

    assert downloads == ["de_core_news_sm"]


def test_invalid_spacy_boolean_env_value_raises(monkeypatch: MonkeyPatch) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, "sometimes")
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)

    with pytest.raises(RuntimeError, match="Invalid boolean value"):
        SpacyModelManager.get_model("de_core_news_sm")


def test_missing_model_raises_in_strict_mode(monkeypatch: MonkeyPatch) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.setenv(SpacyModelManager.STRICT_MODEL_ENV, "1")

    with pytest.raises(RuntimeError, match="de_core_news_sm"):
        SpacyModelManager.get_model("de_core_news_sm")


def test_missing_model_raises_for_clinical_profile(
    monkeypatch: MonkeyPatch,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.setenv("MODE", "clinical")

    with pytest.raises(RuntimeError, match="de_core_news_sm"):
        SpacyModelManager.get_model("de_core_news_sm")


def test_patient_extractor_matches_with_blank_fallback(
    monkeypatch: MonkeyPatch,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)

    extractor = PatientDataExtractor()
    patient_info = extractor("Patient: Lux, Thomas geb. 15.02.2024 Fallnr.: A123")

    assert patient_info["first_name"] == "Thomas"
    assert patient_info["last_name"] == "Lux"
    assert patient_info["dob"] == "2024-02-15"
    assert patient_info["casenumber"] == "A123"
