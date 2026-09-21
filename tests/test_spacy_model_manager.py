import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pytest
from pytest import MonkeyPatch
from spacy.language import Language

from lx_anonymizer.ner import bundled_spacy, spacy_extractor
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
    monkeypatch.setattr(spacy_extractor.settings, "SPACY_AUTO_DOWNLOAD", False)
    monkeypatch.setattr(spacy_extractor.settings, "SPACY_MODEL", "de_core_news_custom")


@pytest.fixture(autouse=True)
def reset_spacy_model_manager() -> Iterator[None]:
    previous_instance = SpacyModelManager._instance  # pyright: ignore[reportPrivateUsage]
    SpacyModelManager._instance = None  # pyright: ignore[reportPrivateUsage]
    yield
    SpacyModelManager._instance = previous_instance  # pyright: ignore[reportPrivateUsage]


def _raise_missing_model(_model_name: str) -> Language:
    raise OSError("missing model")


def test_default_model_performs_offline_german_ner(monkeypatch: MonkeyPatch) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setattr(spacy_extractor.settings, "SPACY_MODEL", "de_core_news_sm")
    monkeypatch.setenv(SpacyModelManager.STRICT_MODEL_ENV, "1")
    real_load = spacy_extractor.spacy.load
    calls: list[Path] = []

    def load_path_only(model: str | Path) -> Language:
        assert isinstance(model, Path), (
            "Default model must not require an installed model package"
        )
        calls.append(model)
        return real_load(model)

    def reject_download(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("The default model must never download at runtime")

    monkeypatch.setattr(spacy_extractor.spacy, "load", load_path_only)
    monkeypatch.setattr(spacy_extractor.subprocess, "run", reject_download)
    model = SpacyModelManager.get_model()
    assert model.lang == "de"
    assert model.has_pipe("ner")
    assert model.meta["version"] == "3.8.0"
    assert any(
        entity.label_ == "PER" for entity in model("Angela Merkel lebt in Berlin.").ents
    )
    assert SpacyModelManager.get_model() is model
    assert calls == [bundled_spacy.bundled_german_model_path()]


@pytest.mark.parametrize("auto_download", ["0", "1"])
def test_missing_bundled_model_fails_without_download_or_blank_fallback(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    auto_download: str,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, auto_download)
    monkeypatch.setattr(
        bundled_spacy, "bundled_german_model_path", lambda: tmp_path / "missing"
    )
    with pytest.raises(
        RuntimeError, match="bundled German spaCy model is missing or corrupt"
    ):
        SpacyModelManager.get_model("de_core_news_sm")


def test_bundled_model_manifest_covers_all_files() -> None:
    root = bundled_spacy.bundled_german_model_path()
    manifest = json.loads((root.parent / "manifest.json").read_text())
    actual = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert actual == manifest["files"]
    assert "LICENSE" in actual
    assert "LICENSES_SOURCES" in actual


def test_bundled_model_rejects_untrained_pipeline(monkeypatch: MonkeyPatch) -> None:
    def blank_model(_path: Path) -> Language:
        return spacy_extractor.spacy.blank("de")

    monkeypatch.setattr(spacy_extractor.spacy, "load", blank_model)
    with pytest.raises(RuntimeError, match="trained model contract"):
        SpacyModelManager.get_model("de_core_news_sm")


def test_missing_model_uses_blank_fallback_without_download(
    monkeypatch: MonkeyPatch,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    calls: list[str] = []

    def fake_load(model_name: str) -> Language:
        calls.append(model_name)
        raise OSError("missing model")

    monkeypatch.setattr(spacy_extractor.spacy, "load", fake_load)

    nlp = SpacyModelManager.get_model("de_core_news_custom")

    assert isinstance(nlp, Language)
    assert nlp.lang == "de"
    assert "sentencizer" in nlp.pipe_names
    assert calls == ["de_core_news_custom"]


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

    nlp = SpacyModelManager.get_model("de_core_news_custom")

    assert isinstance(nlp, Language)
    assert nlp.lang == "de"
    assert calls == ["de_core_news_custom", "de_core_news_custom"]
    assert downloads == ["de_core_news_custom"]


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
        SpacyModelManager.get_model("de_core_news_custom")

    assert downloads == ["de_core_news_custom"]


def test_download_model_uses_current_python_runtime(
    monkeypatch: MonkeyPatch,
) -> None:
    commands: list[tuple[list[str], bool]] = []
    cache_invalidated = False

    def fake_run(command: list[str], *, check: bool) -> None:
        commands.append((command, check))

    def fake_invalidate_caches() -> None:
        nonlocal cache_invalidated
        cache_invalidated = True

    monkeypatch.setattr(spacy_extractor.subprocess, "run", fake_run)
    monkeypatch.setattr(
        spacy_extractor.importlib,
        "invalidate_caches",
        fake_invalidate_caches,
    )

    SpacyModelManager._download_model(  # pyright: ignore[reportPrivateUsage]
        "de_core_news_custom"
    )

    assert commands == [
        (
            [
                sys.executable,
                "-m",
                "spacy",
                "download",
                "de_core_news_custom",
            ],
            True,
        )
    ]
    assert cache_invalidated is True


def test_download_model_reports_subprocess_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    def fake_run(command: list[str], *, check: bool) -> None:
        raise subprocess.CalledProcessError(returncode=23, cmd=command)

    monkeypatch.setattr(spacy_extractor.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 23"):
        SpacyModelManager._download_model(  # pyright: ignore[reportPrivateUsage]
            "de_core_news_custom"
        )


def test_invalid_spacy_boolean_env_value_raises(monkeypatch: MonkeyPatch) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setenv(SpacyModelManager.AUTO_DOWNLOAD_ENV, "sometimes")
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)

    with pytest.raises(RuntimeError, match="Invalid boolean value"):
        SpacyModelManager.get_model("de_core_news_custom")


def test_missing_model_raises_in_strict_mode(monkeypatch: MonkeyPatch) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.setenv(SpacyModelManager.STRICT_MODEL_ENV, "1")

    with pytest.raises(RuntimeError, match="de_core_news_custom"):
        SpacyModelManager.get_model("de_core_news_custom")


def test_missing_model_raises_for_clinical_profile(
    monkeypatch: MonkeyPatch,
) -> None:
    _configure_nonclinical_spacy_env(monkeypatch)
    monkeypatch.setattr(spacy_extractor.spacy, "load", _raise_missing_model)
    monkeypatch.setenv("MODE", "clinical")

    with pytest.raises(RuntimeError, match="de_core_news_custom"):
        SpacyModelManager.get_model("de_core_news_custom")


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
