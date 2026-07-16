from pathlib import Path

import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

from lx_anonymizer.config import Settings


def test_settings_accept_prefixed_spacy_env_file_keys(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("LX_ANONYMIZER_SPACY_MODEL", raising=False)
    monkeypatch.delenv("LX_ANONYMIZER_SPACY_AUTO_DOWNLOAD", raising=False)
    monkeypatch.delenv("LX_ANONYMIZER_SPACY_STRICT", raising=False)
    monkeypatch.delenv("SPACY_MODEL", raising=False)
    monkeypatch.delenv("SPACY_AUTO_DOWNLOAD", raising=False)
    monkeypatch.delenv("SPACY_STRICT", raising=False)
    monkeypatch.chdir(tmp_path)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "LX_ANONYMIZER_SPACY_MODEL=custom_de_model",
                "LX_ANONYMIZER_SPACY_AUTO_DOWNLOAD=1",
                "LX_ANONYMIZER_SPACY_STRICT=true",
            ]
        ),
        encoding="utf-8",
    )

    loaded_settings = Settings()

    assert loaded_settings.SPACY_MODEL == "custom_de_model"
    assert loaded_settings.SPACY_AUTO_DOWNLOAD is True
    assert loaded_settings.SPACY_STRICT is True


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_settings_reject_invalid_ollama_ocr_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(OLLAMA_OCR_CONFIDENCE=confidence)


def test_example_env_defines_fail_closed_release_phi_detector() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    values = {
        key.strip(): value.strip()
        for raw_line in (repository_root / "example.env")
        .read_text(encoding="utf-8")
        .splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
        for key, separator, value in (raw_line.partition("="),)
        if separator
    }

    assert values["PHI_REGION_DETECTOR_REQUIRED"] == "True"
    assert values["PHI_REGION_DETECTOR_CONFIDENCE"] == "0.05"
    assert values["PHI_REGION_DETECTOR_INPUT_SIZE"] == "960"
    assert values["PHI_REGION_DETECTOR_RESIZE_MODE"] == "letterbox"
    assert values["PHI_REGION_DETECTOR_CLASS_IDS"] == "0"
    assert len(values["PHI_REGION_DETECTOR_MODEL_SHA256"]) == 64
