from pathlib import Path

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
