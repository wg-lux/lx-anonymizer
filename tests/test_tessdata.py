import os
from pathlib import Path

import pytest

from lx_anonymizer.ocr.tessdata import get_tessdata_path


@pytest.mark.parametrize("use_parent", [False, True])
def test_configured_tessdata_is_shared_without_environment_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_parent: bool
) -> None:
    data = tmp_path / "tessdata"
    data.mkdir()
    for language in ("deu", "eng"):
        (data / f"{language}.traineddata").touch()
    prefix = str(tmp_path if use_parent else data)
    monkeypatch.setenv("TESSDATA_PREFIX", prefix)
    assert get_tessdata_path("deu+eng") == str(data)
    assert os.environ["TESSDATA_PREFIX"] == prefix


def test_configured_tessdata_requires_every_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "eng.traineddata").touch()
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="deu\\+eng"):
        get_tessdata_path("deu+eng")
    assert os.environ["TESSDATA_PREFIX"] == str(tmp_path)


@pytest.mark.parametrize("language", ["", "eng+", "../eng", "/eng", "."])
def test_invalid_language_rejected(language: str) -> None:
    with pytest.raises(ValueError, match="language"):
        get_tessdata_path(language)


def test_tessdata_follows_selected_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "bin" / "tesseract"
    binary.parent.mkdir()
    binary.touch(mode=0o755)
    data = tmp_path / "share" / "tessdata"
    data.mkdir(parents=True)
    (data / "eng.traineddata").touch()
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    monkeypatch.setenv("PATH", str(binary.parent))
    assert get_tessdata_path("eng") == str(data)
    assert "TESSDATA_PREFIX" not in os.environ
