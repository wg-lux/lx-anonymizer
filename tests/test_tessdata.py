import os
from pathlib import Path

import pytest

from lx_anonymizer.ocr import tessdata
from lx_anonymizer.ocr.tessdata import get_tessdata_path


@pytest.mark.parametrize("prefix", [None, "missing", "other"])
def test_nix_package_data_takes_precedence_without_environment_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prefix: str | None,
) -> None:
    store_data = tmp_path / "nix-store-data"
    store_data.mkdir()
    for language in ("deu", "eng"):
        (store_data / f"{language}.traineddata").touch()
    packaged = tmp_path / "package-tessdata"
    packaged.symlink_to(store_data, target_is_directory=True)
    monkeypatch.setattr(tessdata, "_PACKAGED_TESSDATA", packaged)
    if prefix is None:
        monkeypatch.delenv("TESSDATA_PREFIX", raising=False)
    else:
        configured = tmp_path / prefix
        if prefix == "other":
            configured.mkdir()
            for language in ("deu", "eng"):
                (configured / f"{language}.traineddata").touch()
        monkeypatch.setenv("TESSDATA_PREFIX", str(configured))
    original_prefix = os.environ.get("TESSDATA_PREFIX")

    assert get_tessdata_path("deu+eng") == str(packaged)
    assert os.environ.get("TESSDATA_PREFIX") == original_prefix


@pytest.mark.parametrize("broken_link", [False, True])
def test_invalid_nix_package_data_does_not_use_host_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    broken_link: bool,
) -> None:
    packaged = tmp_path / "package-tessdata"
    if broken_link:
        packaged.symlink_to(tmp_path / "missing-store-data", target_is_directory=True)
    else:
        packaged.mkdir()
        (packaged / "eng.traineddata").touch()
    host_data = tmp_path / "host-data"
    host_data.mkdir()
    for language in ("deu", "eng"):
        (host_data / f"{language}.traineddata").touch()
    monkeypatch.setattr(tessdata, "_PACKAGED_TESSDATA", packaged)
    monkeypatch.setenv("TESSDATA_PREFIX", str(host_data))

    with pytest.raises(FileNotFoundError, match="Packaged Tesseract data"):
        get_tessdata_path("deu+eng")


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
