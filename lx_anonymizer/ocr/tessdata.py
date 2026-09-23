"""Resolve native Tesseract data without changing subprocess configuration."""

import os
import re
from pathlib import Path
from shutil import which

_PACKAGED_TESSDATA = Path(__file__).resolve().parents[1] / "resources" / "tessdata"


def get_tessdata_path(language: str) -> str:
    """Use Nix package data, then configured or conventional installation paths.

    The selected directory must contain every requested language. Missing data
    raises before native initialization rather than using an unverified default.
    """
    languages = language.split("+")
    if any(
        re.fullmatch(r"[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)?", name) is None
        for name in languages
    ):
        raise ValueError("Tesseract language must be '+'-separated language names")

    prefix = os.environ.get("TESSDATA_PREFIX")
    candidates: list[Path] = []
    packaged = _PACKAGED_TESSDATA.exists() or _PACKAGED_TESSDATA.is_symlink()
    if packaged:
        # package.nix pins these data to the same Tesseract derivation as the CLI.
        # An unrelated host prefix must not replace that reproducible dependency.
        candidates.append(_PACKAGED_TESSDATA)
    elif prefix:
        configured = Path(prefix).expanduser()
        candidates.extend((configured, configured / "tessdata"))
    else:
        executable = which("tesseract")
        if executable:
            # Inspect both the PATH installation and its resolved target: Homebrew,
            # Conda and Nix commonly expose the executable through symlinks.
            for binary in (Path(executable), Path(executable).resolve()):
                candidates.extend(
                    (
                        binary.parent / "tessdata",  # Windows distribution
                        binary.parent.parent / "share" / "tessdata",
                        binary.parent.parent
                        / "share"
                        / "tesseract-ocr"
                        / "5"
                        / "tessdata",
                        binary.parent.parent
                        / "share"
                        / "tesseract-ocr"
                        / "4.00"
                        / "tessdata",
                    )
                )
        candidates.extend(
            Path(path)
            for path in (
                "/run/current-system/sw/share/tessdata",
                "/usr/share/tesseract-ocr/5/tessdata",
                "/usr/share/tesseract-ocr/4.00/tessdata",
                "/usr/share/tessdata",
                "/usr/local/share/tessdata",
                "/opt/homebrew/share/tessdata",
                "/opt/local/share/tessdata",
            )
        )
    for candidate in candidates:
        if all((candidate / f"{name}.traineddata").is_file() for name in languages):
            return str(candidate)
    if packaged:
        raise FileNotFoundError(
            f"Packaged Tesseract data does not contain trained data for {language}: "
            f"{_PACKAGED_TESSDATA}"
        )
    if prefix:
        raise FileNotFoundError(
            f"TESSDATA_PREFIX does not contain trained data for {language}: {prefix}"
        )
    raise FileNotFoundError(
        f"No trained data found for {language}; configure TESSDATA_PREFIX. "
        f"Searched: {', '.join(str(path) for path in candidates)}"
    )
