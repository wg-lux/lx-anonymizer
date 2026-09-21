"""Resolve native Tesseract data without changing subprocess configuration."""

import os
from pathlib import Path
from shutil import which


def get_tessdata_path(language: str) -> str:
    """Accept a configured data directory or its parent; otherwise use installation paths.

    The selected directory must contain every requested language. Missing data
    raises before native initialization rather than using an unverified default.
    """
    languages = language.split("+")
    if any(
        not name or Path(name).name != name or name in {".", ".."} for name in languages
    ):
        raise ValueError("Tesseract language must be '+'-separated language names")

    prefix = os.environ.get("TESSDATA_PREFIX")
    candidates: list[Path] = []
    if prefix:
        configured = Path(prefix).expanduser()
        candidates.extend((configured, configured / "tessdata"))
    else:
        executable = which("tesseract")
        if executable:
            candidates.append(
                Path(executable).resolve().parent.parent / "share" / "tessdata"
            )
        candidates.extend(
            Path(path)
            for path in (
                "/run/current-system/sw/share/tessdata",
                "/usr/share/tesseract-ocr/5/tessdata",
                "/usr/share/tesseract-ocr/4.00/tessdata",
                "/usr/share/tessdata",
                "/usr/local/share/tessdata",
            )
        )
    for candidate in candidates:
        if all((candidate / f"{name}.traineddata").is_file() for name in languages):
            return str(candidate)
    if prefix:
        raise FileNotFoundError(
            f"TESSDATA_PREFIX does not contain trained data for {language}: {prefix}"
        )
    raise FileNotFoundError(
        f"No trained data found for {language}; configure TESSDATA_PREFIX"
    )
