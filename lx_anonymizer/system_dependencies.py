"""Lightweight deployment checks; no model imports, downloads or service starts."""

import argparse
import importlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from shutil import which
from typing import Literal

from lx_anonymizer.ocr.tessdata import get_tessdata_path


@dataclass(frozen=True)
class DependencyCheck:
    name: str
    status: Literal["ok", "error", "skipped"]
    detail: str


def check_dependencies(
    language: str = "deu+eng", *, video: bool = False, llm: bool = False
) -> list[DependencyCheck]:
    checks: list[DependencyCheck] = []
    for command in ("tesseract", "ffmpeg", "ffprobe") if video else ("tesseract",):
        executable = which(command)
        checks.append(
            DependencyCheck(
                command,
                "ok" if executable else "error",
                executable or f"Install {command} and expose it on the worker's PATH.",
            )
        )
    try:
        data = get_tessdata_path(language)
        checks.append(DependencyCheck("tessdata", "ok", data))
    except (OSError, ValueError) as exc:
        checks.append(DependencyCheck("tessdata", "error", str(exc)))
    # Import checks catch missing shared libraries as well as absent Python wheels.
    for module in ("tesserocr", "cv2", "pymupdf"):
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            checks.append(DependencyCheck(module, "error", str(exc)))
        else:
            checks.append(
                DependencyCheck(module, "ok", "Native module imports successfully")
            )
    if llm:
        checks.append(check_llm())
    else:
        checks.append(
            DependencyCheck("llm", "skipped", "Use --llm to probe the configured model")
        )
    return checks


def check_llm() -> DependencyCheck:
    import requests
    from pydantic import BaseModel, ValidationError

    from lx_anonymizer.config import settings
    from lx_anonymizer.llm.connection import request_options, resolve_connection

    class OllamaModel(BaseModel):
        name: str

    class OllamaModels(BaseModel):
        models: list[OllamaModel]

    class OpenAIModel(BaseModel):
        id: str

    class OpenAIModels(BaseModel):
        data: list[OpenAIModel]

    if not settings.LLM_ENABLED:
        return DependencyCheck(
            "llm", "skipped", "LLM_ENABLED is false; no request sent"
        )
    try:
        provider, base_url = resolve_connection(None, None)
        endpoint = "/api/tags" if provider == "ollama" else "/v1/models"
        response = requests.get(
            base_url + endpoint, timeout=5, **request_options(base_url)
        )
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Model discovery must return HTTP 200 without redirects")
        if provider == "ollama":
            names = [
                model.name
                for model in OllamaModels.model_validate_json(response.content).models
            ]
        else:
            names = [
                model.id
                for model in OpenAIModels.model_validate_json(response.content).data
            ]
        model = settings.LLM_MODEL.strip()
        available = model in names or (
            provider == "ollama" and ":" not in model and f"{model}:latest" in names
        )
        if not model or not available:
            return DependencyCheck(
                "llm", "error", "Configured LLM_MODEL is not installed on the server"
            )
    except (requests.RequestException, ValueError, ValidationError, OSError) as exc:
        # Requests exceptions can contain configured credentials/URLs; avoid echoing them.
        return DependencyCheck(
            "llm",
            "error",
            f"Model discovery failed ({type(exc).__name__}); check endpoint, TLS files and service availability",
        )
    return DependencyCheck(
        "llm", "ok", "Configured model is listed; inference capability is not tested"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="deu+eng")
    parser.add_argument(
        "--video", action="store_true", help="Require FFmpeg and FFprobe on PATH"
    )
    parser.add_argument(
        "--llm", action="store_true", help="Probe model discovery (no inference)"
    )
    args = parser.parse_args(argv)
    checks = check_dependencies(args.language, video=args.video, llm=args.llm)
    print(json.dumps([asdict(check) for check in checks], indent=2))
    return int(any(check.status == "error" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
