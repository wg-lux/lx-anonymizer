"""Shared configuration and transport boundary for clinical LLM requests."""

from ipaddress import ip_address
from pathlib import Path
from typing import TypedDict
from urllib.parse import urlsplit

from lx_anonymizer.config import settings


class RequestOptions(TypedDict):
    verify: bool | str
    cert: tuple[str, str] | None
    allow_redirects: bool
    proxies: dict[str, str]


def resolve_connection(provider: str | None, base_url: str | None) -> tuple[str, str]:
    protocol = (settings.LLM_PROVIDER if provider is None else provider).strip().lower()
    if protocol not in {"ollama", "vllm"}:
        raise ValueError("LLM_PROVIDER must be ollama or vllm")
    url = (
        (settings.llm_base_url_for(protocol) if base_url is None else base_url)
        .strip()
        .rstrip("/")
    )
    if not url:
        url = (
            "http://127.0.0.1:11434"
            if protocol == "ollama"
            else "http://127.0.0.1:8000"
        )
    if protocol == "vllm" and url.endswith("/v1"):
        url = url[:-3]
    return protocol, url


def request_options(base_url: str) -> RequestOptions:
    url = urlsplit(base_url)
    if not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError(
            "LLM_BASE_URL must be a server URL without credentials, query or fragment"
        )
    if url.port == 0:
        raise ValueError("LLM server port must be between 1 and 65535")
    try:
        loopback = ip_address(url.hostname).is_loopback
    except ValueError:
        loopback = False
    ca = settings.LLM_CA_FILE.strip()
    cert = settings.LLM_CLIENT_CERT_FILE.strip()
    key = settings.LLM_CLIENT_KEY_FILE.strip()
    if url.scheme == "http" and loopback:
        if ca or cert or key:
            raise ValueError("LLM TLS files require HTTPS")
    elif url.scheme == "https":
        if not all((ca, cert, key)):
            raise ValueError(
                "Remote LLM HTTPS requires CA, client certificate and key files"
            )
        for filename in (ca, cert, key):
            if not Path(filename).is_file():
                raise ValueError("An LLM TLS file is missing")
        if Path(key).stat().st_mode & 0o077:
            raise ValueError("LLM client key must be private to its owner")
    else:
        raise ValueError(
            "LLM HTTP is restricted to literal loopback addresses; use HTTPS with mTLS remotely"
        )
    return {
        "verify": ca or True,
        "cert": (cert, key) if cert else None,
        "allow_redirects": False,
        "proxies": {"http": "", "https": ""},
    }
