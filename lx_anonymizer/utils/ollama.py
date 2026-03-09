import shutil
import socket
import subprocess
import time
import os
from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired


def _ollama_ps_ok(timeout: float = 5.0) -> bool:
    """
    Schnelltest: ruft `ollama ps` auf. Liefert True, wenn der Befehl erfolgreich ist,
    andernfalls False (z. B. wenn der Server nicht läuft).
    """
    if shutil.which("ollama") is None:
        return False
    try:
        cp = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        # Erfolgssignal ist Returncode 0
        return cp.returncode == 0
    except (CalledProcessError, TimeoutExpired):
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gpu_available() -> bool:
    """
    Lightweight NVIDIA GPU presence check for Ollama auto-pull gating.
    """
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        cp = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return cp.returncode == 0 and bool((cp.stdout or "").strip())
    except Exception:
        return False


def _models_storage_root() -> Path:
    models_env = os.getenv("OLLAMA_MODELS")
    if models_env:
        return Path(models_env).expanduser()
    return Path.home() / ".ollama" / "models"


def _free_storage_gb(path: Path) -> float:
    """
    Return free storage in GiB on the filesystem hosting `path`.
    """
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return usage.free / (1024**3)


def _ollama_has_model(model: str, timeout: float = 5.0) -> bool:
    if not model or shutil.which("ollama") is None:
        return False
    try:
        cp = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if cp.returncode != 0:
            return False
        lines = (cp.stdout or "").splitlines()
        return any(line.strip().startswith(model) for line in lines)
    except Exception:
        return False


def _maybe_pull_model_if_capable(
    model: str | None,
    min_free_gb: float,
    pull_timeout: int = 900,
) -> None:
    """
    Try to pull a model only if GPU and sufficient storage are available.
    No exceptions are raised; failures are logged and ignored.
    """
    import logging

    logger = logging.getLogger(__name__)
    if not model:
        return

    if _ollama_has_model(model):
        return

    if not _gpu_available():
        logger.info("Skipping Ollama auto-pull for %s: no NVIDIA GPU detected", model)
        return

    free_gb = _free_storage_gb(_models_storage_root())
    if free_gb < min_free_gb:
        logger.info(
            "Skipping Ollama auto-pull for %s: insufficient free storage (%.1f GiB < %.1f GiB)",
            model,
            free_gb,
            min_free_gb,
        )
        return

    logger.info(
        "Attempting Ollama auto-pull for %s (GPU detected, free storage %.1f GiB)",
        model,
        free_gb,
    )
    try:
        cp = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=pull_timeout,
            check=False,
        )
        if cp.returncode == 0:
            logger.info("Ollama auto-pull succeeded for model %s", model)
        else:
            logger.warning(
                "Ollama auto-pull failed for %s (rc=%s): %s",
                model,
                cp.returncode,
                (cp.stderr or cp.stdout or "").strip()[:300],
            )
    except TimeoutExpired:
        logger.warning(
            "Ollama auto-pull timed out for %s after %ss", model, pull_timeout
        )
    except Exception as e:
        logger.warning("Ollama auto-pull error for %s: %s", model, e)


def ensure_ollama(
    timeout: int = 15,
    auto_pull_if_capable: bool | None = None,
    auto_pull_model: str | None = None,
    auto_pull_min_free_gb: float | None = None,
):
    """
    Stellt sicher, dass der Ollama-Server läuft.
    Vorgehen:
    1) Versuche `ollama ps` – wenn erfolgreich, ist der Server erreichbar.
    2) Falls nicht erreichbar, starte `ollama serve` und warte bis Port 11434 lauscht.

    Optional auto-pull behavior (opt-in):
    - If enabled, tries `ollama pull <model>` after the server is reachable
    - Only when a NVIDIA GPU is detected and enough free storage is available

    Environment variables (used when corresponding args are None):
        LX_OLLAMA_AUTO_PULL_IF_CAPABLE=0/1
        LX_OLLAMA_AUTO_PULL_MODEL=<model-name>
        LX_OLLAMA_AUTO_PULL_MIN_FREE_GB=<float>

    Returns:
        subprocess.Popen Handle, wenn `ollama serve` gestartet wurde, sonst None.

    Raises:
        RuntimeError: Nur wenn Ollama gestartet werden sollte aber innerhalb des Timeouts nicht erreichbar wurde.
    """

    def listening(port: int = 11434) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                return False

    if shutil.which("ollama") is None:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("ollama CLI not found in PATH - LLM features will be disabled")
        return None

    if auto_pull_if_capable is None:
        auto_pull_if_capable = _env_flag("LX_OLLAMA_AUTO_PULL_IF_CAPABLE", False)
    if auto_pull_model is None:
        auto_pull_model = os.getenv("LX_OLLAMA_AUTO_PULL_MODEL")
    if auto_pull_min_free_gb is None:
        try:
            auto_pull_min_free_gb = float(
                os.getenv("LX_OLLAMA_AUTO_PULL_MIN_FREE_GB", "10")
            )
        except ValueError:
            auto_pull_min_free_gb = 10.0

    # 1) Schneller Check via `ollama ps`
    if _ollama_ps_ok():
        if auto_pull_if_capable:
            _maybe_pull_model_if_capable(
                auto_pull_model, min_free_gb=float(auto_pull_min_free_gb)
            )
        return None  # Server ist bereits erreichbar

    # 2) Falls Port bereits lauscht (Race mit extern gestartetem Server)
    if listening():
        if auto_pull_if_capable:
            _maybe_pull_model_if_capable(
                auto_pull_model, min_free_gb=float(auto_pull_min_free_gb)
            )
        return None

    # 3) Server starten
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Warten bis der Server erreichbar ist (Port- oder `ps`-Check)
    start = time.time()
    while True:
        if listening() or _ollama_ps_ok(timeout=2.0):
            break
        if time.time() - start > timeout:
            proc.terminate()
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Ollama failed to start within {timeout}s timeout - LLM features will be disabled"
            )
            return None
        time.sleep(0.3)

    if auto_pull_if_capable:
        _maybe_pull_model_if_capable(
            auto_pull_model, min_free_gb=float(auto_pull_min_free_gb)
        )

    return proc  # Handle zurückgeben, damit der Subprozess nicht vorzeitig GC-ed wird
