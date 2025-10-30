import shutil
import socket
import subprocess
import time
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


def ensure_ollama(timeout: int = 15):
    """
    Stellt sicher, dass der Ollama-Server läuft.
    Vorgehen:
    1) Versuche `ollama ps` – wenn erfolgreich, ist der Server erreichbar.
    2) Falls nicht erreichbar, starte `ollama serve` und warte bis Port 11434 lauscht.

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

    # 1) Schneller Check via `ollama ps`
    if _ollama_ps_ok():
        return None  # Server ist bereits erreichbar

    # 2) Falls Port bereits lauscht (Race mit extern gestartetem Server)
    if listening():
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
            logger.warning(f"Ollama failed to start within {timeout}s timeout - LLM features will be disabled")
            return None
        time.sleep(0.3)

    return proc  # Handle zurückgeben, damit der Subprozess nicht vorzeitig GC-ed wird
