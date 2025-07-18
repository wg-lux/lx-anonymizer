import subprocess, time, socket, shutil

def ensure_ollama(timeout: int = 15):
    """
    Start `ollama serve` if it's not already listening on localhost:11434.
    Returns a subprocess.Popen handle or None if the server was already up.
    """
    def listening(port: int = 11434) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                return False

    if listening():
        return None                        # already running

    if shutil.which("ollama") is None:
        raise RuntimeError("ollama CLI not found in PATH")

    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # Wait until the port comes up (or give up)
    start = time.time()
    while not listening():
        if time.time() - start > timeout:
            proc.terminate()
            raise RuntimeError("Ollama failed to start within timeout")
        time.sleep(0.3)

    return proc           # keep the handle alive so the subprocess isn’t GC-ed
