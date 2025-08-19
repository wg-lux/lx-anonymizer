import os
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("env_setup")


def _parse_dotenv(content: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        env[key] = val
    return env


def _dump_dotenv(env: Dict[str, str]) -> str:
    # Sort for stable output
    lines = [f'{k}={env[k]}' for k in sorted(env.keys())]
    return "\n".join(lines) + "\n"


def main() -> None:
    # Projektroot: /home/admin/dev/lx-annotate
    base_dir = Path(__file__).resolve().parents[2]
    env_path = base_dir / ".env"

    logger.info(f"Project root: {base_dir}")
    logger.info(f"Target .env:   {env_path}")

    # Bestehende .env laden (optional)
    existing: Dict[str, str] = {}
    if env_path.exists():
        existing = _parse_dotenv(env_path.read_text(encoding="utf-8"))
        logger.info(f"Existing .env found with {len(existing)} entries.")

    # Defaults bestimmen (können per Umgebungsvariablen überschrieben sein)
    home = Path.home()

    # Hugging Face: einheitlicher Cache-Ort, um Doppel-Cache zu vermeiden
    hf_home = Path(os.environ.get("HF_HOME", existing.get("HF_HOME", str(home / ".cache" / "huggingface"))))
    hf_hub_cache = Path(os.environ.get("HF_HUB_CACHE", existing.get("HF_HUB_CACHE", str(hf_home / "hub"))))
    # Wichtig: TRANSFORMERS_CACHE auf den gleichen Ort legen wie der hub-cache
    transformers_cache = Path(os.environ.get("TRANSFORMERS_CACHE", existing.get("TRANSFORMERS_CACHE", str(hf_hub_cache))))
    # Optionaler Download-Beschleuniger
    hf_transfer = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", existing.get("HF_HUB_ENABLE_HF_TRANSFER", "1"))

    # Ollama-Modelstore (kann auf größeres Volume zeigen)
    ollama_models = Path(os.environ.get("OLLAMA_MODELS", existing.get("OLLAMA_MODELS", str(home / ".ollama" / "models"))))
    ollama_keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", existing.get("OLLAMA_KEEP_ALIVE", "4h"))

    # Verzeichnisse sicherstellen
    for p in [hf_home, hf_hub_cache, transformers_cache, ollama_models]:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory: {p} ({e})")

    # Merge: bestehende beibehalten, gezielte Keys überschreiben/setzen
    merged = dict(existing)
    merged.update({
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub_cache),
        "TRANSFORMERS_CACHE": str(transformers_cache),
        "HF_HUB_ENABLE_HF_TRANSFER": str(hf_transfer),
        "OLLAMA_MODELS": str(ollama_models),
        "OLLAMA_KEEP_ALIVE": str(ollama_keep_alive),
    })

    # .env schreiben
    env_path.write_text(_dump_dotenv(merged), encoding="utf-8")

    logger.info("Environment variables set:")
    logger.info(f"  HF_HOME={merged['HF_HOME']}")
    logger.info(f"  HF_HUB_CACHE={merged['HF_HUB_CACHE']}")
    logger.info(f"  TRANSFORMERS_CACHE={merged['TRANSFORMERS_CACHE']}")
    logger.info(f"  HF_HUB_ENABLE_HF_TRANSFER={merged['HF_HUB_ENABLE_HF_TRANSFER']}")
    logger.info(f"  OLLAMA_MODELS={merged['OLLAMA_MODELS']}")
    logger.info(f"  OLLAMA_KEEP_ALIVE={merged['OLLAMA_KEEP_ALIVE']}")
    logger.info(f"Done. Please restart the service for the .env changes to take effect.")


if __name__ == "__main__":
    main()