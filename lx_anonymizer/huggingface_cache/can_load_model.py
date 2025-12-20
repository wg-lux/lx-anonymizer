import os
import shutil
from pathlib import Path

from lx_anonymizer.setup.custom_logger import get_logger

logger = get_logger(__name__)


class HF_Cache:
    def __init__(
        self, estimated_model_bytes: int = 14 * 1024**3, safety_factor: float = 1.5
    ) -> None:
        self.ESTIMATED_MODEL_BYTES = (
            estimated_model_bytes  # e.g. 14 GiB for MiniCPM-o 2.6
        )
        self.SAFETY_FACTOR = safety_factor  # keep some headroom

    def _can_load_model(self) -> bool:
        """
        Check if we have enough free space to safely download / load the HF model.
        """
        base = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        )
        usage = shutil.disk_usage(base)
        free_bytes = usage.free
        required = int(self.ESTIMATED_MODEL_BYTES * self.SAFETY_FACTOR)

        logger.info(
            "HF cache filesystem free=%d GiB, required≈%d GiB",
            free_bytes // (1024**3),
            required // (1024**3),
        )

        if free_bytes < required:
            logger.warning(
                "Not enough free disk space for HF model "
                "(free=%d GiB, required≈%d GiB). Disabling MiniCPM.",
                free_bytes // (1024**3),
                required // (1024**3),
            )
            return False
        return True

    @staticmethod
    def _dir_size_bytes(path: Path) -> int:
        """Approximate directory size in bytes."""
        if path.is_file():
            return path.stat().st_size
        if not path.exists():
            return 0
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
        return total

    def log_hf_cache_info(self) -> None:
        base = Path(
            os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        )
        hub = Path(os.environ.get("HF_HUB_CACHE", str(base / "hub")))
        tfc = Path(os.environ.get("HF_HOME", str(base)))
        candidates = [
            hub / "models--openbmb--MiniCPM-o-2_6",
            base / "models--openbmb--MiniCPM-o-2_6",
            tfc / "models--openbmb--MiniCPM-o-2_6",
        ]
        logger.info(f"HF_HOME={base} HF_HUB_CACHE={hub} HF_HOME={tfc}")
        for p in candidates:
            size = self._dir_size_bytes(p)
            exists = p.exists()
            logger.info(
                "HF cache candidate: %s exists=%s size_bytes=%s",
                p,
                exists,
                size,
            )
            if exists:
                usage = shutil.disk_usage(p)
                logger.info(
                    "Filesystem for %s → total=%d GiB, used=%d GiB, free=%d GiB",
                    p,
                    usage.total // (1024**3),
                    usage.used // (1024**3),
                    usage.free // (1024**3),
                )
