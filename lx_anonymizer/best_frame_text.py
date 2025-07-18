from __future__ import annotations
import random
from .utils.ollama import ensure_ollama
from .ollama_llm_processor import OllamaLLMProcessor


class BestFrameText:
    """
    Collects 'good' OCR passages for two purposes:

    1. A quick human‑readable preview (`reduce()`)
    2. *NOT* for metadata extraction – that now happens per‑frame
    """

    def __init__(
        self,
        reservoir_size: int = 500,
        min_conf: float = 0.6,
        min_len: int = 10,
    ) -> None:
        self.N = reservoir_size
        self.min_conf = min_conf
        self.min_len = min_len
        self.i = 0
        self.reservoir: List[str] = []

        ensure_ollama()                         # still needed by other code paths
        self.processor = OllamaLLMProcessor()   # kept for compatibility

    # ---------- online phase -------------------------------------------------

    def _should_keep(self) -> bool:
        return random.random() < self.N / (self.i + 1)

    def push(
        self,
        text: str,
        ocr_conf: float,
        is_sensitive: bool | None = None,
    ) -> None:
        """
        Push one OCR passage.

        * If `is_sensitive` is True we **always** keep it (it likely carries names / MRNs).
        * Otherwise we apply min_conf / min_len + reservoir sampling.
        """
        self.i += 1

        # keep everything flagged sensitive
        if is_sensitive:
            if len(self.reservoir) < self.N:
                self.reservoir.append(text)
            elif self._should_keep():
                self.reservoir[random.randrange(self.N)] = text
            return

        # normal gate
        if ocr_conf < self.min_conf or len(text) < self.min_len:
            return

        if len(self.reservoir) < self.N:
            self.reservoir.append(text)
        elif self._should_keep():
            self.reservoir[random.randrange(self.N)] = text

    # ---------- offline phase ------------------------------------------------

    def reduce(self, preview_size: int = 5) -> dict[str, str]:
        """
        Return a very small JSON‑like dict with a 'best' passage and
        a rough 'average' preview assembled from random samples.
        """
        if not self.reservoir:
            return {"best": "", "average": ""}

        # 'best' – take longest passage (usually the status overlay block)
        best = max(self.reservoir, key=len)

        # 'average' – concatenate a few random samples, truncated for safety
        preview = random.sample(
            self.reservoir, min(preview_size, len(self.reservoir))
        )
        average = "\n\n".join(preview)[:1500]     # limit to ∼1.5 KiB

        return {"best": best, "average": average}