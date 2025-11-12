from __future__ import annotations

import logging
import math
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .ollama_llm_meta_extraction import OllamaOptimizedExtractor

logger = logging.getLogger(__name__)

_WORD_RX = re.compile(r"[A-Za-zÄÖÜäöüß]{2,}")
_MEDICAL_RXS = [
    re.compile(r"(Patient|Untersuchung|Diagnose|Befund|Behandlung)", re.I),
    re.compile(r"\b(mm|cm|grad|°|%)\b", re.I),
    re.compile(r"(links|rechts|lateral|medial|anterior|posterior)", re.I),
    re.compile(r"(Jahr|Jahre|Tag|Tage|Monat|Monate)", re.I),
]


def _is_punct_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat.startswith("P") or cat.startswith("S")


@dataclass
class Candidate:
    text: str
    conf: float
    score: float
    meta: Dict[str, Any]


class BestFrameText:
    """
    Hybrid of top-K scoring and reservoir sampling:

    - score(text, conf) -> float in [0,1]
    - keep a uniform reservoir for diversity (size N)
    - keep a deterministic top-K list (size K) for 'best' picks
    - always accept sensitive items
    - provide reduce() for human preview (best + average)
    """

    def __init__(
        self,
        reservoir_size: int = 500,
        topk: int = 5,
        min_conf: float = 0.6,
        min_len: int = 10,
        seed: Optional[int] = None,
        enable_quality_mode: bool = True,
    ) -> None:
        self.N = max(1, reservoir_size)
        self.K = max(1, topk)
        self.min_conf = min_conf
        self.min_len = min_len
        self.enable_quality_mode = enable_quality_mode or bool(os.getenv("OCR_FIX_V1") == "1")

        self._i = 0
        self._reservoir: List[Candidate] = []
        self._topk: List[Candidate] = []
        if seed is not None:
            random.seed(seed)

        # FIX: initialize extractor instance (was empty assignment causing syntax error)
        self.ollama_extraction = OllamaOptimizedExtractor()

    # ---------- scoring ------------------------------------------------------

    def _quality_score(self, text: str, conf: float) -> float:
        if not text:
            return 0.0

        # Normalize confidence (defensive)
        conf = 0.0 if math.isnan(conf) else max(0.0, min(1.0, conf))
        score = conf * 0.4

        words = _WORD_RX.findall(text)
        if words:
            # medical pattern bonus (capped)
            med_bonus = sum(0.1 for rx in _MEDICAL_RXS if rx.search(text))
            score += min(med_bonus, 0.3)

            # average word length bonus
            awl = sum(map(len, words)) / len(words)
            if awl >= 4.0:
                score += 0.1

            # readable ratio (>=3 letters)
            rw = [w for w in words if len(w) >= 3 and not all(c == w[0] for c in w)]
            ratio = len(rw) / len(words) if words else 0.0
            score += ratio * 0.2

        # punctuation/symbol penalty (robust Unicode)
        if len(text) >= 4:
            punct = sum(1 for ch in text if _is_punct_or_symbol(ch))
            pr = punct / len(text)
            if pr > 0.5:
                score *= 0.3
            elif pr > 0.3:
                score *= 0.7

        # mild length bonus (cap)
        score += min(len(text) / 1000.0, 0.1)

        return max(0.0, min(1.0, score))

    # ---------- accept / store ----------------------------------------------

    def _maybe_add_reservoir(self, cand: Candidate) -> None:
        self._i += 1
        if len(self._reservoir) < self.N:
            self._reservoir.append(cand)
            return
        # classic reservoir probability
        if random.random() < self.N / self._i:
            idx = random.randrange(self.N)
            self._reservoir[idx] = cand

    def _update_topk(self, cand: Candidate) -> None:
        self._topk.append(cand)
        self._topk.sort(key=lambda c: c.score, reverse=True)
        if len(self._topk) > self.K:
            self._topk = self._topk[: self.K]

    def push(self, text: str, ocr_conf: float, *, is_sensitive: Optional[bool] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not text:
            return

        if not self.enable_quality_mode:
            # simple length-biased fallback with conf gate
            if ocr_conf < self.min_conf or len(text) < self.min_len:
                return
            s = min(1.0, (len(text) / 2000.0) + max(0.0, min(1.0, ocr_conf)) * 0.25)
        else:
            s = self._quality_score(text, ocr_conf)
            # score gate for non-sensitive
            if not is_sensitive and (ocr_conf < self.min_conf or len(text) < self.min_len or s < 0.25):
                return

        cand = Candidate(text=text, conf=ocr_conf, score=s, meta=metadata or {})

        # sensitive always eligible, bypass min gates already done; still sampled for diversity
        self._maybe_add_reservoir(cand)
        self._update_topk(cand)

    # ---------- output -------------------------------------------------------

    def best_text(self) -> str:
        return self._topk[0].text if self._topk else ""

    def topk_candidates(self) -> List[Candidate]:
        return list(self._topk)

    def reduce(self, preview_size: int = 50) -> Dict[str, str]:
        """
        Human-friendly preview:
          - 'best': highest score (not just longest)
          - 'average': join a few random reservoir samples (diverse), clipped
        """

        # FIX: ensure local accumulator exists; attempt LLM summary if reservoir has metadata
        try:
            text = ""
            for c in self._reservoir:
                text += c.meta.get("extracted_metadata", "")
            if text:
                meta = self.ollama_extraction.extract_metadata(text)
                if isinstance(meta, dict):
                    best = meta.get("representative_text") or meta.get("best") or ""
                else:
                    best = str(meta)
                average = "\n\n".join(c.text for c in random.sample(self._reservoir, min(preview_size, len(self._reservoir))))[:1500]
                return {"best": best, "average": average}
        except Exception as e:
            logger.info(f"Ollama extraction failed: {e}, fallback to original reduce.")

        if not self._reservoir and not self._topk:
            return {"best": "", "average": ""}

        best = self._topk[0].text if self._topk else max(self._reservoir, key=lambda c: len(c.text)).text

        bag = self._reservoir if self._reservoir else self._topk
        samples = random.sample(bag, min(preview_size, len(bag)))
        average = "\n\n".join(c.text for c in samples)[:1500]
        return {"best": best, "average": average}

    def stats(self) -> Dict[str, Any]:
        if not (self._reservoir or self._topk):
            return {"n": 0}
        allc = self._reservoir + self._topk
        scores = [c.score for c in allc]
        return {
            "n_reservoir": len(self._reservoir),
            "n_topk": len(self._topk),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "quality_mode": self.enable_quality_mode,
        }
