from __future__ import annotations
import random, heapq, json, itertools
from .utils.ollama import ensure_ollama
from .ollama_llm_processor import OllamaLLMProcessor


class BestFrameText:
    def __init__(self, reservoir_size: int = 200):
        self.N = reservoir_size
        self.i = 0
        self.reservoir: list[str] = []
        ensure_ollama()
        self.processor = OllamaLLMProcessor()


    def _should_keep(self) -> bool:
        return random.random() < self.N / (self.i + 1)

    def push(self, text: str, ocr_conf: float):
        """Call once per frame."""
        self.i += 1
        if ocr_conf < 0.8 or len(text) < 30:
            return                     # quick-reject noisy short snippets

        if len(self.reservoir) < self.N:
            self.reservoir.append(text)
        elif self._should_keep():
            self.reservoir[random.randrange(self.N)] = text

    # ----- offline phase -----

    def _score(self, text: str) -> float:
        resp = self.processor.call_llm(
            "Act as an OCR expert. Give me a single float in [0,1] that reflects how clean and informative this passage is:",
            text)
        return float(resp.strip())

    def reduce(self, top_k: int = 20) -> dict[str, str]:
        # Phase 1: score
        scored = [(self._score(t), t) for t in self.reservoir]
        top_texts = [t for _, t in heapq.nlargest(top_k, scored)]

        # Phase 2: ask LLM for best + average
        prompt = (
            "Below are OCR passages separated by #. "
            "Return JSON: {\"best\": \"...\", \"average\": \"...\"}\n"
            + "#".join(top_texts)
        )
        return json.loads(self.processor.call_llm(prompt))

