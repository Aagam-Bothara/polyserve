"""Fixed synthetic workload used for every trial so results are comparable.

Default: 16 prompts x ~256-token prefill x 128-token decode, at concurrency 1 / 4 / 8.
When a tokenizer is available, `fit_prompts` resizes each prompt to the target token count
so "256-token prefill" means 256 tokens for that model, not ~256.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from polyserve.calibrate.tokens import TokenCounter

_WORDS = (
    "system latency throughput memory kernel batch token decode prefill schedule cache page block "
    "tensor matrix vector attention layer head hidden context window request response server client "
    "measure sample power energy joule watt clock frequency thermal budget margin planner search stage "
    "quantize precision weight bias gradient optimizer inference runtime backend adapter profile hash "
    "network socket stream chunk parse encode dictionary index offset stride buffer queue worker thread"
).split()

_PREFIX = "Prompt {i}: "
_SUFFIX = "\n\nContinue the text:"


@dataclass
class Workload:
    n_prompts: int = 16
    prefill_tokens: int = 256
    decode_tokens: int = 128
    concurrencies: Tuple[int, ...] = (1, 4, 8)
    seed: int = 0
    temperature: float = 0.0
    prompts: List[str] = field(default_factory=list)
    fitted: bool = False  # prompts were sized with a real tokenizer

    def __post_init__(self) -> None:
        if not self.prompts:
            self.prompts = self._generate()

    def _words(self, i: int, n_words: int) -> List[str]:
        rng = random.Random(self.seed * 100_003 + i)
        return [rng.choice(_WORDS) for _ in range(n_words)]

    def _generate(self) -> List[str]:
        # ~1.25 tokens per short English word on Llama/Qwen tokenizers, so start at 0.8 x tokens words.
        n_words = max(8, int(self.prefill_tokens * 0.8))
        return [_PREFIX.format(i=i) + " ".join(self._words(i, n_words)) + _SUFFIX for i in range(self.n_prompts)]

    def fit_prompts(self, counter: TokenCounter, tolerance: int = 2, max_iter: int = 8) -> "Workload":
        """Resize each prompt so the tokenizer counts within `tolerance` of prefill_tokens."""
        if not counter.available:
            return self
        fitted: List[str] = []
        for i in range(self.n_prompts):
            n_words = max(4, int(self.prefill_tokens * 0.8))
            prompt = ""
            for _ in range(max_iter):
                prompt = _PREFIX.format(i=i) + " ".join(self._words(i, n_words)) + _SUFFIX
                n = counter.count(prompt)
                if n is None:
                    return self
                if abs(n - self.prefill_tokens) <= tolerance:
                    break
                # Proportional correction, at least one word.
                delta = self.prefill_tokens - n
                step = int(round(delta * n_words / max(n, 1)))
                n_words = max(4, n_words + (step if step != 0 else (1 if delta > 0 else -1)))
            fitted.append(prompt)
        self.prompts = fitted
        self.fitted = True
        return self

    def measured_prompt_tokens(self, counter: TokenCounter) -> Optional[int]:
        if not counter.available or not self.prompts:
            return None
        counts = [counter.count(p) or 0 for p in self.prompts]
        return int(round(sum(counts) / len(counts)))

    def describe(self) -> str:
        return (
            f"{self.n_prompts} prompts x {'' if self.fitted else '~'}{self.prefill_tokens} prefill x "
            f"{self.decode_tokens} decode, concurrency {'/'.join(str(c) for c in self.concurrencies)}"
        )
