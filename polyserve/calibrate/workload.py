"""Fixed synthetic workload used for every trial so results are comparable.

Default: 16 prompts x ~256-token prefill x 128-token decode, at concurrency 1 / 4 / 8.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

_WORDS = (
    "system latency throughput memory kernel batch token decode prefill schedule cache page block "
    "tensor matrix vector attention layer head hidden context window request response server client "
    "measure sample power energy joule watt clock frequency thermal budget margin planner search stage "
    "quantize precision weight bias gradient optimizer inference runtime backend adapter profile hash "
    "network socket stream chunk parse encode dictionary index offset stride buffer queue worker thread"
).split()


@dataclass
class Workload:
    n_prompts: int = 16
    prefill_tokens: int = 256
    decode_tokens: int = 128
    concurrencies: Tuple[int, ...] = (1, 4, 8)
    seed: int = 0
    temperature: float = 0.0
    prompts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.prompts:
            self.prompts = self._generate()

    def _generate(self) -> List[str]:
        rng = random.Random(self.seed)
        # ~0.75 tokens per short English word on Llama-style tokenizers, so use 0.8 x tokens words.
        n_words = max(8, int(self.prefill_tokens * 0.8))
        prompts = []
        for i in range(self.n_prompts):
            words = [rng.choice(_WORDS) for _ in range(n_words)]
            prompts.append(f"Prompt {i}: " + " ".join(words) + "\n\nContinue the text:")
        return prompts

    def describe(self) -> str:
        return (
            f"{self.n_prompts} prompts x ~{self.prefill_tokens} prefill x {self.decode_tokens} decode, "
            f"concurrency {'/'.join(str(c) for c in self.concurrencies)}"
        )
