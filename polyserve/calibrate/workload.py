"""Synthetic workloads. Every trial in a calibration uses the same one so results are comparable.

Presets (`polyserve serve <model> --workload NAME`):

| name             | prefill | decode | concurrency   | TTFT ceiling | shaped like                     |
|------------------|---------|--------|---------------|--------------|---------------------------------|
| default          |   256   |  128   | 1 / 4 / 8     |   500 ms     | the spec's calibration workload |
| chat             |   512   |  128   | 1 / 4 / 8     |   500 ms     | assistant turns                 |
| long-context     |  8192   |  256   | 1 / 2 / 4     |  2000 ms     | document Q&A, summarisation     |
| generation       |   128   | 1024   | 1 / 4 / 8     |   500 ms     | code / story generation         |
| high-concurrency |   256   |   64   | 32 / 64 / 128 |  1000 ms     | many short requests             |
| rag              |  6144   |   64   | 1 / 4 / 8     |  1500 ms     | retrieval-augmented answers     |

When a tokenizer is available, `fit_prompts` resizes each prompt to the target token count
so "512-token prefill" means 512 tokens for that model, not ~512.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

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
CTX_HEADROOM = 64  # tokens of slack for chat templates / special tokens


@dataclass
class Workload:
    n_prompts: int = 16
    prefill_tokens: int = 256
    decode_tokens: int = 128
    concurrencies: Tuple[int, ...] = (1, 4, 8)
    seed: int = 0
    temperature: float = 0.0
    name: str = "default"
    ttft_ceiling_ms: float = 500.0  # default constraint for the `balanced` objective
    prompts: List[str] = field(default_factory=list)
    fitted: bool = False  # prompts were sized with a real tokenizer

    def __post_init__(self) -> None:
        if not self.prompts:
            self.prompts = self._generate()

    @property
    def min_ctx(self) -> int:
        """Smallest per-request context a config must offer to run this workload."""
        return self.prefill_tokens + self.decode_tokens + CTX_HEADROOM

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
            f"{self.name}: {self.n_prompts} prompts x {'' if self.fitted else '~'}{self.prefill_tokens} prefill x "
            f"{self.decode_tokens} decode, concurrency {'/'.join(str(c) for c in self.concurrencies)}, "
            f"TTFT ceiling {self.ttft_ceiling_ms:.0f} ms"
        )

    def spec(self) -> Dict[str, object]:
        """Serialisable description stored in profiles and benchmark results."""
        return {
            "name": self.name,
            "n_prompts": self.n_prompts,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "concurrencies": list(self.concurrencies),
            "ttft_ceiling_ms": self.ttft_ceiling_ms,
            "seed": self.seed,
        }


_PRESETS: Dict[str, Workload] = {
    "default": Workload(name="default"),
    "chat": Workload(name="chat", prefill_tokens=512, decode_tokens=128),
    "long-context": Workload(
        name="long-context", n_prompts=8, prefill_tokens=8192, decode_tokens=256,
        concurrencies=(1, 2, 4), ttft_ceiling_ms=2000.0,
    ),
    "generation": Workload(name="generation", prefill_tokens=128, decode_tokens=1024),
    "high-concurrency": Workload(
        name="high-concurrency", n_prompts=256, prefill_tokens=256, decode_tokens=64,
        concurrencies=(32, 64, 128), ttft_ceiling_ms=1000.0,
    ),
    "rag": Workload(name="rag", prefill_tokens=6144, decode_tokens=64, ttft_ceiling_ms=1500.0),
}

WORKLOAD_NAMES: Tuple[str, ...] = tuple(_PRESETS)


def get_workload(name: str = "default") -> Workload:
    """A fresh copy of a preset (prompts are regenerated, so fitting one never affects another)."""
    if name not in _PRESETS:
        raise ValueError(f"unknown workload {name!r}; choose from {', '.join(WORKLOAD_NAMES)}")
    base = _PRESETS[name]
    return replace(base, prompts=[], fitted=False)


def workload_table() -> List[Dict[str, object]]:
    return [_PRESETS[n].spec() for n in WORKLOAD_NAMES]
