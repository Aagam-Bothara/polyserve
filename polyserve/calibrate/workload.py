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
| sharegpt         | ≤1024   | ≤512   | 1 / 8 / 32    |  1000 ms     | real ChatGPT user turns         |
| extract          | ≤1536   | ≤384   | 1 / 4 / 8     |  1500 ms     | quote sentences from an article |
| code-edit        |  ≤768   | ≤512   | 1 / 4 / 8     |   500 ms     | add type hints to a function    |

The last three use real text (polyserve.calibrate.datasets) and let answers stop when the model
does; their prefill and decode figures are caps.

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
MIN_LEVEL_REQUESTS = 8  # fewest requests a level sends when requests_per_slot caps it


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
    # Per-token decode latency ceiling, applied under every objective. Prefill latency is what
    # TTFT constrains; this constrains decode, so neither phase can be sacrificed for the other.
    tpot_ceiling_ms: Optional[float] = 100.0
    # Tokens every prompt starts with (a system prompt, a retrieved document). This is what prefix
    # caching exploits: only the first request should pay to prefill them.
    shared_prefix_tokens: int = 0
    prefix_text: str = ""  # the shared prefix itself; generated from `seed` unless given
    prefix_fixed: bool = False  # the prefix was given (e.g. a warmup reusing it) and must not change
    prompts: List[str] = field(default_factory=list)
    fitted: bool = False  # prompts were sized with a real tokenizer
    # Real text instead of random words (see polyserve.calibrate.datasets). prefill_tokens is then a
    # cap, and with natural_stop answers end when the model stops rather than at decode_tokens.
    source: str = "synthetic"
    natural_stop: bool = False
    sample_offset: int = 0  # where this workload's prompts start in the source's shuffled pool
    # Requests per concurrency slot at each level (0 = every level sends all n_prompts). Real answers
    # run to hundreds of tokens, so sending all 64 ShareGPT prompts one at a time took 5 minutes of
    # an 8-minute trial on an A40; a level at concurrency c needs a few rounds of c, not all of them.
    requests_per_slot: int = 0

    def __post_init__(self) -> None:
        if self.shared_prefix_tokens and not self.prefix_text:
            self.prefix_text = self._prefix(max(4, int(self.shared_prefix_tokens * 0.8)))
        if not self.prompts and self.source == "synthetic":  # real text downloads: only when needed
            self.prompts = self._generate()

    def level_requests(self, concurrency: int) -> int:
        """How many of the prompts a level at this concurrency sends."""
        if not self.requests_per_slot:
            return len(self.prompts)
        return min(len(self.prompts), max(MIN_LEVEL_REQUESTS, concurrency * self.requests_per_slot))

    def ensure_prompts(self) -> "Workload":
        """Generate or download the prompts if they are not there yet."""
        if not self.prompts:
            self.prompts = self._generate()
        return self

    @property
    def _unique_tokens(self) -> int:
        return max(8, self.prefill_tokens - self.shared_prefix_tokens)

    def _prefix(self, n_words: int) -> str:
        rng = random.Random(7_919 * (self.seed + 1))
        return "Context: " + " ".join(rng.choice(_WORDS) for _ in range(n_words)) + "\n\n"

    @property
    def min_ctx(self) -> int:
        """Smallest per-request context a config must offer to run this workload."""
        return self.prefill_tokens + self.decode_tokens + CTX_HEADROOM

    def _words(self, i: int, n_words: int) -> List[str]:
        rng = random.Random(self.seed * 100_003 + i)
        return [rng.choice(_WORDS) for _ in range(n_words)]

    def _generate(self) -> List[str]:
        if self.source != "synthetic":
            from polyserve.calibrate import datasets

            return datasets.sample(self.source, self.n_prompts, self.sample_offset,
                                   self.prefill_tokens * datasets.CHARS_PER_TOKEN)
        # ~1.25 tokens per short English word on Llama/Qwen tokenizers, so start at 0.8 x tokens words.
        n_words = max(8, int(self._unique_tokens * 0.8))
        return [self.prefix_text + _PREFIX.format(i=i) + " ".join(self._words(i, n_words)) + _SUFFIX
                for i in range(self.n_prompts)]

    def fit_prompts(self, counter: TokenCounter, tolerance: int = 2, max_iter: int = 8) -> "Workload":
        """Resize each prompt so the tokenizer counts within `tolerance` of prefill_tokens.

        Real-text prompts keep their own length; only those over the cap are cut, from the body so
        their instruction survives."""
        if not counter.available:
            return self
        if self.source != "synthetic":
            from polyserve.calibrate import datasets

            self.ensure_prompts()
            fitted_real: List[str] = []
            for i, prompt in enumerate(self.prompts):
                n, chars = counter.count(prompt), len(prompt)
                for _ in range(max_iter):
                    if n is None or n <= self.prefill_tokens:
                        break
                    chars = int(chars * self.prefill_tokens / n * 0.95)
                    prompt = datasets.sample(self.source, 1, self.sample_offset + i, chars)[0]
                    n = counter.count(prompt)
                fitted_real.append(prompt)
            self.prompts = fitted_real
            self.fitted = True
            return self
        if self.shared_prefix_tokens and not self.prefix_fixed:
            n_words = max(4, int(self.shared_prefix_tokens * 0.8))
            text = self._prefix(n_words)
            for _ in range(max_iter):
                text = self._prefix(n_words)
                n = counter.count(text)
                if n is None:
                    return self
                if abs(n - self.shared_prefix_tokens) <= tolerance:
                    break
                delta = self.shared_prefix_tokens - n
                step = int(round(delta * n_words / max(n, 1)))
                n_words = max(4, n_words + (step if step != 0 else (1 if delta > 0 else -1)))
            self.prefix_text = text
        fitted: List[str] = []
        for i in range(self.n_prompts):
            n_words = max(4, int(self._unique_tokens * 0.8))
            prompt = ""
            for _ in range(max_iter):
                prompt = self.prefix_text + _PREFIX.format(i=i) + " ".join(self._words(i, n_words)) + _SUFFIX
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
        if self.source != "synthetic":
            return (f"{self.name}: {self.n_prompts} {self.source} prompts of up to {self.prefill_tokens} tokens, "
                    f"answers up to {self.decode_tokens} tokens{' (natural stop)' if self.natural_stop else ''}, "
                    f"concurrency {'/'.join(str(c) for c in self.concurrencies)}, "
                    f"TTFT ceiling {self.ttft_ceiling_ms:.0f} ms"
                    + (f", TPOT ceiling {self.tpot_ceiling_ms:.0f} ms" if self.tpot_ceiling_ms else ""))
        return (
            f"{self.name}: {self.n_prompts} prompts x {'' if self.fitted else '~'}{self.prefill_tokens} prefill"
            f"{f' ({self.shared_prefix_tokens} shared)' if self.shared_prefix_tokens else ''} x "
            f"{self.decode_tokens} decode, concurrency {'/'.join(str(c) for c in self.concurrencies)}, "
            f"TTFT ceiling {self.ttft_ceiling_ms:.0f} ms"
            + (f", TPOT ceiling {self.tpot_ceiling_ms:.0f} ms" if self.tpot_ceiling_ms else "")
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
            "tpot_ceiling_ms": self.tpot_ceiling_ms,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "seed": self.seed,
            "source": self.source,
            "natural_stop": self.natural_stop,
            **({"requests_per_slot": self.requests_per_slot} if self.requests_per_slot else {}),
        }


_PRESETS: Dict[str, Workload] = {
    "default": Workload(name="default"),
    "chat": Workload(name="chat", prefill_tokens=512, decode_tokens=128, tpot_ceiling_ms=50.0),
    "long-context": Workload(
        name="long-context", n_prompts=8, prefill_tokens=8192, decode_tokens=256,
        concurrencies=(1, 2, 4), ttft_ceiling_ms=2000.0,
    ),
    "generation": Workload(name="generation", prefill_tokens=128, decode_tokens=1024, tpot_ceiling_ms=50.0),
    "high-concurrency": Workload(
        name="high-concurrency", n_prompts=256, prefill_tokens=256, decode_tokens=64,
        concurrencies=(32, 64, 128), ttft_ceiling_ms=1000.0, tpot_ceiling_ms=150.0,
    ),
    "rag": Workload(name="rag", prefill_tokens=6144, decode_tokens=64, ttft_ceiling_ms=1500.0),
    # Shared-prefix shapes, where prefix caching decides time to first token.
    "chat-system": Workload(name="chat-system", prefill_tokens=2048, shared_prefix_tokens=1536, decode_tokens=128,
                            tpot_ceiling_ms=50.0),
    "rag-shared": Workload(name="rag-shared", prefill_tokens=6144, shared_prefix_tokens=5632, decode_tokens=64,
                           ttft_ceiling_ms=1500.0),
    # Real text, for strategies that depend on content. Prompt and answer lengths are the data's and
    # the model's own; the figures here are caps.
    "sharegpt": Workload(name="sharegpt", source="sharegpt", natural_stop=True, n_prompts=64, prefill_tokens=1024,
                         decode_tokens=512, concurrencies=(1, 8, 32), ttft_ceiling_ms=1000.0, requests_per_slot=2),
    "extract": Workload(name="extract", source="cnn-extract", natural_stop=True, n_prompts=32, prefill_tokens=1536,
                        decode_tokens=384, ttft_ceiling_ms=1500.0, tpot_ceiling_ms=50.0, requests_per_slot=4),
    "code-edit": Workload(name="code-edit", source="humaneval-edit", natural_stop=True, n_prompts=32,
                          prefill_tokens=768, decode_tokens=512, tpot_ceiling_ms=50.0, requests_per_slot=4),
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
