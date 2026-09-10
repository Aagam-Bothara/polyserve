"""Exact token counting for calibration.

Streaming chunks are not tokens: vLLM may emit several tokens per chunk under load and
llama.cpp may split a multi-byte character across chunks. Every trial therefore counts
output tokens from, in order of preference:

  1. the server's own `usage.completion_tokens` (requested via stream_options.include_usage),
  2. the model's tokenizer applied to the concatenated generated text,
  3. the number of streamed content chunks, flagged as approximate.

The same tokenizer sizes the synthetic prompts to their target length.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

SOURCES = ("usage", "tokenizer", "chunks")


class TokenCounter:
    """Counts tokens with a Hugging Face tokenizer; `available` is False if none could be loaded."""

    _cache: Dict[str, "TokenCounter"] = {}

    def __init__(self, encode: Optional[Callable[[str], int]] = None, name: str = "none"):
        self._encode = encode
        self.name = name

    @property
    def available(self) -> bool:
        return self._encode is not None

    def count(self, text: str) -> Optional[int]:
        if self._encode is None or not text:
            return None if self._encode is None else 0
        try:
            return int(self._encode(text))
        except Exception as exc:  # pragma: no cover
            logger.debug("tokenizer failed: %s", exc)
            return None

    @classmethod
    def for_model(cls, hf_id: Optional[str], token: Optional[str] = None) -> "TokenCounter":
        if not hf_id:
            return cls()
        if hf_id in cls._cache:
            return cls._cache[hf_id]
        counter = cls._load(hf_id, token)
        cls._cache[hf_id] = counter
        return counter

    @classmethod
    def _load(cls, hf_id: str, token: Optional[str]) -> "TokenCounter":
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(hf_id, token=token)
            return cls(lambda s: len(tok.encode(s, add_special_tokens=False)), name=f"transformers:{hf_id}")
        except Exception as exc:
            logger.debug("transformers tokenizer unavailable for %s: %s", hf_id, exc)
        try:
            from tokenizers import Tokenizer  # type: ignore

            tok = Tokenizer.from_pretrained(hf_id)
            return cls(lambda s: len(tok.encode(s, add_special_tokens=False).ids), name=f"tokenizers:{hf_id}")
        except Exception as exc:
            logger.debug("tokenizers unavailable for %s: %s", hf_id, exc)
        logger.warning("no tokenizer for %s; output tokens will be counted from stream chunks (approximate)", hf_id)
        return cls()


def worst_source(sources) -> str:
    """The least trustworthy source among a set, so a trial is labelled by its weakest count."""
    seen = set(sources)
    for s in reversed(SOURCES):
        if s in seen:
            return s
    return "none"
