"""Speculative decoding: guess several tokens cheaply, verify them in one step of the big model.

Decode is memory-bandwidth bound, so verifying k proposed tokens costs about the same as
generating one. When most guesses are accepted, per-token latency falls; when the batch is already
large the GPU has no idle bandwidth to spend on verification and it becomes a loss. PolyServe
therefore tries it as a calibration stage and lets the objective decide.

Two proposers:
  * n-gram prompt lookup (vLLM): proposes continuations copied from the prompt. Free: no second
    model, and strong when outputs echo inputs (RAG, extraction, code edits).
  * a draft model: a small model of the same family, sharing the tokenizer.

Spec strings kept in Config.spec_decode: "ngram:<k>" or "draft:<hf_id>:<k>".
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# (target pattern, draft model). The draft must share the target's tokenizer.
DRAFTS = [
    (r"^Qwen/Qwen2\.5-(?:1\.5|3|7|14|32|72)B-Instruct$", "Qwen/Qwen2.5-0.5B-Instruct"),
    (r"^Qwen/Qwen2\.5-Coder-(?:1\.5|3|7|14|32)B-Instruct$", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    (r"^Qwen/Qwen3-(?:1\.7|4|8|14|32)B$", "Qwen/Qwen3-0.6B"),
    (r"^meta-llama/Llama-3\.2-3B-Instruct$", "meta-llama/Llama-3.2-1B-Instruct"),
    (r"^meta-llama/(?:Meta-)?Llama-3\.1-(?:8|70)B-Instruct$", "meta-llama/Llama-3.2-1B-Instruct"),
    (r"^meta-llama/Llama-3\.3-70B-Instruct$", "meta-llama/Llama-3.2-1B-Instruct"),
]

# vLLM 0.11's V1 engine rejects a separate draft model ("not supported yet"); n-gram works there.
VLLM_DRAFT_MIN = (0, 12)

NGRAM_TOKENS = 4
NGRAM_TOKENS_LLAMACPP = 64  # llama.cpp's ngram-mod default
DRAFT_TOKENS_VLLM = 4
DRAFT_TOKENS_LLAMACPP = 16


def draft_for(hf_id: str) -> Optional[str]:
    """A smaller model from the same family that can draft for `hf_id`, or None."""
    for pattern, draft in DRAFTS:
        if re.match(pattern, hf_id):
            return draft
    return None


def vllm_supports_draft(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM accepts a draft model for speculative decoding."""
    if version is None:
        try:
            from importlib.metadata import version as installed

            version = installed("vllm")
        except Exception:
            return False
    m = re.match(r"(\d+)\.(\d+)", version or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= VLLM_DRAFT_MIN


def ngram(k: int = NGRAM_TOKENS) -> str:
    return f"ngram:{k}"


def draft(model_id: str, k: int) -> str:
    return f"draft:{model_id}:{k}"


def parse(spec: str) -> Tuple[str, Optional[str], int]:
    """("ngram", None, k) or ("draft", model_id, k)."""
    kind, _, rest = spec.partition(":")
    if kind == "ngram":
        return "ngram", None, int(rest or NGRAM_TOKENS)
    if kind == "draft":
        model_id, _, k = rest.rpartition(":")
        if not model_id:
            raise ValueError(f"bad speculative spec {spec!r}")
        return "draft", model_id, int(k)
    raise ValueError(f"unknown speculative method in {spec!r}")


def vllm_config(spec: str) -> Dict[str, object]:
    """The value for vLLM's --speculative-config."""
    kind, model_id, k = parse(spec)
    if kind == "ngram":
        return {"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    return {"method": "draft_model", "model": model_id, "num_speculative_tokens": k}
