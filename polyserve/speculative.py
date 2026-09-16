"""Speculative decoding: guess several tokens cheaply, verify them in one step of the big model.

Decode is memory-bandwidth bound, so verifying k proposed tokens costs about the same as
generating one. When most guesses are accepted, per-token latency falls; when the batch is already
large the GPU has no idle bandwidth to spend on verification and it becomes a loss. PolyServe
therefore tries it as a calibration stage and lets the objective decide.

Proposers:
  * n-gram prompt lookup: proposes continuations copied from the prompt. Free: no second model, and
    strong when outputs echo inputs (RAG, extraction, code edits). On vLLM 0.29 and later it runs on
    the GPU (`ngram_gpu`): vLLM 0.29 turns async scheduling off for its CPU n-gram and suffix
    proposers and keeps it for `ngram_gpu` and draft models. The CPU version lost 87% of throughput
    from four users up on an A40; whether losing async scheduling is why has not been measured.
  * suffix decoding (vLLM, needs `pip install arctic-inference==0.1.1`): matches the prompt and the
    model's earlier outputs in a suffix tree and adapts how many tokens it proposes per request.
  * a draft model: a small model of the same family, sharing the tokenizer.

Spec strings kept in Config.spec_decode: "ngram:<k>", "ngram_gpu:<k>", "suffix:<k>" or "draft:<hf_id>:<k>".
"""

from __future__ import annotations

import importlib.util
import re
from typing import Dict, Optional, Tuple

# (target pattern, draft model). The draft must share the target's tokenizer. Llama drafts come from the target's
# own organisation, so unsloth's ungated copies of Meta's gated models draft with an ungated model too.
DRAFTS = [
    # Qwen2.5 splits its vocabulary by size: 0.5B, 1.5B and 3B carry 151936 tokens, 7B and larger 152064 (read
    # from each config.json). vLLM refuses a draft whose vocabulary differs from the target's, so the small draft
    # serves only the small targets. Offering it for 14B produced a configuration that could never launch:
    # "Target and draft model should have the same vocabulary size. Target model vocab_size=152064. Draft model
    # vocab_size=151936" (Qwen2.5-14B-Instruct on an RTX 4090, 2026-09-16). Qwen2.5-Coder splits the same way.
    (r"^Qwen/Qwen2\.5-(?:1\.5|3)B-Instruct$", "Qwen/Qwen2.5-0.5B-Instruct"),
    (r"^Qwen/Qwen2\.5-Coder-(?:1\.5|3)B-Instruct$", "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    # Qwen3 keeps one vocabulary (151936) from 0.6B through 32B, so one draft covers every size here.
    (r"^Qwen/Qwen3-(?:1\.7|4|8|14|32)B$", "Qwen/Qwen3-0.6B"),
    (r"^(meta-llama|unsloth)/Llama-3\.2-3B-Instruct$", r"\1/Llama-3.2-1B-Instruct"),
    (r"^(meta-llama|unsloth)/(?:Meta-)?Llama-3\.1-(?:8|70)B-Instruct$", r"\1/Llama-3.2-1B-Instruct"),
    (r"^(meta-llama|unsloth)/Llama-3\.3-70B-Instruct$", r"\1/Llama-3.2-1B-Instruct"),
]

# vLLM 0.11's V1 engine rejects a separate draft model ("not supported yet"); n-gram works there.
VLLM_DRAFT_MIN = (0, 12)
# Both methods are in vLLM 0.29's SpeculativeConfig; earlier releases were not checked.
VLLM_NGRAM_GPU_MIN = (0, 29)
VLLM_SUFFIX_MIN = (0, 29)

NGRAM_TOKENS = 4
NGRAM_TOKENS_LLAMACPP = 64  # llama.cpp's ngram-mod default
SUFFIX_TOKENS = 24  # vLLM's default for suffix decoding: its maximum tree depth
DRAFT_TOKENS_VLLM = 4
DRAFT_TOKENS_LLAMACPP = 16

LOOKUP = ("ngram", "ngram_gpu", "suffix")  # proposers that need no second model


def draft_for(hf_id: str) -> Optional[str]:
    """A smaller model from the same family that can draft for `hf_id`, or None."""
    for pattern, draft in DRAFTS:
        m = re.match(pattern, hf_id)
        if m:
            return m.expand(draft)
    return None


def _vllm_version() -> Optional[str]:
    try:
        from importlib.metadata import version as installed

        return installed("vllm")
    except Exception:
        return None


def _vllm_at_least(minimum: Tuple[int, int], version: Optional[str] = None) -> bool:
    m = re.match(r"(\d+)\.(\d+)", (version if version is not None else _vllm_version()) or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= minimum


def vllm_supports_draft(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM accepts a draft model for speculative decoding."""
    return _vllm_at_least(VLLM_DRAFT_MIN, version)


def vllm_supports_ngram_gpu(version: Optional[str] = None) -> bool:
    """Whether the installed (or given) vLLM runs n-gram lookup on the GPU."""
    return _vllm_at_least(VLLM_NGRAM_GPU_MIN, version)


def vllm_supports_suffix(version: Optional[str] = None) -> bool:
    """Whether vLLM can use suffix decoding here: a recent enough release, and Arctic Inference installed."""
    return _vllm_at_least(VLLM_SUFFIX_MIN, version) and importlib.util.find_spec("arctic_inference") is not None


def ngram(k: int = NGRAM_TOKENS) -> str:
    return f"ngram:{k}"


def ngram_gpu(k: int = NGRAM_TOKENS) -> str:
    return f"ngram_gpu:{k}"


def suffix(k: int = SUFFIX_TOKENS) -> str:
    return f"suffix:{k}"


def draft(model_id: str, k: int) -> str:
    return f"draft:{model_id}:{k}"


def parse(spec: str) -> Tuple[str, Optional[str], int]:
    """(kind, None, k) for the lookup proposers ("ngram", "ngram_gpu", "suffix"), or ("draft", model_id, k)."""
    kind, _, rest = spec.partition(":")
    if kind in LOOKUP:
        return kind, None, int(rest or (SUFFIX_TOKENS if kind == "suffix" else NGRAM_TOKENS))
    if kind == "draft":
        model_id, _, k = rest.rpartition(":")
        if not model_id:
            raise ValueError(f"bad speculative spec {spec!r}")
        return "draft", model_id, int(k)
    raise ValueError(f"unknown speculative method in {spec!r}")


def vllm_config(spec: str) -> Dict[str, object]:
    """The value for vLLM's --speculative-config."""
    kind, model_id, k = parse(spec)
    if kind in ("ngram", "ngram_gpu"):  # one code path in vLLM, with the same lookup settings
        return {"method": kind, "num_speculative_tokens": k, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    if kind == "suffix":
        return {"method": "suffix", "num_speculative_tokens": k}
    return {"method": "draft_model", "model": model_id, "num_speculative_tokens": k}
