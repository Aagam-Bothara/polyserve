"""Read what the memory planner needs from a Hugging Face model config."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from polyserve.models import ArchInfo, ModelSpec

logger = logging.getLogger(__name__)

DTYPE_BYTES: Dict[str, int] = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "bfloat16": 2,
    "bf16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "auto": 2,
}


def dtype_bytes(name: str) -> int:
    return DTYPE_BYTES.get(name.lower(), 2)


# Bytes per KV element for llama.cpp's block-quantized caches: 32 values plus their scale(s). vLLM's
# int8_per_token_head keeps a scale per token and head beside one byte per value; sized here as a 4-byte
# scale over a 64-wide head, which overstates it for wider heads (Qwen2.5's are 128).
KV_ELEMENT_BYTES: Dict[str, float] = {"q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32,
                                      "q5_1": 24 / 32, "int8_per_token_head": 1 + 4 / 64}


def kv_element_bytes(name: str) -> float:
    return KV_ELEMENT_BYTES.get(name.lower(), float(dtype_bytes(name)))


def _first(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def arch_from_config(cfg: Dict[str, Any]) -> ArchInfo:
    """Build ArchInfo from a raw config.json dict. Handles nested text_config (multimodal)."""
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        merged = dict(cfg)
        merged.update(cfg["text_config"])
        cfg = merged
    hidden = int(_first(cfg, "hidden_size", "n_embd", "d_model", default=4096))
    layers = int(_first(cfg, "num_hidden_layers", "n_layer", "num_layers", default=32))
    heads = int(_first(cfg, "num_attention_heads", "n_head", default=32))
    kv_heads = int(_first(cfg, "num_key_value_heads", "num_kv_heads", default=heads))
    head_dim = int(_first(cfg, "head_dim", default=hidden // max(heads, 1)))
    vocab = int(_first(cfg, "vocab_size", default=32000))
    max_pos = int(_first(cfg, "max_position_embeddings", "n_positions", "n_ctx", default=4096))
    dtype = str(_first(cfg, "torch_dtype", "dtype", default="bfloat16"))
    archs = cfg.get("architectures") or ["unknown"]
    intermediate = int(_first(cfg, "intermediate_size", "n_inner", default=4 * hidden))
    tie = bool(cfg.get("tie_word_embeddings", False))
    params = estimate_params(hidden, layers, heads, kv_heads, head_dim, vocab, intermediate, tie)
    return ArchInfo(
        architecture=str(archs[0]),
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=vocab,
        max_position_embeddings=max_pos,
        num_params=params,
        torch_dtype=dtype,
    )


def estimate_params(
    hidden: int,
    layers: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    vocab: int,
    intermediate: int,
    tie_embeddings: bool = False,
) -> int:
    """Dense-transformer parameter estimate (within a few percent for Llama-style models)."""
    q_o = 2 * hidden * heads * head_dim
    k_v = 2 * hidden * kv_heads * head_dim
    mlp = 3 * hidden * intermediate  # gated MLP (gate, up, down)
    norms = 2 * hidden
    per_layer = q_o + k_v + mlp + norms
    embed = vocab * hidden * (1 if tie_embeddings else 2)
    return layers * per_layer + embed + hidden


def _load_local_config(path: str) -> Optional[Dict[str, Any]]:
    cand = os.path.join(path, "config.json") if os.path.isdir(path) else path
    if os.path.isfile(cand):
        with open(cand, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def fetch_config(spec: ModelSpec, token: Optional[str] = None) -> Dict[str, Any]:
    """config.json for a model: local dir, or HF hub (cached)."""
    local = _load_local_config(spec.hf_id)
    if local is not None:
        return local
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(spec.hf_id, "config.json", revision=spec.revision, token=token)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def safetensors_param_count(spec: ModelSpec, token: Optional[str] = None) -> Optional[int]:
    """Exact parameter count from the hub's safetensors metadata, if published."""
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(spec.hf_id, revision=spec.revision)
        st = getattr(info, "safetensors", None)
        if st and getattr(st, "total", None):
            return int(st.total)
    except Exception as exc:
        logger.debug("safetensors metadata unavailable for %s: %s", spec.hf_id, exc)
    return None


def load_arch(spec: ModelSpec, token: Optional[str] = None) -> ArchInfo:
    arch = arch_from_config(fetch_config(spec, token=token))
    exact = safetensors_param_count(spec, token=token)
    if exact:
        arch.num_params = exact
    return arch
