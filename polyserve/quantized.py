"""Find pre-quantized checkpoints of a model on the Hugging Face Hub: 4-bit AWQ and GPTQ, 8-bit W8A8.

On a memory-bound decode the weights are read once per step, so bytes per weight set the speed
limit. The RTX 3090 benchmarks showed that 8-bit weights account for essentially all of the gain
over stock vLLM; 4-bit halves the traffic again. vLLM and SGLang read the checkpoint's
`quantization_config` and pick the kernel themselves (Marlin on Ampere and newer), so all PolyServe
needs is the right repository and its size for the memory planner.

W8A8 checkpoints (compressed-tensors with int8 weights and int8 activations, as Red Hat publishes
them) are the 8-bit option for Ampere cards, where vLLM 0.29 cannot quantize weights to fp8 itself;
its int8 kernels need compute capability 7.5 (0.29 source).

Quality is not free: `benchmarks/task_quality.py --quants bf16 fp8 awq w8a8` measures it, and
`--quant` restricts what calibration may choose. Every method here is opt-in for that reason.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from polyserve.gguf import list_hub_models
from polyserve.models import ModelSpec

logger = logging.getLogger(__name__)

INT4_METHODS: Tuple[str, ...] = ("awq", "gptq")
INT8_METHODS: Tuple[str, ...] = ("w8a8",)
PREQUANTIZED: Tuple[str, ...] = INT4_METHODS + INT8_METHODS
# Bytes per parameter when the repository size is unknown: 4 bits plus group scales and zeros, or
# 8 bits plus per-channel scales.
FALLBACK_BYTES_PER_PARAM = {"awq": 0.55, "gptq": 0.55, "w8a8": 1.05}
# How each method's repositories are named on the Hub; the default is "<model>-<method>".
QUERIES = {"w8a8": ("{name}-quantized.w8a8", "{name}-w8a8")}
# Quantizers whose uploads are usually faithful, after the model's own author.
PREFERRED_AUTHORS = ("hugging-quants", "RedHatAI", "neuralmagic", "casperhansen", "TheBloke", "unsloth")


@dataclass
class PrequantizedRepo:
    repo_id: str
    method: str
    bits: int = 4
    group_size: Optional[int] = None
    size_bytes: Optional[int] = None


def _rank(repo_id: str, model_author: str, downloads: int) -> Tuple[int, int, str]:
    author = repo_id.split("/")[0]
    if author.lower() == model_author.lower():
        tier = 0
    elif author in PREFERRED_AUTHORS:
        tier = 1 + PREFERRED_AUTHORS.index(author)
    else:
        tier = 100
    return (tier, -downloads, repo_id)


def _default_fetch_config(repo_id: str, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    try:
        with open(hf_hub_download(repo_id, "config.json", token=token), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.debug("no config.json for %s: %s", repo_id, exc)
        return None


def _repo_size(api: Any, repo_id: str) -> Optional[int]:
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:
        logger.debug("model_info(%s) failed: %s", repo_id, exc)
        return None
    sizes = [getattr(s, "size", None) for s in (getattr(info, "siblings", None) or [])
             if getattr(s, "rfilename", "").endswith(".safetensors")]
    sizes = [s for s in sizes if s]
    return int(sum(sizes)) if sizes else None


def _int8(tensor: Any) -> bool:
    return isinstance(tensor, dict) and int(tensor.get("num_bits") or 0) == 8 and str(tensor.get("type", "")).lower() == "int"


def verified_bits(method: str, q: Dict[str, Any]) -> Optional[int]:
    """The bit width a checkpoint's quantization_config shows for `method`, or None when it is something else.

    A repository's name is not evidence: an "-Int8" GPTQ upload or an fp8 model named w8a8 must not pass."""
    if method in INT4_METHODS:
        ok = str(q.get("quant_method", "")).lower() == method and int(q.get("bits") or 0) == 4
        return 4 if ok else None
    if method == "w8a8":
        groups = list((q.get("config_groups") or {}).values())
        ok = (str(q.get("quant_method", "")).lower() == "compressed-tensors" and bool(groups)
              and all(_int8(g.get("weights")) and _int8(g.get("input_activations")) for g in groups))
        return 8 if ok else None
    return None


def find_prequantized_repos(
    spec: ModelSpec,
    methods: Sequence[str] = PREQUANTIZED,
    token: Optional[str] = None,
    api: Any = None,
    fetch_config: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    limit: int = 30,
    verify: int = 3,
) -> Dict[str, PrequantizedRepo]:
    """method -> best verified pre-quantized repository of the same model. Empty when none is found."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    fetch = fetch_config or (lambda rid: _default_fetch_config(rid, token))
    author, _, name = spec.hf_id.partition("/")
    base = name.lower()
    found: Dict[str, PrequantizedRepo] = {}
    for method in methods:
        ranked: Dict[str, Tuple[int, int, str]] = {}
        for query in (t.format(name=name, method=method) for t in QUERIES.get(method, ("{name}-{method}",
                                                                                         "{name} {method}"))):
            try:
                for m in list_hub_models(api, query, limit):
                    rid = m.id
                    low = rid.lower()
                    # Same model, this method, and not a GGUF re-upload.
                    if method not in low or "gguf" in low or base not in low:
                        continue
                    ranked[rid] = _rank(rid, author, getattr(m, "downloads", 0) or 0)
            except Exception as exc:
                logger.warning("hub search for %r failed: %s", query, exc)
        for rid in sorted(ranked, key=lambda r: ranked[r])[:verify]:
            q = (fetch(rid) or {}).get("quantization_config") or {}
            bits = verified_bits(method, q)
            if bits is None:
                continue
            found[method] = PrequantizedRepo(repo_id=rid, method=method, bits=bits, group_size=q.get("group_size"),
                                             size_bytes=_repo_size(api, rid))
            break
    return found


def hf_weight_options(spec: ModelSpec, params: int, wanted: Sequence[str],
                      bytes_per_param: Callable[[str], float]) -> Tuple[Dict[str, int], Dict[str, str]]:
    """(weights_bytes, repo per quant) for a Hugging Face backend: full/fp8 from the base repo,
    4-bit and W8A8 from whatever pre-quantized repository the Hub has."""
    weights: Dict[str, int] = {}
    paths: Dict[str, str] = {}
    for p in wanted:
        if p not in PREQUANTIZED:
            weights[p] = int(params * bytes_per_param(p))
    pre = [p for p in wanted if p in PREQUANTIZED]
    if pre:
        try:
            repos = find_prequantized_repos(spec, methods=pre)
        except Exception as exc:
            logger.warning("pre-quantized checkpoint search failed: %s", exc)
            repos = {}
        for method, repo in repos.items():
            paths[method] = repo.repo_id
            weights[method] = repo.size_bytes or int(params * FALLBACK_BYTES_PER_PARAM[method])
    return weights, paths


def int4_list(values: List[str]) -> List[str]:
    return [v for v in values if v in INT4_METHODS]
