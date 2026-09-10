"""GGUF resolver for llama.cpp.

Order:
  1. Search the hub for a pre-quantized GGUF of this model (Q4_K_M, Q5_K_M, Q6_K, Q8_0).
  2. If none: download FP16 weights, run convert_hf_to_gguf.py, then llama-quantize
     only the quants the memory planner kept.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from polyserve.models import ModelSpec

logger = logging.getLogger(__name__)

GGUF_QUANTS: Tuple[str, ...] = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")

# Bits per weight, used to estimate GGUF size before download (or after conversion).
QUANT_BPW: Dict[str, float] = {
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.69,
    "Q6_K": 6.59,
    "Q8_0": 8.5,
    "F16": 16.0,
    "BF16": 16.0,
    "F32": 32.0,
}

# Quantizers whose repos are usually reliable, in preference order after the model author.
PREFERRED_AUTHORS = ("bartowski", "unsloth", "lmstudio-community", "QuantFactory", "TheBloke", "ggml-org")


def default_cache_dir() -> Path:
    return Path(os.environ.get("POLYSERVE_HOME", Path.home() / ".polyserve")) / "gguf"


@dataclass
class GGUFCandidate:
    repo_id: str
    filename: str
    quant: str
    size_bytes: Optional[int] = None


def match_gguf_files(files: Iterable[str], quants: Sequence[str] = GGUF_QUANTS) -> Dict[str, str]:
    """Map quant name -> filename for single-file GGUFs (multi-part shards are skipped)."""
    out: Dict[str, str] = {}
    for f in files:
        base = os.path.basename(f)
        if not base.lower().endswith(".gguf"):
            continue
        if re.search(r"-\d{5}-of-\d{5}\.gguf$", base, re.I):
            continue  # sharded; v1 only handles single files
        for q in quants:
            if q in out:
                continue
            if re.search(rf"(?:^|[-_.]){re.escape(q)}(?:[-_.]|\.gguf$)", base, re.I):
                out[q] = f
                break
    return out


def _rank_repo(repo_id: str, model_author: str, downloads: int) -> Tuple[int, int]:
    author = repo_id.split("/")[0]
    if author.lower() == model_author.lower():
        tier = 0
    elif author in PREFERRED_AUTHORS:
        tier = 1 + PREFERRED_AUTHORS.index(author)
    else:
        tier = 100
    return (tier, -downloads)


def search_hub_gguf(
    spec: ModelSpec,
    quants: Sequence[str] = GGUF_QUANTS,
    token: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, GGUFCandidate]:
    """Find pre-quantized GGUF files for the model. Returns quant -> best candidate."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    author, _, name = spec.hf_id.partition("/")
    queries = [f"{name} GGUF", f"{name}-GGUF", name]
    seen: Dict[str, Tuple[Tuple[int, int], object]] = {}
    for q in queries:
        try:
            for m in api.list_models(search=q, sort="downloads", direction=-1, limit=limit):
                rid = m.id
                if rid in seen:
                    continue
                if "gguf" not in rid.lower() and not any(
                    (t or "").lower() == "gguf" for t in (getattr(m, "tags", None) or [])
                ):
                    continue
                # Require the model name to appear in the repo name to avoid picking a different model.
                if name.lower().replace("-instruct", "") not in rid.lower().replace("-instruct", ""):
                    continue
                seen[rid] = (_rank_repo(rid, author, getattr(m, "downloads", 0) or 0), m)
        except Exception as exc:
            logger.warning("hub search for %r failed: %s", q, exc)
    ranked = sorted(seen.items(), key=lambda kv: kv[1][0])

    found: Dict[str, GGUFCandidate] = {}
    for rid, _ in ranked:
        if all(q in found for q in quants):
            break
        try:
            info = api.model_info(rid, files_metadata=True)
        except Exception as exc:
            logger.debug("model_info(%s) failed: %s", rid, exc)
            continue
        siblings = getattr(info, "siblings", None) or []
        sizes = {s.rfilename: getattr(s, "size", None) for s in siblings}
        matches = match_gguf_files(sizes.keys(), quants)
        for q, fname in matches.items():
            if q not in found:
                found[q] = GGUFCandidate(repo_id=rid, filename=fname, quant=q, size_bytes=sizes.get(fname))
    return found


def download_gguf(cand: GGUFCandidate, cache_dir: Optional[Path] = None, token: Optional[str] = None) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(cand.repo_id, cand.filename, token=token, cache_dir=str(cache_dir))
    return Path(path)


# --------------------------------------------------------------------------- conversion fallback


def find_convert_script() -> Optional[Path]:
    env = os.environ.get("LLAMA_CPP_DIR")
    cands: List[Path] = []
    if env:
        cands.append(Path(env) / "convert_hf_to_gguf.py")
    which = shutil.which("convert_hf_to_gguf.py")
    if which:
        cands.append(Path(which))
    from polyserve.hardware import llama_server_binary

    b = llama_server_binary()
    if b:
        bp = Path(b).resolve()
        cands += [bp.parent / "convert_hf_to_gguf.py", bp.parent.parent / "convert_hf_to_gguf.py"]
    for c in cands:
        if c.is_file():
            return c
    return None


def find_quantize_binary() -> Optional[str]:
    env = os.environ.get("LLAMA_QUANTIZE")
    if env and os.path.exists(env):
        return env
    for cand in ("llama-quantize", "llama-quantize.exe"):
        p = shutil.which(cand)
        if p:
            return p
    from polyserve.hardware import llama_server_binary

    b = llama_server_binary()
    if b:
        for cand in ("llama-quantize", "llama-quantize.exe"):
            p = Path(b).resolve().parent / cand
            if p.exists():
                return str(p)
    return None


def convert_and_quantize(
    spec: ModelSpec,
    quants: Sequence[str],
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Dict[str, Path]:
    """Download FP16 weights, convert to GGUF, and quantize to the requested quants."""
    from huggingface_hub import snapshot_download

    script = find_convert_script()
    quantize = find_quantize_binary()
    if script is None:
        raise RuntimeError(
            "No pre-quantized GGUF found and convert_hf_to_gguf.py is not available. "
            "Set $LLAMA_CPP_DIR to a llama.cpp checkout."
        )
    cache_dir = cache_dir or default_cache_dir()
    out_dir = cache_dir / "converted" / spec.safe_id
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = snapshot_download(spec.hf_id, revision=spec.revision, token=token)
    f16 = out_dir / f"{spec.safe_id}-F16.gguf"
    if not f16.exists():
        logger.info("converting %s -> %s", snapshot, f16)
        subprocess.run(
            [sys.executable, str(script), snapshot, "--outtype", "f16", "--outfile", str(f16)],
            check=True,
        )
    result: Dict[str, Path] = {}
    for q in quants:
        target = out_dir / f"{spec.safe_id}-{q}.gguf"
        if target.exists():
            result[q] = target
            continue
        if quantize is None:
            logger.warning("llama-quantize not found; cannot produce %s", q)
            continue
        logger.info("quantizing %s -> %s", f16, target)
        subprocess.run([quantize, str(f16), str(target), q], check=True)
        result[q] = target
    if not result:
        result["F16"] = f16
    return result


def estimate_gguf_bytes(num_params: int, quant: str) -> int:
    bpw = QUANT_BPW.get(quant.upper(), 16.0)
    return int(num_params * bpw / 8)
