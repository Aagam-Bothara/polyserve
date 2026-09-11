#!/usr/bin/env python
"""Does the quantisation PolyServe picks change output quality?

Throughput comparisons between precisions are not like-for-like unless quality holds. This
measures token-level perplexity on the same held-out text at each precision, using the same
engine (vLLM) and the same sequences, and reports the relative change.

    python benchmarks/quality_check.py --model Qwen/Qwen2.5-3B-Instruct --quants bf16 fp8

Perplexity is a weak proxy for task quality, but it is cheap, deterministic and sensitive to the
kind of degradation weight-only quantisation causes. A gap under ~1% is normal for fp8 weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

# Public-domain text (Project Gutenberg). Any fixed natural-language corpus works: the comparison
# is between precisions on identical sequences, not against a published perplexity number.
TEXT_URL = "https://www.gutenberg.org/files/11/11-0.txt"


def load_text(cache: Path) -> str:
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    import httpx

    r = httpx.get(TEXT_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text, encoding="utf-8")
    return r.text


def make_sequences(text: str, tokenizer, n_seq: int, seq_len: int) -> List[List[int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    start = len(ids) // 10  # skip the licence header
    seqs = []
    for i in range(n_seq):
        a = start + i * seq_len
        if a + seq_len > len(ids):
            break
        seqs.append(ids[a:a + seq_len])
    return seqs


def perplexity(model: str, quant: Optional[str], seqs: List[List[int]], max_len: int) -> float:
    from vllm import LLM, SamplingParams

    kwargs = dict(model=model, max_model_len=max_len, gpu_memory_utilization=0.85,
                  enforce_eager=True, disable_log_stats=True)
    if quant and quant != "bf16":
        kwargs["quantization"] = quant
    else:
        kwargs["dtype"] = "bfloat16"
    llm = LLM(**kwargs)
    try:
        params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        # vLLM 0.11 removed generate(prompt_token_ids=...); token prompts go through `prompts`.
        outs = llm.generate([{"prompt_token_ids": s} for s in seqs], sampling_params=params)
        total_lp, total_n = 0.0, 0
        for o in outs:
            lps = o.prompt_logprobs or []
            for entry in lps:
                if not entry:
                    continue  # first token has no conditional logprob
                # prompt_logprobs=0 returns exactly the actual token's logprob
                lp = next(iter(entry.values()))
                total_lp += float(getattr(lp, "logprob", lp))
                total_n += 1
        return math.exp(-total_lp / max(total_n, 1))
    finally:
        del llm
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quants", nargs="+", default=["bf16", "fp8"])
    ap.add_argument("--sequences", type=int, default=24)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/quality.json"))
    ap.add_argument("--cache", type=Path, default=Path("/tmp/quality-corpus.txt"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    seqs = make_sequences(load_text(args.cache), tok, args.sequences, args.seq_len)
    if not seqs:
        print("corpus too short", file=sys.stderr)
        return 2
    print(f"{len(seqs)} sequences x {args.seq_len} tokens = {len(seqs) * args.seq_len} tokens", flush=True)

    results = {}
    for q in args.quants:
        print(f"loading {args.model} at {q} ...", flush=True)
        ppl = perplexity(args.model, q, seqs, args.seq_len + 8)
        results[q] = ppl
        print(f"{q}: perplexity {ppl:.4f}", flush=True)

    base = args.quants[0]
    out = {
        "model": args.model, "sequences": len(seqs), "seq_len": args.seq_len,
        "corpus": TEXT_URL, "perplexity": results,
        "delta_pct": {q: (results[q] - results[base]) / results[base] * 100 for q in results},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for q, d in out["delta_pct"].items():
        print(f"{q}: {results[q]:.4f} ({d:+.2f}% vs {base})")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
