#!/usr/bin/env python
"""Task-level quality: GSM8K accuracy at each weight precision, same engine, same prompts.

Perplexity on a well-known book rose 26-36% with the Hub's 4-bit checkpoints and 1-2% with fp8
(benchmarks/quality_check.py). A book the model has memorised may exaggerate that, so this measures
whether the answers change: grade-school maths problems, zero-shot with the model's chat template,
greedy decoding, the final number compared with the reference answer. Each precision is compared
with the first problem by problem, so a small real difference is not lost in the confidence interval.

    python benchmarks/task_quality.py --model Qwen/Qwen2.5-3B-Instruct --quants bf16 fp8 awq gptq
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

INSTRUCTION = "Solve the problem step by step. End your answer with a line of the form 'The answer is N'."
_ANSWER = re.compile(r"answer is\s*:?\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def final_number(text: str) -> Optional[float]:
    """The number after the last 'The answer is', else the last number anywhere in the answer."""
    found = _ANSWER.findall(text)
    candidate = found[-1] if found else (_NUMBER.findall(text) or [None])[-1]
    if candidate is None:
        return None
    try:
        return float(candidate.replace(",", "").rstrip("."))
    except ValueError:
        return None


def reference(answer: str) -> float:
    """GSM8K reference answers end with '#### N'."""
    return float(answer.split("####")[-1].strip().replace(",", ""))


def wilson(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for an accuracy."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def answers(model: str, quant: str, questions: List[str], max_len: int, max_tokens: int) -> List[str]:
    from vllm import LLM, SamplingParams

    from polyserve.quantized import PREQUANTIZED

    kwargs = dict(model=model, max_model_len=max_len, gpu_memory_utilization=0.85, disable_log_stats=True)
    if quant in PREQUANTIZED:
        kwargs["dtype"] = "auto"  # the checkpoint's quantization_config chooses the kernel
    elif quant != "bf16":
        kwargs["quantization"] = quant
    else:
        kwargs["dtype"] = "bfloat16"
    llm = LLM(**kwargs)
    try:
        conversations = [[{"role": "user", "content": f"{q}\n\n{INSTRUCTION}"}] for q in questions]
        outs = llm.chat(conversations, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        return [o.outputs[0].text for o in outs]
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
    ap.add_argument("--limit", type=int, default=0, help="first N test problems (0 = all 1319)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/task_quality.json"))
    args = ap.parse_args()

    from datasets import load_dataset

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    if args.limit:
        rows = rows[: args.limit]
    questions = [r["question"] for r in rows]
    refs = [reference(r["answer"]) for r in rows]

    results, checkpoints, samples, per_problem = {}, {}, {}, {}
    for q in args.quants:
        target = args.model
        from polyserve.quantized import PREQUANTIZED

        if q in PREQUANTIZED:
            from polyserve.models import ModelSpec
            from polyserve.quantized import find_prequantized_repos

            repo = find_prequantized_repos(ModelSpec(hf_id=args.model), methods=[q]).get(q)
            if repo is None:
                print(f"{q}: no pre-quantized checkpoint of {args.model} on the Hub; skipped", flush=True)
                continue
            target = checkpoints[q] = repo.repo_id
        print(f"{q}: {target} on {len(questions)} problems ...", flush=True)
        try:
            texts = answers(target, q, questions, max_len=2048, max_tokens=args.max_tokens)
        except Exception as exc:  # one precision failing (an engine bug, say) must not lose the others
            results[q] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"{q}: FAILED: {results[q]['error'][:200]}", flush=True)
            continue
        got = [final_number(t) for t in texts]
        ok = [g is not None and abs(g - r) < 1e-6 for g, r in zip(got, refs)]
        correct = sum(ok)
        lo, hi = wilson(correct, len(refs))
        per_problem[q] = ok
        results[q] = {"correct": correct, "n": len(refs), "accuracy": correct / len(refs), "ci95": [lo, hi],
                      "unparsed": sum(g is None for g in got)}
        base = next(iter(per_problem))
        if q != base:
            results[q]["vs_" + base] = {"lost": sum(b and not x for b, x in zip(per_problem[base], ok)),
                                        "gained": sum(x and not b for b, x in zip(per_problem[base], ok))}
        samples[q] = [{"question": questions[i], "reference": refs[i], "answer": texts[i]} for i in range(3)]
        print(f"{q}: {correct}/{len(refs)} = {correct / len(refs):.1%} (95% CI {lo:.1%} to {hi:.1%})"
              + (f"; against {base}: {results[q]['vs_' + base]['lost']} lost, "
                 f"{results[q]['vs_' + base]['gained']} gained" if q != base else ""), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "task": "gsm8k", "split": "test", "problems": len(refs), "instruction": INSTRUCTION,
        "max_tokens": args.max_tokens, "accuracy": results, "checkpoints": checkpoints, "samples": samples,
    }, indent=2), encoding="utf-8")
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
