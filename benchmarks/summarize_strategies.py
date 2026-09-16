#!/usr/bin/env python
"""Turn strategy benchmark outputs into the markdown tables behind docs/benchmarks.md.

    cd benchmarks/strategies && python ../summarize_strategies.py \
        --results results results-llamacpp results-layout results-disagg results-real results-real/v2 \
            results-real/v3 results-real/sglang results-l4 results-l4/p95 results-cpu results-dolly \
            results-dolly/budget-10m results-dolly/p95 results-dolly/p95-old-warmup results-dolly/p95-old-warmup-budget \
            results-llama/heldout results-llama/oasst results-llama/baselines results-llama/heldout-fixed \
            results-llama/headtohead results-llama/qwen-heldout results-llama/explore results-llama/specfirst \
            results-14b \
        --ablation ablation ablation-llamacpp ablation-real ablation-real/flashinfer ablation-l4 ablation-dolly \
        --quality quality-3b.json quality-7b.json --gguf-quality quality-gguf-3b.txt \
        --task-quality task-quality-3b.json task-quality-7b.json task-quality-7b-l4.json task-quality-3b-w8a8.json \
        --out SUMMARY.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyserve.bench.ablation import (
    ablation_report,
    compare_table,
    gguf_quality_table,
    quality_table,
    task_quality_table,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="*", default=[], help="directories of `polyserve compare` JSON")
    ap.add_argument("--ablation", type=Path, nargs="*", default=[], help="directories of ablation JSON")
    ap.add_argument("--quality", type=Path, nargs="*", default=[], help="quality_check.py JSON files")
    ap.add_argument("--gguf-quality", type=Path, default=None, help="llama-perplexity summary lines")
    ap.add_argument("--task-quality", type=Path, nargs="*", default=[], help="task_quality.py JSON files")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    parts = []
    results = [f for d in args.results for f in sorted(d.glob("*.json"))]
    if results:
        parts += ["## PolyServe against stock settings", "", compare_table(results), ""]
    ablations = [f for d in args.ablation for f in sorted(d.glob("*__ablation.json"))]
    if ablations:
        parts += ["## What each strategy is worth", "", ablation_report(ablations), ""]
    if args.quality:
        parts += ["## Quality", "", quality_table(args.quality), ""]
    if args.gguf_quality:
        parts += [gguf_quality_table(args.gguf_quality), ""]
    if args.task_quality:
        parts += ["## Task accuracy (GSM8K)", "", task_quality_table(args.task_quality), ""]
    text = "\n".join(parts)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
