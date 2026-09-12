#!/usr/bin/env python
"""Turn strategy benchmark outputs into the markdown tables used in the README.

    python benchmarks/summarize_strategies.py --results benchmarks/strategies/results \
        --ablation benchmarks/strategies/ablation --quality benchmarks/strategies/quality-*.json \
        --gguf-quality benchmarks/strategies/quality-gguf.txt --out benchmarks/strategies/SUMMARY.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyserve.bench.ablation import ablation_report, compare_table, gguf_quality_table, quality_table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="*", default=[], help="directories of `polyserve compare` JSON")
    ap.add_argument("--ablation", type=Path, nargs="*", default=[], help="directories of ablation JSON")
    ap.add_argument("--quality", type=Path, nargs="*", default=[], help="quality_check.py JSON files")
    ap.add_argument("--gguf-quality", type=Path, default=None, help="llama-perplexity summary lines")
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
    text = "\n".join(parts)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
