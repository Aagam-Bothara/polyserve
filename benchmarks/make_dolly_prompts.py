#!/usr/bin/env python
"""Write the prompt file used for the `--workload-file` results in docs/benchmarks.md.

    python benchmarks/make_dolly_prompts.py --out dolly.jsonl
    polyserve compare Qwen/Qwen2.5-3B-Instruct --workload chat --workload-file dolly.jsonl --repeats 3

    # 300 more, disjoint from those, to measure a pick on prompts its calibration never saw:
    python benchmarks/make_dolly_prompts.py --offset 300 --out dolly-heldout.jsonl
    polyserve compare ... --workload-file dolly.jsonl --eval-workload-file dolly-heldout.jsonl

The prompts come from databricks-dolly-15k (CC BY-SA 3.0), a dataset none of the presets use: rows
after a seed-0 shuffle (the first 300 by default), each row's context (when it has one) followed by
its instruction. They are not committed; this script rebuilds the same file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def to_prompt(row: Dict[str, str]) -> str:
    """A Dolly row as one prompt: the passage to work from, then what to do with it."""
    context, instruction = (row.get("context") or "").strip(), row["instruction"].strip()
    return f"{context}\n\n{instruction}" if context else instruction


def select_rows(rows: Sequence[Dict[str, str]], n: int, offset: int = 0, seed: int = 0) -> List[Dict[str, str]]:
    """Rows offset to offset + n of the seeded shuffle, so different offsets never share a row."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[offset: offset + n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("dolly.jsonl"))
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--offset", type=int, default=0, help="skip this many shuffled rows (300: the held-out set)")
    args = ap.parse_args()

    from datasets import load_dataset

    rows = select_rows(list(load_dataset("databricks/databricks-dolly-15k", split="train")), args.n, args.offset)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"prompt": to_prompt(row)}) + "\n")
    print(f"wrote {len(rows)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
