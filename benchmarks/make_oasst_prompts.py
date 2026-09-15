#!/usr/bin/env python
"""Write held-out prompts from OpenAssistant (oasst2, Apache-2.0): the opening user turn of English conversations.

    python benchmarks/make_oasst_prompts.py --out oasst.jsonl
    polyserve compare <model> --workload chat --workload-file dolly.jsonl --eval-workload-file oasst.jsonl --repeats 3

Nothing else in the repository uses this dataset, so a pick calibrated on other prompts meets these for the
first time. The first 300 distinct openers after a seed-0 shuffle; they are not committed, this script
rebuilds the same file.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def first_turns(rows: Iterable[Dict[str, object]], n: int, seed: int = 0, lang: str = "en") -> List[str]:
    """Up to n distinct conversation openers (user messages with no parent, in `lang`), after a seeded shuffle."""
    seen, openers = set(), []
    for r in rows:
        text = str(r.get("text") or "").strip()
        if (r.get("parent_id") is None and r.get("role") == "prompter" and r.get("lang") == lang
                and text and text not in seen):
            seen.add(text)
            openers.append(text)
    random.Random(seed).shuffle(openers)
    return openers[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("oasst.jsonl"))
    ap.add_argument("-n", type=int, default=300)
    args = ap.parse_args()

    from datasets import load_dataset

    prompts = first_turns(load_dataset("OpenAssistant/oasst2", split="train"), args.n)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in prompts:
            fh.write(json.dumps({"prompt": p}) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
