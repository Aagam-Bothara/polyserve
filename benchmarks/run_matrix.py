#!/usr/bin/env python
"""Run `polyserve compare` for every (model, workload) pair on this machine that has no results file yet.

    python benchmarks/run_matrix.py --models meta-llama/Llama-3.2-3B-Instruct [--workloads chat rag] [--ollama]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from polyserve.bench.compare import results_path
from polyserve.bench.references import ollama_tag_for
from polyserve.calibrate.workload import WORKLOAD_NAMES
from polyserve.hardware import hardware_hash, probe
from polyserve.models import ModelSpec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOAD_NAMES))
    ap.add_argument("--objective", default="balanced")
    ap.add_argument("--ollama", action="store_true", help="include the ollama-default row")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hw = probe()
    hw_hash = hardware_hash(hw)
    print(f"machine {hw_hash}: {hw.gpu.name if hw.gpu else hw.cpu.model_name}")
    failures = 0
    for model in args.models:
        for wl in args.workloads:
            path = results_path(hw_hash, ModelSpec(hf_id=model), wl, args.objective, args.out)
            if path.exists():
                print(f"skip {model} {wl}: {path} exists")
                continue
            cmd = [sys.executable, "-m", "polyserve.cli", "compare", model, "--workload", wl,
                   "--objective", args.objective]
            if args.out:
                cmd += ["--out", str(args.out)]
            if args.ollama:
                tag = ollama_tag_for(model)
                if tag:
                    cmd += ["--ollama-model", tag]
                else:
                    print(f"no known ollama tag for {model}; ollama row skipped")
            print("+", " ".join(cmd))
            if args.dry_run:
                continue
            rc = subprocess.call(cmd)
            if rc != 0:
                failures += 1
                print(f"FAILED ({rc}): {model} {wl}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
