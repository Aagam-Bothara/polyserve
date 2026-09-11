#!/usr/bin/env python
"""Isolate what the search contributes beyond simply choosing 8-bit weights.

The headline comparison in RESULTS.md is PolyServe's pick against `vllm serve <model>`, which
defaults to 16-bit. On Ampere, vLLM's fp8 is weight-only, so it halves weight traffic in a
memory-bound decode and most of the measured gain may be that choice alone rather than the
configuration search.

This script measures three configurations on the same workload, back to back on one card:

    vllm-default       stock bf16     (model max context, max_num_seqs 256, gmu 0.90)
    vllm-fp8-default   stock fp8      (identical, plus --quantization fp8)
    polyserve          the calibrated pick, read from the matching results file

The gap between the second and third is the search's own contribution.

    python benchmarks/isolate_quantisation.py --model Qwen/Qwen2.5-3B-Instruct \
        --workloads default chat high-concurrency --out benchmarks/isolation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from polyserve.bench.compare import results_path
from polyserve.calibrate.objectives import Constraints, rank
from polyserve.calibrate.search import SubprocessTrialRunner
from polyserve.calibrate.workload import get_workload
from polyserve.hardware import hardware_hash, probe
from polyserve.models import Config, ModelSpec, TrialResult
from polyserve.pipeline import prepare_and_plan, select


def _load_polyserve_pick(results_dir: Path, model: str, workload: str, objective: str) -> Optional[Config]:
    """The calibrated configuration this workload chose, from any machine's results file."""
    spec = ModelSpec(hf_id=model)
    exact = results_path("*", spec, workload, objective, results_dir)
    for f in sorted(results_dir.glob(exact.name.replace("*", "*"))):
        data = json.loads(f.read_text(encoding="utf-8"))
        for row in data["rows"]:
            if row["label"] == "polyserve":
                return Config.model_validate(row["config"])
    return None


def _score(res: TrialResult, objective: str, cons: Constraints) -> Dict[str, object]:
    ranked = rank([res], objective, cons)
    if not ranked:
        return {"ok": False, "error": res.error}
    m = ranked[0].metrics
    return {
        "ok": True,
        "concurrency": m.concurrency,
        "tok_s": m.tok_s,
        "ttft_ms": m.ttft_ms,
        "ttft_p95_ms": m.ttft_p95_ms,
        "tpot_ms": m.tpot_ms,
        "joules_per_token": m.joules_per_token,
        "peak_mem_mb": m.peak_mem_mb,
        "meets_slo": ranked[0].feasible,
        "token_count_source": m.token_count_source,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workloads", nargs="+", default=["default"])
    ap.add_argument("--objective", default="balanced")
    ap.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/isolation"))
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    hw = probe()
    if hw.gpu is None:
        print("this experiment needs an NVIDIA GPU", file=sys.stderr)
        return 2
    spec = ModelSpec(hf_id=args.model)
    candidates, reg = select(hw, spec, force="vllm")
    if not candidates:
        print("vLLM is not usable on this machine", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for wl_name in args.workloads:
        wl = get_workload(wl_name)
        cons = Constraints(ttft_ceiling_ms=wl.ttft_ceiling_ms)
        planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl)
        model = planned.prepared["vllm"]
        max_pos = model.arch.max_position_embeddings

        configs: Dict[str, Config] = {
            "vllm-default": Config(backend="vllm", quant="bf16", ctx=max_pos, batch=256,
                                   gpu_memory_utilization=0.90),
            "vllm-fp8-default": Config(backend="vllm", quant="fp8", ctx=max_pos, batch=256,
                                       gpu_memory_utilization=0.90),
        }
        pick = _load_polyserve_pick(args.results, args.model, wl_name, args.objective)
        if pick is None:
            print(f"no polyserve row for {wl_name}; skipping that column", file=sys.stderr)
        else:
            configs["polyserve"] = pick

        runner = SubprocessTrialRunner(backends=reg, models=planned.prepared, hw=hw, workload=wl,
                                       log_dir=args.log_dir)
        rows: List[Dict[str, object]] = []
        for label, cfg in configs.items():
            print(f"[{wl_name}] {label}: {cfg.key()} ...", flush=True)
            res = runner.run(cfg, f"isolate:{label}")
            row = {"label": label, "config": cfg.model_dump(), "config_key": cfg.key()}
            row.update(_score(res, args.objective, cons))
            rows.append(row)
            print(f"[{wl_name}] {label}: {row.get('tok_s', 0) or 0:.1f} tok/s "
                  f"@c{row.get('concurrency')} slo={row.get('meets_slo')}", flush=True)

        out = {
            "hardware_hash": hardware_hash(hw), "gpu": hw.gpu.name, "model_id": args.model,
            "workload": wl_name, "objective": args.objective, "ttft_ceiling_ms": wl.ttft_ceiling_ms,
            "workload_spec": wl.spec(), "rows": rows,
        }
        path = args.out / f"{hardware_hash(hw)}__{spec.safe_id}__{wl_name}__isolation.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
