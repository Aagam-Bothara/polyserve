#!/usr/bin/env python
"""Measure what each search strategy is worth: PolyServe's pick with one strategy flipped at a time.

Reads the pick from `polyserve compare` results, so run those first. Then:

    python benchmarks/ablate_strategies.py --model Qwen/Qwen2.5-3B-Instruct \
        --workloads chat chat-system rag-shared generation high-concurrency --out benchmarks/ablation

Each variant flips one strategy (4-bit weights, KV-cache quantization, prefix caching,
speculative decoding): removed if the pick uses it, added if it does not. `--baselines` adds stock
vLLM at bf16 and fp8. `--spec-sweep 1 2 4 8 16 32 64` measures speculative decoding on and off
at each concurrency level, to find where it turns from a gain into a loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from polyserve.bench.ablation import (
    Variant, constraints_for, crossover, level_table, load_pick, markdown, score_row, strategy_variants,
    sweep_workload,
)
from polyserve.calibrate.search import SubprocessTrialRunner
from polyserve.calibrate.workload import get_workload
from polyserve.hardware import hardware_hash, probe
from polyserve.models import Config, ModelSpec
from polyserve.pipeline import prepare_and_plan, select
from polyserve.quantized import INT4_METHODS


def _spec_sweep(pick: Config, backend, model, reg, prepared, hw, wl, levels: List[int], log_dir) -> Dict[str, object]:
    sw = sweep_workload(wl, levels)
    runner = SubprocessTrialRunner(backends=reg, models=prepared, hw=hw, workload=sw, log_dir=log_dir)
    off = pick.model_copy(update={"spec_decode": None})
    print(f"[sweep] off: {off.key()} at c={list(sw.concurrencies)} ...", flush=True)
    off_levels = level_table(runner.run(off, "sweep:off").metrics)
    out: Dict[str, object] = {"levels": list(sw.concurrencies), "off": {"config_key": off.key(), "levels": off_levels},
                              "on": []}
    for v in ([pick] if pick.spec_decode else backend.spec_variants(off, model)):
        print(f"[sweep] on: {v.key()} ...", flush=True)
        res = runner.run(v, "sweep:on")
        lv = level_table(res.metrics)
        entry = {"config_key": v.key(), "spec": v.spec_decode, "ok": res.ok, "error": (res.error or "")[:2000],
                 "levels": lv, "tok_s_crossover": crossover(off_levels, lv, "tok_s"),
                 "latency_crossover": crossover(off_levels, lv, "e2e_ms")}
        out["on"].append(entry)  # type: ignore[union-attr]
        for c in sw.concurrencies:
            a, b = off_levels.get(str(c), {}), lv.get(str(c), {})
            print(f"[sweep] {v.spec_decode} c={c}: tok/s {a.get('tok_s', 0) or 0:.0f} -> {b.get('tok_s', 0) or 0:.0f}, "
                  f"latency {a.get('e2e_ms')} -> {b.get('e2e_ms')} ms", flush=True)
        print(f"[sweep] {v.spec_decode}: throughput stops gaining at c={entry['tok_s_crossover']}, "
              f"latency at c={entry['latency_crossover']}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workloads", nargs="+", default=["default"])
    ap.add_argument("--objective", default="balanced")
    ap.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--out", type=Path, default=Path("benchmarks/ablation"))
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--baselines", action="store_true", help="also measure stock vLLM at bf16 and fp8")
    ap.add_argument("--spec-sweep", type=int, nargs="*", default=None, metavar="C",
                    help="concurrency levels for a speculative-decoding on/off sweep")
    args = ap.parse_args()

    hw = probe()
    hw_hash = hardware_hash(hw)
    spec = ModelSpec(hf_id=args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for wl_name in args.workloads:
        wl = get_workload(wl_name)
        cons = constraints_for(wl)
        pick = load_pick(args.results, args.model, wl_name, args.objective, hw_hash)
        if pick is None:
            print(f"[{wl_name}] no polyserve row in {args.results}; run `polyserve compare` first", file=sys.stderr)
            failures += 1
            continue
        candidates, reg = select(hw, spec, force=pick.backend)
        if not candidates:
            print(f"[{wl_name}] {pick.backend} is not usable here", file=sys.stderr)
            failures += 1
            continue
        # 4-bit checkpoints included on purpose: what they are worth is one of the things measured.
        planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl,
                                   quants=["auto", *INT4_METHODS])
        backend, model = reg[pick.backend], planned.prepared[pick.backend]
        runner = SubprocessTrialRunner(backends=reg, models=planned.prepared, hw=hw, workload=wl, log_dir=args.log_dir)

        todo: List[Variant] = []
        if args.baselines and pick.backend == "vllm":
            ctx = model.arch.max_position_embeddings
            todo += [Variant("vllm-default", "baseline",
                             Config(backend="vllm", quant="bf16", ctx=ctx, batch=256, gpu_memory_utilization=0.90)),
                     Variant("vllm-fp8-default", "baseline",
                             Config(backend="vllm", quant="fp8", ctx=ctx, batch=256, gpu_memory_utilization=0.90))]
        todo.append(Variant("polyserve", "pick", pick))
        todo += strategy_variants(pick, backend, hw, model, wl)

        rows: List[Dict[str, object]] = []
        for v in todo:
            row: Dict[str, object] = {"label": v.label, "strategy": v.strategy, "note": v.note,
                                      "config": v.config.model_dump() if v.config else None,
                                      "config_key": v.config.key() if v.config else "-"}
            if v.config is None:
                row["ok"] = False
                print(f"[{wl_name}] {v.label}: skipped ({v.note})", flush=True)
            else:
                print(f"[{wl_name}] {v.label}: {v.config.key()} ...", flush=True)
                row.update(score_row(runner.run(v.config, f"ablate:{v.label}"), args.objective, cons))
                print(f"[{wl_name}] {v.label}: " + (f"{row['tok_s']:.1f} tok/s @c{row['concurrency']}" if row["ok"]
                                                    else f"FAILED {str(row.get('error'))[:300]}"), flush=True)
            rows.append(row)

        out: Dict[str, object] = {
            "hardware_hash": hw_hash, "gpu": hw.gpu.name if hw.gpu else hw.cpu.model_name, "model_id": args.model,
            "workload": wl_name, "objective": args.objective, "workload_spec": wl.spec(), "rows": rows,
        }
        if args.spec_sweep:
            out["spec_sweep"] = _spec_sweep(pick, backend, model, reg, planned.prepared, hw, wl, args.spec_sweep,
                                            args.log_dir)
        path = args.out / f"{hw_hash}__{spec.safe_id}__{wl_name}__ablation.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(markdown(rows), flush=True)
        print("wrote", path, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
