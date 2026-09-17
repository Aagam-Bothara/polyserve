#!/usr/bin/env python
"""Is the staged search worth its time? Compare its pick with a rule of thumb and with random search given the
same time, all re-measured side by side.

    python benchmarks/search_baselines.py meta-llama/Llama-3.1-8B-Instruct --workload chat \
        --workload-file dolly.jsonl --eval-workload-file oasst.jsonl --seeds 0 1 --repeats 3 --out results-baselines

1. PolyServe calibrates as `polyserve compare` would (or reuses its cached profile); its calibration time is the
   budget unless --budget is given.
2. The rule of thumb is one configuration chosen without measuring (polyserve.bench.baselines.rule_of_thumb).
3. Random search draws from the same space calibration explores, one run per seed, each until the budget is spent,
   with the same measurement rules as calibration (busiest level first, close calls measured again).
4. Every pick, and stock vLLM and SGLang, is measured with --repeats interleaved runs, on the held-out prompts when
   --eval-workload-file is given. The comparison is saved like `polyserve compare`'s, next to a summary of each
   search (space size, trials, seconds, pick).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from polyserve.bench import compare, to_markdown
from polyserve.bench.baselines import random_search, rule_of_thumb, search_space
from polyserve.bench.compare import save
from polyserve.calibrate.objectives import Constraints, close_call_level
from polyserve.calibrate.search import SubprocessTrialRunner
from polyserve.calibrate.workload import get_workload, workload_from_file
from polyserve.hardware import probe
from polyserve.models import Config, ModelSpec
from polyserve.pipeline import SearchOptions, level_rule, prepare_and_plan, resolve_profile, select


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--workload", default="chat")
    ap.add_argument("--workload-file", type=Path, default=None)
    ap.add_argument("--eval-workload-file", type=Path, default=None, help="measure every row on these prompts")
    ap.add_argument("--objective", default="balanced")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--budget", type=float, default=None, help="seconds per random search (default: PolyServe's)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--include", nargs="*", default=["vllm-default", "sglang-default"], help="stock rows to keep")
    ap.add_argument("--out", type=Path, default=Path("results-baselines"))
    ap.add_argument("--log-dir", type=Path, default=Path("/workspace/logs/baselines"))
    args = ap.parse_args(argv)

    template = get_workload(args.workload)
    wl = workload_from_file(args.workload_file, template=template) if args.workload_file else template
    eval_wl = workload_from_file(args.eval_workload_file, template=template) if args.eval_workload_file else wl
    cons = Constraints(ttft_ceiling_ms=wl.ttft_ceiling_ms, tpot_ceiling_ms=wl.tpot_ceiling_ms)
    hw, spec = probe(), ModelSpec(hf_id=args.model)

    profile = resolve_profile(spec, args.objective, workload=wl, constraints=cons, hw=hw,
                              on_stage=lambda s: print(f"-> {s}", flush=True))
    budget = args.budget or float(profile.calibration_seconds or 0)
    print(f"PolyServe: {profile.config.key()} after {profile.calibration_trials} trials in {budget:.0f}s", flush=True)

    candidates, reg = select(hw, spec)
    planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl)
    space = search_space(hw, planned, reg, wl)
    runner = SubprocessTrialRunner(backends=reg, models=planned.prepared, hw=hw, workload=wl, log_dir=args.log_dir,
                                   enough=level_rule(args.objective, cons, SearchOptions()),
                                   close_call=close_call_level(args.objective, cons))
    extra: Dict[str, Config] = {}
    summary: Dict[str, object] = {"model": args.model, "workload": wl.name, "eval_workload": eval_wl.name,
                                  "space": len(space), "budget_s": budget, "polyserve": profile.config.key(),
                                  "polyserve_trials": profile.calibration_trials, "searches": []}
    rule = rule_of_thumb(hw, planned, reg)
    if rule is not None:
        extra["rule-of-thumb"] = rule
        summary["rule_of_thumb"] = rule.key()
    for seed in args.seeds:
        run = random_search(space, runner, args.objective, cons, budget, seed=seed)
        print(f"random search, seed {seed}: {len(run.results)} of {len(space)} configurations in {run.seconds:.0f}s; "
              f"pick {run.winner.config.key() if run.winner else '-'}", flush=True)
        # Keep the per-draw series, not just the totals: comparing how fast each search reaches its own answer
        # needs best-so-far against elapsed time, and a run that saved only aggregates cannot be re-analysed.
        t0 = min((r.started_at for r in run.results), default=0.0)
        series, best_so_far = [], 0.0
        for i, r in enumerate(sorted(run.results, key=lambda r: r.started_at), 1):
            if r.ok:
                best_so_far = max(best_so_far, r.metrics.tok_s)
            series.append({"trial": i, "elapsed_s": round(r.started_at - t0, 1), "config": r.config.key(),
                           "tok_s": round(r.metrics.tok_s, 1) if r.ok else None,
                           "best_so_far": round(best_so_far, 1)})
        summary["searches"].append({"seed": seed, "trials": len(run.results), "seconds": run.seconds,
                                    "untried": run.untried, "pick": run.winner.config.key() if run.winner else None,
                                    "series": series})
        if run.winner is not None:
            extra[f"random-search-seed{seed}"] = run.winner.config

    result = compare(hw, spec, profile, planned.prepared, reg, workload=eval_wl, constraints=cons,
                     include=args.include or None, extra=extra, repeats=args.repeats, log_dir=args.log_dir / "compare")
    result.notes.append(f"random search: {len(space)} configurations in the space, "
                        + "; ".join(f"seed {s['seed']} measured {s['trials']} in {s['seconds']:.0f}s"
                                    for s in summary["searches"])
                        + f", against PolyServe's {profile.calibration_trials} trials in {budget:.0f}s")
    path = save(result, args.out)
    (args.out / f"{path.stem}__search.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(to_markdown(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
