#!/usr/bin/env python
"""Launch every search feature once on the real engines, to catch launch failures before a long run.

    python benchmarks/smoke_features.py --model Qwen/Qwen2.5-3B-Instruct --backends vllm llamacpp-cuda

Each feature is one short trial (4 prompts sharing a prefix, concurrency 1 and 4). Prints OK with
the numbers, or the error and the tail of the engine log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from polyserve.calibrate.search import SubprocessTrialRunner
from polyserve.calibrate.workload import Workload
from polyserve.hardware import probe
from polyserve.models import Config, ModelSpec
from polyserve.pipeline import prepare_and_plan, select
from polyserve.quantized import INT4_METHODS


def _base(configs: List[Config], llamacpp: bool) -> Optional[Config]:
    if not configs:
        return None
    if llamacpp:
        pool = [c for c in configs if c.quant == "Q4_K_M"] or configs
        return min(pool, key=lambda c: (abs(c.batch - 4), abs(c.ctx - 8192)))
    pool = [c for c in configs if c.quant == "bf16"] or configs
    return min(pool, key=lambda c: (abs(c.batch - 16), abs(c.ctx - 4096), -(c.gpu_memory_utilization or 0)))


def _features(name: str, base: Config, backend, model, hw) -> List[Tuple[str, Optional[Config]]]:
    out: List[Tuple[str, Optional[Config]]] = [("base", base)]
    if not name.startswith("llamacpp"):
        for q in ("fp8", *INT4_METHODS):
            out.append((f"quant {q}", base.model_copy(update={"quant": q}) if q in model.weights_bytes else None))
    for kd in backend.kv_dtypes(hw):
        out.append((f"kv {kd}", base.model_copy(update={"kv_dtype": kd})))
    prefix = backend.prefix_variants(base)
    for v in prefix:
        out.append((f"prefix {','.join(k for k in v.extra if k not in base.extra)}", v))
    if not prefix:
        out.append(("prefix off", base.model_copy(update={"prefix_cache": False})))
    spec = backend.spec_variants(base, model)
    out += [(f"spec {v.spec_decode}", v) for v in spec]
    if not spec:
        out.append(("spec", None))
    return out


def _log_tail(log_dir: Path, lines: int = 15) -> str:
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    return "\n".join(logs[-1].read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--backends", nargs="+", default=["vllm", "llamacpp-cuda"])
    ap.add_argument("--log-dir", type=Path, default=Path("smoke-logs"))
    ap.add_argument("--only", nargs="*", default=None, help="run only features whose label contains one of these")
    args = ap.parse_args()

    hw = probe()
    spec = ModelSpec(hf_id=args.model)
    wl = Workload(name="smoke", n_prompts=4, prefill_tokens=256, shared_prefix_tokens=128, decode_tokens=32,
                  concurrencies=(1, 4))
    failed = 0
    for name in args.backends:
        candidates, reg = select(hw, spec, force=name)
        if not candidates:
            print(f"{name}: not usable here", flush=True)
            failed += 1
            continue
        planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl,
                                   quants=["auto", *INT4_METHODS])  # smoke-test 4-bit too
        if name in planned.errors:
            print(f"{name}: {planned.errors[name]}", flush=True)
        model = planned.prepared.get(name)
        base = _base([c for c, _ in planned.feasible.get(name, [])], name.startswith("llamacpp"))
        if model is None or base is None:
            print(f"{name}: nothing feasible to launch", flush=True)
            failed += 1
            continue
        backend = reg[name]
        log_dir = args.log_dir / name
        runner = SubprocessTrialRunner(backends=reg, models=planned.prepared, hw=hw, workload=wl, log_dir=log_dir)
        print(f"== {name}: base {base.key()}; drafts {model.draft_paths or '-'}; 4-bit repos {model.hf_paths or '-'}",
              flush=True)
        for label, cfg in _features(name, base, backend, model, hw):
            if args.only and not any(o in label for o in args.only):
                continue
            if cfg is None:
                print(f"SKIP {name:14s} {label:40s} not available (no checkpoint or draft resolved)", flush=True)
                continue
            res = runner.run(cfg, f"smoke:{label}")
            m = res.metrics
            if res.ok:
                print(f"OK   {name:14s} {label:40s} {m.tok_s:7.1f} tok/s  TTFT {m.ttft_ms or 0:6.0f} ms  "
                      f"TPOT {m.tpot_ms or 0:5.1f} ms  peak {m.peak_mem_mb or 0:6.0f} MB", flush=True)
            else:
                failed += 1
                print(f"FAIL {name:14s} {label:40s} {(res.error or '').splitlines()[0][:200] if res.error else ''}",
                      flush=True)
                print("     " + _log_tail(log_dir).replace("\n", "\n     "), flush=True)
    print(f"SMOKE DONE, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
