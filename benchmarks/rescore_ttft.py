"""Re-score recorded comparisons with the TTFT ceiling judged at p95 instead of the median.

Every comparison row stores each concurrency level's median and 95th-percentile time to first token,
so no GPU is needed: this ranks the same measurements under both rules and prints PolyServe's pick
and each stock row at both. The p50 columns reproduce the tables the runs printed. The PolyServe pick
itself is the one calibration chose by the median; a calibration run with the p95 rule may pick a
different configuration, which only a new run can show.

    PYTHONUTF8=1 python benchmarks/rescore_ttft.py benchmarks/strategies/results*/ \
        benchmarks/strategies/results-real/*/ benchmarks/strategies/results-dolly/*/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from polyserve.bench.ablation import score_row
from polyserve.calibrate.objectives import Constraints
from polyserve.models import Config, TrialMetrics, TrialResult

STOCK = (("stock vLLM bf16", ("vllm-default",)), ("stock vLLM fp8", ("vllm-fp8-default",)),
         ("stock SGLang", ("sglang-default",)), ("stock llama.cpp", ("llamacpp-cuda-default", "llamacpp-cpu-default")))


def _trial(row: Dict[str, object]) -> Optional[TrialResult]:
    if not row.get("ok") or not row.get("metrics"):
        return None
    return TrialResult(config=Config.model_validate(row["config"]), stage="compare",
                       metrics=TrialMetrics.model_validate(row["metrics"]))


def rescore(path: Path) -> Optional[Dict[str, object]]:
    """{workload, gpu, pick, rows: {label: {50: scored, 95: scored}}} for one results file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    if "rows" not in d or d.get("objective") != "balanced":
        return None
    out: Dict[str, object] = {"file": str(path), "workload": d.get("workload"), "ceiling": d.get("ttft_ceiling_ms"),
                              "gpu": str(d.get("gpu") or ("CPU" if d.get("cpu") else "?")).replace("NVIDIA ", ""),
                              "model": str(d.get("model_id", "?")).split("/")[-1], "rows": {}}
    for row in d["rows"]:
        t = _trial(row)
        if t is None:
            continue
        out["rows"][row["label"]] = {
            pct: score_row(t, "balanced", Constraints(ttft_ceiling_ms=float(d["ttft_ceiling_ms"]),
                                                      tpot_ceiling_ms=d.get("tpot_ceiling_ms"), ttft_percentile=pct))
            for pct in (50, 95)}
        if row["label"] == "polyserve":
            out["pick"] = row.get("config_key")
    return out


def _cell(ours: Dict[str, object], ref: Optional[Dict[str, object]]) -> str:
    if ref is None:
        return "-"
    gain = (float(ours["tok_s"]) / float(ref["tok_s"]) - 1) * 100 if ref["tok_s"] else 0.0
    return f"{ref['tok_s']:.0f} ({gain:+.0f}%)" + ("" if ref["meets_slo"] else ", misses")


def table(results: Sequence[Dict[str, object]]) -> str:
    lines = ["| GPU | model | workload | ceiling | rule | PolyServe tok/s | its TTFT p50 / p95 | "
             + " | ".join(name for name, _ in STOCK) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(STOCK)]
    for r in results:
        rows = r["rows"]
        if "polyserve" not in rows:
            continue
        for pct in (50, 95):
            ours = rows["polyserve"][pct]
            refs = [next((rows[lab][pct] for lab in labels if lab in rows), None) for _, labels in STOCK]
            lines.append(f"| {r['gpu']} | {r['model']} | {r['workload']} | {r['ceiling']:.0f} ms | p{pct} | "
                         f"{ours['tok_s']:.0f}{'' if ours['meets_slo'] else ', misses'} (at {ours['concurrency']}) | "
                         f"{ours['ttft_ms']:.0f} / {ours['ttft_p95_ms']:.0f} ms | "
                         + " | ".join(_cell(ours, ref) for ref in refs) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path, help="folders of `polyserve compare` results")
    args = ap.parse_args(argv)
    files = sorted({f for d in args.dirs for f in Path(d).glob("*.json")})
    results = [r for r in (rescore(f) for f in files) if r is not None]
    broke = [r for r in results if "polyserve" in r["rows"] and not r["rows"]["polyserve"][95]["meets_slo"]
             or ("polyserve" in r["rows"] and r["rows"]["polyserve"][95]["concurrency"]
                 != r["rows"]["polyserve"][50]["concurrency"])]
    print(table(results))
    print(f"\n{len(broke)} of {sum('polyserve' in r['rows'] for r in results)} picks scored at a lower load "
          f"or missed the ceiling once judged at p95: " + ", ".join(f"{r['gpu']} {r['model']} {r['workload']}"
                                                                   for r in broke))


if __name__ == "__main__":
    main()
