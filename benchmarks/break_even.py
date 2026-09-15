"""How long a PolyServe pick has to serve before its calibration has paid for itself.

Calibration occupies the GPU without serving. Against the fastest stock setup that met the objective's
limits in the same comparison, the pick's extra throughput repays that time after

    break-even hours = calibration seconds x stock tok/s / (PolyServe tok/s - stock tok/s) / 3600

of serving at the load the comparison scored. That prices throughput, for a server kept busy; at light
traffic a pick's value is its latency instead, which this does not count. A pick no faster than stock
never breaks even, and where no stock setup met the limits there is nothing to repay against.

    PYTHONUTF8=1 python benchmarks/break_even.py benchmarks/strategies/results benchmarks/strategies/results-l4/p95 ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def break_even(path: Path) -> Optional[Dict[str, object]]:
    """The break-even of one `polyserve compare` results file, or None when it has no calibrated pick."""
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["label"]: r for r in d.get("rows", [])}
    ps = rows.get("polyserve")
    if ps is None or not ps.get("ok") or not ps.get("calibration_seconds") or not ps.get("scored_tok_s"):
        return None
    run = path.parent  # a rerun kept in a folder inside a results folder (results-dolly/p95) is labelled
    workload = str(d.get("workload")) + (f" ({run.name})" if run.parent.name.startswith("results") else "")
    out: Dict[str, object] = {
        "gpu": str(d.get("gpu") or ("CPU" if d.get("cpu") else "?")).replace("NVIDIA ", ""),
        "model": str(d.get("model_id", "?")).split("/")[-1], "workload": workload,
        "calibration_s": float(ps["calibration_seconds"]), "tok_s": float(ps["scored_tok_s"]),
        "stock": None, "stock_tok_s": None, "hours": None,
    }
    stock = [r for label, r in rows.items()
             if label != "polyserve" and r.get("ok") and r.get("meets_slo") and r.get("scored_tok_s")]
    if stock:
        best = max(stock, key=lambda r: r["scored_tok_s"])
        gain = out["tok_s"] - float(best["scored_tok_s"])
        out.update(stock=best["label"], stock_tok_s=float(best["scored_tok_s"]),
                   hours=out["calibration_s"] * float(best["scored_tok_s"]) / gain / 3600 if gain > 0 else math.inf)
    return out


def _cell(r: Dict[str, object]) -> str:
    if r["stock"] is None:
        return "no stock setup met the limits"
    if r["hours"] == math.inf:
        return "never: no faster than stock"
    return f"{r['hours']:.1f} h"


def table(results: List[Dict[str, object]]) -> str:
    lines = ["| GPU | model | workload | calibration | PolyServe tok/s | fastest stock that met the limits | "
             "break-even |", "|---|---|---|---|---|---|---|"]
    for r in results:
        stock = f"{r['stock']} {r['stock_tok_s']:.0f}" if r["stock"] else "-"
        lines.append(f"| {r['gpu']} | {r['model']} | {r['workload']} | {r['calibration_s'] / 60:.0f} min | "
                     f"{r['tok_s']:.0f} | {stock} | {_cell(r)} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", type=Path, help="folders of `polyserve compare` results")
    args = ap.parse_args(argv)
    files = sorted({f for d in args.dirs for f in Path(d).glob("*.json")})
    print(table([r for r in (break_even(f) for f in files) if r is not None]))


if __name__ == "__main__":
    main()
