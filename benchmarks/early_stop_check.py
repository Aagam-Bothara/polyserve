"""What measuring concurrency levels highest first, and stopping early, does to recorded calibrations.

Every trial in a profile's calibration table recorded all its levels. This keeps, for each trial, only
the levels a top-down run measures (down to the first that meets the objective, objectives.enough_level),
re-ranks the table, and reports how many trial scores and picks change and how long the skipped levels
took to measure. Server start-up is not skipped, so the share of calibration time saved is smaller.

    PYTHONUTF8=1 python benchmarks/early_stop_check.py benchmarks/strategies/results-real/profiles/*.json \
        benchmarks/strategies/results-dolly/profiles/*.json benchmarks/strategies/results-real/sglang/profiles/*.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from polyserve.calibrate.objectives import Constraints, enough_level, rank
from polyserve.models import TrialResult


def _ceilings(profile: Dict[str, object]) -> Tuple[float, Optional[float]]:
    """The TTFT and TPOT ceilings the calibration ran under, from its workload spec or its workload note."""
    spec = profile.get("workload_spec") or {}
    ttft, tpot = spec.get("ttft_ceiling_ms"), spec.get("tpot_ceiling_ms")
    note = " ".join(n for n in profile.get("notes", []) if n.startswith("workload:"))
    if ttft is None:
        m = re.search(r"TTFT ceiling (\d+) ms", note)
        ttft = float(m.group(1)) if m else 500.0
    if tpot is None:
        m = re.search(r"TPOT ceiling (\d+) ms", note)
        tpot = float(m.group(1)) if m else None
    return float(ttft), tpot


def _top_down(trial: TrialResult, enough) -> Tuple[TrialResult, float]:
    """The trial as a top-down run would have recorded it, and the seconds its skipped levels took."""
    kept = {}
    for c, m in sorted(trial.metrics.by_concurrency.items(), key=lambda kv: -int(kv[0])):
        kept[c] = m
        if m.ok and enough(m):
            break
    skipped = sum(m.duration_s or 0.0 for c, m in trial.metrics.by_concurrency.items() if c not in kept)
    return trial.model_copy(update={"metrics": trial.metrics.model_copy(update={"by_concurrency": kept})}), skipped


def check(path: Path, percentile: int) -> Optional[Dict[str, object]]:
    p = json.loads(path.read_text(encoding="utf-8"))
    if "calibration_table" not in p or p.get("objective") not in ("throughput", "balanced"):
        return None
    ttft, tpot = _ceilings(p)
    cons = Constraints(ttft_ceiling_ms=ttft, tpot_ceiling_ms=tpot, ttft_percentile=percentile)
    enough = enough_level(p["objective"], cons)
    ok = [t for t in (TrialResult.model_validate(x) for x in p["calibration_table"]) if t.ok]
    if not ok:
        return None
    cut = [_top_down(t, enough) for t in ok]
    full = rank(ok, p["objective"], cons)
    short = rank([t for t, _ in cut], p["objective"], cons)
    before = {x.result.config.key(): (x.feasible, x.score) for x in full}
    after = {x.result.config.key(): (x.feasible, x.score) for x in short}
    return {
        "profile": path.name, "trials": len(ok), "changed": sum(before[k] != after.get(k) for k in before),
        "same_pick": full[0].result.config.key() == short[0].result.config.key(),
        "skipped_s": sum(s for _, s in cut),
        "measured_s": sum(m.duration_s or 0.0 for t in ok for m in t.metrics.by_concurrency.values()),
        "calibration_s": float(p.get("calibration_seconds") or 0.0),
    }


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("profiles", nargs="+", type=Path, help="calibration profiles (JSON with a calibration_table)")
    ap.add_argument("--ttft-percentile", type=int, default=95, choices=(50, 95))
    args = ap.parse_args(argv)
    rows = [r for r in (check(p, args.ttft_percentile) for p in sorted(set(args.profiles))) if r is not None]
    print("| profile | trials | scores changed | same pick | measuring time skipped | of calibration time |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        share = f"{r['skipped_s'] / r['calibration_s']:.0%}" if r["calibration_s"] else "-"
        print(f"| {r['profile']} | {r['trials']} | {r['changed']} | {'yes' if r['same_pick'] else 'NO'} | "
              f"{r['skipped_s'] / 60:.0f} of {r['measured_s'] / 60:.0f} min | {share} |")
    total = sum(r["calibration_s"] for r in rows)
    if total:
        print(f"\n{sum(r['changed'] for r in rows)} of {sum(r['trials'] for r in rows)} trial scores changed; "
              f"{sum(r['same_pick'] for r in rows)} of {len(rows)} picks kept; "
              f"{sum(r['skipped_s'] for r in rows) / total:.0%} of calibration time skipped")


if __name__ == "__main__":
    main()
