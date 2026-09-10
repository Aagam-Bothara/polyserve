"""Memory-planner calibration: predicted vs measured, per backend, with recommended constants.

Every trial records the planner's estimate (weights + kv + workspace, before the safety margin) and
what the backend actually used (NVML peak, plus the backend's own weight / KV / workspace figures
from its log). This module turns those pairs into:

  * prediction error per backend: mean absolute % error, signed bias, worst under-prediction;
  * a fitted runtime_workspace per backend (p95 of measured workspace), replacing the hand-set
    constant;
  * a recommended safety margin: the worst under-prediction plus a buffer, floored at 512 MB.

`polyserve memory-report --apply` writes the fitted constants to ~/.polyserve/memory-model.json,
keyed by hardware hash, and the planner uses them on the next run.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from polyserve.models import MiB, TrialResult

WORKSPACE_PERCENTILE = 0.95
MARGIN_BUFFER = 0.02  # added on top of the worst observed under-prediction
MIN_MARGIN_FRACTION = 0.02
MAX_MARGIN_FRACTION = 0.15


def model_path() -> Path:
    return Path(os.environ.get("POLYSERVE_HOME", Path.home() / ".polyserve")) / "memory-model.json"


# --------------------------------------------------------------------------- observations


@dataclass
class Observation:
    backend: str
    config_key: str
    hardware_hash: str
    predicted_mb: float  # weights + kv + workspace (no margin)
    predicted_weights_mb: float
    predicted_kv_mb: float
    predicted_workspace_mb: float
    measured_mb: Optional[float]  # device peak (or log total)
    measured_weights_mb: Optional[float]
    measured_kv_mb: Optional[float]
    measured_workspace_mb: Optional[float]
    launched: bool
    oom: bool
    source: str

    @property
    def error_pct(self) -> Optional[float]:
        """(predicted - measured) / measured; negative = under-prediction (dangerous)."""
        if self.measured_mb is None or self.measured_mb <= 0:
            return None
        return (self.predicted_mb - self.measured_mb) / self.measured_mb * 100.0


def _is_oom(error: Optional[str]) -> bool:
    if not error:
        return False
    e = error.lower()
    return "out of memory" in e or "oom" in e or "cuda error: out of memory" in e or "failed to allocate" in e


def observations_from_trials(trials: Iterable[TrialResult], hardware_hash: str) -> List[Observation]:
    out: List[Observation] = []
    for t in trials:
        if t.memory is None or t.memory.predicted is None:
            continue
        p = t.memory.predicted
        m = t.memory.measured
        out.append(
            Observation(
                backend=t.config.backend,
                config_key=t.config.key(),
                hardware_hash=hardware_hash,
                predicted_mb=(p.weights + p.kv_cache + p.runtime_workspace) / MiB,
                predicted_weights_mb=p.weights / MiB,
                predicted_kv_mb=p.kv_cache / MiB,
                predicted_workspace_mb=p.runtime_workspace / MiB,
                measured_mb=m.total_mb if m else None,
                measured_weights_mb=m.weights_mb if m else None,
                measured_kv_mb=m.kv_mb if m else None,
                measured_workspace_mb=m.workspace_mb if m else None,
                launched=t.launched,
                oom=_is_oom(t.error),
                source=m.source if m else "none",
            )
        )
    return out


# --------------------------------------------------------------------------- analysis


@dataclass
class BackendCalibration:
    backend: str
    n: int = 0
    n_measured: int = 0
    mape_pct: Optional[float] = None
    bias_pct: Optional[float] = None  # mean signed error; negative = planner under-predicts
    worst_under_pct: Optional[float] = None  # most negative error (0 if never under)
    worst_over_pct: Optional[float] = None
    weights_mape_pct: Optional[float] = None
    kv_mape_pct: Optional[float] = None
    fitted_workspace_mb: Optional[float] = None  # p95 of measured workspace
    current_workspace_mb: Optional[float] = None
    ooms: int = 0
    ooms_predicted_feasible: int = 0  # OOMs the planner did not foresee = planner failures
    recommended_margin_fraction: Optional[float] = None
    observations: List[Observation] = field(default_factory=list)


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _mape(pairs: List[tuple]) -> Optional[float]:
    errs = [abs(p - m) / m * 100 for p, m in pairs if m]
    return statistics.fmean(errs) if errs else None


def analyse(observations: List[Observation]) -> Dict[str, BackendCalibration]:
    by: Dict[str, List[Observation]] = {}
    for o in observations:
        by.setdefault(o.backend, []).append(o)
    out: Dict[str, BackendCalibration] = {}
    for backend, obs in by.items():
        cal = BackendCalibration(backend=backend, n=len(obs), observations=obs)
        errs = [o.error_pct for o in obs if o.error_pct is not None]
        cal.n_measured = len(errs)
        if errs:
            cal.mape_pct = statistics.fmean(abs(e) for e in errs)
            cal.bias_pct = statistics.fmean(errs)
            cal.worst_under_pct = min(0.0, min(errs))
            cal.worst_over_pct = max(0.0, max(errs))
        cal.weights_mape_pct = _mape([(o.predicted_weights_mb, o.measured_weights_mb) for o in obs
                                      if o.measured_weights_mb])
        cal.kv_mape_pct = _mape([(o.predicted_kv_mb, o.measured_kv_mb) for o in obs if o.measured_kv_mb])
        ws = [o.measured_workspace_mb for o in obs if o.measured_workspace_mb is not None]
        if ws:
            cal.fitted_workspace_mb = _pct(ws, WORKSPACE_PERCENTILE)
        cal.current_workspace_mb = obs[0].predicted_workspace_mb
        cal.ooms = sum(1 for o in obs if o.oom)
        cal.ooms_predicted_feasible = cal.ooms  # every observed trial was planner-feasible by construction
        if cal.worst_under_pct is not None:
            rec = -cal.worst_under_pct / 100.0 + MARGIN_BUFFER
            cal.recommended_margin_fraction = min(MAX_MARGIN_FRACTION, max(MIN_MARGIN_FRACTION, rec))
        out[backend] = cal
    return out


# --------------------------------------------------------------------------- overrides used by the planner


def load_overrides(hardware_hash: str) -> Dict[str, Dict[str, float]]:
    """{backend: {"workspace_bytes": int, "margin_fraction": float, "n": int}} for this machine."""
    p = model_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get(hardware_hash, {})


def save_overrides(hardware_hash: str, cals: Dict[str, BackendCalibration]) -> Path:
    p = model_path()
    data: Dict[str, Dict] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    entry = data.setdefault(hardware_hash, {})
    for backend, cal in cals.items():
        if cal.fitted_workspace_mb is None and cal.recommended_margin_fraction is None:
            continue
        rec: Dict[str, float] = {"n": cal.n_measured}
        if cal.fitted_workspace_mb is not None:
            rec["workspace_bytes"] = int(cal.fitted_workspace_mb * MiB)
        if cal.recommended_margin_fraction is not None:
            rec["margin_fraction"] = round(cal.recommended_margin_fraction, 4)
        entry[backend] = rec
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def workspace_override(hardware_hash: str, backend: str) -> Optional[int]:
    rec = load_overrides(hardware_hash).get(backend)
    return int(rec["workspace_bytes"]) if rec and "workspace_bytes" in rec else None


def margin_override(hardware_hash: str, backend: str) -> Optional[float]:
    rec = load_overrides(hardware_hash).get(backend)
    return float(rec["margin_fraction"]) if rec and "margin_fraction" in rec else None


# --------------------------------------------------------------------------- report


def _f(x: Optional[float], nd: int = 1, sign: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_markdown(cals: Dict[str, BackendCalibration], hardware: Optional[str] = None) -> str:
    lines = ["## Memory planner accuracy" + (f" ({hardware})" if hardware else ""), ""]
    if not cals:
        return "\n".join(lines + ["No trials with memory observations yet.", ""])
    lines += [
        "| backend | trials | measured | MAPE | bias | worst under | worst over | weights MAPE | KV MAPE | "
        "workspace now → fitted (p95) | OOMs | margin now → recommended |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b, c in sorted(cals.items()):
        ws = f"{_f(c.current_workspace_mb, 0)} → {_f(c.fitted_workspace_mb, 0)} MB"
        lines.append(
            f"| {b} | {c.n} | {c.n_measured} | {_f(c.mape_pct)}% | {_f(c.bias_pct, 1, True)}% | "
            f"{_f(c.worst_under_pct, 1, True)}% | {_f(c.worst_over_pct, 1, True)}% | {_f(c.weights_mape_pct)}% | "
            f"{_f(c.kv_mape_pct)}% | {ws} | {c.ooms} | 5.0% → {_f((c.recommended_margin_fraction or 0) * 100)}% |"
        )
    total = sum(c.n for c in cals.values())
    ooms = sum(c.ooms for c in cals.values())
    measured = [o for c in cals.values() for o in c.observations if o.error_pct is not None]
    if measured:
        worst = min(o.error_pct for o in measured)
        mape = statistics.fmean(abs(o.error_pct) for o in measured)
        lines += [
            "",
            f"Across {len(measured)} measured trials ({total} planned) the planner predicts peak device memory "
            f"within {mape:.1f}% on average; worst under-prediction {worst:+.1f}%; {ooms} OOM"
            f"{'s' if ooms != 1 else ''} among planner-feasible configs.",
        ]
    lines += [
        "",
        "MAPE = mean |predicted − measured| / measured over weights + KV + workspace (margin excluded). "
        "Negative bias means the planner under-predicts; the recommended margin is the worst under-prediction "
        "plus 2%, clamped to [2%, 15%], with the 512 MB floor unchanged.",
        "",
    ]
    return "\n".join(lines)
