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

# Backends that reserve a fixed share of the device up front and size the KV pool to fill it.
# For these, peak device memory is a policy choice (gpu_memory_utilization x VRAM), not a
# requirement, so the planner is scored on non-KV memory (weights + workspace) instead.
RESERVATION_BACKENDS = {"vllm", "sglang", "vllm-cpu"}

WORKSPACE_PERCENTILE = 0.95
MARGIN_BUFFER = 0.02  # added on top of the worst observed under-prediction
MIN_MARGIN_FRACTION = 0.02
MAX_MARGIN_FRACTION = 0.15


def model_path() -> Path:
    return Path(os.environ.get("POLYSERVE_HOME", Path.home() / ".polyserve")) / "memory-model.json"


# --------------------------------------------------------------------------- observations


def target_for(backend: str) -> str:
    """What the planner is scored against: 'non_kv' for reservation backends, else 'total'."""
    return "non_kv" if backend in RESERVATION_BACKENDS else "total"


@dataclass
class Observation:
    backend: str
    config_key: str
    hardware_hash: str
    target: str = "total"  # "total" (peak device memory) | "non_kv" (weights + workspace)
    predicted_mb: float = 0.0  # weights + kv + workspace (no margin)
    predicted_weights_mb: float = 0.0
    predicted_kv_mb: float = 0.0
    predicted_workspace_mb: float = 0.0
    measured_mb: Optional[float] = None  # device peak (or log total)
    measured_weights_mb: Optional[float] = None
    measured_kv_mb: Optional[float] = None
    measured_workspace_mb: Optional[float] = None
    launched: bool = True
    oom: bool = False
    source: str = "none"

    @property
    def predicted_target_mb(self) -> float:
        if self.target == "non_kv":
            return self.predicted_weights_mb + self.predicted_workspace_mb
        return self.predicted_mb

    @property
    def measured_target_mb(self) -> Optional[float]:
        if self.target == "non_kv":
            if self.measured_weights_mb is not None and self.measured_workspace_mb is not None:
                return self.measured_weights_mb + self.measured_workspace_mb
            if self.measured_mb is not None and self.measured_kv_mb is not None:
                return max(0.0, self.measured_mb - self.measured_kv_mb)
            return None
        return self.measured_mb

    @property
    def measured_workspace_effective(self) -> Optional[float]:
        """Measured workspace, or the residual for backends that allocate weights and KV exactly.

        llama.cpp allocates exactly the GGUF it is given and exactly ctx x n_parallel of KV, so
        peak - weights - kv is the compute-buffer workspace even when the build prints no
        per-buffer lines. Reservation backends size KV elastically, so no residual is inferred.
        """
        if self.measured_workspace_mb is not None:
            return self.measured_workspace_mb
        if self.target == "total" and self.measured_mb is not None:
            residual = self.measured_mb - self.predicted_weights_mb - self.predicted_kv_mb
            return residual if residual > 0 else None
        return None

    @property
    def kv_headroom(self) -> Optional[float]:
        """Measured KV pool / KV the planner budgeted. >1 means the pool was bigger than assumed."""
        if self.measured_kv_mb and self.predicted_kv_mb > 0:
            return self.measured_kv_mb / self.predicted_kv_mb
        return None

    @property
    def error_pct(self) -> Optional[float]:
        """(predicted - measured) / measured on the target quantity; negative = under-prediction."""
        m = self.measured_target_mb
        if m is None or m <= 0:
            return None
        return (self.predicted_target_mb - m) / m * 100.0


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
        if p.weights <= 0:
            continue  # reference runtimes we do not plan for (e.g. Ollama picks its own quant)
        out.append(
            Observation(
                backend=t.config.backend,
                config_key=t.config.key(),
                hardware_hash=hardware_hash,
                target=target_for(t.config.backend),
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
    target: str = "total"
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
    kv_headroom_median: Optional[float] = None
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
        cal = BackendCalibration(backend=backend, target=target_for(backend), n=len(obs), observations=obs)
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
        ws = [o.measured_workspace_effective for o in obs if o.measured_workspace_effective is not None]
        if ws:
            cal.fitted_workspace_mb = _pct(ws, WORKSPACE_PERCENTILE)
        heads = [o.kv_headroom for o in obs if o.kv_headroom is not None]
        if heads:
            cal.kv_headroom_median = statistics.median(heads)
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
        "| backend | scored on | trials | measured | MAPE | bias | worst under | worst over | weights MAPE | "
        "workspace now → fitted (p95) | KV pool vs budget | OOMs | margin now → recommended |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for b, c in sorted(cals.items()):
        ws = f"{_f(c.current_workspace_mb, 0)} → {_f(c.fitted_workspace_mb, 0)} MB"
        head = f"{_f(c.kv_headroom_median, 1)}x" if c.kv_headroom_median else "-"
        target = "weights+workspace" if c.target == "non_kv" else "peak device"
        lines.append(
            f"| {b} | {target} | {c.n} | {c.n_measured} | {_f(c.mape_pct)}% | {_f(c.bias_pct, 1, True)}% | "
            f"{_f(c.worst_under_pct, 1, True)}% | {_f(c.worst_over_pct, 1, True)}% | {_f(c.weights_mape_pct)}% | "
            f"{ws} | {head} | {c.ooms} | 5.0% → {_f((c.recommended_margin_fraction or 0) * 100)}% |"
        )
    total = sum(c.n for c in cals.values())
    ooms = sum(c.ooms for c in cals.values())
    measured = [o for c in cals.values() for o in c.observations if o.error_pct is not None]
    if measured:
        worst = min(o.error_pct for o in measured)
        mape = statistics.fmean(abs(o.error_pct) for o in measured)
        lines += [
            "",
            f"Across {len(measured)} measured trials ({total} planned) the planner predicts its target quantity "
            f"within {mape:.1f}% on average; worst under-prediction {worst:+.1f}%; {ooms} OOM"
            f"{'s' if ooms != 1 else ''} among planner-feasible configs.",
        ]
    lines += [
        "",
        "Reservation backends (vLLM, SGLang) size their KV pool to fill `gpu_memory_utilization x VRAM`, so their "
        "peak is a policy choice, not a requirement: they are scored on weights + workspace, and the KV column "
        "shows how much larger the pool they allocated was than the planner budgeted. llama.cpp allocates exactly "
        "what it is asked for, so it is scored on peak device memory. Negative bias means the planner "
        "under-predicts; the recommended margin is the worst under-prediction plus 2%, clamped to [2%, 15%], with "
        "the 512 MB floor unchanged.",
        "",
    ]
    return "\n".join(lines)
