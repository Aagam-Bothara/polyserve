"""`polyserve compare`: PolyServe's calibrated pick vs stock defaults vs Ollama, one workload, one file.

Result files are the raw material for the benchmark matrix and `polyserve report`:

    benchmarks/results/<hardware_hash>__<model>__<workload>__<objective>.json
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from polyserve import __version__
from polyserve.backends.base import BaseBackend
from polyserve.bench.references import OllamaReference, ollama_tag_for, reference_configs
from polyserve.calibrate.objectives import Constraints
from polyserve.calibrate.search import SubprocessTrialRunner, TrialRunner
from polyserve.calibrate.workload import Workload, get_workload
from polyserve.hardware import hardware_hash
from polyserve.models import (
    Config,
    HardwareDescriptor,
    MemoryObservation,
    ModelSpec,
    PreparedModel,
    Profile,
    TrialMetrics,
)

logger = logging.getLogger(__name__)


class ComparisonRow(BaseModel):
    label: str  # "polyserve" | "vllm-default" | "sglang-default" | "llamacpp-cuda-default" | "ollama-default"
    runtime: str  # backend name
    runtime_version: Optional[str] = None
    config: Config
    config_key: str
    launch_args: List[str] = Field(default_factory=list)
    ok: bool = False
    error: Optional[str] = None
    metrics: TrialMetrics = Field(default_factory=TrialMetrics)
    memory: Optional[MemoryObservation] = None
    # Objective-relevant view: the concurrency level the objective would run this row at.
    scored_concurrency: Optional[int] = None
    scored_tok_s: Optional[float] = None
    scored_ttft_ms: Optional[float] = None
    scored_ttft_p95_ms: Optional[float] = None
    scored_tpot_ms: Optional[float] = None
    scored_joules_per_token: Optional[float] = None
    meets_slo: Optional[bool] = None
    calibration_seconds: float = 0.0
    calibration_trials: int = 0


class ComparisonResult(BaseModel):
    polyserve_version: str
    hardware_hash: str
    gpu: Optional[str]
    cpu: str
    model_id: str
    objective: str
    workload: str
    workload_spec: Dict[str, Any]
    ttft_ceiling_ms: float
    tpot_ceiling_ms: Optional[float] = None
    rows: List[ComparisonRow] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    notes: List[str] = Field(default_factory=list)

    @property
    def polyserve_row(self) -> Optional[ComparisonRow]:
        return next((r for r in self.rows if r.label == "polyserve"), None)

    def default_rows(self) -> List[ComparisonRow]:
        return [r for r in self.rows if r.label != "polyserve"]


def results_dir() -> Path:
    return Path(os.environ.get("POLYSERVE_RESULTS", "benchmarks/results"))


def results_path(hw_hash: str, spec: ModelSpec, workload: str, objective: str, out: Optional[Path] = None) -> Path:
    return (out or results_dir()) / f"{hw_hash}__{spec.safe_id}__{workload}__{objective}.json"


def _score_row(row: ComparisonRow, objective: str, cons: Constraints) -> None:
    """Fill the scored_* fields from the concurrency level the objective would pick."""
    from polyserve.calibrate.objectives import rank
    from polyserve.models import TrialResult

    if not row.ok:
        return
    tr = TrialResult(config=row.config, stage="compare", metrics=row.metrics)
    ranked = rank([tr], objective, cons)
    if not ranked:
        return
    m = ranked[0].metrics
    row.scored_concurrency = m.concurrency
    row.scored_tok_s = m.tok_s
    row.scored_ttft_ms = m.ttft_ms
    row.scored_ttft_p95_ms = m.ttft_p95_ms
    row.scored_tpot_ms = m.tpot_ms
    row.scored_joules_per_token = m.joules_per_token
    row.meets_slo = ranked[0].feasible


def _run_disagg_row(profile: Profile, backends: Dict[str, BaseBackend], models: Dict[str, PreparedModel],
                    hw: HardwareDescriptor, workload: Workload, log_dir: Optional[Path], power: Optional[object],
                    disagg_runner: Optional[object], objective: str, cons: Constraints,
                    progress: Optional[Callable[[str, Optional[ComparisonRow]], None]]) -> ComparisonRow:
    """Measure a disaggregated profile end to end: both engines plus the router."""
    from polyserve.disagg import DisaggTrialRunner

    spec = profile.disagg
    backend = backends[profile.backend]
    runner = disagg_runner or DisaggTrialRunner(backend, models[profile.backend], hw, workload, log_dir=log_dir,
                                                power_for_gpu=(lambda i: power) if power is not None else None)
    row = ComparisonRow(label="polyserve", runtime=f"{profile.backend}-disaggregated",
                        runtime_version=backend.version(hw), config=spec.decode, config_key=spec.key())
    if progress:
        progress("polyserve", None)
    tr = runner.run(spec, "compare:polyserve")  # type: ignore[attr-defined]
    row.ok, row.error, row.metrics = tr.ok, tr.error, tr.metrics
    _score_row(row, objective, cons)
    if progress:
        progress("polyserve", row)
    return row


def _run_replica_row(profile: Profile, backends: Dict[str, BaseBackend], models: Dict[str, PreparedModel],
                     hw: HardwareDescriptor, workload: Workload, log_dir: Optional[Path],
                     replica_runner: Optional[object], objective: str, cons: Constraints,
                     progress: Optional[Callable[[str, Optional[ComparisonRow]], None]]) -> ComparisonRow:
    """Measure a replicas profile end to end through the load balancer."""
    from polyserve.layout import LayoutTrialRunner

    backend = backends[profile.backend]
    gpus = [g.index for g in hw.gpus if g.vendor == "nvidia"][: profile.replicas]
    runner = replica_runner or LayoutTrialRunner(backend, models[profile.backend], hw, workload, log_dir=log_dir)
    row = ComparisonRow(label="polyserve", runtime=f"{profile.backend}-x{profile.replicas}",
                        runtime_version=backend.version(hw), config=profile.config,
                        config_key=f"{profile.config.key()} x{profile.replicas}")
    if progress:
        progress("polyserve", None)
    tr = runner.run_replicas(profile.config, gpus, "compare:polyserve")  # type: ignore[attr-defined]
    row.ok, row.error, row.metrics = tr.ok, tr.error, tr.metrics
    _score_row(row, objective, cons)
    if progress:
        progress("polyserve", row)
    return row


def compare(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    profile: Profile,
    prepared: Dict[str, PreparedModel],
    reg: Dict[str, BaseBackend],
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    runner: Optional[TrialRunner] = None,
    ollama_tag: Optional[str] = None,
    include: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
    progress: Optional[Callable[[str, Optional[ComparisonRow]], None]] = None,
    power: Optional[object] = None,
    disagg_runner: Optional[object] = None,
    replica_runner: Optional[object] = None,
) -> ComparisonResult:
    """Measure PolyServe's winner and every reference config under the same workload."""
    workload = workload or get_workload(profile.workload)
    cons = constraints or Constraints(ttft_ceiling_ms=workload.ttft_ceiling_ms,
                                      tpot_ceiling_ms=workload.tpot_ceiling_ms)
    objective = profile.objective

    backends: Dict[str, BaseBackend] = dict(reg)
    models: Dict[str, PreparedModel] = dict(prepared)
    if profile.prepared is not None:
        models[profile.backend] = profile.prepared

    # References: stock defaults for each candidate backend, plus Ollama when available.
    refs: Dict[str, Config] = reference_configs(hw, prepared, reg)
    tag = ollama_tag or ollama_tag_for(spec.hf_id)
    if tag:
        ollama = OllamaReference(tag)
        if ollama.available(hw):
            backends["ollama"] = ollama
            try:
                models["ollama"] = ollama.prepare(spec, hw)
                refs["ollama-default"] = ollama.default_config(hw, models["ollama"])
            except Exception as exc:
                logger.warning("ollama reference skipped: %s", exc)
        else:
            logger.info("ollama binary not found; skipping the ollama-default row")
    if include:
        refs = {k: v for k, v in refs.items() if k in include}

    if runner is None:
        runner = SubprocessTrialRunner(backends=backends, models=models, hw=hw, workload=workload,
                                       log_dir=log_dir, power=power)

    result = ComparisonResult(
        polyserve_version=__version__,
        hardware_hash=hardware_hash(hw),
        gpu=hw.gpu.name if hw.gpu else None,
        cpu=hw.cpu.model_name,
        model_id=spec.hf_id,
        objective=objective,
        workload=workload.name,
        workload_spec=workload.spec(),
        ttft_ceiling_ms=cons.ttft_ceiling_ms,
        tpot_ceiling_ms=cons.tpot_ceiling_ms,
    )

    def run_row(label: str, cfg: Config) -> ComparisonRow:
        backend = backends[cfg.backend]
        row = ComparisonRow(label=label, runtime=cfg.backend, runtime_version=backend.version(hw), config=cfg,
                            config_key=cfg.key())
        if progress:
            progress(label, None)
        try:
            row.launch_args = backend.launch_spec(cfg, models[cfg.backend], 0).args
        except Exception:
            pass
        tr = runner.run(cfg, f"compare:{label}")
        row.ok, row.error, row.metrics, row.memory = tr.ok, tr.error, tr.metrics, tr.memory
        _score_row(row, objective, cons)
        if progress:
            progress(label, row)
        return row

    # PolyServe's winner, re-measured now so it faces the same conditions as the references.
    if profile.disagg is not None:
        ps = _run_disagg_row(profile, backends, models, hw, workload, log_dir, power, disagg_runner, objective, cons,
                             progress)
    elif profile.replicas > 1:
        ps = _run_replica_row(profile, backends, models, hw, workload, log_dir, replica_runner, objective, cons,
                              progress)
        result.notes.append(f"PolyServe serves {profile.replicas} replicas on {profile.replicas} GPUs; the stock rows "
                            "use one GPU, so this row is not a like-for-like throughput comparison")
    else:
        ps = run_row("polyserve", profile.config)
    ps.calibration_seconds = profile.calibration_seconds
    ps.calibration_trials = profile.calibration_trials
    result.rows.append(ps)
    for label, cfg in refs.items():
        if cfg.backend not in models:
            continue
        result.rows.append(run_row(label, cfg))

    ps_ok = ps.ok and ps.meets_slo
    if not ps_ok:
        result.notes.append("polyserve row failed or missed the SLO in this run; inspect before reporting")
    return result


def save(result: ComparisonResult, out: Optional[Path] = None) -> Path:
    path = results_path(result.hardware_hash, ModelSpec(hf_id=result.model_id), result.workload, result.objective, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def _f(x: Optional[float], nd: int = 0) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def to_markdown(result: ComparisonResult) -> str:
    lines = [
        f"**{result.gpu or result.cpu}** · {result.model_id} · workload `{result.workload}` · objective "
        f"`{result.objective}` (TTFT ≤ {result.ttft_ceiling_ms:.0f} ms"
        f"{f', TPOT ≤ {result.tpot_ceiling_ms:.0f} ms' if result.tpot_ceiling_ms else ''})",
        "",
        "| runtime / config | tok/s | TTFT p50 | TTFT p95 | TPOT | peak mem | W | J/tok | SLO | calib |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result.rows:
        name = f"**PolyServe** → {r.config_key}" if r.label == "polyserve" else f"{r.label} ({r.config_key})"
        if not r.ok:
            lines.append(f"| {name} | failed: {(r.error or '').splitlines()[0][:60]} | | | | | | | | |")
            continue
        m = r.metrics
        calib = f"{r.calibration_seconds:.0f}s / {r.calibration_trials}" if r.label == "polyserve" else "-"
        mem = _f(m.peak_mem_mb / 1024, 1) if m.peak_mem_mb else "-"
        lines.append(
            f"| {name} | {_f(r.scored_tok_s)} @c{r.scored_concurrency} | {_f(r.scored_ttft_ms)} | "
            f"{_f(r.scored_ttft_p95_ms)} | {_f(r.scored_tpot_ms, 1)} | {mem} GB | {_f(m.power_w)} | "
            f"{_f(r.scored_joules_per_token, 3)} | {'yes' if r.meets_slo else 'no'} | {calib} |"
        )
    if result.notes:
        lines += [""] + [f"_{n}_" for n in result.notes]
    return "\n".join(lines)
