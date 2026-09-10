"""Orchestration: probe -> select -> prepare -> plan -> calibrate -> cache -> serve."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from polyserve import __version__
from polyserve import cache as profile_cache
from polyserve.backends import BaseBackend, registry as backend_registry
from polyserve.calibrate.objectives import Constraints
from polyserve.calibrate.search import ProgressFn, StagedSearch, SubprocessTrialRunner, TrialRunner
from polyserve.calibrate.workload import Workload
from polyserve.hardware import hardware_hash, llmtrace_version, probe
from polyserve.memory import plan as memory_plan
from polyserve.models import Config, HardwareDescriptor, MemoryEstimate, ModelSpec, PreparedModel, Profile
from polyserve.selector import select_backends

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    hw: HardwareDescriptor
    spec: ModelSpec
    candidates: List[str]
    prepared: Dict[str, PreparedModel] = field(default_factory=dict)
    feasible: Dict[str, List[Tuple[Config, MemoryEstimate]]] = field(default_factory=dict)
    considered: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)

    @property
    def all_feasible(self) -> List[Config]:
        return [c for cfgs in self.feasible.values() for c, _ in cfgs]

    @property
    def total_considered(self) -> int:
        return sum(self.considered.values())


def select(
    hw: HardwareDescriptor, spec: ModelSpec, force: Optional[str] = None
) -> Tuple[List[str], Dict[str, BaseBackend]]:
    reg = backend_registry()
    return select_backends(hw, spec, reg, force=force), reg


def prepare_and_plan(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    candidates: List[str],
    reg: Dict[str, BaseBackend],
    materialize: bool = False,
) -> PlanResult:
    """Prepare each candidate backend's model, enumerate its grid, keep what fits."""
    result = PlanResult(hw=hw, spec=spec, candidates=list(candidates))
    for name in candidates:
        backend = reg[name]
        try:
            prepared = backend.prepare(spec, hw)
            grid = backend.candidate_configs(hw, prepared)
            kept = memory_plan(hw, prepared, grid, backend.memory_model(hw))
            result.prepared[name] = prepared
            result.feasible[name] = kept
            result.considered[name] = len(grid)
            logger.info("%s: %d/%d configs feasible", name, len(kept), len(grid))
            if materialize and kept:
                quants = sorted({c.quant for c, _ in kept})
                backend.materialize(prepared, quants)
                # Re-plan with true file sizes (GGUF sizes are estimates until downloaded).
                result.feasible[name] = memory_plan(hw, prepared, grid, backend.memory_model(hw))
        except Exception as exc:
            logger.exception("%s: prepare/plan failed", name)
            result.errors[name] = str(exc)
    return result


def _profile_for(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    objective: str,
    backend: BaseBackend,
    cfg: Config,
    prepared: PreparedModel,
    table=None,
    notes: Optional[List[str]] = None,
) -> Profile:
    launch = backend.launch_spec(cfg, prepared, 0)
    return Profile(
        polyserve_version=__version__,
        hardware_hash=hardware_hash(hw),
        hardware=hw,
        model_id=spec.hf_id,
        objective=objective,
        backend=backend.name,
        backend_version=backend.version(hw),
        config=cfg,
        prepared=prepared,
        launch_args=launch.args,
        launch_env=launch.env,
        calibration_table=list(table or []),
        llmtrace_version=llmtrace_version(),
        notes=list(notes or []),
    )


def default_profile(hw: HardwareDescriptor, spec: ModelSpec, plan: PlanResult, reg: Dict[str, BaseBackend],
                    objective: str = "balanced") -> Profile:
    """Uncalibrated: first candidate backend with its default config (or the largest feasible one)."""
    for name in plan.candidates:
        prepared = plan.prepared.get(name)
        if prepared is None:
            continue
        backend = reg[name]
        cfg = backend.default_config(hw, prepared)
        feasible = plan.feasible.get(name) or []
        if feasible and not any(c.key() == cfg.key() for c, _ in feasible):
            # Default does not fit; take the feasible config with the most headroom for ctx/batch.
            cfg = max(feasible, key=lambda ce: (ce[0].ctx, ce[0].batch))[0]
        if cfg.quant not in prepared.weights_bytes:
            cfg.quant = next(iter(prepared.weights_bytes))
        backend.materialize(prepared, [cfg.quant])
        return _profile_for(hw, spec, objective, backend, cfg, prepared, notes=["uncalibrated defaults"])
    raise RuntimeError(f"no usable backend for {spec.hf_id}; errors: {plan.errors}")


def calibrate(
    hw: HardwareDescriptor,
    spec: ModelSpec,
    objective: str,
    plan: PlanResult,
    reg: Dict[str, BaseBackend],
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    progress: Optional[ProgressFn] = None,
    runner: Optional[TrialRunner] = None,
    log_dir: Optional[Path] = None,
) -> Profile:
    workload = workload or Workload()
    constraints = constraints or Constraints()
    feasible = plan.all_feasible
    if not feasible:
        raise RuntimeError("memory planner left no feasible configuration; try a smaller model or quant")
    if runner is None:
        runner = SubprocessTrialRunner(
            backends={n: reg[n] for n in plan.candidates},
            models=plan.prepared,
            hw=hw,
            workload=workload,
            log_dir=log_dir or (profile_cache.logs_dir() / spec.safe_id),
        )
    search = StagedSearch(objective=objective, runner=runner, constraints=constraints, progress=progress)
    t0 = time.monotonic()
    winner, notes = search.run(feasible)
    if winner is None:
        raise RuntimeError("calibration failed: " + "; ".join(notes))
    notes.append(f"calibration took {time.monotonic() - t0:.0f}s over {len(search.results)} trials")
    notes.append(f"workload: {workload.describe()}")
    backend = reg[winner.config.backend]
    prepared = plan.prepared[winner.config.backend]
    return _profile_for(hw, spec, objective, backend, winner.config, prepared, table=search.results, notes=notes)


def resolve_profile(
    spec: ModelSpec,
    objective: str = "balanced",
    force_backend: Optional[str] = None,
    recalibrate: bool = False,
    skip_calibration: bool = False,
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    progress: Optional[ProgressFn] = None,
    hw: Optional[HardwareDescriptor] = None,
    on_stage: Optional[Callable[[str], None]] = None,
) -> Profile:
    """Cached profile if valid, else run the full pipeline and cache the result."""
    say = on_stage or (lambda s: None)
    say("probe")
    hw = hw or probe()
    if not recalibrate and not skip_calibration:
        cached = profile_cache.load(hw, spec, objective)
        if cached is not None and (force_backend is None or cached.backend == force_backend):
            say("cache hit")
            return cached
    say("select")
    candidates, reg = select(hw, spec, force=force_backend)
    if not candidates:
        raise RuntimeError("no backend supports this machine/model; run `polyserve probe` for details")
    say("prepare + plan")
    plan = prepare_and_plan(hw, spec, candidates, reg, materialize=not skip_calibration)
    if skip_calibration:
        say("defaults")
        return default_profile(hw, spec, plan, reg, objective)
    say("calibrate")
    profile = calibrate(hw, spec, objective, plan, reg, workload=workload, constraints=constraints, progress=progress)
    profile_cache.save(profile)
    return profile
