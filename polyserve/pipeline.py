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
from polyserve.calibrate.objectives import Constraints, close_call_level
from polyserve.calibrate.search import ProgressFn, StagedSearch, SubprocessTrialRunner, TrialRunner
from polyserve.calibrate.workload import Workload, get_workload
from polyserve.hardware import hardware_hash, llmtrace_version, probe
from polyserve.memory import plan as memory_plan
from polyserve.models import Config, HardwareDescriptor, MemoryEstimate, ModelSpec, PreparedModel, Profile
from polyserve.quantized import PREQUANTIZED
from polyserve.selector import select_backends

logger = logging.getLogger(__name__)

AUTO_QUANT = "auto"


def allowed_quants(supported: List[str], quants: Optional[List[str]]) -> Optional[List[str]]:
    """The precisions a backend may prepare under --quant. None: the backend lists none, so it decides.

    `auto` (the default) is every supported precision except the Hub's pre-quantized checkpoints:
    4-bit AWQ and GPTQ, which on Qwen2.5 3B and 7B raised perplexity by 26-36% where fp8 cost 1-2%,
    and 8-bit W8A8, whose quality has not been measured. A server should not trade quality away
    unless asked: `--quant auto,awq` or `--quant auto,w8a8` adds them back.
    """
    if not supported:
        return None
    wanted = quants if quants is not None else [AUTO_QUANT]
    return [q for q in supported if q in wanted or (AUTO_QUANT in wanted and q not in PREQUANTIZED)]


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


@dataclass
class SearchOptions:
    """Switches for the optional search dimensions. The defaults explore everything that is safe."""

    # None = auto: every supported precision but 4-bit checkpoints. A list may include "auto".
    quants: Optional[List[str]] = None
    kv_quant: bool = True  # quantized KV caches
    speculative: bool = True  # n-gram and draft-model speculative decoding
    prefix_cache: bool = True  # keep prefix caching on, and tune it for shared-prefix workloads
    phase_tuning: bool = True  # the prefill-knob stage
    combine: bool = True  # combinations of the strategies that were promising on their own
    budget_s: Optional[float] = None  # --budget: seconds of calibration, after which later trials are skipped
    ttft_percentile: int = 95  # --ttft-percentile: which time to first token the ceiling applies to
    confirm: bool = True  # --confirm: re-measure the best three configurations and choose on those runs
    all_levels: bool = False  # --all-levels: measure every concurrency level of every trial (no early stop)
    explore: bool = False  # --explore: random draws around the leader after the stages (off: see search.py)
    # --max-quality-loss: the share of answers a cheaper precision may change before the search drops it,
    # measured greedily on the calibration prompts against the most faithful precision that runs (quality.py).
    # None = quality is measured separately and never gates the search, as it did before this option existed.
    max_quality_loss: Optional[float] = None

    def key(self, objective: Optional[str] = None) -> Dict[str, str]:
        """Options that change the pick, recorded in the profile and in its cache path: the non-default
        ones, and for `balanced` (the objective with a TTFT ceiling) the percentile unless it is the
        median. Balanced profiles from before the percentile existed were picked by the median and
        carry no key for it, so they are served only where the median is asked for."""
        out: Dict[str, str] = {}
        if self.quants is not None:
            out["quant"] = "+".join(self.quants)
        if not self.kv_quant:
            out["kv"] = "off"
        if not self.speculative:
            out["spec"] = "off"
        if not self.prefix_cache:
            out["prefix"] = "off"
        if not self.phase_tuning:
            out["prefill"] = "off"
        if not self.combine:
            out["combine"] = "off"
        if not self.confirm:
            out["confirm"] = "off"
        if self.all_levels:
            out["levels"] = "all"
        if self.explore:
            out["explore"] = "on"
        if self.max_quality_loss is not None:  # a quality-gated profile is not served where none was asked for
            out["quality"] = f"{self.max_quality_loss:.3f}"
        if self.budget_s is not None:  # a budgeted profile is never served where a full one was asked for
            out["budget"] = f"{int(self.budget_s)}s"
        if objective == "balanced" and self.ttft_percentile != 50:
            out["ttft"] = f"p{self.ttft_percentile}"
        return out

    def allows(self, quant: str) -> bool:
        """Whether --quant permits this precision (see allowed_quants)."""
        return quant in (allowed_quants([quant], self.quants) or [])


def _usable(cached: Optional[Profile], opts: SearchOptions, force_backend: Optional[str]) -> bool:
    """Serve a cached profile only if it matches the forced backend and the current --quant: a
    profile calibrated when `auto` still picked 4-bit checkpoints must not outlive that default."""
    return (cached is not None and (force_backend is None or cached.backend == force_backend)
            and opts.allows(cached.config.quant))


def level_rule(objective: str, constraints: Constraints, opts: SearchOptions,
               power_points: Optional[list] = None) -> Optional[Callable]:
    """How a calibration trial measures its concurrency levels: busiest first, stopping once one settles the
    score (objectives.enough_level), or every level (None). Every level with --all-levels, which exists to
    check that shortcut, and with a power stage, which compares energy per token over every level."""
    if power_points or opts.all_levels:
        return None
    from polyserve.calibrate.objectives import enough_level

    return enough_level(objective, constraints)


def combination_fits(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult"):
    """Whether a combined config can run: it must fit in memory, and on vLLM and SGLang the prefill
    budget must cover the batch (each stage keeps that true on its own; a combination may not)."""
    from polyserve.memory import estimate

    small_gpu = hw.gpu is not None and hw.gpu.vram_total_bytes < 32 * 2**30

    def fn(c: Config) -> bool:
        # SGLang 0.5.19 captures prefill CUDA graphs up to its prefill budget, outside its memory fraction: on an
        # RTX 4090 (24 GB) every trial with a budget of 8192 or 16384 ran out of memory at start-up, at 93% and
        # at 88% reserved, in four calibrations.
        if c.backend == "sglang" and small_gpu and (c.prefill_budget or 0) >= 8192:
            return False
        pm = plan.prepared.get(c.backend)
        if pm is None:
            return False
        if c.prefill_budget is not None and c.prefill_budget < c.batch and not c.backend.startswith("llamacpp"):
            return False
        try:
            return estimate(hw, pm, c, reg[c.backend].memory_model(hw)).feasible
        except Exception:
            return False

    return fn


def kv_variants_fn(hw: HardwareDescriptor, reg: Dict[str, BaseBackend], plan: "PlanResult"):
    """Leader variants with a quantized KV cache, plus the batch step that the smaller cache admits."""
    from polyserve.memory import estimate

    def fn(c: Config) -> List[Config]:
        be = reg[c.backend]
        pm = plan.prepared.get(c.backend)
        if pm is None:
            return []
        mm = be.memory_model(hw)

        def fits(x: Config) -> bool:
            try:
                return estimate(hw, pm, x, mm).feasible
            except Exception:
                return False

        out: List[Config] = []
        for kd in be.kv_dtypes(hw):
            if kd == c.kv_dtype:
                continue
            v = c.model_copy(update={"kv_dtype": kd})
            if fits(v):
                out.append(v)
            bigger = [b for b in be.batch_ladder() if b > c.batch]
            if bigger:
                update: Dict[str, object] = {"batch": bigger[0]}
                if v.prefill_budget is not None and v.prefill_budget < bigger[0]:
                    update["prefill_budget"] = None  # vLLM: the budget must cover the batch
                vb = v.model_copy(update=update)
                if fits(vb) and not fits(vb.model_copy(update={"kv_dtype": c.kv_dtype})):
                    out.append(vb)
        return out

    return fn


def search_space(hw: HardwareDescriptor, plan: "PlanResult", reg: Dict[str, BaseBackend], workload: Workload,
                 options: Optional[SearchOptions] = None) -> List[Config]:
    """Every runnable configuration calibration could reach, each once: the planner's configurations, then each
    stage's variants of everything so far (prefill budgets, quantized KV caches with the batch step they allow,
    prefix-cache settings, speculative methods), so every combination of them is included. The explore stage
    draws from it, and so does random search in benchmarks/search_baselines.py."""
    opts = options or SearchOptions()
    stages: List[Callable[[Config], List[Config]]] = []
    if opts.phase_tuning:
        stages.append(lambda c: reg[c.backend].prefill_variants(c))
    if opts.kv_quant:
        stages.append(kv_variants_fn(hw, reg, plan))
    if opts.prefix_cache and workload.shared_prefix_tokens > 0:
        stages.append(lambda c: reg[c.backend].prefix_variants(c))
    if opts.speculative:
        stages.append(lambda c: reg[c.backend].spec_variants(c, plan.prepared[c.backend]))
    configs = list(plan.all_feasible)
    if not opts.prefix_cache:
        configs = [c.model_copy(update={"prefix_cache": False}) for c in configs]
    for stage in stages:
        configs += [v for c in configs for v in stage(c)]
    fits = combination_fits(hw, reg, plan)
    out: Dict[str, Config] = {}
    for c in configs:
        if c.key() not in out and fits(c):
            out[c.key()] = c
    return list(out.values())


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
    workload: Optional[Workload] = None,
    quants: Optional[List[str]] = None,
) -> PlanResult:
    """Prepare each candidate backend's model, enumerate its grid, keep what fits the workload.

    `quants` restricts weight precisions (the --quant option); None allows everything supported.
    """
    result = PlanResult(hw=hw, spec=spec, candidates=list(candidates))
    min_ctx = workload.min_ctx if workload else 0
    for name in candidates:
        backend = reg[name]
        try:
            allowed = allowed_quants(backend.supported_quants(hw), quants)
            if quants is not None and allowed == []:
                raise RuntimeError(f"none of --quant {','.join(quants)} is available on {name}")
            prepared = backend.prepare(spec, hw, quants=allowed)
            grid = backend.candidate_configs(hw, prepared, min_ctx=min_ctx)
            if not grid:
                raise RuntimeError(
                    f"model max context {prepared.arch.max_position_embeddings} < workload minimum {min_ctx}"
                )
            kept = memory_plan(hw, prepared, grid, backend.memory_model(hw))
            result.prepared[name] = prepared
            result.feasible[name] = kept
            result.considered[name] = len(grid)
            logger.info("%s: %d/%d configs feasible", name, len(kept), len(grid))
            if materialize and kept:
                survivors = sorted({c.quant for c, _ in kept})  # not `quants`: that filter applies to every backend
                backend.materialize(prepared, survivors)
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
    workload: Optional[Workload] = None,
    prepared_all: Optional[Dict[str, PreparedModel]] = None,
) -> Profile:
    launch = backend.launch_spec(cfg, prepared, 0)
    workload = workload or get_workload("default")
    return Profile(
        polyserve_version=__version__,
        hardware_hash=hardware_hash(hw),
        hardware=hw,
        model_id=spec.hf_id,
        objective=objective,
        workload=workload.name,
        workload_spec=workload.spec(),
        backend=backend.name,
        backend_version=backend.version(hw),
        config=cfg,
        prepared=prepared,
        prepared_all=dict(prepared_all or {}),
        launch_args=launch.args,
        launch_env=launch.env,
        calibration_table=list(table or []),
        llmtrace_version=llmtrace_version(),
        notes=list(notes or []),
    )


def default_profile(hw: HardwareDescriptor, spec: ModelSpec, plan: PlanResult, reg: Dict[str, BaseBackend],
                    objective: str = "balanced", workload: Optional[Workload] = None) -> Profile:
    """Uncalibrated: first candidate backend with its default config (or the largest feasible one)."""
    workload = workload or get_workload("default")
    for name in plan.candidates:
        prepared = plan.prepared.get(name)
        if prepared is None:
            continue
        backend = reg[name]
        cfg = backend.default_config(hw, prepared, min_ctx=workload.min_ctx)
        feasible = plan.feasible.get(name) or []
        if feasible and not any(c.key() == cfg.key() for c, _ in feasible):
            # Default does not fit; take the feasible config with the most headroom for ctx/batch.
            cfg = max(feasible, key=lambda ce: (ce[0].ctx, ce[0].batch))[0]
        if cfg.quant not in prepared.weights_bytes:
            cfg.quant = next(iter(prepared.weights_bytes))
        backend.materialize(prepared, [cfg.quant])
        return _profile_for(hw, spec, objective, backend, cfg, prepared, notes=["uncalibrated defaults"],
                            workload=workload, prepared_all=plan.prepared)
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
    power_mode: str = "off",
    power_controller: Optional[object] = None,
    power_points: Optional[list] = None,
    phase_tuning: bool = True,
    options: Optional[SearchOptions] = None,
) -> Profile:
    opts = options or SearchOptions()
    workload = workload or get_workload("default")
    constraints = constraints or Constraints(ttft_ceiling_ms=workload.ttft_ceiling_ms,
                                             tpot_ceiling_ms=workload.tpot_ceiling_ms,
                                             ttft_percentile=opts.ttft_percentile)
    feasible = plan.all_feasible
    if not feasible:
        raise RuntimeError("memory planner left no feasible configuration; try a smaller model or quant")
    if not opts.prefix_cache:
        feasible = [c.model_copy(update={"prefix_cache": False}) for c in feasible]
    stages = []
    if opts.kv_quant:
        stages.append(("kv", kv_variants_fn(hw, reg, plan)))
    if opts.prefix_cache and workload.shared_prefix_tokens > 0:
        stages.append(("prefix", lambda c: reg[c.backend].prefix_variants(c)))
    if opts.speculative:
        stages.append(("spec", lambda c: reg[c.backend].spec_variants(c, plan.prepared[c.backend])))
    controller, points, power_notes = None, [], []
    if power_points is not None:  # injected (tests, or a caller that already probed the GPU)
        controller, points = power_controller, list(power_points)
    elif power_mode != "off":
        controller, points, power_notes = setup_power(hw, power_mode, power_controller)
    probe = None
    if opts.max_quality_loss is not None and workload.prompts:
        from polyserve.quality import QualityProbe

        # The calibration prompts themselves, answered greedily once per precision while its engine is up.
        # Sixteen is enough to catch a precision that answers differently without adding a minute per trial.
        probe = QualityProbe(prompts=list(workload.prompts)[:16], tolerance=opts.max_quality_loss)
    if runner is None:
        runner = SubprocessTrialRunner(
            backends={n: reg[n] for n in plan.candidates},
            models=plan.prepared,
            hw=hw,
            workload=workload,
            log_dir=log_dir or (profile_cache.logs_dir() / spec.safe_id),
            power=controller,
            enough=level_rule(objective, constraints, opts, points),
            close_call=close_call_level(objective, constraints),
            quality=probe,
        )
    from polyserve.predict import Predictor

    search = StagedSearch(objective=objective, runner=runner, constraints=constraints, progress=progress,
                          predictor=Predictor(hw), models=plan.prepared, workload=workload,
                          power_points=points, variant_stages=stages, combine=opts.combine,
                          budget_s=opts.budget_s, confirm_top=3 if opts.confirm else 0,
                          feasible_fn=combination_fits(hw, reg, plan), quality=probe,
                          explore_space=search_space(hw, plan, reg, workload, opts) if opts.explore else [],
                          spec_first=True,
                          prefill_variants=((lambda c: reg[c.backend].prefill_variants(c))
                                            if phase_tuning and opts.phase_tuning else None))
    t0 = time.monotonic()
    try:
        winner, notes = search.run(feasible)
    finally:
        if controller is not None and getattr(controller, "applied", None) is not None:
            controller.restore()  # type: ignore[attr-defined]
    if winner is None:
        raise RuntimeError("calibration failed: " + "; ".join(notes))
    elapsed = time.monotonic() - t0
    notes = power_notes + notes
    notes.append(f"calibration took {elapsed:.0f}s over {len(search.results)} trials")
    notes.append(f"workload: {workload.describe()}")
    backend = reg[winner.config.backend]
    prepared = plan.prepared[winner.config.backend]
    profile = _profile_for(hw, spec, objective, backend, winner.config, prepared, table=search.results,
                           notes=notes, workload=workload, prepared_all=plan.prepared)
    profile.calibration_seconds = elapsed
    profile.calibration_trials = len(search.results)
    profile.power_mode = power_mode
    profile.options = opts.key(objective)
    return profile


def setup_power(hw: HardwareDescriptor, mode: str, controller: Optional[object] = None):
    """Probe what this GPU permits and build the stage-4 points. Returns (controller, points, notes)."""
    from polyserve.power import PowerControlUnavailable, candidate_points, controller_for

    if hw.gpu is None:
        return None, [], [f"--power {mode} requested but no GPU is visible; power stage skipped"]
    ctl = controller or controller_for(hw.gpu.index)
    try:
        caps = ctl.capabilities()  # type: ignore[attr-defined]
    except PowerControlUnavailable as exc:
        return None, [], [f"--power {mode} requested but unavailable: {exc}; power stage skipped"]
    notes = []
    if mode in ("cap", "both") and not caps.can_cap:
        notes.append(f"power capping not permitted: {caps.reasons.get('cap', 'unknown reason')}")
    if mode in ("clock", "both") and not caps.can_lock:
        notes.append(f"clock locking not permitted: {caps.reasons.get('clock', 'unknown reason')}")
    points = candidate_points(caps, mode)
    trying = [p for p in points if not p.is_default]
    if not trying:
        notes.append("no power settings available to try; power stage skipped")
        return None, [], notes
    notes.append(f"power stage tried {len(trying)} settings: " + ", ".join(p.label() for p in trying))
    return ctl, points, notes


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
    power_mode: str = "off",
    phases: str = "unified",
    kv_connector: str = "nixl",
    options: Optional[SearchOptions] = None,
    layout: str = "single",
) -> Profile:
    """Cached profile if valid, else run the full pipeline and cache the result.

    `phases` other than "unified" resolves the unified profile first, then measures disaggregated
    prefill/decode pairs built around it (see polyserve.disagg).
    """
    say = on_stage or (lambda s: None)
    workload = workload or get_workload("default")
    say("probe")
    hw = hw or probe()
    opts = options or SearchOptions()
    if phases != "unified" and layout != "single":
        raise RuntimeError("choose either --phases (disaggregated prefill/decode) or --layout (replicas, tp), not both")
    if phases != "unified" and not skip_calibration:
        return _resolve_phased(spec, objective, force_backend, recalibrate, workload, constraints, progress, hw,
                               on_stage, power_mode, phases, kv_connector, opts)
    if layout != "single" and not skip_calibration:
        return _resolve_layout(spec, objective, force_backend, recalibrate, workload, constraints, progress, hw,
                               on_stage, power_mode, opts, layout)
    if not recalibrate and not skip_calibration:
        cached = profile_cache.load(hw, spec, objective, workload.name, power_mode, options=opts.key(objective))
        if _usable(cached, opts, force_backend):
            say("cache hit")
            return cached
    say("select")
    candidates, reg = select(hw, spec, force=force_backend)
    if not candidates:
        raise RuntimeError("no backend supports this machine/model; run `polyserve probe` for details")
    say("prepare + plan")
    plan = prepare_and_plan(hw, spec, candidates, reg, materialize=not skip_calibration, workload=workload,
                            quants=opts.quants)
    if skip_calibration:
        say("defaults")
        return default_profile(hw, spec, plan, reg, objective, workload=workload)
    say("calibrate")
    profile = calibrate(hw, spec, objective, plan, reg, workload=workload, constraints=constraints, progress=progress,
                        power_mode=power_mode, options=opts)
    profile_cache.save(profile)
    return profile


def _resolve_phased(spec: ModelSpec, objective: str, force_backend: Optional[str], recalibrate: bool,
                    workload: Workload, constraints: Optional[Constraints], progress: Optional[ProgressFn],
                    hw: HardwareDescriptor, on_stage: Optional[Callable[[str], None]], power_mode: str,
                    phases: str, kv_connector: str, options: Optional[SearchOptions] = None) -> Profile:
    from polyserve.disagg import calibrate_disaggregated, check_support

    say = on_stage or (lambda s: None)
    if not recalibrate:
        cached = profile_cache.load(hw, spec, objective, workload.name, power_mode, phases,
                                    options=(options or SearchOptions()).key())
        if _usable(cached, options or SearchOptions(), force_backend):
            say("cache hit")
            return cached
    if phases == "disaggregated":
        # Refuse before spending a unified calibration on a machine that cannot disaggregate.
        reason = check_support(hw, "vllm", kv_connector)
        if reason is None and not hw.backend_available("vllm"):
            reason = f"vLLM is not available ({hw.backends.get('vllm').reason if hw.backends.get('vllm') else 'unknown'})"
        if reason:
            raise RuntimeError(f"--phases disaggregated is not possible here: {reason}")
    # Disaggregation runs on vLLM; `auto` lets the unified search choose freely and falls back if needed.
    base = resolve_profile(spec, objective, force_backend=force_backend or ("vllm" if phases == "disaggregated" else None),
                           recalibrate=recalibrate, workload=workload, constraints=constraints, progress=progress,
                           hw=hw, on_stage=on_stage, power_mode=power_mode if phases == "auto" else "off",
                           options=options)
    _, reg = select(hw, spec, force=base.backend)
    say("disaggregate")
    profile = calibrate_disaggregated(hw, spec, objective, base, reg, workload=workload, constraints=constraints,
                                      phases=phases, connector=kv_connector, progress=progress, power_mode=power_mode,
                                      log_dir=profile_cache.logs_dir() / spec.safe_id)
    profile_cache.save(profile)
    return profile


def _resolve_layout(spec: ModelSpec, objective: str, force_backend: Optional[str], recalibrate: bool,
                    workload: Workload, constraints: Optional[Constraints], progress: Optional[ProgressFn],
                    hw: HardwareDescriptor, on_stage: Optional[Callable[[str], None]], power_mode: str,
                    options: SearchOptions, layout: str) -> Profile:
    from polyserve.layout import calibrate_layout

    say = on_stage or (lambda s: None)
    key = {**options.key(objective), "layout": layout}
    if not recalibrate:
        cached = profile_cache.load(hw, spec, objective, workload.name, power_mode, options=key)
        if _usable(cached, options, force_backend):
            say("cache hit")
            return cached
    gpus = [g for g in hw.gpus if g.vendor == "nvidia"]
    if layout != "auto" and len(gpus) < 2:
        raise RuntimeError(f"--layout {layout} is not possible here: multi-GPU layouts need two NVIDIA GPUs; "
                           f"{len(gpus)} visible")
    base = resolve_profile(spec, objective, force_backend=force_backend, recalibrate=recalibrate, workload=workload,
                           constraints=constraints, progress=progress, hw=hw, on_stage=on_stage,
                           power_mode=power_mode, options=options)
    _, reg = select(hw, spec, force=base.backend)
    say("layout")
    profile = calibrate_layout(hw, objective, base, reg, layout, workload=workload, constraints=constraints,
                               progress=progress, log_dir=profile_cache.logs_dir() / spec.safe_id)
    profile_cache.save(profile)
    return profile
