"""Staged search over the feasible configuration set.

1. quant / precision  - one short run per feasible (backend, quant); keep the top 2.
2. memory config      - largest safe gpu_memory_utilization / offload layers / context.
3. batch / concurrency - sweep max_num_seqs (vLLM/SGLang) or n_parallel/n_batch (llama.cpp): the
                        decode knob.
3b. prefill           - on the leading config, sweep the prefill knob: the chunked-prefill token
                        budget (vLLM, SGLang) or the micro-batch (llama.cpp). Default on.
3c. kv                - quantized KV caches, including the batch step a smaller cache admits.
3d. prefix            - prefix-cache settings, when the workload's prompts share a prefix.
3e. spec              - speculative decoding (n-gram lookup, a small draft model).
4. power (optional)   - on the leading config, sweep GPU power caps and/or locked SM clocks
                        through NVML without relaunching, and keep the setting that saves energy.

The winner is the objective's constrained argmax over *every* successful trial, not just stage 3.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from polyserve.backends.base import BaseBackend, free_port
from polyserve.calibrate.measure import run_trial
from polyserve.calibrate.objectives import Constraints, pick
from polyserve.calibrate.workload import Workload
from polyserve.memlog import device_used_mb, merge, parse_log
from polyserve.memory import estimate as memory_estimate
from polyserve.models import (
    Config,
    HardwareDescriptor,
    MemoryObservation,
    PreparedModel,
    TrialMetrics,
    TrialResult,
)
from polyserve.power import PowerSetting, setting_of, with_power

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, Config, Optional[TrialResult]], None]


class TrialRunner(Protocol):
    def run(self, cfg: Config, stage: str) -> TrialResult: ...


class SubprocessTrialRunner:
    """Launch the backend for one config, measure, tear down."""

    def __init__(
        self,
        backends: Dict[str, BaseBackend],
        models: Dict[str, PreparedModel],
        hw: HardwareDescriptor,
        workload: Workload,
        log_dir: Optional[Path] = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 180.0,
        power: Optional[object] = None,
        power_settle_s: float = 2.0,
    ):
        self.backends = backends
        self.models = models
        self.hw = hw
        self.workload = workload
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.power = power  # polyserve.power.PowerController, or None when energy tuning is off
        self.power_settle_s = power_settle_s

    def _apply_power(self, cfg: Config) -> Optional[str]:
        """Apply cfg's power setting to the GPU. Returns an error message if it cannot be applied."""
        setting = setting_of(cfg)
        if setting.is_default:
            if self.power is not None and getattr(self.power, "applied", None) is not None:
                self.power.restore()  # type: ignore[attr-defined]
            return None
        if self.power is None:
            return "power setting requested but no power controller is configured"
        try:
            self.power.apply(setting)  # type: ignore[attr-defined]
        except Exception as exc:
            return f"power control unavailable: {exc}"
        if self.power_settle_s > 0:
            time.sleep(self.power_settle_s)  # let clocks and board power settle under the new limit
        return None

    def _restore_power(self) -> None:
        if self.power is not None and getattr(self.power, "applied", None) is not None:
            try:
                self.power.restore()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.error("power restore failed: %s", exc)

    def _predict(self, cfg: Config, backend: BaseBackend, model: PreparedModel):
        try:
            return memory_estimate(self.hw, model, cfg, backend.memory_model(self.hw))
        except Exception as exc:
            logger.debug("no memory estimate for %s: %s", cfg.key(), exc)
            return None

    def run(self, cfg: Config, stage: str) -> TrialResult:
        backend = self.backends[cfg.backend]
        model = self.models[cfg.backend]
        port = free_port()
        log_path = None
        if self.log_dir:
            safe = re.sub(r"[^A-Za-z0-9._=+-]+", "_", cfg.key())
            log_path = self.log_dir / f"{int(time.time())}_{safe}.log"
        hooks = backend.workload_hooks(self.hw, model)
        if cfg.tp > 1 and hooks.gpu_ids:  # one engine over several GPUs: measure all of them
            gpus = [g.index for g in self.hw.gpus if g.vendor == "nvidia"][: cfg.tp]
            hooks = dataclasses.replace(hooks, gpu_ids=gpus or hooks.gpu_ids)
        observation = MemoryObservation(predicted=self._predict(cfg, backend, model))
        baseline_mb = device_used_mb(hooks.gpu_ids)
        proc = None
        try:
            proc = backend.launch(cfg, model, port, log_path=log_path)
            if not proc.wait_ready(timeout=self.startup_timeout):
                tail = proc.tail_log(20)
                observation.measured = parse_log(cfg.backend, proc.tail_log(400))
                return TrialResult(
                    config=cfg, stage=stage, metrics=TrialMetrics(), launched=False,
                    error=f"failed to start (rc={proc.returncode()})\n{tail}", memory=observation,
                )
            perr = self._apply_power(cfg)
            if perr:
                return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=perr, memory=observation)
            metrics = run_trial(
                f"http://127.0.0.1:{port}", hooks, self.workload, pid=proc.pid,
                request_timeout=self.request_timeout,
            )
            observation.measured = merge(
                parse_log(cfg.backend, proc.tail_log(400)), metrics.peak_mem_mb, baseline_mb,
                metrics.telemetry_source,
            )
            err = None
            if not metrics.ok:
                err = metrics.failure_summary()
            elif metrics.failed:
                logger.info("%s: tolerated %d/%d failed requests", cfg.key(), metrics.failed, metrics.requests)
            return TrialResult(config=cfg, stage=stage, metrics=metrics, error=err, memory=observation)
        except Exception as exc:
            logger.exception("trial %s crashed", cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=str(exc),
                               memory=observation)
        finally:
            self._restore_power()
            if proc is not None:
                proc.stop()

    def sweep(
        self,
        base: Config,
        settings: Sequence[PowerSetting],
        stage: str,
        progress: Optional[ProgressFn] = None,
    ) -> List[TrialResult]:
        """Launch `base` once, then measure it under each power setting in turn.

        Power caps and clock locks take effect on a running server, so one launch serves the whole
        sweep: faster than relaunching per point, and every point shares the same warm engine, so
        differences come from the setting rather than from launch-to-launch variance. Include the
        default setting to get a same-launch baseline.
        """
        backend = self.backends[base.backend]
        model = self.models[base.backend]
        port = free_port()
        log_path = None
        if self.log_dir:
            log_path = self.log_dir / f"{int(time.time())}_{re.sub(r'[^A-Za-z0-9._=+-]+', '_', base.key())}_power.log"
        hooks = backend.workload_hooks(self.hw, model)
        results: List[TrialResult] = []
        proc = None
        try:
            proc = backend.launch(base.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}), model,
                                  port, log_path=log_path)
            if not proc.wait_ready(timeout=self.startup_timeout):
                err = f"failed to start (rc={proc.returncode()})\n{proc.tail_log(20)}"
                return [TrialResult(config=with_power(base, s), stage=stage, metrics=TrialMetrics(), launched=False,
                                    error=err) for s in settings]
            for s in settings:
                cfg = with_power(base, s)
                if progress:
                    progress(stage, cfg, None)
                perr = self._apply_power(cfg)
                if perr:
                    res = TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=perr)
                else:
                    try:
                        metrics = run_trial(f"http://127.0.0.1:{port}", hooks, self.workload, pid=proc.pid,
                                            request_timeout=self.request_timeout)
                        err = None if metrics.ok else metrics.failure_summary()
                        res = TrialResult(config=cfg, stage=stage, metrics=metrics, error=err)
                    except Exception as exc:
                        logger.exception("power point %s crashed", cfg.key())
                        res = TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=str(exc))
                results.append(res)
                if progress:
                    progress(stage, cfg, res)
            return results
        finally:
            self._restore_power()
            if proc is not None:
                proc.stop()


# --------------------------------------------------------------------------- staged search


def _memory_knob(cfg: Config) -> float:
    """Single scalar for 'how much memory this config asks for' (higher = more)."""
    if cfg.gpu_memory_utilization is not None:
        return cfg.gpu_memory_utilization
    if cfg.n_gpu_layers is not None:
        return float(cfg.n_gpu_layers)
    return 0.0


def _baseline_candidates(group: Sequence[Config]) -> List[Config]:
    """Stage-1 baselines for a (backend, quant) group: median ctx and batch, memory knob largest first.

    The planner already vetted every config here, so the largest memory setting (full GPU
    offload, highest gpu_memory_utilization) is the fair one to compare quants at; the
    smaller ones are fallbacks if that launch fails.

    The median context and median batch need not co-occur: the planner prunes the grid
    asymmetrically, so a long-context workload on a small card can keep (32k, batch 4) and
    (16k, batch 64) while dropping (32k, batch 64). Fall back to the nearest surviving batch
    at that context, then to the whole group, rather than returning nothing.
    """
    if not group:
        return []
    ctxs = sorted({c.ctx for c in group})
    batches = sorted({c.batch for c in group})
    ctx = ctxs[len(ctxs) // 2]
    batch = batches[len(batches) // 2]
    same = [c for c in group if c.ctx == ctx and c.batch == batch]
    if not same:
        at_ctx = [c for c in group if c.ctx == ctx] or list(group)
        nearest = min({c.batch for c in at_ctx}, key=lambda b: (abs(b - batch), b))
        same = [c for c in at_ctx if c.batch == nearest]
    return sorted(same, key=lambda c: (_memory_knob(c), c.n_batch or 0), reverse=True)


def _baseline(group: Sequence[Config]) -> Config:
    return _baseline_candidates(group)[0]


@dataclass
class StagedSearch:
    objective: str
    runner: TrialRunner
    constraints: Constraints = field(default_factory=Constraints)
    top_quants: int = 2
    max_memory_trials_per_quant: int = 3
    progress: Optional[ProgressFn] = None
    # Optional performance predictor (polyserve.predict.Predictor) + the models it needs.
    predictor: Optional[object] = None
    models: Dict[str, PreparedModel] = field(default_factory=dict)
    workload: Optional[Workload] = None
    prune_below: float = 0.4  # skip a quant whose predicted best tok/s < this fraction of the best predicted
    # Stage 4: power settings to try on the leading config (the default setting first). Empty = off.
    power_points: List[PowerSetting] = field(default_factory=list)
    # Stage 3b: configs differing from the leader only in the prefill knob. None = stage off.
    prefill_variants: Optional[Callable[[Config], List[Config]]] = None
    # Stages 3c-3e: (name, variants of the current leader), run in order after the prefill stage.
    variant_stages: List[Tuple[str, Callable[[Config], List[Config]]]] = field(default_factory=list)
    results: List[TrialResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    _done: Dict[str, TrialResult] = field(default_factory=dict)

    # ---- predictor helpers

    def _predicted_tok_s(self, cfg: Config) -> Optional[float]:
        """Predicted best-level tok/s under the objective's TTFT ceiling, or None if unavailable/unfitted."""
        if self.predictor is None or cfg.backend not in self.models:
            return None
        try:
            if not self.predictor.is_fitted(cfg.backend):  # type: ignore[attr-defined]
                return None
            wl = self.workload or Workload()
            ceiling = self.constraints.ttft_ceiling_ms if self.objective == "balanced" else None
            kwargs = {}
            if self.constraints.tpot_ceiling_ms is not None:
                kwargs["tpot_ceiling_ms"] = self.constraints.tpot_ceiling_ms
            p = self.predictor.best_level(  # type: ignore[attr-defined]
                self.models[cfg.backend], cfg, wl.prefill_tokens, wl.decode_tokens, wl.concurrencies, ceiling,
                **kwargs,
            )
            return float(p.tok_s)
        except Exception as exc:
            logger.debug("predictor failed for %s: %s", cfg.key(), exc)
            return None

    # ---- helpers

    def _run(self, cfg: Config, stage: str) -> TrialResult:
        key = cfg.key()
        if key in self._done:
            return self._done[key]
        if self.progress:
            self.progress(stage, cfg, None)
        res = self.runner.run(cfg, stage)
        self._done[key] = res
        self.results.append(res)
        if self.progress:
            self.progress(stage, cfg, res)
        return res

    def _best_of(self, results: Sequence[TrialResult]) -> Optional[TrialResult]:
        winner, _ = pick(results, self.objective, self.constraints)
        return winner

    # ---- stages

    def stage_quant(self, feasible: Sequence[Config]) -> List[Tuple[str, str]]:
        groups: Dict[Tuple[str, str], List[Config]] = {}
        for c in feasible:
            groups.setdefault((c.backend, c.quant), []).append(c)
        # Predictor-guided pruning: skip (backend, quant) groups predicted far below the best group.
        predicted: Dict[Tuple[str, str], float] = {}
        for key, group in groups.items():
            p = self._predicted_tok_s(_baseline(group))
            if p is not None:
                predicted[key] = p
        if predicted:
            best_pred = max(predicted.values())
            for key, p in predicted.items():
                if p < self.prune_below * best_pred:
                    self.notes.append(
                        f"skipped {key[0]}/{key[1]}: predicted {p:.0f} tok/s < {self.prune_below:.0%} of best "
                        f"predicted {best_pred:.0f}"
                    )
                    groups.pop(key, None)
        stage_results: Dict[Tuple[str, str], TrialResult] = {}
        for key, group in groups.items():
            candidates = _baseline_candidates(group)[: self.max_memory_trials_per_quant]
            if not candidates:
                logger.warning("no baseline config for %s; skipping", key)
                continue
            # Largest memory setting first; step down only if the launch itself fails.
            for cfg in candidates:
                res = self._run(cfg, "quant")
                stage_results[key] = res
                if res.launched:
                    break
        ok = [r for r in stage_results.values() if r.ok]
        from polyserve.calibrate.objectives import rank

        ranked = rank(ok, self.objective, self.constraints)
        kept = [(r.result.config.backend, r.result.config.quant) for r in ranked[: self.top_quants]]
        logger.info("stage 1 kept: %s", kept)
        return kept

    def stage_memory(self, feasible: Sequence[Config], kept: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], Config]:
        chosen: Dict[Tuple[str, str], Config] = {}
        for key in kept:
            group = [c for c in feasible if (c.backend, c.quant) == key]
            base = next(
                (r.config for r in self.results if r.ok and (r.config.backend, r.config.quant) == key),
                _baseline(group),
            )
            # Fix batch to the baseline's; vary memory knob and ctx, largest first.
            same_batch = [c for c in group if c.batch == base.batch and (c.n_batch == base.n_batch)]
            ordered = sorted(same_batch, key=lambda c: (_memory_knob(c), c.ctx), reverse=True)
            # De-duplicate identical (knob, ctx) pairs.
            seen = set()
            trials: List[Config] = []
            for c in ordered:
                sig = (_memory_knob(c), c.ctx)
                if sig in seen:
                    continue
                seen.add(sig)
                trials.append(c)
            winner: Optional[Config] = None
            for c in trials[: self.max_memory_trials_per_quant]:
                res = self._run(c, "memory")
                if res.ok:
                    winner = c
                    break  # largest safe config found
            if winner is None:
                # Fall back to whatever succeeded in stage 1 for this quant.
                prior = [r for r in self.results if r.ok and (r.config.backend, r.config.quant) == key]
                if prior:
                    winner = prior[0].config
            if winner is not None:
                chosen[key] = winner
        logger.info("stage 2 chose: %s", {k: v.key() for k, v in chosen.items()})
        return chosen

    def stage_batch(self, feasible: Sequence[Config], chosen: Dict[Tuple[str, str], Config]) -> None:
        for key, mem_cfg in chosen.items():
            variants = [
                c
                for c in feasible
                if (c.backend, c.quant) == key
                and c.ctx == mem_cfg.ctx
                and _memory_knob(c) == _memory_knob(mem_cfg)
            ]
            # Try the batch settings the predictor likes best first, so an interrupted run has the winner.
            preds = {c.key(): self._predicted_tok_s(c) for c in variants}
            if all(v is not None for v in preds.values()):
                variants.sort(key=lambda c: -preds[c.key()])
            else:
                variants.sort(key=lambda c: (c.batch, c.n_batch or 0))
            for c in variants:
                self._run(c, "batch")

    def stage_prefill(self, base: Config) -> None:
        """Tune the prefill knob on the leading config; stage 3 already tuned the decode knob."""
        if self.prefill_variants is None:
            return
        try:
            variants = self.prefill_variants(base)
        except Exception as exc:
            logger.warning("no prefill variants for %s: %s", base.key(), exc)
            return
        for v in variants:
            self._run(v, "prefill")

    def stage_variants(self, name: str, fn: Callable[[Config], List[Config]]) -> None:
        """Measure the current leader's variants along one dimension; the objective keeps the best."""
        leader, _ = pick(self.results, self.objective, self.constraints)
        if leader is None:
            return
        try:
            variants = fn(leader.config)
        except Exception as exc:
            logger.warning("no %s variants for %s: %s", name, leader.config.key(), exc)
            return
        for v in variants:
            self._run(v, name)

    def stage_power(self, base: Config) -> None:
        """Measure the leading config under each power setting and let the objective choose."""
        settings = list(self.power_points)
        if not any(not s.is_default for s in settings):
            return
        sweep = getattr(self.runner, "sweep", None)
        if sweep is not None:
            # One launch; the default setting is re-measured in it so every point shares a baseline.
            for res in sweep(base, settings, "power", progress=self.progress):
                self.results.append(res)
                self._done.setdefault(res.config.key(), res)
        else:
            for s in settings:
                if not s.is_default:
                    self._run(with_power(base, s), "power")

    def _power_note(self, winner: TrialResult) -> Optional[str]:
        s = setting_of(winner.config)
        base_key = winner.config.base_key()
        uncapped = [r for r in self.results if r.ok and r.config.base_key() == base_key and setting_of(r.config).is_default]
        tried = [r for r in self.results if r.config.base_key() == base_key and not setting_of(r.config).is_default]
        if not tried:
            return None
        if s.is_default:
            return f"power stage tried {len(tried)} settings; none saved energy within the allowed throughput loss"
        if not uncapped:
            return f"power setting chosen: {s.label()}"
        ref = max(uncapped, key=lambda r: r.metrics.tok_s)
        w, r = winner.metrics, ref.metrics
        tok = (w.tok_s - r.tok_s) / r.tok_s * 100 if r.tok_s else 0.0
        note = f"power setting chosen: {s.label()} ({tok:+.1f}% tok/s"
        if w.joules_per_token and r.joules_per_token:
            note += f", {(w.joules_per_token - r.joules_per_token) / r.joules_per_token * 100:+.1f}% J/token"
        if w.power_w and r.power_w:
            note += f", {w.power_w:.0f} W vs {r.power_w:.0f} W"
        return note + " against the same config at default power)"

    # ---- entry

    def run(self, feasible: Sequence[Config]) -> Tuple[Optional[TrialResult], List[str]]:
        if not feasible:
            return None, ["memory planner left no feasible configuration"]
        kept = self.stage_quant(feasible)
        if not kept:
            return None, ["every stage-1 trial failed"] + [
                f"{r.config.key()}: {r.error}" for r in self.results if r.error
            ]
        chosen = self.stage_memory(feasible, kept)
        self.stage_batch(feasible, chosen)
        if self.prefill_variants is not None:
            leader, _ = pick(self.results, self.objective, self.constraints)
            if leader is not None:
                self.stage_prefill(leader.config)
        for name, fn in self.variant_stages:
            self.stage_variants(name, fn)
        if self.power_points:
            leader, _ = pick(self.results, self.objective, self.constraints)
            if leader is not None:
                self.stage_power(leader.config.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None}))
        winner, notes = pick(self.results, self.objective, self.constraints)
        if winner is not None and self.power_points:
            pn = self._power_note(winner)
            if pn:
                notes.append(pn)
        return winner, self.notes + notes
