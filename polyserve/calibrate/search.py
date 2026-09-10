"""Staged search over the feasible configuration set.

1. quant / precision  - one short run per feasible (backend, quant); keep the top 2.
2. memory config      - largest safe gpu_memory_utilization / offload layers / context.
3. batch / concurrency - sweep max_num_seqs (vLLM/SGLang) or n_parallel/n_batch (llama.cpp).

The winner is the objective's constrained argmax over *every* successful trial, not just stage 3.
"""

from __future__ import annotations

import logging
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
    ):
        self.backends = backends
        self.models = models
        self.hw = hw
        self.workload = workload
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout

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
            safe = cfg.key().replace("/", "_")
            log_path = self.log_dir / f"{int(time.time())}_{safe}.log"
        hooks = backend.workload_hooks(self.hw, model)
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
                err = f"{metrics.failed}/{metrics.requests} requests failed"
            return TrialResult(config=cfg, stage=stage, metrics=metrics, error=err, memory=observation)
        except Exception as exc:
            logger.exception("trial %s crashed", cfg.key())
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=str(exc),
                               memory=observation)
        finally:
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
    """
    ctxs = sorted({c.ctx for c in group})
    batches = sorted({c.batch for c in group})
    ctx = ctxs[len(ctxs) // 2]
    batch = batches[len(batches) // 2]
    same = [c for c in group if c.ctx == ctx and c.batch == batch]
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
    results: List[TrialResult] = field(default_factory=list)
    _done: Dict[str, TrialResult] = field(default_factory=dict)

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
        stage_results: Dict[Tuple[str, str], TrialResult] = {}
        for key, group in groups.items():
            # Largest memory setting first; step down only if the launch itself fails.
            for cfg in _baseline_candidates(group)[: self.max_memory_trials_per_quant]:
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
            variants.sort(key=lambda c: (c.batch, c.n_batch or 0))
            for c in variants:
                self._run(c, "batch")

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
        winner, notes = pick(self.results, self.objective, self.constraints)
        return winner, notes
