"""What PolyServe's staged search is measured against: a fixed rule of thumb, and random search given the same time.

The search space is the one calibration explores: every configuration the memory planner kept (engine, precision,
context, batch, memory reserved), crossed with the prefill budgets, quantized KV caches, speculative methods and
prefix-cache settings the engines offer, keeping only combinations that can run. Random search draws from it in
a seeded order until a typical trial would overrun its budget, then picks by the same objective and limits; the
rule of thumb is one configuration chosen without measuring anything. `benchmarks/search_baselines.py` runs both
on a GPU and compares their picks with PolyServe's.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from polyserve.calibrate.objectives import Constraints, pick
from polyserve.calibrate.search import TrialRunner
from polyserve.calibrate.workload import Workload
from polyserve.models import Config, HardwareDescriptor, TrialResult


def search_space(hw: HardwareDescriptor, plan: Any, reg: Dict[str, Any], workload: Workload,
                 options: Any = None) -> List[Config]:
    """Every runnable configuration calibration could reach, each once: the planner's configurations, then each
    stage's variants of everything so far (prefill budgets, quantized KV caches with the batch step they allow,
    prefix-cache settings, speculative methods), so every combination of them is included."""
    from polyserve.pipeline import SearchOptions, combination_fits, kv_variants_fn

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


def rule_of_thumb(hw: HardwareDescriptor, plan: Any, reg: Dict[str, Any]) -> Optional[Config]:
    """What a knowledgeable user might set by hand, as one launch with no measurement: vLLM (else SGLang, else the
    first engine that fits), fp8 weights where the card runs them (bf16 otherwise), the shortest context the
    workload allows rather than the model's full window, batch as close to 256 as fits, 90% of memory reserved,
    and the engine's 8-bit KV cache where it has one. No speculative decoding: a draft helps a quiet server and
    costs throughput at the batch sizes chosen here."""
    from polyserve.pipeline import combination_fits

    fits = combination_fits(hw, reg, plan)
    names = [b for b in ("vllm", "sglang") if plan.feasible.get(b)] or [b for b, cs in plan.feasible.items() if cs]
    if not names:
        return None
    be = reg[names[0]]
    configs = [c for c, _ in plan.feasible[names[0]]]
    quants = {c.quant for c in configs}
    quant = next((q for q in ("fp8", "bf16", "fp16") if q in quants), sorted(quants)[0])
    cfg = min((c for c in configs if c.quant == quant),
              key=lambda c: (abs(c.batch - 256), abs((c.gpu_memory_utilization or 0.9) - 0.9), c.ctx))
    kvs = list(be.kv_dtypes(hw))
    if kvs and fits(cfg.model_copy(update={"kv_dtype": kvs[0]})):
        cfg = cfg.model_copy(update={"kv_dtype": kvs[0]})
    return cfg


@dataclass
class SearchRun:
    seed: int
    results: List[TrialResult]
    seconds: float
    untried: int  # configurations of the space left when the budget ran out
    winner: Optional[TrialResult]


def random_search(space: Sequence[Config], runner: TrialRunner, objective: str, cons: Constraints, budget_s: float,
                  seed: int = 0, clock: Callable[[], float] = time.monotonic) -> SearchRun:
    """Measure configurations of `space` in a seeded random order until a typical trial would end past the budget
    (the first always runs), then pick the best by the objective, as calibration does."""
    order = random.Random(seed).sample(list(space), len(space))
    results: List[TrialResult] = []
    durations: List[float] = []
    start = clock()
    for cfg in order:
        if durations and clock() + sum(durations) / len(durations) > start + budget_s:
            break
        t0 = clock()
        results.append(runner.run(cfg, f"random:{seed}"))
        durations.append(clock() - t0)
    winner = pick(results, objective, cons)[0] if results else None
    return SearchRun(seed=seed, results=results, seconds=clock() - start, untried=len(order) - len(results),
                     winner=winner)
