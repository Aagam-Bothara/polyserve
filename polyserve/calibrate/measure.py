"""Measure one trial: drive the workload over HTTP while sampling device telemetry.

Telemetry priority: llmtrace GPUSampler (NVML) -> bare pynvml -> psutil RSS (+ RAPL energy on Linux).
Request timing (TTFT / TPOT / tok/s) is measured at the HTTP client, which is the only vantage point
shared by all backends.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import queue
import statistics
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from polyserve.backends.base import LlmtraceHooks
from polyserve.calibrate.tokens import TokenCounter, worst_source
from polyserve.calibrate.workload import Workload
from polyserve.models import TrialMetrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- telemetry


@dataclass
class TelemetrySummary:
    source: str = "none"
    peak_mem_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    power_w: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    energy_j: Optional[float] = None
    samples: int = 0


@dataclass
class _Sample:
    t: float
    mem_mb: Optional[float] = None
    util: Optional[float] = None
    power_w: Optional[float] = None
    clock_mhz: Optional[float] = None


def _summarize(samples: List[_Sample], source: str, extra_energy: Optional[float] = None) -> TelemetrySummary:
    s = TelemetrySummary(source=source, samples=len(samples))
    mems = [x.mem_mb for x in samples if x.mem_mb is not None]
    utils = [x.util for x in samples if x.util is not None]
    powers = [(x.t, x.power_w) for x in samples if x.power_w is not None]
    clocks = [x.clock_mhz for x in samples if x.clock_mhz is not None]
    if clocks:
        s.sm_clock_mhz = statistics.fmean(clocks)
    if mems:
        s.peak_mem_mb = max(mems)
    if utils:
        s.gpu_util_pct = statistics.fmean(utils)
    if powers:
        s.power_w = statistics.fmean(p for _, p in powers)
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(powers, powers[1:]):
            energy += 0.5 * (p0 + p1) * max(0.0, t1 - t0)
        s.energy_j = energy if len(powers) > 1 else None
    if extra_energy is not None:
        s.energy_j = extra_energy
    return s


class Telemetry:
    """Context manager sampling GPU (llmtrace/pynvml) or process (psutil) telemetry."""

    def __init__(self, hooks: LlmtraceHooks, pid: Optional[int] = None, interval_ms: int = 100):
        self.hooks = hooks
        self.pid = pid
        self.interval_ms = interval_ms
        self._samples: List[_Sample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._llmtrace_sampler: Any = None
        self._source = "none"
        self._rapl_start: Optional[float] = None
        self.summary = TelemetrySummary()

    # ---- start/stop

    def __enter__(self) -> "Telemetry":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self.hooks.gpu_ids and self._start_llmtrace():
            return
        if self.hooks.gpu_ids and self._start_pynvml():
            return
        self._start_process()

    def stop(self) -> None:
        if self._llmtrace_sampler is not None:
            samp = self._llmtrace_sampler
            samp.stop()
            raw = samp.drain()
            samples = [
                _Sample(
                    t=(x.monotonic if x.monotonic is not None else x.timestamp),
                    mem_mb=x.memory_used_mb,
                    util=x.gpu_utilization_pct,
                    power_w=x.power_draw_watts,
                    clock_mhz=x.sm_clock_mhz,
                )
                for x in raw
            ]
            self.summary = _summarize(samples, "llmtrace")
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        extra_energy = None
        if self._rapl_start is not None:
            end = _rapl_energy_j()
            if end is not None and end >= self._rapl_start:
                extra_energy = end - self._rapl_start
        self.summary = _summarize(self._samples, self._source, extra_energy)

    # ---- backends

    def _start_llmtrace(self) -> bool:
        if importlib.util.find_spec("llmtrace") is None:
            return False
        try:
            from llmtrace.data_plane.gpu_sampler import GPUSampler  # type: ignore
            from llmtrace.models.config import GPUSamplerConfig  # type: ignore

            sampler = GPUSampler(GPUSamplerConfig(gpu_ids=list(self.hooks.gpu_ids), sample_interval_ms=self.interval_ms))
            sampler.start()
            if not sampler.available:
                logger.debug("llmtrace sampler unavailable: %s", sampler.unavailable_reason)
                return False
            self._llmtrace_sampler = sampler
            self._source = "llmtrace"
            return True
        except Exception as exc:
            logger.debug("llmtrace sampler failed: %s", exc)
            return False

    def _start_pynvml(self) -> bool:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.hooks.gpu_ids]
        except Exception:
            return False

        def read() -> _Sample:
            mem = util = power = clock = None
            try:
                mem = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles) / 2**20
                util = statistics.fmean(pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles)
                power = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in handles) / 1000.0
                clock = statistics.fmean(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM) for h in handles)
            except Exception:
                pass
            return _Sample(t=time.monotonic(), mem_mb=mem, util=util, power_w=power, clock_mhz=clock)

        self._source = "pynvml"
        self._run_thread(read, on_stop=lambda: pynvml.nvmlShutdown())
        return True

    def _start_process(self) -> None:
        proc = None
        if self.pid is not None:
            try:
                import psutil

                proc = psutil.Process(self.pid)
            except Exception:
                proc = None
        self._rapl_start = _rapl_energy_j()
        if proc is None and self._rapl_start is None:
            self._source = "none"
            return

        def read() -> _Sample:
            mem = None
            if proc is not None:
                try:
                    rss = proc.memory_info().rss
                    for child in proc.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                    mem = rss / 2**20
                except Exception:
                    pass
            return _Sample(t=time.monotonic(), mem_mb=mem)

        self._source = "psutil" if proc is not None else "rapl"
        self._run_thread(read)

    def _run_thread(self, read, on_stop=None) -> None:
        interval = self.interval_ms / 1000.0

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self._samples.append(read())
                except Exception as exc:  # pragma: no cover
                    logger.debug("telemetry read failed: %s", exc)
                self._stop.wait(interval)
            if on_stop:
                try:
                    on_stop()
                except Exception:
                    pass

        self._thread = threading.Thread(target=loop, name="polyserve-telemetry", daemon=True)
        self._thread.start()


def _rapl_energy_j() -> Optional[float]:
    """Package energy from Intel/AMD RAPL sysfs (Linux). Needs read permission; often root-only."""
    base = "/sys/class/powercap"
    if not os.path.isdir(base):
        return None
    total = 0.0
    found = False
    try:
        for name in os.listdir(base):
            if not name.startswith("intel-rapl:") or name.count(":") != 1:
                continue
            path = os.path.join(base, name, "energy_uj")
            with open(path, "r") as fh:
                total += int(fh.read().strip()) / 1e6
                found = True
    except (OSError, ValueError):
        return None
    return total if found else None


# --------------------------------------------------------------------------- HTTP driver


@dataclass
class RequestOutcome:
    ok: bool
    ttft_s: float = 0.0
    duration_s: float = 0.0
    tokens: int = 0
    token_source: str = "none"  # "usage" | "tokenizer" | "chunks"
    error: Optional[str] = None


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _one_request(
    client: httpx.AsyncClient, url: str, body: Dict[str, Any], timeout: float, counter: TokenCounter
) -> RequestOutcome:
    t0 = time.perf_counter()
    first: Optional[float] = None
    chunks = 0
    pieces: List[str] = []
    usage_tokens: Optional[int] = None
    try:
        async with client.stream("POST", url, json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")[:300]
                return RequestOutcome(ok=False, error=f"HTTP {resp.status_code}: {text}")
            async for line in resp.aiter_lines():
                obj = _parse_sse_line(line)
                if obj is None:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    ch = choices[0]
                    text = ch.get("text")
                    if text is None and isinstance(ch.get("delta"), dict):
                        text = ch["delta"].get("content")
                    if text:
                        if first is None:
                            first = time.perf_counter()
                        chunks += 1
                        pieces.append(text)
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        return RequestOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
    end = time.perf_counter()
    if first is None:
        return RequestOutcome(ok=False, error="no tokens produced")
    # Exact count: server usage > model tokenizer > chunk count (approximate).
    if usage_tokens:
        tokens, source = usage_tokens, "usage"
    else:
        counted = counter.count("".join(pieces))
        if counted:
            tokens, source = counted, "tokenizer"
        else:
            tokens, source = chunks, "chunks"
    return RequestOutcome(ok=True, ttft_s=first - t0, duration_s=end - t0, tokens=tokens, token_source=source)


async def _drive(
    base_url: str,
    hooks: LlmtraceHooks,
    workload: Workload,
    concurrency: int,
    timeout: float,
    counter: Optional[TokenCounter] = None,
) -> List[RequestOutcome]:
    counter = counter or TokenCounter()
    sem = asyncio.Semaphore(concurrency)
    url = base_url.rstrip("/") + hooks.completions_path
    results: List[RequestOutcome] = []

    async with httpx.AsyncClient() as client:

        async def worker(prompt: str) -> None:
            body: Dict[str, Any] = {
                "model": hooks.model_name or "model",
                "prompt": prompt,
                "max_tokens": workload.decode_tokens,
                "temperature": workload.temperature,
                "stream": True,
                # Fixed-length answers for synthetic prompts; real-text workloads stop when the model does.
                "ignore_eos": not workload.natural_stop,
            }
            if hooks.stream_usage:
                body["stream_options"] = {"include_usage": True}
            async with sem:
                results.append(await _one_request(client, url, body, timeout, counter))

        await asyncio.gather(*(worker(p) for p in workload.prompts))
    return results


def _metrics_from(outcomes: List[RequestOutcome], wall_s: float, concurrency: int) -> TrialMetrics:
    ok = [o for o in outcomes if o.ok]
    m = TrialMetrics(
        requests=len(outcomes),
        failed=len(outcomes) - len(ok),
        concurrency=concurrency,
        duration_s=wall_s,
        output_tokens=sum(o.tokens for o in ok),
    )
    # A few distinct errors, so "12/48 requests failed" says why.
    m.errors = list(dict.fromkeys((o.error or "")[:200] for o in outcomes if not o.ok and o.error))[:3]
    if not ok:
        return m
    m.token_count_source = worst_source(o.token_source for o in ok)
    ttfts = sorted(o.ttft_s * 1000 for o in ok)
    m.ttft_ms = statistics.median(ttfts)
    m.ttft_p95_ms = ttfts[min(len(ttfts) - 1, int(round(0.95 * (len(ttfts) - 1))))]
    m.ttft_samples_ms = [round(t, 1) for t in ttfts]
    tpots = [(o.duration_s - o.ttft_s) * 1000 / (o.tokens - 1) for o in ok if o.tokens > 1]
    m.tpot_ms = statistics.fmean(tpots) if tpots else None
    m.tok_s = m.output_tokens / wall_s if wall_s > 0 else 0.0
    return m


def _level_workload(workload: Workload, k: int, counter: TokenCounter) -> Workload:
    """The prompts for concurrency level k: fresh ones after the first level.

    Replaying the first level's prompts lets the engine's prefix cache serve every later level's
    prefill almost for free: on an A40 one configuration read 3638 tok/s with replayed prompts and
    1814 with fresh ones. A workload's shared prefix is kept; caching that is the point.
    """
    if k == 0:
        return workload
    level = replace(workload, seed=workload.seed + 7_919 * k, prompts=[], fitted=False,
                    prefix_text=workload.prefix_text, prefix_fixed=True,
                    sample_offset=workload.sample_offset + k * workload.n_prompts).ensure_prompts()
    if counter.available:
        level.fit_prompts(counter)
    return level


def _client_main(index: int, base_url: str, hooks: LlmtraceHooks, workload: Workload, concurrency: int,
                 timeout: float, ready: Any, go: Any, out: Any) -> None:
    """One load-generator process (see _drive_clients). Module level so it pickles under spawn."""
    ready.put(index)
    go.wait()
    out.put((index, asyncio.run(_drive(base_url, hooks, workload, concurrency, timeout))))


def _drive_clients(base_url: str, hooks: LlmtraceHooks, workload: Workload, concurrency: int, timeout: float,
                   clients: int) -> Tuple[List[RequestOutcome], float]:
    """Drive one level from several processes, with the prompts and the concurrency split between them.

    One Python client tops out near 3000 streamed tokens/s (through the replica balancer on two A40s:
    2954 tok/s from one client, 3541 from four). The clock starts once every process is up, so
    process start-up is not measured. A process that dies counts its requests as failed.
    """
    import multiprocessing as mp

    shards = [s for s in (workload.prompts[i::clients] for i in range(clients)) if s]
    per = max(1, -(-concurrency // len(shards)))
    ctx = mp.get_context("spawn")
    ready, out, go = ctx.Queue(), ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_client_main, daemon=True,
                         args=(i, base_url, hooks, replace(workload, prompts=s, n_prompts=len(s), fitted=True),
                               per, timeout, ready, go, out))
             for i, s in enumerate(shards)]
    for p in procs:
        p.start()
    started, deadline = 0, time.monotonic() + 180
    while started < len(procs) and time.monotonic() < deadline:
        try:
            ready.get(timeout=1)
            started += 1
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                break
    t0 = time.perf_counter()
    go.set()
    results: Dict[int, List[RequestOutcome]] = {}
    while len(results) < len(procs):
        try:
            i, outcomes = out.get(timeout=5)
            results[i] = outcomes
        except queue.Empty:
            if not any(p.is_alive() for p in procs) and out.empty():
                break
    wall = time.perf_counter() - t0
    for p in procs:
        p.join(timeout=10)
    merged: List[RequestOutcome] = []
    for i, s in enumerate(shards):
        merged += results.get(i) or [RequestOutcome(ok=False, error="load-generator process died")] * len(s)
    return merged, wall


def run_trial(
    base_url: str,
    hooks: LlmtraceHooks,
    workload: Workload,
    pid: Optional[int] = None,
    request_timeout: float = 120.0,
    warmup: bool = True,
    counter: Optional[TokenCounter] = None,
    clients: int = 1,
    enough: Optional[Callable[[TrialMetrics], bool]] = None,
    close_call: Optional[Callable[[TrialMetrics], bool]] = None,
) -> TrialMetrics:
    """Run the workload at each concurrency level and fold into one TrialMetrics.

    Every level gets fresh prompts (see _level_workload). `clients` > 1 drives each level from that
    many processes, for layouts whose combined throughput a single Python client cannot keep up with.

    `enough` (see objectives.enough_level) measures the levels highest first and stops at the first one
    it accepts: when the objective scores a trial by its throughput, a lower level cannot beat a level
    that already meets the constraints. Re-scored this way, 115 recorded trials in six calibrations on
    an A40 kept every score and every pick. A level keeps its own prompts whatever the order.

    `close_call` (see objectives.close_call_level) flags a level whose tail is too close to its ceiling to call
    from its requests; that level is measured again with as many requests on fresh prompts, inside the same
    telemetry window, and judged on both runs together.

    Summary rule: tok/s and TTFT/TPOT come from the concurrency level with the highest
    throughput (that is the load the server would actually be run at); energy per token
    is total joules / total tokens across all levels; peak memory is the max.
    """
    if counter is None:
        counter = TokenCounter.for_model(hooks.tokenizer_id)
    workload.ensure_prompts()
    if counter.available and not workload.fitted:
        workload.fit_prompts(counter)
    prompt_tokens = workload.measured_prompt_tokens(counter) or 0
    if warmup:
        # One short request per slot at the busiest level. Kernels that compile on first use (Triton attention,
        # the sampler, speculative decoding) otherwise compile during the first batched level measured: on an
        # A40 a draft model's first batched level had a p95 time to first token of 1.3-1.6 s and the level
        # after it 0.12-0.18 s, and vLLM 0.29 logged "Triton kernel JIT compilation during inference". Once
        # the busiest level ran first, that spike landed on the level that is scored.
        top = max(workload.concurrencies)
        small = Workload(
            n_prompts=max(2, top), prefill_tokens=workload.prefill_tokens,
            decode_tokens=min(16, workload.decode_tokens), concurrencies=(top,), seed=workload.seed + 1,
            # Same shared prefix as the real prompts, so the warmup leaves it cached as production would.
            shared_prefix_tokens=workload.shared_prefix_tokens, prefix_text=workload.prefix_text, prefix_fixed=True,
        )
        try:
            asyncio.run(_drive(base_url, hooks, small, top, request_timeout, counter))
        except Exception as exc:
            logger.debug("warmup failed: %s", exc)

    per_level: Dict[str, TrialMetrics] = {}
    total_energy = 0.0
    energy_known = True
    peak_mem: Optional[float] = None
    utils: List[float] = []
    powers: List[float] = []
    clocks: List[float] = []
    source = "none"

    levels = list(enumerate(workload.concurrencies))
    if enough is not None:
        levels.sort(key=lambda kc: -kc[1])
    for k, c in levels:
        level = _level_workload(workload, k, counter)
        n = level.level_requests(c)
        if n < len(level.prompts):
            level = replace(level, prompts=level.prompts[:n], n_prompts=n)
        def drive(lw: Workload) -> Tuple[List[RequestOutcome], float]:
            if clients > 1:
                return _drive_clients(base_url, hooks, lw, c, request_timeout, clients)
            t0 = time.perf_counter()
            got = asyncio.run(_drive(base_url, hooks, lw, c, request_timeout, counter))
            return got, time.perf_counter() - t0

        resampled = False
        with Telemetry(hooks, pid=pid) as tel:
            outcomes, wall = drive(level)
            if close_call is not None and close_call(_metrics_from(outcomes, wall, c)):
                # Too close to the ceiling to call from these requests: as many again, on prompts of their own.
                extra = _level_workload(workload, k + len(workload.concurrencies), counter)
                if n < len(extra.prompts):
                    extra = replace(extra, prompts=extra.prompts[:n], n_prompts=n)
                more, more_wall = drive(extra)
                outcomes, wall, resampled = outcomes + more, wall + more_wall, True
        m = _metrics_from(outcomes, wall, c)
        m.resampled = resampled
        s = tel.summary
        source = s.source if s.source != "none" else source
        m.telemetry_source = s.source
        m.prompt_tokens = prompt_tokens
        m.peak_mem_mb = s.peak_mem_mb
        m.gpu_util_pct = s.gpu_util_pct
        m.power_w = s.power_w
        m.sm_clock_mhz = s.sm_clock_mhz
        if s.sm_clock_mhz is not None:
            clocks.append(s.sm_clock_mhz)
        if s.energy_j is not None and m.output_tokens:
            m.joules_per_token = s.energy_j / m.output_tokens
            total_energy += s.energy_j
        else:
            energy_known = False
        if s.peak_mem_mb is not None:
            peak_mem = max(peak_mem or 0.0, s.peak_mem_mb)
        if s.gpu_util_pct is not None:
            utils.append(s.gpu_util_pct)
        if s.power_w is not None:
            powers.append(s.power_w)
        per_level[str(c)] = m
        if enough is not None and m.ok and enough(m):
            break

    best = max(per_level.values(), key=lambda x: x.tok_s)
    total_tokens = sum(x.output_tokens for x in per_level.values())
    summary = TrialMetrics(
        tok_s=best.tok_s,
        ttft_ms=best.ttft_ms,
        ttft_p95_ms=best.ttft_p95_ms,
        tpot_ms=best.tpot_ms,
        peak_mem_mb=peak_mem,
        gpu_util_pct=statistics.fmean(utils) if utils else None,
        power_w=statistics.fmean(powers) if powers else None,
        sm_clock_mhz=statistics.fmean(clocks) if clocks else None,
        joules_per_token=(total_energy / total_tokens) if (energy_known and total_tokens) else None,
        duration_s=sum(x.duration_s for x in per_level.values()),
        requests=sum(x.requests for x in per_level.values()),
        failed=sum(x.failed for x in per_level.values()),
        output_tokens=total_tokens,
        concurrency=best.concurrency,
        telemetry_source=source,
        token_count_source=worst_source(x.token_count_source for x in per_level.values()),
        prompt_tokens=prompt_tokens,
        by_concurrency=per_level,
    )
    return summary
