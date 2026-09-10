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
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from polyserve.backends.base import LlmtraceHooks
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
    energy_j: Optional[float] = None
    samples: int = 0


@dataclass
class _Sample:
    t: float
    mem_mb: Optional[float] = None
    util: Optional[float] = None
    power_w: Optional[float] = None


def _summarize(samples: List[_Sample], source: str, extra_energy: Optional[float] = None) -> TelemetrySummary:
    s = TelemetrySummary(source=source, samples=len(samples))
    mems = [x.mem_mb for x in samples if x.mem_mb is not None]
    utils = [x.util for x in samples if x.util is not None]
    powers = [(x.t, x.power_w) for x in samples if x.power_w is not None]
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
            mem = util = power = None
            try:
                mem = sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in handles) / 2**20
                util = statistics.fmean(pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles)
                power = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in handles) / 1000.0
            except Exception:
                pass
            return _Sample(t=time.monotonic(), mem_mb=mem, util=util, power_w=power)

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
    client: httpx.AsyncClient, url: str, body: Dict[str, Any], timeout: float
) -> RequestOutcome:
    t0 = time.perf_counter()
    first: Optional[float] = None
    tokens = 0
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
                        tokens += 1
                usage = obj.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        return RequestOutcome(ok=False, error=f"{type(exc).__name__}: {exc}")
    end = time.perf_counter()
    if first is None:
        return RequestOutcome(ok=False, error="no tokens produced")
    return RequestOutcome(
        ok=True, ttft_s=first - t0, duration_s=end - t0, tokens=usage_tokens or tokens
    )


async def _drive(
    base_url: str, hooks: LlmtraceHooks, workload: Workload, concurrency: int, timeout: float
) -> List[RequestOutcome]:
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
                "ignore_eos": True,  # vLLM/SGLang honour this; llama.cpp ignores unknown fields
            }
            if hooks.stream_usage:
                body["stream_options"] = {"include_usage": True}
            async with sem:
                results.append(await _one_request(client, url, body, timeout))

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
    if not ok:
        return m
    ttfts = sorted(o.ttft_s * 1000 for o in ok)
    m.ttft_ms = statistics.median(ttfts)
    m.ttft_p95_ms = ttfts[min(len(ttfts) - 1, int(round(0.95 * (len(ttfts) - 1))))]
    tpots = [(o.duration_s - o.ttft_s) * 1000 / (o.tokens - 1) for o in ok if o.tokens > 1]
    m.tpot_ms = statistics.fmean(tpots) if tpots else None
    m.tok_s = m.output_tokens / wall_s if wall_s > 0 else 0.0
    return m


def run_trial(
    base_url: str,
    hooks: LlmtraceHooks,
    workload: Workload,
    pid: Optional[int] = None,
    request_timeout: float = 120.0,
    warmup: bool = True,
) -> TrialMetrics:
    """Run the workload at each concurrency level and fold into one TrialMetrics.

    Summary rule: tok/s and TTFT/TPOT come from the concurrency level with the highest
    throughput (that is the load the server would actually be run at); energy per token
    is total joules / total tokens across all levels; peak memory is the max.
    """
    if warmup:
        small = Workload(
            n_prompts=min(2, workload.n_prompts), prefill_tokens=workload.prefill_tokens,
            decode_tokens=min(16, workload.decode_tokens), concurrencies=(1,), seed=workload.seed + 1,
        )
        try:
            asyncio.run(_drive(base_url, hooks, small, 1, request_timeout))
        except Exception as exc:
            logger.debug("warmup failed: %s", exc)

    per_level: Dict[str, TrialMetrics] = {}
    total_energy = 0.0
    energy_known = True
    peak_mem: Optional[float] = None
    utils: List[float] = []
    powers: List[float] = []
    source = "none"

    for c in workload.concurrencies:
        with Telemetry(hooks, pid=pid) as tel:
            t0 = time.perf_counter()
            outcomes = asyncio.run(_drive(base_url, hooks, workload, c, request_timeout))
            wall = time.perf_counter() - t0
        m = _metrics_from(outcomes, wall, c)
        s = tel.summary
        source = s.source if s.source != "none" else source
        m.telemetry_source = s.source
        m.peak_mem_mb = s.peak_mem_mb
        m.gpu_util_pct = s.gpu_util_pct
        m.power_w = s.power_w
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
        joules_per_token=(total_energy / total_tokens) if (energy_known and total_tokens) else None,
        duration_s=sum(x.duration_s for x in per_level.values()),
        requests=sum(x.requests for x in per_level.values()),
        failed=sum(x.failed for x in per_level.values()),
        output_tokens=total_tokens,
        concurrency=best.concurrency,
        telemetry_source=source,
        by_concurrency=per_level,
    )
    return summary
