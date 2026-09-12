"""Multi-GPU layouts: independent replicas behind a load balancer, or one tensor-parallel engine.

With more than one GPU there are three ways to spend them, and which wins depends on the model:

  * replicas - one full engine per GPU, requests spread by a least-outstanding load balancer.
               No inter-GPU traffic at all, so for a model that fits on one card this is usually
               the throughput winner; it does nothing for a single request's latency.
  * tp       - one engine sharded across the GPUs (tensor parallel). Every layer all-reduces
               between cards, which costs bandwidth, but it frees memory per card for more KV
               cache and cuts per-token latency on models that are bandwidth bound per GPU.
  * disaggregated prefill/decode - see polyserve.disagg (`--phases`).

`--layout auto` measures replicas and tensor parallelism against the single-GPU winner and keeps
the best under the objective.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx
# Module level: with postponed annotations FastAPI resolves `request: Request` from module globals.
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from polyserve.backends.base import BaseBackend, Process
from polyserve.calibrate.measure import run_trial
from polyserve.calibrate.objectives import Constraints, pick
from polyserve.calibrate.workload import Workload
from polyserve.disagg import BackgroundServer, _free_ports, nvidia_gpus
from polyserve.models import Config, HardwareDescriptor, PreparedModel, Profile, TrialMetrics, TrialResult
from polyserve.serve.proxy import HOP_BY_HOP

logger = logging.getLogger(__name__)

LAYOUTS = ("single", "replicas", "tp", "auto")


def _strip(cfg: Config) -> Config:
    return cfg.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None, "tp": 1})


# --------------------------------------------------------------------------- load balancer


def create_lb_app(upstreams: List[str], profile: Optional[Profile] = None,
                  status_fn: Optional[Callable[[], Dict[str, Any]]] = None,
                  request_timeout: Optional[float] = None) -> FastAPI:
    """OpenAI-compatible front end over N identical engines: least outstanding requests wins."""
    if not upstreams:
        raise ValueError("load balancer needs at least one upstream")
    state: Dict[str, Any] = {"inflight": [0] * len(upstreams), "rr": itertools.count(), "served": [0] * len(upstreams)}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: Any):
        state["client"] = httpx.AsyncClient(timeout=request_timeout)
        try:
            yield
        finally:
            await state["client"].aclose()

    app = FastAPI(title="PolyServe (replicas)", lifespan=lifespan)

    def _choose() -> int:
        with lock:
            low = min(state["inflight"])
            tied = [i for i, n in enumerate(state["inflight"]) if n == low]
            i = tied[next(state["rr"]) % len(tied)]
            state["inflight"][i] += 1
            state["served"][i] += 1
            return i

    def _release(i: int) -> None:
        with lock:
            state["inflight"][i] = max(0, state["inflight"][i] - 1)

    app.state.lb = state

    @app.get("/health")
    async def health() -> JSONResponse:
        body: Dict[str, Any] = {"status": "ok", "replicas": upstreams}
        if status_fn is not None:
            body["backend"] = status_fn()
            if not body["backend"].get("alive", True):
                body["status"] = "degraded"
        return JSONResponse(body, status_code=200 if body["status"] == "ok" else 503)

    @app.get("/polyserve/profile")
    async def get_profile() -> JSONResponse:
        if profile is None:
            return JSONResponse({"error": "no profile"}, status_code=404)
        data = profile.model_dump(mode="json", exclude={"calibration_table", "hardware", "prepared", "prepared_all"})
        data["calibration_trials"] = len(profile.calibration_table)
        if status_fn is not None:
            data["runtime"] = status_fn()
        return JSONResponse(data)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def forward(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = state["client"]
        i = _choose()
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        req = client.build_request(request.method, f"{upstreams[i]}/v1/{path}", content=await request.body(),
                                   headers=headers, params=request.query_params)
        try:
            up = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            _release(i)
            return JSONResponse({"error": {"message": f"replica {i} unavailable: {exc}", "type": "upstream_error"}},
                                status_code=502)

        async def close() -> None:
            await up.aclose()
            _release(i)

        if "text/event-stream" in up.headers.get("content-type", ""):
            return StreamingResponse(up.aiter_raw(), status_code=up.status_code, media_type="text/event-stream",
                                     background=BackgroundTask(close))
        content = await up.aread()
        await close()
        return Response(content=content, status_code=up.status_code, media_type=up.headers.get("content-type"))

    return app


# --------------------------------------------------------------------------- replicas


@dataclasses.dataclass
class ReplicaSet:
    processes: List[Process]

    @property
    def urls(self) -> List[str]:
        return [f"http://127.0.0.1:{p.port}" for p in self.processes]

    def alive(self) -> bool:
        return all(p.alive() for p in self.processes)

    def stop(self) -> None:
        for p in self.processes:
            try:
                p.stop()
            except Exception as exc:  # pragma: no cover
                logger.error("stop failed: %s", exc)


def launch_replicas(backend: BaseBackend, cfg: Config, model: PreparedModel, gpus: List[int],
                    log_dir: Optional[Path] = None, startup_timeout: float = 900.0):
    """One engine per GPU. Returns (replica set, error message or None)."""
    ports = _free_ports(len(gpus))
    stamp = int(time.time())
    procs = []
    for gpu, port in zip(gpus, ports):
        spec = backend.replica_launch_spec(_strip(cfg), model, port, gpu)
        log = log_dir / f"{stamp}_replica{gpu}.log" if log_dir else None
        procs.append(Process(spec, port, f"http://127.0.0.1:{port}{backend.health_path}", log_path=log).start())
    rs = ReplicaSet(procs)
    if not all(p.wait_ready(timeout=startup_timeout) for p in procs):
        err = "replicas failed to start: " + ", ".join(f"GPU {g} rc={p.returncode()}" for g, p in zip(gpus, procs))
        tail = next((p.tail_log(10) for p in procs if p.tail_log(10)), "")
        rs.stop()
        return rs, f"{err}\n{tail}"
    return rs, None


class LayoutTrialRunner:
    """Measure N replicas end to end through the load balancer."""

    def __init__(self, backend: BaseBackend, model: PreparedModel, hw: HardwareDescriptor, workload: Workload,
                 log_dir: Optional[Path] = None, startup_timeout: float = 900.0, request_timeout: float = 180.0):
        self.backend = backend
        self.model = model
        self.hw = hw
        self.workload = workload
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout

    def run_replicas(self, cfg: Config, gpus: List[int], stage: str) -> TrialResult:
        cfg = _strip(cfg)
        rs, err = launch_replicas(self.backend, cfg, self.model, gpus, self.log_dir, self.startup_timeout)
        if err:
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), launched=False, error=err,
                               replicas=len(gpus))
        lb = None
        try:
            lb = BackgroundServer(create_lb_app(rs.urls, request_timeout=self.request_timeout)).start()
            hooks = self.backend.workload_hooks(self.hw, self.model)
            if hooks.gpu_ids:
                hooks = dataclasses.replace(hooks, gpu_ids=list(gpus))
            m = run_trial(lb.url, hooks, self.workload, pid=None, request_timeout=self.request_timeout)
            return TrialResult(config=cfg, stage=stage, metrics=m, replicas=len(gpus),
                               error=None if m.ok else f"{m.failed}/{m.requests} requests failed")
        except Exception as exc:
            logger.exception("replica trial crashed")
            return TrialResult(config=cfg, stage=stage, metrics=TrialMetrics(), error=str(exc), replicas=len(gpus))
        finally:
            if lb is not None:
                lb.stop()
            rs.stop()


class ReplicaSupervisor:
    """Serve N replicas, each kept alive by its own Supervisor pinned to one GPU."""

    def __init__(self, backend: BaseBackend, cfg: Config, model: PreparedModel, gpus: List[int],
                 log_dir: Optional[Path] = None, startup_timeout: float = 900.0, health_interval: float = 5.0):
        from polyserve.serve.supervisor import Supervisor

        self.gpus = gpus
        self.supervisors = [
            Supervisor(backend, _strip(cfg), model, log_path=(log_dir / f"serve-replica{g}.log") if log_dir else None,
                       startup_timeout=startup_timeout, health_interval=health_interval, gpu_index=g)
            for g in gpus
        ]

    @property
    def urls(self) -> List[str]:
        return [s.base_url for s in self.supervisors]

    def start(self) -> None:
        started = []
        try:
            for s in self.supervisors:
                s.start()
                started.append(s)
        except Exception:
            for s in started:
                s.stop()
            raise

    def healthy(self) -> bool:
        return all(s.healthy() for s in self.supervisors)

    def stop(self) -> None:
        for s in self.supervisors:
            s.stop()

    def status(self) -> Dict[str, Any]:
        per = [s.status() for s in self.supervisors]
        return {"layout": "replicas", "alive": all(p["alive"] for p in per), "replicas": per}


# --------------------------------------------------------------------------- calibration


def tp_candidates(cfg: Config, n: int) -> List[Config]:
    """Tensor-parallel shapes: the winner sharded n ways, and with the freed memory spent on batch."""
    base = _strip(cfg).model_copy(update={"tp": n})
    out = [base]
    doubled = base.model_copy(update={"batch": base.batch * 2})
    if doubled.prefill_budget is not None and doubled.prefill_budget < doubled.batch:
        doubled = doubled.model_copy(update={"prefill_budget": None})  # vLLM: budget must cover the batch
    out.append(doubled)
    return out


def _fallback(base: Profile, layout: str, reason: str) -> Profile:
    if layout != "auto":
        raise RuntimeError(f"--layout {layout} is not possible here: {reason}")
    out = base.model_copy(deep=True)
    out.layout = "auto"
    out.options = {**base.options, "layout": "auto"}
    out.notes = list(base.notes) + [f"--layout auto: {reason}; serving on one GPU"]
    return out


def _single_reference(profile: Profile) -> Optional[TrialResult]:
    key = profile.config.key()
    matches = [r for r in profile.calibration_table
               if r.ok and r.config.key() == key and r.disagg is None and r.replicas == 1]
    return max(matches, key=lambda r: r.metrics.tok_s) if matches else None


def calibrate_layout(
    hw: HardwareDescriptor,
    objective: str,
    base: Profile,
    reg: Dict[str, BaseBackend],
    layout: str,
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    progress: Optional[Callable] = None,
    replica_runner: Optional[Any] = None,
    tp_runner: Optional[Any] = None,
    log_dir: Optional[Path] = None,
) -> Profile:
    """Measure multi-GPU layouts around the single-GPU winner and decide."""
    from polyserve.calibrate.search import SubprocessTrialRunner
    from polyserve.calibrate.workload import get_workload
    from polyserve.memory import estimate

    if layout not in ("replicas", "tp", "auto"):
        raise ValueError("calibrate_layout needs layout 'replicas', 'tp' or 'auto'")
    workload = workload or get_workload(base.workload)
    constraints = constraints or Constraints(ttft_ceiling_ms=workload.ttft_ceiling_ms,
                                             tpot_ceiling_ms=workload.tpot_ceiling_ms)
    gpus = [g.index for g in nvidia_gpus(hw)]
    if len(gpus) < 2:
        return _fallback(base, layout, f"multi-GPU layouts need two NVIDIA GPUs; {len(gpus)} visible")
    backend = reg[base.backend]
    model = base.prepared_all.get(base.backend) or base.prepared
    if model is None:
        return _fallback(base, layout, "the single-GPU profile has no prepared model")
    if layout == "tp" and not backend.supports_tp:
        return _fallback(base, layout, f"{backend.name} has no tensor parallelism")

    results: List[TrialResult] = []
    t0 = time.monotonic()

    def _record(res: TrialResult) -> None:
        results.append(res)
        if progress:
            progress("layout", res.config, res)

    if layout in ("replicas", "auto"):
        runner = replica_runner or LayoutTrialRunner(backend, model, hw, workload, log_dir=log_dir)
        if progress:
            progress("layout", base.config, None)
        _record(runner.run_replicas(base.config, gpus, "layout"))
    if layout in ("tp", "auto") and backend.supports_tp:
        runner = tp_runner or SubprocessTrialRunner(backends={backend.name: backend}, models={backend.name: model},
                                                    hw=hw, workload=workload, log_dir=log_dir)
        mm = backend.memory_model(hw)
        for cfg in tp_candidates(base.config, len(gpus)):
            if not estimate(hw, model, cfg, mm).feasible:
                continue
            if progress:
                progress("layout", cfg, None)
            _record(runner.run(cfg, "layout"))
    elapsed = time.monotonic() - t0

    ok = [r for r in results if r.ok]
    if not ok:
        errors = "; ".join(sorted({(r.error or "failed").splitlines()[0] for r in results})) or "nothing to try"
        return _fallback(base, layout, f"every multi-GPU trial failed ({errors})")
    best, _ = pick(ok, objective, constraints)
    ref = _single_reference(base)
    chosen = best
    notes: List[str] = []
    if ref is not None:
        winner, _ = pick([ref] + ok, objective, constraints)
        if layout == "auto":
            chosen = winner
        gain = (best.metrics.tok_s - ref.metrics.tok_s) / ref.metrics.tok_s * 100 if ref.metrics.tok_s else 0.0
        shape = f"{best.replicas} replicas" if best.replicas > 1 else f"tp={best.config.tp}"
        verdict = "one GPU wins" if winner is ref else f"{shape} wins"
        notes.append(f"best multi-GPU layout {shape}: {best.metrics.tok_s:.0f} tok/s ({gain:+.1f}% vs one GPU "
                     f"{ref.metrics.tok_s:.0f}); {verdict} under {objective}")

    out = base.model_copy(deep=True)
    out.layout = layout
    out.options = {**base.options, "layout": layout}
    out.calibration_table = list(base.calibration_table) + results
    out.calibration_trials = base.calibration_trials + len(results)
    out.calibration_seconds = base.calibration_seconds + elapsed
    if chosen is not ref and chosen is not None:
        out.config = chosen.config
        out.replicas = chosen.replicas
        out.launch_args = backend.launch_spec(chosen.config, model, 0).args
    else:
        notes.append("auto: serving on one GPU, which measured better")
    out.notes = list(base.notes) + notes
    return out
