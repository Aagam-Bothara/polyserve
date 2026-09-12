"""Disaggregated prefill and decode: two vLLM engines on two GPUs, joined by KV-cache transfer.

Prefill is compute bound and decode is memory-bandwidth bound, so a single engine configuration,
and a single GPU clock, is always a compromise between them. Disaggregation gives each phase its
own GPU and its own settings: the prefill engine runs a large chunked-prefill budget and few
concurrent sequences; the decode engine runs many sequences and a small prefill budget, and with
`--power` it is the GPU that gets capped, since decode is where extra clock buys nothing.

Request flow, as implemented by vLLM V1 KV connectors such as NixlConnector:

  1. The router sends the request to the prefill engine with max_tokens=1, stream=False and
     kv_transfer_params={"do_remote_decode": true, ...}. The engine computes the KV cache and
     returns kv_transfer_params naming the blocks for the decode engine to pull.
  2. The router sends the original request plus those kv_transfer_params to the decode engine,
     which pulls the KV cache over the connector and generates; its stream goes to the client.

Time to first token therefore includes the prefill, the KV transfer and the first decode step,
which is exactly what the client-side measurement sees through the router.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx
# Module level on purpose: with postponed annotations FastAPI resolves `request: Request` from the
# module's globals, so a function-local import leaves it unresolved and every call returns 422.
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from polyserve.backends.base import BaseBackend, Process
from polyserve.calibrate.measure import run_trial
from polyserve.calibrate.objectives import Constraints, pick
from polyserve.calibrate.workload import Workload
from polyserve.models import Config, DisaggSpec, GPUInfo, HardwareDescriptor, PreparedModel, Profile, TrialMetrics, TrialResult
from polyserve.power import PowerSetting, setting_of, with_power

logger = logging.getLogger(__name__)

PHASES = ("unified", "disaggregated", "auto")

# KV transfer connectors: the vLLM --kv-transfer-config each engine gets, and the Python package it
# needs. Anything else can be passed verbatim with a kv_transfer_config override.
CONNECTORS: Dict[str, Dict[str, Any]] = {
    "nixl": {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
}
CONNECTOR_PACKAGES: Dict[str, str] = {"nixl": "nixl"}

# Pool shapes tried during calibration, around the unified winner.
PREFILL_POOL_BUDGETS: Tuple[int, ...] = (8192, 16384)
PREFILL_POOL_MAX_SEQS = 64
DECODE_POOL_BUDGET = 2048
DECODE_POOL_MAX_SEQS = 256


# --------------------------------------------------------------------------- support


def kv_config(connector: str, override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if override:
        return dict(override)
    if connector not in CONNECTORS:
        raise ValueError(f"unknown KV connector {connector!r}; known: {', '.join(CONNECTORS)}")
    return dict(CONNECTORS[connector])


def connector_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def nvidia_gpus(hw: HardwareDescriptor) -> List[GPUInfo]:
    return [g for g in hw.gpus if g.vendor == "nvidia"]


def check_support(hw: HardwareDescriptor, backend: str, connector: str = "nixl",
                  override: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """None if disaggregated serving can run here, else the reason it cannot."""
    if backend != "vllm":
        return f"disaggregated serving needs vLLM, but the calibrated backend is {backend}"
    n = len(nvidia_gpus(hw))
    if n < 2:
        return f"disaggregated serving needs two NVIDIA GPUs; {n} visible"
    if override is None:
        if connector not in CONNECTORS:
            return f"unknown KV connector {connector!r}; known: {', '.join(CONNECTORS)}"
        pkg = CONNECTOR_PACKAGES.get(connector)
        if pkg and not connector_available(pkg):
            return f"the {connector} KV connector needs the `{pkg}` package (pip install {pkg})"
    return None


# --------------------------------------------------------------------------- candidate pairs


def _strip_power(cfg: Config) -> Config:
    return cfg.model_copy(update={"power_limit_w": None, "sm_clock_mhz": None})


def _fits(hw: HardwareDescriptor, gpu: GPUInfo, backend: BaseBackend, model: PreparedModel, cfg: Config) -> bool:
    from polyserve.memory import estimate

    one = hw.model_copy(update={"gpus": [gpu]})
    try:
        return estimate(one, model, cfg, backend.memory_model(one)).feasible
    except Exception as exc:  # no estimate for this config: let the launch decide
        logger.debug("no memory estimate for %s on GPU %d: %s", cfg.key(), gpu.index, exc)
        return True


def candidate_pairs(hw: HardwareDescriptor, backend: BaseBackend, model: PreparedModel, base: Config,
                    connector: str = "nixl", override: Optional[Dict[str, Any]] = None) -> List[DisaggSpec]:
    """Prefill/decode pool shapes to try, each checked against the memory of its own GPU."""
    gpus = nvidia_gpus(hw)
    if len(gpus) < 2:
        return []
    p_gpu, d_gpu = gpus[0], gpus[1]
    base = _strip_power(base)
    prefills = [base.model_copy(update={"batch": min(base.batch, PREFILL_POOL_MAX_SEQS), "prefill_budget": b})
                for b in PREFILL_POOL_BUDGETS]
    decodes = [base.model_copy(update={"batch": b, "prefill_budget": DECODE_POOL_BUDGET})
               for b in sorted({base.batch, DECODE_POOL_MAX_SEQS})]
    kv = kv_config(connector, override)
    out: List[DisaggSpec] = []
    for p in prefills:
        if not _fits(hw, p_gpu, backend, model, p):
            continue
        for d in decodes:
            if _fits(hw, d_gpu, backend, model, d):
                out.append(DisaggSpec(prefill=p, decode=d, prefill_gpu=p_gpu.index, decode_gpu=d_gpu.index,
                                      connector=connector, kv_transfer_config=kv))
    return out


def with_decode_power(spec: DisaggSpec, setting: PowerSetting) -> DisaggSpec:
    return spec.model_copy(update={"decode": with_power(spec.decode, setting)})


# --------------------------------------------------------------------------- router

_REMOTE_DECODE = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def prefill_request_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """The request as the prefill engine should see it: compute KV, emit one token, keep the KV."""
    pre = dict(body)
    pre["kv_transfer_params"] = dict(_REMOTE_DECODE)
    pre["stream"] = False
    pre["max_tokens"] = 1
    if "max_completion_tokens" in pre:
        pre["max_completion_tokens"] = 1
    pre.pop("stream_options", None)
    return pre


def decode_request_body(body: Dict[str, Any], prefill_response: Dict[str, Any]) -> Dict[str, Any]:
    """The original request plus the KV handle the prefill engine returned."""
    dec = dict(body)
    kv = prefill_response.get("kv_transfer_params")
    if kv:
        dec["kv_transfer_params"] = kv
    return dec


def create_pd_app(prefill_url: str, decode_url: str, profile: Optional[Profile] = None,
                  status_fn: Optional[Callable[[], Dict[str, Any]]] = None,
                  request_timeout: Optional[float] = None):
    """OpenAI-compatible front end that splits every completion across the two engines."""
    state: Dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: Any):
        state["client"] = httpx.AsyncClient(timeout=request_timeout)
        try:
            yield
        finally:
            await state["client"].aclose()

    app = FastAPI(title="PolyServe (disaggregated prefill/decode)", lifespan=lifespan)

    def _err(msg: str, code: int = 502) -> JSONResponse:
        logger.warning("prefill/decode router: %s", msg)
        return JSONResponse({"error": {"message": msg, "type": "upstream_error"}}, status_code=code)

    @app.get("/health")
    async def health() -> JSONResponse:
        body: Dict[str, Any] = {"status": "ok", "prefill": prefill_url, "decode": decode_url}
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

    @app.get("/v1/models")
    async def models() -> Response:
        try:
            r = await state["client"].get(decode_url + "/v1/models")
        except httpx.HTTPError as exc:
            return _err(f"decode engine unavailable: {exc}")
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    async def _split(path: str, request: Request) -> Response:
        client: httpx.AsyncClient = state["client"]
        body = await request.json()
        headers = {"X-Request-Id": request.headers.get("x-request-id") or uuid.uuid4().hex}
        if request.headers.get("authorization"):
            headers["Authorization"] = request.headers["authorization"]
        try:
            pre = await client.post(prefill_url + path, json=prefill_request_body(body), headers=headers)
        except httpx.HTTPError as exc:
            return _err(f"prefill engine unavailable: {exc}")
        if pre.status_code != 200:
            logger.warning("prefill engine returned HTTP %d: %s", pre.status_code, pre.text[:300])
            return Response(content=pre.content, status_code=pre.status_code,
                            media_type=pre.headers.get("content-type"))
        try:
            pre_json = pre.json()
        except ValueError:
            return _err("prefill engine returned a non-JSON response")
        req = client.build_request("POST", decode_url + path, json=decode_request_body(body, pre_json),
                                   headers=headers)
        try:
            up = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            return _err(f"decode engine unavailable: {exc}")
        if "text/event-stream" in up.headers.get("content-type", ""):
            return StreamingResponse(up.aiter_raw(), status_code=up.status_code, media_type="text/event-stream",
                                     background=BackgroundTask(up.aclose))
        content = await up.aread()
        await up.aclose()
        return Response(content=content, status_code=up.status_code, media_type=up.headers.get("content-type"))

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _split("/v1/completions", request)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        return await _split("/v1/chat/completions", request)

    return app


class ProcessServer:
    """Run a router (`python -m polyserve.router`) in its own process for a calibration trial.

    Not a thread: the load generator runs in the measuring process, and a router sharing its GIL
    caps the very throughput it is there to measure.
    """

    def __init__(self, kind: str, urls: Sequence[str], request_timeout: Optional[float] = None,
                 port: Optional[int] = None, log_path: Optional[Path] = None):
        self.port = port or _free_ports(1)[0]
        self.argv = [sys.executable, "-m", "polyserve.router", kind, "--port", str(self.port)]
        if request_timeout is not None:
            self.argv += ["--timeout", str(request_timeout)]
        self.argv += list(urls)
        self._proc: Optional[subprocess.Popen] = None
        # Keep the router's log next to the engines' when there is a log directory: its warnings are
        # the only record of why a request failed between the engines.
        self._log = open(log_path, "w+b") if log_path else tempfile.TemporaryFile()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 60.0) -> "ProcessServer":
        self._proc = subprocess.Popen(self.argv, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self._proc.poll() is None:
            try:
                httpx.get(self.url + "/health", timeout=1.0)
                return self
            except httpx.HTTPError:
                time.sleep(0.1)
        self._log.seek(0)
        tail = self._log.read().decode(errors="replace")[-800:]
        self.stop()
        raise RuntimeError(f"router failed to start: {tail}")

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._log.close()


# --------------------------------------------------------------------------- processes


def _free_ports(n: int) -> List[int]:
    """n distinct free ports: sockets stay bound until all are allocated, so none repeats."""
    socks, ports = [], []
    try:
        for _ in range(n):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            socks.append(s)
            ports.append(s.getsockname()[1])
    finally:
        for s in socks:
            s.close()
    return ports


@dataclasses.dataclass
class PDProcesses:
    prefill: Process
    decode: Process

    @property
    def prefill_url(self) -> str:
        return f"http://127.0.0.1:{self.prefill.port}"

    @property
    def decode_url(self) -> str:
        return f"http://127.0.0.1:{self.decode.port}"

    def alive(self) -> bool:
        return self.prefill.alive() and self.decode.alive()

    def stop(self) -> None:
        for p in (self.decode, self.prefill):
            try:
                p.stop()
            except Exception as exc:  # pragma: no cover
                logger.error("stop failed: %s", exc)


def launch_pair(backend: BaseBackend, spec: DisaggSpec, model: PreparedModel, log_dir: Optional[Path] = None,
                startup_timeout: float = 900.0) -> Tuple[PDProcesses, Optional[str]]:
    """Start both engines on their GPUs. Returns (processes, error message or None)."""
    p_port, d_port, p_side, d_side = _free_ports(4)
    p_ls = backend.disagg_launch_spec(_strip_power(spec.prefill), model, p_port, "prefill", spec.kv_transfer_config,
                                      spec.prefill_gpu, p_side)
    d_ls = backend.disagg_launch_spec(_strip_power(spec.decode), model, d_port, "decode", spec.kv_transfer_config,
                                      spec.decode_gpu, d_side)
    stamp = int(time.time())
    logs = (log_dir / f"{stamp}_pd_prefill.log", log_dir / f"{stamp}_pd_decode.log") if log_dir else (None, None)
    hp = backend.health_path
    pre = Process(p_ls, p_port, f"http://127.0.0.1:{p_port}{hp}", log_path=logs[0]).start()
    dec = Process(d_ls, d_port, f"http://127.0.0.1:{d_port}{hp}", log_path=logs[1]).start()
    pair = PDProcesses(pre, dec)
    if not (pre.wait_ready(timeout=startup_timeout) and dec.wait_ready(timeout=startup_timeout)):
        err = (f"engines failed to start (prefill rc={pre.returncode()}, decode rc={dec.returncode()})\n"
               f"{pre.tail_log(10) or dec.tail_log(10)}")
        pair.stop()
        return pair, err
    return pair, None


class DisaggTrialRunner:
    """Measure a prefill/decode pair end to end through the router, with the unified driver."""

    def __init__(self, backend: BaseBackend, model: PreparedModel, hw: HardwareDescriptor, workload: Workload,
                 log_dir: Optional[Path] = None, startup_timeout: float = 900.0, request_timeout: float = 180.0,
                 power_for_gpu: Optional[Callable[[int], Any]] = None, power_settle_s: float = 2.0):
        self.backend = backend
        self.model = model
        self.hw = hw
        self.workload = workload
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.power_for_gpu = power_for_gpu
        self.power_settle_s = power_settle_s

    def _hooks(self, spec: DisaggSpec):
        hooks = self.backend.workload_hooks(self.hw, self.model)
        if hooks.gpu_ids:  # energy is the whole system's: sample both GPUs
            hooks = dataclasses.replace(hooks, gpu_ids=[spec.prefill_gpu, spec.decode_gpu])
        return hooks

    def run(self, spec: DisaggSpec, stage: str) -> TrialResult:
        return self.sweep(spec, [setting_of(spec.decode)], stage)[0]

    def sweep(self, spec: DisaggSpec, settings: Sequence[PowerSetting], stage: str,
              progress: Optional[Callable] = None) -> List[TrialResult]:
        """Launch the pair once and measure it under each power setting of the decode GPU."""
        base = spec.model_copy(update={"decode": _strip_power(spec.decode), "prefill": _strip_power(spec.prefill)})
        pair, err = launch_pair(self.backend, base, self.model, self.log_dir, self.startup_timeout)
        if err:
            return [TrialResult(config=with_decode_power(base, s).decode, stage=stage, metrics=TrialMetrics(),
                                launched=False, error=err, disagg=with_decode_power(base, s)) for s in settings]
        router = None
        ctl = None
        results: List[TrialResult] = []
        try:
            router = ProcessServer("pd", [pair.prefill_url, pair.decode_url], request_timeout=self.request_timeout,
                                   log_path=(self.log_dir / f"{int(time.time())}_router.log") if self.log_dir
                                   else None).start()
            hooks = self._hooks(base)
            for s in settings:
                variant = with_decode_power(base, s)
                if progress:
                    progress(stage, variant.decode, None)
                error: Optional[str] = None
                if not s.is_default:
                    if ctl is None and self.power_for_gpu is not None:
                        ctl = self.power_for_gpu(base.decode_gpu)
                    if ctl is None:
                        error = "power setting requested but no power controller is configured"
                    else:
                        try:
                            ctl.apply(s)
                            if self.power_settle_s > 0:
                                time.sleep(self.power_settle_s)
                        except Exception as exc:
                            error = f"power control unavailable: {exc}"
                elif ctl is not None and getattr(ctl, "applied", None) is not None:
                    ctl.restore()
                if error:
                    res = TrialResult(config=variant.decode, stage=stage, metrics=TrialMetrics(), error=error,
                                      disagg=variant)
                else:
                    try:
                        m = run_trial(router.url, hooks, self.workload, pid=None,
                                      request_timeout=self.request_timeout, clients=2)
                        res = TrialResult(config=variant.decode, stage=stage, metrics=m, disagg=variant,
                                          error=None if m.ok else m.failure_summary())
                    except Exception as exc:
                        logger.exception("disaggregated trial crashed")
                        res = TrialResult(config=variant.decode, stage=stage, metrics=TrialMetrics(), error=str(exc),
                                          disagg=variant)
                results.append(res)
                if progress:
                    progress(stage, variant.decode, res)
            return results
        finally:
            if ctl is not None and getattr(ctl, "applied", None) is not None:
                ctl.restore()
            if router is not None:
                router.stop()
            pair.stop()


# --------------------------------------------------------------------------- serving


class DisaggSupervisor:
    """Keep both engines alive; restart the pair together, since the KV connector ties them."""

    def __init__(self, backend: BaseBackend, spec: DisaggSpec, model: PreparedModel, log_dir: Optional[Path] = None,
                 startup_timeout: float = 900.0, health_interval: float = 5.0, max_restarts: int = 5,
                 power: Optional[Any] = None):
        self.backend = backend
        self.spec = spec
        self.model = model
        self.log_dir = log_dir
        self.startup_timeout = startup_timeout
        self.health_interval = health_interval
        self.max_restarts = max_restarts
        self.power = power  # controller for the decode GPU, when the profile caps it
        self.power_error: Optional[str] = None
        self.pair: Optional[PDProcesses] = None
        self.restarts = 0
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def prefill_url(self) -> str:
        assert self.pair is not None
        return self.pair.prefill_url

    @property
    def decode_url(self) -> str:
        assert self.pair is not None
        return self.pair.decode_url

    def start(self) -> None:
        self._launch()
        s = setting_of(self.spec.decode)
        if not s.is_default:
            if self.power is None:
                self.power_error = "profile caps the decode GPU but no power controller was provided"
            else:
                try:
                    self.power.apply(s)
                except Exception as exc:
                    self.power_error = str(exc)
                    logger.warning("serving without the calibrated decode power setting: %s", exc)
        self._thread = threading.Thread(target=self._watch, name="polyserve-pd-supervisor", daemon=True)
        self._thread.start()

    def _launch(self) -> None:
        pair, err = launch_pair(self.backend, self.spec, self.model, self.log_dir, self.startup_timeout)
        if err:
            raise RuntimeError(err)
        self.pair = pair
        logger.info("prefill engine on GPU %d (%s), decode engine on GPU %d (%s)", self.spec.prefill_gpu,
                    pair.prefill_url, self.spec.decode_gpu, pair.decode_url)

    def healthy(self) -> bool:
        if self.pair is None or not self.pair.alive():
            return False
        try:
            with httpx.Client(timeout=5.0) as c:
                return all(c.get(u + self.backend.health_path).status_code < 500
                           for u in (self.pair.prefill_url, self.pair.decode_url))
        except httpx.HTTPError:
            return False

    def _watch(self) -> None:
        misses = 0
        while not self._stop.wait(self.health_interval):
            if self.healthy():
                misses = 0
                continue
            misses += 1
            if misses < 3 and self.pair is not None and self.pair.alive():
                continue
            if self.restarts >= self.max_restarts:
                self.last_error = f"engine died and restart budget ({self.max_restarts}) exhausted"
                logger.error(self.last_error)
                return
            self.restarts += 1
            logger.warning("prefill/decode pair unhealthy; restart %d/%d", self.restarts, self.max_restarts)
            if self.pair is not None:
                self.pair.stop()
            time.sleep(min(60.0, 2.0 ** self.restarts))
            try:
                self._launch()
                misses = 0
            except Exception as exc:
                self.last_error = str(exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.pair is not None:
            self.pair.stop()
        if self.power is not None and getattr(self.power, "applied", None) is not None:
            try:
                self.power.restore()
            except Exception as exc:
                logger.error("power restore failed: %s (run `polyserve power reset`)", exc)

    def status(self) -> Dict[str, Any]:
        pair = self.pair
        return {
            "phases": "disaggregated",
            "backend": self.backend.name,
            "alive": bool(pair and pair.alive()),
            "prefill": {"gpu": self.spec.prefill_gpu, "pid": pair.prefill.pid if pair else None,
                        "config": self.spec.prefill.key()},
            "decode": {"gpu": self.spec.decode_gpu, "pid": pair.decode.pid if pair else None,
                       "config": self.spec.decode.key()},
            "restarts": self.restarts,
            "last_error": self.last_error,
            "power": {
                "applied": (self.power.applied.label()
                            if self.power is not None and getattr(self.power, "applied", None) else None),
                "error": self.power_error,
            },
        }


# --------------------------------------------------------------------------- calibration


def _unified_reference(profile: Profile) -> Optional[TrialResult]:
    key = profile.config.key()
    matches = [r for r in profile.calibration_table if r.ok and r.config.key() == key and r.disagg is None]
    return max(matches, key=lambda r: r.metrics.tok_s) if matches else None


def _fallback(base: Profile, phases: str, reason: str) -> Profile:
    if phases == "disaggregated":
        raise RuntimeError(f"--phases disaggregated is not possible here: {reason}")
    out = base.model_copy(deep=True)
    out.phases = phases
    out.notes = list(base.notes) + [f"--phases {phases}: {reason}; serving unified"]
    return out


def calibrate_disaggregated(
    hw: HardwareDescriptor,
    spec: Any,
    objective: str,
    base: Profile,
    reg: Dict[str, BaseBackend],
    workload: Optional[Workload] = None,
    constraints: Optional[Constraints] = None,
    phases: str = "disaggregated",
    connector: str = "nixl",
    kv_override: Optional[Dict[str, Any]] = None,
    progress: Optional[Callable] = None,
    runner: Optional[Any] = None,
    power_mode: str = "off",
    power_controller: Optional[Any] = None,
    log_dir: Optional[Path] = None,
) -> Profile:
    """Measure prefill/decode pairs built around the unified winner, then decide.

    `phases="disaggregated"` serves the best pair even if unified measured faster (the note says so);
    `phases="auto"` keeps whichever wins under the objective.
    """
    from polyserve.calibrate.workload import get_workload

    if phases not in ("disaggregated", "auto"):
        raise ValueError("calibrate_disaggregated needs phases 'disaggregated' or 'auto'")
    workload = workload or get_workload(base.workload)
    constraints = constraints or Constraints(ttft_ceiling_ms=workload.ttft_ceiling_ms,
                                             tpot_ceiling_ms=workload.tpot_ceiling_ms)
    reason = check_support(hw, base.backend, connector, kv_override)
    if reason:
        return _fallback(base, phases, reason)
    backend = reg[base.backend]
    model = base.prepared_all.get(base.backend) or base.prepared
    if model is None:
        return _fallback(base, phases, "the unified profile has no prepared model")
    pairs = candidate_pairs(hw, backend, model, base.config, connector, kv_override)
    if not pairs:
        return _fallback(base, phases, "no prefill/decode pair fits in the two GPUs' memory")

    if runner is None:
        from polyserve.power import controller_for

        runner = DisaggTrialRunner(backend, model, hw, workload, log_dir=log_dir,
                                   power_for_gpu=(lambda i: power_controller or controller_for(i))
                                   if power_mode != "off" else None)
    t0 = time.monotonic()
    results: List[TrialResult] = []
    for p in pairs:
        if progress:
            progress("pd", p.decode, None)
        res = runner.run(p, "pd")
        results.append(res)
        if progress:
            progress("pd", p.decode, res)
    best, _ = pick(results, objective, constraints)
    if best is None:
        errors = "; ".join(sorted({(r.error or "failed").splitlines()[0] for r in results}))
        return _fallback(base, phases, f"every disaggregated trial failed ({errors})")

    notes: List[str] = [f"disaggregated: tried {len(pairs)} prefill/decode pairs on GPUs "
                        f"{pairs[0].prefill_gpu} (prefill) and {pairs[0].decode_gpu} (decode)"]
    if power_mode != "off":
        from polyserve.power import candidate_points, controller_for

        sweep = getattr(runner, "sweep", None)
        try:
            ctl = power_controller or controller_for(best.disagg.decode_gpu)
            points = candidate_points(ctl.capabilities(), power_mode)
        except Exception as exc:
            points, sweep = [], None
            notes.append(f"decode-GPU power stage skipped: {exc}")
        if sweep is not None and len(points) > 1:
            results += sweep(best.disagg, points, "pd-power", progress=progress)
            best, _ = pick(results, objective, constraints)
            notes.append(f"decode-GPU power setting: {setting_of(best.config).label()}")

    ref = _unified_reference(base)
    chosen = best
    if ref is not None:
        winner, _ = pick([ref] + results, objective, constraints)
        if phases == "auto":
            chosen = winner
        d, u = best.metrics, ref.metrics
        gain = (d.tok_s - u.tok_s) / u.tok_s * 100 if u.tok_s else 0.0
        verdict = "disaggregated wins" if winner.disagg is not None else "unified wins"
        notes.append(f"best pair {best.disagg.key()}: {d.tok_s:.0f} tok/s ({gain:+.1f}% vs unified "
                     f"{u.tok_s:.0f}); {verdict} under {objective}")
    elapsed = time.monotonic() - t0

    out = base.model_copy(deep=True)
    out.phases = phases
    out.power_mode = power_mode
    out.calibration_table = list(base.calibration_table) + results
    out.calibration_trials = base.calibration_trials + len(results)
    out.calibration_seconds = base.calibration_seconds + elapsed
    if chosen.disagg is not None:
        out.disagg = chosen.disagg
        out.config = chosen.disagg.decode
        out.launch_args = backend.disagg_launch_spec(chosen.disagg.decode, model, 0, "decode",
                                                     chosen.disagg.kv_transfer_config, chosen.disagg.decode_gpu, 0).args
    else:
        notes.append("auto: serving unified, which measured better")
    out.notes = list(base.notes) + notes
    return out
