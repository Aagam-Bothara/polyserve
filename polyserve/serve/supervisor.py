"""Keep the winning backend alive: launch, health-check, restart with backoff."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from polyserve.backends.base import BaseBackend, Process, free_port
from polyserve.models import Config, PreparedModel

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(
        self,
        backend: BaseBackend,
        cfg: Config,
        model: PreparedModel,
        port: Optional[int] = None,
        log_path: Optional[Path] = None,
        startup_timeout: float = 900.0,
        health_interval: float = 5.0,
        max_restarts: int = 5,
        power: Optional[object] = None,
        gpu_index: Optional[int] = None,
    ):
        self.gpu_index = gpu_index  # set for one replica of the replicas layout
        self.backend = backend
        self.cfg = cfg
        self.model = model
        self.port = port or free_port()
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.health_interval = health_interval
        self.max_restarts = max_restarts
        self.process: Optional[Process] = None
        self.restarts = 0
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.power = power  # polyserve.power.PowerController when the profile carries a power setting
        self.power_error: Optional[str] = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_url(self) -> str:
        return self.base_url + self.backend.health_path

    def start(self) -> None:
        self._launch()
        self._apply_power()
        self._thread = threading.Thread(target=self._watch, name="polyserve-supervisor", daemon=True)
        self._thread.start()

    def _apply_power(self) -> None:
        """Serve at the calibrated power setting. Failure is not fatal: the server runs uncapped."""
        if self.cfg.power_limit_w is None and self.cfg.sm_clock_mhz is None:
            return
        from polyserve.power import setting_of

        if self.power is None:
            self.power_error = "profile has a power setting but no power controller was provided"
            logger.warning(self.power_error)
            return
        try:
            self.power.apply(setting_of(self.cfg))  # type: ignore[attr-defined]
        except Exception as exc:
            self.power_error = str(exc)
            logger.warning("serving without the calibrated power setting: %s", exc)

    def _launch(self) -> None:
        with self._lock:
            if self.gpu_index is not None:
                spec = self.backend.replica_launch_spec(self.cfg, self.model, self.port, self.gpu_index)
                self.process = Process(spec, self.port, self.health_url, log_path=self.log_path).start()
            else:
                self.process = self.backend.launch(self.cfg, self.model, self.port, log_path=self.log_path)
            if not self.process.wait_ready(timeout=self.startup_timeout):
                tail = self.process.tail_log(30)
                self.process.stop()
                raise RuntimeError(f"{self.backend.name} failed to start on port {self.port}\n{tail}")
            logger.info("%s ready on %s (pid %s)", self.backend.name, self.base_url, self.process.pid)

    def healthy(self) -> bool:
        if self.process is None or not self.process.alive():
            return False
        try:
            with httpx.Client(timeout=5.0) as c:
                return c.get(self.health_url).status_code < 500
        except httpx.HTTPError:
            return False

    def _watch(self) -> None:
        misses = 0
        while not self._stop.wait(self.health_interval):
            if self.healthy():
                misses = 0
                continue
            misses += 1
            if misses < 3 and self.process is not None and self.process.alive():
                continue  # transient; give it a couple of intervals
            if self.restarts >= self.max_restarts:
                self.last_error = f"backend died and restart budget ({self.max_restarts}) exhausted"
                logger.error(self.last_error)
                return
            self.restarts += 1
            delay = min(60.0, 2.0 ** self.restarts)
            logger.warning("backend unhealthy; restart %d/%d in %.0fs", self.restarts, self.max_restarts, delay)
            if self.process is not None:
                self.process.stop()
            time.sleep(delay)
            try:
                self._launch()
                misses = 0
            except Exception as exc:
                self.last_error = str(exc)
                logger.error("restart failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.process is not None:
            self.process.stop()
        if self.power is not None and getattr(self.power, "applied", None) is not None:
            try:
                self.power.restore()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.error("power restore failed: %s (run `polyserve power reset`)", exc)

    def status(self) -> dict:
        return {
            "backend": self.backend.name,
            "gpu": self.gpu_index,
            "port": self.port,
            "pid": self.process.pid if self.process else None,
            "alive": bool(self.process and self.process.alive()),
            "restarts": self.restarts,
            "last_error": self.last_error,
            "power": {
                "applied": (self.power.applied.label()  # type: ignore[attr-defined]
                            if self.power is not None and getattr(self.power, "applied", None) else None),
                "error": self.power_error,
            },
        }
