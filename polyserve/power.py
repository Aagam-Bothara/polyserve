"""GPU power and clock control for energy tuning.

LLM decode is memory-bandwidth bound: past a certain core clock the SMs wait on memory, so more
frequency buys almost no throughput while power keeps rising steeply. There is therefore a power
cap (or clock) below peak where joules per token is lowest. Two knobs are supported, both applied
through NVML while the backend is already running, so sweeping them needs no relaunch:

  * power cap     - nvmlDeviceSetPowerManagementLimit: the board power limit, in watts. Portable
                    across consumer and datacentre cards; the driver picks the clock under the cap.
  * locked clocks - nvmlDeviceSetGpuLockedClocks: pins the SM clock range, in MHz. Direct control
                    of frequency; support varies by card and driver.

Both are machine-wide and need root. Every change is recorded in a restore file before it is made
(~/.polyserve/power-restore.json), undone in a finally block, on interpreter exit and on SIGTERM,
and `polyserve power reset` restores it after a hard kill.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from polyserve.models import Config

logger = logging.getLogger(__name__)

MODES = ("off", "cap", "clock", "both")
# Fractions of the default power limit / max SM clock to try below 100%.
FRACTIONS = (0.85, 0.70, 0.55)


class PowerControlUnavailable(RuntimeError):
    """The requested power or clock change is not permitted or not supported on this machine."""


@dataclass(frozen=True)
class PowerSetting:
    power_limit_w: Optional[int] = None
    sm_clock_mhz: Optional[int] = None

    @property
    def is_default(self) -> bool:
        return self.power_limit_w is None and self.sm_clock_mhz is None

    def label(self) -> str:
        if self.is_default:
            return "default"
        parts = []
        if self.power_limit_w is not None:
            parts.append(f"cap {self.power_limit_w} W")
        if self.sm_clock_mhz is not None:
            parts.append(f"clock <= {self.sm_clock_mhz} MHz")
        return ", ".join(parts)


def setting_of(cfg: Config) -> PowerSetting:
    return PowerSetting(power_limit_w=cfg.power_limit_w, sm_clock_mhz=cfg.sm_clock_mhz)


def with_power(cfg: Config, setting: PowerSetting) -> Config:
    return cfg.model_copy(update={"power_limit_w": setting.power_limit_w, "sm_clock_mhz": setting.sm_clock_mhz})


@dataclass
class Capabilities:
    gpu_index: int
    power_limit_min_w: Optional[int] = None
    power_limit_max_w: Optional[int] = None
    power_limit_default_w: Optional[int] = None
    power_limit_current_w: Optional[int] = None
    sm_clocks_mhz: List[int] = field(default_factory=list)  # supported graphics clocks, descending
    sm_clock_max_mhz: Optional[int] = None
    can_cap: bool = False
    can_lock: bool = False
    reasons: Dict[str, str] = field(default_factory=dict)


class PowerController(Protocol):
    @property
    def applied(self) -> Optional[PowerSetting]: ...

    def capabilities(self) -> Capabilities: ...

    def apply(self, setting: PowerSetting) -> None: ...

    def restore(self) -> None: ...


# --------------------------------------------------------------------------- restore file


def restore_path() -> Path:
    return Path(os.environ.get("POLYSERVE_HOME", Path.home() / ".polyserve")) / "power-restore.json"


def _write_restore(state: Dict[str, Any]) -> None:
    p = restore_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clear_restore() -> None:
    try:
        restore_path().unlink()
    except FileNotFoundError:
        pass


def pending_restore() -> Optional[Dict[str, Any]]:
    p = restore_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- NVML implementation


def _load_nvml() -> Any:
    try:
        import pynvml  # type: ignore
    except ImportError as exc:
        raise PowerControlUnavailable("pynvml is not installed (pip install nvidia-ml-py)") from exc
    return pynvml


class NvmlPowerController:
    """Power cap and clock lock for one GPU through NVML. `nvml` is injectable for tests."""

    def __init__(self, gpu_index: int = 0, nvml: Any = None):
        self.gpu_index = gpu_index
        self._nvml = nvml
        self._handle: Any = None
        self._original_limit_mw: Optional[int] = None
        self._clocks_locked = False
        self._applied: Optional[PowerSetting] = None
        self._hooks_installed = False
        self._lock = threading.Lock()

    # ---- plumbing

    @property
    def applied(self) -> Optional[PowerSetting]:
        return self._applied

    def _lib(self) -> Any:
        if self._nvml is None:
            self._nvml = _load_nvml()
        if self._handle is None:
            try:
                self._nvml.nvmlInit()
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception as exc:
                raise PowerControlUnavailable(f"NVML unavailable: {exc}") from exc
        return self._nvml

    def _call(self, name: str, *args: Any) -> Any:
        nv = self._lib()
        try:
            return getattr(nv, name)(self._handle, *args)
        except Exception as exc:
            raise PowerControlUnavailable(_explain(name, exc)) from exc

    def _try(self, name: str, *args: Any) -> Any:
        try:
            return self._call(name, *args)
        except PowerControlUnavailable:
            return None

    # ---- query

    def capabilities(self) -> Capabilities:
        caps = Capabilities(gpu_index=self.gpu_index)
        nv = self._lib()
        cons = self._try("nvmlDeviceGetPowerManagementLimitConstraints")
        if cons:
            caps.power_limit_min_w, caps.power_limit_max_w = int(cons[0]) // 1000, int(cons[1]) // 1000
        dflt = self._try("nvmlDeviceGetPowerManagementDefaultLimit")
        if dflt:
            caps.power_limit_default_w = int(dflt) // 1000
        cur = self._try("nvmlDeviceGetPowerManagementLimit")
        if cur:
            caps.power_limit_current_w = int(cur) // 1000
        mems = self._try("nvmlDeviceGetSupportedMemoryClocks") or []
        if mems:
            clocks = self._try("nvmlDeviceGetSupportedGraphicsClocks", max(mems)) or []
            caps.sm_clocks_mhz = sorted({int(c) for c in clocks}, reverse=True)
        mx = self._try("nvmlDeviceGetMaxClockInfo", getattr(nv, "NVML_CLOCK_SM", 1))
        caps.sm_clock_max_mhz = int(mx) if mx else (caps.sm_clocks_mhz[0] if caps.sm_clocks_mhz else None)

        # Permission probes: both writes below leave the device exactly as it was.
        if cur:
            try:
                self._call("nvmlDeviceSetPowerManagementLimit", int(cur))
                caps.can_cap = True
            except PowerControlUnavailable as exc:
                caps.reasons["cap"] = str(exc)
        else:
            caps.reasons["cap"] = "power limit not reported by the driver"
        if caps.sm_clocks_mhz:
            try:
                self._call("nvmlDeviceSetGpuLockedClocks", caps.sm_clocks_mhz[-1], caps.sm_clocks_mhz[0])
                self._call("nvmlDeviceResetGpuLockedClocks")
                caps.can_lock = True
            except PowerControlUnavailable as exc:
                caps.reasons["clock"] = str(exc)
        else:
            caps.reasons["clock"] = "supported clocks not reported by the driver"
        return caps

    # ---- control

    def apply(self, setting: PowerSetting) -> None:
        with self._lock:
            if setting.is_default:
                self._restore_locked()
                return
            self._remember_original()
            self._install_hooks()
            if setting.power_limit_w is not None:
                self._call("nvmlDeviceSetPowerManagementLimit", int(setting.power_limit_w) * 1000)
            elif self._original_limit_mw is not None and self._applied and self._applied.power_limit_w is not None:
                self._call("nvmlDeviceSetPowerManagementLimit", self._original_limit_mw)
            if setting.sm_clock_mhz is not None:
                lo = self._min_clock()
                self._call("nvmlDeviceSetGpuLockedClocks", min(lo, setting.sm_clock_mhz), int(setting.sm_clock_mhz))
                self._clocks_locked = True
            elif self._clocks_locked:
                self._call("nvmlDeviceResetGpuLockedClocks")
                self._clocks_locked = False
            self._applied = setting
            logger.info("GPU %d power setting: %s", self.gpu_index, setting.label())

    def restore(self) -> None:
        with self._lock:
            self._restore_locked()

    def _restore_locked(self) -> None:
        if self._applied is None and not self._clocks_locked and pending_restore() is None:
            return
        errors = []
        if self._original_limit_mw is not None:
            try:
                self._call("nvmlDeviceSetPowerManagementLimit", self._original_limit_mw)
            except PowerControlUnavailable as exc:
                errors.append(str(exc))
        if self._clocks_locked or (self._applied and self._applied.sm_clock_mhz is not None):
            try:
                self._call("nvmlDeviceResetGpuLockedClocks")
            except PowerControlUnavailable as exc:
                errors.append(str(exc))
        self._clocks_locked = False
        self._applied = None
        if errors:
            logger.error("could not fully restore GPU %d power state: %s (run `polyserve power reset`)",
                         self.gpu_index, "; ".join(errors))
            return
        _clear_restore()
        logger.info("GPU %d power state restored", self.gpu_index)

    def _remember_original(self) -> None:
        if self._original_limit_mw is not None:
            return
        existing = pending_restore()
        if existing and existing.get("gpu_index") == self.gpu_index and existing.get("power_limit_mw"):
            # A previous run died without restoring; the file holds the true original, not the cap.
            self._original_limit_mw = int(existing["power_limit_mw"])
            return
        cur = self._call("nvmlDeviceGetPowerManagementLimit")
        self._original_limit_mw = int(cur)
        _write_restore({"gpu_index": self.gpu_index, "power_limit_mw": self._original_limit_mw})

    def _min_clock(self) -> int:
        mems = self._try("nvmlDeviceGetSupportedMemoryClocks") or []
        clocks = self._try("nvmlDeviceGetSupportedGraphicsClocks", max(mems)) if mems else None
        return min(int(c) for c in clocks) if clocks else 0

    def _install_hooks(self) -> None:
        if self._hooks_installed:
            return
        self._hooks_installed = True
        atexit.register(self.restore)
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            previous = signal.getsignal(signal.SIGTERM)

            def _on_term(signum: int, frame: Any) -> None:
                self.restore()
                if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                    previous(signum, frame)
                raise SystemExit(128 + signum)

            signal.signal(signal.SIGTERM, _on_term)
        except (ValueError, OSError, AttributeError):  # not the main thread, or no SIGTERM on this platform
            pass


def _explain(call: str, exc: Exception) -> str:
    name = type(exc).__name__
    if "NoPermission" in name or "permission" in str(exc).lower():
        return f"{call}: insufficient permissions (power and clock control need root on the host)"
    if "NotSupported" in name or "not supported" in str(exc).lower():
        return f"{call}: not supported on this GPU/driver"
    return f"{call}: {exc}"


def controller_for(gpu_index: int) -> NvmlPowerController:
    """Factory used by the CLI and pipeline; tests monkeypatch it."""
    return NvmlPowerController(gpu_index)


def reset_from_file(gpu_index: Optional[int] = None, nvml: Any = None) -> Optional[Dict[str, Any]]:
    """Undo a change left behind by a crashed run. Always unlocks clocks on the target GPU."""
    state = pending_restore()
    idx = gpu_index if gpu_index is not None else (state or {}).get("gpu_index", 0)
    ctl = NvmlPowerController(int(idx), nvml=nvml)
    if state and state.get("power_limit_mw"):
        ctl._call("nvmlDeviceSetPowerManagementLimit", int(state["power_limit_mw"]))
    ctl._try("nvmlDeviceResetGpuLockedClocks")
    _clear_restore()
    return state


# --------------------------------------------------------------------------- search points


def _snap(target: float, supported: Sequence[int]) -> int:
    return min(supported, key=lambda c: (abs(c - target), c))


def candidate_points(caps: Capabilities, mode: str, fractions: Sequence[float] = FRACTIONS) -> List[PowerSetting]:
    """Settings the power stage should try; the unchanged default is always first."""
    if mode not in MODES:
        raise ValueError(f"power mode must be one of {', '.join(MODES)}")
    points: List[PowerSetting] = [PowerSetting()]
    if mode in ("cap", "both") and caps.can_cap and caps.power_limit_default_w:
        dflt = caps.power_limit_default_w
        lo = caps.power_limit_min_w or 1
        hi = caps.power_limit_max_w or dflt
        for f in fractions:
            w = int(round(min(hi, max(lo, dflt * f))))
            if w < dflt:
                points.append(PowerSetting(power_limit_w=w))
    if mode in ("clock", "both") and caps.can_lock and caps.sm_clocks_mhz:
        top = caps.sm_clock_max_mhz or caps.sm_clocks_mhz[0]
        for f in fractions:
            mhz = _snap(top * f, caps.sm_clocks_mhz)
            if mhz < top:
                points.append(PowerSetting(sm_clock_mhz=mhz))
    seen, out = set(), []
    for p in points:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
