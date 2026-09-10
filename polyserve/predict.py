"""Performance predictor: tok/s, TTFT and TPOT for a config *before* launching it.

A roofline model with three fitted parameters per backend:

    decode step (n sequences in flight):
        t_mem   = (device_weights + kv_read) / (alpha * mem_bw)          # weight + KV traffic
        t_comp  = n * 2 * params / (beta * peak_flops)                   # matmul work
        t_step  = max(t_mem, t_comp) + overhead                          # launch / scheduler cost
    prefill (n prompts of L_p tokens, compute bound):
        t_pre   = n * L_p * 2 * params / (beta * peak_flops) + device_weights / (alpha * mem_bw)
    a wave of n requests decoding L_d tokens:
        t_wave  = t_pre + L_d * t_step
        tok/s   = n * L_d / t_wave
    TTFT p50 at client concurrency c with B server slots (n = min(c, B)):
        own prefill (chunked prefill interleaves requests, so a request pays for its own prompt):
        t_own   = L_p * 2 * params / (beta * peak_flops)
        waves queued ahead of the median request = floor((c // 2) / n)
        TTFT    = t_own + queued * t_wave
    TPOT    = t_step

alpha (memory-bandwidth efficiency), beta (compute efficiency) and overhead (seconds per step) are
fitted per backend from measured trials by grid search on log error. Untrained backends use
priors, and every prediction says whether it came from fitted or prior parameters.

The predictor's job is ranking, not oracle accuracy: it lets the staged search skip quants and
batch settings that cannot win, and it is evaluated by leave-one-out MAPE and Spearman rank
correlation across the recorded trials (`polyserve fit`).
"""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from polyserve.models import Config, HardwareDescriptor, PreparedModel, Profile

GB = 1e9

# (memory bandwidth GB/s, dense fp16/bf16 tensor TFLOPS). Substring match on the NVML name, first hit wins.
GPU_SPECS: List[Tuple[str, float, float]] = [
    ("H100", 3350.0, 989.0),
    ("H200", 4800.0, 989.0),
    ("A100-SXM", 2039.0, 312.0),
    ("A100 80GB PCIe", 1935.0, 312.0),
    ("A100", 1555.0, 312.0),
    ("A30", 933.0, 165.0),
    ("A10", 600.0, 125.0),
    ("L40S", 864.0, 362.0),
    ("L40", 864.0, 181.0),
    ("L4", 300.0, 121.0),
    ("RTX 6000 Ada", 960.0, 364.0),
    ("RTX A6000", 768.0, 155.0),
    ("RTX A5000", 768.0, 92.0),
    ("RTX A4000", 448.0, 76.0),
    ("RTX 4090", 1008.0, 165.0),
    ("RTX 4080", 717.0, 97.0),
    ("RTX 4070", 504.0, 58.0),
    ("RTX 3090 Ti", 1008.0, 80.0),
    ("RTX 3090", 936.0, 71.0),
    ("RTX 3080", 760.0, 60.0),
    ("RTX 3070", 448.0, 41.0),
    ("RTX 3060", 360.0, 26.0),
    ("V100", 900.0, 125.0),
    ("T4", 320.0, 65.0),
    ("GTX 1080 Ti", 484.0, 11.3),
    ("GTX 1080", 320.0, 8.9),
    ("GTX 1070", 256.0, 6.5),
    ("P100", 732.0, 19.0),
]


@dataclass
class DeviceSpec:
    name: str
    mem_bw_gbs: float
    tflops: float
    known: bool  # False when estimated from a rule rather than the table

    @property
    def flops(self) -> float:
        return self.tflops * 1e12

    @property
    def mem_bw(self) -> float:
        return self.mem_bw_gbs * GB


def device_spec(hw: HardwareDescriptor) -> DeviceSpec:
    if hw.gpu is not None:
        name = hw.gpu.name
        for key, bw, tf in GPU_SPECS:
            if key.lower() in name.lower():
                return DeviceSpec(name=name, mem_bw_gbs=bw, tflops=tf, known=True)
        # Unknown GPU: scale by VRAM as a crude proxy (24 GB consumer ~ 900 GB/s, 80 GB ~ 2000 GB/s).
        gib = hw.gpu.vram_total_bytes / 2**30
        return DeviceSpec(name=name, mem_bw_gbs=max(200.0, min(3000.0, gib * 30)), tflops=max(8.0, gib * 3), known=False)
    c = hw.cpu
    bw = 150.0 if c.logical_cores >= 64 else 80.0 if c.logical_cores >= 32 else 40.0
    flops_per_cycle = 32 if c.avx512 else 16
    tflops = c.physical_cores * 2.5e9 * flops_per_cycle / 1e12
    return DeviceSpec(name=c.model_name, mem_bw_gbs=bw, tflops=tflops, known=False)


# --------------------------------------------------------------------------- parameters


@dataclass
class PerfParams:
    alpha: float = 0.7  # achieved / peak memory bandwidth
    beta: float = 0.4  # achieved / peak compute
    overhead_s: float = 0.004  # per decode step
    fitted: bool = False
    n: int = 0
    mape_tok_s: Optional[float] = None
    mape_ttft: Optional[float] = None
    spearman_tok_s: Optional[float] = None


PRIORS: Dict[str, PerfParams] = {
    "vllm": PerfParams(alpha=0.75, beta=0.45, overhead_s=0.003),
    "sglang": PerfParams(alpha=0.75, beta=0.45, overhead_s=0.003),
    "llamacpp-cuda": PerfParams(alpha=0.65, beta=0.30, overhead_s=0.006),
    "llamacpp-cpu": PerfParams(alpha=0.50, beta=0.30, overhead_s=0.010),
    "vllm-cpu": PerfParams(alpha=0.50, beta=0.30, overhead_s=0.010),
}


def prior(backend: str) -> PerfParams:
    p = PRIORS.get(backend, PerfParams())
    return PerfParams(alpha=p.alpha, beta=p.beta, overhead_s=p.overhead_s)


def model_path() -> Path:
    return Path(os.environ.get("POLYSERVE_HOME", Path.home() / ".polyserve")) / "perf-model.json"


def load_params(hardware_hash: str) -> Dict[str, PerfParams]:
    p = model_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8")).get(hardware_hash, {})
    except Exception:
        return {}
    out = {}
    for backend, rec in data.items():
        out[backend] = PerfParams(alpha=rec["alpha"], beta=rec["beta"], overhead_s=rec["overhead_s"], fitted=True,
                                  n=rec.get("n", 0), mape_tok_s=rec.get("mape_tok_s"), mape_ttft=rec.get("mape_ttft"),
                                  spearman_tok_s=rec.get("spearman_tok_s"))
    return out


def save_params(hardware_hash: str, params: Dict[str, PerfParams]) -> Path:
    p = model_path()
    data: Dict[str, Dict] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    entry = data.setdefault(hardware_hash, {})
    for backend, pp in params.items():
        if not pp.fitted:
            continue
        entry[backend] = {"alpha": round(pp.alpha, 4), "beta": round(pp.beta, 4), "overhead_s": round(pp.overhead_s, 6),
                          "n": pp.n, "mape_tok_s": pp.mape_tok_s, "mape_ttft": pp.mape_ttft,
                          "spearman_tok_s": pp.spearman_tok_s}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def params_for(hardware_hash: str, backend: str) -> PerfParams:
    return load_params(hardware_hash).get(backend) or prior(backend)


# --------------------------------------------------------------------------- the model


@dataclass
class Prediction:
    tok_s: float
    ttft_ms: float
    tpot_ms: float
    concurrency: int
    in_flight: int
    memory_bound: bool
    fitted: bool


@dataclass
class Situation:
    """Everything the model needs, decoupled from PreparedModel so trials can be replayed from JSON."""

    params_count: int
    kv_bytes_per_token: int
    device_weights_bytes: int
    host_weights_bytes: int = 0
    prefill_tokens: int = 256
    decode_tokens: int = 128


def situation(model: PreparedModel, cfg: Config, prefill_tokens: int, decode_tokens: int) -> Situation:
    from polyserve.hfconfig import dtype_bytes

    weights = model.weights_for(cfg.quant)
    frac = 1.0
    if cfg.n_gpu_layers is not None and model.arch.num_layers:
        frac = min(1.0, max(0.0, cfg.n_gpu_layers / model.arch.num_layers))
    kv_dtype = cfg.kv_dtype if cfg.kv_dtype != "auto" else model.arch.torch_dtype
    return Situation(
        params_count=model.arch.num_params or 0,
        kv_bytes_per_token=model.arch.kv_bytes_per_token(dtype_bytes(kv_dtype)),
        device_weights_bytes=int(weights * frac),
        host_weights_bytes=int(weights * (1 - frac)),
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
    )


HOST_BW = 40.0 * GB  # host memory bandwidth for CPU-resident layers (llama.cpp partial offload)


def predict(sit: Situation, cfg: Config, dev: DeviceSpec, params: PerfParams, concurrency: int,
            device_is_cpu: bool = False) -> Prediction:
    n = max(1, min(concurrency, cfg.batch))
    bw = params.alpha * dev.mem_bw
    flops = params.beta * dev.flops
    lp, ld = sit.prefill_tokens, sit.decode_tokens
    weights_time = sit.device_weights_bytes / bw + (sit.host_weights_bytes / (params.alpha * HOST_BW) if not device_is_cpu else 0)
    kv_read = n * sit.kv_bytes_per_token * (lp + ld / 2)
    t_mem = weights_time + kv_read / bw
    t_comp = n * 2 * sit.params_count / flops
    t_step = max(t_mem, t_comp) + params.overhead_s
    t_pre = n * lp * 2 * sit.params_count / flops + weights_time
    t_wave = t_pre + ld * t_step
    tok_s = n * ld / t_wave
    t_own = lp * 2 * sit.params_count / flops
    queued = (concurrency // 2) // n if concurrency > n else 0
    ttft = t_own + queued * t_wave
    return Prediction(tok_s=tok_s, ttft_ms=ttft * 1000, tpot_ms=t_step * 1000, concurrency=concurrency, in_flight=n,
                      memory_bound=t_mem >= t_comp, fitted=params.fitted)


# --------------------------------------------------------------------------- observations & fitting


@dataclass
class Observation:
    backend: str
    config: Config
    situation: Situation
    concurrency: int
    tok_s: float
    ttft_ms: Optional[float]
    device_is_cpu: bool = False


def observations_from_profile(profile: Profile) -> List[Observation]:
    """One observation per (trial, concurrency level) with measured tok/s."""
    out: List[Observation] = []
    pm = profile.prepared
    if pm is None:
        return out
    spec = profile.workload_spec or {}
    lp = int(spec.get("prefill_tokens", 256))
    ld = int(spec.get("decode_tokens", 128))
    for t in profile.calibration_table:
        if not t.ok or t.config.backend != pm.backend and t.config.backend != profile.backend:
            pass
        if not t.ok or t.config.quant not in pm.weights_bytes:
            continue
        try:
            sit = situation(pm, t.config, lp, ld)
        except Exception:
            continue
        levels = list(t.metrics.by_concurrency.values()) or [t.metrics]
        for m in levels:
            if m.tok_s > 0:
                out.append(Observation(backend=t.config.backend, config=t.config, situation=sit,
                                       concurrency=m.concurrency, tok_s=m.tok_s, ttft_ms=m.ttft_ms,
                                       device_is_cpu=t.config.backend.endswith("-cpu")))
    return out


def _loss(obs: Sequence[Observation], dev: DeviceSpec, p: PerfParams, ttft_weight: float = 0.5) -> float:
    total = 0.0
    for o in obs:
        pr = predict(o.situation, o.config, dev, p, o.concurrency, o.device_is_cpu)
        total += math.log(max(pr.tok_s, 1e-6) / max(o.tok_s, 1e-6)) ** 2
        if o.ttft_ms and o.ttft_ms > 0:
            total += ttft_weight * math.log(max(pr.ttft_ms, 1e-3) / o.ttft_ms) ** 2
    return total / max(len(obs), 1)


ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BETAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
OVERHEADS = [0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03, 0.05]


def fit(obs: Sequence[Observation], dev: DeviceSpec, backend: str) -> PerfParams:
    """Grid search on the three parameters, then one local refinement pass."""
    if not obs:
        return prior(backend)
    best, best_loss = prior(backend), math.inf
    for a in ALPHAS:
        for b in BETAS:
            for o in OVERHEADS:
                p = PerfParams(alpha=a, beta=b, overhead_s=o)
                loss = _loss(obs, dev, p)
                if loss < best_loss:
                    best, best_loss = p, loss
    # Refine each parameter by +-25% steps.
    for _ in range(3):
        for attr in ("alpha", "beta", "overhead_s"):
            base = getattr(best, attr)
            for factor in (0.75, 0.9, 1.1, 1.25):
                cand = PerfParams(alpha=best.alpha, beta=best.beta, overhead_s=best.overhead_s)
                setattr(cand, attr, min(1.0, base * factor) if attr != "overhead_s" else base * factor)
                loss = _loss(obs, dev, cand)
                if loss < best_loss:
                    best, best_loss = cand, loss
    best.fitted = True
    best.n = len(obs)
    return best


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


@dataclass
class Evaluation:
    backend: str
    n: int
    mape_tok_s: Optional[float]
    mape_ttft: Optional[float]
    spearman_tok_s: Optional[float]
    params: PerfParams
    prior_mape_tok_s: Optional[float] = None


def evaluate(obs: Sequence[Observation], dev: DeviceSpec, backend: str) -> Evaluation:
    """Leave-one-out: fit on all but one trial's observations, predict that trial. Also fit on everything."""
    by_trial: Dict[str, List[Observation]] = {}
    for o in obs:
        by_trial.setdefault(o.config.key(), []).append(o)
    errs_t, errs_l, pred_all, meas_all = [], [], [], []
    keys = list(by_trial)
    for k in keys:
        train = [o for kk in keys if kk != k for o in by_trial[kk]]
        p = fit(train, dev, backend) if train else prior(backend)
        for o in by_trial[k]:
            pr = predict(o.situation, o.config, dev, p, o.concurrency, o.device_is_cpu)
            errs_t.append(abs(pr.tok_s - o.tok_s) / o.tok_s * 100)
            pred_all.append(pr.tok_s)
            meas_all.append(o.tok_s)
            if o.ttft_ms:
                errs_l.append(abs(pr.ttft_ms - o.ttft_ms) / o.ttft_ms * 100)
    full = fit(obs, dev, backend)
    prior_errs = [abs(predict(o.situation, o.config, dev, prior(backend), o.concurrency, o.device_is_cpu).tok_s - o.tok_s)
                  / o.tok_s * 100 for o in obs]
    ev = Evaluation(
        backend=backend, n=len(obs),
        mape_tok_s=statistics.fmean(errs_t) if errs_t else None,
        mape_ttft=statistics.fmean(errs_l) if errs_l else None,
        spearman_tok_s=_spearman(pred_all, meas_all),
        params=full,
        prior_mape_tok_s=statistics.fmean(prior_errs) if prior_errs else None,
    )
    full.mape_tok_s = ev.mape_tok_s
    full.mape_ttft = ev.mape_ttft
    full.spearman_tok_s = ev.spearman_tok_s
    return ev


def fit_all(observations: Iterable[Observation], dev: DeviceSpec) -> Dict[str, Evaluation]:
    by: Dict[str, List[Observation]] = {}
    for o in observations:
        by.setdefault(o.backend, []).append(o)
    return {b: evaluate(obs, dev, b) for b, obs in by.items()}


# --------------------------------------------------------------------------- convenience


class Predictor:
    """Bound to a machine: predicts any feasible config for a workload with that machine's parameters."""

    def __init__(self, hw: HardwareDescriptor, params: Optional[Dict[str, PerfParams]] = None):
        from polyserve.hardware import hardware_hash

        self.hw = hw
        self.dev = device_spec(hw)
        self.params = params if params is not None else load_params(hardware_hash(hw))

    def params_for(self, backend: str) -> PerfParams:
        return self.params.get(backend) or prior(backend)

    def is_fitted(self, backend: str) -> bool:
        return self.params_for(backend).fitted

    def predict(self, model: PreparedModel, cfg: Config, prefill: int, decode: int, concurrency: int) -> Prediction:
        sit = situation(model, cfg, prefill, decode)
        return predict(sit, cfg, self.dev, self.params_for(cfg.backend), concurrency, cfg.backend.endswith("-cpu"))

    def best_level(self, model: PreparedModel, cfg: Config, prefill: int, decode: int,
                   concurrencies: Sequence[int], ttft_ceiling_ms: Optional[float] = None) -> Prediction:
        """Highest predicted tok/s across levels, honouring a TTFT ceiling when one is given."""
        preds = [self.predict(model, cfg, prefill, decode, c) for c in concurrencies]
        ok = [p for p in preds if ttft_ceiling_ms is None or p.ttft_ms <= ttft_ceiling_ms]
        pool = ok or preds
        return max(pool, key=lambda p: p.tok_s)


def render_markdown(evals: Dict[str, Evaluation], dev: DeviceSpec) -> str:
    lines = [f"## Performance predictor ({dev.name}: {dev.mem_bw_gbs:.0f} GB/s, {dev.tflops:.0f} TFLOPS"
             f"{'' if dev.known else ', estimated'})", ""]
    if not evals:
        return "\n".join(lines + ["No trials to fit yet; predictions use priors.", ""])
    lines += ["| backend | observations | α (bw eff) | β (compute eff) | overhead | LOO MAPE tok/s | prior MAPE | LOO MAPE TTFT | Spearman ρ |",
              "|---|---|---|---|---|---|---|---|---|"]

    def f(x: Optional[float], nd: int = 1) -> str:
        return "-" if x is None else f"{x:.{nd}f}"

    for b, e in sorted(evals.items()):
        p = e.params
        lines.append(f"| {b} | {e.n} | {p.alpha:.2f} | {p.beta:.2f} | {p.overhead_s * 1000:.1f} ms | {f(e.mape_tok_s)}% | "
                     f"{f(e.prior_mape_tok_s)}% | {f(e.mape_ttft)}% | {f(e.spearman_tok_s, 2)} |")
    lines += ["", "LOO = leave-one-trial-out. Spearman ρ is rank agreement between predicted and measured tok/s "
              "across configs, which is what the search needs to prune safely.", ""]
    return "\n".join(lines)
