"""What each search strategy is worth on this machine.

Calibration says which configuration wins; it does not say how much each strategy contributed.
This takes the calibrated pick and flips one strategy at a time: removed when the pick uses it,
added when it does not. Measured back to back on one card, the gap to the pick is that
strategy's contribution on that workload.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from polyserve.backends.base import BaseBackend
from polyserve.backends.vllm import AMPERE_FP8_KV
from polyserve.bench.compare import results_path
from polyserve.calibrate.objectives import Constraints, _e2e_latency, rank
from polyserve.calibrate.workload import Workload
from polyserve.memory import estimate
from polyserve.models import Config, HardwareDescriptor, ModelSpec, PreparedModel, TrialMetrics, TrialResult
from polyserve.quantized import INT4_METHODS

PREFIX_EXTRAS = ("cache_reuse", "kv_unified")  # llama.cpp's prefix-sharing flags


@dataclass
class Variant:
    label: str  # "-kv", "+spec:ngram:4", ...
    strategy: str  # "weights" | "kv" | "prefix" | "spec" | "pick" | "baseline"
    config: Optional[Config]  # None when it cannot run here; `note` says why
    note: str = ""


def constraints_for(wl: Workload) -> Constraints:
    return Constraints(ttft_ceiling_ms=wl.ttft_ceiling_ms, tpot_ceiling_ms=wl.tpot_ceiling_ms)


def load_pick(results_dir: Path, model: str, workload: str, objective: str,
              hw_hash: Optional[str] = None) -> Optional[Config]:
    """PolyServe's configuration for this workload, from a `polyserve compare` results file.

    A file from this machine is preferred over one from another; ties go by file name, so the
    choice never depends on the order the filesystem lists the directory in.
    """
    name = results_path("*", ModelSpec(hf_id=model), workload, objective, results_dir).name
    files = sorted(results_dir.glob(name), key=lambda f: (not (hw_hash and f.name.startswith(hw_hash)), f.name))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if row.get("label") == "polyserve" and row.get("config"):
                return Config.model_validate(row["config"])
    return None


def strategy_variants(pick: Config, backend: BaseBackend, hw: HardwareDescriptor, model: PreparedModel,
                      workload: Workload) -> List[Variant]:
    """The pick with each strategy flipped, one at a time."""
    mm = backend.memory_model(hw)

    def fits(c: Config) -> bool:
        try:
            return estimate(hw, model, c, mm).feasible
        except Exception:
            return False

    def variant(label: str, strategy: str, cfg: Config) -> Variant:
        return Variant(label, strategy, cfg) if fits(cfg) else Variant(label, strategy, None, "does not fit in memory")

    def fit_or_shrink(label: str, strategy: str, cfg: Config, what: str) -> Variant:
        """Removing a memory-saving strategy may not fit at the pick's batch: step down until it does.
        That smaller batch is what the machine would have run without the strategy."""
        if fits(cfg):
            return Variant(label, strategy, cfg)
        smaller = [cfg.model_copy(update={"batch": b}) for b in sorted(backend.batch_ladder(), reverse=True)
                   if b < cfg.batch]
        shrunk = next((c for c in smaller if fits(c)), None)
        note = (f"batch {shrunk.batch}: {what} does not fit at batch {cfg.batch}" if shrunk
                else f"{what} does not fit")
        return Variant(label, strategy, shrunk, note)

    out: List[Variant] = []

    # Weights: 4-bit against the best 8- or 16-bit precision, or the reverse.
    if pick.quant in INT4_METHODS:
        alt = next((q for q in ("fp8", "bf16", "fp16") if q in model.weights_bytes), None)
        if alt:
            out.append(fit_or_shrink(f"-int4 ({alt})", "weights", pick.model_copy(update={"quant": alt}),
                                     f"{alt} weights"))
    elif pick.quant in ("fp8", "bf16", "fp16"):
        for q in INT4_METHODS:
            if q in model.weights_bytes:
                out.append(variant(f"+{q}", "weights", pick.model_copy(update={"quant": q})))

    # KV cache: without quantization the pick's batch may not fit either.
    if pick.kv_dtype != "auto":
        out.append(fit_or_shrink("-kv", "kv", pick.model_copy(update={"kv_dtype": "auto"}), "an unquantized cache"))
        if pick.backend == "vllm" and pick.kv_dtype == AMPERE_FP8_KV:
            # On Ampere the fp8 cache also moves attention to FlashInfer; this separates the two effects.
            flashinfer = pick.model_copy(update={"kv_dtype": "auto",
                                                 "extra": {**pick.extra, "attention_backend": "FLASHINFER"}})
            out.append(fit_or_shrink("-kv (FlashInfer kept)", "kv", flashinfer, "an unquantized cache"))
    else:
        for kd in backend.kv_dtypes(hw):
            out.append(variant(f"+kv:{kd}", "kv", pick.model_copy(update={"kv_dtype": kd})))

    # Prefix caching matters only when prompts share a prefix.
    if workload.shared_prefix_tokens > 0:
        if any(k in pick.extra for k in PREFIX_EXTRAS):
            extra = {k: v for k, v in pick.extra.items() if k not in PREFIX_EXTRAS}
            out.append(variant("-prefix", "prefix", pick.model_copy(update={"extra": extra})))
        elif pick.prefix_cache is False:
            out.append(variant("+prefix", "prefix", pick.model_copy(update={"prefix_cache": None})))
        else:
            added = backend.prefix_variants(pick)
            for v in added:
                keys = ",".join(k for k in v.extra if v.extra.get(k) != pick.extra.get(k))
                out.append(variant(f"+prefix:{keys}", "prefix", v))
            if not added:  # vLLM and SGLang cache prefixes by default: measure what turning it off costs
                out.append(variant("-prefix", "prefix", pick.model_copy(update={"prefix_cache": False})))

    # Speculative decoding.
    if pick.spec_decode:
        out.append(variant("-spec", "spec", pick.model_copy(update={"spec_decode": None})))
    else:
        for v in backend.spec_variants(pick, model):
            out.append(variant(f"+spec:{v.spec_decode}", "spec", v))
    return out


def _finite(x: Optional[float]) -> Optional[float]:
    return x if (x is not None and math.isfinite(x)) else None


def level_table(m: TrialMetrics) -> Dict[str, Dict[str, object]]:
    """Per-concurrency results, for seeing where a strategy stops paying."""
    return {c: {"ok": x.ok, "tok_s": x.tok_s, "ttft_ms": x.ttft_ms, "tpot_ms": x.tpot_ms,
                "e2e_ms": _finite(_e2e_latency(x))}
            for c, x in sorted(m.by_concurrency.items(), key=lambda kv: int(kv[0]))}


def score_row(res: TrialResult, objective: str, cons: Constraints) -> Dict[str, object]:
    """One measured row, scored at the concurrency level the objective would pick."""
    ranked = rank([res], objective, cons)
    if not ranked:
        return {"ok": False, "error": (res.error or "failed")[:2000]}
    m = ranked[0].metrics
    return {
        "ok": True, "concurrency": m.concurrency, "tok_s": m.tok_s, "ttft_ms": m.ttft_ms,
        "ttft_p95_ms": m.ttft_p95_ms, "tpot_ms": m.tpot_ms, "e2e_ms": _finite(_e2e_latency(m)),
        "joules_per_token": m.joules_per_token, "peak_mem_mb": m.peak_mem_mb, "meets_slo": ranked[0].feasible,
        "token_count_source": m.token_count_source, "levels": level_table(res.metrics),
    }


def sweep_workload(wl: Workload, concurrencies: Sequence[int]) -> Workload:
    """The same prompt shape at every level, with enough prompts to keep the top level busy."""
    levels = tuple(sorted({int(c) for c in concurrencies}))
    return replace(wl, concurrencies=levels, n_prompts=max(wl.n_prompts, 2 * levels[-1]), prompts=[], fitted=False)


def crossover(off: Dict[str, Dict[str, object]], on: Dict[str, Dict[str, object]],
              metric: str = "tok_s") -> Optional[int]:
    """The lowest concurrency at which `on` stops beating `off`: higher tok/s, or lower latency."""
    higher_is_better = metric == "tok_s"
    for c in sorted(set(off) & set(on), key=int):
        a, b = off[c].get(metric), on[c].get(metric)
        if a is None or b is None or not off[c].get("ok") or not on[c].get("ok"):
            continue
        if (b <= a) if higher_is_better else (b >= a):
            return int(c)
    return None


def _ms(x: Optional[float], digits: int = 0) -> str:
    return "-" if x is None else f"{x:.{digits}f} ms"


def _ref_cell(ref: Optional[Dict[str, object]], ours: float) -> str:
    if ref is None:
        return "-"
    if not ref.get("ok"):
        return "failed"
    tok = float(ref["scored_tok_s"] or 0.0)
    gain = f"{(ours / tok - 1) * 100:+.0f}%" if tok else "-"
    return f"{tok:.0f} ({gain}){'' if ref.get('meets_slo') else ', missed SLO'}"


def _gpu(d: Dict[str, object]) -> str:
    """The card a results file was measured on, without the vendor prefix; "CPU" for a CPU-only run."""
    return str(d.get("gpu") or ("CPU" if d.get("cpu") else "?")).replace("NVIDIA ", "")


def compare_table(paths: Sequence[Path]) -> str:
    """PolyServe's pick against stock vLLM at bf16 and fp8, one row per (model, workload) results file."""
    lines = ["| GPU | model | workload | PolyServe pick | tok/s | TTFT p50 | stock vLLM bf16 (gain) "
             "| stock vLLM fp8 (gain) | stock llama.cpp (gain) |",
             "|---|---|---|---|---|---|---|---|---|"]
    notes: List[str] = []
    for p in sorted(paths):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if "rows" not in d:  # a calibration profile or other JSON kept next to the results
            continue
        rows = {r["label"]: r for r in d.get("rows", [])}
        ps = rows.get("polyserve")
        model = str(d.get("model_id", "?")).split("/")[-1]
        run = Path(p).parent  # a rerun kept in a folder inside a results folder (results-real/v3) is labelled
        workload = f"{d.get('workload')}" + (f" ({run.name})" if run.parent.name.startswith("results") else "")
        if ps is None or not ps.get("ok"):
            lines.append(f"| {_gpu(d)} | {model} | {workload} | failed | | | | | |")
            continue
        ours = float(ps["scored_tok_s"] or 0.0)
        llama = rows.get("llamacpp-cuda-default") or rows.get("llamacpp-cpu-default")
        lines.append(f"| {_gpu(d)} | {model} | {workload} | `{ps['config_key']}` | {ours:.0f} | "
                     f"{_ms(ps.get('scored_ttft_ms'))} | {_ref_cell(rows.get('vllm-default'), ours)} | "
                     f"{_ref_cell(rows.get('vllm-fp8-default'), ours)} | {_ref_cell(llama, ours)} |")
        notes += [f"{model} / {d.get('workload')}: {n}" for n in d.get("notes", [])
                  if any(w in n for w in ("layout", "replicas", "disaggregat", "one GPU"))]
    return "\n".join(lines + ([""] + [f"- {n}" for n in notes] if notes else []))


def ablation_report(paths: Sequence[Path]) -> str:
    """One table per ablation file, plus the speculative-decoding sweep when one was run."""
    out: List[str] = []
    for p in sorted(paths):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        folder = Path(p).parent  # a side check kept inside an ablation folder (ablation-real/flashinfer)
        where = ", ".join([_gpu(d)] + ([folder.name] if folder.parent.name.startswith("ablation") else []))
        out += [f"**{str(d.get('model_id', '?')).split('/')[-1]} / {d.get('workload')}** ({where})", "",
                markdown(d["rows"]), ""]
        sweep = d.get("spec_sweep")
        if sweep:
            off = sweep["off"]["levels"]
            for on in sweep["on"]:
                out += [f"Speculative decoding `{on['spec']}` by concurrency (throughput stops gaining at "
                        f"c={on['tok_s_crossover']}, latency at c={on['latency_crossover']}):", "",
                        "| concurrency | tok/s off | tok/s on | request latency off | request latency on |",
                        "|---|---|---|---|---|"]
                for c in sweep["levels"]:
                    a, b = off.get(str(c), {}), on["levels"].get(str(c), {})
                    out.append(f"| {c} | {a.get('tok_s') or 0:.0f} | {b.get('tok_s') or 0:.0f} | "
                               f"{_ms(a.get('e2e_ms'))} | {_ms(b.get('e2e_ms'))} |")
                out.append("")
    return "\n".join(out)


def quality_table(paths: Sequence[Path]) -> str:
    """Perplexity per weight precision (benchmarks/quality_check.py output), relative to bf16."""
    lines = ["| model | weights | perplexity | vs bf16 | checkpoint |", "|---|---|---|---|---|"]
    for p in sorted(paths):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        model = str(d.get("model", "?")).split("/")[-1]
        for q, ppl in d.get("perplexity", {}).items():
            delta = d.get("delta_pct", {}).get(q)
            lines.append(f"| {model} | {q} | {ppl:.3f} | {'' if not delta else f'{delta:+.1f}%'} | "
                         f"{d.get('checkpoints', {}).get(q, d.get('model', ''))} |")
    return "\n".join(lines)


def gguf_quality_table(path: Path, reference: str = "q8_0") -> str:
    """llama-perplexity results, one '<quant> ... PPL = <x>' line per GGUF, relative to Q8_0."""
    ppl: Dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\S+)\s.*PPL = ([0-9.]+)", line)
        if m:
            ppl[m.group(1)] = float(m.group(2))
    ref = ppl.get(reference)
    lines = ["| GGUF | perplexity | vs Q8_0 |", "|---|---|---|"]
    for q, v in ppl.items():
        lines.append(f"| {q.upper()} | {v:.3f} | {'' if not ref or q == reference else f'{(v / ref - 1) * 100:+.1f}%'} |")
    return "\n".join(lines)


def mcnemar_p(lost: int, gained: int) -> float:
    """Two-sided exact McNemar test: how lopsided the split is among problems only one side got right."""
    n = lost + gained
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lost, gained) + 1)) / 2 ** n)


def task_quality_table(paths: Sequence[Path]) -> str:
    """Task accuracy per weight precision (benchmarks/task_quality.py output), paired against the first."""
    lines = ["| model | weights | accuracy | 95% CI | vs reference | lost / gained | p (McNemar) |",
             "|---|---|---|---|---|---|---|"]
    for p in sorted(paths):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        model = str(d.get("model", "?")).split("/")[-1]
        acc = d.get("accuracy", {})
        for q, r in acc.items():
            if "error" in r:
                lines.append(f"| {model} | {q} | failed | | | | {str(r['error']).splitlines()[0][:60]} |")
                continue
            lo, hi = r["ci95"]
            row = f"| {model} | {q} | {r['accuracy']:.1%} | {lo:.1%}–{hi:.1%} |"
            ref_key = next((k for k in r if k.startswith("vs_")), None)
            if ref_key is None:
                lines.append(row + " reference | | |")
                continue
            ref, vs = ref_key[3:], r[ref_key]
            p_value = mcnemar_p(vs["lost"], vs["gained"])
            lines.append(row + f" {(r['accuracy'] - acc[ref]['accuracy']) * 100:+.1f} pts vs {ref} | "
                         f"{vs['lost']} / {vs['gained']} | {'<0.001' if p_value < 0.001 else f'{p_value:.3f}'} |")
    return "\n".join(lines)


def markdown(rows: List[Dict[str, object]], reference: str = "polyserve") -> str:
    ref = next((r for r in rows if r["label"] == reference and r.get("ok")), None)
    lines = ["| row | config | tok/s | vs pick | TTFT p50 | TPOT | request latency | SLO |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("ok"):
            why = str(r.get("note") or r.get("error") or "failed").splitlines()[0][:70]
            lines.append(f"| {r['label']} | `{r.get('config_key', '-')}` | failed | | | | | {why} |")
            continue
        delta = f"{(r['tok_s'] / ref['tok_s'] - 1) * 100:+.1f}%" if ref and r is not ref and ref["tok_s"] else ""
        lines.append(f"| {r['label']} | `{r['config_key']}` | {r['tok_s']:.0f} | {delta} | {_ms(r['ttft_ms'])} | "
                     f"{_ms(r['tpot_ms'], 1)} | {_ms(r['e2e_ms'])} | {'met' if r['meets_slo'] else 'missed'} |")
    return "\n".join(lines)
