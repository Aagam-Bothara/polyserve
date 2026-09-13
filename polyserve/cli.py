"""polyserve CLI: serve | probe | plan | recalibrate | profiles | bench"""

from __future__ import annotations

import json
import logging
import math
import signal
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from polyserve import __version__
from polyserve.models import OBJECTIVES, GiB, ModelSpec, Profile, TrialResult

app = typer.Typer(help="PolyServe: hardware-adaptive LLM serving. One command, one OpenAI-compatible API.",
                  no_args_is_help=True, add_completion=False)
console = Console()
err = Console(stderr=True)
if sys.platform == "win32":
    # Legacy Windows consoles are cp1252: print what they cannot encode (α, →) as '?' rather than crash.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(console=err, show_path=False, rich_tracebacks=verbose)])
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _objective(value: str) -> str:
    if value not in OBJECTIVES:
        raise typer.BadParameter(f"objective must be one of {', '.join(OBJECTIVES)}")
    return value


def _fmt(x: Optional[float], nd: int = 1, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}{suffix}"


def _trial_table(results: List[TrialResult], winner: Optional[str] = None) -> Table:
    t = Table(title="Calibration", show_lines=False)
    for col in ("stage", "config", "tok/s", "TTFT ms", "TPOT ms", "peak MB", "W", "MHz", "J/tok", "status"):
        t.add_column(col, justify="right" if col not in ("stage", "config", "status") else "left",
                     overflow="fold", min_width=(40 if col == "config" else None))
    for r in results:
        m = r.metrics
        status = "ok" if r.ok else (r.error or "failed").splitlines()[0][:40]
        key = _trial_key(r)
        style = "bold green" if winner and key == winner else None
        t.add_row(r.stage, key, _fmt(m.tok_s), _fmt(m.ttft_ms, 0), _fmt(m.tpot_ms, 1), _fmt(m.peak_mem_mb, 0),
                  _fmt(m.power_w, 0), _fmt(m.sm_clock_mhz, 0), _fmt(m.joules_per_token, 3), status, style=style)
    return t


def _trial_key(r: TrialResult) -> str:
    if r.disagg is not None:
        return r.disagg.key()
    return f"{r.config.key()} x{r.replicas}" if r.replicas > 1 else r.config.key()


def _print_profile(p: Profile) -> None:
    console.print(f"[bold]objective[/]: {p.objective}   [bold]workload[/]: {p.workload}")
    console.print(f"[bold]backend[/]: {p.backend} {p.backend_version or ''}")
    console.print(f"[bold]config[/]:  {p.config.key()}")
    if p.disagg is not None:
        console.print(f"[bold]phases[/]:  disaggregated over {p.disagg.connector}; "
                      f"prefill GPU {p.disagg.prefill_gpu}: {p.disagg.prefill.key()}")
        console.print(f"          decode GPU {p.disagg.decode_gpu}: {p.disagg.decode.key()}")
    if p.replicas > 1:
        console.print(f"[bold]layout[/]:  {p.replicas} replicas, one per GPU, behind a load balancer")
    elif p.config.tp > 1:
        console.print(f"[bold]layout[/]:  tensor parallel across {p.config.tp} GPUs")
    if p.options:
        console.print(f"[bold]options[/]: {', '.join(f'{k}={v}' for k, v in sorted(p.options.items()))}")
    console.print(f"[bold]launch[/]:  {' '.join(p.launch_args)}")
    if p.launch_env:
        console.print(f"[bold]env[/]:     {' '.join(f'{k}={v}' for k, v in p.launch_env.items())}")
    for n in p.notes:
        console.print(f"[dim]note: {n}[/]")


def _winner_key(p: Profile) -> str:
    if p.disagg is not None:
        return p.disagg.key()
    return f"{p.config.key()} x{p.replicas}" if p.replicas > 1 else p.config.key()


def _progress(stage: str, cfg, res) -> None:
    if res is None:
        err.print(f"[cyan]{stage:>7}[/] {cfg.key()} ...")
    else:
        m = res.metrics
        if res.ok:
            err.print(f"[green]{stage:>7}[/] {cfg.key()}  {m.tok_s:.1f} tok/s  TTFT {_fmt(m.ttft_ms, 0)} ms"
                      f"{'  ' + _fmt(m.joules_per_token, 3) + ' J/tok' if m.joules_per_token else ''}")
        else:
            err.print(f"[red]{stage:>7}[/] {cfg.key()}  FAILED: {(res.error or '').splitlines()[0][:80]}")


def _workload(name: str):
    from polyserve.calibrate.workload import WORKLOAD_NAMES, get_workload

    if name not in WORKLOAD_NAMES:
        raise typer.BadParameter(f"workload must be one of {', '.join(WORKLOAD_NAMES)}")
    return get_workload(name)


def _constraints(ttft_ceiling: Optional[float], tok_s_floor: Optional[float], workload=None,
                 tpot_ceiling: Optional[float] = None):
    from polyserve.calibrate.objectives import Constraints

    ceiling = ttft_ceiling if ttft_ceiling is not None else (workload.ttft_ceiling_ms if workload else 500.0)
    tpot = tpot_ceiling if tpot_ceiling is not None else (workload.tpot_ceiling_ms if workload else None)
    return Constraints(ttft_ceiling_ms=ceiling, tok_s_floor_abs=tok_s_floor, tpot_ceiling_ms=tpot)


def _power_mode(value: str) -> str:
    from polyserve.power import MODES

    if value not in MODES:
        raise typer.BadParameter(f"power must be one of {', '.join(MODES)}")
    return value


def _power_controller(profile: Profile):
    """A controller for serving/comparing a profile that carries a power setting, else None."""
    if profile.config.power_limit_w is None and profile.config.sm_clock_mhz is None:
        return None
    from polyserve.power import controller_for

    if profile.disagg is not None:  # disaggregated: the decode GPU is the one that is capped
        return controller_for(profile.disagg.decode_gpu)
    return controller_for(profile.hardware.gpu.index if profile.hardware.gpu else 0)


WORKLOAD_OPT = typer.Option("default", "--workload", "-w", help="Workload preset; see `polyserve workloads`")
POWER_OPT = typer.Option(
    "off", "--power", callback=_power_mode,
    help="Energy tuning: off | cap (power limit) | clock (locked SM clock) | both. Needs root; machine-wide; "
         "restored on exit and by `polyserve power reset`.",
)


def _phases(value: str) -> str:
    from polyserve.disagg import PHASES

    if value not in PHASES:
        raise typer.BadParameter(f"phases must be one of {', '.join(PHASES)}")
    return value


TPOT_OPT = typer.Option(None, "--tpot-ceiling", help="Per-token decode latency ceiling in ms (default: the workload's)")
PHASES_OPT = typer.Option(
    "unified", "--phases", callback=_phases,
    help="unified (default): one engine, prefill and decode knobs tuned separately. disaggregated: prefill and "
         "decode on separate GPUs joined by KV transfer (vLLM, two GPUs, a KV connector). auto: measure both, "
         "keep the better.",
)
KV_OPT = typer.Option("nixl", "--kv-connector", help="KV-cache transfer connector for --phases disaggregated/auto")


def _on_off(value: str) -> str:
    if value not in ("on", "off"):
        raise typer.BadParameter("expected on or off")
    return value


def _layout(value: str) -> str:
    from polyserve.layout import LAYOUTS

    if value not in LAYOUTS:
        raise typer.BadParameter(f"layout must be one of {', '.join(LAYOUTS)}")
    return value


def _quant_list(value: str) -> str:
    from polyserve.gguf import GGUF_QUANTS

    known = {"auto", "bf16", "fp16", "fp8", "awq", "gptq", *GGUF_QUANTS}
    if value != "auto":
        bad = [q for q in value.split(",") if q.strip() not in known]
        if bad:
            raise typer.BadParameter(f"unknown precision {', '.join(bad)}; known: auto, {', '.join(sorted(known))}")
    return value


def _opts(quant: str, kv_quant: str, speculative: str, prefix_cache: str, combine: str = "on"):
    from polyserve.pipeline import SearchOptions

    return SearchOptions(
        quants=None if quant == "auto" else [q.strip() for q in quant.split(",") if q.strip()],
        kv_quant=kv_quant == "on", speculative=speculative == "on", prefix_cache=prefix_cache == "on",
        combine=combine == "on",
    )


QUANT_OPT = typer.Option("auto", "--quant", callback=_quant_list,
                         help="Weight precisions calibration may choose. auto: everything supported except 4-bit "
                              "AWQ/GPTQ checkpoints, which cost quality (auto,awq,gptq adds them). Or a list such "
                              "as bf16, bf16,fp8 or gptq (also fp16, Q4_K_M, Q5_K_M, Q6_K, Q8_0).")
KVQ_OPT = typer.Option("on", "--kv-quant", callback=_on_off,
                       help="Try quantized KV caches (fp8 on vLLM and SGLang, q8_0 and q4_0 on llama.cpp)")
SPEC_OPT = typer.Option("on", "--speculative", callback=_on_off,
                        help="Try speculative decoding (n-gram prompt lookup, a small draft model)")
PREFIX_OPT = typer.Option("on", "--prefix-cache", callback=_on_off,
                          help="Keep prefix caching on and tune it for workloads whose prompts share a prefix")
LAYOUT_OPT = typer.Option("single", "--layout", callback=_layout,
                          help="Multi-GPU layout: single (default), replicas (one engine per GPU behind a load "
                               "balancer), tp (tensor parallel), auto (measure both, keep the better)")
COMBINE_OPT = typer.Option("on", "--combine", callback=_on_off,
                           help="After tuning one setting at a time, measure the leader with each adopted "
                                "change undone, then combinations of the settings that came close on their "
                                "own (up to 8 extra trials in all)")
TTFT_OPT = typer.Option(None, "--ttft-ceiling", help="balanced: TTFT ceiling in ms (default: the workload's)")


# --------------------------------------------------------------------------- commands


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    _setup_logging(verbose)


@app.command()
def version() -> None:
    """Print the PolyServe version."""
    console.print(__version__)


@app.command()
def workloads() -> None:
    """List workload presets."""
    from polyserve.calibrate.workload import workload_table

    t = Table(title="Workload presets (latency ceilings in ms)")
    for col in ("name", "prefill", "shared", "decode", "concurrency", "TTFT", "TPOT"):
        t.add_column(col, justify="right" if col != "name" else "left")
    for w in workload_table():
        t.add_row(str(w["name"]), str(w["prefill_tokens"]), str(w.get("shared_prefix_tokens") or "-"),
                  str(w["decode_tokens"]),
                  "/".join(str(c) for c in w["concurrencies"]), f"{w['ttft_ceiling_ms']:.0f}",
                  f"{w['tpot_ceiling_ms']:.0f}" if w.get("tpot_ceiling_ms") else "-")
    console.print(t)


@app.command()
def probe(as_json: bool = typer.Option(False, "--json", help="Machine-readable output")) -> None:
    """Print the HardwareDescriptor."""
    from polyserve.hardware import hardware_hash, probe as _probe
    from polyserve.selector import explain

    hw = _probe()
    if as_json:
        data = hw.model_dump(mode="json")
        data["hardware_hash"] = hardware_hash(hw)
        console.print_json(json.dumps(data))
        return
    console.print(f"[bold]hardware hash[/]: {hardware_hash(hw)}   [dim]{hw.os}, python {hw.python}[/]")
    if hw.gpus:
        for g in hw.gpus:
            console.print(f"[bold]GPU {g.index}[/]: {g.name} (cc {g.cc[0]}.{g.cc[1]}) "
                          f"{g.vram_free_bytes / GiB:.1f}/{g.vram_total_bytes / GiB:.1f} GiB free, "
                          f"driver {g.driver_version or '?'}, CUDA {g.cuda_version or '?'}")
    else:
        console.print("[bold]GPU[/]: none")
    c = hw.cpu
    console.print(f"[bold]CPU[/]: {c.model_name} ({c.physical_cores}c/{c.logical_cores}t, {c.arch}) "
                  f"AVX2={'y' if c.avx2 else 'n'} AVX512={'y' if c.avx512 else 'n'}  "
                  f"RAM {c.ram_free_bytes / GiB:.1f}/{c.ram_total_bytes / GiB:.1f} GiB free")
    console.print("[bold]backends[/]:")
    for line in explain(hw):
        console.print(f"  {line}")


@app.command()
def plan(
    model: str,
    backend: Optional[str] = typer.Option(None, "--backend", help="Force one backend"),
    workload: str = WORKLOAD_OPT,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Print feasible configs (after the memory planner) without running them."""
    from polyserve.hardware import probe as _probe
    from polyserve.pipeline import prepare_and_plan, select

    wl = _workload(workload)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    candidates, reg = select(hw, spec, force=backend)
    if not candidates:
        err.print("[red]no candidate backends for this machine/model[/]")
        raise typer.Exit(2)
    result = prepare_and_plan(hw, spec, candidates, reg, workload=wl)
    if as_json:
        out = {
            name: [{"config": c.model_dump(), "estimate": e.model_dump()} for c, e in cfgs]
            for name, cfgs in result.feasible.items()
        }
        out["_errors"] = result.errors
        console.print_json(json.dumps(out))
        return
    console.print(f"candidates: {', '.join(candidates)}   budget: {hw.available_memory_bytes / GiB:.1f} GiB   "
                  f"workload: {wl.describe()}")
    for name in candidates:
        if name in result.errors:
            console.print(f"[red]{name}: {result.errors[name]}[/]")
            continue
        cfgs = result.feasible.get(name, [])
        t = Table(title=f"{name}: {len(cfgs)}/{result.considered.get(name, 0)} feasible")
        for col in ("config", "weights GiB", "kv GiB", "workspace GiB", "margin GiB", "total GiB", "budget GiB"):
            t.add_column(col, justify="right" if col != "config" else "left", overflow="fold",
                         min_width=(40 if col == "config" else None))
        for c, e in cfgs:
            t.add_row(c.key(), f"{e.weights / GiB:.2f}", f"{e.kv_cache / GiB:.2f}", f"{e.runtime_workspace / GiB:.2f}",
                      f"{e.safety_margin / GiB:.2f}", f"{e.total / GiB:.2f}", f"{e.budget / GiB:.2f}")
        console.print(t)


@app.command()
def bench(
    model: str,
    objective: str = typer.Option("balanced", "--objective", callback=_objective),
    workload: str = WORKLOAD_OPT,
    backend: Optional[str] = typer.Option(None, "--backend"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None, help="latency/efficiency: absolute tok/s floor"),
    save: bool = typer.Option(False, "--save", help="Also write the winning profile to the cache"),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
) -> None:
    """Run calibration and print the table; do not serve."""
    from polyserve import cache as profile_cache
    from polyserve.hardware import probe as _probe
    from polyserve.pipeline import calibrate, prepare_and_plan, select

    wl = _workload(workload)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    candidates, reg = select(hw, spec, force=backend)
    if not candidates:
        err.print("[red]no candidate backends for this machine/model[/]")
        raise typer.Exit(2)
    opts = _opts(quant, kv_quant, speculative, prefix_cache, combine)
    result = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl, quants=opts.quants)
    err.print(f"{len(result.all_feasible)}/{result.total_considered} configs feasible; "
              f"calibrating for {objective} on workload {wl.name}")
    cons = _constraints(ttft_ceiling, tok_s_floor, wl, tpot_ceiling)
    profile = calibrate(hw, spec, objective, result, reg, workload=wl, constraints=cons, progress=_progress,
                        power_mode=power, options=opts)
    if phases != "unified":
        from polyserve.disagg import calibrate_disaggregated

        profile = calibrate_disaggregated(hw, spec, objective, profile, reg, workload=wl, constraints=cons,
                                          phases=phases, connector=kv_connector, progress=_progress,
                                          power_mode=power)
    if layout != "single":
        from polyserve.layout import calibrate_layout

        profile = calibrate_layout(hw, objective, profile, reg, layout, workload=wl, constraints=cons,
                                   progress=_progress)
    console.print(_trial_table(profile.calibration_table, winner=_winner_key(profile)))
    _print_profile(profile)
    if save:
        profile_cache.save(profile)


@app.command()
def recalibrate(
    model: str,
    objective: str = typer.Option("balanced", "--objective", callback=_objective),
    workload: str = WORKLOAD_OPT,
    backend: Optional[str] = typer.Option(None, "--backend"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
) -> None:
    """Force a calibration rerun and overwrite the cached profile."""
    from polyserve.pipeline import resolve_profile

    wl = _workload(workload)
    profile = resolve_profile(ModelSpec(hf_id=model), objective, force_backend=backend, recalibrate=True,
                              workload=wl, constraints=_constraints(ttft_ceiling, tok_s_floor, wl, tpot_ceiling),
                              progress=_progress, on_stage=lambda s: err.print(f"[dim]-> {s}[/]"),
                              power_mode=power, phases=phases, kv_connector=kv_connector,
                              options=_opts(quant, kv_quant, speculative, prefix_cache, combine), layout=layout)
    console.print(_trial_table(profile.calibration_table, winner=_winner_key(profile)))
    _print_profile(profile)


@app.command()
def compare(
    model: str,
    objective: str = typer.Option("balanced", "--objective", callback=_objective),
    workload: str = WORKLOAD_OPT,
    backend: Optional[str] = typer.Option(None, "--backend", help="Restrict PolyServe's candidates to one backend"),
    ollama_model: Optional[str] = typer.Option(None, "--ollama-model", help="Ollama tag for the ollama row"),
    include: Optional[List[str]] = typer.Option(None, "--include", help="Only these reference rows"),
    out: Optional[Path] = typer.Option(None, "--out", help="Results directory (default benchmarks/results)"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
) -> None:
    """Measure PolyServe's pick vs stock defaults (and Ollama) on one workload; write a results JSON."""
    from polyserve.bench import compare as _compare, to_markdown
    from polyserve.bench.compare import save
    from polyserve.hardware import probe as _probe
    from polyserve.pipeline import prepare_and_plan, resolve_profile, select

    wl = _workload(workload)
    cons = _constraints(ttft_ceiling, tok_s_floor, wl, tpot_ceiling)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    profile = resolve_profile(spec, objective, force_backend=backend, workload=wl, constraints=cons,
                              progress=_progress, hw=hw, on_stage=lambda s: err.print(f"[dim]-> {s}[/]"),
                              power_mode=power, phases=phases, kv_connector=kv_connector,
                              options=_opts(quant, kv_quant, speculative, prefix_cache, combine), layout=layout)
    _print_profile(profile)
    candidates, reg = select(hw, spec, force=backend)
    planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl,
                               quants=_opts(quant, kv_quant, speculative, prefix_cache, combine).quants)

    def _row_progress(label: str, row) -> None:
        if row is None:
            err.print(f"[cyan]compare[/] {label} ...")
        elif row.ok:
            err.print(f"[green]compare[/] {label}: {row.scored_tok_s:.1f} tok/s @c{row.scored_concurrency}, "
                      f"TTFT {_fmt(row.scored_ttft_ms, 0)} ms, SLO {'met' if row.meets_slo else 'missed'}")
        else:
            err.print(f"[red]compare[/] {label} FAILED: {(row.error or '').splitlines()[0][:80]}")

    from polyserve import cache as profile_cache

    result = _compare(hw, spec, profile, planned.prepared, reg, workload=wl, constraints=cons,
                      ollama_tag=ollama_model, include=include or None, progress=_row_progress,
                      log_dir=profile_cache.logs_dir() / spec.safe_id / f"compare-{wl.name}",
                      power=_power_controller(profile))
    path = save(result, out)
    console.print(to_markdown(result))
    console.print(f"[dim]saved {path}[/]")


@app.command()
def report(
    results: Optional[Path] = typer.Option(None, "--results", help="Results dir (default benchmarks/results)"),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to write RESULTS.md and the SVG"),
) -> None:
    """Aggregate compare results: median gain over the best SLO-meeting default, RESULTS.md, plot."""
    from polyserve.bench.report import write_report

    md, svg, summary = write_report(results, out)
    console.print(summary.headline())
    console.print(f"[dim]wrote {md} and {svg}[/]")


@app.command()
def predict(
    model: str,
    workload: str = WORKLOAD_OPT,
    backend: Optional[str] = typer.Option(None, "--backend"),
    top: int = typer.Option(20, "--top", help="Rows to show"),
) -> None:
    """Predict tok/s and TTFT for every feasible config without launching anything."""
    from polyserve.hardware import hardware_hash, probe as _probe
    from polyserve.pipeline import prepare_and_plan, select
    from polyserve.predict import Predictor

    wl = _workload(workload)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    candidates, reg = select(hw, spec, force=backend)
    if not candidates:
        err.print("[red]no candidate backends for this machine/model[/]")
        raise typer.Exit(2)
    planned = prepare_and_plan(hw, spec, candidates, reg, workload=wl)
    pred = Predictor(hw)
    rows = []
    for name, cfgs in planned.feasible.items():
        for cfg, _ in cfgs:
            p = pred.best_level(planned.prepared[name], cfg, wl.prefill_tokens, wl.decode_tokens, wl.concurrencies,
                                wl.ttft_ceiling_ms)
            rows.append((cfg, p))
    rows.sort(key=lambda r: -r[1].tok_s)
    d = pred.dev
    console.print(f"[bold]{d.name}[/]: {d.mem_bw_gbs:.0f} GB/s, {d.tflops:.0f} TFLOPS"
                  f"{'' if d.known else ' (estimated)'}   workload {wl.name}   hardware {hardware_hash(hw)}")
    t = Table(title=f"Predicted ({len(rows)} feasible configs, top {top})")
    for col in ("config", "pred tok/s", "@c", "pred TTFT ms", "pred TPOT ms", "bound", "params"):
        t.add_column(col, justify="left" if col in ("config", "bound", "params") else "right", overflow="fold",
                     min_width=(40 if col == "config" else None))
    for cfg, p in rows[:top]:
        t.add_row(cfg.key(), f"{p.tok_s:.0f}", str(p.concurrency), f"{p.ttft_ms:.0f}", f"{p.tpot_ms:.1f}",
                  "memory" if p.memory_bound else "compute", "fitted" if p.fitted else "prior")
    console.print(t)
    if not any(p.fitted for _, p in rows):
        err.print("[dim]no fitted parameters for this machine yet; run `polyserve fit` after a calibration[/]")


@app.command()
def fit(
    apply: bool = typer.Option(False, "--apply", help="Write fitted parameters to ~/.polyserve/perf-model.json"),
    all_machines: bool = typer.Option(False, "--all", help="Report every machine's profiles (no --apply)"),
) -> None:
    """Fit the performance predictor from cached calibration profiles; report leave-one-out accuracy."""
    from polyserve import cache as profile_cache
    from polyserve import predict as P
    from polyserve.hardware import hardware_hash, probe as _probe

    if apply and all_machines:
        err.print("[red]--apply needs this machine's profiles only; drop --all[/]")
        raise typer.Exit(2)
    hh = hardware_hash(_probe())
    # One fit per machine, against the GPU its trials ran on: a roofline fitted to another card's
    # bandwidth and compute says nothing about either card.
    groups: dict = {}
    for _, p in profile_cache.list_profiles():
        if (all_machines or p.hardware_hash == hh) and p.hardware is not None:
            groups.setdefault(p.hardware_hash, (p.hardware, []))[1].extend(P.observations_from_profile(p))
    if not groups:
        console.print(f"no calibration profiles for hardware {hh} under {profile_cache.profiles_dir()}")
        return
    for machine, (machine_hw, obs) in sorted(groups.items()):
        dev = P.device_spec(machine_hw)
        evals = P.fit_all(obs, dev)
        if len(groups) > 1:
            console.print(f"[bold]hardware {machine}[/]")
        console.print(P.render_markdown(evals, dev))
        if apply and machine == hh:
            path = P.save_params(hh, {b: e.params for b, e in evals.items() if not e.keeps_prior})
            console.print(f"[green]wrote {path}[/]; calibration on hardware {hh} now prunes with these parameters")


@app.command("memory-report")
def memory_report(
    results: Optional[Path] = typer.Option(None, "--results", help="compare results dir (default benchmarks/results)"),
    apply: bool = typer.Option(False, "--apply", help="Write fitted workspace/margin to ~/.polyserve/memory-model.json"),
    all_machines: bool = typer.Option(False, "--all", help="Include profiles/results from other hardware hashes"),
) -> None:
    """Planner prediction vs measured peak memory across cached profiles and compare results."""
    from polyserve import cache as profile_cache
    from polyserve import memcal
    from polyserve.bench.report import load_results
    from polyserve.hardware import hardware_hash, probe as _probe
    from polyserve.models import TrialResult

    hw = _probe()
    hh = hardware_hash(hw)
    obs = []
    for _, p in profile_cache.list_profiles():
        if all_machines or p.hardware_hash == hh:
            obs += memcal.observations_from_trials(p.calibration_table, p.hardware_hash)
    for r in load_results(results):
        if all_machines or r.hardware_hash == hh:
            trials = [TrialResult(config=row.config, stage=row.label, metrics=row.metrics, launched=row.ok or row.error is None,
                                  error=row.error, memory=row.memory) for row in r.rows if row.memory is not None]
            obs += memcal.observations_from_trials(trials, r.hardware_hash)
    cals = memcal.analyse(obs)
    console.print(memcal.render_markdown(cals, hardware=None if all_machines else (hw.gpu.name if hw.gpu else hw.cpu.model_name)))
    if apply:
        if all_machines:
            err.print("[red]--apply needs this machine's observations only; drop --all[/]")
            raise typer.Exit(2)
        path = memcal.save_overrides(hh, cals)
        console.print(f"[green]wrote {path}[/]; the planner now uses these constants for hardware {hh}")


@app.command()
def profiles() -> None:
    """List cached profiles."""
    from polyserve import cache as profile_cache

    rows = profile_cache.list_profiles()
    if not rows:
        console.print(f"no profiles under {profile_cache.profiles_dir()}")
        return
    t = Table(title=str(profile_cache.profiles_dir()))
    for col in ("hardware", "model", "objective", "workload", "backend", "config", "trials", "created"):
        t.add_column(col)
    import datetime as dt

    for path, p in rows:
        t.add_row(p.hardware_hash, p.model_id, p.objective, p.workload, f"{p.backend} {p.backend_version or ''}",
                  p.config.key(), str(len(p.calibration_table)),
                  dt.datetime.fromtimestamp(p.created_at).strftime("%Y-%m-%d %H:%M"))
    console.print(t)


@app.command()
def serve(
    model: str,
    objective: str = typer.Option("balanced", "--objective", callback=_objective),
    workload: str = WORKLOAD_OPT,
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("0.0.0.0", "--host"),
    backend: Optional[str] = typer.Option(None, "--backend", help="Force a backend by name"),
    skip_calibration: bool = typer.Option(False, "--skip-calibration", help="Serve with backend defaults"),
    ttft_ceiling: Optional[float] = TTFT_OPT,
    tok_s_floor: Optional[float] = typer.Option(None, help="latency/efficiency: absolute tok/s floor"),
    power: str = POWER_OPT,
    tpot_ceiling: Optional[float] = TPOT_OPT,
    phases: str = PHASES_OPT,
    kv_connector: str = KV_OPT,
    quant: str = QUANT_OPT,
    kv_quant: str = KVQ_OPT,
    speculative: str = SPEC_OPT,
    prefix_cache: str = PREFIX_OPT,
    layout: str = LAYOUT_OPT,
    combine: str = COMBINE_OPT,
) -> None:
    """Discover hardware, calibrate once (cached), then serve an OpenAI-compatible API."""
    import uvicorn

    from polyserve import cache as profile_cache
    from polyserve.backends import get_backend
    from polyserve.pipeline import resolve_profile
    from polyserve.serve import Supervisor, create_app

    wl = _workload(workload)
    spec = ModelSpec(hf_id=model)
    profile = resolve_profile(spec, objective, force_backend=backend, skip_calibration=skip_calibration,
                              workload=wl, constraints=_constraints(ttft_ceiling, tok_s_floor, wl, tpot_ceiling),
                              progress=_progress, on_stage=lambda s: err.print(f"[dim]-> {s}[/]"),
                              power_mode=power, phases=phases, kv_connector=kv_connector,
                              options=_opts(quant, kv_quant, speculative, prefix_cache, combine), layout=layout)
    _print_profile(profile)
    if profile.prepared is None:
        err.print("[red]profile has no prepared model; run `polyserve recalibrate`[/]")
        raise typer.Exit(2)
    be = get_backend(profile.backend)
    be.materialize(profile.prepared, [profile.config.quant])  # no-op if already on disk
    if profile.disagg is not None:
        from polyserve.disagg import DisaggSupervisor, create_pd_app

        sup = DisaggSupervisor(be, profile.disagg, profile.prepared, log_dir=profile_cache.logs_dir() / spec.safe_id,
                               power=_power_controller(profile))
        err.print(f"[dim]starting prefill engine on GPU {profile.disagg.prefill_gpu} and decode engine on GPU "
                  f"{profile.disagg.decode_gpu} ...[/]")
        sup.start()
        app_ = create_pd_app(sup.prefill_url, sup.decode_url, profile=profile, status_fn=sup.status)
        upstream = f"prefill {sup.prefill_url}, decode {sup.decode_url}"
    elif profile.replicas > 1:
        from polyserve.layout import ReplicaSupervisor, create_lb_app

        gpus = [g.index for g in profile.hardware.gpus if g.vendor == "nvidia"][: profile.replicas]
        sup = ReplicaSupervisor(be, profile.config, profile.prepared, gpus, log_dir=profile_cache.logs_dir() / spec.safe_id)
        err.print(f"[dim]starting {len(gpus)} {profile.backend} replicas on GPUs {gpus} ...[/]")
        sup.start()
        app_ = create_lb_app(sup.urls, profile=profile, status_fn=sup.status)
        upstream = ", ".join(sup.urls)
    else:
        sup = Supervisor(be, profile.config, profile.prepared,
                         log_path=profile_cache.logs_dir() / spec.safe_id / "serve.log",
                         power=_power_controller(profile))
        err.print(f"[dim]starting {profile.backend} ...[/]")
        sup.start()
        app_ = create_app(sup.base_url, profile=profile, status_fn=sup.status)
        upstream = sup.base_url

    def _shutdown(*_: object) -> None:
        sup.stop()

    signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), sys.exit(0)))
    console.print(f"[bold green]PolyServe listening on http://{host}:{port}/v1[/]  "
                  f"(backend {profile.backend} on {upstream})")
    try:
        uvicorn.run(app_, host=host, port=port, log_level="warning")
    finally:
        _shutdown()


# --------------------------------------------------------------------------- power control

power_app = typer.Typer(help="GPU power cap and clock lock used by --power (needs root).", no_args_is_help=True)
app.add_typer(power_app, name="power")


@power_app.command("status")
def power_status() -> None:
    """Show what this GPU allows: power limit range, supported clocks, and whether control is permitted."""
    from polyserve import power as P
    from polyserve.hardware import probe as _probe

    hw = _probe()
    if hw.gpu is None:
        err.print("[red]no visible GPU; power tuning needs an NVIDIA GPU[/]")
        raise typer.Exit(1)
    try:
        caps = P.controller_for(hw.gpu.index).capabilities()
    except P.PowerControlUnavailable as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[bold]GPU {caps.gpu_index}[/]: {hw.gpu.name}")
    console.print(f"power limit: default {caps.power_limit_default_w} W, current {caps.power_limit_current_w} W, "
                  f"allowed {caps.power_limit_min_w}-{caps.power_limit_max_w} W")
    if caps.sm_clocks_mhz:
        console.print(f"SM clocks: {caps.sm_clocks_mhz[-1]}-{caps.sm_clocks_mhz[0]} MHz "
                      f"({len(caps.sm_clocks_mhz)} steps), max {caps.sm_clock_max_mhz} MHz")
    for knob, ok in (("cap", caps.can_cap), ("clock", caps.can_lock)):
        why = "" if ok else f"  ({caps.reasons.get(knob, 'unknown')})"
        console.print(f"{'power capping' if knob == 'cap' else 'clock locking'}: "
                      f"{'[green]permitted[/]' if ok else '[red]not permitted[/]'}{why}")
    trying = [p.label() for p in P.candidate_points(caps, "both") if not p.is_default]
    console.print(f"--power both would try: {', '.join(trying) if trying else 'nothing'}")
    if P.pending_restore():
        err.print("[yellow]a previous run left a power change behind; run `polyserve power reset`[/]")


@power_app.command("reset")
def power_reset(gpu: Optional[int] = typer.Option(None, "--gpu", help="GPU index (default: the recorded one)")) -> None:
    """Undo a power cap or clock lock left behind by a crashed run."""
    from polyserve import power as P

    try:
        state = P.reset_from_file(gpu)
    except P.PowerControlUnavailable as exc:
        err.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    if state:
        console.print(f"restored GPU {state.get('gpu_index')} power limit to {int(state['power_limit_mw']) // 1000} W "
                      "and unlocked clocks")
    else:
        console.print("no recorded change; unlocked clocks anyway")


if __name__ == "__main__":
    app()
