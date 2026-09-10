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
    for col in ("stage", "config", "tok/s", "TTFT ms", "TPOT ms", "peak MB", "W", "J/tok", "status"):
        t.add_column(col, justify="right" if col not in ("stage", "config", "status") else "left",
                     overflow="fold", min_width=(40 if col == "config" else None))
    for r in results:
        m = r.metrics
        status = "ok" if r.ok else (r.error or "failed").splitlines()[0][:40]
        key = r.config.key()
        style = "bold green" if winner and key == winner else None
        t.add_row(r.stage, key, _fmt(m.tok_s), _fmt(m.ttft_ms, 0), _fmt(m.tpot_ms, 1), _fmt(m.peak_mem_mb, 0),
                  _fmt(m.power_w, 0), _fmt(m.joules_per_token, 3), status, style=style)
    return t


def _print_profile(p: Profile) -> None:
    console.print(f"[bold]objective[/]: {p.objective}   [bold]workload[/]: {p.workload}")
    console.print(f"[bold]backend[/]: {p.backend} {p.backend_version or ''}")
    console.print(f"[bold]config[/]:  {p.config.key()}")
    console.print(f"[bold]launch[/]:  {' '.join(p.launch_args)}")
    if p.launch_env:
        console.print(f"[bold]env[/]:     {' '.join(f'{k}={v}' for k, v in p.launch_env.items())}")
    for n in p.notes:
        console.print(f"[dim]note: {n}[/]")


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


def _constraints(ttft_ceiling: Optional[float], tok_s_floor: Optional[float], workload=None):
    from polyserve.calibrate.objectives import Constraints

    ceiling = ttft_ceiling if ttft_ceiling is not None else (workload.ttft_ceiling_ms if workload else 500.0)
    return Constraints(ttft_ceiling_ms=ceiling, tok_s_floor_abs=tok_s_floor)


WORKLOAD_OPT = typer.Option("default", "--workload", "-w", help="Workload preset; see `polyserve workloads`")
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

    t = Table(title="Workload presets")
    for col in ("name", "prompts", "prefill", "decode", "concurrency", "TTFT ceiling ms"):
        t.add_column(col, justify="right" if col != "name" else "left")
    for w in workload_table():
        t.add_row(str(w["name"]), str(w["n_prompts"]), str(w["prefill_tokens"]), str(w["decode_tokens"]),
                  "/".join(str(c) for c in w["concurrencies"]), f"{w['ttft_ceiling_ms']:.0f}")
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
    result = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl)
    err.print(f"{len(result.all_feasible)}/{result.total_considered} configs feasible; "
              f"calibrating for {objective} on workload {wl.name}")
    profile = calibrate(hw, spec, objective, result, reg, workload=wl,
                        constraints=_constraints(ttft_ceiling, tok_s_floor, wl), progress=_progress)
    console.print(_trial_table(profile.calibration_table, winner=profile.config.key()))
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
) -> None:
    """Force a calibration rerun and overwrite the cached profile."""
    from polyserve.pipeline import resolve_profile

    wl = _workload(workload)
    profile = resolve_profile(ModelSpec(hf_id=model), objective, force_backend=backend, recalibrate=True,
                              workload=wl, constraints=_constraints(ttft_ceiling, tok_s_floor, wl),
                              progress=_progress, on_stage=lambda s: err.print(f"[dim]-> {s}[/]"))
    console.print(_trial_table(profile.calibration_table, winner=profile.config.key()))
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
) -> None:
    """Measure PolyServe's pick vs stock defaults (and Ollama) on one workload; write a results JSON."""
    from polyserve.bench import compare as _compare, to_markdown
    from polyserve.bench.compare import save
    from polyserve.hardware import probe as _probe
    from polyserve.pipeline import prepare_and_plan, resolve_profile, select

    wl = _workload(workload)
    cons = _constraints(ttft_ceiling, tok_s_floor, wl)
    hw = _probe()
    spec = ModelSpec(hf_id=model)
    profile = resolve_profile(spec, objective, force_backend=backend, workload=wl, constraints=cons,
                              progress=_progress, hw=hw, on_stage=lambda s: err.print(f"[dim]-> {s}[/]"))
    _print_profile(profile)
    candidates, reg = select(hw, spec, force=backend)
    planned = prepare_and_plan(hw, spec, candidates, reg, materialize=True, workload=wl)

    def _row_progress(label: str, row) -> None:
        if row is None:
            err.print(f"[cyan]compare[/] {label} ...")
        elif row.ok:
            err.print(f"[green]compare[/] {label}: {row.scored_tok_s:.1f} tok/s @c{row.scored_concurrency}, "
                      f"TTFT {_fmt(row.scored_ttft_ms, 0)} ms, SLO {'met' if row.meets_slo else 'missed'}")
        else:
            err.print(f"[red]compare[/] {label} FAILED: {(row.error or '').splitlines()[0][:80]}")

    result = _compare(hw, spec, profile, planned.prepared, reg, workload=wl, constraints=cons,
                      ollama_tag=ollama_model, include=include or None, progress=_row_progress)
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


@app.command("memory-report")
def memory_report(
    results: Optional[Path] = typer.Option(None, "--results", help="compare results dir (default benchmarks/results)"),
    apply: bool = typer.Option(False, "--apply", help="Write fitted workspace/margin to ~/.polyserve/memory-model.json"),
    all_machines: bool = typer.Option(False, "--all", help="Include profiles/results from other hardware hashes"),
) -> None:
    """Planner prediction vs measured peak memory across cached profiles and compare results."""
    from polyserve import cache as profile_cache
    from polyserve import memcal
    from polyserve.bench.compare import ComparisonResult
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
                              workload=wl, constraints=_constraints(ttft_ceiling, tok_s_floor, wl),
                              progress=_progress, on_stage=lambda s: err.print(f"[dim]-> {s}[/]"))
    _print_profile(profile)
    if profile.prepared is None:
        err.print("[red]profile has no prepared model; run `polyserve recalibrate`[/]")
        raise typer.Exit(2)
    be = get_backend(profile.backend)
    be.materialize(profile.prepared, [profile.config.quant])  # no-op if already on disk
    sup = Supervisor(be, profile.config, profile.prepared,
                     log_path=profile_cache.logs_dir() / spec.safe_id / "serve.log")
    err.print(f"[dim]starting {profile.backend} ...[/]")
    sup.start()
    app_ = create_app(sup.base_url, profile=profile, status_fn=sup.status)

    def _shutdown(*_: object) -> None:
        sup.stop()

    signal.signal(signal.SIGTERM, lambda *_: (_shutdown(), sys.exit(0)))
    console.print(f"[bold green]PolyServe listening on http://{host}:{port}/v1[/]  "
                  f"(backend {profile.backend} on {sup.base_url})")
    try:
        uvicorn.run(app_, host=host, port=port, log_level="warning")
    finally:
        _shutdown()


if __name__ == "__main__":
    app()
