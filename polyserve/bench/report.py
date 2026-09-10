"""`polyserve report`: aggregate compare results into RESULTS.md, a summary statistic, and a plot.

The headline number is:

    Across N (GPU, model, workload) combinations, PolyServe improves throughput by a median of X%
    over the best stock/default configuration that satisfies the requested latency SLO.

"Best default" is chosen per combination as the highest-throughput non-PolyServe row whose TTFT at
its scored concurrency is under the workload's ceiling. If no default meets the SLO the fastest
default is used and the combination is flagged, because beating a row that violates the SLO is
not a like-for-like win.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from polyserve.bench.compare import ComparisonResult, ComparisonRow, results_dir


@dataclass
class Combo:
    result: ComparisonResult
    polyserve: Optional[ComparisonRow]
    baseline: Optional[ComparisonRow]
    baseline_met_slo: bool
    tok_s_gain_pct: Optional[float]  # (ps - baseline) / baseline
    ttft_delta_ms: Optional[float]  # ps - baseline (negative = better)
    joules_gain_pct: Optional[float]  # (baseline - ps) / baseline (positive = less energy)

    @property
    def key(self) -> str:
        r = self.result
        return f"{r.gpu or r.cpu} · {r.model_id} · {r.workload}"

    @property
    def valid(self) -> bool:
        return self.polyserve is not None and self.polyserve.ok and self.baseline is not None


@dataclass
class Summary:
    combos: List[Combo] = field(default_factory=list)

    @property
    def scored(self) -> List[Combo]:
        return [c for c in self.combos if c.valid and c.tok_s_gain_pct is not None]

    @property
    def like_for_like(self) -> List[Combo]:
        """Combos where the baseline itself met the SLO and PolyServe met it too."""
        return [c for c in self.scored if c.baseline_met_slo and c.polyserve.meets_slo]

    def median_gain(self, combos: Optional[List[Combo]] = None) -> Optional[float]:
        cs = self.like_for_like if combos is None else combos
        return statistics.median(c.tok_s_gain_pct for c in cs) if cs else None

    def wins(self, combos: Optional[List[Combo]] = None) -> Tuple[int, int]:
        cs = self.like_for_like if combos is None else combos
        return sum(1 for c in cs if c.tok_s_gain_pct > 0), len(cs)

    def headline(self) -> str:
        cs = self.like_for_like
        if not cs:
            return "No combination has both a PolyServe row and a stock default that meet the SLO yet."
        med = self.median_gain(cs)
        w, n = self.wins(cs)
        gains = sorted(c.tok_s_gain_pct for c in cs)
        return (
            f"Across {n} GPU/model/workload combination{'s' if n != 1 else ''}, PolyServe improves throughput "
            f"by a median of {med:+.0f}% (range {gains[0]:+.0f}% to {gains[-1]:+.0f}%) over the best stock/default "
            f"configuration that satisfies the requested latency SLO, winning {w} of {n}."
        )


# --------------------------------------------------------------------------- aggregation


def load_results(path: Optional[Path] = None) -> List[ComparisonResult]:
    root = path or results_dir()
    out: List[ComparisonResult] = []
    if not root.exists():
        return out
    for f in sorted(root.glob("*.json")):
        try:
            out.append(ComparisonResult.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            print(f"skipping {f}: {exc}")
    return out


def best_default(result: ComparisonResult) -> Tuple[Optional[ComparisonRow], bool]:
    ok = [r for r in result.default_rows() if r.ok and r.scored_tok_s is not None]
    if not ok:
        return None, False
    met = [r for r in ok if r.meets_slo]
    if met:
        return max(met, key=lambda r: r.scored_tok_s), True
    return max(ok, key=lambda r: r.scored_tok_s), False


def analyse(result: ComparisonResult) -> Combo:
    ps = result.polyserve_row
    base, met = best_default(result)
    gain = ttft = joules = None
    if ps and ps.ok and base and base.scored_tok_s:
        gain = (ps.scored_tok_s - base.scored_tok_s) / base.scored_tok_s * 100.0
        if ps.scored_ttft_ms is not None and base.scored_ttft_ms is not None:
            ttft = ps.scored_ttft_ms - base.scored_ttft_ms
        if ps.scored_joules_per_token and base.scored_joules_per_token:
            joules = (base.scored_joules_per_token - ps.scored_joules_per_token) / base.scored_joules_per_token * 100
    return Combo(result=result, polyserve=ps, baseline=base, baseline_met_slo=met, tok_s_gain_pct=gain,
                 ttft_delta_ms=ttft, joules_gain_pct=joules)


def summarize(results: List[ComparisonResult]) -> Summary:
    return Summary(combos=[analyse(r) for r in results])


# --------------------------------------------------------------------------- markdown


def _f(x: Optional[float], nd: int = 0, sign: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_markdown(summary: Summary) -> str:
    lines = ["# PolyServe benchmark results", "", summary.headline(), ""]
    flagged = [c for c in summary.scored if not (c.baseline_met_slo and c.polyserve.meets_slo)]
    if flagged:
        lines += [
            f"{len(flagged)} combination{'s' if len(flagged) != 1 else ''} excluded from the headline because the "
            "baseline or PolyServe missed the SLO (shown below, marked).",
            "",
        ]
    lines += [
        "| machine | model | workload | PolyServe pick | tok/s | best default | tok/s | gain | TTFT Δ | J/tok gain | calib |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in summary.combos:
        r = c.result
        if not c.valid:
            why = "no PolyServe row" if c.polyserve is None else ("PolyServe failed" if not c.polyserve.ok else "no default row")
            lines.append(f"| {r.gpu or r.cpu} | {r.model_id} | {r.workload} | {why} | | | | | | | |")
            continue
        mark = "" if (c.baseline_met_slo and c.polyserve.meets_slo) else " ⚠"
        lines.append(
            f"| {r.gpu or r.cpu} | {r.model_id} | {r.workload}{mark} | {c.polyserve.config_key} | "
            f"{_f(c.polyserve.scored_tok_s)} | {c.baseline.label} | {_f(c.baseline.scored_tok_s)} | "
            f"{_f(c.tok_s_gain_pct, 0, True)}% | {_f(c.ttft_delta_ms, 0, True)} ms | "
            f"{_f(c.joules_gain_pct, 0, True)}% | {_f(c.polyserve.calibration_seconds)}s / {c.polyserve.calibration_trials} |"
        )
    lines += ["", "⚠ = baseline or PolyServe missed the workload's TTFT SLO in that run; not counted in the headline.", ""]
    lines += [memory_section(summary), ""]
    lines += ["## Per-combination detail", ""]
    from polyserve.bench.compare import to_markdown

    for c in summary.combos:
        lines += [to_markdown(c.result), ""]
    return "\n".join(lines)


def memory_section(summary: Summary) -> str:
    """Planner accuracy over every row in every result (each compare row carries its observation)."""
    from polyserve import memcal
    from polyserve.models import TrialResult

    trials = []
    for c in summary.combos:
        for row in c.result.rows:
            if row.memory is not None:
                trials.append((TrialResult(config=row.config, stage=row.label, metrics=row.metrics,
                                           launched=row.ok or row.error is None, error=row.error, memory=row.memory),
                               c.result.hardware_hash))
    obs = []
    for t, hh in trials:
        obs += memcal.observations_from_trials([t], hh)
    return memcal.render_markdown(memcal.analyse(obs))


# --------------------------------------------------------------------------- plot

# Categorical palette: fixed hue per runtime (validated for CVD separation and contrast on #fcfcfb).
# Identity is never colour-alone: PolyServe is the only filled marker, defaults are hollow and direct-labelled.
RUNTIME_COLOR: Dict[str, str] = {
    "polyserve": "#e34948",
    "vllm": "#2a78d6",
    "sglang": "#1baf7a",
    "llamacpp-cuda": "#eb6834",
    "llamacpp-cpu": "#eb6834",
    "vllm-cpu": "#2a78d6",
    "ollama": "#4a3aa7",
}
SURFACE, INK, INK_2, INK_MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _panel_points(result: ComparisonResult) -> List[Tuple[str, str, float, float, bool]]:
    pts = []
    for r in result.rows:
        if r.ok and r.scored_tok_s and r.scored_ttft_ms:
            pts.append((r.label, r.runtime, r.scored_ttft_ms, r.scored_tok_s, bool(r.meets_slo)))
    return pts


def render_svg(results: List[ComparisonResult], panel_w: int = 300, panel_h: int = 220, cols: int = 3) -> str:
    """Small-multiples scatter: one panel per (machine, model, workload); x = TTFT p50 (log), y = tok/s.

    PolyServe is the filled red point; defaults are hollow circles coloured by runtime; the dashed
    vertical line is the workload's TTFT ceiling. Points to the right of it miss the SLO.
    """
    import math

    panels = [r for r in results if _panel_points(r)]
    if not panels:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="60"><text x="10" y="35" font-family="sans-serif" font-size="14">no results yet</text></svg>'
    rows_n = math.ceil(len(panels) / cols)
    W, H = cols * panel_w, rows_n * panel_h + 40
    ml, mr, mt, mb = 48, 14, 34, 36
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inter, Helvetica, Arial, sans-serif" font-size="11">',
        f'<style>text{{fill:{INK_2}}}.axis{{stroke:{AXIS};stroke-width:1}}.grid{{stroke:{GRID};stroke-width:1}}'
        f'.slo{{stroke:{INK_MUTED};stroke-dasharray:4 3}}.t{{fill:{INK_MUTED}}}</style>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
    ]
    for i, res in enumerate(panels):
        px, py = (i % cols) * panel_w, (i // cols) * panel_h
        pts = _panel_points(res)
        xs = [p[2] for p in pts] + [res.ttft_ceiling_ms]
        ys = [p[3] for p in pts]
        xmin, xmax = min(xs) / 1.5, max(xs) * 1.5
        ymax = max(ys) * 1.15
        iw, ih = panel_w - ml - mr, panel_h - mt - mb

        def sx(v: float) -> float:
            return px + ml + (math.log10(v) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * iw

        def sy(v: float) -> float:
            return py + mt + ih - v / ymax * ih

        title = f"{res.gpu or res.cpu} · {res.model_id.split('/')[-1]} · {res.workload}"
        out.append(f'<text x="{px + ml}" y="{py + 16}" font-weight="600" font-size="12" fill="{INK}">{title}</text>')
        # y grid + labels
        for k in range(5):
            v = ymax * k / 4
            y = sy(v)
            out.append(f'<line class="grid" x1="{px + ml}" x2="{px + ml + iw}" y1="{y:.1f}" y2="{y:.1f}"/>')
            out.append(f'<text class="t" x="{px + ml - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10">{v:.0f}</text>')
        # x ticks at decades
        d = math.floor(math.log10(xmin))
        while 10**d <= xmax:
            v = 10**d
            if v >= xmin:
                x = sx(v)
                out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{py + mt}" y2="{py + mt + ih}"/>')
                out.append(f'<text class="t" x="{x:.1f}" y="{py + mt + ih + 14}" text-anchor="middle" font-size="10">{v:g}</text>')
            d += 1
        out.append(f'<line class="axis" x1="{px + ml}" x2="{px + ml + iw}" y1="{py + mt + ih}" y2="{py + mt + ih}"/>')
        out.append(f'<line class="axis" x1="{px + ml}" x2="{px + ml}" y1="{py + mt}" y2="{py + mt + ih}"/>')
        out.append(f'<text x="{px + ml + iw / 2:.1f}" y="{py + panel_h - 6}" text-anchor="middle" font-size="10">TTFT p50 (ms, log)</text>')
        out.append(f'<text transform="translate({px + 12},{py + mt + ih / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="10">tok/s</text>')
        # SLO line
        xs_ = sx(res.ttft_ceiling_ms)
        out.append(f'<line class="slo" x1="{xs_:.1f}" x2="{xs_:.1f}" y1="{py + mt}" y2="{py + mt + ih}"/>')
        out.append(f'<text class="t" x="{xs_ + 3:.1f}" y="{py + mt + 10}" font-size="9">SLO {res.ttft_ceiling_ms:.0f} ms</text>')
        # points
        for label, runtime, ttft, tok, met in pts:
            x, y = sx(ttft), sy(tok)
            color = RUNTIME_COLOR.get(runtime, "#888")
            if label == "polyserve":
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="{SURFACE}" stroke-width="2"><title>PolyServe: {tok:.0f} tok/s, TTFT {ttft:.0f} ms</title></circle>')
                out.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="10" font-weight="600" fill="{INK}">PolyServe</text>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{SURFACE}" stroke="{color}" stroke-width="2"><title>{label}: {tok:.0f} tok/s, TTFT {ttft:.0f} ms</title></circle>')
                out.append(f'<text x="{x + 9:.1f}" y="{y + 4:.1f}" font-size="9">{label.replace("-default", "")}</text>')
    # legend
    lx, ly = 12, H - 14
    for name, color in (("PolyServe", RUNTIME_COLOR["polyserve"]), ("vLLM", RUNTIME_COLOR["vllm"]),
                        ("SGLang", RUNTIME_COLOR["sglang"]), ("llama.cpp", RUNTIME_COLOR["llamacpp-cuda"]),
                        ("Ollama", RUNTIME_COLOR["ollama"])):
        if name == "PolyServe":
            out.append(f'<circle cx="{lx + 5}" cy="{ly}" r="5" fill="{color}"/>')
        else:
            out.append(f'<circle cx="{lx + 5}" cy="{ly}" r="4" fill="{SURFACE}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text x="{lx + 14}" y="{ly + 4}" font-size="10">{name}</text>')
        lx += 14 + 7 * len(name) + 18
    out.append("</svg>")
    return "\n".join(out)


def write_report(results_path: Optional[Path] = None, out_dir: Optional[Path] = None) -> Tuple[Path, Path, Summary]:
    results = load_results(results_path)
    summary = summarize(results)
    out_dir = out_dir or (results_path or results_dir()).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "RESULTS.md"
    svg = out_dir / "throughput_vs_ttft.svg"
    md.write_text(render_markdown(summary), encoding="utf-8")
    svg.write_text(render_svg(results), encoding="utf-8")
    return md, svg, summary
