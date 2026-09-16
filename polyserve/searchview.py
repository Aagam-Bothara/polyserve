"""The search, drawn while it runs: one row per stage, one bar per trial, the leader called out.

The picture is only drawn when stderr is a terminal. Piped or redirected output (CI, nohup, the benchmark
pods) keeps the one-line-per-trial log, which scripts grep for, so turning this on changes no recorded run.
The same renderer redraws a finished search from a profile's calibration table (`polyserve profiles --trace`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from polyserve.models import TrialResult

ProgressFn = Callable[[str, object, Optional[TrialResult]], None]

BARS = "▁▂▃▄▅▆▇█"
ASCII_BARS = ".:-=+*#@"
RUNNING = "…"
ASCII_RUNNING = "?"


def _unicode_ok(console: Console) -> bool:
    """Legacy Windows consoles are cp1252; ask the stream whether it can carry the bar characters."""
    if getattr(console, "legacy_windows", False):
        return False
    enc = getattr(getattr(console, "file", None), "encoding", None) or "utf-8"
    try:
        BARS.encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


@dataclass(eq=False)
class _Trial:
    key: str
    tok_s: float = 0.0
    ok: bool = True
    feasible: bool = True
    running: bool = False


@dataclass(eq=False)
class _Stage:
    name: str
    trials: List[_Trial] = field(default_factory=list)

    @property
    def best(self) -> Optional[_Trial]:
        done = [t for t in self.trials if not t.running and t.ok and t.feasible]
        return max(done, key=lambda t: t.tok_s) if done else None


class SearchView:
    """Collects calibration trials as they land and renders them as stage rows of bars.

    `progress` has the signature the search calls (stage, config, result-or-None), so it drops into the
    place the log printer had. `stop()` leaves the final picture on screen.
    """

    def __init__(self, console: Console, objective: Optional[str] = None, constraints: object = None,
                 fallback: Optional[ProgressFn] = None, graphical: Optional[bool] = None) -> None:
        self.console = console
        self.objective = objective
        self.constraints = constraints
        self.fallback = fallback
        self.graphical = console.is_terminal if graphical is None else graphical
        self.ascii = not _unicode_ok(console)
        self.stages: List[_Stage] = []
        self.notes: List[str] = []
        self.started = time.time()
        self.live: Optional[Live] = None
        self.live_enabled = self.graphical
        self.winner: Optional[str] = None

    # ----------------------------------------------------------------- collecting

    def progress(self, stage: str, cfg: object, res: Optional[TrialResult]) -> None:
        """One trial starting (res None) or finishing. The search calls this; keep it cheap and never raise."""
        if not self.graphical:
            if self.fallback is not None:
                self.fallback(stage, cfg, res)
            return
        key = cfg.key() if hasattr(cfg, "key") else str(cfg)
        row = self._stage(stage)
        if res is None:
            row.trials.append(_Trial(key=key, running=True))
        else:
            t = next((t for t in row.trials if t.running and t.key == key), None)
            if t is None:
                t = _Trial(key=key)
                row.trials.append(t)
            metrics, feasible = self._scored(res)
            t.running, t.ok, t.feasible = False, res.ok, feasible
            t.tok_s = metrics.tok_s if res.ok else 0.0
        self._refresh()

    def on_stage(self, name: str) -> None:
        """What the search says between stages: which engines it is tuning, what it skipped and why."""
        if not self.graphical:
            self.console.print(f"[dim]-> {name}[/]")
            return
        self.notes.append(name)
        self._refresh()

    def _stage(self, name: str) -> _Stage:
        for s in self.stages:
            if s.name == name:
                return s
        self.stages.append(_Stage(name=name))
        return self.stages[-1]

    def _scored(self, res: TrialResult):
        """The level the objective scores a trial at, not its fastest: the same number the log line prints."""
        if not res.ok or self.objective is None:
            return res.metrics, True
        from polyserve.calibrate.objectives import rank

        ranked = rank([res], self.objective, self.constraints)
        return (ranked[0].metrics, ranked[0].feasible) if ranked else (res.metrics, True)

    # ----------------------------------------------------------------- drawing

    def _refresh(self) -> None:
        if not self.live_enabled:
            return
        if self.live is None:
            self.live = Live(console=self.console, refresh_per_second=8)
            self.live.start()
        self.live.update(self.render())

    def stop(self) -> None:
        """End the live picture, leaving the last frame printed."""
        if self.live is not None:
            self.live.update(self.render())
            self.live.stop()
            self.live = None

    @property
    def leader(self) -> Optional[_Trial]:
        best = [s.best for s in self.stages if s.best is not None]
        return max(best, key=lambda t: t.tok_s) if best else None

    def _bar(self, t: _Trial, best: float) -> Text:
        bars = ASCII_BARS if self.ascii else BARS
        if t.running:
            return Text(ASCII_RUNNING if self.ascii else RUNNING, style="dim")
        if not t.ok:
            return Text(bars[0], style="red")
        level = 0 if best <= 0 else min(len(bars) - 1, int(round(t.tok_s / best * (len(bars) - 1))))
        style = "yellow" if not t.feasible else ("bold green" if t.key == (self.winner or "") else "cyan")
        return Text(bars[level], style=style)

    def render(self) -> RenderableType:
        """The whole picture: a header, one row per stage, and the last notes the search made."""
        trials = [t for s in self.stages for t in s.trials]
        done = [t for t in trials if not t.running]
        best = max((t.tok_s for t in done if t.ok and t.feasible), default=0.0)
        lead = self.leader
        if self.winner is None and lead is not None:
            self.winner = lead.key

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim")   # stage
        table.add_column(no_wrap=True)                   # bars
        table.add_column(justify="right")                # best in stage
        for s in self.stages:
            marks = Text()
            for t in s.trials:
                marks.append_text(self._bar(t, best))
            top = s.best
            table.add_row(s.name, marks, f"{top.tok_s:.0f}" if top else "-")

        head = Text()
        head.append(f"{len(done)} trials", style="bold")
        head.append(f" in {(time.time() - self.started) / 60:.0f} min" if self.live is not None else "")
        if lead is not None:
            head.append("   leader ", style="dim")
            head.append(f"{lead.tok_s:.0f} tok/s ", style="bold green")
            head.append(lead.key, style="green")
        parts: List[RenderableType] = [head, table]
        for n in self.notes[-2:]:
            parts.append(Text(f"note: {n}", style="dim"))
        return Group(*parts)

    # ----------------------------------------------------------------- a finished search

    @classmethod
    def from_trials(cls, trials: Sequence[TrialResult], console: Console, winner: Optional[str] = None,
                    objective: Optional[str] = None, constraints: object = None,
                    notes: Sequence[str] = ()) -> "SearchView":
        """Redraw a search that already ran, from a profile's calibration table."""
        view = cls(console, objective=objective, constraints=constraints, graphical=True)
        view.live_enabled = False
        for r in trials:
            view.progress(r.stage, r.config, r)
        view.notes = list(notes)
        view.winner = winner
        return view


def stage_counts(view: SearchView) -> Dict[str, int]:
    """Trials per stage, for tests and for anything that wants the shape without the drawing."""
    return {s.name: len(s.trials) for s in view.stages}
