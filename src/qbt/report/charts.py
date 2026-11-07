"""Dependency-free chart rendering for the HTML tearsheet.

Charts are inline SVG plus a thin layer of HTML for text. The plot area is an
SVG with ``preserveAspectRatio="none"`` drawn in a fixed 1000 x 1000 user
space, so it stretches to any width while strokes stay crisp via
``vector-effect: non-scaling-stroke``. Axis labels and direct labels are HTML
positioned in percentages, which keeps text at its real size on a phone
instead of shrinking with the drawing. No JavaScript is involved.

Colors are never hard-coded here: every mark carries a class and the page's
CSS custom properties decide light and dark values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd

VIEW = 1000.0  # user-space size of the plot area in both directions


@dataclass
class Series:
    values: np.ndarray
    label: str
    css: str
    """CSS class applied to the mark (e.g. ``"strategy"`` or ``"bench"``)."""
    kind: str = "line"  # "line" or "area" (area fills toward ``baseline``)
    end_label: str | None = None


@dataclass
class Tick:
    value: float
    label: str


def _fmt_path(xs: np.ndarray, ys: np.ndarray) -> str:
    finite = np.isfinite(ys)
    parts: list[str] = []
    pen_down = False
    for x, y, ok in zip(xs, ys, finite, strict=True):
        if not ok:
            pen_down = False
            continue
        parts.append(f"{'L' if pen_down else 'M'}{x:.1f} {y:.1f}")
        pen_down = True
    return "".join(parts)


def nice_step(span: float, target_ticks: int = 5) -> float:
    """A 1/2/2.5/5 x 10^k step giving roughly ``target_ticks`` intervals."""
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * magnitude:
            return mult * magnitude
    return 10 * magnitude


def linear_ticks(lo: float, hi: float, fmt: Callable[[float], str], target: int = 5) -> list[Tick]:
    step = nice_step(hi - lo, target)
    start = math.ceil(lo / step - 1e-9) * step
    ticks = []
    value = start
    while value <= hi + 1e-9 * step:
        ticks.append(Tick(0.0 if abs(value) < step * 1e-9 else value, fmt(value)))
        value += step
    return ticks


_LOG_MANTISSAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)


def log_ticks(lo: float, hi: float, fmt: Callable[[float], str], max_ticks: int = 6) -> list[Tick]:
    """Readable ticks for a log axis: 1, 1.25, 1.5, 2, ... x 10^k, thinned."""
    if lo <= 0 or hi <= lo:
        return []
    candidates = []
    for k in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in _LOG_MANTISSAS:
            v = m * 10**k
            if lo <= v <= hi:
                candidates.append(v)
    while len(candidates) > max_ticks:
        candidates = candidates[::2]
    return [Tick(v, fmt(v)) for v in candidates]


def year_ticks(index: pd.DatetimeIndex, max_ticks: int = 10) -> list[tuple[float, str]]:
    """``(position in [0, 1], label)`` at the first bar of each calendar year."""
    if len(index) < 2:
        return []
    years = index.year
    firsts = np.flatnonzero(np.r_[True, years[1:] != years[:-1]])
    # Skip a partial first year if it starts late, to avoid crowding the origin.
    if len(firsts) > 1 and firsts[1] < len(index) * 0.04:
        firsts = firsts[1:]
    every = max(1, math.ceil(len(firsts) / max_ticks))
    picked = firsts[::every]
    n = len(index) - 1
    return [(i / n, str(years[i])) for i in picked]


def _spread_labels(positions: list[float], min_gap: float, lo: float, hi: float) -> list[float]:
    """Nudge sorted pixel positions apart so labels never overlap."""
    order = np.argsort(positions)
    placed = [positions[i] for i in order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    overflow = placed[-1] - hi if placed else 0
    if overflow > 0:
        placed = [p - overflow for p in placed]
    placed = [max(lo, p) for p in placed]
    out = [0.0] * len(positions)
    for rank, idx in enumerate(order):
        out[idx] = placed[rank]
    return out


def line_chart(
    index: pd.DatetimeIndex,
    series: Sequence[Series],
    y_ticks: list[Tick],
    y_domain: tuple[float, float],
    height: int = 280,
    log: bool = False,
    baseline: float | None = None,
    zero_line: float | None = None,
    aria_label: str = "",
) -> str:
    """Render a time-series chart as an HTML fragment."""
    lo, hi = y_domain
    if log:
        lo, hi = math.log(lo), math.log(hi)
    span = (hi - lo) or 1.0

    def y_of(v: np.ndarray | float) -> np.ndarray | float:
        v = np.log(v) if log else v
        return VIEW * (1.0 - (v - lo) / span)

    n = len(index)
    xs = np.linspace(0.0, VIEW, n) if n > 1 else np.zeros(1)

    grid = "".join(
        f'<line class="grid" x1="0" x2="{VIEW:.0f}" y1="{y_of(t.value):.1f}" '
        f'y2="{y_of(t.value):.1f}"/>'
        for t in y_ticks
    )
    marks = []
    for s in series:
        ys = np.asarray(y_of(np.asarray(s.values, dtype=float)), dtype=float)
        path = _fmt_path(xs, ys)
        if s.kind == "area" and path:
            base = y_of(baseline if baseline is not None else (math.exp(lo) if log else lo))
            area = f"{path}L{xs[-1]:.1f} {base:.1f}L{xs[0]:.1f} {base:.1f}Z"
            marks.append(f'<path class="fill {s.css}" d="{area}"/>')
        marks.append(f'<path class="mark {s.css}" d="{path}"/>')
    zero = ""
    if zero_line is not None:
        zy = y_of(zero_line)
        zero = f'<line class="zero" x1="0" x2="{VIEW:.0f}" y1="{zy:.1f}" y2="{zy:.1f}"/>'

    svg = (
        f'<svg class="plot-svg" viewBox="0 0 {VIEW:.0f} {VIEW:.0f}" preserveAspectRatio="none" '
        f'role="img" aria-label="{escape(aria_label)}">{grid}{zero}{"".join(marks)}</svg>'
    )

    y_labels = "".join(
        f'<span style="top:{y_of(t.value) / VIEW * 100:.2f}%">{escape(t.label)}</span>'
        for t in y_ticks
        if 0 <= y_of(t.value) <= VIEW
    )

    ends = [s for s in series if s.end_label]
    end_html = ""
    if ends:
        raw = []
        for s in ends:
            last = pd.Series(s.values).dropna()
            raw.append(float(y_of(last.iloc[-1])) / VIEW * height if len(last) else 0.0)
        placed = _spread_labels(raw, min_gap=18.0, lo=8.0, hi=height - 8.0)
        end_html = "".join(
            f'<span class="end {s.css}" style="top:{p:.1f}px">{escape(s.end_label or "")}</span>'
            for s, p in zip(ends, placed, strict=True)
        )

    x_labels = "".join(
        f'<span style="left:{pos * 100:.2f}%">{escape(label)}</span>'
        for pos, label in year_ticks(index)
    )
    return (
        f'<div class="chart" style="--h:{height}px">'
        f'<div class="y-axis" aria-hidden="true">{y_labels}</div>'
        f'<div class="plot">{svg}</div>'
        f'<div class="end-labels" aria-hidden="true">{end_html}</div>'
        f'<div class="x-axis" aria-hidden="true">{x_labels}</div>'
        f"</div>"
    )


def _cell_text(value: float) -> str:
    text = f"{value * 100:.1f}"
    return "0.0" if text == "-0.0" else text.replace("-", "\u2212")


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def heatmap(table: pd.DataFrame, yearly: pd.Series, cap: float | None = None) -> str:
    """Monthly returns heatmap (years x months, plus a full-year column) as SVG.

    Cell fills use ``color-mix`` between the neutral and the gain/loss pole
    tokens, so the scale follows the page's light or dark theme. Each cell has
    a ``<title>`` for a native tooltip.
    """
    values = table.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if cap is None:
        cap = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 0.05
        cap = max(cap, 0.01)

    cell_w, cell_h, gap = 52, 30, 2
    label_w, head_h, year_gap = 48, 24, 10
    cols = 13
    width = label_w + cols * cell_w + year_gap
    height = head_h + len(table) * cell_h

    year_cap = cap * math.sqrt(12)  # annual moves are ~sqrt(12) x monthly ones

    def cell(x: float, y: float, value: float, title: str, scale: float = cap) -> str:
        if not np.isfinite(value):
            return (
                f'<rect class="hm-empty" x="{x + gap / 2}" y="{y + gap / 2}" '
                f'width="{cell_w - gap}" height="{cell_h - gap}" rx="2"/>'
            )
        strength = min(abs(value) / scale, 1.0) ** 0.85
        pole = "var(--gain)" if value >= 0 else "var(--loss)"
        pct = round(strength * 100)
        fill = f"color-mix(in oklab, {pole} {pct}%, var(--neutral))"
        text_class = "hm-text on-dark" if strength > 0.6 else "hm-text"
        return (
            f"<g><title>{escape(title)}: {value * 100:+.2f}%</title>"
            f'<rect x="{x + gap / 2}" y="{y + gap / 2}" width="{cell_w - gap}" '
            f'height="{cell_h - gap}" rx="2" style="fill:{fill}"/>'
            f'<text class="{text_class}" x="{x + cell_w / 2}" y="{y + cell_h / 2}">'
            f"{_cell_text(value)}</text></g>"
        )

    parts = []
    for j, name in enumerate((*MONTHS, "Year")):
        x = label_w + j * cell_w + (year_gap if j == 12 else 0) + cell_w / 2
        parts.append(f'<text class="hm-head" x="{x}" y="{head_h / 2 + 2}">{name}</text>')
    for i, (year, row) in enumerate(table.iterrows()):
        y = head_h + i * cell_h
        parts.append(f'<text class="hm-year" x="{label_w - 10}" y="{y + cell_h / 2}">{year}</text>')
        for j in range(12):
            month = j + 1
            value = float(row.get(month, np.nan))
            parts.append(cell(label_w + j * cell_w, y, value, f"{MONTHS[j]} {year}"))
        annual = float(yearly.get(year, np.nan))
        parts.append(cell(label_w + 12 * cell_w + year_gap, y, annual, str(year), year_cap))

    return (
        f'<svg class="heatmap" viewBox="0 0 {width} {height}" width="{width}" '
        f'role="img" aria-label="Monthly returns by year, in percent">{"".join(parts)}</svg>'
    )
