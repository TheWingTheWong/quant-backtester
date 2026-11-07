"""Self-contained HTML tearsheet.

One file, no JavaScript, no external requests: styles are inline and charts
are inline SVG (see :mod:`qbt.report.charts`). The page follows the layout of
a printed fund factsheet - title block, growth chart beside the statistics
table, then drawdowns, rolling risk-adjusted return and the monthly grid -
and prints cleanly to A4/Letter.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import cache
from html import escape
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from qbt import __version__
from qbt.engine import BacktestResult
from qbt.metrics import (
    annual_returns,
    compute_metrics,
    drawdown_series,
    format_metric,
    monthly_returns,
    rolling_sharpe,
    top_drawdowns,
)
from qbt.report.charts import Series, heatmap, line_chart, linear_ticks, log_ticks

METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Return", ("total_return", "cagr", "best_day", "worst_day")),
    ("Risk", ("ann_vol", "max_drawdown", "max_dd_duration", "skew")),
    ("Risk-adjusted", ("sharpe", "sortino", "calmar")),
    ("Trading", ("hit_rate", "profit_factor", "turnover", "avg_exposure")),
)

LABELS = {
    "total_return": "Total return",
    "cagr": "CAGR",
    "best_day": "Best day",
    "worst_day": "Worst day",
    "ann_vol": "Volatility, annualized",
    "max_drawdown": "Max drawdown",
    "max_dd_duration": "Longest drawdown, days",
    "skew": "Skew, daily",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "hit_rate": "Hit rate",
    "profit_factor": "Profit factor",
    "turnover": "Turnover, x per year",
    "avg_exposure": "Gross exposure, avg",
}


@cache
def _stylesheet() -> str:
    return resources.files("qbt.report").joinpath("tearsheet.css").read_text(encoding="utf-8")


def _money(value: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= div:
            text = f"{value / div:.2f}".rstrip("0").rstrip(".")
            return f"{text}{unit}"
    return f"{value:,.0f}"


def _pct(value: float, digits: int = 0) -> str:
    return f"{value * 100:.{digits}f}%"


def _date(ts: pd.Timestamp) -> str:
    return f"{ts.day} {ts:%b %Y}"


def _metrics_table(strategy: dict[str, float], bench: dict[str, float] | None, name: str) -> str:
    head = f'<th scope="col">{escape(name)}</th>'
    if bench is not None:
        head += '<th scope="col">Benchmark</th>'
    rows = []
    for group, keys in METRIC_GROUPS:
        span = 3 if bench is not None else 2
        rows.append(f'<tr class="group"><th colspan="{span}" scope="colgroup">{group}</th></tr>')
        for key in keys:
            if key not in strategy:
                continue
            cells = f"<td>{format_metric(key, strategy[key])}</td>"
            if bench is not None:
                value = bench.get(key)
                cells += f"<td>{format_metric(key, value) if value is not None else '–'}</td>"
            rows.append(f'<tr><th scope="row">{LABELS.get(key, key)}</th>{cells}</tr>')
    return (
        '<table class="stats">'
        f'<thead><tr><th scope="col"><span class="sr-only">Metric</span></th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _drawdowns_table(returns: pd.Series) -> str:
    rows = []
    for d in top_drawdowns(returns, 5):
        end = _date(d.end) if d.end is not None else "Not recovered"
        rows.append(
            f"<tr><td>{_pct(d.depth, 1)}</td><td>{_date(d.start)}</td><td>{_date(d.trough)}</td>"
            f"<td>{end}</td><td>{d.length:,}</td></tr>"
        )
    if not rows:
        return '<p class="note">No drawdowns: the equity curve never fell below a prior peak.</p>'
    return (
        '<table class="ledger"><thead><tr><th scope="col">Depth</th><th scope="col">Peak</th>'
        '<th scope="col">Trough</th><th scope="col">Recovered</th>'
        f'<th scope="col">Days</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _settings_table(settings: Mapping[str, str]) -> str:
    rows = "".join(
        f'<tr><th scope="row">{escape(k)}</th><td>{escape(v)}</td></tr>'
        for k, v in settings.items()
    )
    return f'<table class="ledger settings"><tbody>{rows}</tbody></table>'


def _legend(items: list[tuple[str, str]]) -> str:
    entries = "".join(
        f'<li><svg class="key" viewBox="0 0 24 8" aria-hidden="true">'
        f'<line class="mark {css}" x1="1" x2="23" y1="4" y2="4"/></svg>{escape(label)}</li>'
        for css, label in items
    )
    return f'<ul class="legend">{entries}</ul>'


def render_tearsheet(
    result: BacktestResult,
    title: str | None = None,
    subtitle: str = "",
    settings: Mapping[str, str] | None = None,
    note: str = "",
) -> str:
    """Render ``result`` as a complete HTML document string."""
    returns = result.returns
    index = returns.index
    title = title or result.name
    bench = result.benchmark
    strat_stats = result.metrics()
    bench_stats = compute_metrics(bench) if bench is not None else None

    # Growth of capital, log scale.
    equity = result.equity
    curves = [equity]
    series = [Series(equity.to_numpy(), title, "strategy", end_label=_money(equity.iloc[-1]))]
    legend = [("strategy", title)]
    if bench is not None:
        bench_equity = result.initial_capital * (1.0 + bench.reindex(index).fillna(0.0)).cumprod()
        curves.append(bench_equity)
        series.append(
            Series(
                bench_equity.to_numpy(),
                "Benchmark",
                "bench",
                end_label=_money(bench_equity.iloc[-1]),
            )
        )
        legend.append(("bench", "Benchmark: equal-weight buy and hold"))
    lo = min(float(c.min()) for c in curves)
    hi = max(float(c.max()) for c in curves)
    pad = (hi / lo) ** 0.04
    domain = (lo / pad, hi * pad)
    growth_chart = line_chart(
        index,
        series,
        log_ticks(*domain, fmt=_money),
        domain,
        height=300,
        log=True,
        aria_label=f"Growth of {_money(result.initial_capital)}, log scale",
    )

    # Underwater curve.
    dd = drawdown_series(returns)
    dd_lo = min(float(dd.min()), -0.01)
    dd_series = [Series(dd.to_numpy(), "Drawdown", "loss", kind="area")]
    dd_legend = [("loss", title)]
    if bench is not None:
        bench_dd = drawdown_series(bench.reindex(index).fillna(0.0))
        dd_lo = min(dd_lo, float(bench_dd.min()))
        dd_series.append(Series(bench_dd.to_numpy(), "Benchmark", "bench"))
        dd_legend.append(("bench", "Benchmark"))
    dd_domain = (dd_lo * 1.06, 0.0)
    dd_chart = line_chart(
        index,
        dd_series,
        linear_ticks(dd_domain[0], 0.0, fmt=lambda v: _pct(v), target=4),
        dd_domain,
        height=180,
        baseline=0.0,
        aria_label="Drawdown from running peak",
    )

    # Rolling one-year Sharpe.
    roll = rolling_sharpe(returns, 252)
    roll_series = [Series(roll.to_numpy(), "Rolling Sharpe", "strategy")]
    roll_values = [roll]
    roll_legend = [("strategy", title)]
    if bench is not None:
        bench_roll = rolling_sharpe(bench.reindex(index).fillna(0.0), 252)
        roll_series.append(Series(bench_roll.to_numpy(), "Benchmark", "bench"))
        roll_values.append(bench_roll)
        roll_legend.append(("bench", "Benchmark"))
    finite = pd.concat(roll_values).replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        roll_domain = (-1.0, 1.0)
    else:
        r_lo, r_hi = min(float(finite.min()), 0.0), max(float(finite.max()), 0.0)
        margin = 0.08 * (r_hi - r_lo or 1.0)
        roll_domain = (r_lo - margin, r_hi + margin)
    roll_chart = line_chart(
        index,
        roll_series,
        linear_ticks(*roll_domain, fmt=lambda v: f"{v:.1f}".replace("-", "−"), target=4),
        roll_domain,
        height=180,
        zero_line=0.0,
        aria_label="Rolling 252-day Sharpe ratio",
    )

    monthly = monthly_returns(returns)
    yearly = annual_returns(returns)
    heat = heatmap(monthly, yearly)

    start, end = index[0], index[-1]
    assets = ", ".join(map(str, result.prices.columns))
    period = f"{_date(start)} – {_date(end)}"
    settings = dict(settings or {})
    settings.setdefault("Assets", assets)
    settings.setdefault("Starting capital", _money(result.initial_capital))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    cagr = strat_stats["cagr"]
    sharpe = strat_stats["sharpe"]
    mdd = strat_stats["max_drawdown"]
    summary = (
        f"{_pct(cagr, 1)} a year with a Sharpe of {sharpe:.2f} and a worst peak-to-trough "
        f"loss of {_pct(abs(mdd), 1)}"
    )
    if bench_stats is not None:
        summary += (
            f", against {_pct(bench_stats['cagr'], 1)}, {bench_stats['sharpe']:.2f} and "
            f"{_pct(abs(bench_stats['max_drawdown']), 1)} for the benchmark"
        )
    summary += "."

    def figure(heading: str, caption: str, body: str, legend_items=None, cls: str = "") -> str:
        legend_html = _legend(legend_items) if legend_items and len(legend_items) > 1 else ""
        return (
            f'<figure class="{cls}"><figcaption><h2>{escape(heading)}</h2>'
            f'<p class="note">{escape(caption)}</p></figcaption>{legend_html}{body}</figure>'
        )

    body = f"""
<header class="masthead">
  <div class="title-block">
    <h1>{escape(title)}</h1>
    <p class="params">{escape(subtitle)}</p>
  </div>
  <dl class="period">
    <div><dt>Period</dt><dd>{period}</dd></div>
    <div><dt>Bars</dt><dd>{len(index):,}</dd></div>
  </dl>
</header>
<p class="summary">{escape(summary)}</p>

<section class="overview">
  <div class="charts-col">
    {
        figure(
            f"Growth of {_money(result.initial_capital)}",
            "Net of costs, log scale.",
            growth_chart,
            legend,
            "growth",
        )
    }
    {figure("Drawdown", "Percentage below the previous equity peak.", dd_chart, dd_legend)}
  </div>
  <aside class="stats-panel" aria-label="Performance statistics">
    {_metrics_table(strat_stats, bench_stats, "Strategy")}
  </aside>
</section>

{figure("Rolling Sharpe", "Trailing 252 trading days, annualized.", roll_chart, roll_legend)}

<section>
  <figure class="months">
    <figcaption><h2>Monthly returns</h2>
    <p class="note">Percent, net of costs. Blue months gained, red months lost;
    the last column compounds the year.</p></figcaption>
    <div class="scroll" tabindex="0" role="region" aria-label="Monthly returns grid">{heat}</div>
  </figure>
</section>

<section class="pair">
  <div>
    <h2>Worst drawdowns</h2>
    <div class="scroll">{_drawdowns_table(returns)}</div>
  </div>
  <div>
    <h2>Setup</h2>
    {_settings_table(settings)}
  </div>
</section>

<footer>
  <p>{escape(note) + " " if note else ""}Simulated results. They include the modelled
  trading costs but not taxes, and past or simulated performance says little about
  future returns.</p>
  <p>quant-backtester {__version__} · generated {generated}</p>
</footer>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="quant-backtester {__version__}">
<title>{escape(title)} backtest</title>
<style>{_stylesheet()}</style>
</head>
<body>
<main>{body}</main>
</body>
</html>
"""


def write_tearsheet(result: BacktestResult, path: str | Path, **kwargs) -> Path:
    """Render and write the tearsheet; returns the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_tearsheet(result, **kwargs), encoding="utf-8")
    return path.resolve()
