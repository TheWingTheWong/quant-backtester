"""Command-line interface: ``qbt run | grid | walkforward | strategies | data``.

Strategy parameters are exposed as flags generated from the strategy's
dataclass fields, so ``qbt run --strategy sma --help`` documents exactly the
parameters that strategy accepts and ``--fast 20 --slow 100`` just works.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from qbt import __version__
from qbt.costs import CostModel
from qbt.data import TRADING_DAYS, generate_prices, load_csv, load_yfinance
from qbt.engine import BacktestResult, run_backtest
from qbt.metrics import METRIC_FORMATS, compute_metrics, format_metric
from qbt.strategies import REGISTRY, Strategy, get_strategy

_TYPES: dict[str, type] = {"int": int, "float": float, "bool": bool, "str": str}

TABLE_METRICS = (
    "total_return",
    "cagr",
    "ann_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "max_dd_duration",
    "hit_rate",
    "turnover",
    "avg_exposure",
)


# --------------------------------------------------------------------------- helpers


def _field_type(field: dataclasses.Field) -> type:
    annotation = field.type if isinstance(field.type, str) else field.type.__name__
    return _TYPES.get(annotation, str)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _add_strategy_params(parser: argparse.ArgumentParser, cls: type[Strategy]) -> None:
    group = parser.add_argument_group(f"{cls.key} parameters ({cls.title})")
    for field in dataclasses.fields(cls):
        kind = _field_type(field)
        flags = [_flag(field.name)]
        if "_" in field.name and kind is not bool:
            flags.append("--" + field.name)
        if kind is bool:
            group.add_argument(
                *flags,
                dest=f"p_{field.name}",
                action=argparse.BooleanOptionalAction,
                default=None,
                help=f"default: {field.default}",
            )
        else:
            group.add_argument(
                *flags,
                dest=f"p_{field.name}",
                type=kind,
                default=None,
                metavar=kind.__name__.upper(),
                help=f"default: {field.default}",
            )


def _strategy_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key[2:]: value
        for key, value in vars(args).items()
        if key.startswith("p_") and value is not None
    }


def _peek_strategy(argv: Sequence[str]) -> str | None:
    for i, token in enumerate(argv):
        if token in ("--strategy", "-s") and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--strategy="):
            return token.split("=", 1)[1]
    return None


def _parse_value(text: str, kind: type) -> Any:
    if kind is bool:
        lowered = text.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"not a boolean: {text!r}")
    return kind(text)


def parse_grid(specs: Sequence[str], cls: type[Strategy]) -> dict[str, list[Any]]:
    """Parse ``name=v1,v2,v3`` or ``name=start:stop:step`` (inclusive) specs."""
    fields = {f.name: _field_type(f) for f in dataclasses.fields(cls)}
    grid: dict[str, list[Any]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"grid spec must look like name=values, got {spec!r}")
        name, values = spec.split("=", 1)
        name = name.strip().replace("-", "_")
        if name not in fields:
            raise ValueError(f"{cls.key} has no parameter {name!r}; choose from {list(fields)}")
        kind = fields[name]
        if ":" in values:
            start, stop, step = (float(x) for x in values.split(":"))
            if step <= 0:
                raise ValueError("range step must be positive")
            items, value = [], start
            while value <= stop + 1e-9:
                items.append(kind(round(value, 10)))
                value += step
        else:
            items = [_parse_value(v, kind) for v in values.split(",") if v.strip()]
        if not items:
            raise ValueError(f"no values for {name!r}")
        grid[name] = items
    return grid


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Plain-text table: first column left-aligned, the rest right-aligned."""
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]

    def line(cells: Sequence[str]) -> str:
        parts = [
            str(c).ljust(w) if i == 0 else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        ]
        return "  ".join(parts).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), rule, *(line(r) for r in rows)])


def _load_prices(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.csv:
        prices = load_csv(args.csv)
        source = f"CSV {args.csv}"
    elif args.yf:
        tickers = [t.strip().upper() for t in args.yf.split(",") if t.strip()]
        prices = load_yfinance(tickers, start=args.start or "2010-01-01", end=args.end)
        source = f"Yahoo Finance {', '.join(tickers)}"
    else:
        prices = generate_prices(
            n_assets=args.assets,
            n_days=int(args.years * TRADING_DAYS),
            seed=args.seed,
            start=args.start or "2015-01-02",
        )
        source = f"Synthetic regime-switching GBM, seed {args.seed}"
    if args.start or args.end:
        prices = prices.loc[args.start : args.end]
    if len(prices) < 2:
        raise ValueError("not enough price data after filtering")
    return prices, source


def _costs(args: argparse.Namespace) -> CostModel:
    return CostModel(
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        borrow_bps=args.borrow_bps,
        cash_rate=args.cash_rate,
    )


def _rebalance(args: argparse.Namespace, cls: type[Strategy]) -> str | int:
    value = args.rebalance or cls.default_rebalance
    return int(value) if str(value).isdigit() else value


def _describe_params(strategy: Strategy) -> str:
    return " ".join(f"{k}={v}" for k, v in strategy.params.items())


# --------------------------------------------------------------------------- commands


def cmd_strategies(args: argparse.Namespace) -> int:
    for key, cls in sorted(REGISTRY.items()):
        params = ", ".join(f"{_flag(f.name)} {f.default}" for f in dataclasses.fields(cls))
        print(f"{key:<11} {cls.description}")
        print(f"{'':<11} {params}  (rebalance {cls.default_rebalance})")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    prices, source = _load_prices(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out, float_format="%.6f")
    print(f"Wrote {len(prices):,} rows x {prices.shape[1]} assets ({source}) to {out}")
    return 0


def _metrics_rows(result: BacktestResult) -> list[list[str]]:
    strat = result.metrics()
    bench = compute_metrics(result.benchmark) if result.benchmark is not None else {}
    rows = []
    for key in TABLE_METRICS:
        label = METRIC_FORMATS[key][0]
        row = [label, format_metric(key, strat[key])]
        if result.benchmark is not None:
            row.append(format_metric(key, bench[key]) if key in bench else "-")
        rows.append(row)
    return rows


def cmd_run(args: argparse.Namespace) -> int:
    cls = get_strategy(args.strategy)
    strategy = cls(**_strategy_params(args))
    prices, source = _load_prices(args)
    rebalance = _rebalance(args, cls)
    costs = _costs(args)
    result = run_backtest(
        strategy,
        prices,
        costs=costs,
        rebalance=rebalance,
        delay=args.delay,
        vol_target=args.overlay_vol,
        max_leverage=args.overlay_leverage,
        initial_capital=args.capital,
    )

    if args.json:
        payload = {
            "strategy": cls.key,
            "params": strategy.params,
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "metrics": result.metrics(),
        }
        if result.benchmark is not None:
            payload["benchmark"] = compute_metrics(result.benchmark)
        print(json.dumps(payload, indent=2, default=float))
    else:
        first, last = prices.index[0], prices.index[-1]
        print(f"{cls.title}  {_describe_params(strategy)}")
        print(
            f"{source}: {prices.shape[1]} assets, {len(prices):,} bars "
            f"({first:%Y-%m-%d} to {last:%Y-%m-%d})"
        )
        print(
            f"rebalance {rebalance}, delay {args.delay}, costs "
            f"{costs.commission_bps:g}+{costs.slippage_bps:g} bps per trade\n"
        )
        headers = ["Metric", "Strategy"]
        if result.benchmark is not None:
            headers.append("Benchmark")
        print(format_table(headers, _metrics_rows(result)))

    if not args.no_report:
        from qbt.report import write_tearsheet

        out = args.out or f"reports/{cls.key}.html"
        cost_text = f"{costs.commission_bps:g} bps commission, {costs.slippage_bps:g} bps slippage"
        if costs.borrow_bps:
            cost_text += f", {costs.borrow_bps:g} bps/yr borrow"
        settings = {
            "Data": source,
            "Rebalance": str(rebalance),
            "Execution delay": f"{args.delay} bar(s) after the signal close",
            "Costs": cost_text,
        }
        if args.overlay_vol:
            settings["Vol target overlay"] = f"{args.overlay_vol:.0%}"
        if args.overlay_leverage:
            settings["Max gross leverage"] = f"{args.overlay_leverage:g}x"
        subtitle = " · ".join(f"{k.replace('_', ' ')} {v}" for k, v in strategy.params.items())
        path = write_tearsheet(result, out, title=cls.title, subtitle=subtitle, settings=settings)
        print(f"\nTearsheet written to {path}", file=sys.stderr if args.json else sys.stdout)
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    from qbt.optimize import grid_search

    cls = get_strategy(args.strategy)
    grid = parse_grid(args.grid, cls)
    prices, source = _load_prices(args)
    table = grid_search(
        cls,
        prices,
        grid,
        metric=args.metric,
        costs=_costs(args),
        rebalance=_rebalance(args, cls),
        delay=args.delay,
    )
    n_combos = len(table)
    print(f"{cls.title}: {n_combos} valid combinations on {source}, ranked by {args.metric}\n")
    shown = table.head(args.top)
    cols = [*grid, "cagr", "sharpe", "max_drawdown", "turnover"]
    if args.metric not in cols:
        cols.append(args.metric)
    # Read column-wise so integer parameters are not upcast to floats.
    columns = {c: shown[c].tolist() for c in cols}
    rows = [
        [str(columns[c][i]) if c in grid else format_metric(c, columns[c][i]) for c in cols]
        for i in range(len(shown))
    ]
    headers = [METRIC_FORMATS.get(c, (c,))[0] if c not in grid else c for c in cols]
    print(format_table(headers, rows))
    if args.save:
        table.to_csv(args.save, index=False)
        print(f"\nFull results written to {args.save}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    from qbt.optimize import walk_forward

    cls = get_strategy(args.strategy)
    grid = parse_grid(args.grid, cls)
    prices, source = _load_prices(args)
    result = walk_forward(
        cls,
        prices,
        grid,
        train_bars=args.train,
        test_bars=args.test,
        metric=args.metric,
        anchored=args.anchored,
        costs=_costs(args),
        rebalance=_rebalance(args, cls),
        delay=args.delay,
    )
    kind = "anchored" if args.anchored else "rolling"
    print(
        f"{cls.title}: {kind} walk-forward on {source}, "
        f"train {args.train} / test {args.test} bars, optimizing {args.metric}\n"
    )
    folds = result.folds_table()
    rows = []
    for _, row in folds.iterrows():
        cells = [row["test"], *(str(row[k]) for k in grid)]
        cells += [
            format_metric(args.metric, row[f"train_{args.metric}"]),
            format_metric(args.metric, row[f"test_{args.metric}"]),
        ]
        rows.append(cells)
    print(format_table(["Test window", *grid, "In-sample", "Out-of-sample"], rows))

    stats = result.metrics()
    print("\nStitched out-of-sample performance")
    print(
        format_table(
            ["Metric", "Value"],
            [
                [METRIC_FORMATS[k][0], format_metric(k, stats[k])]
                for k in TABLE_METRICS
                if k in stats
            ],
        )
    )
    return 0


# --------------------------------------------------------------------------- parser


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("data")
    src = g.add_mutually_exclusive_group()
    src.add_argument("--csv", metavar="PATH", help="wide or OHLCV CSV file")
    src.add_argument("--yf", metavar="TICKERS", help="comma-separated tickers via yfinance")
    g.add_argument("--assets", type=int, default=4, help="synthetic: number of assets (4)")
    g.add_argument("--years", type=float, default=10, help="synthetic: years of data (10)")
    g.add_argument("--seed", type=int, default=42, help="synthetic: RNG seed (42)")
    g.add_argument("--start", help="first date to use (YYYY-MM-DD)")
    g.add_argument("--end", help="last date to use (YYYY-MM-DD)")


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("execution and costs")
    g.add_argument(
        "--rebalance",
        help="D, W, M, Q or an integer number of bars (default: per strategy)",
    )
    g.add_argument("--delay", type=int, default=0, help="bars between signal and fill (0)")
    g.add_argument("--commission-bps", type=float, default=1.0, help="per trade (1.0)")
    g.add_argument("--slippage-bps", type=float, default=2.0, help="per trade (2.0)")
    g.add_argument("--borrow-bps", type=float, default=0.0, help="annual, on shorts (0)")
    g.add_argument("--cash-rate", type=float, default=0.0, help="annual rate on cash (0)")


def build_parser(argv: Sequence[str] = ()) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qbt",
        description="Vectorized backtests for systematic trading strategies.",
    )
    parser.add_argument("--version", action="version", version=f"qbt {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    chosen = _peek_strategy(argv)
    chosen_cls = REGISTRY.get(chosen) if chosen else None
    strategy_help = "one of: " + ", ".join(sorted(REGISTRY))

    run = sub.add_parser("run", help="backtest one strategy and write a tearsheet")
    run.add_argument(
        "-s", "--strategy", required=True, choices=sorted(REGISTRY), help=strategy_help
    )
    _add_data_args(run)
    _add_execution_args(run)
    g = run.add_argument_group("sizing overlays")
    g.add_argument(
        "--overlay-vol",
        type=float,
        metavar="VOL",
        help="scale the book to this annualized vol, e.g. 0.10",
    )
    g.add_argument(
        "--overlay-leverage",
        type=float,
        metavar="X",
        help="cap gross exposure, e.g. 1.5",
    )
    g = run.add_argument_group("output")
    g.add_argument("--capital", type=float, default=1_000_000.0, help="starting equity")
    g.add_argument("-o", "--out", help="tearsheet path (default: reports/<strategy>.html)")
    g.add_argument("--no-report", action="store_true", help="skip the HTML tearsheet")
    g.add_argument("--json", action="store_true", help="print metrics as JSON")
    if chosen_cls is not None:
        _add_strategy_params(run, chosen_cls)
    run.set_defaults(func=cmd_run)

    for name, func, text in (
        ("grid", cmd_grid, "rank every combination of a parameter grid"),
        ("walkforward", cmd_walkforward, "walk-forward optimization, out-of-sample"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument(
            "-s", "--strategy", required=True, choices=sorted(REGISTRY), help=strategy_help
        )
        p.add_argument(
            "-g",
            "--grid",
            action="append",
            required=True,
            metavar="NAME=VALUES",
            help="e.g. fast=10,20,50 or slow=50:200:50 (repeatable)",
        )
        p.add_argument("--metric", default="sharpe", help="metric to optimize (sharpe)")
        _add_data_args(p)
        _add_execution_args(p)
        if name == "grid":
            p.add_argument("--top", type=int, default=10, help="rows to print (10)")
            p.add_argument("--save", metavar="CSV", help="write the full ranking to CSV")
        else:
            p.add_argument("--train", type=int, default=756, help="in-sample bars (756)")
            p.add_argument("--test", type=int, default=252, help="out-of-sample bars (252)")
            p.add_argument("--anchored", action="store_true", help="expanding train window")
        p.set_defaults(func=func)

    lst = sub.add_parser("strategies", help="list strategies and their parameters")
    lst.set_defaults(func=cmd_strategies)

    data = sub.add_parser("data", help="export price data (e.g. synthetic) to CSV")
    _add_data_args(data)
    data.add_argument("-o", "--out", default="data/prices.csv", help="CSV path")
    data.set_defaults(func=cmd_data)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, KeyError, ImportError, FileNotFoundError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"qbt: error: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
