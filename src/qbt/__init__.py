"""quant-backtester: a vectorized backtesting engine for systematic strategies."""

from qbt.costs import CostModel
from qbt.data import generate_prices, load_csv, load_yfinance
from qbt.engine import BacktestResult, buy_and_hold, run_backtest, simulate
from qbt.metrics import compute_metrics

__version__ = "0.1.0"

__all__ = [
    "BacktestResult",
    "CostModel",
    "__version__",
    "buy_and_hold",
    "compute_metrics",
    "generate_prices",
    "load_csv",
    "load_yfinance",
    "run_backtest",
    "simulate",
]
