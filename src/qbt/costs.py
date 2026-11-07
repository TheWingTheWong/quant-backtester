"""Transaction cost and financing model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qbt.data import TRADING_DAYS


@dataclass(frozen=True)
class CostModel:
    """Linear trading costs plus financing.

    All rates are in basis points so they read like a broker's fee schedule.

    Attributes:
        commission_bps: commission charged per unit of traded notional.
        slippage_bps: expected execution shortfall per unit of traded notional
            (half-spread plus market impact for a small order).
        borrow_bps: annualized fee charged on short exposure.
        cash_rate: annualized rate earned on idle cash and paid on borrowed
            cash (negative cash weight means leverage).
    """

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    borrow_bps: float = 0.0
    cash_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("commission_bps", "slippage_bps", "borrow_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def trade_cost_rate(self) -> float:
        """Total cost per unit of traded notional, as a fraction."""
        return (self.commission_bps + self.slippage_bps) / 1e4

    @property
    def daily_cash_rate(self) -> float:
        return (1.0 + self.cash_rate) ** (1.0 / TRADING_DAYS) - 1.0

    def trading_cost(self, turnover: pd.Series) -> pd.Series:
        """Cost as a fraction of equity given one-way turnover per bar."""
        return turnover * self.trade_cost_rate

    def borrow_cost(self, short_exposure: pd.Series) -> pd.Series:
        """Daily borrow fee given short exposure (positive number)."""
        return short_exposure * (self.borrow_bps / 1e4) / TRADING_DAYS

    @classmethod
    def zero(cls) -> CostModel:
        """A frictionless model, handy for tests and theoretical studies."""
        return cls(commission_bps=0.0, slippage_bps=0.0, borrow_bps=0.0, cash_rate=0.0)
