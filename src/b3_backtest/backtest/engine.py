from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from b3_backtest.strategies.core import validate_ohlcv


@dataclass(frozen=True)
class BacktestResult:
    initial_cash: float
    final_equity: float
    final_profit: float
    final_return_pct: float
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    annual_returns: pd.DataFrame
    max_trade_gain_pct: float | None
    max_trade_loss_pct: float | None


def _dates(frame: pd.DataFrame) -> pd.Series:
    if "date" not in frame.columns:
        raise ValueError("backtest requires a date column")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("date must be unique and sorted ascending")
    return dates


def simulate_from_positions(
    frame: pd.DataFrame,
    positions: list[int],
    *,
    initial_cash: float = 1000.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_target: int = 0,
) -> BacktestResult:
    """Execute close-generated 0/1 signals on the next bar open.

    This engine is deliberately single-asset. It contains no portfolio allocation,
    ranking, rebalancing or capital sharing between symbols. Buys use the maximum
    whole number of shares affordable with the available cash; any remainder stays
    as cash. A still-open position is marked to the final close, not force-sold.

    ``initial_target`` is the strategy state known from the close immediately before
    the first bar in ``frame``. It allows indicators to warm up on earlier history
    while capital still starts exactly on the requested backtest date. No future
    information is used.
    """
    validate_ohlcv(frame)
    dates = _dates(frame)
    if len(positions) != len(frame):
        raise ValueError("positions length must equal frame length")
    if any(value not in (0, 1) for value in positions):
        raise ValueError("positions must contain only 0 or 1")
    if initial_target not in (0, 1):
        raise ValueError("initial_target must be 0 or 1")
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("cost parameters must be non-negative")

    date_values = dates.tolist()
    open_values = frame["open"].to_numpy(dtype=float, copy=False)
    close_values = frame["close"].to_numpy(dtype=float, copy=False)

    cash = float(initial_cash)
    shares = 0
    entry_price: float | None = None
    entry_date: pd.Timestamp | None = None
    entry_cost_total = 0.0
    trade_rows: list[tuple[object, ...]] = []
    equity_rows: list[tuple[object, ...]] = []

    commission_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0

    for index, (date, open_price, close_price) in enumerate(
        zip(date_values, open_values, close_values, strict=True)
    ):
        open_price = float(open_price)
        close_price = float(close_price)
        target = initial_target if index == 0 else positions[index - 1]

        if target == 1 and shares == 0:
            buy_price = open_price * (1.0 + slippage_rate)
            unit_cash_need = buy_price * (1.0 + commission_rate)
            quantity = int(cash // unit_cash_need)
            if quantity > 0:
                gross = quantity * buy_price
                commission = gross * commission_rate
                total = gross + commission
                cash -= total
                shares = quantity
                entry_price = buy_price
                entry_date = date
                entry_cost_total = total

        elif target == 0 and shares > 0:
            sell_price = open_price * (1.0 - slippage_rate)
            gross = shares * sell_price
            commission = gross * commission_rate
            proceeds = gross - commission
            cash += proceeds
            pnl = proceeds - entry_cost_total
            return_pct = pnl / entry_cost_total * 100.0
            trade_rows.append(
                (
                    entry_date,
                    date,
                    shares,
                    entry_price,
                    sell_price,
                    pnl,
                    return_pct,
                )
            )
            shares = 0
            entry_price = None
            entry_date = None
            entry_cost_total = 0.0

        equity = cash + shares * close_price
        equity_rows.append((date, cash, shares, close_price, equity))

    equity_curve = pd.DataFrame(
        equity_rows,
        columns=["date", "cash", "shares", "close", "equity"],
    )
    final_equity = float(equity_curve.iloc[-1]["equity"])
    final_profit = final_equity - initial_cash
    final_return_pct = final_profit / initial_cash * 100.0

    annual = equity_curve[["date", "equity"]].copy()
    annual["year"] = annual["date"].dt.year
    year_end = annual.groupby("year", sort=True).tail(1).set_index("year")["equity"]
    annual_rows: list[tuple[object, ...]] = []
    previous_equity = float(initial_cash)
    for year, ending_equity_raw in year_end.items():
        ending_equity = float(ending_equity_raw)
        return_pct = (ending_equity / previous_equity - 1.0) * 100.0
        annual_rows.append((int(year), ending_equity, return_pct))
        previous_equity = ending_equity
    annual_returns = pd.DataFrame(
        annual_rows,
        columns=["year", "end_equity", "return_pct"],
    )

    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "entry_date",
            "exit_date",
            "shares",
            "entry_price",
            "exit_price",
            "pnl_brl",
            "return_pct",
        ],
    )
    max_gain = float(trades["return_pct"].max()) if not trades.empty else None
    max_loss = float(trades["return_pct"].min()) if not trades.empty else None

    return BacktestResult(
        initial_cash=float(initial_cash),
        final_equity=final_equity,
        final_profit=final_profit,
        final_return_pct=final_return_pct,
        trades=trades,
        equity_curve=equity_curve,
        annual_returns=annual_returns,
        max_trade_gain_pct=max_gain,
        max_trade_loss_pct=max_loss,
    )
