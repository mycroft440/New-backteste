from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from .b3lab_champion import (
    CHAMPION_INDICATOR_NAME,
    CHAMPION_MANAGEMENT_STRATEGY_NAME,
    SOURCE_COST_BPS,
    SOURCE_SLIPPAGE_BPS,
    select_b3lab_champion,
)
from .engine import ManagedPortfolioResult, _execute_target, _prepare_frames
from .top1_momentum import is_monthly_rebalance_close


def _final_close_liquidation(
    *,
    date: pd.Timestamp,
    frames: Mapping[str, pd.DataFrame],
    date_index: Mapping[str, dict[pd.Timestamp, int]],
    holdings: dict[str, int],
    cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> tuple[float, list[dict[str, object]]]:
    orders: list[dict[str, object]] = []
    for ticker in sorted(holdings):
        qty = holdings[ticker]
        if qty <= 0:
            continue
        close = float(frames[ticker].iloc[date_index[ticker][date]]["close"])
        if not math.isfinite(close) or close <= 0:
            raise ValueError(f"{ticker} {date.date()}: invalid close for final liquidation")
        price = close * (1.0 - slippage_rate)
        gross = qty * price
        commission = gross * commission_rate
        cash += gross - commission
        holdings[ticker] = 0
        orders.append(
            {
                "date": date,
                "side": "SELL",
                "ticker": ticker,
                "shares": qty,
                "price": price,
                "gross_brl": gross,
                "commission_brl": commission,
                "cash_after_brl": cash,
                "reason": "final_close_liquidation",
            }
        )
    return cash, orders


def _annual_returns(curve: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    source = curve[["date", "equity"]].copy()
    source["year"] = pd.to_datetime(source["date"]).dt.year
    year_end = source.groupby("year", sort=True).tail(1).set_index("year")["equity"]
    previous = float(initial_cash)
    rows: list[dict[str, object]] = []
    for year, end_equity_raw in year_end.items():
        end_equity = float(end_equity_raw)
        rows.append(
            {
                "year": int(year),
                "end_equity": end_equity,
                "return_pct": (end_equity / previous - 1.0) * 100.0,
            }
        )
        previous = end_equity
    return pd.DataFrame(rows)


def simulate_b3lab_champion_portfolio(
    frames: Mapping[str, pd.DataFrame],
    *,
    initial_cash: float = 1000.0,
    commission_bps: float = SOURCE_COST_BPS,
    slippage_bps: float = SOURCE_SLIPPAGE_BPS,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> ManagedPortfolioResult:
    """Simulate the strongest persisted B3 Strategy Lab portfolio configuration.

    The monthly close computes the champion cross-sectional indicator and ranks
    all eligible tickers. The Top-1 target is executed only at the next common
    session's open. There are no daily MACD/RSI/etc. exits between rebalances;
    that is source-faithful for the winning portfolio sweep. The last position is
    liquidated at the final close, also matching the source research engine.
    """
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("cost parameters must be non-negative")

    prepared, all_dates = _prepare_frames(frames)
    start_ts = pd.Timestamp(start) if start is not None else all_dates[0]
    end_ts = pd.Timestamp(end) if end is not None else all_dates[-1]
    dates = [date for date in all_dates if start_ts <= date <= end_ts]
    if len(dates) < 2:
        raise ValueError("requested window has fewer than two common sessions")

    date_index = {
        ticker: {pd.Timestamp(value): index for index, value in enumerate(frame["date"])}
        for ticker, frame in prepared.items()
    }
    cash = float(initial_cash)
    holdings = {ticker: 0 for ticker in prepared}
    pending: tuple[str | None, str, bool] | None = None
    decision_frames: list[pd.DataFrame] = []
    rebalance_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    commission_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0

    def record_decision(decision_date: pd.Timestamp, execution_date: pd.Timestamp) -> str | None:
        selection = select_b3lab_champion(prepared, decision_date)
        candidates = selection.candidates.copy()
        candidates.insert(1, "execution_date", execution_date)
        decision_frames.append(candidates)
        eligible = candidates.loc[candidates["eligible_for_ranking"], "ticker"].tolist()
        chosen = selection.selected_ticker
        selected_score = None
        if chosen is not None:
            selected_score = float(candidates.loc[candidates["selected"], "score"].iloc[0])
        rebalance_rows.append(
            {
                "decision_date": decision_date,
                "execution_date": execution_date,
                "eligible_tickers": ";".join(eligible),
                "selected_ticker": chosen or "",
                "selected_score": selected_score,
            }
        )
        return chosen

    first_global = all_dates.index(dates[0])
    if first_global > 0:
        prior = all_dates[first_global - 1]
        if is_monthly_rebalance_close(prior, dates[0]):
            pending = (record_decision(prior, dates[0]), "monthly_rebalance", True)

    for offset, date in enumerate(dates):
        if pending is not None:
            target, reason, force = pending
            cash, orders = _execute_target(
                date=date,
                target=target,
                reason=reason,
                frames=prepared,
                date_index=date_index,
                holdings=holdings,
                cash=cash,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                force_rebalance=force,
            )
            order_rows.extend(orders)
            pending = None

        last_day = offset + 1 == len(dates)
        if last_day:
            cash, liquidation_orders = _final_close_liquidation(
                date=date,
                frames=prepared,
                date_index=date_index,
                holdings=holdings,
                cash=cash,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
            )
            order_rows.extend(liquidation_orders)

        held = [ticker for ticker, qty in holdings.items() if qty > 0]
        if len(held) > 1:
            raise AssertionError("champion Top-1 manager cannot hold more than one ticker")
        held_ticker = held[0] if held else None
        shares = holdings[held_ticker] if held_ticker else 0
        invested = 0.0
        if held_ticker is not None:
            invested = shares * float(prepared[held_ticker].iloc[date_index[held_ticker][date]]["close"])
        equity_rows.append(
            {
                "date": date,
                "cash": cash,
                "held_ticker": held_ticker or "",
                "shares": shares,
                "invested_value": invested,
                "equity": cash + invested,
            }
        )

        if last_day:
            continue
        next_date = dates[offset + 1]
        if is_monthly_rebalance_close(date, next_date):
            pending = (record_decision(date, next_date), "monthly_rebalance", True)

    curve = pd.DataFrame(equity_rows)
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    rebalance_summary = pd.DataFrame(rebalance_rows)
    orders = pd.DataFrame(
        order_rows,
        columns=[
            "date",
            "side",
            "ticker",
            "shares",
            "price",
            "gross_brl",
            "commission_brl",
            "cash_after_brl",
            "reason",
        ],
    )
    annual_returns = _annual_returns(curve, initial_cash)
    final_equity = float(curve.iloc[-1]["equity"])
    peak = float(initial_cash)
    max_drawdown = 0.0
    for value in curve["equity"].astype(float):
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)

    return ManagedPortfolioResult(
        CHAMPION_INDICATOR_NAME,
        CHAMPION_MANAGEMENT_STRATEGY_NAME,
        float(initial_cash),
        final_equity,
        final_equity - initial_cash,
        (final_equity / initial_cash - 1.0) * 100.0,
        max_drawdown * 100.0,
        decisions,
        rebalance_summary,
        orders,
        curve,
        annual_returns,
    )
