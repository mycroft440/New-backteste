from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pandas as pd

from b3_backtest.strategies.catalog import run_strategy
from b3_backtest.strategies.core import validate_ohlcv
from .top1_momentum import MANAGEMENT_STRATEGY_NAME, is_monthly_rebalance_close, select_top1_momentum


@dataclass(frozen=True)
class ManagedPortfolioResult:
    signal_strategy: str
    management_strategy: str
    initial_cash: float
    final_equity: float
    final_profit: float
    final_return_pct: float
    max_drawdown_pct: float
    decisions: pd.DataFrame
    rebalance_summary: pd.DataFrame
    orders: pd.DataFrame
    equity_curve: pd.DataFrame
    annual_returns: pd.DataFrame


def _prepare_frames(frames: Mapping[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    if not frames:
        raise ValueError("portfolio requires at least one ticker")
    prepared: dict[str, pd.DataFrame] = {}
    common: set[pd.Timestamp] | None = None
    for ticker in sorted(frames):
        frame = frames[ticker].copy().reset_index(drop=True)
        validate_ohlcv(frame)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
            raise ValueError(f"{ticker}: dates must be unique and sorted")
        prepared[ticker] = frame
        dates = set(pd.Timestamp(v) for v in frame["date"])
        common = dates if common is None else common.intersection(dates)
    dates = sorted(common or set())
    if len(dates) < 2:
        raise ValueError("portfolio requires at least two common sessions")
    return prepared, dates


def _execute_target(*, date: pd.Timestamp, target: str | None, reason: str,
                    frames: Mapping[str, pd.DataFrame], date_index: Mapping[str, dict[pd.Timestamp, int]],
                    holdings: dict[str, int], cash: float, commission_rate: float,
                    slippage_rate: float, force_rebalance: bool) -> tuple[float, list[dict[str, object]]]:
    orders: list[dict[str, object]] = []
    held = next((ticker for ticker, qty in holdings.items() if qty > 0), None)

    def open_price(ticker: str) -> float:
        value = float(frames[ticker].iloc[date_index[ticker][date]]["open"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{ticker} {date.date()}: invalid open")
        return value

    def sell(ticker: str) -> None:
        nonlocal cash
        qty = holdings[ticker]
        if qty <= 0:
            return
        price = open_price(ticker) * (1.0 - slippage_rate)
        gross = qty * price
        commission = gross * commission_rate
        cash += gross - commission
        holdings[ticker] = 0
        orders.append({"date": date, "side": "SELL", "ticker": ticker, "shares": qty,
                       "price": price, "gross_brl": gross, "commission_brl": commission,
                       "cash_after_brl": cash, "reason": reason})

    if held is not None and held != target:
        sell(held)
        held = None
    if target is None:
        if held is not None:
            sell(held)
        return cash, orders
    if target not in holdings:
        raise ValueError(f"unknown target ticker: {target}")
    if held == target and not force_rebalance:
        return cash, orders

    buy_price = open_price(target) * (1.0 + slippage_rate)
    unit_need = buy_price * (1.0 + commission_rate)
    qty = int(cash // unit_need)
    if qty > 0:
        gross = qty * buy_price
        commission = gross * commission_rate
        cash -= gross + commission
        holdings[target] += qty
        orders.append({"date": date, "side": "BUY", "ticker": target, "shares": qty,
                       "price": buy_price, "gross_brl": gross, "commission_brl": commission,
                       "cash_after_brl": cash, "reason": reason})
    return cash, orders


def simulate_managed_portfolio(frames: Mapping[str, pd.DataFrame], *, signal_strategy: str,
                               initial_cash: float = 1000.0, commission_bps: float = 0.0,
                               slippage_bps: float = 0.0, start: str | pd.Timestamp | None = None,
                               end: str | pd.Timestamp | None = None,
                               signal_states: Mapping[str, list[int]] | None = None) -> ManagedPortfolioResult:
    """One shared-capital portfolio: monthly Top-1 ranking plus daily signal exits."""
    if not math.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if commission_bps < 0 or slippage_bps < 0:
        raise ValueError("cost parameters must be non-negative")

    prepared, all_dates = _prepare_frames(frames)
    start_ts = pd.Timestamp(start) if start is not None else all_dates[0]
    end_ts = pd.Timestamp(end) if end is not None else all_dates[-1]
    dates = [d for d in all_dates if start_ts <= d <= end_ts]
    if len(dates) < 2:
        raise ValueError("requested window has fewer than two common sessions")

    states: dict[str, list[int]] = {}
    for ticker, frame in prepared.items():
        values = run_strategy(frame, signal_strategy) if signal_states is None else list(signal_states[ticker])
        if len(values) != len(frame) or any(v not in (0, 1) for v in values):
            raise ValueError(f"{ticker}: invalid signal states")
        states[ticker] = values
    date_index = {ticker: {pd.Timestamp(v): i for i, v in enumerate(frame["date"])}
                  for ticker, frame in prepared.items()}

    cash = float(initial_cash)
    holdings = {ticker: 0 for ticker in prepared}
    designated: str | None = None
    active_target: str | None = None
    pending: tuple[str | None, str, bool] | None = None
    decision_frames: list[pd.DataFrame] = []
    rebalance_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    commission_rate = commission_bps / 10_000.0
    slippage_rate = slippage_bps / 10_000.0

    def record_decision(decision_date: pd.Timestamp, execution_date: pd.Timestamp) -> str | None:
        selection = select_top1_momentum(prepared, states, decision_date)
        candidates = selection.candidates.copy()
        candidates.insert(1, "execution_date", execution_date)
        decision_frames.append(candidates)
        uptrend = candidates.loc[candidates["indicator_uptrend"], "ticker"].tolist()
        eligible = candidates.loc[candidates["eligible_for_ranking"], "ticker"].tolist()
        chosen = selection.selected_ticker
        selected_momentum = None
        if chosen is not None:
            selected_momentum = float(candidates.loc[candidates["selected"], "momentum_126_pct"].iloc[0])
        rebalance_rows.append({"decision_date": decision_date, "execution_date": execution_date,
                               "uptrend_tickers": ";".join(uptrend), "eligible_tickers": ";".join(eligible),
                               "selected_ticker": chosen or "", "selected_momentum_126_pct": selected_momentum})
        return chosen

    first_global = all_dates.index(dates[0])
    if first_global > 0:
        prior = all_dates[first_global - 1]
        if is_monthly_rebalance_close(prior, dates[0]):
            designated = record_decision(prior, dates[0])
            pending = (designated, "monthly_rebalance", True)

    for offset, date in enumerate(dates):
        if pending is not None:
            target, reason, force = pending
            cash, orders = _execute_target(date=date, target=target, reason=reason, frames=prepared,
                                           date_index=date_index, holdings=holdings, cash=cash,
                                           commission_rate=commission_rate, slippage_rate=slippage_rate,
                                           force_rebalance=force)
            order_rows.extend(orders)
            active_target = target
            pending = None

        held = [ticker for ticker, qty in holdings.items() if qty > 0]
        if len(held) > 1:
            raise AssertionError("Top-1 manager cannot hold more than one ticker")
        held_ticker = held[0] if held else None
        shares = holdings[held_ticker] if held_ticker else 0
        invested = 0.0
        if held_ticker is not None:
            invested = shares * float(prepared[held_ticker].iloc[date_index[held_ticker][date]]["close"])
        equity_rows.append({"date": date, "cash": cash, "held_ticker": held_ticker or "",
                            "shares": shares, "invested_value": invested, "equity": cash + invested})

        if offset + 1 == len(dates):
            continue
        next_date = dates[offset + 1]
        if is_monthly_rebalance_close(date, next_date):
            designated = record_decision(date, next_date)
            pending = (designated, "monthly_rebalance", True)
        else:
            desired: str | None = None
            if designated is not None:
                idx = date_index[designated][date]
                if states[designated][idx] == 1:
                    desired = designated
            if desired != active_target:
                pending = (desired, "indicator_reentry" if desired else "indicator_exit", False)

    curve = pd.DataFrame(equity_rows)
    decisions = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    rebalance_summary = pd.DataFrame(rebalance_rows)
    orders = pd.DataFrame(order_rows, columns=["date", "side", "ticker", "shares", "price", "gross_brl",
                                                    "commission_brl", "cash_after_brl", "reason"])
    annual_source = curve[["date", "equity"]].copy()
    annual_source["year"] = pd.to_datetime(annual_source["date"]).dt.year
    year_end = annual_source.groupby("year", sort=True).tail(1).set_index("year")["equity"]
    previous = float(initial_cash)
    annual_rows = []
    for year, end_equity_raw in year_end.items():
        end_equity = float(end_equity_raw)
        annual_rows.append({"year": int(year), "end_equity": end_equity,
                            "return_pct": (end_equity / previous - 1.0) * 100.0})
        previous = end_equity
    annual_returns = pd.DataFrame(annual_rows)

    final_equity = float(curve.iloc[-1]["equity"])
    peak = float(initial_cash)
    max_drawdown = 0.0
    for value in curve["equity"].astype(float):
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)

    return ManagedPortfolioResult(signal_strategy, MANAGEMENT_STRATEGY_NAME, float(initial_cash), final_equity,
                                  final_equity - initial_cash, (final_equity / initial_cash - 1.0) * 100.0,
                                  max_drawdown * 100.0, decisions, rebalance_summary, orders, curve, annual_returns)
