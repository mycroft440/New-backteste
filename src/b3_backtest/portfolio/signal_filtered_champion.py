from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

from b3_backtest.indicators.b3lab_champion import evaluate_universe
from b3_backtest.strategies.catalog import run_strategy
from .b3lab_champion import (
    CHAMPION_MANAGEMENT_STRATEGY_NAME,
    SOURCE_COST_BPS,
    SOURCE_SLIPPAGE_BPS,
    _rank,
)
from .champion_engine import _annual_returns, _final_close_liquidation
from .engine import ManagedPortfolioResult, _execute_target, _prepare_frames
from .top1_momentum import is_monthly_rebalance_close


def simulate_signal_filtered_champion(
    frames: Mapping[str, pd.DataFrame],
    *,
    signal_strategy: str,
    initial_cash: float = 1000.0,
    commission_bps: float = SOURCE_COST_BPS,
    slippage_bps: float = SOURCE_SLIPPAGE_BPS,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    signal_states: Mapping[str, list[int]] | None = None,
    management_cache: dict[pd.Timestamp, pd.DataFrame] | None = None,
) -> ManagedPortfolioResult:
    """Run the champion manager only inside the signal strategy uptrend set.

    At each monthly decision close:
    1. The BUY/SELL strategy marks every ticker as 0/1.
    2. Only state=1 tickers are passed to the portfolio manager.
    3. The champion ROC/volatility/SMA rules rank that subset and select Top-1.
    4. The target is executed at the next common session open.

    The selected asset is held until the next scheduled rebalance. The signal is
    intentionally sampled only on rebalance dates; there are no intra-month exits.

    ``management_cache`` may be shared across runs that use the same market frames.
    It stores only the signal-independent champion indicator snapshot for each
    decision date. Each signal strategy still applies its own uptrend gate and its
    own ranking result, so caching cannot change portfolio decisions.
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

    states: dict[str, list[int]] = {}
    for ticker, frame in prepared.items():
        values = run_strategy(frame, signal_strategy) if signal_states is None else list(signal_states[ticker])
        if len(values) != len(frame) or any(value not in (0, 1) for value in values):
            raise ValueError(f"{ticker}: invalid signal states")
        states[ticker] = values

    date_index = {
        ticker: {pd.Timestamp(value): index for index, value in enumerate(frame["date"])}
        for ticker, frame in prepared.items()
    }
    cache = management_cache if management_cache is not None else {}

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
        uptrend = {
            ticker
            for ticker in prepared
            if states[ticker][date_index[ticker][decision_date]] == 1
        }

        base = cache.get(decision_date)
        if base is None:
            base = evaluate_universe(prepared, decision_date)
            cache[decision_date] = base.copy(deep=True)
        candidates = base.copy(deep=True)
        candidates["indicator_uptrend"] = candidates["ticker"].astype(str).isin(uptrend)
        candidates["management_eligible"] = (
            candidates["indicator_uptrend"] & candidates["eligible_for_ranking"].fillna(False)
        )
        selection = _rank(candidates, "management_eligible")
        candidates = selection.candidates.copy()
        candidates.insert(1, "execution_date", execution_date)
        candidates.insert(2, "signal_strategy", signal_strategy)
        decision_frames.append(candidates)

        management_eligible = candidates.loc[candidates["management_eligible"], "ticker"].astype(str).tolist()
        chosen = selection.selected_ticker
        score = None
        if chosen is not None:
            score = float(candidates.loc[candidates["selected"], "score"].iloc[0])
        rebalance_rows.append(
            {
                "signal_strategy": signal_strategy,
                "decision_date": decision_date,
                "execution_date": execution_date,
                "uptrend_count": len(uptrend),
                "uptrend_tickers": ";".join(sorted(uptrend)),
                "management_eligible_count": len(management_eligible),
                "management_eligible_tickers": ";".join(management_eligible),
                "selected_ticker": chosen or "",
                "selected_score": score,
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
            raise AssertionError("Top-1 manager cannot hold more than one ticker")
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
        signal_strategy,
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
