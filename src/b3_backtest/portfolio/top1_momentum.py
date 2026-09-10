from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pandas as pd

SOURCE_REPOSITORY = "mycroft440/b3-strategy-lab"
SOURCE_COMMIT = "8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d"
SOURCE_MODULE = "scripts/research_portfolio_allocation_core.py"
SOURCE_CONFIG = "top1_momentum_lb126_skip0_trend0_vol63_equal_monthly_abs_cap1_adjusted"

MANAGEMENT_STRATEGY_NAME = "top1_momentum_lb126_monthly_signal_filter"
LOOKBACK = 126
VOL_WINDOW = 63


@dataclass(frozen=True)
class RebalanceSelection:
    decision_date: pd.Timestamp
    selected_ticker: str | None
    candidates: pd.DataFrame


def _aligned_index(frame: pd.DataFrame, decision_date: pd.Timestamp) -> int | None:
    dates = pd.to_datetime(frame["date"], errors="raise")
    positions = [i for i, value in enumerate(dates) if pd.Timestamp(value) == decision_date]
    if not positions:
        return None
    if len(positions) != 1:
        raise ValueError(f"duplicate date for {decision_date.date()}")
    return positions[0]


def _annualized_volatility(closes: pd.Series, index: int, window: int) -> float:
    if index < window:
        return 0.0
    values = closes.iloc[index - window : index + 1].astype(float)
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    value = float(returns.std(ddof=1) * math.sqrt(252.0))
    return value if math.isfinite(value) else 0.0


def select_top1_momentum(
    frames: Mapping[str, pd.DataFrame],
    signal_states: Mapping[str, list[int]],
    decision_date: pd.Timestamp | str,
    *,
    lookback: int = LOOKBACK,
    vol_window: int = VOL_WINDOW,
) -> RebalanceSelection:
    """Rank only tickers whose buy/sell indicator is long at this close.

    This is the source Top-1 momentum management rule with trend_window=0:
    positive lookback momentum is required and the highest score gets 100% target
    weight. The external binary buy/sell strategy is therefore the trend filter.
    Signals use this repository's split-only close, keeping dividends/JCP excluded.
    """
    decision_date = pd.Timestamp(decision_date)
    rows: list[dict[str, object]] = []

    for ticker in sorted(frames):
        frame = frames[ticker]
        states = signal_states.get(ticker)
        if states is None or len(states) != len(frame):
            raise ValueError(f"{ticker}: missing or misaligned signal states")
        index = _aligned_index(frame, decision_date)
        state = 0
        close: float | None = None
        momentum_pct: float | None = None
        volatility_pct: float | None = None
        eligible = False
        reason = "missing_candle"

        if index is not None:
            state = int(states[index])
            if state not in (0, 1):
                raise ValueError(f"{ticker}: signal states must be binary")
            closes = frame["close"].astype(float)
            close = float(closes.iloc[index])
            if state == 0:
                reason = "indicator_not_uptrend"
            elif index < lookback or index < vol_window:
                reason = "insufficient_history"
            else:
                past = float(closes.iloc[index - lookback])
                if past <= 0 or close <= 0:
                    reason = "invalid_price"
                else:
                    momentum = close / past - 1.0
                    momentum_pct = momentum * 100.0
                    volatility = _annualized_volatility(closes, index, vol_window)
                    volatility_pct = volatility * 100.0
                    if momentum <= 0:
                        reason = "nonpositive_momentum"
                    elif volatility <= 0:
                        reason = "zero_volatility"
                    else:
                        eligible = True
                        reason = "eligible"

        rows.append(
            {
                "decision_date": decision_date,
                "ticker": ticker,
                "indicator_uptrend": state == 1,
                "close": close,
                "momentum_126_pct": momentum_pct,
                "volatility_63_pct": volatility_pct,
                "eligible_for_ranking": eligible,
                "rank": None,
                "selected": False,
                "target_weight": 0.0,
                "reason": reason,
            }
        )

    eligible_rows = [row for row in rows if bool(row["eligible_for_ranking"])]
    eligible_rows.sort(key=lambda row: (-float(row["momentum_126_pct"]), str(row["ticker"])))
    for rank, row in enumerate(eligible_rows, start=1):
        row["rank"] = rank
    selected_ticker = str(eligible_rows[0]["ticker"]) if eligible_rows else None
    if eligible_rows:
        eligible_rows[0]["selected"] = True
        eligible_rows[0]["target_weight"] = 1.0

    result = pd.DataFrame(rows)
    result["rank"] = pd.array(result["rank"], dtype="Int64")
    return RebalanceSelection(decision_date, selected_ticker, result)


def is_monthly_rebalance_close(current_date: pd.Timestamp | str, next_date: pd.Timestamp | str) -> bool:
    current = pd.Timestamp(current_date)
    following = pd.Timestamp(next_date)
    return (current.year, current.month) != (following.year, following.month)
