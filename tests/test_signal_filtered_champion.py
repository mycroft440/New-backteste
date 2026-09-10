from __future__ import annotations

import math

import pandas as pd

from b3_backtest.portfolio import simulate_signal_filtered_champion


def _frame(multiplier: float, phase: float) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=340)
    closes = []
    for i in range(len(dates)):
        trend = 100.0 * (1.0 + multiplier * i)
        wave = 1.0 + 0.015 * math.sin(i / 5.0 + phase)
        closes.append(trend * wave)
    opens = [value * 0.999 for value in closes]
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": [max(o, c) * 1.002 for o, c in zip(opens, closes)],
            "low": [min(o, c) * 0.998 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1_000_000] * len(dates),
        }
    )


def test_downtrend_ticker_cannot_be_selected_even_with_stronger_score() -> None:
    frames = {
        "AAA3": _frame(0.0015, 0.0),
        "BBB3": _frame(0.0030, 0.7),
    }
    states = {
        "AAA3": [1] * len(frames["AAA3"]),
        "BBB3": [0] * len(frames["BBB3"]),
    }
    result = simulate_signal_filtered_champion(
        frames,
        signal_strategy="synthetic",
        initial_cash=1000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        signal_states=states,
    )

    selected = result.decisions.loc[result.decisions["selected"]]
    assert not selected.empty
    assert set(selected["ticker"].astype(str)) == {"AAA3"}
    assert selected["indicator_uptrend"].all()
    assert selected["management_eligible"].all()


def test_monthly_decision_executes_only_on_next_session_open() -> None:
    frames = {
        "AAA3": _frame(0.0015, 0.0),
        "BBB3": _frame(0.0012, 0.4),
    }
    states = {ticker: [1] * len(frame) for ticker, frame in frames.items()}
    result = simulate_signal_filtered_champion(
        frames,
        signal_strategy="synthetic",
        initial_cash=1000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        signal_states=states,
    )

    first = result.rebalance_summary.iloc[0]
    execution = pd.Timestamp(first["execution_date"])
    assert execution > pd.Timestamp(first["decision_date"])
    orders = result.orders[result.orders["reason"] == "monthly_rebalance"]
    assert not orders.empty
    assert pd.Timestamp(orders.iloc[0]["date"]) == execution


def test_selected_name_is_always_inside_uptrend_set() -> None:
    frames = {
        "AAA3": _frame(0.0015, 0.0),
        "BBB3": _frame(0.0018, 0.2),
        "CCC3": _frame(0.0021, 0.5),
    }
    states = {
        "AAA3": [1] * len(frames["AAA3"]),
        "BBB3": [1 if i % 40 < 20 else 0 for i in range(len(frames["BBB3"]))],
        "CCC3": [0] * len(frames["CCC3"]),
    }
    result = simulate_signal_filtered_champion(
        frames,
        signal_strategy="synthetic",
        initial_cash=1000.0,
        commission_bps=3.2,
        slippage_bps=10.0,
        signal_states=states,
    )

    for row in result.rebalance_summary.to_dict("records"):
        selected = str(row["selected_ticker"] or "")
        if not selected:
            continue
        uptrend = {value for value in str(row["uptrend_tickers"]).split(";") if value}
        assert selected in uptrend

    assert (result.equity_curve["shares"] >= 0).all()
    assert result.orders.iloc[-1]["reason"] == "final_close_liquidation"
