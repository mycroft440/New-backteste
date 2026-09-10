from __future__ import annotations

import pandas as pd

from b3_backtest.backtest import simulate_from_positions


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]),
            "open": [10.0, 10.0, 20.0, 25.0, 30.0],
            "high": [11.0, 12.0, 22.0, 27.0, 32.0],
            "low": [9.0, 9.5, 19.0, 24.0, 29.0],
            "close": [10.5, 11.0, 21.0, 26.0, 31.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
        }
    )


def test_signal_executes_only_on_next_open() -> None:
    frame = _frame()
    positions = [0, 1, 1, 0, 0]
    result = simulate_from_positions(frame, positions, initial_cash=1000.0)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_date"] == pd.Timestamp("2020-01-06")
    assert trade["entry_price"] == 20.0
    assert trade["shares"] == 50
    assert trade["exit_date"] == pd.Timestamp("2020-01-08")
    assert trade["exit_price"] == 30.0
    assert round(result.final_equity, 8) == 1500.0
    assert round(result.final_profit, 8) == 500.0
    assert round(result.final_return_pct, 8) == 50.0
    assert round(float(trade["return_pct"]), 8) == 50.0


def test_final_bar_signal_cannot_trade_without_future_bar() -> None:
    frame = _frame()
    result = simulate_from_positions(frame, [0, 0, 0, 0, 1], initial_cash=1000.0)

    assert result.trades.empty
    assert result.final_equity == 1000.0
    assert result.final_profit == 0.0


def test_whole_share_sizing_keeps_cash_remainder() -> None:
    frame = _frame()
    frame.loc[2, "open"] = 33.0
    frame.loc[2, "high"] = 34.0
    frame.loc[2, "low"] = 32.0
    frame.loc[2, "close"] = 33.5
    result = simulate_from_positions(frame, [0, 1, 1, 0, 0], initial_cash=1000.0)

    first_held = result.equity_curve.iloc[2]
    assert first_held["shares"] == 30
    assert round(float(first_held["cash"]), 8) == 10.0
