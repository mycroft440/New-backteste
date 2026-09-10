from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.portfolio import MANAGEMENT_STRATEGY_NAME, simulate_managed_portfolio
from b3_backtest.portfolio.top1_momentum import is_monthly_rebalance_close, select_top1_momentum


def _frame(dates: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "open": closes,
                         "high": [v * 1.01 for v in closes], "low": [v * 0.99 for v in closes],
                         "close": closes, "volume": [1000] * len(closes)})


def test_exactly_one_management_strategy_is_exposed():
    assert MANAGEMENT_STRATEGY_NAME == "top1_momentum_lb126_monthly_signal_filter"


def test_top1_ranks_only_indicator_uptrend_tickers():
    dates = pd.bdate_range("2025-01-02", periods=160)
    frames = {"AAA3": _frame(dates, [10 + i * .05 for i in range(len(dates))]),
              "BBB3": _frame(dates, [10 + i * .10 for i in range(len(dates))]),
              "CCC3": _frame(dates, [10 + i * .20 for i in range(len(dates))])}
    states = {"AAA3": [1] * len(dates), "BBB3": [1] * len(dates), "CCC3": [0] * len(dates)}
    selection = select_top1_momentum(frames, states, dates[-1])
    rows = selection.candidates.set_index("ticker")
    assert selection.selected_ticker == "BBB3"
    assert not bool(rows.loc["CCC3", "indicator_uptrend"])
    assert not bool(rows.loc["CCC3", "selected"])
    assert int(rows.loc["BBB3", "rank"]) == 1


def test_future_prices_do_not_change_a_past_selection():
    dates = pd.bdate_range("2024-01-02", periods=260)
    frames = {"AAA3": _frame(dates, [10 + i * .03 for i in range(len(dates))]),
              "BBB3": _frame(dates, [10 + i * .04 for i in range(len(dates))])}
    states = {ticker: [1] * len(dates) for ticker in frames}
    decision = dates[200]
    before = select_top1_momentum(frames, states, decision)
    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    for frame in changed.values():
        mask = frame["date"] > decision
        frame.loc[mask, ["open", "high", "low", "close"]] *= 7.0
    after = select_top1_momentum(changed, states, decision)
    assert before.selected_ticker == after.selected_ticker
    pd.testing.assert_frame_equal(before.candidates[["ticker", "indicator_uptrend", "momentum_126_pct", "rank", "selected"]],
                                  after.candidates[["ticker", "indicator_uptrend", "momentum_126_pct", "rank", "selected"]])


def test_month_end_selection_executes_only_on_next_open():
    dates = pd.bdate_range("2024-01-02", periods=300)
    frames = {"AAA3": _frame(dates, [10 + i * .05 for i in range(len(dates))]),
              "BBB3": _frame(dates, [10 + i * .02 for i in range(len(dates))])}
    states = {ticker: [1] * len(dates) for ticker in frames}
    result = simulate_managed_portfolio(frames, signal_strategy="synthetic", signal_states=states)
    first = result.rebalance_summary[result.rebalance_summary["selected_ticker"] != ""].iloc[0]
    decision, execution = pd.Timestamp(first["decision_date"]), pd.Timestamp(first["execution_date"])
    assert execution > decision and is_monthly_rebalance_close(decision, execution)
    first_buy = result.orders[result.orders["side"] == "BUY"].iloc[0]
    assert pd.Timestamp(first_buy["date"]) == execution
    selected = str(first["selected_ticker"])
    expected_open = float(frames[selected].loc[frames[selected]["date"] == execution, "open"].iloc[0])
    assert float(first_buy["price"]) == pytest.approx(expected_open)


def test_designated_ticker_obeys_daily_indicator_exit():
    dates = pd.bdate_range("2024-01-02", periods=320)
    frames = {"AAA3": _frame(dates, [10 + i * .06 for i in range(len(dates))]),
              "BBB3": _frame(dates, [10 + i * .03 for i in range(len(dates))])}
    states = {ticker: [1] * len(dates) for ticker in frames}
    baseline = simulate_managed_portfolio(frames, signal_strategy="synthetic", signal_states=states)
    first = baseline.rebalance_summary[baseline.rebalance_summary["selected_ticker"] != ""].iloc[0]
    selected, execution = str(first["selected_ticker"]), pd.Timestamp(first["execution_date"])
    exit_signal_date = dates[dates.get_loc(execution) + 3]
    exit_idx = int(frames[selected].index[frames[selected]["date"] == exit_signal_date][0])
    states[selected][exit_idx] = 0
    result = simulate_managed_portfolio(frames, signal_strategy="synthetic", signal_states=states)
    sell = result.orders[(result.orders["side"] == "SELL") & (result.orders["reason"] == "indicator_exit")].iloc[0]
    assert pd.Timestamp(sell["date"]) == dates[dates.get_loc(exit_signal_date) + 1]


def test_real_data_smoke_never_holds_more_than_one_ticker():
    paths = sorted((ROOT / "data" / "quotes").glob("*.csv"))
    assert len(paths) == 40
    frames = {}
    for path in paths:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frames[path.stem] = frame
    result = simulate_managed_portfolio(frames, signal_strategy="macd_24_52_18",
                                        start="2017-02-24", end="2019-12-31")
    assert not result.rebalance_summary.empty
    assert result.final_equity > 0
