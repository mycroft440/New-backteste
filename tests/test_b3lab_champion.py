from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.indicators.b3lab_champion import (
    SOURCE_CONFIG,
    evaluate_champion_indicator,
)
from b3_backtest.portfolio import (
    CHAMPION_MANAGEMENT_STRATEGY_NAME,
    champion_source_metadata,
    select_b3lab_champion,
    simulate_b3lab_champion_portfolio,
)


def _frame(dates: pd.DatetimeIndex, growth: float, phase: float = 0.0) -> pd.DataFrame:
    closes = [100.0 * ((1.0 + growth) ** i) * (1.0 + 0.002 * math.sin(i / 5.0 + phase)) for i in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [value * 1.0005 for value in closes],
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [100000] * len(dates),
        }
    )


def test_champion_indicator_matches_manual_formula():
    dates = pd.bdate_range("2024-01-02", periods=320)
    frame = _frame(dates, 0.0020)
    decision = dates[-1]
    snapshot = evaluate_champion_indicator(frame, decision, ticker="AAA3")
    assert snapshot.eligible

    closes = frame["close"].astype(float)
    index = len(frame) - 1
    manual_252 = closes.iloc[index] / closes.iloc[index - 252] - 1.0
    manual_126 = closes.iloc[index] / closes.iloc[index - 126] - 1.0
    manual_63 = closes.iloc[index] / closes.iloc[index - 63] - 1.0
    manual_composite = (manual_252 + manual_126 + manual_63) / 3.0
    manual_22 = closes.iloc[index] / closes.iloc[index - 22] - 1.0
    manual_sma200 = closes.iloc[index - 199 : index + 1].mean()
    manual_vol18 = closes.iloc[index - 18 : index + 1].pct_change().dropna().std(ddof=1) * math.sqrt(252.0)

    assert snapshot.roc_252 == pytest.approx(manual_252)
    assert snapshot.roc_126 == pytest.approx(manual_126)
    assert snapshot.roc_63 == pytest.approx(manual_63)
    assert snapshot.composite_roc == pytest.approx(manual_composite)
    assert snapshot.roc_22 == pytest.approx(manual_22)
    assert snapshot.sma_200 == pytest.approx(manual_sma200)
    assert snapshot.annualized_volatility_18 == pytest.approx(manual_vol18)
    assert snapshot.score == pytest.approx(manual_22 / manual_vol18)


def test_future_prices_do_not_change_past_selection():
    dates = pd.bdate_range("2023-01-02", periods=420)
    frames = {
        "AAA3": _frame(dates, 0.0015, 0.1),
        "BBB3": _frame(dates, 0.0020, 1.1),
    }
    decision = dates[330]
    before = select_b3lab_champion(frames, decision)

    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    for frame in changed.values():
        mask = frame["date"] > decision
        frame.loc[mask, ["open", "high", "low", "close"]] *= 9.0
    after = select_b3lab_champion(changed, decision)

    assert before.selected_ticker == after.selected_ticker
    columns = ["ticker", "eligible_for_ranking", "score", "rank", "selected"]
    pd.testing.assert_frame_equal(before.candidates[columns], after.candidates[columns])


def test_top1_selection_is_highest_eligible_score():
    dates = pd.bdate_range("2023-01-02", periods=420)
    frames = {
        "AAA3": _frame(dates, 0.0014, 0.2),
        "BBB3": _frame(dates, 0.0018, 0.8),
        "CCC3": _frame(dates, 0.0022, 1.4),
    }
    selection = select_b3lab_champion(frames, dates[-1])
    eligible = selection.candidates[selection.candidates["eligible_for_ranking"]].sort_values(
        ["score", "ticker"], ascending=[False, True]
    )
    assert not eligible.empty
    assert selection.selected_ticker == str(eligible.iloc[0]["ticker"])
    assert int(selection.candidates["selected"].sum()) == 1
    assert float(selection.candidates.loc[selection.candidates["selected"], "target_weight"].iloc[0]) == 1.0


def test_monthly_decision_executes_next_open_and_final_position_is_liquidated():
    dates = pd.bdate_range("2023-01-02", periods=500)
    frames = {
        "AAA3": _frame(dates, 0.0016, 0.3),
        "BBB3": _frame(dates, 0.0020, 1.3),
    }
    result = simulate_b3lab_champion_portfolio(
        frames,
        initial_cash=1000.0,
        commission_bps=0.0,
        slippage_bps=0.0,
    )
    buys = result.orders[result.orders["side"] == "BUY"]
    assert not buys.empty
    first_buy = buys.iloc[0]
    matching = result.rebalance_summary[
        result.rebalance_summary["execution_date"] == pd.Timestamp(first_buy["date"])
    ]
    assert not matching.empty
    assert str(matching.iloc[0]["selected_ticker"]) == str(first_buy["ticker"])

    expected_open = float(
        frames[str(first_buy["ticker"])].loc[
            frames[str(first_buy["ticker"])]["date"] == pd.Timestamp(first_buy["date"]), "open"
        ].iloc[0]
    )
    assert float(first_buy["price"]) == pytest.approx(expected_open)
    assert str(result.orders.iloc[-1]["reason"]) == "final_close_liquidation"
    assert str(result.equity_curve.iloc[-1]["held_ticker"]) == ""
    assert int(result.equity_curve.iloc[-1]["shares"]) == 0
    assert float(result.equity_curve.iloc[-1]["invested_value"]) == pytest.approx(0.0)


def test_source_metadata_is_locked_to_persisted_winner():
    metadata = champion_source_metadata()
    assert CHAMPION_MANAGEMENT_STRATEGY_NAME == SOURCE_CONFIG
    assert metadata["source_config"] == "top1_short22_riskadj_rocfilter_w1_1_1_trend200_vol18_posscore"
    assert metadata["result_commit"] == "bb2533bacaadecbef102869a0fe89407d2eddf22"
    assert metadata["source_full_return"] == pytest.approx(187.20552082879905)
    assert metadata["source_cost_bps"] == pytest.approx(3.2)
    assert metadata["source_slippage_bps"] == pytest.approx(10.0)


def test_real_40_ticker_smoke_is_single_position_and_finite():
    paths = sorted((ROOT / "data" / "quotes").glob("*.csv"))
    assert len(paths) == 40
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frames[path.stem] = frame

    result = simulate_b3lab_champion_portfolio(
        frames,
        start="2017-02-24",
        end="2019-12-31",
    )
    assert math.isfinite(result.final_equity)
    assert result.final_equity > 0
    assert not result.rebalance_summary.empty
    assert (result.equity_curve["held_ticker"].astype(str).str.count(";") == 0).all()
    assert str(result.equity_curve.iloc[-1]["held_ticker"]) == ""
