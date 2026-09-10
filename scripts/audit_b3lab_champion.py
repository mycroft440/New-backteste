from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.data_quality import apply_known_scale_repairs

DATA_DIR = ROOT / "data" / "quotes"
RESULT_DIR = ROOT / "results" / "portfolio" / "b3lab_champion"
AUDIT_PATH = RESULT_DIR / "audit.json"

ROC_WINDOWS = (252, 126, 63)
SHORT_WINDOW = 22
TREND_WINDOW = 200
VOL_WINDOW = 18
EXPECTED_CONFIG = "top1_short22_riskadj_rocfilter_w1_1_1_trend200_vol18_posscore"
EXPECTED_INDICATOR = "roc252_126_63_filter__roc22_over_vol18__sma200"


def _assert_close(actual: float, expected: float, *, label: str, rel: float = 1e-9, abs_: float = 1e-8) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=abs_):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _load_frames() -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    paths = sorted(DATA_DIR.glob("*.csv"))
    if len(paths) != 40:
        raise AssertionError(f"expected 40 quote files, found {len(paths)}")

    frames: dict[str, pd.DataFrame] = {}
    common: set[pd.Timestamp] | None = None
    for path in paths:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame, _ = apply_known_scale_repairs(frame, path.stem)
        frame = frame.sort_values("date").reset_index(drop=True)
        if frame["date"].duplicated().any():
            raise AssertionError(f"{path.stem}: duplicate dates")
        frames[path.stem] = frame
        dates = set(pd.Timestamp(value) for value in frame["date"])
        common = dates if common is None else common.intersection(dates)

    common_dates = sorted(common or set())
    if not common_dates:
        raise AssertionError("no common trading sessions")
    return frames, common_dates


def _manual_snapshot(frame: pd.DataFrame, decision_date: pd.Timestamp) -> dict[str, float | bool | None]:
    matches = frame.index[frame["date"] == decision_date].tolist()
    if len(matches) != 1:
        return {"eligible": False, "score": None}
    index = int(matches[0])
    closes = frame["close"].astype(float)
    required = max(max(ROC_WINDOWS), SHORT_WINDOW, TREND_WINDOW - 1, VOL_WINDOW)
    if index < required:
        return {"eligible": False, "score": None}

    close = float(closes.iloc[index])
    if not math.isfinite(close) or close <= 0:
        return {"eligible": False, "score": None}

    roc_252 = close / float(closes.iloc[index - 252]) - 1.0
    roc_126 = close / float(closes.iloc[index - 126]) - 1.0
    roc_63 = close / float(closes.iloc[index - 63]) - 1.0
    composite = (roc_252 + roc_126 + roc_63) / 3.0
    roc_22 = close / float(closes.iloc[index - SHORT_WINDOW]) - 1.0
    sma_200 = float(closes.iloc[index - TREND_WINDOW + 1 : index + 1].mean())
    returns = closes.iloc[index - VOL_WINDOW : index + 1].pct_change().dropna()
    vol_18 = float(returns.std(ddof=1) * math.sqrt(252.0))

    eligible = (
        math.isfinite(composite)
        and composite > 0
        and math.isfinite(roc_22)
        and roc_22 > 0
        and math.isfinite(sma_200)
        and close > sma_200
        and math.isfinite(vol_18)
        and vol_18 > 0
    )
    score = roc_22 / vol_18 if eligible else None
    return {
        "eligible": eligible,
        "score": score,
        "close": close,
        "roc_252": roc_252,
        "roc_126": roc_126,
        "roc_63": roc_63,
        "composite_roc": composite,
        "roc_22": roc_22,
        "sma_200": sma_200,
        "annualized_volatility_18": vol_18,
    }


def _split_tickers(value: object) -> set[str]:
    text = "" if pd.isna(value) else str(value)
    return {part for part in text.split(";") if part}


def main() -> int:
    summary = json.loads((RESULT_DIR / "summary.json").read_text())
    if summary["management_strategy"] != EXPECTED_CONFIG:
        raise AssertionError("unexpected management strategy")
    if summary["indicator"] != EXPECTED_INDICATOR:
        raise AssertionError("unexpected indicator")

    frames, common_dates_all = _load_frames()
    start = pd.Timestamp(summary["start"])
    end = pd.Timestamp(summary["end"])
    common_dates = [date for date in common_dates_all if start <= date <= end]
    if len(common_dates) < 2:
        raise AssertionError("effective common calendar is too short")
    common_position = {date: index for index, date in enumerate(common_dates_all)}

    candidates = pd.read_csv(RESULT_DIR / "rebalance_candidates.csv")
    rebalances = pd.read_csv(RESULT_DIR / "rebalance_summary.csv")
    orders = pd.read_csv(RESULT_DIR / "orders.csv")
    curve = pd.read_csv(RESULT_DIR / "equity_curve.csv")
    annual = pd.read_csv(RESULT_DIR / "annual_returns.csv")
    for frame in (candidates, rebalances, orders, curve):
        for column in ("date", "decision_date", "execution_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="raise")

    # 1) Independently recompute every monthly ranking from raw repaired closes.
    decision_checks = 0
    expected_candidate_rows = 0
    for _, row in rebalances.iterrows():
        decision = pd.Timestamp(row["decision_date"])
        execution = pd.Timestamp(row["execution_date"])
        all_index = common_position.get(decision)
        if all_index is None or all_index + 1 >= len(common_dates_all):
            raise AssertionError(f"{decision.date()}: decision is not on common calendar")
        if common_dates_all[all_index + 1] != execution:
            raise AssertionError(f"{decision.date()}: execution is not next common session")

        manual: list[tuple[str, float]] = []
        for ticker in sorted(frames):
            snapshot = _manual_snapshot(frames[ticker], decision)
            if bool(snapshot["eligible"]):
                manual.append((ticker, float(snapshot["score"])))
        manual.sort(key=lambda item: (-item[1], item[0]))
        expected_eligible = {ticker for ticker, _ in manual}
        expected_selected = manual[0][0] if manual else ""
        expected_score = manual[0][1] if manual else None

        actual_eligible = _split_tickers(row.get("eligible_tickers", ""))
        actual_selected = "" if pd.isna(row.get("selected_ticker")) else str(row.get("selected_ticker"))
        if actual_eligible != expected_eligible:
            raise AssertionError(f"{decision.date()}: eligible ticker set mismatch")
        if actual_selected != expected_selected:
            raise AssertionError(
                f"{decision.date()}: selected {actual_selected!r}, expected {expected_selected!r}"
            )
        if expected_score is None:
            if not pd.isna(row.get("selected_score")):
                raise AssertionError(f"{decision.date()}: expected no score")
        else:
            _assert_close(float(row["selected_score"]), expected_score, label=f"{decision.date()} selected score")

        actual_rows = candidates[candidates["decision_date"] == decision]
        if len(actual_rows) != 40:
            raise AssertionError(f"{decision.date()}: expected 40 candidate rows, found {len(actual_rows)}")
        expected_candidate_rows += 40
        selected_rows = actual_rows[actual_rows["selected"].astype(str).str.lower().isin(["true", "1"])]
        if len(selected_rows) != (1 if expected_selected else 0):
            raise AssertionError(f"{decision.date()}: selected-row count mismatch")
        decision_checks += 1

    if len(candidates) != expected_candidate_rows:
        raise AssertionError("candidate row total mismatch")

    # 2) Rebuild the cash ledger and daily marked equity from the recorded orders and local prices.
    commission_rate = float(summary["commission_bps"]) / 10_000.0
    slippage_rate = float(summary["slippage_bps"]) / 10_000.0
    cash = float(summary["initial_cash_brl"])
    holdings = {ticker: 0 for ticker in frames}
    date_index = {
        ticker: {pd.Timestamp(value): int(i) for i, value in enumerate(frame["date"])}
        for ticker, frame in frames.items()
    }
    orders_by_date = {date: group for date, group in orders.groupby("date", sort=False)}
    curve_by_date = curve.set_index("date")
    order_checks = 0
    curve_checks = 0

    for date in common_dates:
        for _, order in orders_by_date.get(date, pd.DataFrame()).iterrows():
            ticker = str(order["ticker"])
            qty = int(order["shares"])
            if qty <= 0:
                raise AssertionError(f"{date.date()}: nonpositive order quantity")
            idx = date_index[ticker][date]
            candle = frames[ticker].iloc[idx]
            side = str(order["side"])
            reason = str(order["reason"])
            if reason == "final_close_liquidation":
                raw_price = float(candle["close"])
            else:
                raw_price = float(candle["open"])
            expected_price = raw_price * (1.0 + slippage_rate if side == "BUY" else 1.0 - slippage_rate)
            _assert_close(float(order["price"]), expected_price, label=f"{date.date()} {side} price")
            gross = qty * expected_price
            commission = gross * commission_rate
            _assert_close(float(order["gross_brl"]), gross, label=f"{date.date()} {side} gross")
            _assert_close(float(order["commission_brl"]), commission, label=f"{date.date()} {side} commission")

            if side == "BUY":
                cash -= gross + commission
                holdings[ticker] += qty
            elif side == "SELL":
                if holdings[ticker] < qty:
                    raise AssertionError(f"{date.date()}: selling more {ticker} than held")
                cash += gross - commission
                holdings[ticker] -= qty
            else:
                raise AssertionError(f"{date.date()}: invalid side {side}")
            _assert_close(float(order["cash_after_brl"]), cash, label=f"{date.date()} cash after order")
            if sum(int(value > 0) for value in holdings.values()) > 1:
                raise AssertionError(f"{date.date()}: Top-1 held more than one ticker")
            order_checks += 1

        if date not in curve_by_date.index:
            raise AssertionError(f"{date.date()}: missing equity curve row")
        curve_row = curve_by_date.loc[date]
        held = [ticker for ticker, qty in holdings.items() if qty > 0]
        if len(held) > 1:
            raise AssertionError(f"{date.date()}: multiple holdings")
        held_ticker = held[0] if held else ""
        shares = holdings[held_ticker] if held_ticker else 0
        invested = 0.0
        if held_ticker:
            idx = date_index[held_ticker][date]
            invested = shares * float(frames[held_ticker].iloc[idx]["close"])
        equity = cash + invested
        curve_held = "" if pd.isna(curve_row["held_ticker"]) else str(curve_row["held_ticker"])
        if curve_held != held_ticker:
            raise AssertionError(f"{date.date()}: held ticker mismatch")
        if int(curve_row["shares"]) != shares:
            raise AssertionError(f"{date.date()}: share count mismatch")
        _assert_close(float(curve_row["cash"]), cash, label=f"{date.date()} curve cash")
        _assert_close(float(curve_row["invested_value"]), invested, label=f"{date.date()} invested value")
        _assert_close(float(curve_row["equity"]), equity, label=f"{date.date()} equity")
        curve_checks += 1

    if len(curve) != len(common_dates):
        raise AssertionError("equity curve row count does not equal common calendar")
    if any(qty != 0 for qty in holdings.values()):
        raise AssertionError("portfolio is not fully liquidated at end")
    if orders.empty or str(orders.iloc[-1]["reason"]) != "final_close_liquidation":
        raise AssertionError("last order is not final close liquidation")

    # 3) Recompute summary/annual metrics from the audited equity curve.
    final_equity = float(curve.iloc[-1]["equity"])
    initial_cash = float(summary["initial_cash_brl"])
    _assert_close(final_equity, cash, label="final cash/equity")
    _assert_close(float(summary["final_equity_brl"]), final_equity, label="summary final equity")
    _assert_close(float(summary["final_profit_brl"]), final_equity - initial_cash, label="summary profit")
    _assert_close(
        float(summary["final_return_pct"]),
        (final_equity / initial_cash - 1.0) * 100.0,
        label="summary return",
    )

    peak = initial_cash
    max_drawdown = 0.0
    for value in curve["equity"].astype(float):
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1.0)
    _assert_close(float(summary["max_drawdown_pct"]), max_drawdown * 100.0, label="max drawdown")

    annual_source = curve[["date", "equity"]].copy()
    annual_source["year"] = annual_source["date"].dt.year
    year_end = annual_source.groupby("year", sort=True).tail(1).set_index("year")["equity"]
    expected_annual: list[tuple[int, float, float]] = []
    previous = initial_cash
    for year, end_equity_raw in year_end.items():
        end_equity = float(end_equity_raw)
        return_pct = (end_equity / previous - 1.0) * 100.0
        expected_annual.append((int(year), end_equity, return_pct))
        previous = end_equity

    if len(annual) != len(expected_annual):
        raise AssertionError("annual row count mismatch")
    for actual, expected in zip(annual.itertuples(index=False), expected_annual):
        year, end_equity, return_pct = expected
        if int(actual.year) != year:
            raise AssertionError("annual year mismatch")
        _assert_close(float(actual.end_equity), end_equity, label=f"{year} end equity")
        _assert_close(float(actual.return_pct), return_pct, label=f"{year} return")

    full_years = [(year, ret) for year, _, ret in expected_annual if start.year < year < end.year]
    losing_years = [year for year, ret in full_years if ret < 0]
    mean_full_year = sum(ret for _, ret in full_years) / len(full_years) if full_years else None
    if list(summary["losing_full_calendar_years"]) != losing_years:
        raise AssertionError("losing full-calendar years mismatch")
    if mean_full_year is None:
        if summary["mean_full_calendar_year_return_pct"] is not None:
            raise AssertionError("expected no mean full-year return")
    else:
        _assert_close(
            float(summary["mean_full_calendar_year_return_pct"]),
            mean_full_year,
            label="mean full-calendar-year return",
        )

    report = {
        "status": "PASS",
        "strategy": EXPECTED_CONFIG,
        "indicator": EXPECTED_INDICATOR,
        "tickers": len(frames),
        "decision_checks": decision_checks,
        "candidate_rows_checked": len(candidates),
        "orders_checked": order_checks,
        "equity_rows_checked": curve_checks,
        "annual_rows_checked": len(annual),
        "final_equity_brl": final_equity,
        "final_return_pct": (final_equity / initial_cash - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "losing_full_calendar_years": losing_years,
        "mean_full_calendar_year_return_pct": mean_full_year,
        "last_order_reason": str(orders.iloc[-1]["reason"]),
    }
    AUDIT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
