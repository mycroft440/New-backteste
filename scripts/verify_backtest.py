from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from b3_backtest.backtest import simulate_from_positions
from b3_backtest.data_quality import apply_known_scale_repairs, assert_trusted_window
from b3_backtest.strategies.catalog import SOURCE_COMMIT, list_strategies, run_strategy

INITIAL_CASH = 1000.0
EXPECTED_TICKERS = 40
EXPECTED_STRATEGIES = 40
DATA_DIR = Path("data/quotes")
OUTPUT_DIR = Path("results/backtests")
SOURCE_ROOT = Path(os.environ.get("B3_STRATEGY_LAB_ROOT", ".audit/b3-strategy-lab"))
CSV_TOLERANCE = 5.1e-4
ENGINE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ReferenceResult:
    final_equity: float
    final_profit: float
    final_return_pct: float
    trades: list[dict[str, object]]
    annual_returns: list[dict[str, object]]


def _load_quotes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise AssertionError(f"{path}: dates are not unique and sorted")
    return frame


def _common_window(frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(pd.Timestamp(frame.iloc[0]["date"]) for frame in frames.values())
    end = min(pd.Timestamp(frame.iloc[-1]["date"]) for frame in frames.values())
    return start, end


def _reference_simulator(
    frame: pd.DataFrame,
    positions: list[int],
    *,
    initial_target: int,
) -> ReferenceResult:
    """Independent, deliberately simple implementation of the execution contract.

    This function does not call the production engine and does not reuse its trade or
    annual-return helpers. It exists only to cross-check production accounting.
    """
    dates = pd.to_datetime(frame["date"]).tolist()
    opens = [float(value) for value in frame["open"]]
    closes = [float(value) for value in frame["close"]]

    cash = INITIAL_CASH
    shares = 0
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None
    entry_cost = 0.0
    trades: list[dict[str, object]] = []
    equity_by_date: list[tuple[pd.Timestamp, float]] = []

    for i, (date, open_price, close_price) in enumerate(zip(dates, opens, closes, strict=True)):
        target = initial_target if i == 0 else positions[i - 1]

        if target == 1 and shares == 0:
            quantity = math.floor(cash / open_price)
            if quantity > 0:
                shares = quantity
                entry_date = pd.Timestamp(date)
                entry_price = open_price
                entry_cost = quantity * open_price
                cash -= entry_cost
        elif target == 0 and shares > 0:
            proceeds = shares * open_price
            pnl = proceeds - entry_cost
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": pd.Timestamp(date),
                    "shares": shares,
                    "entry_price": entry_price,
                    "exit_price": open_price,
                    "pnl_brl": pnl,
                    "return_pct": pnl / entry_cost * 100.0,
                }
            )
            cash += proceeds
            shares = 0
            entry_date = None
            entry_price = None
            entry_cost = 0.0

        equity_by_date.append((pd.Timestamp(date), cash + shares * close_price))

    final_equity = float(equity_by_date[-1][1])
    final_profit = final_equity - INITIAL_CASH
    final_return_pct = final_profit / INITIAL_CASH * 100.0

    year_end: dict[int, float] = {}
    for date, equity in equity_by_date:
        year_end[date.year] = equity
    annual_returns: list[dict[str, object]] = []
    previous = INITIAL_CASH
    for year in sorted(year_end):
        ending = float(year_end[year])
        annual_returns.append(
            {
                "year": year,
                "end_equity": ending,
                "return_pct": (ending / previous - 1.0) * 100.0,
            }
        )
        previous = ending

    return ReferenceResult(
        final_equity=final_equity,
        final_profit=final_profit,
        final_return_pct=final_return_pct,
        trades=trades,
        annual_returns=annual_returns,
    )


def _assert_close(actual: float, expected: float, *, tolerance: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: actual={actual!r} expected={expected!r}")


def _assert_optional_close(actual: object, expected: object, *, label: str) -> None:
    actual_missing = pd.isna(actual)
    expected_missing = expected is None or pd.isna(expected)
    if actual_missing and expected_missing:
        return
    if actual_missing != expected_missing:
        raise AssertionError(f"{label}: actual={actual!r} expected={expected!r}")
    _assert_close(float(actual), float(expected), tolerance=CSV_TOLERANCE, label=label)


def _source_modules():
    if not SOURCE_ROOT.exists():
        raise AssertionError(f"source checkout not found: {SOURCE_ROOT}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != SOURCE_COMMIT:
        raise AssertionError(f"source commit mismatch: {actual_commit} != {SOURCE_COMMIT}")
    sys.path.insert(0, str(SOURCE_ROOT.resolve()))
    from b3_strategy_lab import additional_strategies as original  # type: ignore
    from b3_strategy_lab.candles import Candle  # type: ignore

    return original, Candle, actual_commit


def _to_source_candles(frame: pd.DataFrame, ticker: str, Candle) -> list[object]:
    candles: list[object] = []
    for row in frame.itertuples(index=False):
        open_price = float(row.open)
        high = float(row.high)
        low = float(row.low)
        close = float(row.close)
        volume = int(round(float(row.volume)))
        candles.append(
            Candle(
                date=pd.Timestamp(row.date).date().isoformat(),
                ticker=ticker,
                source_symbol=f"{ticker}.SA",
                open=open_price,
                high=high,
                low=low,
                close=close,
                adj_close=close,
                volume=volume,
                raw_open=open_price,
                raw_high=high,
                raw_low=low,
                raw_close=close,
                adjustment_factor=1.0,
                raw_volume=volume,
            )
        )
    return candles


def _compare_trades(strategy: str, ticker: str, production: pd.DataFrame, reference: list[dict[str, object]]) -> None:
    if len(production) != len(reference):
        raise AssertionError(
            f"{strategy}/{ticker}: trade count mismatch {len(production)} != {len(reference)}"
        )
    numeric = ["shares", "entry_price", "exit_price", "pnl_brl", "return_pct"]
    for i, expected in enumerate(reference):
        actual = production.iloc[i]
        for column in ("entry_date", "exit_date"):
            if pd.Timestamp(actual[column]) != pd.Timestamp(expected[column]):
                raise AssertionError(
                    f"{strategy}/{ticker}/trade{i}/{column}: {actual[column]} != {expected[column]}"
                )
        for column in numeric:
            _assert_close(
                float(actual[column]),
                float(expected[column]),
                tolerance=ENGINE_TOLERANCE,
                label=f"{strategy}/{ticker}/trade{i}/{column}",
            )


def _compare_annual(strategy: str, ticker: str, production: pd.DataFrame, reference: list[dict[str, object]]) -> None:
    if len(production) != len(reference):
        raise AssertionError(f"{strategy}/{ticker}: annual row count mismatch")
    for i, expected in enumerate(reference):
        actual = production.iloc[i]
        if int(actual["year"]) != int(expected["year"]):
            raise AssertionError(f"{strategy}/{ticker}/annual{i}: year mismatch")
        _assert_close(
            float(actual["end_equity"]),
            float(expected["end_equity"]),
            tolerance=ENGINE_TOLERANCE,
            label=f"{strategy}/{ticker}/annual{i}/equity",
        )
        _assert_close(
            float(actual["return_pct"]),
            float(expected["return_pct"]),
            tolerance=ENGINE_TOLERANCE,
            label=f"{strategy}/{ticker}/annual{i}/return",
        )


def _causality_check(frame: pd.DataFrame, specs) -> int:
    if len(frame) < 500:
        raise AssertionError("not enough rows for causality test")
    cutoff = len(frame) - 250
    changed = frame.copy()
    future = changed.index > cutoff
    for column in ("open", "high", "low", "close"):
        changed.loc[future, column] = changed.loc[future, column].astype(float) * 1.371
    checks = 0
    for spec in specs:
        original = run_strategy(frame, spec.name)
        mutated = run_strategy(changed, spec.name)
        if original[: cutoff + 1] != mutated[: cutoff + 1]:
            raise AssertionError(f"{spec.name}: future-price mutation changed a past signal")
        checks += 1
    return checks


def main() -> None:
    original, Candle, source_commit = _source_modules()
    source_defs = {item.name: item for item in original._build_definitions()}
    specs = list_strategies()
    if len(specs) != EXPECTED_STRATEGIES:
        raise AssertionError(f"expected {EXPECTED_STRATEGIES} strategies, found {len(specs)}")
    missing_source = [spec.name for spec in specs if spec.name not in source_defs]
    if missing_source:
        raise AssertionError(f"strategies missing from pinned source: {missing_source}")

    paths = sorted(DATA_DIR.glob("*.csv"))
    if len(paths) != EXPECTED_TICKERS:
        raise AssertionError(f"expected {EXPECTED_TICKERS} tickers, found {len(paths)}")

    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frame, _repairs = apply_known_scale_repairs(_load_quotes(path), path.stem)
        frames[path.stem] = frame
    common_start, common_end = _common_window(frames)
    full_years = set(range(common_start.year + 1, common_end.year))

    detail_csv = pd.read_csv(OUTPUT_DIR / "strategy_ticker_results.csv")
    annual_csv = pd.read_csv(OUTPUT_DIR / "strategy_annual_results.csv")
    summary_csv = pd.read_csv(OUTPUT_DIR / "strategy_summary.csv")
    if len(detail_csv) != EXPECTED_TICKERS * EXPECTED_STRATEGIES:
        raise AssertionError(f"detail CSV has {len(detail_csv)} rows")
    if len(summary_csv) != EXPECTED_STRATEGIES:
        raise AssertionError(f"summary CSV has {len(summary_csv)} rows")

    source_signal_pairs = 0
    source_signal_values = 0
    engine_reference_pairs = 0
    detail_rows_verified = 0
    annual_reference_rows: list[dict[str, object]] = []
    reference_detail_rows: list[dict[str, object]] = []
    max_trade: tuple[float, str, str, str, str] | None = None
    min_trade: tuple[float, str, str, str, str] | None = None

    # Causality is tested on a long real frame across all 40 strategy variants.
    causality_checks = _causality_check(frames["PETR4"], specs)

    for ticker, frame in frames.items():
        start_pos = int(frame.index[frame["date"] >= common_start][0])
        end_pos = int(frame.index[frame["date"] <= common_end][-1])
        test_frame = frame.iloc[start_pos : end_pos + 1].reset_index(drop=True)
        assert_trusted_window(test_frame, ticker)

        source_candles = _to_source_candles(frame, ticker, Candle)
        source_by_strategy = {
            spec.name: source_defs[spec.name].function(source_candles) for spec in specs
        }

        for spec in specs:
            full_positions = run_strategy(frame, spec.name)
            source_positions = source_by_strategy[spec.name]
            if full_positions != source_positions:
                first = next(
                    i for i, (a, b) in enumerate(zip(full_positions, source_positions, strict=True)) if a != b
                )
                raise AssertionError(
                    f"{spec.name}/{ticker}: source signal mismatch at row {first}, "
                    f"date={frame.iloc[first]['date']} current={full_positions[first]} source={source_positions[first]}"
                )
            source_signal_pairs += 1
            source_signal_values += len(full_positions)

            positions = full_positions[start_pos : end_pos + 1]
            initial_target = full_positions[start_pos - 1] if start_pos > 0 else 0
            production = simulate_from_positions(
                test_frame,
                positions,
                initial_cash=INITIAL_CASH,
                commission_bps=0.0,
                slippage_bps=0.0,
                initial_target=initial_target,
            )
            reference = _reference_simulator(
                test_frame,
                positions,
                initial_target=initial_target,
            )

            _assert_close(
                production.final_equity,
                reference.final_equity,
                tolerance=ENGINE_TOLERANCE,
                label=f"{spec.name}/{ticker}/final_equity",
            )
            _assert_close(
                production.final_profit,
                reference.final_profit,
                tolerance=ENGINE_TOLERANCE,
                label=f"{spec.name}/{ticker}/final_profit",
            )
            _assert_close(
                production.final_return_pct,
                reference.final_return_pct,
                tolerance=ENGINE_TOLERANCE,
                label=f"{spec.name}/{ticker}/final_return",
            )
            _compare_trades(spec.name, ticker, production.trades, reference.trades)
            _compare_annual(spec.name, ticker, production.annual_returns, reference.annual_returns)
            engine_reference_pairs += 1

            max_gain = max((float(t["return_pct"]) for t in reference.trades), default=None)
            max_loss = min((float(t["return_pct"]) for t in reference.trades), default=None)
            if reference.trades:
                gain_trade = max(reference.trades, key=lambda item: float(item["return_pct"]))
                loss_trade = min(reference.trades, key=lambda item: float(item["return_pct"]))
                candidate_max = (
                    float(gain_trade["return_pct"]), spec.name, ticker,
                    pd.Timestamp(gain_trade["entry_date"]).date().isoformat(),
                    pd.Timestamp(gain_trade["exit_date"]).date().isoformat(),
                )
                candidate_min = (
                    float(loss_trade["return_pct"]), spec.name, ticker,
                    pd.Timestamp(loss_trade["entry_date"]).date().isoformat(),
                    pd.Timestamp(loss_trade["exit_date"]).date().isoformat(),
                )
                if max_trade is None or candidate_max[0] > max_trade[0]:
                    max_trade = candidate_max
                if min_trade is None or candidate_min[0] < min_trade[0]:
                    min_trade = candidate_min

            csv_row = detail_csv[
                (detail_csv["strategy"] == spec.name) & (detail_csv["ticker"] == ticker)
            ]
            if len(csv_row) != 1:
                raise AssertionError(f"{spec.name}/{ticker}: expected exactly one detail CSV row")
            row = csv_row.iloc[0]
            for column, expected in (
                ("final_equity_brl", reference.final_equity),
                ("final_profit_brl", reference.final_profit),
                ("final_return_pct", reference.final_return_pct),
            ):
                _assert_close(
                    float(row[column]), float(expected), tolerance=CSV_TOLERANCE,
                    label=f"{spec.name}/{ticker}/csv/{column}",
                )
            if int(row["completed_trades"]) != len(reference.trades):
                raise AssertionError(f"{spec.name}/{ticker}: completed trade count differs in CSV")
            _assert_optional_close(row["max_trade_gain_pct"], max_gain, label=f"{spec.name}/{ticker}/csv/max_gain")
            _assert_optional_close(row["max_trade_loss_pct"], max_loss, label=f"{spec.name}/{ticker}/csv/max_loss")
            detail_rows_verified += 1

            reference_detail_rows.append(
                {
                    "strategy": spec.name,
                    "family": spec.family,
                    "ticker": ticker,
                    "final_equity_brl": reference.final_equity,
                    "final_profit_brl": reference.final_profit,
                    "final_return_pct": reference.final_return_pct,
                    "completed_trades": len(reference.trades),
                    "max_trade_gain_pct": max_gain,
                    "max_trade_loss_pct": max_loss,
                }
            )
            for annual_row in reference.annual_returns:
                annual_reference_rows.append(
                    {
                        "strategy": spec.name,
                        "family": spec.family,
                        "ticker": ticker,
                        "year": int(annual_row["year"]),
                        "is_full_year": int(annual_row["year"]) in full_years,
                        "return_pct": float(annual_row["return_pct"]),
                    }
                )

    reference_detail = pd.DataFrame(reference_detail_rows)
    reference_annual = pd.DataFrame(annual_reference_rows)
    expected_strategy_year = (
        reference_annual.groupby(["strategy", "family", "year", "is_full_year"], as_index=False)
        .agg(
            mean_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            tickers=("ticker", "nunique"),
        )
        .sort_values(["strategy", "year"])
        .reset_index(drop=True)
    )
    actual_strategy_year = annual_csv.sort_values(["strategy", "year"]).reset_index(drop=True)
    if len(actual_strategy_year) != len(expected_strategy_year):
        raise AssertionError("strategy annual CSV row count mismatch")
    for i, expected in expected_strategy_year.iterrows():
        actual = actual_strategy_year.iloc[i]
        for column in ("strategy", "family"):
            if str(actual[column]) != str(expected[column]):
                raise AssertionError(f"annual CSV row {i}: {column} mismatch")
        if int(actual["year"]) != int(expected["year"]) or int(actual["tickers"]) != int(expected["tickers"]):
            raise AssertionError(f"annual CSV row {i}: year/ticker-count mismatch")
        actual_full = str(actual["is_full_year"]).strip().lower() in {"true", "1"}
        if actual_full != bool(expected["is_full_year"]):
            raise AssertionError(f"annual CSV row {i}: full-year flag mismatch")
        for column in ("mean_return_pct", "median_return_pct"):
            _assert_close(
                float(actual[column]), float(expected[column]), tolerance=CSV_TOLERANCE,
                label=f"annual CSV row {i}/{column}",
            )

    # Rebuild the strategy summary from the independent pair results and annual series.
    expected_summary: dict[str, dict[str, object]] = {}
    for spec in specs:
        subset = reference_detail[reference_detail["strategy"] == spec.name]
        years = expected_strategy_year[
            (expected_strategy_year["strategy"] == spec.name)
            & expected_strategy_year["is_full_year"]
            & (expected_strategy_year["tickers"] == EXPECTED_TICKERS)
        ]
        losing_years = years.loc[years["mean_return_pct"] < 0, "year"].astype(int).tolist()
        best = years.loc[years["mean_return_pct"].idxmax()]
        worst = years.loc[years["mean_return_pct"].idxmin()]
        gains = subset["max_trade_gain_pct"].dropna()
        losses = subset["max_trade_loss_pct"].dropna()
        expected_summary[spec.name] = {
            "mean_final_equity_brl": float(subset["final_equity_brl"].mean()),
            "mean_final_profit_brl": float(subset["final_profit_brl"].mean()),
            "mean_final_return_pct": float(subset["final_return_pct"].mean()),
            "median_final_return_pct": float(subset["final_return_pct"].median()),
            "profitable_tickers": int((subset["final_profit_brl"] > 0).sum()),
            "losing_tickers": int((subset["final_profit_brl"] < 0).sum()),
            "losing_years": ";".join(str(year) for year in losing_years),
            "mean_annual_return_pct": float(years["mean_return_pct"].mean()),
            "best_year": int(best["year"]),
            "best_year_return_pct": float(best["mean_return_pct"]),
            "worst_year": int(worst["year"]),
            "worst_year_return_pct": float(worst["mean_return_pct"]),
            "max_trade_gain_pct": float(gains.max()) if not gains.empty else None,
            "max_trade_loss_pct": float(losses.min()) if not losses.empty else None,
            "completed_trades_total": int(subset["completed_trades"].sum()),
        }

    for _, actual in summary_csv.iterrows():
        name = str(actual["strategy"])
        if name not in expected_summary:
            raise AssertionError(f"summary has unknown strategy: {name}")
        expected = expected_summary[name]
        if int(actual["tickers_tested"]) != EXPECTED_TICKERS:
            raise AssertionError(f"{name}: summary ticker count is not 40")
        if str(actual["common_start"]) != common_start.date().isoformat() or str(actual["common_end"]) != common_end.date().isoformat():
            raise AssertionError(f"{name}: summary common window mismatch")
        for column in ("profitable_tickers", "losing_tickers", "best_year", "worst_year", "completed_trades_total"):
            if int(actual[column]) != int(expected[column]):
                raise AssertionError(f"{name}/{column}: {actual[column]} != {expected[column]}")
        if str(actual["losing_years"] if not pd.isna(actual["losing_years"]) else "") != str(expected["losing_years"]):
            raise AssertionError(f"{name}: losing years mismatch")
        for column in (
            "mean_final_equity_brl", "mean_final_profit_brl", "mean_final_return_pct",
            "median_final_return_pct", "mean_annual_return_pct", "best_year_return_pct",
            "worst_year_return_pct", "max_trade_gain_pct", "max_trade_loss_pct",
        ):
            _assert_optional_close(actual[column], expected[column], label=f"{name}/summary/{column}")

    report = {
        "status": "PASS",
        "source_repository": "mycroft440/b3-strategy-lab",
        "source_commit": source_commit,
        "tickers": EXPECTED_TICKERS,
        "strategies": EXPECTED_STRATEGIES,
        "pair_simulations": EXPECTED_TICKERS * EXPECTED_STRATEGIES,
        "source_signal_pairs_exact": source_signal_pairs,
        "source_signal_values_compared": source_signal_values,
        "engine_reference_pairs_exact": engine_reference_pairs,
        "matrix_detail_rows_verified": detail_rows_verified,
        "matrix_annual_rows_verified": len(actual_strategy_year),
        "matrix_summary_rows_verified": len(summary_csv),
        "causality_strategy_checks": causality_checks,
        "common_start": common_start.date().isoformat(),
        "common_end": common_end.date().isoformat(),
        "full_calendar_years": sorted(full_years),
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "portfolio_management": False,
        "dividends_jcp_added_by_engine": False,
        "largest_completed_trade": None if max_trade is None else {
            "return_pct": max_trade[0], "strategy": max_trade[1], "ticker": max_trade[2],
            "entry_date": max_trade[3], "exit_date": max_trade[4],
        },
        "worst_completed_trade": None if min_trade is None else {
            "return_pct": min_trade[0], "strategy": min_trade[1], "ticker": min_trade[2],
            "entry_date": min_trade[3], "exit_date": min_trade[4],
        },
        "verification_contract": [
            "ported signals equal pinned B3 Strategy Lab signals on the same repaired split-only candles",
            "production execution engine equals an independently implemented cash/share simulator",
            "all 1,600 detail rows equal independent recomputation",
            "annual aggregation and all 40 strategy summary rows equal independent recomputation",
            "future candle mutation does not change past signals for any of the 40 strategies",
            "trusted comparison window contains no unhandled >=2x or <=0.5x one-day scale discontinuity",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
