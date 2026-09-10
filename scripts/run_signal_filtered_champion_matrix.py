from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.data_quality import apply_known_scale_repairs, assert_trusted_window
from b3_backtest.portfolio import simulate_signal_filtered_champion
from b3_backtest.portfolio.b3lab_champion import SOURCE_COST_BPS, SOURCE_SLIPPAGE_BPS
from b3_backtest.strategies.catalog import list_strategies

DATA_DIR = ROOT / "data" / "quotes"
DEFAULT_OUTPUT = ROOT / "results" / "portfolio" / "signal_filtered_champion"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 40 BUY/SELL strategies as the uptrend gate for the same champion rebalance manager."
    )
    parser.add_argument("--initial-cash", type=float, default=1000.0)
    parser.add_argument("--commission-bps", type=float, default=SOURCE_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=SOURCE_SLIPPAGE_BPS)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = sorted(DATA_DIR.glob("*.csv"))
    if len(paths) != 40:
        raise SystemExit(f"expected 40 quote files, found {len(paths)}")

    frames: dict[str, pd.DataFrame] = {}
    repair_count = 0
    for path in paths:
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame, repairs = apply_known_scale_repairs(frame, path.stem)
        repair_count += len(repairs)
        frames[path.stem] = frame

    common_start = max(pd.Timestamp(frame.iloc[0]["date"]) for frame in frames.values())
    common_end = min(pd.Timestamp(frame.iloc[-1]["date"]) for frame in frames.values())
    start = pd.Timestamp(args.start) if args.start else common_start
    end = pd.Timestamp(args.end) if args.end else common_end

    for ticker, frame in frames.items():
        trusted = frame[(frame["date"] >= start) & (frame["date"] <= end)].reset_index(drop=True)
        assert_trusted_window(trusted, ticker)

    ranking_rows: list[dict[str, object]] = []
    rebalance_frames: list[pd.DataFrame] = []

    specs = list_strategies()
    if len(specs) != 40:
        raise SystemExit(f"expected 40 signal strategies, found {len(specs)}")

    for spec in specs:
        result = simulate_signal_filtered_champion(
            frames,
            signal_strategy=spec.name,
            initial_cash=args.initial_cash,
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            start=start,
            end=end,
        )

        full_years = result.annual_returns[
            (result.annual_returns["year"] > start.year) & (result.annual_returns["year"] < end.year)
        ]
        losing_years = [
            int(value) for value in full_years.loc[full_years["return_pct"] < 0, "year"]
        ]

        # Hard contract: every selected ticker must come from the strategy's uptrend set.
        for row in result.rebalance_summary.to_dict("records"):
            selected = str(row.get("selected_ticker") or "")
            if selected:
                uptrend = {value for value in str(row["uptrend_tickers"]).split(";") if value}
                if selected not in uptrend:
                    raise AssertionError(
                        f"{spec.name} {row['decision_date']}: selected {selected} outside uptrend set"
                    )

        ranking_rows.append(
            {
                "rank": 0,
                "signal_strategy": spec.name,
                "family": spec.family,
                "initial_cash_brl": result.initial_cash,
                "final_equity_brl": result.final_equity,
                "final_profit_brl": result.final_profit,
                "final_return_pct": result.final_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "mean_full_calendar_year_return_pct": (
                    float(full_years["return_pct"].mean()) if not full_years.empty else None
                ),
                "losing_full_calendar_years": ";".join(str(year) for year in losing_years),
                "rebalance_count": len(result.rebalance_summary),
                "order_count": len(result.orders),
            }
        )
        compact = result.rebalance_summary.copy()
        rebalance_frames.append(compact)
        print(f"{spec.name}: {result.final_return_pct:.4f}%", flush=True)

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["final_return_pct", "signal_strategy"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    rebalances = pd.concat(rebalance_frames, ignore_index=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out / "strategy_ranking.csv", index=False)
    rebalances.to_csv(out / "rebalance_selections.csv", index=False)

    best = ranking.iloc[0].to_dict()
    summary = {
        "contract": {
            "step_1": "each BUY/SELL strategy marks the 40 tickers as uptrend state=1 or not state=0 at the monthly decision close",
            "step_2": "only state=1 tickers are passed to the portfolio manager",
            "step_3": "inside that subset, the B3 Strategy Lab champion filters ROC252/126/63 > 0, ROC22 > 0 and close > SMA200",
            "step_4": "the manager ranks the survivors by ROC22 / annualized sample volatility18 and buys Top-1",
            "execution": "decision at monthly close; trade at next common session open",
            "between_rebalances": "hold the selected asset until the next monthly decision",
            "no_candidate": "stay 100% in cash",
        },
        "strategies_tested": len(ranking),
        "tickers": len(frames),
        "initial_cash_brl": args.initial_cash,
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "known_scale_repairs_applied": repair_count,
        "best": best,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
