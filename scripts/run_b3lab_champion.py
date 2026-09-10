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
from b3_backtest.portfolio import (
    CHAMPION_INDICATOR_NAME,
    CHAMPION_MANAGEMENT_STRATEGY_NAME,
    champion_source_metadata,
    simulate_b3lab_champion_portfolio,
)
from b3_backtest.portfolio.b3lab_champion import SOURCE_COST_BPS, SOURCE_SLIPPAGE_BPS

DATA_DIR = ROOT / "data" / "quotes"
DEFAULT_OUTPUT = ROOT / "results" / "portfolio" / "b3lab_champion"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the B3 Strategy Lab champion Top-1 portfolio configuration.")
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

    result = simulate_b3lab_champion_portfolio(
        frames,
        initial_cash=args.initial_cash,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        start=start,
        end=end,
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(out / "rebalance_candidates.csv", index=False)
    result.rebalance_summary.to_csv(out / "rebalance_summary.csv", index=False)
    result.orders.to_csv(out / "orders.csv", index=False)
    result.equity_curve.to_csv(out / "equity_curve.csv", index=False)
    result.annual_returns.to_csv(out / "annual_returns.csv", index=False)

    full_years = result.annual_returns[
        (result.annual_returns["year"] > start.year) & (result.annual_returns["year"] < end.year)
    ]
    latest = result.rebalance_summary.iloc[-1].to_dict() if not result.rebalance_summary.empty else {}
    summary = {
        "indicator": CHAMPION_INDICATOR_NAME,
        "management_strategy": CHAMPION_MANAGEMENT_STRATEGY_NAME,
        "source": champion_source_metadata(),
        "adaptation": {
            "universe": "the 40 local New-backteste tickers, not the source lab historical universe",
            "prices": "local split-only close; dividends/JCP excluded",
            "timing": "decision at monthly common-session close; execution at next common-session open",
            "finalization": "source-faithful final close liquidation",
        },
        "initial_cash_brl": result.initial_cash,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "final_equity_brl": result.final_equity,
        "final_profit_brl": result.final_profit,
        "final_return_pct": result.final_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "losing_full_calendar_years": [
            int(value) for value in full_years.loc[full_years["return_pct"] < 0, "year"]
        ],
        "mean_full_calendar_year_return_pct": (
            float(full_years["return_pct"].mean()) if not full_years.empty else None
        ),
        "rebalance_count": len(result.rebalance_summary),
        "order_count": len(result.orders),
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "known_scale_repairs_applied": repair_count,
        "latest_rebalance": latest,
        "contract": {
            "filter_1": "equal-weight mean of ROC252, ROC126 and ROC63 must be > 0",
            "filter_2": "ROC22 must be > 0",
            "filter_3": "close must be > SMA200",
            "ranking": "highest ROC22 / annualized sample volatility of the last 18 returns",
            "selection_count": 1,
            "target_weight": "100% of available capital in the Top-1 ticker",
            "rebalance": "monthly at last common trading close",
            "execution": "next common trading session open",
            "between_rebalances": "hold the monthly target; no external binary strategy exit/re-entry",
            "no_candidate": "100% cash",
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
