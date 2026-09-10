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
from b3_backtest.portfolio import simulate_managed_portfolio
from b3_backtest.strategies.catalog import get_strategy

DATA_DIR = ROOT / "data" / "quotes"
DEFAULT_OUTPUT = ROOT / "results" / "portfolio"
DEFAULT_SIGNAL_STRATEGY = "macd_24_52_18"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the single monthly Top-1 portfolio manager.")
    parser.add_argument("--signal-strategy", default=DEFAULT_SIGNAL_STRATEGY)
    parser.add_argument("--initial-cash", type=float, default=1000.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    get_strategy(args.signal_strategy)

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

    result = simulate_managed_portfolio(frames, signal_strategy=args.signal_strategy,
                                        initial_cash=args.initial_cash, commission_bps=args.commission_bps,
                                        slippage_bps=args.slippage_bps, start=start, end=end)
    out = args.output_dir / args.signal_strategy
    out.mkdir(parents=True, exist_ok=True)
    result.decisions.to_csv(out / "rebalance_candidates.csv", index=False)
    result.rebalance_summary.to_csv(out / "rebalance_summary.csv", index=False)
    result.orders.to_csv(out / "orders.csv", index=False)
    result.equity_curve.to_csv(out / "equity_curve.csv", index=False)
    result.annual_returns.to_csv(out / "annual_returns.csv", index=False)

    full_years = result.annual_returns[(result.annual_returns["year"] > start.year) &
                                      (result.annual_returns["year"] < end.year)]
    latest = result.rebalance_summary.iloc[-1].to_dict() if not result.rebalance_summary.empty else {}
    summary = {
        "signal_strategy": result.signal_strategy,
        "management_strategy": result.management_strategy,
        "management_source": {
            "repository": "mycroft440/b3-strategy-lab",
            "commit": "8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d",
            "module": "scripts/research_portfolio_allocation_core.py",
            "source_config": "top1_momentum_lb126_skip0_trend0_vol63_equal_monthly_abs_cap1_adjusted",
            "adaptation": "signal price uses split-only close; dividends/JCP remain excluded"
        },
        "initial_cash_brl": result.initial_cash,
        "start": start.date().isoformat(), "end": end.date().isoformat(),
        "final_equity_brl": result.final_equity, "final_profit_brl": result.final_profit,
        "final_return_pct": result.final_return_pct, "max_drawdown_pct": result.max_drawdown_pct,
        "losing_full_calendar_years": [int(v) for v in full_years.loc[full_years["return_pct"] < 0, "year"]],
        "mean_full_calendar_year_return_pct": float(full_years["return_pct"].mean()) if not full_years.empty else None,
        "rebalance_count": len(result.rebalance_summary), "order_count": len(result.orders),
        "commission_bps": args.commission_bps, "slippage_bps": args.slippage_bps,
        "known_scale_repairs_applied": repair_count, "latest_rebalance": latest,
        "contract": {
            "indicator_uptrend": "binary long state (1) from selected buy/sell strategy on rebalance close",
            "ranking": "highest positive 126-session momentum among indicator-uptrend tickers",
            "selection_count": 1, "rebalance": "monthly at last common trading close",
            "execution": "next common trading session open",
            "between_rebalances": "designated ticker obeys daily indicator exit/re-entry without reranking",
            "no_candidate": "100% cash"
        }
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
