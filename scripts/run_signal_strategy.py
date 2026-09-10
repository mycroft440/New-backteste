from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.strategies import generate_events, list_strategies  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate BUY/SELL strategy signals only; no portfolio management."
    )
    result.add_argument("--list", action="store_true", help="list available strategies")
    result.add_argument("--ticker", help="B3 ticker matching data/quotes/<TICKER>.csv")
    result.add_argument("--strategy", help="strategy name from the local signal catalog")
    result.add_argument("--data-dir", default="data/quotes", help="directory containing quote CSVs")
    result.add_argument("--output", help="optional event CSV output path")
    return result


def main() -> int:
    args = parser().parse_args()

    if args.list:
        for spec in list_strategies():
            print(f"{spec.name}\t{spec.family}\t{spec.description}")
        return 0

    if not args.ticker or not args.strategy:
        parser().error("--ticker and --strategy are required unless --list is used")

    ticker = args.ticker.upper().strip()
    quote_path = ROOT / args.data_dir / f"{ticker}.csv"
    if not quote_path.is_file():
        raise SystemExit(f"quote file not found: {quote_path}")

    frame = pd.read_csv(quote_path)
    events = generate_events(frame, args.strategy)

    if args.output:
        target = Path(args.output)
        if not target.is_absolute():
            target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(target, index=False, date_format="%Y-%m-%d")
        print(f"saved {len(events)} events to {target}")
    else:
        print(events.to_csv(index=False, date_format="%Y-%m-%d"), end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
