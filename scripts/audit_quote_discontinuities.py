from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/quotes")
OUTPUT = Path("results/backtests/quote_anomalies.csv")
# For liquid B3 stocks a one-session 2x/0.5x jump is implausible unless there is
# a corporate-action/ticker-continuity problem. We record both close-to-close and
# next-open gaps so the backtest cannot silently accept such discontinuities.
UPPER_RATIO = 2.0
LOWER_RATIO = 0.5


def main() -> None:
    rows: list[dict[str, object]] = []
    files = sorted(DATA_DIR.glob("*.csv"))
    if len(files) != 40:
        raise RuntimeError(f"expected 40 quote files, found {len(files)}")

    for path in files:
        ticker = path.stem
        frame = pd.read_csv(path, parse_dates=["date"])
        prev_close = frame["close"].shift(1)
        close_ratio = frame["close"] / prev_close
        open_ratio = frame["open"] / prev_close

        flagged = (
            (close_ratio >= UPPER_RATIO)
            | (close_ratio <= LOWER_RATIO)
            | (open_ratio >= UPPER_RATIO)
            | (open_ratio <= LOWER_RATIO)
        ) & prev_close.notna()

        for idx in frame.index[flagged]:
            rows.append(
                {
                    "ticker": ticker,
                    "date": frame.loc[idx, "date"].date().isoformat(),
                    "previous_close": float(prev_close.loc[idx]),
                    "open": float(frame.loc[idx, "open"]),
                    "close": float(frame.loc[idx, "close"]),
                    "open_to_prev_close_ratio": float(open_ratio.loc[idx]),
                    "close_to_prev_close_ratio": float(close_ratio.loc[idx]),
                }
            )

    output = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "date",
            "previous_close",
            "open",
            "close",
            "open_to_prev_close_ratio",
            "close_to_prev_close_ratio",
        ],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not output.empty:
        output = output.sort_values(
            ["ticker", "date"], ascending=[True, True]
        ).reset_index(drop=True)
        numeric = output.select_dtypes(include="number").columns
        output[numeric] = output[numeric].round(8)
    output.to_csv(OUTPUT, index=False)

    print(f"quote files audited: {len(files)}")
    print(f"discontinuities >=2x or <=0.5x: {len(output)}")
    if not output.empty:
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
