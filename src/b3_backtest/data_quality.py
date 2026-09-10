from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class ScaleRepair:
    boundary: str
    pre_boundary_price_factor: float
    reason: str


# Yahoo's UGPA3 series contains a persistent scale discontinuity on 2021-06-28:
# the pre-boundary history is approximately half the contemporaneous price scale,
# while no corporate split occurred on that date. A factor of 2 restores continuity
# and is consistent with the 2:1 split in UGPA3's corporate-action history. This is
# strictly a vendor price-scale repair; dividends/JCP are never added.
KNOWN_SCALE_REPAIRS: dict[str, tuple[ScaleRepair, ...]] = {
    "UGPA3": (
        ScaleRepair(
            boundary="2021-06-28",
            pre_boundary_price_factor=2.0,
            reason="Yahoo pre-2021-06-28 history is on a half-price scale; no split occurred at boundary",
        ),
    ),
}


def apply_known_scale_repairs(frame: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    repaired = frame.copy()
    repaired["date"] = pd.to_datetime(repaired["date"], errors="raise")
    applied: list[dict[str, object]] = []

    for repair in KNOWN_SCALE_REPAIRS.get(ticker, ()):
        boundary = pd.Timestamp(repair.boundary)
        mask = repaired["date"] < boundary
        if not mask.any():
            continue
        repaired.loc[mask, OHLC] = repaired.loc[mask, OHLC].astype(float) * repair.pre_boundary_price_factor
        if "volume" in repaired.columns:
            repaired.loc[mask, "volume"] = (
                repaired.loc[mask, "volume"].astype(float) / repair.pre_boundary_price_factor
            ).round()
        applied.append(
            {
                "ticker": ticker,
                "boundary": repair.boundary,
                "pre_boundary_price_factor": repair.pre_boundary_price_factor,
                "reason": repair.reason,
            }
        )

    return repaired, applied


def extreme_discontinuities(
    frame: pd.DataFrame,
    *,
    upper_ratio: float = 2.0,
    lower_ratio: float = 0.5,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    previous_close = frame["close"].shift(1)
    open_ratio = frame["open"] / previous_close
    close_ratio = frame["close"] / previous_close
    flagged = previous_close.notna() & (
        (open_ratio >= upper_ratio)
        | (open_ratio <= lower_ratio)
        | (close_ratio >= upper_ratio)
        | (close_ratio <= lower_ratio)
    )
    result = frame.loc[flagged, ["date", "open", "close"]].copy()
    if result.empty:
        return result
    result["previous_close"] = previous_close.loc[flagged].to_numpy()
    result["open_to_prev_close_ratio"] = open_ratio.loc[flagged].to_numpy()
    result["close_to_prev_close_ratio"] = close_ratio.loc[flagged].to_numpy()
    return result


def assert_trusted_window(frame: pd.DataFrame, ticker: str) -> None:
    anomalies = extreme_discontinuities(frame)
    if not anomalies.empty:
        rows = anomalies.head(5).to_dict("records")
        raise ValueError(f"{ticker}: extreme discontinuities remain in trusted backtest window: {rows}")
