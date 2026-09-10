from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import pandas as pd

SOURCE_REPOSITORY = "mycroft440/b3-strategy-lab"
SOURCE_RESULT_COMMIT = "bb2533bacaadecbef102869a0fe89407d2eddf22"
SOURCE_RESULT_REPORT = "reports/portfolio_allocation_research_raw_events_raw_1d_roc_filter_short_fine_sweep.csv"
SOURCE_CODE_COMMIT = "8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d"
SOURCE_CODE_MODULE = "scripts/research_portfolio_allocation_core.py"
SOURCE_CONFIG = "top1_short22_riskadj_rocfilter_w1_1_1_trend200_vol18_posscore"

ROC_WINDOWS = (252, 126, 63)
ROC_WEIGHTS = (1.0, 1.0, 1.0)
SHORT_WINDOW = 22
TREND_WINDOW = 200
VOL_WINDOW = 18


@dataclass(frozen=True)
class ChampionIndicatorSnapshot:
    decision_date: pd.Timestamp
    ticker: str
    close: float | None
    roc_252: float | None
    roc_126: float | None
    roc_63: float | None
    composite_roc: float | None
    roc_22: float | None
    sma_200: float | None
    annualized_volatility_18: float | None
    score: float | None
    eligible: bool
    reason: str

    def as_record(self) -> dict[str, object]:
        return {
            "decision_date": self.decision_date,
            "ticker": self.ticker,
            "close": self.close,
            "roc_252_pct": None if self.roc_252 is None else self.roc_252 * 100.0,
            "roc_126_pct": None if self.roc_126 is None else self.roc_126 * 100.0,
            "roc_63_pct": None if self.roc_63 is None else self.roc_63 * 100.0,
            "composite_roc_pct": None if self.composite_roc is None else self.composite_roc * 100.0,
            "roc_22_pct": None if self.roc_22 is None else self.roc_22 * 100.0,
            "sma_200": self.sma_200,
            "volatility_18_annualized_pct": (
                None if self.annualized_volatility_18 is None else self.annualized_volatility_18 * 100.0
            ),
            "score": self.score,
            "eligible_for_ranking": self.eligible,
            "reason": self.reason,
        }


def _aligned_index(frame: pd.DataFrame, decision_date: pd.Timestamp) -> int | None:
    dates = pd.to_datetime(frame["date"], errors="raise")
    matches = [index for index, value in enumerate(dates) if pd.Timestamp(value) == decision_date]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"duplicate date for {decision_date.date()}")
    return matches[0]


def _annualized_volatility(closes: pd.Series, index: int, window: int = VOL_WINDOW) -> float:
    if index < window:
        return 0.0
    values = closes.iloc[index - window : index + 1].astype(float)
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    value = float(returns.std(ddof=1) * math.sqrt(252.0))
    return value if math.isfinite(value) else 0.0


def evaluate_champion_indicator(
    frame: pd.DataFrame,
    decision_date: pd.Timestamp | str,
    *,
    ticker: str = "",
) -> ChampionIndicatorSnapshot:
    """Evaluate the exact cross-sectional signal used by the B3 Strategy Lab winner.

    Source semantics, adapted to this repository's split-only close series:
    - arithmetic mean of ROC(252), ROC(126), ROC(63) must be positive;
    - ROC(22) must be positive;
    - close must be strictly above SMA(200);
    - score = ROC(22) / annualized sample volatility of the last 18 returns.

    The 12/6/3-month composite is a filter, not the final ranking score. With
    ``positive_rule=score`` the individual ROC components do not each need to be
    positive; only their equally weighted composite must be positive.
    """
    date = pd.Timestamp(decision_date)
    index = _aligned_index(frame, date)
    if index is None:
        return ChampionIndicatorSnapshot(date, ticker, None, None, None, None, None, None, None, None, None, False,
                                         "missing_candle")

    closes = frame["close"].astype(float)
    close = float(closes.iloc[index])
    if not math.isfinite(close) or close <= 0:
        return ChampionIndicatorSnapshot(date, ticker, close, None, None, None, None, None, None, None, None, False,
                                         "invalid_price")

    longest = max(ROC_WINDOWS)
    if index < longest or index + 1 < TREND_WINDOW or index < VOL_WINDOW or index < SHORT_WINDOW:
        return ChampionIndicatorSnapshot(date, ticker, close, None, None, None, None, None, None, None, None, False,
                                         "insufficient_history")

    rocs: list[float] = []
    for window in ROC_WINDOWS:
        past = float(closes.iloc[index - window])
        if not math.isfinite(past) or past <= 0:
            return ChampionIndicatorSnapshot(date, ticker, close, None, None, None, None, None, None, None, None,
                                             False, "invalid_price")
        rocs.append(close / past - 1.0)

    weighted_sum = sum(roc * weight for roc, weight in zip(rocs, ROC_WEIGHTS))
    total_weight = sum(abs(weight) for weight in ROC_WEIGHTS)
    composite = weighted_sum / total_weight
    roc_252, roc_126, roc_63 = rocs
    if composite <= 0:
        return ChampionIndicatorSnapshot(date, ticker, close, roc_252, roc_126, roc_63, composite, None, None, None,
                                         None, False, "nonpositive_composite_roc")

    short_past = float(closes.iloc[index - SHORT_WINDOW])
    if not math.isfinite(short_past) or short_past <= 0:
        return ChampionIndicatorSnapshot(date, ticker, close, roc_252, roc_126, roc_63, composite, None, None, None,
                                         None, False, "invalid_price")
    roc_22 = close / short_past - 1.0
    if roc_22 <= 0:
        return ChampionIndicatorSnapshot(date, ticker, close, roc_252, roc_126, roc_63, composite, roc_22, None,
                                         None, None, False, "nonpositive_roc22")

    trend_values = closes.iloc[index - TREND_WINDOW + 1 : index + 1]
    sma_200 = float(trend_values.mean())
    if close <= sma_200:
        return ChampionIndicatorSnapshot(date, ticker, close, roc_252, roc_126, roc_63, composite, roc_22, sma_200,
                                         None, None, False, "below_or_at_sma200")

    volatility = _annualized_volatility(closes, index)
    if volatility <= 0:
        return ChampionIndicatorSnapshot(date, ticker, close, roc_252, roc_126, roc_63, composite, roc_22, sma_200,
                                         volatility, None, False, "zero_volatility")

    score = roc_22 / volatility
    return ChampionIndicatorSnapshot(date, ticker, close, roc_252, roc_126, roc_63, composite, roc_22, sma_200,
                                     volatility, score, True, "eligible")


def evaluate_universe(
    frames: Mapping[str, pd.DataFrame],
    decision_date: pd.Timestamp | str,
) -> pd.DataFrame:
    records = [
        evaluate_champion_indicator(frame, decision_date, ticker=ticker).as_record()
        for ticker, frame in sorted(frames.items())
    ]
    return pd.DataFrame(records)
