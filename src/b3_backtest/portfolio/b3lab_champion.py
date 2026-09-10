from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from b3_backtest.indicators.b3lab_champion import (
    SOURCE_CODE_COMMIT,
    SOURCE_CODE_MODULE,
    SOURCE_CONFIG,
    SOURCE_REPOSITORY,
    SOURCE_RESULT_COMMIT,
    SOURCE_RESULT_REPORT,
    evaluate_universe,
)

CHAMPION_MANAGEMENT_STRATEGY_NAME = SOURCE_CONFIG
CHAMPION_INDICATOR_NAME = "roc252_126_63_filter__roc22_over_vol18__sma200"
SOURCE_FULL_RETURN = 187.20552082879905
SOURCE_FULL_CAGR = 0.21820719045957193
SOURCE_FULL_MAX_DRAWDOWN = -0.6457170553521918
SOURCE_FULL_SHARPE = 0.7839165097900773
SOURCE_TEST_RETURN = 11.934350255545647
SOURCE_COST_BPS = 3.2
SOURCE_SLIPPAGE_BPS = 10.0


@dataclass(frozen=True)
class ChampionSelection:
    decision_date: pd.Timestamp
    selected_ticker: str | None
    candidates: pd.DataFrame


def select_b3lab_champion(
    frames: Mapping[str, pd.DataFrame],
    decision_date: pd.Timestamp | str,
) -> ChampionSelection:
    """Select the source-faithful Top-1 winner at a decision close.

    The indicator itself supplies both eligibility and ranking. This winning B3
    Strategy Lab configuration does not depend on one of the binary MACD/RSI/etc.
    strategy states used by the legacy manager in this repository.
    """
    decision_date = pd.Timestamp(decision_date)
    result = evaluate_universe(frames, decision_date)
    result["rank"] = pd.array([None] * len(result), dtype="Int64")
    result["selected"] = False
    result["target_weight"] = 0.0

    eligible_indices = result.index[result["eligible_for_ranking"]].tolist()
    eligible_indices.sort(
        key=lambda index: (-float(result.at[index, "score"]), str(result.at[index, "ticker"]))
    )
    for rank, index in enumerate(eligible_indices, start=1):
        result.at[index, "rank"] = rank

    selected_ticker: str | None = None
    if eligible_indices:
        winner = eligible_indices[0]
        selected_ticker = str(result.at[winner, "ticker"])
        result.at[winner, "selected"] = True
        result.at[winner, "target_weight"] = 1.0

    return ChampionSelection(decision_date, selected_ticker, result)


def champion_source_metadata() -> dict[str, object]:
    return {
        "repository": SOURCE_REPOSITORY,
        "result_commit": SOURCE_RESULT_COMMIT,
        "result_report": SOURCE_RESULT_REPORT,
        "code_commit": SOURCE_CODE_COMMIT,
        "code_module": SOURCE_CODE_MODULE,
        "source_config": SOURCE_CONFIG,
        "source_full_return": SOURCE_FULL_RETURN,
        "source_full_cagr": SOURCE_FULL_CAGR,
        "source_full_max_drawdown": SOURCE_FULL_MAX_DRAWDOWN,
        "source_full_sharpe": SOURCE_FULL_SHARPE,
        "source_test_return": SOURCE_TEST_RETURN,
        "source_cost_bps": SOURCE_COST_BPS,
        "source_slippage_bps": SOURCE_SLIPPAGE_BPS,
    }
