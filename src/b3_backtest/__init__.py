"""B3 signal-strategy laboratory.

This first stage intentionally contains only market-data validation and BUY/SELL
signal generation. Portfolio sizing, allocation and rebalancing are out of scope.
"""

from .strategies import generate_events, get_strategy, list_strategies, run_strategy

__all__ = ["generate_events", "get_strategy", "list_strategies", "run_strategy"]
