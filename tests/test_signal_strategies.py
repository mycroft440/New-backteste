from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.strategies import (  # noqa: E402
    generate_events,
    list_original_strategies,
    list_strategies,
    run_strategy,
)


class SignalStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.read_csv(ROOT / "data" / "quotes" / "PETR4.csv")

    def test_original_benchmark_has_exactly_40_unique_strategies(self) -> None:
        specs = list_original_strategies()
        self.assertEqual(len(specs), 40)
        self.assertEqual(len({spec.name for spec in specs}), 40)
        self.assertEqual(
            {spec.engine for spec in specs},
            {"moving_average_cross", "macd", "donchian", "rsi_reversion"},
        )

    def test_full_catalog_is_unique_and_contains_original_benchmark(self) -> None:
        original = list_original_strategies()
        full = list_strategies()
        self.assertGreaterEqual(len(full), len(original))
        self.assertEqual(len({spec.name for spec in full}), len(full))
        self.assertTrue({spec.name for spec in original}.issubset({spec.name for spec in full}))

    def test_all_original_40_strategies_return_binary_position_state(self) -> None:
        for spec in list_original_strategies():
            with self.subTest(strategy=spec.name):
                positions = run_strategy(self.frame, spec.name)
                self.assertEqual(len(positions), len(self.frame))
                self.assertTrue(set(positions).issubset({0, 1}))

    def test_event_layer_contains_buy_sell_signals_only(self) -> None:
        events = generate_events(self.frame, "donchian_breakout_20_10")
        self.assertListEqual(
            list(events.columns),
            ["date", "close", "event", "strategy"],
        )
        self.assertGreater(len(events), 0)
        self.assertTrue(set(events["event"]).issubset({"BUY", "SELL"}))
        self.assertEqual(events.iloc[0]["event"], "BUY")
        for previous, current in zip(events["event"], events["event"].iloc[1:]):
            self.assertNotEqual(previous, current)

    def test_market_data_remains_exactly_40_quote_files(self) -> None:
        files = sorted((ROOT / "data" / "quotes").glob("*.csv"))
        self.assertEqual(len(files), 40)

    def test_signal_package_has_no_portfolio_management_module(self) -> None:
        strategy_dir = ROOT / "src" / "b3_backtest" / "strategies"
        names = {path.name.lower() for path in strategy_dir.glob("*.py")}
        self.assertFalse(any("portfolio" in name for name in names))
        self.assertFalse(any("allocation" in name for name in names))
        self.assertFalse(any("rebalance" in name for name in names))


if __name__ == "__main__":
    unittest.main()
