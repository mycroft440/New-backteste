from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.strategies import (  # noqa: E402
    list_original_strategies,
    list_strategies,
    list_strategy_batches,
    run_strategy,
)


class FullStrategyCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_path = SRC / "b3_backtest" / "strategies" / "strategy_catalog_index.json"
        if not cls.index_path.exists():
            raise unittest.SkipTest("full vendored strategy catalog has not been materialized yet")
        cls.index = json.loads(cls.index_path.read_text(encoding="utf-8"))
        cls.frame = pd.read_csv(ROOT / "data" / "quotes" / "PETR4.csv").tail(700).reset_index(drop=True)

    def test_catalog_expands_beyond_additional_module_only(self) -> None:
        self.assertEqual(len(list_original_strategies()), 40)
        self.assertGreater(self.index["vendored_strategy_count"], 90)
        self.assertGreater(self.index["total_strategy_count"], 130)
        self.assertEqual(len(list_strategies()), self.index["total_strategy_count"])

    def test_names_are_unique_and_batches_are_bounded(self) -> None:
        specs = list_strategies()
        self.assertEqual(len(specs), len({spec.name for spec in specs}))
        batches = list_strategy_batches()
        for name, values in batches.items():
            with self.subTest(batch=name):
                if name == "original_001":
                    self.assertEqual(len(values), 40)
                else:
                    self.assertLessEqual(len(values), 40)
                    self.assertGreater(len(values), 0)

    def test_index_batch_files_match_catalog(self) -> None:
        batch_dir = SRC / "b3_backtest" / "strategies" / "catalog_batches"
        indexed_files = {item["file"] for item in self.index["batches"]}
        actual_files = {path.name for path in batch_dir.glob("*.json")}
        self.assertEqual(actual_files, indexed_files)
        self.assertTrue(all(int(item["count"]) <= 40 for item in self.index["batches"]))

    def test_every_vendored_strategy_returns_binary_state(self) -> None:
        vendored = [spec for spec in list_strategies() if spec.engine == "vendored"]
        self.assertEqual(len(vendored), self.index["vendored_strategy_count"])
        for spec in vendored:
            with self.subTest(strategy=spec.name):
                positions = run_strategy(self.frame, spec.name)
                self.assertEqual(len(positions), len(self.frame))
                self.assertTrue(set(positions).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
