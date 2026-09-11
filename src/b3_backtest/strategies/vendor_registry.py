from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from .core import validate_ohlcv

VENDOR_PACKAGE = "b3_backtest.strategies.vendor.b3_strategy_lab"
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "b3_strategy_lab"
SOURCE_MODULES = (
    "strategies",
    "additional_strategies",
    "extended_strategies",
    "indicator_strategies",
    "researched_strategies",
    "smi_ergodic_strategies",
    "trend_strategies",
)


@dataclass(frozen=True)
class VendoredStrategy:
    name: str
    family: str
    description: str
    source_module: str
    function: Callable[[list[object]], list[int]]


def vendor_available() -> bool:
    return (VENDOR_ROOT / "strategies.py").exists()


def _iter_strategy_objects(value: object) -> Iterable[object]:
    if isinstance(value, dict):
        yield from value.values()
    elif isinstance(value, (list, tuple, set)):
        yield from value


def _looks_like_strategy(value: object) -> bool:
    return (
        isinstance(getattr(value, "name", None), str)
        and isinstance(getattr(value, "family", None), str)
        and callable(getattr(value, "function", None))
    )


def _register(
    registry: dict[str, VendoredStrategy],
    *,
    name: str,
    family: str,
    description: str,
    source_module: str,
    function: Callable[[list[object]], list[int]],
) -> None:
    normalized = name.strip().lower()
    if not normalized:
        return
    candidate = VendoredStrategy(
        normalized,
        family.strip() or "outros",
        description.strip() or normalized,
        source_module,
        function,
    )
    current = registry.get(normalized)
    if current is None:
        registry[normalized] = candidate
        return
    # Duplicate registrations are common in the source tree. Keep the first
    # stable definition; batch generation records only unique strategy names.


@lru_cache(maxsize=1)
def discover_vendored_strategies() -> tuple[VendoredStrategy, ...]:
    """Discover all stock signal strategies exported by the pinned source package."""
    if not vendor_available():
        return ()

    modules = [importlib.import_module(f"{VENDOR_PACKAGE}.{name}") for name in SOURCE_MODULES]
    registry: dict[str, VendoredStrategy] = {}

    base = modules[0]
    base_map = getattr(base, "STRATEGIES", {})
    info_map = getattr(base, "STRATEGY_INFO", {})
    if isinstance(base_map, dict):
        for name, function in base_map.items():
            if not callable(function):
                continue
            info = info_map.get(name) if isinstance(info_map, dict) else None
            _register(
                registry,
                name=str(name),
                family=str(getattr(info, "family", "base")),
                description=str(getattr(info, "description", f"Estrategia base {name}.")),
                source_module="strategies.py",
                function=function,
            )

    for module in modules[1:]:
        short_module = module.__name__.rsplit(".", 1)[-1] + ".py"
        for value in vars(module).values():
            if _looks_like_strategy(value):
                objects = (value,)
            else:
                objects = tuple(_iter_strategy_objects(value))
            for item in objects:
                if not _looks_like_strategy(item):
                    continue
                function = getattr(item, "function")
                origin = getattr(function, "__module__", module.__name__).rsplit(".", 1)[-1] + ".py"
                _register(
                    registry,
                    name=str(getattr(item, "name")),
                    family=str(getattr(item, "family")),
                    description=str(getattr(item, "description", getattr(item, "name"))),
                    source_module=origin or short_module,
                    function=function,
                )

    extensions = importlib.import_module(f"{VENDOR_PACKAGE}.extensions")
    registered = getattr(extensions, "registered_strategies", None)
    if callable(registered):
        for item in registered():
            if not _looks_like_strategy(item):
                continue
            function = getattr(item, "function")
            origin = getattr(function, "__module__", "extensions").rsplit(".", 1)[-1] + ".py"
            _register(
                registry,
                name=str(getattr(item, "name")),
                family=str(getattr(item, "family")),
                description=str(getattr(item, "description", getattr(item, "name"))),
                source_module=origin,
                function=function,
            )

    return tuple(sorted(registry.values(), key=lambda item: (item.family, item.name)))


def _frame_to_candles(frame: pd.DataFrame, ticker: str = "TICKER") -> list[object]:
    validate_ohlcv(frame)
    candle_module = importlib.import_module(f"{VENDOR_PACKAGE}.candles")
    Candle = getattr(candle_module, "Candle")
    dates = (
        pd.to_datetime(frame["date"], errors="raise")
        if "date" in frame.columns
        else pd.to_datetime(frame.index, errors="raise")
    )
    candles: list[object] = []
    for index, row in frame.reset_index(drop=True).iterrows():
        open_price = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        volume = int(float(row["volume"]))
        date_value = dates.iloc[index] if hasattr(dates, "iloc") else dates[index]
        candles.append(
            Candle(
                date=pd.Timestamp(date_value).date().isoformat(),
                ticker=ticker,
                source_symbol=ticker,
                open=open_price,
                high=high,
                low=low,
                close=close,
                adj_close=close,
                volume=volume,
                raw_open=open_price,
                raw_high=high,
                raw_low=low,
                raw_close=close,
                adjustment_factor=1.0,
            )
        )
    return candles


def run_vendored_strategy(frame: pd.DataFrame, name: str) -> list[int]:
    registry = {item.name: item for item in discover_vendored_strategies()}
    normalized = name.strip().lower()
    try:
        strategy = registry[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown vendored strategy: {name}") from exc

    ticker = "TICKER"
    if "ticker" in frame.columns and not frame.empty:
        ticker = str(frame.iloc[0]["ticker"])
    candles = _frame_to_candles(frame, ticker=ticker)
    positions = list(strategy.function(candles))
    if len(positions) != len(candles):
        raise RuntimeError(f"{name}: strategy returned {len(positions)} states for {len(candles)} candles")

    normalized_positions: list[int] = []
    for index, value in enumerate(positions):
        try:
            is_zero = bool(value == 0)
            is_one = bool(value == 1)
        except Exception as exc:
            raise RuntimeError(
                f"{name}: strategy returned non-scalar state at index {index}: {value!r}"
            ) from exc
        if not (is_zero or is_one):
            raise RuntimeError(
                f"{name}: strategy returned non-binary state at index {index}: {value!r}"
            )
        normalized_positions.append(1 if is_one else 0)
    return normalized_positions
