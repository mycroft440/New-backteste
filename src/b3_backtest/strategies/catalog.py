from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .core import run_engine, signal_events

SOURCE_REPOSITORY = "mycroft440/b3-strategy-lab"
SOURCE_COMMIT = "8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d"
SOURCE_MODULE = "b3_strategy_lab/additional_strategies.py"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    description: str
    engine: str
    parameters: dict[str, object]


def _spec(
    name: str,
    family: str,
    description: str,
    engine: str,
    **parameters: object,
) -> StrategySpec:
    return StrategySpec(name, family, description, engine, parameters)


SPECS: tuple[StrategySpec, ...] = (
    # Moving-average crosses: source-faithful variants from b3-strategy-lab.
    _spec("sma_cross_5_20", "tendencia", "Cruzamento de medias SMA 5/20.", "moving_average_cross", average_type="sma", fast=5, slow=20),
    _spec("sma_cross_10_50", "tendencia", "Cruzamento de medias SMA 10/50.", "moving_average_cross", average_type="sma", fast=10, slow=50),
    _spec("sma_cross_20_100", "tendencia", "Cruzamento de medias SMA 20/100.", "moving_average_cross", average_type="sma", fast=20, slow=100),
    _spec("sma_cross_50_100", "tendencia", "Cruzamento de medias SMA 50/100.", "moving_average_cross", average_type="sma", fast=50, slow=100),
    _spec("sma_cross_100_200", "tendencia", "Cruzamento de medias SMA 100/200.", "moving_average_cross", average_type="sma", fast=100, slow=200),
    _spec("ema_cross_5_20", "tendencia", "Cruzamento de medias EMA 5/20.", "moving_average_cross", average_type="ema", fast=5, slow=20),
    _spec("ema_cross_10_30", "tendencia", "Cruzamento de medias EMA 10/30.", "moving_average_cross", average_type="ema", fast=10, slow=30),
    _spec("ema_cross_20_50", "tendencia", "Cruzamento de medias EMA 20/50.", "moving_average_cross", average_type="ema", fast=20, slow=50),
    _spec("ema_cross_50_100", "tendencia", "Cruzamento de medias EMA 50/100.", "moving_average_cross", average_type="ema", fast=50, slow=100),
    _spec("ema_cross_100_200", "tendencia", "Cruzamento de medias EMA 100/200.", "moving_average_cross", average_type="ema", fast=100, slow=200),

    # MACD variants.
    _spec("macd_5_35_5", "tendencia", "MACD 5/35/5.", "macd", fast=5, slow=35, signal_window=5, trend_window=0),
    _spec("macd_8_17_9", "tendencia", "MACD 8/17/9.", "macd", fast=8, slow=17, signal_window=9, trend_window=0),
    _spec("macd_10_30_9", "tendencia", "MACD 10/30/9.", "macd", fast=10, slow=30, signal_window=9, trend_window=0),
    _spec("macd_12_26_5", "tendencia", "MACD 12/26/5.", "macd", fast=12, slow=26, signal_window=5, trend_window=0),
    _spec("macd_12_26_12", "tendencia", "MACD 12/26/12.", "macd", fast=12, slow=26, signal_window=12, trend_window=0),
    _spec("macd_19_39_9", "tendencia", "MACD 19/39/9.", "macd", fast=19, slow=39, signal_window=9, trend_window=0),
    _spec("macd_24_52_18", "tendencia", "MACD 24/52/18.", "macd", fast=24, slow=52, signal_window=18, trend_window=0),
    _spec("macd_12_26_9_trend50", "tendencia", "MACD 12/26/9 com filtro SMA 50.", "macd", fast=12, slow=26, signal_window=9, trend_window=50),
    _spec("macd_12_26_9_trend100", "tendencia", "MACD 12/26/9 com filtro SMA 100.", "macd", fast=12, slow=26, signal_window=9, trend_window=100),
    _spec("macd_12_26_9_trend200", "tendencia", "MACD 12/26/9 com filtro SMA 200.", "macd", fast=12, slow=26, signal_window=9, trend_window=200),

    # Donchian breakout variants.
    _spec("donchian_breakout_10_5", "rompimento", "Donchian 10/5.", "donchian", entry_window=10, exit_window=5, trend_window=0),
    _spec("donchian_breakout_20_10", "rompimento", "Donchian 20/10.", "donchian", entry_window=20, exit_window=10, trend_window=0),
    _spec("donchian_breakout_40_20", "rompimento", "Donchian 40/20.", "donchian", entry_window=40, exit_window=20, trend_window=0),
    _spec("donchian_breakout_55_20", "rompimento", "Donchian 55/20.", "donchian", entry_window=55, exit_window=20, trend_window=0),
    _spec("donchian_breakout_100_50", "rompimento", "Donchian 100/50.", "donchian", entry_window=100, exit_window=50, trend_window=0),
    _spec("donchian_breakout_20_10_trend50", "rompimento", "Donchian 20/10 com filtro SMA 50.", "donchian", entry_window=20, exit_window=10, trend_window=50),
    _spec("donchian_breakout_20_10_trend100", "rompimento", "Donchian 20/10 com filtro SMA 100.", "donchian", entry_window=20, exit_window=10, trend_window=100),
    _spec("donchian_breakout_55_20_trend100", "rompimento", "Donchian 55/20 com filtro SMA 100.", "donchian", entry_window=55, exit_window=20, trend_window=100),
    _spec("donchian_breakout_55_20_trend200", "rompimento", "Donchian 55/20 com filtro SMA 200.", "donchian", entry_window=55, exit_window=20, trend_window=200),
    _spec("donchian_breakout_100_50_trend200", "rompimento", "Donchian 100/50 com filtro SMA 200.", "donchian", entry_window=100, exit_window=50, trend_window=200),

    # RSI mean-reversion variants.
    _spec("rsi2_reversion_5_70", "reversao", "Reversao RSI(2): 5/70, max 10 pregoes.", "rsi_reversion", rsi_period=2, lower=5.0, upper=70.0, trend_window=0, max_hold=10),
    _spec("rsi2_reversion_10_70", "reversao", "Reversao RSI(2): 10/70, max 10 pregoes.", "rsi_reversion", rsi_period=2, lower=10.0, upper=70.0, trend_window=0, max_hold=10),
    _spec("rsi3_reversion_15_70", "reversao", "Reversao RSI(3): 15/70, max 15 pregoes.", "rsi_reversion", rsi_period=3, lower=15.0, upper=70.0, trend_window=0, max_hold=15),
    _spec("rsi5_reversion_20_65", "reversao", "Reversao RSI(5): 20/65, max 20 pregoes.", "rsi_reversion", rsi_period=5, lower=20.0, upper=65.0, trend_window=0, max_hold=20),
    _spec("rsi7_reversion_25_65", "reversao", "Reversao RSI(7): 25/65, max 20 pregoes.", "rsi_reversion", rsi_period=7, lower=25.0, upper=65.0, trend_window=0, max_hold=20),
    _spec("rsi14_reversion_25_60", "reversao", "Reversao RSI(14): 25/60, max 30 pregoes.", "rsi_reversion", rsi_period=14, lower=25.0, upper=60.0, trend_window=0, max_hold=30),
    _spec("rsi2_reversion_5_70_trend100", "reversao", "RSI(2) 5/70 com filtro SMA 100.", "rsi_reversion", rsi_period=2, lower=5.0, upper=70.0, trend_window=100, max_hold=10),
    _spec("rsi2_reversion_5_70_trend200", "reversao", "RSI(2) 5/70 com filtro SMA 200.", "rsi_reversion", rsi_period=2, lower=5.0, upper=70.0, trend_window=200, max_hold=10),
    _spec("rsi5_reversion_20_65_trend100", "reversao", "RSI(5) 20/65 com filtro SMA 100.", "rsi_reversion", rsi_period=5, lower=20.0, upper=65.0, trend_window=100, max_hold=20),
    _spec("rsi14_reversion_30_70_trend200", "reversao", "RSI(14) 30/70 com filtro SMA 200.", "rsi_reversion", rsi_period=14, lower=30.0, upper=70.0, trend_window=200, max_hold=30),
)

if len(SPECS) != 40:
    raise RuntimeError(f"invalid initial catalog size: {len(SPECS)}; expected 40")
if len({spec.name for spec in SPECS}) != len(SPECS):
    raise RuntimeError("initial strategy catalog contains duplicate names")

CATALOG: dict[str, StrategySpec] = {spec.name: spec for spec in SPECS}


def list_strategies() -> tuple[StrategySpec, ...]:
    return SPECS


def get_strategy(name: str) -> StrategySpec:
    try:
        return CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"unknown strategy: {name}") from exc


def run_strategy(frame: pd.DataFrame, name: str) -> list[int]:
    spec = get_strategy(name)
    return run_engine(frame, spec.engine, spec.parameters)


def generate_events(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    positions = run_strategy(frame, name)
    return signal_events(frame, positions, strategy_name=name)
