from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from functools import partial
from typing import Callable

from .candles import Candle


SignalFunction = Callable[[list[Candle]], list[int]]


@dataclass(frozen=True)
class AdditionalStrategy:
    name: str
    family: str
    description: str
    engine: str
    parameters: dict[str, object]
    function: SignalFunction


def _definition(
    name: str,
    family: str,
    description: str,
    engine: str,
    **parameters: object,
) -> AdditionalStrategy:
    function = partial(_run, engine=engine, parameters=parameters)
    function.__name__ = name
    function.__doc__ = description
    return AdditionalStrategy(name, family, description, engine, parameters, function)


def _build_definitions() -> list[AdditionalStrategy]:
    definitions: list[AdditionalStrategy] = []

    momentum_setups = [
        ("time_series_momentum_3m", 63, 0, 0),
        ("time_series_momentum_6m", 126, 0, 0),
        ("time_series_momentum_9m", 189, 0, 0),
        ("time_series_momentum_12m", 252, 21, 0),
        ("time_series_momentum_12m_no_skip", 252, 0, 0),
        ("time_series_momentum_6m_trend100", 126, 0, 100),
        ("time_series_momentum_6m_trend200", 126, 0, 200),
        ("time_series_momentum_12m_trend100", 252, 21, 100),
        ("time_series_momentum_12m_trend200", 252, 21, 200),
        ("time_series_momentum_18m_skip1m", 378, 21, 200),
    ]
    for name, lookback, skip, trend_window in momentum_setups:
        definitions.append(
            _definition(
                name,
                "momentum",
                "Momentum absoluto; compara o fechamento defasado com o passado e aplica filtro de tendencia opcional.",
                "time_series_momentum",
                lookback=lookback,
                skip=skip,
                trend_window=trend_window,
            )
        )

    moving_average_setups = [
        ("sma_cross_5_20", "sma", 5, 20),
        ("sma_cross_10_50", "sma", 10, 50),
        ("sma_cross_20_100", "sma", 20, 100),
        ("sma_cross_50_100", "sma", 50, 100),
        ("sma_cross_100_200", "sma", 100, 200),
        ("ema_cross_5_20", "ema", 5, 20),
        ("ema_cross_10_30", "ema", 10, 30),
        ("ema_cross_20_50", "ema", 20, 50),
        ("ema_cross_50_100", "ema", 50, 100),
        ("ema_cross_100_200", "ema", 100, 200),
    ]
    for name, average_type, fast, slow in moving_average_setups:
        definitions.append(
            _definition(
                name,
                "tendencia",
                f"Cruzamento de medias {average_type.upper()} {fast}/{slow}.",
                "moving_average_cross",
                average_type=average_type,
                fast=fast,
                slow=slow,
            )
        )

    macd_setups = [
        ("macd_5_35_5", 5, 35, 5, 0),
        ("macd_8_17_9", 8, 17, 9, 0),
        ("macd_10_30_9", 10, 30, 9, 0),
        ("macd_12_26_5", 12, 26, 5, 0),
        ("macd_12_26_12", 12, 26, 12, 0),
        ("macd_19_39_9", 19, 39, 9, 0),
        ("macd_24_52_18", 24, 52, 18, 0),
        ("macd_12_26_9_trend50", 12, 26, 9, 50),
        ("macd_12_26_9_trend100", 12, 26, 9, 100),
        ("macd_12_26_9_trend200", 12, 26, 9, 200),
    ]
    for name, fast, slow, signal_window, trend_window in macd_setups:
        definitions.append(
            _definition(
                name,
                "tendencia",
                "MACD parametrizado com confirmacao de tendencia opcional.",
                "macd",
                fast=fast,
                slow=slow,
                signal_window=signal_window,
                trend_window=trend_window,
            )
        )

    breakout_setups = [
        ("donchian_breakout_10_5", 10, 5, 0),
        ("donchian_breakout_20_10", 20, 10, 0),
        ("donchian_breakout_40_20", 40, 20, 0),
        ("donchian_breakout_55_20", 55, 20, 0),
        ("donchian_breakout_100_50", 100, 50, 0),
        ("donchian_breakout_20_10_trend50", 20, 10, 50),
        ("donchian_breakout_20_10_trend100", 20, 10, 100),
        ("donchian_breakout_55_20_trend100", 55, 20, 100),
        ("donchian_breakout_55_20_trend200", 55, 20, 200),
        ("donchian_breakout_100_50_trend200", 100, 50, 200),
    ]
    for name, entry_window, exit_window, trend_window in breakout_setups:
        definitions.append(
            _definition(
                name,
                "rompimento",
                "Canal de Donchian com maxima anterior para entrada e minima anterior para saida.",
                "donchian",
                entry_window=entry_window,
                exit_window=exit_window,
                trend_window=trend_window,
            )
        )

    atr_setups = [
        ("atr_breakout_10_atr7_x2", 10, 7, 2.0, 0),
        ("atr_breakout_20_atr10_x2", 20, 10, 2.0, 0),
        ("atr_breakout_20_atr14_x25", 20, 14, 2.5, 0),
        ("atr_breakout_20_atr14_x4", 20, 14, 4.0, 0),
        ("atr_breakout_55_atr14_x3", 55, 14, 3.0, 0),
        ("atr_breakout_55_atr21_x4", 55, 21, 4.0, 0),
        ("atr_breakout_20_atr14_x3_trend50", 20, 14, 3.0, 50),
        ("atr_breakout_20_atr14_x3_trend100", 20, 14, 3.0, 100),
        ("atr_breakout_55_atr14_x3_trend100", 55, 14, 3.0, 100),
        ("atr_breakout_55_atr21_x4_trend200", 55, 21, 4.0, 200),
    ]
    for name, lookback, atr_period, atr_mult, trend_window in atr_setups:
        definitions.append(
            _definition(
                name,
                "rompimento",
                "Rompimento de maxima com stop movel calculado por ATR.",
                "atr_breakout",
                lookback=lookback,
                atr_period=atr_period,
                atr_mult=atr_mult,
                trend_window=trend_window,
            )
        )

    rsi_setups = [
        ("rsi2_reversion_5_70", 2, 5.0, 70.0, 0, 10),
        ("rsi2_reversion_10_70", 2, 10.0, 70.0, 0, 10),
        ("rsi3_reversion_15_70", 3, 15.0, 70.0, 0, 15),
        ("rsi5_reversion_20_65", 5, 20.0, 65.0, 0, 20),
        ("rsi7_reversion_25_65", 7, 25.0, 65.0, 0, 20),
        ("rsi14_reversion_25_60", 14, 25.0, 60.0, 0, 30),
        ("rsi2_reversion_5_70_trend100", 2, 5.0, 70.0, 100, 10),
        ("rsi2_reversion_5_70_trend200", 2, 5.0, 70.0, 200, 10),
        ("rsi5_reversion_20_65_trend100", 5, 20.0, 65.0, 100, 20),
        ("rsi14_reversion_30_70_trend200", 14, 30.0, 70.0, 200, 30),
    ]
    for name, period, lower, upper, trend_window, max_hold in rsi_setups:
        definitions.append(
            _definition(
                name,
                "reversao",
                "Reversao comprada por RSI com filtro de tendencia e permanencia maxima opcionais.",
                "rsi_reversion",
                rsi_period=period,
                lower=lower,
                upper=upper,
                trend_window=trend_window,
                max_hold=max_hold,
            )
        )

    bollinger_setups = [
        ("rsi_bollinger_2_5_bb20_2", 2, 5.0, 70.0, 20, 2.0, 0),
        ("rsi_bollinger_3_10_bb20_2", 3, 10.0, 70.0, 20, 2.0, 0),
        ("rsi_bollinger_5_20_bb20_2", 5, 20.0, 65.0, 20, 2.0, 0),
        ("rsi_bollinger_14_30_bb20_2", 14, 30.0, 70.0, 20, 2.0, 0),
        ("rsi_bollinger_14_35_bb10_15", 14, 35.0, 65.0, 10, 1.5, 0),
        ("rsi_bollinger_14_30_bb30_25", 14, 30.0, 70.0, 30, 2.5, 0),
        ("rsi_bollinger_2_5_bb20_2_trend100", 2, 5.0, 70.0, 20, 2.0, 100),
        ("rsi_bollinger_2_5_bb20_2_trend200", 2, 5.0, 70.0, 20, 2.0, 200),
        ("rsi_bollinger_5_20_bb20_2_trend100", 5, 20.0, 65.0, 20, 2.0, 100),
        ("rsi_bollinger_14_30_bb20_2_trend200", 14, 30.0, 70.0, 20, 2.0, 200),
    ]
    for name, period, lower, upper, window, num_std, trend_window in bollinger_setups:
        definitions.append(
            _definition(
                name,
                "combinada",
                "RSI em sobrevenda confirmado pela banda inferior de Bollinger.",
                "rsi_bollinger",
                rsi_period=period,
                lower=lower,
                upper=upper,
                window=window,
                num_std=num_std,
                trend_window=trend_window,
            )
        )

    stochastic_setups = [
        ("sma20_stochastic_14_20_80", "sma", 20, 14, 20.0, 80.0),
        ("sma50_stochastic_14_20_80", "sma", 50, 14, 20.0, 80.0),
        ("sma100_stochastic_14_20_80", "sma", 100, 14, 20.0, 80.0),
        ("sma200_stochastic_14_20_80", "sma", 200, 14, 20.0, 80.0),
        ("ema20_stochastic_14_20_80", "ema", 20, 14, 20.0, 80.0),
        ("ema50_stochastic_14_20_80", "ema", 50, 14, 20.0, 80.0),
        ("ema100_stochastic_14_20_80", "ema", 100, 14, 20.0, 80.0),
        ("ema200_stochastic_14_20_80", "ema", 200, 14, 20.0, 80.0),
        ("ema50_stochastic_21_30_75", "ema", 50, 21, 30.0, 75.0),
        ("sma200_stochastic_21_30_75", "sma", 200, 21, 30.0, 75.0),
    ]
    for name, average_type, trend_window, k_period, lower, upper in stochastic_setups:
        definitions.append(
            _definition(
                name,
                "combinada",
                "Media movel define a tendencia; Estocastico define o pullback e a saida.",
                "stochastic_trend",
                average_type=average_type,
                trend_window=trend_window,
                k_period=k_period,
                lower=lower,
                upper=upper,
            )
        )

    supertrend_setups = [
        ("supertrend_stochastic_7_2_14", 7, 2.0, "stochastic", 14, 25.0, 80.0),
        ("supertrend_stochastic_10_2_14", 10, 2.0, "stochastic", 14, 25.0, 80.0),
        ("supertrend_stochastic_10_3_14", 10, 3.0, "stochastic", 14, 25.0, 80.0),
        ("supertrend_stochastic_14_3_21", 14, 3.0, "stochastic", 21, 30.0, 75.0),
        ("supertrend_stochastic_21_4_21", 21, 4.0, "stochastic", 21, 30.0, 75.0),
        ("supertrend_rsi_7_2_7", 7, 2.0, "rsi", 7, 40.0, 75.0),
        ("supertrend_rsi_10_2_14", 10, 2.0, "rsi", 14, 40.0, 75.0),
        ("supertrend_rsi_10_3_14", 10, 3.0, "rsi", 14, 45.0, 75.0),
        ("supertrend_rsi_14_3_14", 14, 3.0, "rsi", 14, 45.0, 70.0),
        ("supertrend_rsi_21_4_21", 21, 4.0, "rsi", 21, 45.0, 70.0),
    ]
    for name, atr_period, atr_mult, oscillator, period, lower, upper in supertrend_setups:
        definitions.append(
            _definition(
                name,
                "combinada",
                "SuperTrend define a direcao e um oscilador confirma entrada e saida.",
                "supertrend_oscillator",
                atr_period=atr_period,
                atr_mult=atr_mult,
                oscillator=oscillator,
                oscillator_period=period,
                lower=lower,
                upper=upper,
            )
        )

    adx_setups = [
        ("adx_trend_7_15_sma50", 7, 15.0, 50, 0),
        ("adx_trend_7_20_sma100", 7, 20.0, 100, 0),
        ("adx_trend_14_15_sma50", 14, 15.0, 50, 0),
        ("adx_trend_14_20_sma100", 14, 20.0, 100, 0),
        ("adx_trend_14_25_sma200", 14, 25.0, 200, 0),
        ("adx_trend_21_20_sma100", 21, 20.0, 100, 0),
        ("adx_trend_21_25_sma200", 21, 25.0, 200, 0),
        ("adx_rsi_trend_14_20_sma100", 14, 20.0, 100, 14),
        ("adx_rsi_trend_14_25_sma200", 14, 25.0, 200, 14),
        ("adx_rsi_trend_21_20_sma200", 21, 20.0, 200, 7),
    ]
    for name, adx_period, threshold, trend_window, rsi_period in adx_setups:
        definitions.append(
            _definition(
                name,
                "combinada",
                "ADX e direcionais confirmam forca compradora acima da media de tendencia.",
                "adx_trend",
                adx_period=adx_period,
                threshold=threshold,
                trend_window=trend_window,
                rsi_period=rsi_period,
            )
        )

    cci_setups = [
        ("cci_trend_10_m100_100_sma50", 10, -100.0, 100.0, 50),
        ("cci_trend_14_m100_100_sma100", 14, -100.0, 100.0, 100),
        ("cci_trend_20_m100_100_sma100", 20, -100.0, 100.0, 100),
        ("cci_trend_20_m100_100_sma200", 20, -100.0, 100.0, 200),
        ("cci_trend_30_m100_100_sma200", 30, -100.0, 100.0, 200),
        ("cci_momentum_10_0_m100_sma50", 10, 0.0, -100.0, 50),
        ("cci_momentum_14_0_m100_sma100", 14, 0.0, -100.0, 100),
        ("cci_momentum_20_0_m100_sma100", 20, 0.0, -100.0, 100),
        ("cci_momentum_20_50_m50_sma200", 20, 50.0, -50.0, 200),
        ("cci_momentum_30_50_m50_sma200", 30, 50.0, -50.0, 200),
    ]
    for name, period, entry_level, exit_level, trend_window in cci_setups:
        definitions.append(
            _definition(
                name,
                "combinada",
                "CCI identifica impulso ou pullback dentro de uma tendencia de alta.",
                "cci_trend",
                period=period,
                entry_level=entry_level,
                exit_level=exit_level,
                trend_window=trend_window,
            )
        )

    mfi_setups = [
        ("mfi_trend_7_20_80_sma50", 7, 20.0, 80.0, 50),
        ("mfi_trend_10_20_80_sma100", 10, 20.0, 80.0, 100),
        ("mfi_trend_14_20_80_sma100", 14, 20.0, 80.0, 100),
        ("mfi_trend_14_25_75_sma200", 14, 25.0, 75.0, 200),
        ("mfi_trend_21_25_75_sma200", 21, 25.0, 75.0, 200),
        ("mfi_momentum_7_50_30_sma50", 7, 50.0, 30.0, 50),
        ("mfi_momentum_10_50_30_sma100", 10, 50.0, 30.0, 100),
        ("mfi_momentum_14_55_35_sma100", 14, 55.0, 35.0, 100),
        ("mfi_momentum_14_55_35_sma200", 14, 55.0, 35.0, 200),
        ("mfi_momentum_21_60_40_sma200", 21, 60.0, 40.0, 200),
    ]
    for name, period, entry_level, exit_level, trend_window in mfi_setups:
        definitions.append(
            _definition(
                name,
                "volume",
                "Money Flow Index combina preco e volume com filtro de tendencia.",
                "mfi_trend",
                period=period,
                entry_level=entry_level,
                exit_level=exit_level,
                trend_window=trend_window,
            )
        )

    advanced_setups = [
        ("dsma_trend", "dsma", {"window": 40, "trend_window": 200}),
        ("dsma_trend_fast", "dsma", {"window": 20, "trend_window": 100}),
        ("dsma_trend_slow", "dsma", {"window": 80, "trend_window": 200}),
        (
            "profit_only_exit_sma_atr_catastrophe",
            "profit_only",
            {"sma_window": 50, "atr_period": 14, "atr_mult": 4.0, "catastrophe_mult": 6.0},
        ),
        (
            "profit_only_exit_sma100_atr3_catastrophe6",
            "profit_only",
            {"sma_window": 100, "atr_period": 14, "atr_mult": 3.0, "catastrophe_mult": 6.0},
        ),
        (
            "profit_only_exit_sma200_atr4_catastrophe8",
            "profit_only",
            {"sma_window": 200, "atr_period": 21, "atr_mult": 4.0, "catastrophe_mult": 8.0},
        ),
        (
            "ema_rsi_stochastic_adx_10_30",
            "multi_indicator",
            {"fast": 10, "slow": 30, "rsi_period": 14, "stoch_period": 14, "adx_period": 14, "adx_threshold": 20.0},
        ),
        (
            "ema_rsi_stochastic_adx_20_50",
            "multi_indicator",
            {"fast": 20, "slow": 50, "rsi_period": 14, "stoch_period": 14, "adx_period": 14, "adx_threshold": 20.0},
        ),
        (
            "ema_rsi_stochastic_adx_50_100",
            "multi_indicator",
            {"fast": 50, "slow": 100, "rsi_period": 14, "stoch_period": 21, "adx_period": 21, "adx_threshold": 20.0},
        ),
        (
            "ema_rsi_stochastic_adx_50_200",
            "multi_indicator",
            {"fast": 50, "slow": 200, "rsi_period": 14, "stoch_period": 21, "adx_period": 21, "adx_threshold": 25.0},
        ),
    ]
    for name, engine, parameters in advanced_setups:
        definitions.append(
            _definition(
                name,
                "avancada",
                "Refinamento avancado com tendencia, osciladores e controle de saida.",
                engine,
                **parameters,
            )
        )

    if len(definitions) != 130:
        raise RuntimeError(f"Catalogo adicional invalido: {len(definitions)} estrategias; esperado 130.")
    if len({item.name for item in definitions}) != len(definitions):
        raise RuntimeError("O catalogo adicional possui nomes duplicados.")
    return definitions


def _run(
    candles: list[Candle],
    *,
    engine: str,
    parameters: dict[str, object],
) -> list[int]:
    engines = {
        "time_series_momentum": _time_series_momentum,
        "moving_average_cross": _moving_average_cross,
        "macd": _macd,
        "donchian": _donchian,
        "atr_breakout": _atr_breakout,
        "rsi_reversion": _rsi_reversion,
        "rsi_bollinger": _rsi_bollinger,
        "stochastic_trend": _stochastic_trend,
        "supertrend_oscillator": _supertrend_oscillator,
        "adx_trend": _adx_trend,
        "cci_trend": _cci_trend,
        "mfi_trend": _mfi_trend,
        "dsma": _dsma,
        "profit_only": _profit_only,
        "multi_indicator": _multi_indicator,
    }
    return engines[engine](candles, **parameters)


def _time_series_momentum(
    candles: list[Candle],
    *,
    lookback: int,
    skip: int,
    trend_window: int,
) -> list[int]:
    closes = [item.close for item in candles]
    trend = _sma(closes, trend_window) if trend_window else [None] * len(candles)
    signals = []
    for index, close in enumerate(closes):
        recent = index - skip
        past = index - lookback
        valid = recent >= 0 and past >= 0 and closes[past] > 0
        trend_ok = trend_window == 0 or (trend[index] is not None and close > trend[index])
        signals.append(int(valid and closes[recent] > closes[past] and trend_ok))
    return signals


def _moving_average_cross(
    candles: list[Candle],
    *,
    average_type: str,
    fast: int,
    slow: int,
) -> list[int]:
    closes = [item.close for item in candles]
    average = _ema if average_type == "ema" else _sma
    fast_values = average(closes, fast)
    slow_values = average(closes, slow)
    return [
        int(fast_value is not None and slow_value is not None and fast_value > slow_value)
        for fast_value, slow_value in zip(fast_values, slow_values)
    ]


def _macd(
    candles: list[Candle],
    *,
    fast: int,
    slow: int,
    signal_window: int,
    trend_window: int,
) -> list[int]:
    closes = [item.close for item in candles]
    fast_values = _ema(closes, fast)
    slow_values = _ema(closes, slow)
    line = [
        None if fast_value is None or slow_value is None else fast_value - slow_value
        for fast_value, slow_value in zip(fast_values, slow_values)
    ]
    signal = _ema_optional(line, signal_window)
    trend = _sma(closes, trend_window) if trend_window else [None] * len(candles)
    return [
        int(
            line_value is not None
            and signal_value is not None
            and line_value > signal_value
            and (trend_window == 0 or (trend[index] is not None and closes[index] > trend[index]))
        )
        for index, (line_value, signal_value) in enumerate(zip(line, signal))
    ]


def _donchian(
    candles: list[Candle],
    *,
    entry_window: int,
    exit_window: int,
    trend_window: int,
) -> list[int]:
    closes = [item.close for item in candles]
    trend = _sma(closes, trend_window) if trend_window else [None] * len(candles)
    position = 0
    signals = []
    for index, candle in enumerate(candles):
        if index >= max(entry_window, exit_window):
            trend_ok = trend_window == 0 or (
                trend[index] is not None and candle.close > trend[index]
            )
            if position == 0:
                prior_high = max(item.high for item in candles[index - entry_window : index])
                if candle.close > prior_high and trend_ok:
                    position = 1
            else:
                prior_low = min(item.low for item in candles[index - exit_window : index])
                if candle.close < prior_low:
                    position = 0
        signals.append(position)
    return signals


def _atr_breakout(
    candles: list[Candle],
    *,
    lookback: int,
    atr_period: int,
    atr_mult: float,
    trend_window: int,
) -> list[int]:
    closes = [item.close for item in candles]
    trend = _sma(closes, trend_window) if trend_window else [None] * len(candles)
    atr_values = _atr(candles, atr_period)
    position = 0
    peak = 0.0
    signals = []
    for index, candle in enumerate(candles):
        atr_value = atr_values[index]
        if index >= lookback and atr_value is not None:
            trend_ok = trend_window == 0 or (
                trend[index] is not None and candle.close > trend[index]
            )
            prior_high = max(item.high for item in candles[index - lookback : index])
            if position == 0 and candle.close > prior_high and trend_ok:
                position = 1
                peak = candle.close
            elif position == 1:
                peak = max(peak, candle.close)
                if candle.close < peak - atr_mult * atr_value:
                    position = 0
                    peak = 0.0
        signals.append(position)
    return signals


def _rsi_reversion(
    candles: list[Candle],
    *,
    rsi_period: int,
    lower: float,
    upper: float,
    trend_window: int,
    max_hold: int,
) -> list[int]:
    closes = [item.close for item in candles]
    values = _rsi(closes, rsi_period)
    trend = _sma(closes, trend_window) if trend_window else [None] * len(candles)
    position = 0
    held = 0
    signals = []
    for index, value in enumerate(values):
        trend_ok = trend_window == 0 or (
            trend[index] is not None and closes[index] > trend[index]
        )
        if value is not None:
            if position == 0 and value <= lower and trend_ok:
                position = 1
                held = 0
            elif position == 1:
                held += 1
                if value >= upper or held >= max_hold:
                    position = 0
                    held = 0
        signals.append(position)
    return signals


def _rsi_bollinger(
    candles: list[Candle],
    *,
    rsi_period: int,
    lower: float,
    upper: float,
    window: int,
    num_std: float,
    trend_window: int,
) -> list[int]:
    closes = [item.close for item in candles]
    rsi_values = _rsi(closes, rsi_period)
    middle = _sma(closes, window)
    trend = _sma(closes, trend_window) if trend_window else [None] * len(candles)
    position = 0
    signals = []
    for index, close in enumerate(closes):
        if index + 1 >= window and middle[index] is not None and rsi_values[index] is not None:
            deviation = statistics.pstdev(closes[index - window + 1 : index + 1])
            lower_band = middle[index] - num_std * deviation
            trend_ok = trend_window == 0 or (
                trend[index] is not None and close > trend[index]
            )
            if position == 0 and close <= lower_band and rsi_values[index] <= lower and trend_ok:
                position = 1
            elif position == 1 and (close >= middle[index] or rsi_values[index] >= upper):
                position = 0
        signals.append(position)
    return signals


def _stochastic_trend(
    candles: list[Candle],
    *,
    average_type: str,
    trend_window: int,
    k_period: int,
    lower: float,
    upper: float,
) -> list[int]:
    closes = [item.close for item in candles]
    trend = (_ema if average_type == "ema" else _sma)(closes, trend_window)
    stochastic = _stochastic(candles, k_period)
    position = 0
    armed = False
    signals = []
    for index, value in enumerate(stochastic):
        trend_ok = trend[index] is not None and closes[index] > trend[index]
        if value is not None:
            if position == 0:
                armed = armed or (trend_ok and value <= lower)
                if armed and trend_ok and index > 0 and stochastic[index - 1] is not None and value > stochastic[index - 1]:
                    position = 1
                    armed = False
            elif value >= upper or not trend_ok:
                position = 0
        signals.append(position)
    return signals


def _supertrend_oscillator(
    candles: list[Candle],
    *,
    atr_period: int,
    atr_mult: float,
    oscillator: str,
    oscillator_period: int,
    lower: float,
    upper: float,
) -> list[int]:
    trend = _supertrend(candles, atr_period, atr_mult)
    values = (
        _stochastic(candles, oscillator_period)
        if oscillator == "stochastic"
        else _rsi([item.close for item in candles], oscillator_period)
    )
    position = 0
    armed = False
    signals = []
    for index, value in enumerate(values):
        if value is not None:
            if position == 0:
                armed = armed or (trend[index] == 1 and value <= lower)
                if armed and trend[index] == 1 and index > 0 and values[index - 1] is not None and value > values[index - 1]:
                    position = 1
                    armed = False
            elif trend[index] == 0 or value >= upper:
                position = 0
        signals.append(position)
    return signals


def _adx_trend(
    candles: list[Candle],
    *,
    adx_period: int,
    threshold: float,
    trend_window: int,
    rsi_period: int,
) -> list[int]:
    adx_values, plus_di, minus_di = _adx(candles, adx_period)
    closes = [item.close for item in candles]
    trend = _sma(closes, trend_window)
    rsi_values = _rsi(closes, rsi_period) if rsi_period else [None] * len(candles)
    signals = []
    for index in range(len(candles)):
        rsi_ok = rsi_period == 0 or (
            rsi_values[index] is not None and 45.0 <= rsi_values[index] <= 75.0
        )
        signals.append(
            int(
                adx_values[index] is not None
                and plus_di[index] is not None
                and minus_di[index] is not None
                and trend[index] is not None
                and adx_values[index] >= threshold
                and plus_di[index] > minus_di[index]
                and closes[index] > trend[index]
                and rsi_ok
            )
        )
    return signals


def _cci_trend(
    candles: list[Candle],
    *,
    period: int,
    entry_level: float,
    exit_level: float,
    trend_window: int,
) -> list[int]:
    values = _cci(candles, period)
    closes = [item.close for item in candles]
    trend = _sma(closes, trend_window)
    position = 0
    signals = []
    for index, value in enumerate(values):
        if value is not None and trend[index] is not None:
            trend_ok = closes[index] > trend[index]
            if position == 0 and trend_ok and value >= entry_level:
                position = 1
            elif position == 1 and (not trend_ok or value <= exit_level):
                position = 0
        signals.append(position)
    return signals


def _mfi_trend(
    candles: list[Candle],
    *,
    period: int,
    entry_level: float,
    exit_level: float,
    trend_window: int,
) -> list[int]:
    values = _mfi(candles, period)
    closes = [item.close for item in candles]
    trend = _sma(closes, trend_window)
    position = 0
    signals = []
    reversion_mode = entry_level < exit_level
    for index, value in enumerate(values):
        if value is not None and trend[index] is not None:
            trend_ok = closes[index] > trend[index]
            entry = value <= entry_level if reversion_mode else value >= entry_level
            exit_ = value >= exit_level if reversion_mode else value <= exit_level
            if position == 0 and trend_ok and entry:
                position = 1
            elif position == 1 and (not trend_ok or exit_):
                position = 0
        signals.append(position)
    return signals


def _dsma(
    candles: list[Candle],
    *,
    window: int,
    trend_window: int,
) -> list[int]:
    closes = [item.close for item in candles]
    first = _ema(closes, window)
    second = _ema_optional(first, window)
    trend = _sma(closes, trend_window)
    signals = []
    for index, value in enumerate(second):
        rising = index > 0 and value is not None and second[index - 1] is not None and value > second[index - 1]
        signals.append(
            int(
                value is not None
                and trend[index] is not None
                and closes[index] > value
                and closes[index] > trend[index]
                and rising
            )
        )
    return signals


def _profit_only(
    candles: list[Candle],
    *,
    sma_window: int,
    atr_period: int,
    atr_mult: float,
    catastrophe_mult: float,
) -> list[int]:
    closes = [item.close for item in candles]
    trend = _sma(closes, sma_window)
    atr_values = _atr(candles, atr_period)
    position = 0
    entry = 0.0
    peak = 0.0
    signals = []
    for index, close in enumerate(closes):
        atr_value = atr_values[index]
        if trend[index] is not None and atr_value is not None:
            if position == 0 and close > trend[index]:
                position = 1
                entry = close
                peak = close
            elif position == 1:
                peak = max(peak, close)
                profitable_trend_exit = close > entry and close < trend[index]
                trailing_exit = close > entry and close < peak - atr_mult * atr_value
                catastrophe_exit = close < entry - catastrophe_mult * atr_value
                if profitable_trend_exit or trailing_exit or catastrophe_exit:
                    position = 0
                    entry = 0.0
                    peak = 0.0
        signals.append(position)
    return signals


def _multi_indicator(
    candles: list[Candle],
    *,
    fast: int,
    slow: int,
    rsi_period: int,
    stoch_period: int,
    adx_period: int,
    adx_threshold: float,
) -> list[int]:
    closes = [item.close for item in candles]
    fast_average = _ema(closes, fast)
    slow_average = _ema(closes, slow)
    rsi_values = _rsi(closes, rsi_period)
    stochastic = _stochastic(candles, stoch_period)
    adx_values, plus_di, minus_di = _adx(candles, adx_period)
    position = 0
    signals = []
    for index in range(len(candles)):
        ready = all(
            value is not None
            for value in (
                fast_average[index],
                slow_average[index],
                rsi_values[index],
                stochastic[index],
                adx_values[index],
                plus_di[index],
                minus_di[index],
            )
        )
        if ready:
            trend_ok = fast_average[index] > slow_average[index]
            confirmation = (
                45.0 <= rsi_values[index] <= 75.0
                and stochastic[index] >= 30.0
                and adx_values[index] >= adx_threshold
                and plus_di[index] > minus_di[index]
            )
            if position == 0 and trend_ok and confirmation:
                position = 1
            elif position == 1 and (
                not trend_ok or rsi_values[index] < 40.0 or plus_di[index] <= minus_di[index]
            ):
                position = 0
        signals.append(position)
    return signals


def _sma(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / window if index + 1 >= window else None)
    return result


def _ema(values: list[float], window: int) -> list[float | None]:
    return _ema_optional([float(value) for value in values], window)


def _ema_optional(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    available: list[float] = []
    current: float | None = None
    alpha = 2.0 / (window + 1)
    for index, value in enumerate(values):
        if value is None:
            continue
        if current is None:
            available.append(value)
            if len(available) == window:
                current = sum(available) / window
                result[index] = current
        else:
            current = value * alpha + current * (1 - alpha)
            result[index] = current
    return result


def _true_ranges(candles: list[Candle]) -> list[float]:
    result = []
    for index, candle in enumerate(candles):
        previous = candles[index - 1].close if index else candle.close
        result.append(max(candle.high - candle.low, abs(candle.high - previous), abs(candle.low - previous)))
    return result


def _atr(candles: list[Candle], period: int) -> list[float | None]:
    return _sma(_true_ranges(candles), period)


def _rsi(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return result
    gains = [max(values[index] - values[index - 1], 0.0) for index in range(1, len(values))]
    losses = [max(values[index - 1] - values[index], 0.0) for index in range(1, len(values))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    result[period] = _rsi_value(avg_gain, avg_loss)
    for index in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[index - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[index - 1]) / period
        result[index] = _rsi_value(avg_gain, avg_loss)
    return result


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _stochastic(candles: list[Candle], period: int) -> list[float | None]:
    result: list[float | None] = []
    for index, candle in enumerate(candles):
        if index + 1 < period:
            result.append(None)
            continue
        window = candles[index - period + 1 : index + 1]
        low = min(item.low for item in window)
        high = max(item.high for item in window)
        result.append(50.0 if high == low else 100.0 * (candle.close - low) / (high - low))
    return result


def _supertrend(candles: list[Candle], period: int, multiplier: float) -> list[int]:
    atr_values = _atr(candles, period)
    upper = [None] * len(candles)
    lower = [None] * len(candles)
    trend = 0
    result = []
    for index, candle in enumerate(candles):
        atr_value = atr_values[index]
        if atr_value is None:
            result.append(0)
            continue
        midpoint = (candle.high + candle.low) / 2
        basic_upper = midpoint + multiplier * atr_value
        basic_lower = midpoint - multiplier * atr_value
        if index == 0 or upper[index - 1] is None:
            upper[index] = basic_upper
            lower[index] = basic_lower
        else:
            previous_close = candles[index - 1].close
            upper[index] = basic_upper if basic_upper < upper[index - 1] or previous_close > upper[index - 1] else upper[index - 1]
            lower[index] = basic_lower if basic_lower > lower[index - 1] or previous_close < lower[index - 1] else lower[index - 1]
        if trend == 0 and candle.close > upper[index]:
            trend = 1
        elif trend == 1 and candle.close < lower[index]:
            trend = 0
        result.append(trend)
    return result


def _adx(
    candles: list[Candle],
    period: int,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    length = len(candles)
    true_ranges = _true_ranges(candles)
    plus_dm = [0.0] * length
    minus_dm = [0.0] * length
    for index in range(1, length):
        up = candles[index].high - candles[index - 1].high
        down = candles[index - 1].low - candles[index].low
        plus_dm[index] = up if up > down and up > 0 else 0.0
        minus_dm[index] = down if down > up and down > 0 else 0.0
    atr_values = _sma(true_ranges, period)
    plus_average = _sma(plus_dm, period)
    minus_average = _sma(minus_dm, period)
    plus_di: list[float | None] = []
    minus_di: list[float | None] = []
    dx: list[float | None] = []
    for atr_value, plus_value, minus_value in zip(atr_values, plus_average, minus_average):
        if atr_value is None or atr_value <= 0 or plus_value is None or minus_value is None:
            plus_di.append(None)
            minus_di.append(None)
            dx.append(None)
            continue
        plus = 100.0 * plus_value / atr_value
        minus = 100.0 * minus_value / atr_value
        plus_di.append(plus)
        minus_di.append(minus)
        total = plus + minus
        dx.append(0.0 if total == 0 else 100.0 * abs(plus - minus) / total)
    return _rolling_optional_mean(dx, period), plus_di, minus_di


def _cci(candles: list[Candle], period: int) -> list[float | None]:
    typical = [(item.high + item.low + item.close) / 3 for item in candles]
    average = _sma(typical, period)
    result: list[float | None] = []
    for index, value in enumerate(typical):
        if average[index] is None:
            result.append(None)
            continue
        window = typical[index - period + 1 : index + 1]
        deviation = sum(abs(item - average[index]) for item in window) / period
        result.append(0.0 if deviation == 0 else (value - average[index]) / (0.015 * deviation))
    return result


def _mfi(candles: list[Candle], period: int) -> list[float | None]:
    typical = [(item.high + item.low + item.close) / 3 for item in candles]
    positive = [0.0] * len(candles)
    negative = [0.0] * len(candles)
    for index in range(1, len(candles)):
        flow = typical[index] * max(candles[index].volume, 0)
        if typical[index] > typical[index - 1]:
            positive[index] = flow
        elif typical[index] < typical[index - 1]:
            negative[index] = flow
    result: list[float | None] = []
    for index in range(len(candles)):
        if index + 1 < period:
            result.append(None)
            continue
        positive_sum = sum(positive[index - period + 1 : index + 1])
        negative_sum = sum(negative[index - period + 1 : index + 1])
        if negative_sum == 0:
            result.append(100.0 if positive_sum > 0 else 50.0)
        else:
            result.append(100.0 - 100.0 / (1.0 + positive_sum / negative_sum))
    return result


def _rolling_optional_mean(values: list[float | None], period: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < period:
            result.append(None)
            continue
        window = values[index - period + 1 : index + 1]
        result.append(sum(value for value in window if value is not None) / period if all(value is not None for value in window) else None)
    return result


ADDITIONAL_STRATEGIES = _build_definitions()
ADDITIONAL_PARAMETERS = {item.name: dict(item.parameters) for item in ADDITIONAL_STRATEGIES}
