from __future__ import annotations

import inspect
import statistics
from dataclasses import dataclass
from numbers import Integral

from .candles import Candle


@dataclass(frozen=True)
class StrategyInfo:
    name: str
    family: str
    description: str
    sweepable: bool = True


def buy_and_hold(candles: list[Candle]) -> list[int]:
    return [1 for _ in candles]


def sma_cross(candles: list[Candle], *, fast: int = 50, slow: int = 200) -> list[int]:
    if fast <= 0 or slow <= 0:
        raise ValueError("fast e slow precisam ser maiores que zero.")
    if fast >= slow:
        raise ValueError("fast precisa ser menor que slow.")

    closes = [candle.close for candle in candles]
    fast_ma = rolling_mean(closes, fast)
    slow_ma = rolling_mean(closes, slow)
    signals: list[int] = []

    for fast_value, slow_value in zip(fast_ma, slow_ma):
        signals.append(1 if fast_value is not None and slow_value is not None and fast_value > slow_value else 0)

    return signals


def ema_cross(candles: list[Candle], *, fast: int = 12, slow: int = 26) -> list[int]:
    if fast <= 0 or slow <= 0:
        raise ValueError("fast e slow precisam ser maiores que zero.")
    if fast >= slow:
        raise ValueError("fast precisa ser menor que slow.")

    closes = [candle.close for candle in candles]
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    return [
        1 if fast_value is not None and slow_value is not None and fast_value > slow_value else 0
        for fast_value, slow_value in zip(fast_ema, slow_ema)
    ]


def price_sma(candles: list[Candle], *, sma_window: int = 200) -> list[int]:
    if sma_window <= 0:
        raise ValueError("sma_window precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    moving_average = rolling_mean(closes, sma_window)
    return [
        1 if average is not None and close > average else 0
        for close, average in zip(closes, moving_average)
    ]


def momentum(candles: list[Candle], *, lookback: int = 126) -> list[int]:
    if lookback <= 0:
        raise ValueError("lookback precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    signals: list[int] = []
    for index, close in enumerate(closes):
        if index < lookback:
            signals.append(0)
            continue
        signals.append(1 if close > closes[index - lookback] else 0)
    return signals


def roc_trend(
    candles: list[Candle],
    *,
    lookback: int = 126,
    sma_window: int = 200,
) -> list[int]:
    if lookback <= 0 or sma_window <= 0:
        raise ValueError("lookback e sma_window precisam ser maiores que zero.")

    closes = [candle.close for candle in candles]
    trend_ma = rolling_mean(closes, sma_window)
    signals: list[int] = []
    for index, close in enumerate(closes):
        if index < lookback or trend_ma[index] is None:
            signals.append(0)
            continue
        signals.append(1 if close > closes[index - lookback] and close > trend_ma[index] else 0)
    return signals


def breakout(candles: list[Candle], *, lookback: int = 55, exit_lookback: int = 20) -> list[int]:
    if lookback <= 0 or exit_lookback <= 0:
        raise ValueError("lookback e exit_lookback precisam ser maiores que zero.")

    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        if index < max(lookback, exit_lookback):
            signals.append(position)
            continue

        prior_high = max(item.high for item in candles[index - lookback : index])
        prior_low = min(item.low for item in candles[index - exit_lookback : index])

        if position == 0 and candle.close > prior_high:
            position = 1
        elif position == 1 and candle.close < prior_low:
            position = 0

        signals.append(position)

    return signals


def atr_breakout(
    candles: list[Candle],
    *,
    lookback: int = 20,
    atr_period: int = 14,
    atr_mult: float = 3.0,
) -> list[int]:
    if lookback <= 0 or atr_period <= 0 or atr_mult <= 0:
        raise ValueError("lookback, atr_period e atr_mult precisam ser maiores que zero.")

    atr_values = atr(candles, atr_period)
    position = 0
    peak_close = 0.0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if index < max(lookback, atr_period) or atr_values[index] is None:
            signals.append(position)
            continue

        prior_high = max(item.high for item in candles[index - lookback : index])
        if position == 0 and candle.close > prior_high:
            position = 1
            peak_close = candle.close
        elif position == 1:
            peak_close = max(peak_close, candle.close)
            trailing_stop = peak_close - atr_mult * atr_values[index]
            if candle.close < trailing_stop:
                position = 0
                peak_close = 0.0

        signals.append(position)

    return signals


def bollinger_reversion(
    candles: list[Candle],
    *,
    window: int = 20,
    num_std: float = 2.0,
    exit_z: float = 0.0,
) -> list[int]:
    if window <= 1:
        raise ValueError("window precisa ser maior que 1.")
    if num_std <= 0:
        raise ValueError("num_std precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    bands = rolling_bands(closes, window, num_std)
    position = 0
    signals: list[int] = []

    for close, band in zip(closes, bands):
        if band is None:
            signals.append(position)
            continue
        middle, lower, _upper, std = band
        exit_level = middle + exit_z * std
        if position == 0 and close < lower:
            position = 1
        elif position == 1 and close > exit_level:
            position = 0
        signals.append(position)

    return signals


def macd(
    candles: list[Candle],
    *,
    fast: int = 12,
    slow: int = 26,
    signal_window: int = 9,
) -> list[int]:
    if fast <= 0 or slow <= 0 or signal_window <= 0:
        raise ValueError("fast, slow e signal_window precisam ser maiores que zero.")
    if fast >= slow:
        raise ValueError("fast precisa ser menor que slow.")

    closes = [candle.close for candle in candles]
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = [
        None if fast_value is None or slow_value is None else fast_value - slow_value
        for fast_value, slow_value in zip(fast_ema, slow_ema)
    ]
    signal_line = ema_optional(macd_line, signal_window)
    return [
        1 if line is not None and signal is not None and line > signal else 0
        for line, signal in zip(macd_line, signal_line)
    ]


def rsi_reversion(
    candles: list[Candle],
    *,
    rsi_period: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")

    rsi_values = rsi([candle.close for candle in candles], rsi_period)
    position = 0
    signals: list[int] = []
    for value in rsi_values:
        if value is None:
            signals.append(position)
            continue
        if position == 0 and value <= lower:
            position = 1
        elif position == 1 and value >= upper:
            position = 0
        signals.append(position)
    return signals


def rsi_reversion_hold(
    candles: list[Candle],
    *,
    rsi_period: int = 14,
    lower: float = 50.0,
    upper: float = 80.0,
    max_hold: int = 20,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")
    if max_hold <= 0:
        raise ValueError("max_hold precisa ser maior que zero.")

    rsi_values = rsi([candle.close for candle in candles], rsi_period)
    position = 0
    held = 0
    signals: list[int] = []
    for value in rsi_values:
        if value is None:
            signals.append(position)
            continue
        if position == 0 and value <= lower:
            position = 1
            held = 0
        elif position == 1:
            held += 1
            if value >= upper or held >= max_hold:
                position = 0
                held = 0
        signals.append(position)
    return signals


def rsi_reversion_atr(
    candles: list[Candle],
    *,
    rsi_period: int = 14,
    lower: float = 50.0,
    upper: float = 80.0,
    atr_period: int = 14,
    atr_mult: float = 3.0,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")
    if atr_period <= 0 or atr_mult <= 0:
        raise ValueError("atr_period e atr_mult precisam ser maiores que zero.")

    rsi_values = rsi([candle.close for candle in candles], rsi_period)
    atr_values = atr(candles, atr_period)
    position = 0
    entry_close = 0.0
    entry_atr = 0.0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        rsi_value = rsi_values[index]
        atr_value = atr_values[index]
        if rsi_value is None or atr_value is None:
            signals.append(position)
            continue

        if position == 0 and rsi_value <= lower:
            position = 1
            entry_close = candle.close
            entry_atr = atr_value
        elif position == 1:
            stop = entry_close - atr_mult * entry_atr
            if rsi_value >= upper or candle.close < stop:
                position = 0
                entry_close = 0.0
                entry_atr = 0.0
        signals.append(position)

    return signals


def rsi_reversion_trend_entry(
    candles: list[Candle],
    *,
    rsi_period: int = 14,
    lower: float = 50.0,
    upper: float = 80.0,
    trend_window: int = 200,
) -> list[int]:
    if trend_window <= 0:
        raise ValueError("trend_window precisa ser maior que zero.")
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")

    closes = [candle.close for candle in candles]
    rsi_values = rsi(closes, rsi_period)
    trend_ma = rolling_mean(closes, trend_window)
    position = 0
    signals: list[int] = []

    for index, close in enumerate(closes):
        rsi_value = rsi_values[index]
        trend_value = trend_ma[index]
        if rsi_value is None or trend_value is None:
            signals.append(position)
            continue
        if position == 0 and close > trend_value and rsi_value <= lower:
            position = 1
        elif position == 1 and rsi_value >= upper:
            position = 0
        signals.append(position)

    return signals


def rsi_cross_reversion(
    candles: list[Candle],
    *,
    rsi_period: int = 14,
    lower: float = 50.0,
    upper: float = 80.0,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")

    rsi_values = rsi([candle.close for candle in candles], rsi_period)
    position = 0
    was_oversold = False
    signals: list[int] = []
    for value in rsi_values:
        if value is None:
            signals.append(position)
            continue
        if position == 0:
            if value <= lower:
                was_oversold = True
            elif was_oversold and value > lower:
                position = 1
                was_oversold = False
        elif value >= upper:
            position = 0
        signals.append(position)
    return signals


def rsi_bollinger(
    candles: list[Candle],
    *,
    rsi_period: int = 14,
    lower: float = 50.0,
    upper: float = 80.0,
    window: int = 20,
    num_std: float = 2.0,
    exit_z: float = 0.0,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")
    if window <= 1:
        raise ValueError("window precisa ser maior que 1.")
    if num_std <= 0:
        raise ValueError("num_std precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    rsi_values = rsi(closes, rsi_period)
    bands = rolling_bands(closes, window, num_std)
    position = 0
    signals: list[int] = []

    for index, close in enumerate(closes):
        rsi_value = rsi_values[index]
        band = bands[index]
        if rsi_value is None or band is None:
            signals.append(position)
            continue
        middle, lower_band, _upper_band, std = band
        exit_level = middle + exit_z * std
        if position == 0 and rsi_value <= lower and close <= lower_band:
            position = 1
        elif position == 1 and (rsi_value >= upper or close >= exit_level):
            position = 0
        signals.append(position)

    return signals


def connors_rsi_reversion(
    candles: list[Candle],
    *,
    rsi_period: int = 3,
    streak_rsi_period: int = 2,
    rank_period: int = 100,
    lower: float = 20.0,
    upper: float = 70.0,
) -> list[int]:
    if streak_rsi_period <= 0 or rank_period <= 1:
        raise ValueError("streak_rsi_period precisa ser maior que zero e rank_period maior que 1.")
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")

    closes = [candle.close for candle in candles]
    price_rsi = rsi(closes, rsi_period)
    streak_values = close_streaks(closes)
    streak_rsi = rsi(streak_values, streak_rsi_period)
    percent_rank = roc_percent_rank(closes, rank_period)
    position = 0
    signals: list[int] = []

    for price_value, streak_value, rank_value in zip(price_rsi, streak_rsi, percent_rank):
        if price_value is None or streak_value is None or rank_value is None:
            signals.append(position)
            continue
        crsi = (price_value + streak_value + rank_value) / 3
        if position == 0 and crsi <= lower:
            position = 1
        elif position == 1 and crsi >= upper:
            position = 0
        signals.append(position)

    return signals


def ibs_reversion(
    candles: list[Candle],
    *,
    ibs_lower: float = 0.2,
    ibs_upper: float = 0.8,
    max_hold: int = 3,
    trend_window: int = 0,
) -> list[int]:
    if not 0 <= ibs_lower < ibs_upper <= 1:
        raise ValueError("ibs_lower e ibs_upper precisam estar entre 0 e 1, com lower < upper.")
    if max_hold <= 0:
        raise ValueError("max_hold precisa ser maior que zero.")
    if trend_window < 0:
        raise ValueError("trend_window nao pode ser negativo.")

    closes = [candle.close for candle in candles]
    trend_ma = rolling_mean(closes, trend_window) if trend_window else [None for _ in candles]
    ibs_values = internal_bar_strength(candles)
    position = 0
    held = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        trend_ok = trend_window == 0 or (trend_ma[index] is not None and candle.close > trend_ma[index])
        if position == 0 and trend_ok and ibs_values[index] <= ibs_lower:
            position = 1
            held = 0
        elif position == 1:
            held += 1
            if ibs_values[index] >= ibs_upper or held >= max_hold:
                position = 0
                held = 0
        signals.append(position)

    return signals


def rsi_ibs_reversion(
    candles: list[Candle],
    *,
    rsi_period: int = 2,
    lower: float = 5.0,
    upper: float = 60.0,
    ibs_lower: float = 0.25,
    ibs_upper: float = 0.75,
    trend_window: int = 200,
    max_hold: int = 10,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")
    if not 0 <= ibs_lower < ibs_upper <= 1:
        raise ValueError("ibs_lower e ibs_upper precisam estar entre 0 e 1, com lower < upper.")
    if trend_window < 0:
        raise ValueError("trend_window nao pode ser negativo.")
    if max_hold <= 0:
        raise ValueError("max_hold precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    rsi_values = rsi(closes, rsi_period)
    ibs_values = internal_bar_strength(candles)
    trend_ma = rolling_mean(closes, trend_window) if trend_window else [None for _ in candles]
    position = 0
    held = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        rsi_value = rsi_values[index]
        trend_ok = trend_window == 0 or (trend_ma[index] is not None and candle.close > trend_ma[index])
        if rsi_value is None:
            signals.append(position)
            continue

        if position == 0 and trend_ok and rsi_value <= lower and ibs_values[index] <= ibs_lower:
            position = 1
            held = 0
        elif position == 1:
            held += 1
            if rsi_value >= upper or ibs_values[index] >= ibs_upper or held >= max_hold:
                position = 0
                held = 0
        signals.append(position)

    return signals


def rsi2_trend_reversion(
    candles: list[Candle],
    *,
    rsi_period: int = 2,
    lower: float = 5.0,
    upper: float = 70.0,
    trend_window: int = 200,
    sma_window: int = 5,
    max_hold: int = 10,
) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")
    if trend_window < 0 or sma_window <= 0:
        raise ValueError("trend_window nao pode ser negativo e sma_window precisa ser maior que zero.")
    if max_hold <= 0:
        raise ValueError("max_hold precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    rsi_values = rsi(closes, rsi_period)
    trend_ma = rolling_mean(closes, trend_window) if trend_window else [None for _ in candles]
    exit_ma = rolling_mean(closes, sma_window)
    position = 0
    held = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        rsi_value = rsi_values[index]
        trend_ok = trend_window == 0 or (trend_ma[index] is not None and candle.close > trend_ma[index])
        if rsi_value is None or exit_ma[index] is None:
            signals.append(position)
            continue

        if position == 0 and trend_ok and rsi_value <= lower:
            position = 1
            held = 0
        elif position == 1:
            held += 1
            if candle.close > exit_ma[index] or rsi_value >= upper or held >= max_hold:
                position = 0
                held = 0
        signals.append(position)

    return signals


def down_streak_reversion(
    candles: list[Candle],
    *,
    streak_length: int = 3,
    ibs_lower: float = 0.35,
    ibs_upper: float = 0.75,
    trend_window: int = 200,
    max_hold: int = 10,
) -> list[int]:
    if streak_length <= 0:
        raise ValueError("streak_length precisa ser maior que zero.")
    if not 0 <= ibs_lower < ibs_upper <= 1:
        raise ValueError("ibs_lower e ibs_upper precisam estar entre 0 e 1, com lower < upper.")
    if trend_window < 0:
        raise ValueError("trend_window nao pode ser negativo.")
    if max_hold <= 0:
        raise ValueError("max_hold precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    trend_ma = rolling_mean(closes, trend_window) if trend_window else [None for _ in candles]
    streaks = close_streaks(closes)
    ibs_values = internal_bar_strength(candles)
    position = 0
    held = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        trend_ok = trend_window == 0 or (trend_ma[index] is not None and candle.close > trend_ma[index])
        if position == 0 and trend_ok and streaks[index] <= -streak_length and ibs_values[index] <= ibs_lower:
            position = 1
            held = 0
        elif position == 1:
            held += 1
            if ibs_values[index] >= ibs_upper or held >= max_hold:
                position = 0
                held = 0
        signals.append(position)

    return signals


def range_expansion_breakout(
    candles: list[Candle],
    *,
    range_mult: float = 0.5,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    trend_window: int = 50,
    volume_window: int = 20,
    volume_mult: float = 1.0,
    max_hold: int = 40,
) -> list[int]:
    if range_mult <= 0 or atr_period <= 0 or atr_mult <= 0:
        raise ValueError("range_mult, atr_period e atr_mult precisam ser maiores que zero.")
    if trend_window < 0 or volume_window < 0:
        raise ValueError("trend_window e volume_window nao podem ser negativos.")
    if volume_mult < 0:
        raise ValueError("volume_mult nao pode ser negativo.")
    if max_hold <= 0:
        raise ValueError("max_hold precisa ser maior que zero.")

    closes = [candle.close for candle in candles]
    volumes = [float(candle.volume) for candle in candles]
    trend_ma = rolling_mean(closes, trend_window) if trend_window else [None for _ in candles]
    volume_ma = rolling_mean(volumes, volume_window) if volume_window else [None for _ in candles]
    atr_values = atr(candles, atr_period)
    position = 0
    held = 0
    peak_close = 0.0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if index == 0 or atr_values[index] is None:
            signals.append(position)
            continue

        prior_range = candles[index - 1].high - candles[index - 1].low
        trigger = candle.open + range_mult * prior_range
        trend_ok = trend_window == 0 or (trend_ma[index] is not None and candle.close > trend_ma[index])
        volume_ok = volume_window == 0 or (volume_ma[index] is not None and candle.volume >= volume_ma[index] * volume_mult)

        if position == 0 and trend_ok and volume_ok and candle.close > trigger:
            position = 1
            held = 0
            peak_close = candle.close
        elif position == 1:
            held += 1
            peak_close = max(peak_close, candle.close)
            stop = peak_close - atr_mult * atr_values[index]
            if candle.close < stop or held >= max_hold:
                position = 0
                held = 0
                peak_close = 0.0
        signals.append(position)

    return signals


def chandelier_breakout(
    candles: list[Candle],
    *,
    lookback: int = 20,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    volume_window: int = 0,
    volume_mult: float = 1.0,
) -> list[int]:
    if lookback <= 0 or atr_period <= 0 or atr_mult <= 0:
        raise ValueError("lookback, atr_period e atr_mult precisam ser maiores que zero.")
    if volume_window < 0 or volume_mult < 0:
        raise ValueError("volume_window e volume_mult nao podem ser negativos.")

    volumes = [float(candle.volume) for candle in candles]
    volume_ma = rolling_mean(volumes, volume_window) if volume_window else [None for _ in candles]
    atr_values = atr(candles, atr_period)
    position = 0
    highest_high = 0.0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if index < lookback or atr_values[index] is None:
            signals.append(position)
            continue

        prior_high = max(item.high for item in candles[index - lookback : index])
        volume_ok = volume_window == 0 or (volume_ma[index] is not None and candle.volume >= volume_ma[index] * volume_mult)

        if position == 0 and volume_ok and candle.close > prior_high:
            position = 1
            highest_high = candle.high
        elif position == 1:
            highest_high = max(highest_high, candle.high)
            stop = highest_high - atr_mult * atr_values[index]
            if candle.close < stop:
                position = 0
                highest_high = 0.0
        signals.append(position)

    return signals


def supertrend_follow(
    candles: list[Candle],
    *,
    atr_period: int = 10,
    atr_mult: float = 3.0,
) -> list[int]:
    if atr_period <= 0 or atr_mult <= 0:
        raise ValueError("atr_period e atr_mult precisam ser maiores que zero.")

    atr_values = atr(candles, atr_period)
    final_upper: list[float | None] = [None for _ in candles]
    final_lower: list[float | None] = [None for _ in candles]
    position = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        atr_value = atr_values[index]
        if atr_value is None:
            signals.append(position)
            continue

        middle = (candle.high + candle.low) / 2
        basic_upper = middle + atr_mult * atr_value
        basic_lower = middle - atr_mult * atr_value

        if index == 0 or final_upper[index - 1] is None or final_lower[index - 1] is None:
            final_upper[index] = basic_upper
            final_lower[index] = basic_lower
        else:
            previous = candles[index - 1]
            previous_upper = final_upper[index - 1]
            previous_lower = final_lower[index - 1]
            final_upper[index] = basic_upper if basic_upper < previous_upper or previous.close > previous_upper else previous_upper
            final_lower[index] = basic_lower if basic_lower > previous_lower or previous.close < previous_lower else previous_lower

        if position == 0 and candle.close > final_upper[index]:
            position = 1
        elif position == 1 and candle.close < final_lower[index]:
            position = 0
        signals.append(position)

    return signals


def keltner_breakout(
    candles: list[Candle],
    *,
    window: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
    exit_z: float = 0.0,
    trend_window: int = 0,
) -> list[int]:
    if window <= 0 or atr_period <= 0 or atr_mult <= 0:
        raise ValueError("window, atr_period e atr_mult precisam ser maiores que zero.")
    if trend_window < 0:
        raise ValueError("trend_window nao pode ser negativo.")

    closes = [candle.close for candle in candles]
    center = ema(closes, window)
    atr_values = atr(candles, atr_period)
    trend_ma = rolling_mean(closes, trend_window) if trend_window else [None for _ in candles]
    position = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        center_value = center[index]
        atr_value = atr_values[index]
        if center_value is None or atr_value is None:
            signals.append(position)
            continue

        upper_band = center_value + atr_mult * atr_value
        exit_band = center_value - exit_z * atr_value
        trend_ok = trend_window == 0 or (trend_ma[index] is not None and candle.close > trend_ma[index])

        if position == 0 and trend_ok and candle.close > upper_band:
            position = 1
        elif position == 1 and candle.close < exit_band:
            position = 0
        signals.append(position)

    return signals


def trend_pullback(
    candles: list[Candle],
    *,
    trend_window: int = 200,
    rsi_period: int = 14,
    lower: float = 40.0,
    upper: float = 70.0,
) -> list[int]:
    if trend_window <= 0:
        raise ValueError("trend_window precisa ser maior que zero.")
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")

    closes = [candle.close for candle in candles]
    trend_ma = rolling_mean(closes, trend_window)
    rsi_values = rsi(closes, rsi_period)
    position = 0
    signals: list[int] = []

    for index, close in enumerate(closes):
        trend_value = trend_ma[index]
        rsi_value = rsi_values[index]
        in_uptrend = trend_value is not None and close > trend_value

        if rsi_value is None or trend_value is None:
            signals.append(position)
            continue
        if position == 0 and in_uptrend and rsi_value <= lower:
            position = 1
        elif position == 1 and (not in_uptrend or rsi_value >= upper):
            position = 0
        signals.append(position)

    return signals


def sma_stop(
    candles: list[Candle],
    *,
    sma_window: int = 200,
    stop_pct: float = 0.2,
) -> list[int]:
    if sma_window <= 0:
        raise ValueError("sma_window precisa ser maior que zero.")
    if stop_pct <= 0 or stop_pct >= 1:
        raise ValueError("stop_pct precisa estar entre 0 e 1.")

    closes = [candle.close for candle in candles]
    trend_ma = rolling_mean(closes, sma_window)
    position = 0
    peak = 0.0
    signals: list[int] = []

    for index, close in enumerate(closes):
        trend_value = trend_ma[index]
        if trend_value is None:
            signals.append(position)
            continue

        if position == 0 and close > trend_value:
            position = 1
            peak = close
        elif position == 1:
            peak = max(peak, close)
            trailing_stop = close < peak * (1 - stop_pct)
            trend_break = close < trend_value
            if trailing_stop or trend_break:
                position = 0
                peak = 0.0

        signals.append(position)

    return signals


STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "atr_breakout": atr_breakout,
    "bollinger_reversion": bollinger_reversion,
    "ema_cross": ema_cross,
    "macd": macd,
    "sma_cross": sma_cross,
    "sma": sma_cross,
    "momentum": momentum,
    "price_sma": price_sma,
    "roc_trend": roc_trend,
    "breakout": breakout,
    "chandelier_breakout": chandelier_breakout,
    "connors_rsi_reversion": connors_rsi_reversion,
    "down_streak_reversion": down_streak_reversion,
    "ibs_reversion": ibs_reversion,
    "keltner_breakout": keltner_breakout,
    "rsi_bollinger": rsi_bollinger,
    "range_expansion_breakout": range_expansion_breakout,
    "rsi_cross_reversion": rsi_cross_reversion,
    "rsi_ibs_reversion": rsi_ibs_reversion,
    "rsi2_trend_reversion": rsi2_trend_reversion,
    "rsi_reversion": rsi_reversion,
    "rsi_reversion_atr": rsi_reversion_atr,
    "rsi_reversion_hold": rsi_reversion_hold,
    "rsi_reversion_trend_entry": rsi_reversion_trend_entry,
    "supertrend_follow": supertrend_follow,
    "trend_pullback": trend_pullback,
    "sma_stop": sma_stop,
}


STRATEGY_INFO = {
    "buy_and_hold": StrategyInfo(
        "buy_and_hold",
        "benchmark",
        (
            "Mantem o sinal de compra ativo em todos os candles; no teste por ativo, "
            "compra na primeira abertura e permanece comprado ate o fim."
        ),
        sweepable=False,
    ),
    "sma_cross": StrategyInfo("sma_cross", "tendencia", "Cruzamento de medias moveis simples."),
    "ema_cross": StrategyInfo("ema_cross", "tendencia", "Cruzamento de medias moveis exponenciais."),
    "macd": StrategyInfo("macd", "tendencia", "Segue a linha MACD contra a linha de sinal."),
    "price_sma": StrategyInfo("price_sma", "tendencia", "Fica comprado quando o preco fecha acima da media simples."),
    "sma_stop": StrategyInfo("sma_stop", "tendencia", "Segue media simples com stop percentual a partir do topo."),
    "supertrend_follow": StrategyInfo("supertrend_follow", "tendencia", "Segue tendencia pelo indicador SuperTrend baseado em ATR."),
    "momentum": StrategyInfo("momentum", "momentum", "Compra quando o fechamento supera o fechamento de N candles atras."),
    "roc_trend": StrategyInfo("roc_trend", "momentum", "Momentum positivo com filtro de media movel."),
    "breakout": StrategyInfo("breakout", "rompimento", "Compra rompimento de maxima e sai na perda de minima."),
    "atr_breakout": StrategyInfo("atr_breakout", "rompimento", "Rompimento de maxima com stop movel por ATR."),
    "chandelier_breakout": StrategyInfo("chandelier_breakout", "rompimento", "Rompimento com saida Chandelier/ATR e filtro opcional de volume."),
    "keltner_breakout": StrategyInfo("keltner_breakout", "rompimento", "Rompimento de canal de Keltner com filtro opcional de tendencia."),
    "range_expansion_breakout": StrategyInfo("range_expansion_breakout", "rompimento", "Compra expansao de range no fechamento com stop por ATR."),
    "bollinger_reversion": StrategyInfo("bollinger_reversion", "reversao", "Compra na banda inferior de Bollinger e sai no retorno ao centro."),
    "connors_rsi_reversion": StrategyInfo("connors_rsi_reversion", "reversao", "Reversao por Connors RSI."),
    "down_streak_reversion": StrategyInfo("down_streak_reversion", "reversao", "Compra apos sequencia de quedas com IBS baixo."),
    "ibs_reversion": StrategyInfo("ibs_reversion", "reversao", "Reversao curta por Internal Bar Strength."),
    "rsi2_trend_reversion": StrategyInfo("rsi2_trend_reversion", "reversao", "Reversao por RSI curto com filtro de tendencia e saida por media curta."),
    "rsi_bollinger": StrategyInfo("rsi_bollinger", "reversao", "Combina sobrevenda por RSI e Bollinger."),
    "rsi_cross_reversion": StrategyInfo("rsi_cross_reversion", "reversao", "Entra quando o RSI recupera acima do limite inferior."),
    "rsi_ibs_reversion": StrategyInfo("rsi_ibs_reversion", "reversao", "Combina RSI curto e IBS baixo para entrada."),
    "rsi_reversion": StrategyInfo("rsi_reversion", "reversao", "Compra sobrevenda por RSI e sai em sobrecompra."),
    "rsi_reversion_atr": StrategyInfo("rsi_reversion_atr", "reversao", "Reversao por RSI com stop de volatilidade por ATR."),
    "rsi_reversion_hold": StrategyInfo("rsi_reversion_hold", "reversao", "Reversao por RSI com tempo maximo de permanencia."),
    "rsi_reversion_trend_entry": StrategyInfo("rsi_reversion_trend_entry", "reversao", "Reversao por RSI com filtro de tendencia apenas na entrada."),
    "trend_pullback": StrategyInfo("trend_pullback", "reversao", "Compra pullback em tendencia positiva."),
}


from .additional_strategies import ADDITIONAL_PARAMETERS, ADDITIONAL_STRATEGIES

for _additional in ADDITIONAL_STRATEGIES:
    if _additional.name in STRATEGIES or _additional.name in STRATEGY_INFO:
        raise RuntimeError(f"Estrategia adicional duplicada: {_additional.name}")
    STRATEGIES[_additional.name] = _additional.function
    STRATEGY_INFO[_additional.name] = StrategyInfo(
        _additional.name,
        _additional.family,
        _additional.description,
    )


from .researched_strategies import RESEARCHED_STRATEGIES

for _researched in RESEARCHED_STRATEGIES:
    if _researched.name in STRATEGIES or _researched.name in STRATEGY_INFO:
        raise RuntimeError(f"Estrategia pesquisada duplicada: {_researched.name}")
    STRATEGIES[_researched.name] = _researched.function
    STRATEGY_INFO[_researched.name] = StrategyInfo(
        _researched.name,
        _researched.family,
        _researched.description,
    )


from .extended_strategies import EXTENDED_STRATEGIES

for _extended in EXTENDED_STRATEGIES:
    if _extended.name in STRATEGIES or _extended.name in STRATEGY_INFO:
        raise RuntimeError(f"Estrategia estendida duplicada: {_extended.name}")
    STRATEGIES[_extended.name] = _extended.function
    STRATEGY_INFO[_extended.name] = StrategyInfo(
        _extended.name,
        _extended.family,
        _extended.description,
    )


# Importar o módulo do usuário executa apenas os decorators declarados nele.
from . import user_extensions as _user_extensions  # noqa: E402,F401
from .extensions import registered_strategies  # noqa: E402

for _extension in registered_strategies():
    if _extension.name in STRATEGIES or _extension.name in STRATEGY_INFO:
        raise RuntimeError(f"Estratégia de extensão duplicada: {_extension.name}")
    STRATEGIES[_extension.name] = _extension.function
    STRATEGY_INFO[_extension.name] = StrategyInfo(
        _extension.name,
        _extension.family,
        _extension.description,
    )


def available_strategies() -> list[str]:
    return sorted(name for name in STRATEGIES if name != "sma")


def sweep_strategies() -> list[str]:
    return sorted(name for name, info in STRATEGY_INFO.items() if info.sweepable)


def portfolio_strategies() -> list[str]:
    """Estrategias combinadas por padrao com os gerenciamentos de carteira.

    Buy and hold nao possui parametros para otimizar, mas funciona como um sinal
    de elegibilidade permanente na matriz: a selecao, os pesos e os
    rebalanceamentos continuam sendo definidos pelo gerenciamento de carteira.
    """

    return sorted({*sweep_strategies(), "buy_and_hold"})


def strategy_info(strategy_name: str) -> StrategyInfo:
    name = strategy_name.strip().lower()
    if name == "sma":
        name = "sma_cross"
    if name not in STRATEGY_INFO:
        available = ", ".join(available_strategies())
        raise ValueError(f"Estrategia desconhecida: {strategy_name}. Disponiveis: {available}.")
    return STRATEGY_INFO[name]


def strategy_parameters(strategy_name: str) -> dict[str, object]:
    name = strategy_name.strip().lower()
    if name == "sma":
        name = "sma_cross"
    if name in ADDITIONAL_PARAMETERS:
        return dict(ADDITIONAL_PARAMETERS[name])
    strategy = STRATEGIES[name]
    parameters: dict[str, object] = {}
    for key, parameter in inspect.signature(strategy).parameters.items():
        if key in {"candles", "session_calendar"}:
            continue
        parameters[key] = "" if parameter.default is inspect.Signature.empty else parameter.default
    return parameters


def strategies_by_family() -> dict[str, list[StrategyInfo]]:
    grouped: dict[str, list[StrategyInfo]] = {}
    for name in available_strategies():
        info = strategy_info(name)
        grouped.setdefault(info.family, []).append(info)
    return {family: grouped[family] for family in sorted(grouped)}


def build_signals(strategy_name: str, candles: list[Candle], **params) -> list[int]:
    name = strategy_name.strip().lower()
    if name not in STRATEGIES:
        available = ", ".join(available_strategies())
        raise ValueError(f"Estrategia desconhecida: {strategy_name}. Disponiveis: {available}.")

    strategy = STRATEGIES[name]
    signature = inspect.signature(strategy)
    supported_params = {
        key: value
        for key, value in params.items()
        if key in signature.parameters and value is not None
    }
    raw_signals = strategy(candles, **supported_params)
    if len(raw_signals) != len(candles):
        raise RuntimeError("A estrategia retornou uma quantidade de sinais diferente dos candles.")
    for index, signal in enumerate(raw_signals):
        if not isinstance(signal, (bool, Integral)) or int(signal) not in (0, 1):
            raise RuntimeError(
                f"A estrategia retornou sinal nao binario no indice {index}: {signal!r}."
            )
    return [int(signal) for signal in raw_signals]


def rolling_mean(values: list[float], window: int) -> list[float | None]:
    means: list[float | None] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        if index + 1 < window:
            means.append(None)
        else:
            means.append(running_sum / window)
    return means


def ema(values: list[float], window: int) -> list[float | None]:
    return ema_optional([float(value) for value in values], window)


def ema_optional(values: list[float | None], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window precisa ser maior que zero.")

    result: list[float | None] = []
    alpha = 2 / (window + 1)
    seed_values: list[float] = []
    current: float | None = None

    for value in values:
        if value is None:
            result.append(None)
            continue
        if current is None:
            seed_values.append(value)
            if len(seed_values) < window:
                result.append(None)
                continue
            current = sum(seed_values) / window
        else:
            current = value * alpha + current * (1 - alpha)
        result.append(current)

    return result


def rolling_bands(values: list[float], window: int, num_std: float) -> list[tuple[float, float, float, float] | None]:
    bands: list[tuple[float, float, float, float] | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            bands.append(None)
            continue
        sample = values[index + 1 - window : index + 1]
        middle = sum(sample) / window
        std = statistics.pstdev(sample)
        lower = middle - num_std * std
        upper = middle + num_std * std
        bands.append((middle, lower, upper, std))
    return bands


def atr(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    if not candles:
        return []

    true_ranges: list[float] = []
    for index, candle in enumerate(candles):
        if index == 0:
            true_ranges.append(candle.high - candle.low)
            continue
        previous_close = candles[index - 1].close
        true_ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )

    result: list[float | None] = [None for _ in candles]
    if len(true_ranges) < period:
        return result

    current = sum(true_ranges[:period]) / period
    result[period - 1] = current
    for index in range(period, len(true_ranges)):
        current = (current * (period - 1) + true_ranges[index]) / period
        result[index] = current
    return result


def close_streaks(values: list[float]) -> list[float]:
    if not values:
        return []

    streaks = [0.0]
    streak = 0
    for index in range(1, len(values)):
        if values[index] > values[index - 1]:
            streak = streak + 1 if streak > 0 else 1
        elif values[index] < values[index - 1]:
            streak = streak - 1 if streak < 0 else -1
        else:
            streak = 0
        streaks.append(float(streak))
    return streaks


def roc_percent_rank(values: list[float], period: int) -> list[float | None]:
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")

    returns = [None]
    for index in range(1, len(values)):
        previous = values[index - 1]
        returns.append(None if previous == 0 else values[index] / previous - 1)

    ranks: list[float | None] = []
    for index, value in enumerate(returns):
        if value is None or index < period:
            ranks.append(None)
            continue
        sample = [item for item in returns[index - period : index] if item is not None]
        if not sample:
            ranks.append(None)
            continue
        below = sum(1 for item in sample if item < value)
        ranks.append(100 * below / len(sample))
    return ranks


def internal_bar_strength(candles: list[Candle]) -> list[float]:
    values: list[float] = []
    for candle in candles:
        range_size = candle.high - candle.low
        if range_size <= 0:
            values.append(0.5)
        else:
            values.append((candle.close - candle.low) / range_size)
    return values


def rsi(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    if len(values) <= period:
        return [None for _ in values]

    result: list[float | None] = [None for _ in values]
    gains: list[float] = []
    losses: list[float] = []

    for index in range(1, period + 1):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result[period] = _rsi_value(avg_gain, avg_loss)

    for index in range(period + 1, len(values)):
        change = values[index] - values[index - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        result[index] = _rsi_value(avg_gain, avg_loss)

    return result


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    relative_strength = avg_gain / avg_loss
    return 100 - (100 / (1 + relative_strength))
