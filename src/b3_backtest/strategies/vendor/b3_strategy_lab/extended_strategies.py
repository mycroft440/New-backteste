from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from .candles import Candle


SignalFunction = Callable[..., list[int]]


@dataclass(frozen=True)
class ExtendedStrategy:
    name: str
    family: str
    description: str
    function: SignalFunction


def fisher_transform_reversal(
    candles: list[Candle],
    *,
    period: int = 10,
    lower: float = -1.5,
    upper: float = 1.5,
) -> list[int]:
    """Buy an oversold Fisher turn and leave on an overbought downturn."""
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")

    prices = [(candle.high + candle.low) / 2 for candle in candles]
    fisher: list[float | None] = []
    normalized = 0.0
    previous_fisher = 0.0

    for index, price in enumerate(prices):
        if index + 1 < period:
            fisher.append(None)
            continue
        sample = prices[index + 1 - period : index + 1]
        highest = max(sample)
        lowest = min(sample)
        raw = 0.0 if highest == lowest else 2 * ((price - lowest) / (highest - lowest) - 0.5)
        normalized = max(-0.999, min(0.999, 0.33 * raw + 0.67 * normalized))
        current = 0.5 * math.log((1 + normalized) / (1 - normalized)) + 0.5 * previous_fisher
        fisher.append(current)
        previous_fisher = current

    position = 0
    signals: list[int] = []
    for index, value in enumerate(fisher):
        previous = fisher[index - 1] if index else None
        if value is not None and previous is not None:
            if position == 0 and previous <= lower and value > previous:
                position = 1
            elif position == 1 and previous >= upper and value < previous:
                position = 0
        signals.append(position)
    return signals


def laguerre_rsi_reversal(
    candles: list[Candle],
    *,
    gamma: float = 0.5,
    lower: float = 0.2,
    upper: float = 0.8,
) -> list[int]:
    """Trade recoveries from the Laguerre RSI extreme zones."""
    if not 0 < gamma < 1:
        raise ValueError("gamma precisa estar entre 0 e 1.")
    if not 0 <= lower < upper <= 1:
        raise ValueError("lower e upper precisam estar entre 0 e 1, com lower < upper.")
    if not candles:
        return []

    l0 = l1 = l2 = l3 = candles[0].close
    oscillator: list[float] = [0.5]
    for candle in candles[1:]:
        previous_l0, previous_l1, previous_l2, previous_l3 = l0, l1, l2, l3
        l0 = (1 - gamma) * candle.close + gamma * previous_l0
        l1 = -gamma * l0 + previous_l0 + gamma * previous_l1
        l2 = -gamma * l1 + previous_l1 + gamma * previous_l2
        l3 = -gamma * l2 + previous_l2 + gamma * previous_l3

        cu = max(l0 - l1, 0.0) + max(l1 - l2, 0.0) + max(l2 - l3, 0.0)
        cd = max(l1 - l0, 0.0) + max(l2 - l1, 0.0) + max(l3 - l2, 0.0)
        oscillator.append(0.5 if cu + cd == 0 else cu / (cu + cd))

    position = 0
    signals = [0]
    for index in range(1, len(oscillator)):
        previous = oscillator[index - 1]
        current = oscillator[index]
        if position == 0 and previous <= lower < current:
            position = 1
        elif position == 1 and previous >= upper > current:
            position = 0
        signals.append(position)
    return signals


def ichimoku_cloud(
    candles: list[Candle],
    *,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> list[int]:
    """Follow price above the causally aligned Ichimoku cloud."""
    if min(tenkan_period, kijun_period, span_b_period, displacement) <= 0:
        raise ValueError("os periodos e displacement precisam ser maiores que zero.")
    if not tenkan_period < kijun_period < span_b_period:
        raise ValueError("use tenkan_period < kijun_period < span_b_period.")

    tenkan = _midpoint_channels(candles, tenkan_period)
    kijun = _midpoint_channels(candles, kijun_period)
    span_b_base = _midpoint_channels(candles, span_b_period)
    span_a_base = [
        None if fast is None or slow is None else (fast + slow) / 2
        for fast, slow in zip(tenkan, kijun)
    ]

    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        source = index - displacement
        if source < 0 or tenkan[index] is None or kijun[index] is None:
            signals.append(position)
            continue
        span_a = span_a_base[source]
        span_b = span_b_base[source]
        if span_a is None or span_b is None:
            signals.append(position)
            continue

        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        if position == 0 and candle.close > cloud_top and tenkan[index] > kijun[index]:
            position = 1
        elif position == 1 and (candle.close < cloud_bottom or tenkan[index] < kijun[index]):
            position = 0
        signals.append(position)
    return signals


def parabolic_sar_trend(
    candles: list[Candle],
    *,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> list[int]:
    """Stay long while Wilder's Parabolic SAR is in its rising state."""
    if af_step <= 0 or af_max <= 0 or af_step > af_max:
        raise ValueError("use 0 < af_step <= af_max.")
    if not candles:
        return []
    if len(candles) == 1:
        return [0]

    rising = candles[1].close >= candles[0].close
    sar = candles[0].low if rising else candles[0].high
    extreme = candles[0].high if rising else candles[0].low
    acceleration = af_step
    signals = [0]

    for index in range(1, len(candles)):
        candle = candles[index]
        candidate = sar + acceleration * (extreme - sar)
        if rising:
            candidate = min(candidate, candles[index - 1].low)
            if index >= 2:
                candidate = min(candidate, candles[index - 2].low)
            if candle.low < candidate:
                rising = False
                sar = extreme
                extreme = candle.low
                acceleration = af_step
            else:
                sar = candidate
                if candle.high > extreme:
                    extreme = candle.high
                    acceleration = min(af_max, acceleration + af_step)
        else:
            candidate = max(candidate, candles[index - 1].high)
            if index >= 2:
                candidate = max(candidate, candles[index - 2].high)
            if candle.high > candidate:
                rising = True
                sar = extreme
                extreme = candle.high
                acceleration = af_step
            else:
                sar = candidate
                if candle.low < extreme:
                    extreme = candle.low
                    acceleration = min(af_max, acceleration + af_step)
        signals.append(int(rising and candle.close > sar))
    return signals


def aroon_trend(
    candles: list[Candle],
    *,
    period: int = 25,
    strong_level: float = 70.0,
) -> list[int]:
    """Use recent-high/recent-low timing to identify a strong Aroon trend."""
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")
    if not 0 < strong_level <= 100:
        raise ValueError("strong_level precisa estar entre 0 e 100.")

    position = 0
    signals: list[int] = []
    for index in range(len(candles)):
        if index < period:
            signals.append(position)
            continue
        sample = candles[index - period : index + 1]
        high_offset = _latest_extreme_offset([candle.high for candle in sample], maximum=True)
        low_offset = _latest_extreme_offset([candle.low for candle in sample], maximum=False)
        aroon_up = 100 * high_offset / period
        aroon_down = 100 * low_offset / period
        if position == 0 and aroon_up >= strong_level and aroon_up > aroon_down:
            position = 1
        elif position == 1 and aroon_down >= strong_level and aroon_down > aroon_up:
            position = 0
        signals.append(position)
    return signals


def trix_signal(
    candles: list[Candle],
    *,
    period: int = 15,
    signal_period: int = 9,
) -> list[int]:
    """Trade the TRIX line against its exponential signal line."""
    if period <= 1 or signal_period <= 1:
        raise ValueError("period e signal_period precisam ser maiores que 1.")
    if not candles:
        return []

    first = _ema([candle.close for candle in candles], period)
    second = _ema_optional(first, period)
    third = _ema_optional(second, period)
    trix: list[float | None] = [None]
    for index in range(1, len(candles)):
        current = third[index]
        previous = third[index - 1]
        trix.append(None if current is None or previous in (None, 0) else 100 * (current / previous - 1))
    signal = _ema_optional(trix, signal_period)
    return [
        int(value is not None and average is not None and value > average)
        for value, average in zip(trix, signal)
    ]


def schaff_trend_cycle(
    candles: list[Candle],
    *,
    fast_period: int = 23,
    slow_period: int = 50,
    cycle_period: int = 10,
    smoothing: float = 0.5,
    lower: float = 25.0,
    upper: float = 75.0,
) -> list[int]:
    """Use Schaff's double-stochastic MACD cycle with hysteresis zones."""
    if min(fast_period, slow_period, cycle_period) <= 1:
        raise ValueError("os periodos precisam ser maiores que 1.")
    if fast_period >= slow_period:
        raise ValueError("fast_period precisa ser menor que slow_period.")
    if not 0 < smoothing <= 1:
        raise ValueError("smoothing precisa estar entre 0 e 1.")
    if not 0 <= lower < upper <= 100:
        raise ValueError("lower e upper precisam estar entre 0 e 100, com lower < upper.")

    closes = [candle.close for candle in candles]
    fast = _ema(closes, fast_period)
    slow = _ema(closes, slow_period)
    macd = [None if one is None or two is None else one - two for one, two in zip(fast, slow)]
    stochastic_one = _stochastic_optional(macd, cycle_period)
    smooth_one = _recursive_smooth(stochastic_one, smoothing)
    stochastic_two = _stochastic_optional(smooth_one, cycle_period)
    cycle = _recursive_smooth(stochastic_two, smoothing)

    position = 0
    signals: list[int] = []
    for index, value in enumerate(cycle):
        previous = cycle[index - 1] if index else None
        if value is not None and previous is not None:
            if position == 0 and previous <= lower < value:
                position = 1
            elif position == 1 and previous >= upper > value:
                position = 0
        signals.append(position)
    return signals


def coppock_curve(
    candles: list[Candle],
    *,
    short_roc: int = 11,
    long_roc: int = 14,
    wma_period: int = 10,
) -> list[int]:
    """Buy a negative Coppock upturn and leave on a positive downturn."""
    if min(short_roc, long_roc, wma_period) <= 0:
        raise ValueError("os periodos precisam ser maiores que zero.")
    if short_roc >= long_roc:
        raise ValueError("short_roc precisa ser menor que long_roc.")

    closes = [candle.close for candle in candles]
    short = _rate_of_change(closes, short_roc)
    long = _rate_of_change(closes, long_roc)
    combined = [None if one is None or two is None else one + two for one, two in zip(short, long)]
    curve = _wma_optional(combined, wma_period)
    position = 0
    signals: list[int] = []
    for index, value in enumerate(curve):
        previous = curve[index - 1] if index else None
        if value is not None and previous is not None:
            if position == 0 and previous < 0 and value > previous:
                position = 1
            elif position == 1 and previous > 0 and value < previous:
                position = 0
        signals.append(position)
    return signals


def know_sure_thing(
    candles: list[Candle],
    *,
    roc1: int = 10,
    roc2: int = 15,
    roc3: int = 20,
    roc4: int = 30,
    sma1: int = 10,
    sma2: int = 10,
    sma3: int = 10,
    sma4: int = 15,
    signal_period: int = 9,
) -> list[int]:
    """Trade Pring's four-horizon Know Sure Thing against its signal line."""
    periods = (roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4, signal_period)
    if min(periods) <= 0:
        raise ValueError("todos os periodos precisam ser maiores que zero.")
    if not roc1 < roc2 < roc3 < roc4:
        raise ValueError("use roc1 < roc2 < roc3 < roc4.")

    closes = [candle.close for candle in candles]
    components = [
        _sma_optional(_rate_of_change(closes, roc_period), average_period)
        for roc_period, average_period in ((roc1, sma1), (roc2, sma2), (roc3, sma3), (roc4, sma4))
    ]
    kst: list[float | None] = []
    for values in zip(*components):
        kst.append(None if any(value is None for value in values) else sum((index + 1) * value for index, value in enumerate(values) if value is not None))
    signal = _sma_optional(kst, signal_period)
    return [int(value is not None and average is not None and value > average) for value, average in zip(kst, signal)]


def true_strength_index(
    candles: list[Candle],
    *,
    long_period: int = 25,
    short_period: int = 13,
    signal_period: int = 7,
) -> list[int]:
    """Trade Blau's double-smoothed momentum ratio against its signal line."""
    if min(long_period, short_period, signal_period) <= 0:
        raise ValueError("os periodos precisam ser maiores que zero.")
    if not candles:
        return []

    closes = [candle.close for candle in candles]
    momentum = [0.0] + [closes[index] - closes[index - 1] for index in range(1, len(closes))]
    absolute = [abs(value) for value in momentum]
    smooth_momentum = _ema_optional(_ema(momentum, long_period), short_period)
    smooth_absolute = _ema_optional(_ema(absolute, long_period), short_period)
    tsi: list[float | None] = []
    for numerator, denominator in zip(smooth_momentum, smooth_absolute):
        tsi.append(None if numerator is None or denominator in (None, 0) else 100 * numerator / denominator)
    signal = _ema_optional(tsi, signal_period)
    return [int(value is not None and average is not None and value > average) for value, average in zip(tsi, signal)]


def awesome_oscillator(
    candles: list[Candle],
    *,
    fast_period: int = 5,
    slow_period: int = 34,
) -> list[int]:
    """Follow Bill Williams' median-price momentum above its zero line."""
    if fast_period <= 0 or slow_period <= 0 or fast_period >= slow_period:
        raise ValueError("use 0 < fast_period < slow_period.")
    median = [(candle.high + candle.low) / 2 for candle in candles]
    fast = _sma(median, fast_period)
    slow = _sma(median, slow_period)
    return [int(one is not None and two is not None and one - two > 0) for one, two in zip(fast, slow)]


def choppiness_breakout(
    candles: list[Candle],
    *,
    period: int = 14,
    high_level: float = 61.8,
    low_level: float = 38.2,
    trend_window: int = 20,
    atr_period: int = 14,
    atr_mult: float = 3.0,
) -> list[int]:
    """Buy an upside breakout when a choppy regime expands into a trend."""
    if min(period, trend_window, atr_period) <= 1 or atr_mult <= 0:
        raise ValueError("period, trend_window e atr_period precisam ser maiores que 1; atr_mult deve ser positivo.")
    if not 0 <= low_level < high_level <= 100:
        raise ValueError("use 0 <= low_level < high_level <= 100.")

    closes = [candle.close for candle in candles]
    true_ranges = _true_ranges(candles)
    trend = _sma(closes, trend_window)
    atr_values = _atr(candles, atr_period)
    chop: list[float | None] = []
    for index in range(len(candles)):
        if index + 1 < period:
            chop.append(None)
            continue
        sample = candles[index + 1 - period : index + 1]
        price_range = max(item.high for item in sample) - min(item.low for item in sample)
        tr_sum = sum(true_ranges[index + 1 - period : index + 1])
        chop.append(None if price_range <= 0 or tr_sum <= 0 else 100 * math.log10(tr_sum / price_range) / math.log10(period))

    armed = False
    position = 0
    peak = 0.0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        value = chop[index]
        if value is not None and value >= high_level:
            armed = True
        if position == 0 and armed and value is not None and value <= low_level and index >= period:
            prior_high = max(item.high for item in candles[index - period : index])
            if candle.close > prior_high:
                position = 1
                peak = candle.close
                armed = False
        elif position == 1:
            peak = max(peak, candle.close)
            average = trend[index]
            atr_value = atr_values[index]
            trend_exit = average is not None and candle.close < average
            volatility_exit = atr_value is not None and candle.close < peak - atr_mult * atr_value
            if (value is not None and value >= high_level) or trend_exit or volatility_exit:
                position = 0
                peak = 0.0
        signals.append(position)
    return signals


def elder_force_index(
    candles: list[Candle],
    *,
    period: int = 13,
    trend_window: int = 50,
) -> list[int]:
    """Require positive smoothed force and a rising price regime."""
    if period <= 0 or trend_window <= 0:
        raise ValueError("period e trend_window precisam ser maiores que zero.")
    if not candles:
        return []
    closes = [candle.close for candle in candles]
    force = [0.0] + [
        (closes[index] - closes[index - 1]) * candles[index].volume
        for index in range(1, len(candles))
    ]
    smooth = _ema(force, period)
    trend = _sma(closes, trend_window)
    return [
        int(value is not None and average is not None and value > 0 and close > average)
        for close, value, average in zip(closes, smooth, trend)
    ]


def ease_of_movement(candles: list[Candle], *, period: int = 14) -> list[int]:
    """Follow the smoothed Ease of Movement oscillator above zero."""
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    if not candles:
        return []
    raw = [0.0]
    for index in range(1, len(candles)):
        candle = candles[index]
        previous = candles[index - 1]
        distance = (candle.high + candle.low - previous.high - previous.low) / 2
        raw.append(0.0 if candle.volume <= 0 else distance * (candle.high - candle.low) / candle.volume)
    smooth = _sma(raw, period)
    return [int(value is not None and value > 0) for value in smooth]


def negative_volume_index(candles: list[Candle], *, ema_period: int = 255) -> list[int]:
    """Follow Fosback's Negative Volume Index above its one-year EMA."""
    if ema_period <= 1:
        raise ValueError("ema_period precisa ser maior que 1.")
    if not candles:
        return []
    index_values = [1000.0]
    for index in range(1, len(candles)):
        value = index_values[-1]
        previous_close = candles[index - 1].close
        if candles[index].volume < candles[index - 1].volume and previous_close != 0:
            value *= candles[index].close / previous_close
        index_values.append(value)
    average = _ema(index_values, ema_period)
    return [int(value is not None and mean is not None and value > mean) for value, mean in zip(index_values, average)]


def klinger_volume_oscillator(
    candles: list[Candle],
    *,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
) -> list[int]:
    """Trade Klinger's volume-force oscillator against its signal EMA."""
    if min(fast_period, slow_period, signal_period) <= 0 or fast_period >= slow_period:
        raise ValueError("use periodos positivos e fast_period < slow_period.")
    if not candles:
        return []

    volume_force = [0.0]
    previous_trend = 0
    previous_dm = max(candles[0].high - candles[0].low, 0.0)
    cumulative = previous_dm
    for index in range(1, len(candles)):
        candle = candles[index]
        previous = candles[index - 1]
        trend = 1 if candle.high + candle.low + candle.close > previous.high + previous.low + previous.close else -1
        dm = max(candle.high - candle.low, 0.0)
        cumulative = cumulative + dm if trend == previous_trend else previous_dm + dm
        ratio = 0.0 if cumulative == 0 else abs(2 * (dm / cumulative - 1))
        volume_force.append(candle.volume * ratio * trend * 100)
        previous_trend = trend
        previous_dm = dm

    fast = _ema(volume_force, fast_period)
    slow = _ema(volume_force, slow_period)
    oscillator = [None if one is None or two is None else one - two for one, two in zip(fast, slow)]
    signal = _ema_optional(oscillator, signal_period)
    return [int(value is not None and average is not None and value > average) for value, average in zip(oscillator, signal)]


def mass_index_reversal(
    candles: list[Candle],
    *,
    ema_period: int = 9,
    sum_period: int = 25,
    bulge_level: float = 27.0,
    trigger_level: float = 26.5,
    exit_window: int = 9,
    hold_limit: int = 20,
) -> list[int]:
    """Buy the long side of a Mass Index reversal bulge and use EMA/time exits."""
    if min(ema_period, sum_period, exit_window, hold_limit) <= 0:
        raise ValueError("os periodos e hold_limit precisam ser maiores que zero.")
    if trigger_level >= bulge_level:
        raise ValueError("trigger_level precisa ser menor que bulge_level.")

    ranges = [max(candle.high - candle.low, 0.0) for candle in candles]
    first = _ema(ranges, ema_period)
    second = _ema_optional(first, ema_period)
    ratio = [None if one is None or two in (None, 0) else one / two for one, two in zip(first, second)]
    mass = _rolling_sum_optional(ratio, sum_period)
    price_average = _ema([candle.close for candle in candles], exit_window)
    armed = False
    position = 0
    held = 0
    signals: list[int] = []

    for index, value in enumerate(mass):
        if value is not None and value >= bulge_level:
            armed = True
        if position == 0 and armed and value is not None and value < trigger_level:
            average = price_average[index]
            previous_average = price_average[index - 1] if index else None
            if average is not None and previous_average is not None and average < previous_average:
                position = 1
                held = 0
                armed = False
        elif position == 1:
            held += 1
            average = price_average[index]
            if (average is not None and candles[index].close >= average) or held >= hold_limit:
                position = 0
                held = 0
        signals.append(position)
    return signals


def vertical_horizontal_filter(
    candles: list[Candle],
    *,
    period: int = 28,
    entry_level: float = 0.4,
    exit_level: float = 0.25,
    trend_window: int = 50,
) -> list[int]:
    """Use VHF to enter directional regimes and leave when trendiness fades."""
    if period <= 1 or trend_window <= 1:
        raise ValueError("period e trend_window precisam ser maiores que 1.")
    if not 0 <= exit_level < entry_level <= 1:
        raise ValueError("use 0 <= exit_level < entry_level <= 1.")

    closes = [candle.close for candle in candles]
    trend = _sma(closes, trend_window)
    position = 0
    signals: list[int] = []
    for index, close in enumerate(closes):
        # TTR uses an n-close high/low range divided by n one-bar close
        # changes, so the first value requires one extra close.
        if index < period:
            signals.append(position)
            continue
        sample = closes[index + 1 - period : index + 1]
        denominator = sum(abs(closes[item] - closes[item - 1]) for item in range(index + 1 - period, index + 1))
        vhf = 0.0 if denominator == 0 else (max(sample) - min(sample)) / denominator
        average = trend[index]
        if position == 0 and average is not None and vhf >= entry_level and close > average:
            position = 1
        elif position == 1 and (vhf <= exit_level or (average is not None and close < average)):
            position = 0
        signals.append(position)
    return signals


def nr7_breakout(
    candles: list[Candle],
    *,
    setup_period: int = 7,
    expiry: int = 5,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    hold_limit: int = 20,
) -> list[int]:
    """Break the high of the narrowest range in seven bars, with ATR/time exits."""
    if setup_period <= 1 or expiry <= 0 or atr_period <= 0 or atr_mult <= 0 or hold_limit <= 0:
        raise ValueError("periodos, expiry, atr_mult e hold_limit precisam ser positivos.")
    atr_values = _atr(candles, atr_period)
    armed_high: float | None = None
    armed_low = 0.0
    expires_at = -1
    position = 0
    peak = 0.0
    held = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if position == 0:
            if armed_high is not None and index <= expires_at and candle.close > armed_high:
                position = 1
                peak = candle.close
                held = 0
            elif index > expires_at:
                armed_high = None

            if position == 0 and index + 1 >= setup_period:
                ranges = [item.high - item.low for item in candles[index + 1 - setup_period : index + 1]]
                if ranges[-1] <= min(ranges[:-1]):
                    armed_high = candle.high
                    armed_low = candle.low
                    expires_at = index + expiry
        else:
            held += 1
            peak = max(peak, candle.close)
            atr_value = atr_values[index]
            atr_exit = atr_value is not None and candle.close < peak - atr_mult * atr_value
            if candle.close < armed_low or atr_exit or held >= hold_limit:
                position = 0
                armed_high = None
                peak = 0.0
                held = 0
        signals.append(position)
    return signals


def inside_bar_breakout(
    candles: list[Candle],
    *,
    expiry: int = 5,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    hold_limit: int = 20,
) -> list[int]:
    """Break the mother-bar high after an inside bar, with deterministic exits."""
    if expiry <= 0 or atr_period <= 0 or atr_mult <= 0 or hold_limit <= 0:
        raise ValueError("expiry, atr_period, atr_mult e hold_limit precisam ser positivos.")
    atr_values = _atr(candles, atr_period)
    armed_high: float | None = None
    armed_midpoint = 0.0
    expires_at = -1
    position = 0
    peak = 0.0
    held = 0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if position == 0:
            if armed_high is not None and index <= expires_at and candle.close > armed_high:
                position = 1
                peak = candle.close
                held = 0
            elif index > expires_at:
                armed_high = None

            if position == 0 and index > 0:
                mother = candles[index - 1]
                if candle.high < mother.high and candle.low > mother.low:
                    armed_high = mother.high
                    armed_midpoint = (mother.high + mother.low) / 2
                    expires_at = index + expiry
        else:
            held += 1
            peak = max(peak, candle.close)
            atr_value = atr_values[index]
            atr_exit = atr_value is not None and candle.close < peak - atr_mult * atr_value
            if candle.close < armed_midpoint or atr_exit or held >= hold_limit:
                position = 0
                armed_high = None
                peak = 0.0
                held = 0
        signals.append(position)
    return signals


def halloween_effect(
    candles: list[Candle],
    *,
    entry_month: int = 11,
    exit_month: int = 5,
) -> list[int]:
    """Hold during the November-April half-year and stay in cash otherwise."""
    if not 1 <= entry_month <= 12 or not 1 <= exit_month <= 12:
        raise ValueError("entry_month e exit_month precisam estar entre 1 e 12.")
    if entry_month == exit_month:
        raise ValueError("entry_month e exit_month precisam ser diferentes.")

    sessions = [date.fromisoformat(candle.date.split(" ", 1)[0]) for candle in candles]

    def is_invested(month: int) -> bool:
        if entry_month > exit_month:
            return month >= entry_month or month < exit_month
        return entry_month <= month < exit_month

    def next_business_month(session: date) -> int:
        # Calendar information is known ex ante. We only need the month of the
        # next potential trading day; weekends are skipped. No future candle,
        # price, volume, or observed session is read.
        candidate = session + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate.month

    # A close signal is the desired position for the following open. This remains
    # prefix-causal even when the current candle is the final candle in the input.
    return [int(is_invested(next_business_month(session))) for session in sessions]


EXTENDED_STRATEGIES = (
    ExtendedStrategy("fisher_transform_reversal", "reversao", "Fisher Transform: compra a virada na sobrevenda e sai na virada da sobrecompra.", fisher_transform_reversal),
    ExtendedStrategy("laguerre_rsi_reversal", "reversao", "Laguerre RSI de Ehlers: recuperacao da sobrevenda com saida apos sobrecompra.", laguerre_rsi_reversal),
    ExtendedStrategy("ichimoku_cloud", "tendencia", "Ichimoku causal: preco acima da nuvem e Tenkan acima da Kijun.", ichimoku_cloud),
    ExtendedStrategy("parabolic_sar_trend", "tendencia", "Parabolic SAR de Wilder: comprado apenas no estado ascendente.", parabolic_sar_trend),
    ExtendedStrategy("aroon_trend", "tendencia", "Aroon: segue novas maximas e sai quando novas minimas dominam.", aroon_trend),
    ExtendedStrategy("trix_signal", "momentum", "TRIX: cruzamento da variacao da tripla EMA contra sua linha de sinal.", trix_signal),
    ExtendedStrategy("schaff_trend_cycle", "momentum", "Schaff Trend Cycle: ciclo estocastico duplo do MACD com zonas de histerese.", schaff_trend_cycle),
    ExtendedStrategy("coppock_curve", "momentum", "Coppock: compra a inflexao negativa para cima e sai na inflexao positiva.", coppock_curve),
    ExtendedStrategy("know_sure_thing", "momentum", "Know Sure Thing de Pring: quatro horizontes de ROC contra a linha de sinal.", know_sure_thing),
    ExtendedStrategy("true_strength_index", "momentum", "True Strength Index: momentum duplamente suavizado contra a linha de sinal.", true_strength_index),
    ExtendedStrategy("awesome_oscillator", "momentum", "Awesome Oscillator: momentum do preco mediano acima ou abaixo de zero.", awesome_oscillator),
    ExtendedStrategy("choppiness_breakout", "volatilidade", "Rompimento depois que o Choppiness Index sai de compressao para tendencia.", choppiness_breakout),
    ExtendedStrategy("elder_force_index", "volume", "Force Index de Elder positivo com confirmacao da tendencia de preco.", elder_force_index),
    ExtendedStrategy("ease_of_movement", "volume", "Ease of Movement suavizado: comprado quando preco e volume favorecem alta.", ease_of_movement),
    ExtendedStrategy("negative_volume_index", "volume", "Negative Volume Index de Fosback acima de sua EMA anual.", negative_volume_index),
    ExtendedStrategy("klinger_volume_oscillator", "volume", "Klinger Volume Oscillator acima da EMA de sinal.", klinger_volume_oscillator),
    ExtendedStrategy("mass_index_reversal", "reversao", "Mass Index: lado comprador da reversal bulge com saida por EMA ou tempo.", mass_index_reversal),
    ExtendedStrategy("vertical_horizontal_filter", "regime", "VHF: entra em tendencia altista direcional e sai quando o regime enfraquece.", vertical_horizontal_filter),
    ExtendedStrategy("nr7_breakout", "price_action", "NR7 de Crabel: rompe a maxima do menor range em sete barras.", nr7_breakout),
    ExtendedStrategy("inside_bar_breakout", "price_action", "Inside Bar: rompe a maxima da barra-mae com saidas objetivas.", inside_bar_breakout),
    ExtendedStrategy("halloween_effect", "sazonalidade", "Efeito Halloween: comprado de novembro a abril e em caixa de maio a outubro.", halloween_effect),
)


def _midpoint_channels(candles: list[Candle], period: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(candles)):
        if index + 1 < period:
            result.append(None)
            continue
        sample = candles[index + 1 - period : index + 1]
        result.append((max(item.high for item in sample) + min(item.low for item in sample)) / 2)
    return result


def _latest_extreme_offset(values: list[float], *, maximum: bool) -> int:
    extreme = max(values) if maximum else min(values)
    return max(index for index, value in enumerate(values) if value == extreme)


def _sma(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        result.append(None if index + 1 < period else running / period)
    return result


def _sma_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    sample: list[float] = []
    for value in values:
        if value is None:
            sample = []
            result.append(None)
            continue
        sample.append(value)
        if len(sample) > period:
            sample.pop(0)
        result.append(sum(sample) / period if len(sample) == period else None)
    return result


def _ema(values: list[float], period: int) -> list[float | None]:
    return _ema_optional([float(value) for value in values], period)


def _ema_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    alpha = 2 / (period + 1)
    seed: list[float] = []
    current: float | None = None
    result: list[float | None] = []
    for value in values:
        if value is None:
            seed = []
            current = None
            result.append(None)
            continue
        if current is None:
            seed.append(value)
            if len(seed) < period:
                result.append(None)
                continue
            current = sum(seed) / period
        else:
            current = alpha * value + (1 - alpha) * current
        result.append(current)
    return result


def _wma_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    denominator = period * (period + 1) / 2
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < period:
            result.append(None)
            continue
        sample = values[index + 1 - period : index + 1]
        if any(value is None for value in sample):
            result.append(None)
        else:
            result.append(sum((weight + 1) * value for weight, value in enumerate(sample) if value is not None) / denominator)
    return result


def _rate_of_change(values: list[float], period: int) -> list[float | None]:
    result: list[float | None] = []
    for index, value in enumerate(values):
        if index < period or values[index - period] == 0:
            result.append(None)
        else:
            result.append(100 * (value / values[index - period] - 1))
    return result


def _stochastic_optional(values: list[float | None], period: int) -> list[float | None]:
    result: list[float | None] = []
    previous = 0.0
    for index, value in enumerate(values):
        if value is None or index + 1 < period:
            result.append(None)
            continue
        sample = values[index + 1 - period : index + 1]
        if any(item is None for item in sample):
            result.append(None)
            continue
        lowest = min(item for item in sample if item is not None)
        highest = max(item for item in sample if item is not None)
        current = previous if highest == lowest else 100 * (value - lowest) / (highest - lowest)
        result.append(current)
        previous = current
    return result


def _recursive_smooth(values: list[float | None], factor: float) -> list[float | None]:
    result: list[float | None] = []
    current: float | None = None
    for value in values:
        if value is None:
            result.append(None)
            continue
        current = value if current is None else current + factor * (value - current)
        result.append(current)
    return result


def _true_ranges(candles: list[Candle]) -> list[float]:
    if not candles:
        return []
    result = [candles[0].high - candles[0].low]
    for index in range(1, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        result.append(max(candle.high - candle.low, abs(candle.high - previous_close), abs(candle.low - previous_close)))
    return result


def _atr(candles: list[Candle], period: int) -> list[float | None]:
    ranges = _true_ranges(candles)
    result: list[float | None] = [None for _ in ranges]
    if len(ranges) < period:
        return result
    current = sum(ranges[:period]) / period
    result[period - 1] = current
    for index in range(period, len(ranges)):
        current = (current * (period - 1) + ranges[index]) / period
        result[index] = current
    return result


def _rolling_sum_optional(values: list[float | None], period: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < period:
            result.append(None)
            continue
        sample = values[index + 1 - period : index + 1]
        result.append(None if any(value is None for value in sample) else sum(value for value in sample if value is not None))
    return result
