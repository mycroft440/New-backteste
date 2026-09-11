from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from .candles import Candle


SignalFunction = Callable[..., list[int]]


@dataclass(frozen=True)
class ResearchedStrategy:
    name: str
    family: str
    description: str
    function: SignalFunction


def precision_trend_ehlers(
    candles: list[Candle],
    *,
    long_period: int = 250,
    short_period: int = 40,
) -> list[int]:
    """John Ehlers' Precision Trend Analysis, traded by ROC zero crosses."""
    if short_period <= 0 or long_period <= 0:
        raise ValueError("long_period e short_period precisam ser maiores que zero.")
    if short_period >= long_period:
        raise ValueError("short_period precisa ser menor que long_period.")

    closes = [candle.close for candle in candles]
    slow_high_pass = _high_pass(closes, float(long_period))
    fast_high_pass = _high_pass(closes, float(short_period))
    trend = [slow - fast for slow, fast in zip(slow_high_pass, fast_high_pass)]
    position = 0
    signals: list[int] = []

    for index in range(len(candles)):
        if index < 3:
            signals.append(position)
            continue
        roc = (short_period / (2 * math.pi)) * (trend[index] - trend[index - 1])
        if roc > 0:
            position = 1
        elif roc < 0:
            position = 0
        signals.append(position)
    return signals


def ultimate_oscillator_ehlers(
    candles: list[Candle],
    *,
    band_edge: int = 20,
    bandwidth: float = 2.0,
    rms_period: int = 100,
) -> list[int]:
    """Ehlers' 2025 oscillator, normalized by RMS and traded around zero."""
    if band_edge <= 0 or rms_period <= 0:
        raise ValueError("band_edge e rms_period precisam ser maiores que zero.")
    if bandwidth <= 1:
        raise ValueError("bandwidth precisa ser maior que 1.")

    closes = [candle.close for candle in candles]
    wide = _high_pass(closes, band_edge * bandwidth)
    narrow = _high_pass(closes, float(band_edge))
    raw = [wide_value - narrow_value for wide_value, narrow_value in zip(wide, narrow)]
    rms = _rolling_rms(raw, rms_period)
    position = 0
    signals: list[int] = []

    for value, scale in zip(raw, rms):
        if scale is None or scale == 0:
            signals.append(position)
            continue
        oscillator = value / scale
        if oscillator > 0:
            position = 1
        elif oscillator < 0:
            position = 0
        signals.append(position)
    return signals


def gap_momentum(
    candles: list[Candle],
    *,
    period: int = 40,
    signal_period: int = 20,
) -> list[int]:
    """Perry Kaufman's gap-ratio system, long while its signal line rises."""
    if period <= 1 or signal_period <= 1:
        raise ValueError("period e signal_period precisam ser maiores que 1.")

    gaps = [0.0]
    gaps.extend(candles[index].open - candles[index - 1].close for index in range(1, len(candles)))
    up_gaps = [max(gap, 0.0) for gap in gaps]
    down_gaps = [max(-gap, 0.0) for gap in gaps]
    up_sums = _rolling_sum(up_gaps, period)
    down_sums = _rolling_sum(down_gaps, period)
    ratios: list[float | None] = []

    for up_sum, down_sum in zip(up_sums, down_sums):
        if up_sum is None or down_sum is None:
            ratios.append(None)
        elif down_sum == 0:
            ratios.append(1.0)
        else:
            ratios.append(100 * up_sum / down_sum)

    signal_line = _sma_optional(ratios, signal_period)
    position = 0
    signals: list[int] = []
    previous: float | None = None
    for value in signal_line:
        if value is None:
            signals.append(position)
            continue
        if previous is not None:
            if value > previous:
                position = 1
            elif value < previous:
                position = 0
        signals.append(position)
        previous = value
    return signals


def heikin_ashi_stochastic(
    candles: list[Candle],
    *,
    k_period: int = 14,
    slowing: int = 3,
    d_period: int = 3,
    lower: float = 20.0,
    upper: float = 80.0,
) -> list[int]:
    """Heikin-Ashi direction confirmed by a slow stochastic crossover."""
    if k_period <= 1 or slowing <= 0 or d_period <= 0:
        raise ValueError("k_period precisa ser maior que 1; slowing e d_period precisam ser positivos.")
    if not 0 <= lower < upper <= 100:
        raise ValueError("lower e upper precisam estar entre 0 e 100, com lower < upper.")

    ha_open, ha_close = _heikin_ashi_open_close(candles)
    raw_k: list[float | None] = []
    for index, candle in enumerate(candles):
        if index + 1 < k_period:
            raw_k.append(None)
            continue
        sample = candles[index + 1 - k_period : index + 1]
        highest = max(item.high for item in sample)
        lowest = min(item.low for item in sample)
        raw_k.append(50.0 if highest == lowest else 100 * (candle.close - lowest) / (highest - lowest))

    slow_k = _sma_optional(raw_k, slowing)
    slow_d = _sma_optional(slow_k, d_period)
    position = 0
    signals: list[int] = []

    for index in range(len(candles)):
        if index == 0 or slow_k[index] is None or slow_d[index] is None:
            signals.append(position)
            continue
        previous_k = slow_k[index - 1]
        previous_d = slow_d[index - 1]
        if previous_k is None or previous_d is None:
            signals.append(position)
            continue

        bullish_flip = ha_close[index] > ha_open[index] and ha_close[index - 1] <= ha_open[index - 1]
        bearish_flip = ha_close[index] < ha_open[index] and ha_close[index - 1] >= ha_open[index - 1]
        bullish_cross = previous_k <= previous_d and slow_k[index] > slow_d[index]
        bearish_cross = previous_k >= previous_d and slow_k[index] < slow_d[index]
        oversold = max(slow_k[index], slow_d[index]) <= lower
        overbought = min(slow_k[index], slow_d[index]) >= upper

        if position == 0 and bullish_flip and bullish_cross and oversold:
            position = 1
        elif position == 1 and (bearish_flip or (bearish_cross and overbought)):
            position = 0
        signals.append(position)
    return signals


def vortex_trend(candles: list[Candle], *, period: int = 14) -> list[int]:
    """Botes-Siepman Vortex Indicator: long while VI+ is above VI-."""
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")
    if not candles:
        return []

    true_range = [candles[0].high - candles[0].low]
    positive_movement = [0.0]
    negative_movement = [0.0]
    for index in range(1, len(candles)):
        candle = candles[index]
        previous = candles[index - 1]
        true_range.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous.close),
                abs(candle.low - previous.close),
            )
        )
        positive_movement.append(abs(candle.high - previous.low))
        negative_movement.append(abs(candle.low - previous.high))

    position = 0
    signals: list[int] = []
    for index in range(len(candles)):
        if index < period:
            signals.append(position)
            continue
        start = index + 1 - period
        tr_sum = sum(true_range[start : index + 1])
        if tr_sum == 0:
            signals.append(position)
            continue
        vi_plus = sum(positive_movement[start : index + 1]) / tr_sum
        vi_minus = sum(negative_movement[start : index + 1]) / tr_sum
        if vi_plus > vi_minus:
            position = 1
        elif vi_plus < vi_minus:
            position = 0
        signals.append(position)
    return signals


def kama_trend(
    candles: list[Candle],
    *,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> list[int]:
    """Kaufman's adaptive moving average with price and slope confirmation."""
    if er_period <= 0 or fast_period <= 0 or slow_period <= 0:
        raise ValueError("er_period, fast_period e slow_period precisam ser maiores que zero.")
    if fast_period >= slow_period:
        raise ValueError("fast_period precisa ser menor que slow_period.")

    closes = [candle.close for candle in candles]
    kama: list[float | None] = [None for _ in candles]
    fast = 2 / (fast_period + 1)
    slow = 2 / (slow_period + 1)

    for index in range(er_period, len(closes)):
        change = abs(closes[index] - closes[index - er_period])
        volatility = sum(abs(closes[item] - closes[item - 1]) for item in range(index + 1 - er_period, index + 1))
        efficiency = 0.0 if volatility == 0 else change / volatility
        smoothing = (efficiency * (fast - slow) + slow) ** 2
        previous = closes[index - 1] if kama[index - 1] is None else kama[index - 1]
        kama[index] = previous + smoothing * (closes[index] - previous)

    position = 0
    signals: list[int] = []
    for index, close in enumerate(closes):
        value = kama[index]
        previous = kama[index - 1] if index else None
        if value is None or previous is None:
            signals.append(position)
            continue
        if position == 0 and close > value and value > previous:
            position = 1
        elif position == 1 and (close < value or value < previous):
            position = 0
        signals.append(position)
    return signals


def frama_trend(candles: list[Candle], *, window: int = 16) -> list[int]:
    """Ehlers' fractal adaptive moving average with price/slope trend rules."""
    if window < 4 or window % 2:
        raise ValueError("window precisa ser par e maior ou igual a 4.")

    prices = [(candle.high + candle.low) / 2 for candle in candles]
    frama: list[float | None] = [None for _ in candles]
    half = window // 2

    for index in range(window - 1, len(candles)):
        sample = candles[index + 1 - window : index + 1]
        first = sample[:half]
        second = sample[half:]
        n1 = (max(item.high for item in first) - min(item.low for item in first)) / half
        n2 = (max(item.high for item in second) - min(item.low for item in second)) / half
        n3 = (max(item.high for item in sample) - min(item.low for item in sample)) / window
        if n1 + n2 > 0 and n3 > 0:
            dimension = (math.log(n1 + n2) - math.log(n3)) / math.log(2)
            alpha = math.exp(-4.6 * (dimension - 1))
        else:
            alpha = 1.0
        alpha = min(1.0, max(0.01, alpha))
        previous = prices[index - 1] if frama[index - 1] is None else frama[index - 1]
        frama[index] = alpha * prices[index] + (1 - alpha) * previous

    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        value = frama[index]
        previous = frama[index - 1] if index else None
        if value is None or previous is None:
            signals.append(position)
            continue
        if position == 0 and candle.close > value and value > previous:
            position = 1
        elif position == 1 and (candle.close < value or value < previous):
            position = 0
        signals.append(position)
    return signals


def rvi_reversal(
    candles: list[Candle],
    *,
    period: int = 10,
    entry_level: float = -0.4,
    exit_level: float = 0.0,
) -> list[int]:
    """Relative Vigor Index reversal from weakness, exited on a bearish cross."""
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")
    if entry_level >= exit_level:
        raise ValueError("entry_level precisa ser menor que exit_level.")

    close_open = [candle.close - candle.open for candle in candles]
    high_low = [candle.high - candle.low for candle in candles]
    numerator = _weighted_four(close_open)
    denominator = _weighted_four(high_low)
    rvi: list[float | None] = [None for _ in candles]

    for index in range(len(candles)):
        if index < period + 2:
            continue
        start = index + 1 - period
        num_sample = numerator[start : index + 1]
        den_sample = denominator[start : index + 1]
        if any(value is None for value in num_sample) or any(value is None for value in den_sample):
            continue
        num_sum = sum(value for value in num_sample if value is not None)
        den_sum = sum(value for value in den_sample if value is not None)
        rvi[index] = 0.0 if den_sum == 0 else num_sum / den_sum

    rvi_signal = _weighted_four_optional(rvi)
    position = 0
    signals: list[int] = []
    for index in range(len(candles)):
        if index == 0 or rvi[index] is None or rvi_signal[index] is None:
            signals.append(position)
            continue
        previous_rvi = rvi[index - 1]
        previous_signal = rvi_signal[index - 1]
        if previous_rvi is None or previous_signal is None:
            signals.append(position)
            continue
        bullish_cross = previous_rvi <= previous_signal and rvi[index] > rvi_signal[index]
        bearish_cross = previous_rvi >= previous_signal and rvi[index] < rvi_signal[index]
        if position == 0 and previous_rvi < entry_level and bullish_cross:
            position = 1
        elif position == 1 and previous_rvi > exit_level and bearish_cross:
            position = 0
        signals.append(position)
    return signals


def chaikin_money_flow(
    candles: list[Candle],
    *,
    period: int = 21,
    trend_window: int = 100,
) -> list[int]:
    """Long on a positive CMF zero-cross, optionally above a trend average."""
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")
    if trend_window < 0:
        raise ValueError("trend_window nao pode ser negativo.")

    closes = [candle.close for candle in candles]
    trend = _sma(closes, trend_window) if trend_window else [None for _ in candles]
    flow_volume: list[float] = []
    volume = [float(candle.volume) for candle in candles]
    for candle in candles:
        size = candle.high - candle.low
        multiplier = 0.0 if size == 0 else ((candle.close - candle.low) - (candle.high - candle.close)) / size
        flow_volume.append(multiplier * candle.volume)
    flow_sums = _rolling_sum(flow_volume, period)
    volume_sums = _rolling_sum(volume, period)
    cmf = [
        None if flow is None or total is None else (0.0 if total == 0 else flow / total)
        for flow, total in zip(flow_sums, volume_sums)
    ]

    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        value = cmf[index]
        previous = cmf[index - 1] if index else None
        trend_ok = trend_window == 0 or (trend[index] is not None and candle.close > trend[index])
        if value is None:
            signals.append(position)
            continue
        if position == 0 and previous is not None and previous <= 0 < value and trend_ok:
            position = 1
        elif position == 1 and (value < 0 or not trend_ok):
            position = 0
        signals.append(position)
    return signals


def squeeze_breakout(
    candles: list[Candle],
    *,
    window: int = 20,
    num_std: float = 2.0,
    atr_period: int = 20,
    keltner_mult: float = 1.5,
    squeeze_bars: int = 3,
    atr_mult: float = 3.0,
) -> list[int]:
    """BB/Keltner squeeze release with a close-confirmed upside breakout."""
    if window <= 1 or atr_period <= 0 or squeeze_bars <= 0:
        raise ValueError("window precisa ser maior que 1; atr_period e squeeze_bars precisam ser positivos.")
    if num_std <= 0 or keltner_mult <= 0 or atr_mult <= 0:
        raise ValueError("num_std, keltner_mult e atr_mult precisam ser maiores que zero.")

    closes = [candle.close for candle in candles]
    middle = _sma(closes, window)
    deviation = _rolling_std(closes, window)
    keltner_middle = _ema(closes, window)
    atr_values = _atr(candles, atr_period)
    position = 0
    squeeze_run = 0
    armed = False
    peak_close = 0.0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if (
            middle[index] is None
            or deviation[index] is None
            or keltner_middle[index] is None
            or atr_values[index] is None
        ):
            signals.append(position)
            continue

        upper_bollinger = middle[index] + num_std * deviation[index]
        lower_bollinger = middle[index] - num_std * deviation[index]
        upper_keltner = keltner_middle[index] + keltner_mult * atr_values[index]
        lower_keltner = keltner_middle[index] - keltner_mult * atr_values[index]
        in_squeeze = upper_bollinger < upper_keltner and lower_bollinger > lower_keltner

        if position == 1:
            peak_close = max(peak_close, candle.close)
            trailing_stop = peak_close - atr_mult * atr_values[index]
            if candle.close < middle[index] or candle.close < trailing_stop:
                position = 0
                peak_close = 0.0
        elif in_squeeze:
            squeeze_run += 1
            armed = armed or squeeze_run >= squeeze_bars
        else:
            if armed and candle.close > upper_bollinger:
                position = 1
                peak_close = candle.close
            squeeze_run = 0
            armed = False
        signals.append(position)
    return signals


def turtle_soup(
    candles: list[Candle],
    *,
    lookback: int = 20,
    sma_window: int = 5,
    atr_period: int = 14,
    stop_atr: float = 0.5,
    hold_limit: int = 5,
) -> list[int]:
    """Close-confirmed long-only Turtle Soup false-breakout reversal."""
    if lookback <= 1 or sma_window <= 0 or atr_period <= 0 or hold_limit <= 0:
        raise ValueError("lookback precisa ser maior que 1 e os demais periodos precisam ser positivos.")
    if stop_atr < 0:
        raise ValueError("stop_atr nao pode ser negativo.")

    closes = [candle.close for candle in candles]
    exit_average = _sma(closes, sma_window)
    atr_values = _atr(candles, atr_period)
    position = 0
    held = 0
    setup_low = 0.0
    signals: list[int] = []

    for index, candle in enumerate(candles):
        if index < lookback or exit_average[index] is None or atr_values[index] is None:
            signals.append(position)
            continue
        if position == 0:
            prior_low = min(item.low for item in candles[index - lookback : index])
            if candle.low < prior_low and candle.close > prior_low:
                position = 1
                held = 0
                setup_low = candle.low
        else:
            held += 1
            stop = setup_low - stop_atr * atr_values[index]
            if candle.close >= exit_average[index] or candle.close < stop or held >= hold_limit:
                position = 0
                held = 0
                setup_low = 0.0
        signals.append(position)
    return signals


def turn_of_month(
    candles: list[Candle],
    *,
    sessions_before: int = 1,
    sessions_after: int = 3,
    session_calendar: list[str] | None = None,
) -> list[int]:
    """Hold around month-end using a calendar independent of future prices."""
    if sessions_before <= 0 or sessions_after <= 0:
        raise ValueError("sessions_before e sessions_after precisam ser maiores que zero.")
    if not candles:
        return []

    candle_dates = [
        date.fromisoformat(candle.date.split(" ", 1)[0]) for candle in candles
    ]
    if candle_dates != sorted(set(candle_dates)):
        raise ValueError("candles precisam ter datas unicas em ordem crescente.")

    if session_calendar is None:
        # The public matrix always supplies the verified B3 calendar.  This
        # deterministic fallback keeps standalone calls prefix-causal without
        # pretending that a future price observation defines a month boundary.
        calendar_dates = set(candle_dates)
        current = candle_dates[0]
        horizon = candle_dates[-1] + timedelta(days=45)
        while current <= horizon:
            if current.weekday() < 5:
                calendar_dates.add(current)
            current += timedelta(days=1)
        ordered_calendar = sorted(calendar_dates)
    else:
        ordered_calendar = [
            date.fromisoformat(str(value).split(" ", 1)[0])
            for value in session_calendar
        ]
        if ordered_calendar != sorted(set(ordered_calendar)):
            raise ValueError(
                "session_calendar precisa ter datas unicas em ordem crescente."
            )
        missing = sorted(set(candle_dates) - set(ordered_calendar))
        if missing:
            raise ValueError(
                "session_calendar nao cobre todos os candles: "
                f"{[value.isoformat() for value in missing[:5]]}."
            )

    grouped: dict[tuple[int, int], list[int]] = {}
    for index, session in enumerate(ordered_calendar):
        grouped.setdefault((session.year, session.month), []).append(index)

    in_window = [False for _ in ordered_calendar]
    month_groups = list(grouped.values())
    for group_index, indices in enumerate(month_groups):
        if group_index > 0:
            for index in indices[:sessions_after]:
                in_window[index] = True
        if group_index + 1 < len(month_groups):
            for index in indices[-sessions_before:]:
                in_window[index] = True

    # A close signal is the desired position at the next session's open.
    calendar_index = {session: index for index, session in enumerate(ordered_calendar)}
    return [
        int(in_window[calendar_index[session] + 1])
        if calendar_index[session] + 1 < len(ordered_calendar)
        else 0
        for session in candle_dates
    ]


RESEARCHED_STRATEGIES = (
    ResearchedStrategy(
        "precision_trend_ehlers",
        "tendencia",
        "Precision Trend de Ehlers: compra no ROC positivo do filtro e sai no ROC negativo.",
        precision_trend_ehlers,
    ),
    ResearchedStrategy(
        "ultimate_oscillator_ehlers",
        "tendencia",
        "Ultimate Oscillator de Ehlers: comprado acima de zero e em caixa abaixo de zero.",
        ultimate_oscillator_ehlers,
    ),
    ResearchedStrategy(
        "gap_momentum",
        "momentum",
        "Gap Momentum de Kaufman: compra quando a linha-sinal sobe e sai quando ela cai.",
        gap_momentum,
    ),
    ResearchedStrategy(
        "heikin_ashi_stochastic",
        "combinada",
        "Reversao Heikin-Ashi confirmada por cruzamento estocastico em zona extrema.",
        heikin_ashi_stochastic,
    ),
    ResearchedStrategy(
        "vortex_trend",
        "tendencia",
        "Vortex de Botes-Siepman: comprado enquanto VI+ permanece acima de VI-.",
        vortex_trend,
    ),
    ResearchedStrategy(
        "kama_trend",
        "tendencia",
        "Tendencia por KAMA de Kaufman com confirmacao simultanea de preco e inclinacao.",
        kama_trend,
    ),
    ResearchedStrategy(
        "frama_trend",
        "tendencia",
        "Tendencia pela media fractal adaptativa FRAMA de Ehlers.",
        frama_trend,
    ),
    ResearchedStrategy(
        "rvi_reversal",
        "reversao",
        "Reversao por cruzamento do Relative Vigor Index apos fraqueza extrema.",
        rvi_reversal,
    ),
    ResearchedStrategy(
        "chaikin_money_flow",
        "volume",
        "Cruzamento positivo do Chaikin Money Flow com filtro opcional de tendencia.",
        chaikin_money_flow,
    ),
    ResearchedStrategy(
        "squeeze_breakout",
        "rompimento",
        "Rompimento altista apos Bollinger comprimir dentro do canal de Keltner.",
        squeeze_breakout,
    ),
    ResearchedStrategy(
        "turtle_soup",
        "reversao",
        "Turtle Soup long-only: falso rompimento da minima com saida por media, ATR ou tempo.",
        turtle_soup,
    ),
    ResearchedStrategy(
        "turn_of_month",
        "sazonalidade",
        "Janela sazonal do ultimo pregao ate o terceiro pregao do mes seguinte.",
        turn_of_month,
    ),
)


def _high_pass(values: list[float], period: float) -> list[float]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -(a1**2)
    c1 = (1 + c2 - c3) / 4
    result = [0.0 for _ in values]
    for index in range(3, len(values)):
        result[index] = (
            c1 * (values[index] - 2 * values[index - 1] + values[index - 2])
            + c2 * result[index - 1]
            + c3 * result[index - 2]
        )
    return result


def _rolling_sum(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(None if index + 1 < window else running)
    return result


def _sma(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window precisa ser maior que zero.")
    sums = _rolling_sum(values, window)
    return [None if value is None else value / window for value in sums]


def _sma_optional(values: list[float | None], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window precisa ser maior que zero.")
    result: list[float | None] = []
    sample: list[float] = []
    for value in values:
        if value is None:
            sample = []
            result.append(None)
            continue
        sample.append(value)
        if len(sample) > window:
            sample.pop(0)
        result.append(sum(sample) / window if len(sample) == window else None)
    return result


def _ema(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window precisa ser maior que zero.")
    result: list[float | None] = [None for _ in values]
    if len(values) < window:
        return result
    current = sum(values[:window]) / window
    result[window - 1] = current
    alpha = 2 / (window + 1)
    for index in range(window, len(values)):
        current = alpha * values[index] + (1 - alpha) * current
        result[index] = current
    return result


def _rolling_std(values: list[float], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
        else:
            result.append(statistics.pstdev(values[index + 1 - window : index + 1]))
    return result


def _rolling_rms(values: list[float], window: int) -> list[float | None]:
    squared = _rolling_sum([value * value for value in values], window)
    return [None if value is None else math.sqrt(max(value, 0.0) / window) for value in squared]


def _atr(candles: list[Candle], period: int) -> list[float | None]:
    if not candles:
        return []
    ranges = [candles[0].high - candles[0].low]
    for index in range(1, len(candles)):
        candle = candles[index]
        previous_close = candles[index - 1].close
        ranges.append(
            max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        )
    result: list[float | None] = [None for _ in candles]
    if len(ranges) < period:
        return result
    current = sum(ranges[:period]) / period
    result[period - 1] = current
    for index in range(period, len(ranges)):
        current = (current * (period - 1) + ranges[index]) / period
        result[index] = current
    return result


def _heikin_ashi_open_close(candles: list[Candle]) -> tuple[list[float], list[float]]:
    ha_open: list[float] = []
    ha_close: list[float] = []
    for index, candle in enumerate(candles):
        close = (candle.open + candle.high + candle.low + candle.close) / 4
        open_ = (candle.open + candle.close) / 2 if index == 0 else (ha_open[-1] + ha_close[-1]) / 2
        ha_open.append(open_)
        ha_close.append(close)
    return ha_open, ha_close


def _weighted_four(values: list[float]) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index < 3:
            result.append(None)
        else:
            result.append((values[index] + 2 * values[index - 1] + 2 * values[index - 2] + values[index - 3]) / 6)
    return result


def _weighted_four_optional(values: list[float | None]) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        if index < 3 or any(values[item] is None for item in range(index - 3, index + 1)):
            result.append(None)
            continue
        current = values[index]
        one = values[index - 1]
        two = values[index - 2]
        three = values[index - 3]
        assert current is not None and one is not None and two is not None and three is not None
        result.append((current + 2 * one + 2 * two + three) / 6)
    return result
