from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .candles import Candle


SignalFunction = Callable[..., list[int]]


@dataclass(frozen=True)
class SMIErgodicStrategy:
    name: str
    family: str
    description: str
    function: SignalFunction


def _validate_periods(long_period: int, short_period: int, signal_period: int) -> None:
    if long_period <= 0 or short_period <= 0 or signal_period <= 0:
        raise ValueError("long_period, short_period e signal_period precisam ser maiores que zero.")
    if short_period >= long_period:
        raise ValueError("short_period precisa ser menor que long_period.")


def _ema_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    alpha = 2.0 / (period + 1.0)
    result: list[float | None] = []
    seed: list[float] = []
    current: float | None = None
    for value in values:
        if value is None:
            result.append(None)
            continue
        if current is None:
            seed.append(float(value))
            if len(seed) < period:
                result.append(None)
                continue
            current = sum(seed) / period
        else:
            current = alpha * float(value) + (1.0 - alpha) * current
        result.append(current)
    return result


def _sma_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    window: list[float] = []
    for value in values:
        if value is None:
            window = []
            result.append(None)
            continue
        window.append(float(value))
        if len(window) > period:
            window.pop(0)
        result.append(sum(window) / period if len(window) == period else None)
    return result


def _wilder_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    seed: list[float] = []
    current: float | None = None
    for value in values:
        if value is None:
            result.append(None)
            continue
        if current is None:
            seed.append(float(value))
            if len(seed) < period:
                result.append(None)
                continue
            current = sum(seed) / period
        else:
            current = ((period - 1.0) * current + float(value)) / period
        result.append(current)
    return result


def smi_ergodic_components(
    candles: list[Candle],
    *,
    long_period: int = 20,
    short_period: int = 5,
    signal_period: int = 5,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return SMI Ergodic/TSI line, signal EMA and histogram without lookahead."""
    _validate_periods(long_period, short_period, signal_period)
    if not candles:
        return [], [], []
    closes = [float(candle.close) for candle in candles]
    momentum: list[float | None] = [None]
    absolute: list[float | None] = [None]
    for index in range(1, len(closes)):
        change = closes[index] - closes[index - 1]
        momentum.append(change)
        absolute.append(abs(change))
    momentum_double = _ema_optional(_ema_optional(momentum, long_period), short_period)
    absolute_double = _ema_optional(_ema_optional(absolute, long_period), short_period)
    smi: list[float | None] = []
    for numerator, denominator in zip(momentum_double, absolute_double):
        if numerator is None or denominator is None or denominator == 0:
            smi.append(None)
        else:
            smi.append(100.0 * numerator / denominator)
    signal = _ema_optional(smi, signal_period)
    histogram = [
        None if line is None or signal_value is None else line - signal_value
        for line, signal_value in zip(smi, signal)
    ]
    return smi, signal, histogram


def _crossed_above(left: list[float | None], right: list[float | None], index: int) -> bool:
    if index <= 0:
        return False
    a0, a1, b0, b1 = left[index - 1], left[index], right[index - 1], right[index]
    return a0 is not None and a1 is not None and b0 is not None and b1 is not None and a0 <= b0 and a1 > b1


def _crossed_below(left: list[float | None], right: list[float | None], index: int) -> bool:
    if index <= 0:
        return False
    a0, a1, b0, b1 = left[index - 1], left[index], right[index - 1], right[index]
    return a0 is not None and a1 is not None and b0 is not None and b1 is not None and a0 >= b0 and a1 < b1


def _ema_closes(candles: list[Candle], period: int) -> list[float | None]:
    return _ema_optional([float(candle.close) for candle in candles], period)


def _atr(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("atr_period precisa ser maior que zero.")
    if not candles:
        return []
    ranges: list[float | None] = [float(candles[0].high - candles[0].low)]
    for index in range(1, len(candles)):
        candle = candles[index]
        previous_close = float(candles[index - 1].close)
        ranges.append(max(float(candle.high - candle.low), abs(float(candle.high) - previous_close), abs(float(candle.low) - previous_close)))
    return _wilder_optional(ranges, period)


def _adx(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("adx_period precisa ser maior que zero.")
    if not candles:
        return []
    true_ranges: list[float | None] = [float(candles[0].high - candles[0].low)]
    plus_dm: list[float | None] = [0.0]
    minus_dm: list[float | None] = [0.0]
    for index in range(1, len(candles)):
        candle, previous = candles[index], candles[index - 1]
        up_move = float(candle.high - previous.high)
        down_move = float(previous.low - candle.low)
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        true_ranges.append(max(float(candle.high - candle.low), abs(float(candle.high - previous.close)), abs(float(candle.low - previous.close))))
    atr_values = _wilder_optional(true_ranges, period)
    plus_values = _wilder_optional(plus_dm, period)
    minus_values = _wilder_optional(minus_dm, period)
    dx: list[float | None] = []
    for atr_value, plus_value, minus_value in zip(atr_values, plus_values, minus_values):
        if atr_value is None or plus_value is None or minus_value is None or atr_value == 0:
            dx.append(None)
            continue
        plus_di = 100.0 * plus_value / atr_value
        minus_di = 100.0 * minus_value / atr_value
        total = plus_di + minus_di
        dx.append(0.0 if total == 0 else 100.0 * abs(plus_di - minus_di) / total)
    return _wilder_optional(dx, period)


def _macd_components(candles: list[Candle], fast: int, slow: int, signal_period: int) -> tuple[list[float | None], list[float | None]]:
    if fast <= 0 or slow <= 0 or signal_period <= 0 or fast >= slow:
        raise ValueError("MACD exige 0 < fast < slow e signal_period > 0.")
    closes = [float(candle.close) for candle in candles]
    fast_ema, slow_ema = _ema_optional(closes, fast), _ema_optional(closes, slow)
    line = [None if fast_value is None or slow_value is None else fast_value - slow_value for fast_value, slow_value in zip(fast_ema, slow_ema)]
    return line, _ema_optional(line, signal_period)


def _rolling_percentile(values: list[float | None], *, window: int, percentile: float) -> list[float | None]:
    if window <= 1:
        raise ValueError("zone_window precisa ser maior que 1.")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile precisa estar entre 0 e 1.")
    result: list[float | None] = []
    for index in range(len(values)):
        sample = [float(value) for value in values[max(0, index - window):index] if value is not None]
        if len(sample) < window:
            result.append(None)
            continue
        sample.sort()
        position = percentile * (len(sample) - 1)
        lower, upper = int(math.floor(position)), int(math.ceil(position))
        if lower == upper:
            result.append(sample[lower])
        else:
            weight = position - lower
            result.append(sample[lower] * (1.0 - weight) + sample[upper] * weight)
    return result


def smi_ergodic_crossover(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5) -> list[int]:
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    return [int(line is not None and signal_value is not None and line > signal_value) for line, signal_value in zip(smi, signal)]


def smi_ergodic_oversold_cross(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, lower: float = -20.0, upper: float = 20.0) -> list[int]:
    if lower >= upper:
        raise ValueError("lower precisa ser menor que upper.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    position = 0
    signals: list[int] = []
    for index, value in enumerate(smi):
        if value is not None and signal[index] is not None:
            previous = smi[index - 1] if index > 0 else None
            touched_lower = value <= lower or (previous is not None and previous <= lower)
            if position == 0 and touched_lower and _crossed_above(smi, signal, index):
                position = 1
            elif position == 1 and (_crossed_below(smi, signal, index) or value >= upper):
                position = 0
        signals.append(position)
    return signals


def smi_ergodic_ema_trend(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, trend_period: int = 200) -> list[int]:
    if trend_period <= 0:
        raise ValueError("trend_period precisa ser maior que zero.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    trend = _ema_closes(candles, trend_period)
    return [int(line is not None and signal_value is not None and trend_value is not None and candle.close > trend_value and line > signal_value) for candle, line, signal_value, trend_value in zip(candles, smi, signal, trend)]


def smi_ergodic_ema_zero(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, trend_period: int = 200) -> list[int]:
    if trend_period <= 0:
        raise ValueError("trend_period precisa ser maior que zero.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    trend = _ema_closes(candles, trend_period)
    return [int(line is not None and signal_value is not None and trend_value is not None and candle.close > trend_value and line > 0.0 and line > signal_value) for candle, line, signal_value, trend_value in zip(candles, smi, signal, trend)]


def smi_ergodic_zero_regime(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5) -> list[int]:
    smi, _, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    return [int(value is not None and value > 0.0) for value in smi]


def smi_ergodic_pullback(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, trend_period: int = 200, pullback_period: int = 20) -> list[int]:
    if trend_period <= 0 or pullback_period <= 0:
        raise ValueError("trend_period e pullback_period precisam ser maiores que zero.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    trend, pullback = _ema_closes(candles, trend_period), _ema_closes(candles, pullback_period)
    position, armed = 0, False
    signals: list[int] = []
    for index, candle in enumerate(candles):
        trend_value, pullback_value, line, signal_value = trend[index], pullback[index], smi[index], signal[index]
        if trend_value is not None and pullback_value is not None and line is not None and signal_value is not None:
            in_trend = candle.close > trend_value
            if position == 0:
                if in_trend and candle.close <= pullback_value:
                    armed = True
                elif armed and in_trend and candle.close > pullback_value and line > 0.0 and line > signal_value:
                    position, armed = 1, False
                elif not in_trend:
                    armed = False
            elif candle.close < trend_value or line < signal_value:
                position = 0
        signals.append(position)
    return signals


def smi_ergodic_divergence(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, divergence_lookback: int = 20) -> list[int]:
    if divergence_lookback <= 1:
        raise ValueError("divergence_lookback precisa ser maior que 1.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    position, armed = 0, False
    signals: list[int] = []
    for index, candle in enumerate(candles):
        line = smi[index]
        if line is not None and index >= divergence_lookback:
            prior_prices = [float(item.close) for item in candles[index - divergence_lookback:index]]
            prior_smi = [float(value) for value in smi[index - divergence_lookback:index] if value is not None]
            if len(prior_smi) == divergence_lookback and candle.close < min(prior_prices) and line > min(prior_smi):
                armed = True
            if position == 0 and armed and _crossed_above(smi, signal, index):
                position, armed = 1, False
            elif position == 1 and _crossed_below(smi, signal, index):
                position = 0
        signals.append(position)
    return signals


def smi_ergodic_histogram_momentum(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5) -> list[int]:
    _, _, histogram = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    position = 0
    signals: list[int] = []
    for index, value in enumerate(histogram):
        previous = histogram[index - 1] if index > 0 else None
        if value is not None and previous is not None:
            if position == 0 and value > 0.0 and value > previous:
                position = 1
            elif position == 1 and (value <= 0.0 or value < previous):
                position = 0
        signals.append(position)
    return signals


def smi_ergodic_histogram_continuation(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, trend_period: int = 200) -> list[int]:
    if trend_period <= 0:
        raise ValueError("trend_period precisa ser maior que zero.")
    _, _, histogram = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    trend = _ema_closes(candles, trend_period)
    position, armed = 0, False
    signals: list[int] = []
    for index, candle in enumerate(candles):
        value, trend_value = histogram[index], trend[index]
        previous = histogram[index - 1] if index > 0 else None
        if value is not None and trend_value is not None:
            in_trend = candle.close > trend_value
            if position == 0:
                if in_trend and value < 0.0:
                    armed = True
                elif armed and in_trend and previous is not None and previous <= 0.0 < value:
                    position, armed = 1, False
                elif not in_trend:
                    armed = False
            elif not in_trend or value < 0.0:
                position = 0
        signals.append(position)
    return signals


def smi_ergodic_atr_filter(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, atr_period: int = 14, atr_mean_period: int = 50, atr_ratio: float = 1.0) -> list[int]:
    if atr_mean_period <= 0 or atr_ratio <= 0:
        raise ValueError("atr_mean_period e atr_ratio precisam ser maiores que zero.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    atr_values = _atr(candles, atr_period)
    atr_average = _sma_optional(atr_values, atr_mean_period)
    position = 0
    signals: list[int] = []
    for index in range(len(candles)):
        atr_value, average = atr_values[index], atr_average[index]
        if position == 0 and atr_value is not None and average is not None and atr_value >= average * atr_ratio and _crossed_above(smi, signal, index):
            position = 1
        elif position == 1 and _crossed_below(smi, signal, index):
            position = 0
        signals.append(position)
    return signals


def smi_ergodic_adx_filter(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, adx_period: int = 14, adx_threshold: float = 25.0) -> list[int]:
    if adx_threshold < 0:
        raise ValueError("adx_threshold nao pode ser negativo.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    adx_values = _adx(candles, adx_period)
    return [int(line is not None and signal_value is not None and adx_value is not None and adx_value >= adx_threshold and line > signal_value) for line, signal_value, adx_value in zip(smi, signal, adx_values)]


def smi_ergodic_volume_filter(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, volume_period: int = 20, volume_ratio: float = 1.0) -> list[int]:
    if volume_period <= 0 or volume_ratio <= 0:
        raise ValueError("volume_period e volume_ratio precisam ser maiores que zero.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    volume_average = _sma_optional([float(candle.volume) for candle in candles], volume_period)
    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        average = volume_average[index]
        if position == 0 and average is not None and candle.volume >= average * volume_ratio and _crossed_above(smi, signal, index):
            position = 1
        elif position == 1 and _crossed_below(smi, signal, index):
            position = 0
        signals.append(position)
    return signals


def smi_ergodic_macd_confirm(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> list[int]:
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    macd, macd_signal_line = _macd_components(candles, macd_fast, macd_slow, macd_signal)
    return [int(line is not None and signal_value is not None and macd_value is not None and macd_signal_value is not None and line > signal_value and macd_value > macd_signal_value) for line, signal_value, macd_value, macd_signal_value in zip(smi, signal, macd, macd_signal_line)]


def smi_ergodic_123_reversal(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5) -> list[int]:
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        if index >= 3 and smi[index] is not None and signal[index] is not None:
            point1, point2, point3 = candles[index - 3], candles[index - 2], candles[index - 1]
            bullish_123 = point1.low < point2.low and point3.low > point1.low and point2.high > point1.high and candle.close > point2.high
            if position == 0 and bullish_123 and smi[index] > signal[index]:
                position = 1
            elif position == 1 and _crossed_below(smi, signal, index):
                position = 0
        signals.append(position)
    return signals


def smi_ergodic_dynamic_zones(candles: list[Candle], *, long_period: int = 20, short_period: int = 5, signal_period: int = 5, zone_window: int = 100, lower_quantile: float = 0.2, upper_quantile: float = 0.8) -> list[int]:
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("use 0 < lower_quantile < upper_quantile < 1.")
    smi, signal, _ = smi_ergodic_components(candles, long_period=long_period, short_period=short_period, signal_period=signal_period)
    lower_zone = _rolling_percentile(smi, window=zone_window, percentile=lower_quantile)
    upper_zone = _rolling_percentile(smi, window=zone_window, percentile=upper_quantile)
    position = 0
    signals: list[int] = []
    for index, value in enumerate(smi):
        lower, upper = lower_zone[index], upper_zone[index]
        previous = smi[index - 1] if index > 0 else None
        if value is not None and lower is not None and upper is not None:
            touched_lower = value <= lower or (previous is not None and previous <= lower)
            touched_upper = value >= upper or (previous is not None and previous >= upper)
            if position == 0 and touched_lower and _crossed_above(smi, signal, index):
                position = 1
            elif position == 1 and touched_upper and _crossed_below(smi, signal, index):
                position = 0
        signals.append(position)
    return signals


SMI_ERGODIC_STRATEGIES = (
    SMIErgodicStrategy("smi_ergodic_crossover", "momentum", "SMI Ergodic classico: comprado enquanto a linha SMI permanece acima da linha de sinal.", smi_ergodic_crossover),
    SMIErgodicStrategy("smi_ergodic_oversold_cross", "reversao", "Cruzamento altista do SMI Ergodic apos tocar sobrevenda, com saida por cruzamento ou sobrecompra.", smi_ergodic_oversold_cross),
    SMIErgodicStrategy("smi_ergodic_ema_trend", "tendencia", "SMI Ergodic confirmado por fechamento acima da EMA de tendencia.", smi_ergodic_ema_trend),
    SMIErgodicStrategy("smi_ergodic_ema_zero", "tendencia", "SMI Ergodic acima do sinal e de zero, com preco acima da EMA de tendencia.", smi_ergodic_ema_zero),
    SMIErgodicStrategy("smi_ergodic_zero_regime", "momentum", "Regime long-only enquanto a linha SMI Ergodic permanece acima de zero.", smi_ergodic_zero_regime),
    SMIErgodicStrategy("smi_ergodic_pullback", "reversao", "Pullback na EMA curta dentro de tendencia positiva, confirmado pelo SMI Ergodic.", smi_ergodic_pullback),
    SMIErgodicStrategy("smi_ergodic_divergence", "reversao", "Divergencia altista causal: nova minima de preco sem nova minima do SMI, confirmada por cruzamento.", smi_ergodic_divergence),
    SMIErgodicStrategy("smi_ergodic_histogram_momentum", "momentum", "Compra aceleracao positiva do histograma SMI Ergodic e sai quando o impulso perde forca.", smi_ergodic_histogram_momentum),
    SMIErgodicStrategy("smi_ergodic_histogram_continuation", "tendencia", "Continuacao: histograma negativo arma o pullback e o retorno acima de zero dispara em tendencia positiva.", smi_ergodic_histogram_continuation),
    SMIErgodicStrategy("smi_ergodic_atr_filter", "volatilidade", "Cruzamento SMI Ergodic aceito apenas quando o ATR confirma volatilidade suficiente.", smi_ergodic_atr_filter),
    SMIErgodicStrategy("smi_ergodic_adx_filter", "tendencia", "SMI Ergodic alinhado ao sinal somente quando o ADX confirma tendencia.", smi_ergodic_adx_filter),
    SMIErgodicStrategy("smi_ergodic_volume_filter", "volume", "Cruzamento altista do SMI Ergodic confirmado por volume acima da media.", smi_ergodic_volume_filter),
    SMIErgodicStrategy("smi_ergodic_macd_confirm", "combinada", "SMI Ergodic e MACD precisam concordar simultaneamente para manter a posicao comprada.", smi_ergodic_macd_confirm),
    SMIErgodicStrategy("smi_ergodic_123_reversal", "reversao", "Padrao causal 1-2-3 de reversao altista confirmado pelo SMI Ergodic.", smi_ergodic_123_reversal),
    SMIErgodicStrategy("smi_ergodic_dynamic_zones", "adaptativa", "Zonas dinamicas por quantis passados do SMI Ergodic, sem usar observacoes futuras.", smi_ergodic_dynamic_zones),
)
