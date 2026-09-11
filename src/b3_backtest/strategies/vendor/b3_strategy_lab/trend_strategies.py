from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from .candles import Candle
from .research_indicators import chaikin_money_flow, negative_volume_index, realized_volatility


SignalFunction = Callable[..., list[int]]


@dataclass(frozen=True)
class TrendStrategy:
    name: str
    family: str
    description: str
    function: SignalFunction


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


def _ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = [None for _ in values]
    if len(values) < period:
        return result
    current = sum(values[:period]) / period
    result[period - 1] = current
    alpha = 2.0 / (period + 1)
    for index in range(period, len(values)):
        current = alpha * values[index] + (1.0 - alpha) * current
        result[index] = current
    return result


def _ema_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    seed: list[float] = []
    current: float | None = None
    alpha = 2.0 / (period + 1)
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


def _atr(candles: list[Candle], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    if not candles:
        return []
    ranges: list[float] = []
    for index, candle in enumerate(candles):
        if index == 0:
            ranges.append(candle.high - candle.low)
        else:
            previous = candles[index - 1].close
            ranges.append(max(candle.high - candle.low, abs(candle.high - previous), abs(candle.low - previous)))
    result: list[float | None] = [None for _ in ranges]
    if len(ranges) < period:
        return result
    current = sum(ranges[:period]) / period
    result[period - 1] = current
    for index in range(period, len(ranges)):
        current = (current * (period - 1) + ranges[index]) / period
        result[index] = current
    return result


def _roc(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    for index, value in enumerate(values):
        if index < period or values[index - period] <= 0:
            result.append(None)
        else:
            result.append(value / values[index - period] - 1.0)
    return result


def ema_fast_slow_trend(candles: list[Candle], *, fast: int = 20, slow: int = 80) -> list[int]:
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("use 0 < fast < slow.")
    closes = [c.close for c in candles]
    fast_ma = _ema(closes, fast)
    slow_ma = _ema(closes, slow)
    return [int(f is not None and s is not None and f > s and close > f) for close, f, s in zip(closes, fast_ma, slow_ma)]


def ema_triple_alignment_trend(candles: list[Candle], *, fast: int = 20, middle: int = 50, slow: int = 200) -> list[int]:
    if not 0 < fast < middle < slow:
        raise ValueError("use 0 < fast < middle < slow.")
    closes = [c.close for c in candles]
    one, two, three = _ema(closes, fast), _ema(closes, middle), _ema(closes, slow)
    return [int(a is not None and b is not None and c is not None and close > a > b > c) for close, a, b, c in zip(closes, one, two, three)]


def sma_triple_alignment_trend(candles: list[Candle], *, fast: int = 20, middle: int = 50, slow: int = 200) -> list[int]:
    if not 0 < fast < middle < slow:
        raise ValueError("use 0 < fast < middle < slow.")
    closes = [c.close for c in candles]
    one, two, three = _sma(closes, fast), _sma(closes, middle), _sma(closes, slow)
    return [int(a is not None and b is not None and c is not None and close > a > b > c) for close, a, b, c in zip(closes, one, two, three)]


def ema_pullback_trend(candles: list[Candle], *, fast: int = 21, slow: int = 100) -> list[int]:
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("use 0 < fast < slow.")
    closes = [c.close for c in candles]
    fma, sma = _ema(closes, fast), _ema(closes, slow)
    position = 0
    armed = False
    signals: list[int] = []
    for index, close in enumerate(closes):
        fast_value, slow_value = fma[index], sma[index]
        if fast_value is not None and slow_value is not None:
            trend = fast_value > slow_value and close > slow_value
            if position == 0:
                if trend and close <= fast_value:
                    armed = True
                elif armed and trend and index > 0 and closes[index - 1] <= (fma[index - 1] or fast_value) and close > fast_value:
                    position = 1
                    armed = False
                elif not trend:
                    armed = False
            elif close < slow_value or fast_value < slow_value:
                position = 0
        signals.append(position)
    return signals


def sma_pullback_trend(candles: list[Candle], *, fast: int = 20, slow: int = 100) -> list[int]:
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("use 0 < fast < slow.")
    closes = [c.close for c in candles]
    fma, sma = _sma(closes, fast), _sma(closes, slow)
    position = 0
    armed = False
    signals: list[int] = []
    for index, close in enumerate(closes):
        fast_value, slow_value = fma[index], sma[index]
        if fast_value is not None and slow_value is not None:
            trend = fast_value > slow_value and close > slow_value
            if position == 0:
                if trend and close <= fast_value:
                    armed = True
                elif armed and trend and index > 0 and closes[index - 1] <= (fma[index - 1] or fast_value) and close > fast_value:
                    position = 1
                    armed = False
                elif not trend:
                    armed = False
            elif close < slow_value or fast_value < slow_value:
                position = 0
        signals.append(position)
    return signals


def donchian_40_20_trend(candles: list[Candle], *, entry_lookback: int = 40, exit_lookback: int = 20) -> list[int]:
    if entry_lookback <= 1 or exit_lookback <= 0 or exit_lookback >= entry_lookback:
        raise ValueError("use entry_lookback > exit_lookback > 0.")
    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        if index >= entry_lookback:
            entry = max(item.high for item in candles[index - entry_lookback:index])
            exit_level = min(item.low for item in candles[index - exit_lookback:index])
            if position == 0 and candle.close > entry:
                position = 1
            elif position == 1 and candle.close < exit_level:
                position = 0
        signals.append(position)
    return signals


def donchian_80_30_trend(candles: list[Candle], *, entry_lookback: int = 80, exit_lookback: int = 30) -> list[int]:
    if entry_lookback <= 1 or exit_lookback <= 0 or exit_lookback >= entry_lookback:
        raise ValueError("use entry_lookback > exit_lookback > 0.")
    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        if index >= entry_lookback:
            entry = max(item.high for item in candles[index - entry_lookback:index])
            exit_level = min(item.low for item in candles[index - exit_lookback:index])
            if position == 0 and candle.close > entry:
                position = 1
            elif position == 1 and candle.close < exit_level:
                position = 0
        signals.append(position)
    return signals


def highest_close_breakout_trend(candles: list[Candle], *, entry_lookback: int = 60, exit_lookback: int = 20) -> list[int]:
    if entry_lookback <= 1 or exit_lookback <= 0 or exit_lookback >= entry_lookback:
        raise ValueError("use entry_lookback > exit_lookback > 0.")
    closes = [c.close for c in candles]
    position = 0
    signals: list[int] = []
    for index, close in enumerate(closes):
        if index >= entry_lookback:
            highest = max(closes[index - entry_lookback:index])
            lowest = min(closes[index - exit_lookback:index])
            if position == 0 and close > highest:
                position = 1
            elif position == 1 and close < lowest:
                position = 0
        signals.append(position)
    return signals


def atr_channel_trend(candles: list[Candle], *, ema_period: int = 50, atr_period: int = 20, atr_mult: float = 1.5) -> list[int]:
    if ema_period <= 0 or atr_period <= 0 or atr_mult <= 0:
        raise ValueError("periodos e atr_mult precisam ser positivos.")
    closes = [c.close for c in candles]
    mean, atr = _ema(closes, ema_period), _atr(candles, atr_period)
    position = 0
    signals: list[int] = []
    for close, center, width in zip(closes, mean, atr):
        if center is not None and width is not None:
            if position == 0 and close > center + atr_mult * width:
                position = 1
            elif position == 1 and close < center:
                position = 0
        signals.append(position)
    return signals


def atr_trailing_trend(candles: list[Candle], *, trend_period: int = 100, atr_period: int = 20, atr_mult: float = 3.0) -> list[int]:
    if trend_period <= 0 or atr_period <= 0 or atr_mult <= 0:
        raise ValueError("periodos e atr_mult precisam ser positivos.")
    closes = [c.close for c in candles]
    trend, atr = _ema(closes, trend_period), _atr(candles, atr_period)
    position = 0
    peak = 0.0
    signals: list[int] = []
    for index, close in enumerate(closes):
        mean, width = trend[index], atr[index]
        if mean is not None and width is not None:
            if position == 0 and close > mean:
                position = 1
                peak = close
            elif position == 1:
                peak = max(peak, close)
                if close < mean or close < peak - atr_mult * width:
                    position = 0
                    peak = 0.0
        signals.append(position)
    return signals


def macd_zero_trend(candles: list[Candle], *, fast: int = 12, slow: int = 26) -> list[int]:
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("use 0 < fast < slow.")
    closes = [c.close for c in candles]
    one, two = _ema(closes, fast), _ema(closes, slow)
    return [int(a is not None and b is not None and a - b > 0 and close > b) for close, a, b in zip(closes, one, two)]


def macd_signal_long_trend(candles: list[Candle], *, fast: int = 12, slow: int = 26, signal_period: int = 9, trend_period: int = 100) -> list[int]:
    if fast <= 0 or slow <= 0 or signal_period <= 0 or trend_period <= 0 or fast >= slow:
        raise ValueError("periodos invalidos; use fast < slow e todos positivos.")
    closes = [c.close for c in candles]
    one, two = _ema(closes, fast), _ema(closes, slow)
    line = [None if a is None or b is None else a - b for a, b in zip(one, two)]
    signal = _ema_optional(line, signal_period)
    trend = _ema(closes, trend_period)
    return [int(m is not None and s is not None and t is not None and m > s and m > 0 and close > t) for close, m, s, t in zip(closes, line, signal, trend)]


def roc_dual_horizon_trend(candles: list[Candle], *, short: int = 63, long: int = 126) -> list[int]:
    if short <= 0 or long <= 0 or short >= long:
        raise ValueError("use 0 < short < long.")
    closes = [c.close for c in candles]
    one, two = _roc(closes, short), _roc(closes, long)
    return [int(a is not None and b is not None and a > 0 and b > 0) for a, b in zip(one, two)]


def roc_stack_trend(candles: list[Candle], *, short: int = 21, middle: int = 63, long: int = 126) -> list[int]:
    if not 0 < short < middle < long:
        raise ValueError("use 0 < short < middle < long.")
    closes = [c.close for c in candles]
    one, two, three = _roc(closes, short), _roc(closes, middle), _roc(closes, long)
    return [int(a is not None and b is not None and c is not None and a > 0 and b > 0 and c > 0) for a, b, c in zip(one, two, three)]


def ema_slope_price_trend(candles: list[Candle], *, ema_period: int = 80, slope_lookback: int = 20) -> list[int]:
    if ema_period <= 0 or slope_lookback <= 0:
        raise ValueError("ema_period e slope_lookback precisam ser positivos.")
    closes = [c.close for c in candles]
    average = _ema(closes, ema_period)
    signals: list[int] = []
    for index, close in enumerate(closes):
        current = average[index]
        prior = average[index - slope_lookback] if index >= slope_lookback else None
        signals.append(int(current is not None and prior is not None and current > prior and close > current))
    return signals


def sma_slope_price_trend(candles: list[Candle], *, sma_period: int = 100, slope_lookback: int = 20) -> list[int]:
    if sma_period <= 0 or slope_lookback <= 0:
        raise ValueError("sma_period e slope_lookback precisam ser positivos.")
    closes = [c.close for c in candles]
    average = _sma(closes, sma_period)
    signals: list[int] = []
    for index, close in enumerate(closes):
        current = average[index]
        prior = average[index - slope_lookback] if index >= slope_lookback else None
        signals.append(int(current is not None and prior is not None and current > prior and close > current))
    return signals


def efficiency_ratio_trend(candles: list[Candle], *, period: int = 40, threshold: float = 0.35) -> list[int]:
    if period <= 1 or not 0 < threshold < 1:
        raise ValueError("period precisa ser > 1 e threshold entre 0 e 1.")
    closes = [c.close for c in candles]
    signals: list[int] = []
    for index, close in enumerate(closes):
        if index < period:
            signals.append(0)
            continue
        direction = close - closes[index - period]
        noise = sum(abs(closes[pos] - closes[pos - 1]) for pos in range(index - period + 1, index + 1))
        ratio = 0.0 if noise <= 0 else abs(direction) / noise
        signals.append(int(direction > 0 and ratio >= threshold))
    return signals


def nvi_dual_ema_trend(candles: list[Candle], *, fast: int = 50, slow: int = 150) -> list[int]:
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("use 0 < fast < slow.")
    nvi = negative_volume_index(candles)
    fast_ma, slow_ma = _ema_optional(nvi, fast), _ema_optional(nvi, slow)
    return [int(value is not None and a is not None and b is not None and value > a > b) for value, a, b in zip(nvi, fast_ma, slow_ma)]


def cmf_ema_trend(candles: list[Candle], *, cmf_period: int = 21, ema_period: int = 100, entry_cmf: float = 0.05, exit_cmf: float = -0.05) -> list[int]:
    if cmf_period <= 0 or ema_period <= 0 or exit_cmf >= entry_cmf:
        raise ValueError("periodos devem ser positivos e exit_cmf < entry_cmf.")
    closes = [c.close for c in candles]
    average = _ema(closes, ema_period)
    cmf = chaikin_money_flow(candles, period=cmf_period)
    position = 0
    signals: list[int] = []
    for close, mean, flow in zip(closes, average, cmf):
        if mean is not None and flow is not None:
            if position == 0 and close > mean and flow >= entry_cmf:
                position = 1
            elif position == 1 and (close < mean or flow <= exit_cmf):
                position = 0
        signals.append(position)
    return signals


def low_vol_momentum_trend(candles: list[Candle], *, vol_period: int = 63, ema_period: int = 100, momentum_lookback: int = 63, max_vol: float = 0.40) -> list[int]:
    if vol_period <= 1 or ema_period <= 0 or momentum_lookback <= 0 or max_vol <= 0:
        raise ValueError("parametros precisam ser positivos e vol_period > 1.")
    closes = [c.close for c in candles]
    average = _ema(closes, ema_period)
    volatility = realized_volatility(candles, period=vol_period)
    signals: list[int] = []
    for index, close in enumerate(closes):
        momentum = index >= momentum_lookback and close > closes[index - momentum_lookback]
        mean = average[index]
        vol = volatility[index]
        signals.append(int(mean is not None and vol is not None and close > mean and momentum and vol <= max_vol))
    return signals


TREND_STRATEGIES = (
    TrendStrategy("ema_fast_slow_trend", "tendencia", "EMA curta acima da longa com preco acima da EMA curta.", ema_fast_slow_trend),
    TrendStrategy("ema_triple_alignment_trend", "tendencia", "Alinhamento de tres EMAs em ordem de alta.", ema_triple_alignment_trend),
    TrendStrategy("sma_triple_alignment_trend", "tendencia", "Alinhamento de tres SMAs em ordem de alta.", sma_triple_alignment_trend),
    TrendStrategy("ema_pullback_trend", "tendencia", "Compra recuperacao da EMA curta durante tendencia definida pela EMA longa.", ema_pullback_trend),
    TrendStrategy("sma_pullback_trend", "tendencia", "Compra recuperacao da SMA curta dentro de tendencia de alta.", sma_pullback_trend),
    TrendStrategy("donchian_40_20_trend", "tendencia", "Donchian 40/20: rompe maxima de 40 e sai na minima de 20.", donchian_40_20_trend),
    TrendStrategy("donchian_80_30_trend", "tendencia", "Donchian mais lento 80/30 para tendencias prolongadas.", donchian_80_30_trend),
    TrendStrategy("highest_close_breakout_trend", "tendencia", "Rompimento do maior fechamento com saida pelo menor fechamento recente.", highest_close_breakout_trend),
    TrendStrategy("atr_channel_trend", "tendencia", "Entrada acima de canal EMA+ATR e saida na perda da media.", atr_channel_trend),
    TrendStrategy("atr_trailing_trend", "tendencia", "Tendencia acima da EMA com trailing stop baseado em ATR.", atr_trailing_trend),
    TrendStrategy("macd_zero_trend", "tendencia", "MACD acima de zero com preco acima da EMA lenta.", macd_zero_trend),
    TrendStrategy("macd_signal_long_trend", "tendencia", "MACD acima do sinal e de zero, confirmado por EMA longa.", macd_signal_long_trend),
    TrendStrategy("roc_dual_horizon_trend", "tendencia", "ROC positivo simultaneamente em dois horizontes.", roc_dual_horizon_trend),
    TrendStrategy("roc_stack_trend", "tendencia", "Momentum positivo em tres horizontes para confirmar tendencia.", roc_stack_trend),
    TrendStrategy("ema_slope_price_trend", "tendencia", "Preco acima de EMA cuja inclinacao permanece positiva.", ema_slope_price_trend),
    TrendStrategy("sma_slope_price_trend", "tendencia", "Preco acima de SMA com inclinacao positiva.", sma_slope_price_trend),
    TrendStrategy("efficiency_ratio_trend", "tendencia", "Razao de eficiencia alta e direcao positiva para filtrar ruido lateral.", efficiency_ratio_trend),
    TrendStrategy("nvi_dual_ema_trend", "tendencia", "NVI acima de duas EMAs alinhadas em alta.", nvi_dual_ema_trend),
    TrendStrategy("cmf_ema_trend", "tendencia", "Tendencia de preco por EMA confirmada por Chaikin Money Flow.", cmf_ema_trend),
    TrendStrategy("low_vol_momentum_trend", "tendencia", "Momentum e EMA positivos sob teto de volatilidade realizada.", low_vol_momentum_trend),
)
