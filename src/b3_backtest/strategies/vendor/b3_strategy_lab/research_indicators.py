from __future__ import annotations

import math

from .candles import Candle
from .extensions import indicator


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


def _rolling_sum(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    result: list[float | None] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        result.append(None if index + 1 < period else running)
    return result


@indicator("typical_price")
def typical_price(candles: list[Candle]) -> list[float | None]:
    return [(candle.high + candle.low + candle.close) / 3.0 for candle in candles]


@indicator("money_flow_index")
def money_flow_index(candles: list[Candle], period: int = 14) -> list[float | None]:
    """MFI usando volume split-normalizado do Candle, consistente com o OHLC normalizado."""
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    typical = [(c.high + c.low + c.close) / 3.0 for c in candles]
    positive = [0.0 for _ in candles]
    negative = [0.0 for _ in candles]
    for index in range(1, len(candles)):
        flow = typical[index] * float(candles[index].volume)
        if typical[index] > typical[index - 1]:
            positive[index] = flow
        elif typical[index] < typical[index - 1]:
            negative[index] = flow
    positive_sum = _rolling_sum(positive, period)
    negative_sum = _rolling_sum(negative, period)
    result: list[float | None] = []
    for up, down in zip(positive_sum, negative_sum):
        if up is None or down is None:
            result.append(None)
        elif down == 0:
            result.append(100.0 if up > 0 else 50.0)
        else:
            ratio = up / down
            result.append(100.0 - 100.0 / (1.0 + ratio))
    return result


@indicator("chaikin_money_flow")
def chaikin_money_flow(candles: list[Candle], period: int = 21) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    flow_volume: list[float] = []
    volumes = [float(candle.volume) for candle in candles]
    for candle in candles:
        spread = candle.high - candle.low
        multiplier = 0.0 if spread <= 0 else ((candle.close - candle.low) - (candle.high - candle.close)) / spread
        flow_volume.append(multiplier * float(candle.volume))
    flow_sum = _rolling_sum(flow_volume, period)
    volume_sum = _rolling_sum(volumes, period)
    return [
        None if flow is None or volume is None or volume <= 0 else flow / volume
        for flow, volume in zip(flow_sum, volume_sum)
    ]


@indicator("elder_force_index")
def elder_force_index(candles: list[Candle], period: int = 13) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    raw = [0.0]
    for index in range(1, len(candles)):
        raw.append((candles[index].close - candles[index - 1].close) * float(candles[index].volume))
    return _ema(raw, period)


@indicator("ease_of_movement")
def ease_of_movement(candles: list[Candle], period: int = 14) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    raw = [0.0 for _ in candles]
    for index in range(1, len(candles)):
        candle = candles[index]
        previous = candles[index - 1]
        midpoint_move = ((candle.high + candle.low) - (previous.high + previous.low)) / 2.0
        range_size = candle.high - candle.low
        volume = float(candle.volume)
        if range_size > 0 and volume > 0:
            raw[index] = midpoint_move * range_size / volume
    return _sma(raw, period)


@indicator("negative_volume_index")
def negative_volume_index(candles: list[Candle], base: float = 1000.0) -> list[float | None]:
    if base <= 0:
        raise ValueError("base precisa ser maior que zero.")
    if not candles:
        return []
    values = [float(base)]
    current = float(base)
    for index in range(1, len(candles)):
        previous_close = candles[index - 1].close
        if candles[index].volume < candles[index - 1].volume and previous_close > 0:
            current *= 1.0 + (candles[index].close / previous_close - 1.0)
        values.append(current)
    return values


@indicator("realized_volatility")
def realized_volatility(candles: list[Candle], period: int = 63) -> list[float | None]:
    if period <= 1:
        raise ValueError("period precisa ser maior que 1.")
    returns = [0.0]
    for index in range(1, len(candles)):
        previous = candles[index - 1].close
        returns.append(0.0 if previous <= 0 else candles[index].close / previous - 1.0)
    result: list[float | None] = []
    for index in range(len(returns)):
        if index + 1 < period:
            result.append(None)
            continue
        sample = returns[index + 1 - period : index + 1]
        mean = sum(sample) / len(sample)
        variance = sum((value - mean) ** 2 for value in sample) / len(sample)
        result.append(math.sqrt(max(variance, 0.0)) * math.sqrt(252.0))
    return result
