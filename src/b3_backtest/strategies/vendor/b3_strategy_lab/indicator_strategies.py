from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .candles import Candle
from .research_indicators import (
    chaikin_money_flow,
    ease_of_movement,
    elder_force_index,
    money_flow_index,
    negative_volume_index,
    realized_volatility,
    typical_price,
)


SignalFunction = Callable[..., list[int]]


@dataclass(frozen=True)
class IndicatorStrategy:
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


def _ema_optional(values: list[float | None], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period precisa ser maior que zero.")
    alpha = 2.0 / (period + 1)
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


def _validate_bounds(lower: float, upper: float) -> None:
    if not 0 <= lower < upper <= 100:
        raise ValueError("use 0 <= lower < upper <= 100.")


def mfi_reversal(candles: list[Candle], *, period: int = 14, lower: float = 25.0, upper: float = 70.0) -> list[int]:
    _validate_bounds(lower, upper)
    values = money_flow_index(candles, period=period)
    position = 0
    armed = False
    signals: list[int] = []
    for value in values:
        if value is not None:
            if position == 0:
                if value <= lower:
                    armed = True
                elif armed and value > lower:
                    position = 1
                    armed = False
            elif value >= upper:
                position = 0
        signals.append(position)
    return signals


def mfi_trend_follow(candles: list[Candle], *, period: int = 14, trend_window: int = 100, entry: float = 55.0, exit: float = 45.0) -> list[int]:
    if exit >= entry:
        raise ValueError("exit precisa ser menor que entry.")
    closes = [c.close for c in candles]
    trend = _sma(closes, trend_window)
    values = money_flow_index(candles, period=period)
    position = 0
    signals: list[int] = []
    for close, average, value in zip(closes, trend, values):
        if average is not None and value is not None:
            if position == 0 and close > average and value >= entry:
                position = 1
            elif position == 1 and (close < average or value <= exit):
                position = 0
        signals.append(position)
    return signals


def cmf_zero_cross(candles: list[Candle], *, period: int = 21) -> list[int]:
    values = chaikin_money_flow(candles, period=period)
    return [int(value is not None and value > 0.0) for value in values]


def cmf_threshold_hysteresis(candles: list[Candle], *, period: int = 21, entry: float = 0.05, exit: float = -0.02) -> list[int]:
    if exit >= entry:
        raise ValueError("exit precisa ser menor que entry.")
    values = chaikin_money_flow(candles, period=period)
    position = 0
    signals: list[int] = []
    for value in values:
        if value is not None:
            if position == 0 and value >= entry:
                position = 1
            elif position == 1 and value <= exit:
                position = 0
        signals.append(position)
    return signals


def efi_zero_cross(candles: list[Candle], *, period: int = 13) -> list[int]:
    values = elder_force_index(candles, period=period)
    return [int(value is not None and value > 0.0) for value in values]


def efi_trend_confirm(candles: list[Candle], *, period: int = 13, trend_window: int = 100) -> list[int]:
    closes = [c.close for c in candles]
    trend = _sma(closes, trend_window)
    values = elder_force_index(candles, period=period)
    return [int(value is not None and average is not None and value > 0.0 and close > average) for close, average, value in zip(closes, trend, values)]


def eom_zero_cross(candles: list[Candle], *, period: int = 14) -> list[int]:
    values = ease_of_movement(candles, period=period)
    return [int(value is not None and value > 0.0) for value in values]


def eom_trend_confirm(candles: list[Candle], *, period: int = 14, trend_window: int = 100) -> list[int]:
    closes = [c.close for c in candles]
    trend = _sma(closes, trend_window)
    values = ease_of_movement(candles, period=period)
    return [int(value is not None and average is not None and value > 0.0 and close > average) for close, average, value in zip(closes, trend, values)]


def nvi_ema_trend(candles: list[Candle], *, ema_period: int = 100) -> list[int]:
    nvi = negative_volume_index(candles)
    average = _ema_optional(nvi, ema_period)
    return [int(value is not None and mean is not None and value > mean) for value, mean in zip(nvi, average)]


def nvi_price_confirm(candles: list[Candle], *, nvi_ema: int = 100, price_sma: int = 100) -> list[int]:
    closes = [c.close for c in candles]
    nvi = negative_volume_index(candles)
    nvi_mean = _ema_optional(nvi, nvi_ema)
    price_mean = _sma(closes, price_sma)
    return [int(value is not None and nmean is not None and pmean is not None and value > nmean and close > pmean) for value, nmean, close, pmean in zip(nvi, nvi_mean, closes, price_mean)]


def realized_vol_low_momentum(candles: list[Candle], *, vol_period: int = 63, momentum_lookback: int = 63, max_vol: float = 0.45) -> list[int]:
    if momentum_lookback <= 0 or max_vol <= 0:
        raise ValueError("momentum_lookback e max_vol precisam ser positivos.")
    closes = [c.close for c in candles]
    vol = realized_volatility(candles, period=vol_period)
    signals: list[int] = []
    for index, value in enumerate(vol):
        positive = index >= momentum_lookback and closes[index] > closes[index - momentum_lookback]
        signals.append(int(value is not None and value <= max_vol and positive))
    return signals


def realized_vol_breakout(candles: list[Candle], *, vol_period: int = 63, breakout_lookback: int = 20, min_vol: float = 0.20, exit_sma: int = 20) -> list[int]:
    if breakout_lookback <= 0 or min_vol <= 0:
        raise ValueError("breakout_lookback e min_vol precisam ser positivos.")
    closes = [c.close for c in candles]
    vol = realized_volatility(candles, period=vol_period)
    mean = _sma(closes, exit_sma)
    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        if index >= breakout_lookback and vol[index] is not None and mean[index] is not None:
            prior_high = max(item.high for item in candles[index - breakout_lookback:index])
            if position == 0 and candle.close > prior_high and vol[index] >= min_vol:
                position = 1
            elif position == 1 and candle.close < mean[index]:
                position = 0
        signals.append(position)
    return signals


def typical_price_sma_trend(candles: list[Candle], *, period: int = 50) -> list[int]:
    values = [float(v) for v in typical_price(candles)]
    mean = _sma(values, period)
    return [int(average is not None and value > average) for value, average in zip(values, mean)]


def typical_price_pullback(candles: list[Candle], *, period: int = 50, pullback_pct: float = 0.03) -> list[int]:
    if not 0 < pullback_pct < 1:
        raise ValueError("pullback_pct precisa estar entre 0 e 1.")
    values = [float(v) for v in typical_price(candles)]
    mean = _sma(values, period)
    position = 0
    signals: list[int] = []
    for value, average in zip(values, mean):
        if average is not None:
            if position == 0 and value <= average * (1.0 - pullback_pct):
                position = 1
            elif position == 1 and value >= average:
                position = 0
        signals.append(position)
    return signals


def mfi_cmf_confirm(candles: list[Candle], *, mfi_period: int = 14, cmf_period: int = 21, entry_mfi: float = 55.0, exit_mfi: float = 45.0) -> list[int]:
    mfi = money_flow_index(candles, period=mfi_period)
    cmf = chaikin_money_flow(candles, period=cmf_period)
    position = 0
    signals: list[int] = []
    for mfi_value, cmf_value in zip(mfi, cmf):
        if mfi_value is not None and cmf_value is not None:
            if position == 0 and mfi_value >= entry_mfi and cmf_value > 0:
                position = 1
            elif position == 1 and (mfi_value <= exit_mfi or cmf_value < 0):
                position = 0
        signals.append(position)
    return signals


def mfi_efi_confirm(candles: list[Candle], *, mfi_period: int = 14, efi_period: int = 13, entry_mfi: float = 55.0, exit_mfi: float = 45.0) -> list[int]:
    mfi = money_flow_index(candles, period=mfi_period)
    efi = elder_force_index(candles, period=efi_period)
    position = 0
    signals: list[int] = []
    for mfi_value, efi_value in zip(mfi, efi):
        if mfi_value is not None and efi_value is not None:
            if position == 0 and mfi_value >= entry_mfi and efi_value > 0:
                position = 1
            elif position == 1 and (mfi_value <= exit_mfi or efi_value < 0):
                position = 0
        signals.append(position)
    return signals


def cmf_efi_confirm(candles: list[Candle], *, cmf_period: int = 21, efi_period: int = 13) -> list[int]:
    cmf = chaikin_money_flow(candles, period=cmf_period)
    efi = elder_force_index(candles, period=efi_period)
    return [int(c is not None and e is not None and c > 0 and e > 0) for c, e in zip(cmf, efi)]


def eom_nvi_confirm(candles: list[Candle], *, eom_period: int = 14, nvi_ema: int = 100) -> list[int]:
    eom = ease_of_movement(candles, period=eom_period)
    nvi = negative_volume_index(candles)
    nvi_mean = _ema_optional(nvi, nvi_ema)
    return [int(e is not None and n is not None and mean is not None and e > 0 and n > mean) for e, n, mean in zip(eom, nvi, nvi_mean)]


def low_vol_trend(candles: list[Candle], *, vol_period: int = 63, trend_window: int = 100, max_vol: float = 0.35) -> list[int]:
    closes = [c.close for c in candles]
    vol = realized_volatility(candles, period=vol_period)
    trend = _sma(closes, trend_window)
    return [int(v is not None and mean is not None and v <= max_vol and close > mean) for v, mean, close in zip(vol, trend, closes)]


def high_vol_breakout(candles: list[Candle], *, vol_period: int = 63, breakout_lookback: int = 55, min_vol: float = 0.30) -> list[int]:
    if breakout_lookback <= 0:
        raise ValueError("breakout_lookback precisa ser positivo.")
    vol = realized_volatility(candles, period=vol_period)
    position = 0
    signals: list[int] = []
    for index, candle in enumerate(candles):
        if index >= breakout_lookback and vol[index] is not None:
            prior_high = max(item.high for item in candles[index - breakout_lookback:index])
            prior_low = min(item.low for item in candles[index - max(10, breakout_lookback // 3):index])
            if position == 0 and candle.close > prior_high and vol[index] >= min_vol:
                position = 1
            elif position == 1 and candle.close < prior_low:
                position = 0
        signals.append(position)
    return signals


def volume_triple_confirm(candles: list[Candle], *, cmf_period: int = 21, efi_period: int = 13, eom_period: int = 14) -> list[int]:
    cmf = chaikin_money_flow(candles, period=cmf_period)
    efi = elder_force_index(candles, period=efi_period)
    eom = ease_of_movement(candles, period=eom_period)
    position = 0
    signals: list[int] = []
    for c, f, m in zip(cmf, efi, eom):
        if c is not None and f is not None and m is not None:
            positives = int(c > 0) + int(f > 0) + int(m > 0)
            if position == 0 and positives == 3:
                position = 1
            elif position == 1 and positives <= 1:
                position = 0
        signals.append(position)
    return signals


def mfi_price_trend(candles: list[Candle], *, mfi_period: int = 14, price_period: int = 50, entry_mfi: float = 52.0, exit_mfi: float = 45.0) -> list[int]:
    prices = [float(v) for v in typical_price(candles)]
    mean = _sma(prices, price_period)
    mfi = money_flow_index(candles, period=mfi_period)
    position = 0
    signals: list[int] = []
    for price, average, value in zip(prices, mean, mfi):
        if average is not None and value is not None:
            if position == 0 and price > average and value >= entry_mfi:
                position = 1
            elif position == 1 and (price < average or value <= exit_mfi):
                position = 0
        signals.append(position)
    return signals


def cmf_price_trend(candles: list[Candle], *, cmf_period: int = 21, price_period: int = 50, entry_cmf: float = 0.03, exit_cmf: float = -0.02) -> list[int]:
    closes = [c.close for c in candles]
    mean = _sma(closes, price_period)
    cmf = chaikin_money_flow(candles, period=cmf_period)
    position = 0
    signals: list[int] = []
    for close, average, value in zip(closes, mean, cmf):
        if average is not None and value is not None:
            if position == 0 and close > average and value >= entry_cmf:
                position = 1
            elif position == 1 and (close < average or value <= exit_cmf):
                position = 0
        signals.append(position)
    return signals


def nvi_mfi_confirm(candles: list[Candle], *, nvi_ema: int = 100, mfi_period: int = 14, entry_mfi: float = 52.0, exit_mfi: float = 45.0) -> list[int]:
    nvi = negative_volume_index(candles)
    mean = _ema_optional(nvi, nvi_ema)
    mfi = money_flow_index(candles, period=mfi_period)
    position = 0
    signals: list[int] = []
    for nvi_value, average, mfi_value in zip(nvi, mean, mfi):
        if nvi_value is not None and average is not None and mfi_value is not None:
            if position == 0 and nvi_value > average and mfi_value >= entry_mfi:
                position = 1
            elif position == 1 and (nvi_value < average or mfi_value <= exit_mfi):
                position = 0
        signals.append(position)
    return signals


INDICATOR_STRATEGIES = (
    IndicatorStrategy("mfi_reversal", "reversao_volume", "MFI recupera da sobrevenda; sai em sobrecompra.", mfi_reversal),
    IndicatorStrategy("mfi_trend_follow", "tendencia_volume", "MFI forte confirmado por tendencia de preco.", mfi_trend_follow),
    IndicatorStrategy("cmf_zero_cross", "volume", "Comprado enquanto Chaikin Money Flow permanece positivo.", cmf_zero_cross),
    IndicatorStrategy("cmf_threshold_hysteresis", "volume", "CMF com bandas distintas de entrada e saida para reduzir ruido.", cmf_threshold_hysteresis),
    IndicatorStrategy("efi_zero_cross", "volume", "Elder Force Index positivo como regime comprador.", efi_zero_cross),
    IndicatorStrategy("efi_trend_confirm", "tendencia_volume", "Force Index positivo confirmado por media de preco.", efi_trend_confirm),
    IndicatorStrategy("eom_zero_cross", "volume", "Ease of Movement positivo como sinal de compra.", eom_zero_cross),
    IndicatorStrategy("eom_trend_confirm", "tendencia_volume", "Ease of Movement positivo com tendencia de preco.", eom_trend_confirm),
    IndicatorStrategy("nvi_ema_trend", "volume", "Negative Volume Index acima de sua EMA.", nvi_ema_trend),
    IndicatorStrategy("nvi_price_confirm", "tendencia_volume", "NVI acima da EMA e preco acima da media.", nvi_price_confirm),
    IndicatorStrategy("realized_vol_low_momentum", "regime", "Momentum positivo apenas em regime de volatilidade realizada controlada.", realized_vol_low_momentum),
    IndicatorStrategy("realized_vol_breakout", "rompimento_volatilidade", "Rompimento acompanhado de volatilidade realizada minima, com saida por media.", realized_vol_breakout),
    IndicatorStrategy("typical_price_sma_trend", "tendencia", "Typical Price acima de sua media simples.", typical_price_sma_trend),
    IndicatorStrategy("typical_price_pullback", "reversao", "Compra desconto do Typical Price contra sua media e sai na recuperacao.", typical_price_pullback),
    IndicatorStrategy("mfi_cmf_confirm", "volume_hibrido", "MFI e CMF precisam confirmar fluxo comprador.", mfi_cmf_confirm),
    IndicatorStrategy("mfi_efi_confirm", "volume_hibrido", "MFI e Elder Force Index confirmam entrada e saida.", mfi_efi_confirm),
    IndicatorStrategy("cmf_efi_confirm", "volume_hibrido", "Chaikin Money Flow e Force Index simultaneamente positivos.", cmf_efi_confirm),
    IndicatorStrategy("eom_nvi_confirm", "volume_hibrido", "Ease of Movement positivo com NVI acima da EMA.", eom_nvi_confirm),
    IndicatorStrategy("low_vol_trend", "regime", "Tendencia de preco filtrada por baixa volatilidade realizada.", low_vol_trend),
    IndicatorStrategy("high_vol_breakout", "rompimento_volatilidade", "Rompimento de maxima em regime de volatilidade elevada.", high_vol_breakout),
    IndicatorStrategy("volume_triple_confirm", "volume_hibrido", "CMF, Force Index e Ease of Movement confirmam o regime comprador.", volume_triple_confirm),
    IndicatorStrategy("mfi_price_trend", "tendencia_volume", "Typical Price em tendencia com confirmacao do MFI.", mfi_price_trend),
    IndicatorStrategy("cmf_price_trend", "tendencia_volume", "Tendencia de preco com CMF acima de limiar positivo.", cmf_price_trend),
    IndicatorStrategy("nvi_mfi_confirm", "volume_hibrido", "NVI acima da EMA combinado com MFI comprador.", nvi_mfi_confirm),
)
