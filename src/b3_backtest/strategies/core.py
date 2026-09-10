from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
SignalEngine = Callable[..., list[int]]


def validate_ohlcv(frame: pd.DataFrame) -> None:
    """Validate the canonical quote shape without silently repairing strategy input."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    if frame.empty:
        raise ValueError("OHLCV frame is empty")

    for column in REQUIRED_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(lambda value: math.isfinite(float(value))).all():
            raise ValueError(f"column {column!r} contains non-finite values")

    prices = frame[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise ValueError("OHLC values must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be non-negative")
    if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high is below another OHLC field")
    if (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low is above another OHLC field")

    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="raise")
        if dates.duplicated().any():
            raise ValueError("date contains duplicates")
        if not dates.is_monotonic_increasing:
            raise ValueError("date must be sorted ascending")


def _sma(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")
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
    if window <= 0:
        raise ValueError("window must be positive")
    result: list[float | None] = [None] * len(values)
    available: list[float] = []
    current: float | None = None
    alpha = 2.0 / (window + 1)
    for index, value in enumerate(values):
        if value is None:
            continue
        value = float(value)
        if current is None:
            available.append(value)
            if len(available) == window:
                current = sum(available) / window
                result[index] = current
        else:
            current = value * alpha + current * (1.0 - alpha)
            result[index] = current
    return result


def _rsi(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
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


def moving_average_cross(
    frame: pd.DataFrame,
    *,
    average_type: str,
    fast: int,
    slow: int,
) -> list[int]:
    if average_type not in {"sma", "ema"}:
        raise ValueError("average_type must be 'sma' or 'ema'")
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("use 0 < fast < slow")

    closes = [float(value) for value in frame["close"]]
    average = _ema if average_type == "ema" else _sma
    fast_values = average(closes, fast)
    slow_values = average(closes, slow)
    return [
        int(fast_value is not None and slow_value is not None and fast_value > slow_value)
        for fast_value, slow_value in zip(fast_values, slow_values)
    ]


def macd(
    frame: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    signal_window: int,
    trend_window: int,
) -> list[int]:
    if min(fast, slow, signal_window) <= 0 or fast >= slow or trend_window < 0:
        raise ValueError("invalid MACD parameters")

    closes = [float(value) for value in frame["close"]]
    fast_values = _ema(closes, fast)
    slow_values = _ema(closes, slow)
    line = [
        None if fast_value is None or slow_value is None else fast_value - slow_value
        for fast_value, slow_value in zip(fast_values, slow_values)
    ]
    signal = _ema_optional(line, signal_window)
    trend = _sma(closes, trend_window) if trend_window else [None] * len(frame)

    return [
        int(
            line_value is not None
            and signal_value is not None
            and line_value > signal_value
            and (
                trend_window == 0
                or (trend[index] is not None and closes[index] > float(trend[index]))
            )
        )
        for index, (line_value, signal_value) in enumerate(zip(line, signal))
    ]


def donchian(
    frame: pd.DataFrame,
    *,
    entry_window: int,
    exit_window: int,
    trend_window: int,
) -> list[int]:
    if min(entry_window, exit_window) <= 0 or trend_window < 0:
        raise ValueError("invalid Donchian parameters")

    highs = [float(value) for value in frame["high"]]
    lows = [float(value) for value in frame["low"]]
    closes = [float(value) for value in frame["close"]]
    trend = _sma(closes, trend_window) if trend_window else [None] * len(frame)

    position = 0
    signals: list[int] = []
    for index, close in enumerate(closes):
        if index >= max(entry_window, exit_window):
            trend_ok = trend_window == 0 or (
                trend[index] is not None and close > float(trend[index])
            )
            if position == 0:
                prior_high = max(highs[index - entry_window : index])
                if close > prior_high and trend_ok:
                    position = 1
            else:
                prior_low = min(lows[index - exit_window : index])
                if close < prior_low:
                    position = 0
        signals.append(position)
    return signals


def rsi_reversion(
    frame: pd.DataFrame,
    *,
    rsi_period: int,
    lower: float,
    upper: float,
    trend_window: int,
    max_hold: int,
) -> list[int]:
    if rsi_period <= 0 or trend_window < 0 or max_hold <= 0:
        raise ValueError("invalid RSI parameters")
    if not 0 <= lower < upper <= 100:
        raise ValueError("use 0 <= lower < upper <= 100")

    closes = [float(value) for value in frame["close"]]
    values = _rsi(closes, rsi_period)
    trend = _sma(closes, trend_window) if trend_window else [None] * len(frame)

    position = 0
    held = 0
    signals: list[int] = []
    for index, value in enumerate(values):
        trend_ok = trend_window == 0 or (
            trend[index] is not None and closes[index] > float(trend[index])
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


ENGINES: dict[str, SignalEngine] = {
    "moving_average_cross": moving_average_cross,
    "macd": macd,
    "donchian": donchian,
    "rsi_reversion": rsi_reversion,
}


def run_engine(frame: pd.DataFrame, engine: str, parameters: dict[str, object]) -> list[int]:
    validate_ohlcv(frame)
    if engine not in ENGINES:
        raise KeyError(f"unknown strategy engine: {engine}")
    positions = ENGINES[engine](frame, **parameters)
    if len(positions) != len(frame) or any(value not in (0, 1) for value in positions):
        raise RuntimeError("strategy engine returned an invalid position series")
    return positions


def signal_events(
    frame: pd.DataFrame,
    positions: list[int],
    *,
    strategy_name: str,
) -> pd.DataFrame:
    """Convert 0/1 position state into BUY/SELL events only.

    These are close-of-bar signals. A future execution/backtest layer should execute
    them on the next tradable bar to avoid using the same close that created the signal.
    """
    if len(positions) != len(frame):
        raise ValueError("positions length must equal frame length")

    dates = (
        pd.to_datetime(frame["date"], errors="raise")
        if "date" in frame.columns
        else pd.to_datetime(frame.index, errors="raise")
    )
    closes = [float(value) for value in frame["close"]]

    rows: list[dict[str, object]] = []
    previous = 0
    for index, position in enumerate(positions):
        if position not in (0, 1):
            raise ValueError("positions must contain only 0 or 1")
        if position != previous:
            rows.append(
                {
                    "date": dates.iloc[index] if hasattr(dates, "iloc") else dates[index],
                    "close": closes[index],
                    "event": "BUY" if position == 1 else "SELL",
                    "strategy": strategy_name,
                }
            )
        previous = position

    return pd.DataFrame(rows, columns=["date", "close", "event", "strategy"])
