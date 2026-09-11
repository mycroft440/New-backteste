from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

from .candles import Candle


SignalFunction = Callable[..., list[int]]
IndicatorFunction = Callable[..., list[float | None]]


@dataclass(frozen=True)
class StrategyExtension:
    name: str
    family: str
    description: str
    function: SignalFunction


_STRATEGIES: dict[str, StrategyExtension] = {}
_INDICATORS: dict[str, IndicatorFunction] = {}


def _validate_signature(function: Callable, kind: str) -> None:
    parameters = list(inspect.signature(function).parameters.values())
    if not parameters or parameters[0].name != "candles":
        raise ValueError(f"{kind} precisa receber 'candles' como primeiro parâmetro.")
    for parameter in parameters[1:]:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValueError(f"{kind} não aceita *args ou **kwargs.")
        if parameter.default is inspect.Signature.empty:
            raise ValueError(
                f"O parâmetro '{parameter.name}' de {kind.lower()} precisa de valor padrão."
            )


def strategy(
    name: str,
    *,
    family: str,
    description: str,
) -> Callable[[SignalFunction], SignalFunction]:
    """Register a user strategy with one decorator.

    The decorated function must receive ``list[Candle]`` as its first argument
    and return one truthy/falsy position signal for every candle.
    """

    normalized = name.strip().lower()
    if not normalized or not family.strip() or not description.strip():
        raise ValueError("name, family e description são obrigatórios.")

    def register(function: SignalFunction) -> SignalFunction:
        _validate_signature(function, "Estratégia")
        if normalized in _STRATEGIES:
            raise ValueError(f"Estratégia de extensão duplicada: {normalized}")
        _STRATEGIES[normalized] = StrategyExtension(
            normalized,
            family.strip(),
            description.strip(),
            function,
        )
        return function

    return register


def indicator(name: str) -> Callable[[IndicatorFunction], IndicatorFunction]:
    """Register a reusable indicator function for user strategies."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("O nome do indicador é obrigatório.")

    def register(function: IndicatorFunction) -> IndicatorFunction:
        _validate_signature(function, "Indicador")
        if normalized in _INDICATORS:
            raise ValueError(f"Indicador de extensão duplicado: {normalized}")
        _INDICATORS[normalized] = function
        return function

    return register


def registered_strategies() -> tuple[StrategyExtension, ...]:
    return tuple(_STRATEGIES[name] for name in sorted(_STRATEGIES))


def available_indicators() -> tuple[str, ...]:
    return tuple(sorted(_INDICATORS))


def registered_indicators() -> tuple[tuple[str, IndicatorFunction], ...]:
    return tuple((name, _INDICATORS[name]) for name in sorted(_INDICATORS))


def build_indicator(name: str, candles: list[Candle], **parameters) -> list[float | None]:
    normalized = name.strip().lower()
    if normalized not in _INDICATORS:
        raise ValueError(
            f"Indicador desconhecido: {name}. Disponíveis: {', '.join(available_indicators())}"
        )
    values = _INDICATORS[normalized](candles, **parameters)
    if len(values) != len(candles):
        raise RuntimeError("O indicador retornou quantidade diferente dos candles.")
    return values
