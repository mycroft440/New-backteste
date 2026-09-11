"""Extensoes locais carregadas automaticamente pelo catalogo.

Adicione funcoes decoradas com ``@strategy`` ou ``@indicator`` neste arquivo.
Nao e necessario editar ``strategies.py`` nem qualquer registro central.

A biblioteca ``research_indicators`` e importada automaticamente para tornar
indicadores reutilizaveis disponiveis a qualquer estrategia de extensao.
"""

from . import research_indicators as _research_indicators  # noqa: F401
from .extensions import strategy as _register_strategy
from .indicator_strategies import INDICATOR_STRATEGIES as _INDICATOR_STRATEGIES
from .smi_ergodic_strategies import (
    SMI_ERGODIC_STRATEGIES as _SMI_ERGODIC_STRATEGIES,
    smi_ergodic_volume_filter as _smi_ergodic_volume_filter,
)
from .trend_strategies import TREND_STRATEGIES as _TREND_STRATEGIES


def _audited_smi_ergodic_volume_filter(
    candles,
    *,
    long_period: int = 20,
    short_period: int = 5,
    signal_period: int = 5,
    volume_period: int = 20,
    volume_ratio: float = 1.0,
):
    """Expose the SMI volume filter through the extension volume-audit contract."""
    return _smi_ergodic_volume_filter(
        candles,
        long_period=long_period,
        short_period=short_period,
        signal_period=signal_period,
        volume_period=volume_period,
        volume_ratio=volume_ratio,
    )


# Register the indicator-driven, SMI Ergodic and trend-focused catalogs through
# the same extension API used by user strategies. This keeps strategies.py
# generic and makes every new engine immediately available to sweep_strategies()
# and portfolio_strategies(). The volume-filter variant is deliberately exposed
# by a local wrapper so the existing extension volume auditor includes it in its
# causal adjusted-volume checks.
for _item in (*_INDICATOR_STRATEGIES, *_SMI_ERGODIC_STRATEGIES, *_TREND_STRATEGIES):
    _function = (
        _audited_smi_ergodic_volume_filter
        if _item.name == "smi_ergodic_volume_filter"
        else _item.function
    )
    _register_strategy(
        _item.name,
        family=_item.family,
        description=_item.description,
    )(_function)
