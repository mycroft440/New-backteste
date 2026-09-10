# New-backteste

Laboratório de backtest para estratégias da B3 usando os dados locais validados em `data/quotes/`.

## Etapa atual

O repositório possui duas camadas separadas:

1. **40 estratégias de compra e venda**, que geram estado binário por ação (`0 = fora`, `1 = comprado/em tendência de alta`);
2. **1 estratégia de gerenciamento de carteira**, responsável por escolher em qual das ações elegíveis investir.

As 40 estratégias de sinal vieram do catálogo de `mycroft440/b3-strategy-lab` e cobrem cruzamentos SMA/EMA, MACD, Donchian e RSI. A única estratégia de carteira atualmente habilitada também foi portada do B3 Strategy Lab e é a variante Top-1 Momentum mensal.

## Organização

```text
data/quotes/                              40 séries OHLCV B3
src/b3_backtest/strategies/               40 estratégias de compra/venda
src/b3_backtest/backtest/                 motor de execução por ativo
src/b3_backtest/portfolio/                único gerenciador de carteira Top-1
scripts/run_signal_strategy.py            CLI de sinais BUY/SELL
scripts/run_backtest_matrix.py            matriz das 40 estratégias
scripts/run_portfolio_backtest.py         backtest da carteira com capital compartilhado
tests/test_portfolio_management.py        testes do gerenciador de carteira
results/backtests/                        resultados das estratégias isoladas
results/portfolio/                        resultados do gerenciador de carteira
```

## Gerenciamento de carteira: Top-1 Momentum

A estratégia habilitada é `top1_momentum_lb126_monthly_signal_filter`, baseada na configuração `top1_momentum_lb126_skip0_trend0_vol63_equal_monthly_abs_cap1_adjusted` do B3 Strategy Lab, fixada no commit `8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d`.

A decisão é dividida em duas etapas:

1. **Indicador de tendência:** na última sessão de cada mês, a estratégia de compra/venda escolhida informa quais das 40 ações estão com estado `1`. Essas são as ações em tendência de alta segundo aquele indicador.
2. **Seleção da carteira:** entre as ações em alta, o gerenciador exige momentum positivo de 126 pregões e seleciona a de maior momentum. Como é Top-1, 100% do capital disponível é direcionado para uma única ação. Se nenhuma ação passar, a carteira fica em caixa.

A decisão é feita no fechamento do rebalanceamento e só é executada no **open do pregão seguinte**, evitando look-ahead. Entre rebalanceamentos, a ação escolhida continua obedecendo diariamente ao sinal de compra/venda: se o indicador sair de `1` para `0`, a posição é vendida no próximo open; ela pode ser recomprada se o indicador voltar a `1`, mas o ranking não é recalculado até o próximo rebalanceamento mensal.

## Execução da carteira

Por padrão, o teste integrado usa a estratégia de sinal `macd_24_52_18` e **R$ 1.000 de capital total compartilhado pela carteira**:

```bash
python scripts/run_portfolio_backtest.py --signal-strategy macd_24_52_18
```

Os principais arquivos gerados são:

```text
results/portfolio/macd_24_52_18/summary.json
results/portfolio/macd_24_52_18/rebalance_candidates.csv
results/portfolio/macd_24_52_18/rebalance_summary.csv
results/portfolio/macd_24_52_18/orders.csv
results/portfolio/macd_24_52_18/equity_curve.csv
results/portfolio/macd_24_52_18/annual_returns.csv
```

`rebalance_candidates.csv` mostra, para cada data de decisão e para cada ação, `indicator_uptrend`, momentum, ranking e se foi selecionada. `rebalance_summary.csv` mostra diretamente quais ações estavam em alta e qual delas foi escolhida para investimento.

## Dados e custos

Os preços usados no projeto são split-only e não adicionam dividendos/JCP aos retornos. O gerenciador aceita corretagem e slippage configuráveis, mas o baseline atual permanece em `0 bps` para preservar comparabilidade com os testes anteriores.

## Validação

Os sinais de compra/venda e o motor por ativo possuem verificação independente. O gerenciador de carteira possui testes específicos para garantir que apenas ações em tendência de alta possam ser selecionadas, que preços futuros não alterem decisões passadas, que a execução ocorra somente no open seguinte e que a carteira Top-1 nunca mantenha mais de uma ação simultaneamente.
