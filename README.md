# New-backteste

Laboratório de backtest para estratégias da B3 usando os dados locais validados em `data/quotes/`.

## Etapa atual

O repositório possui três camadas separadas:

1. **40 estratégias de compra e venda** em `src/b3_backtest/strategies/`, com estado binário por ação;
2. **um gerenciador legado Top-1 Momentum** que combina uma das 40 estratégias de sinal com momentum de 126 pregões;
3. **o campeão portado do B3 Strategy Lab**, com indicador cross-sectional próprio e gerenciamento Top-1 mensal.

O campeão não substitui as 40 estratégias nem o gerenciador anterior. Ele é uma implementação separada, rastreável e testável.

## Organização

```text
data/quotes/                               40 séries OHLCV B3
src/b3_backtest/strategies/                40 estratégias BUY/SELL
src/b3_backtest/indicators/b3lab_champion.py indicador do campeão
src/b3_backtest/portfolio/top1_momentum.py gerenciador legado
src/b3_backtest/portfolio/b3lab_champion.py seletor do campeão
src/b3_backtest/portfolio/champion_engine.py simulador do campeão
scripts/run_backtest_matrix.py             matriz das 40 estratégias
scripts/run_portfolio_backtest.py          Top-1 Momentum legado
scripts/run_b3lab_champion.py              campeão do B3 Strategy Lab
tests/test_b3lab_champion.py               testes específicos do campeão
results/portfolio/b3lab_champion/           resultados materializados do campeão
```

## Campeão do B3 Strategy Lab

A configuração portada é:

`top1_short22_riskadj_rocfilter_w1_1_1_trend200_vol18_posscore`

Ela foi identificada no resultado persistido do `mycroft440/b3-strategy-lab`, commit de resultados `bb2533bacaadecbef102869a0fe89407d2eddf22`, no relatório `reports/portfolio_allocation_research_raw_events_raw_1d_roc_filter_short_fine_sweep.csv`. O código-fonte da fórmula está fixado no commit `8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d`.

No relatório de origem, essa configuração registrou `full_return = 187.20552082879905`, CAGR de aproximadamente 21,82%, Sharpe de aproximadamente 0,784 e drawdown máximo de aproximadamente -64,57%. Esses números pertencem ao universo e aos dados do B3 Strategy Lab e **não são assumidos como resultado do New-backteste**.

### Indicador

Na última sessão comum de cada mês, cada ação é avaliada apenas com dados disponíveis até aquele fechamento:

1. calcula ROC de 252, 126 e 63 pregões e exige que a média simples dos três seja positiva;
2. exige ROC de 22 pregões positivo;
3. exige `close > SMA200`;
4. calcula volatilidade anualizada dos últimos 18 retornos usando desvio-padrão amostral × `sqrt(252)`;
5. calcula `score = ROC22 / volatilidade18`.

### Gerenciamento

Entre as ações elegíveis, o maior `score` recebe rank 1. A carteira Top-1 direciona 100% do capital disponível para esse ativo. Se nenhuma ação passar pelos filtros, permanece em caixa.

A decisão é feita no fechamento mensal e executada apenas no **open da sessão comum seguinte**, evitando look-ahead. Entre rebalanceamentos, a posição é mantida até a próxima decisão mensal; não há saída diária por MACD/RSI. No último dia do backtest, qualquer posição restante é liquidada no fechamento, seguindo a convenção do motor de pesquisa de origem.

## Execução do campeão

Por padrão, usa R$ 1.000 de capital total, custo de 3,2 bps e slippage de 10 bps, conforme os parâmetros do experimento de origem:

```bash
python scripts/run_b3lab_champion.py
```

Arquivos gerados:

```text
results/portfolio/b3lab_champion/summary.json
results/portfolio/b3lab_champion/rebalance_candidates.csv
results/portfolio/b3lab_champion/rebalance_summary.csv
results/portfolio/b3lab_champion/orders.csv
results/portfolio/b3lab_champion/equity_curve.csv
results/portfolio/b3lab_champion/annual_returns.csv
```

O `summary.json` registra o indicador, o gerenciamento, commits de origem, custos, janela, lucro final, retorno, drawdown, anos completos negativos e média anual.

## Gerenciador legado

O gerenciador anterior continua disponível como `top1_momentum_lb126_monthly_signal_filter`. Ele usa uma das 40 estratégias binárias para determinar ações em tendência de alta e, entre elas, seleciona o maior momentum positivo de 126 pregões. O baseline legado continua executável com:

```bash
python scripts/run_portfolio_backtest.py --signal-strategy macd_24_52_18
```

## Dados

Os preços do projeto são split-only e não adicionam dividendos/JCP aos retornos. A janela comparável entre os 40 ativos começa em 24/02/2017. Reparos conhecidos de escala são aplicados antes da simulação e a janela efetiva passa pelo gate de qualidade dos dados.

## Validação

Os sinais BUY/SELL e o motor por ativo possuem verificação independente. O campeão possui testes próprios que verificam a fórmula matemática, ranking Top-1, ausência de influência de preços futuros em decisões passadas, execução somente no pregão seguinte, parâmetros de origem, posição única e liquidação final.
