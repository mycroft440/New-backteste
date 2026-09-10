# New-backteste

Laboratório de backtest para estratégias da B3 usando os dados locais validados em `data/quotes/`.

## Etapa atual: sinais de compra e venda

Nesta fase o repositório contém **somente a lógica necessária para gerar sinais de entrada e saída**. Gestão de carteira fica explicitamente fora do escopo por enquanto: não há tamanho de posição, capital, pesos, alocação, rebalanceamento, limite de exposição ou construção de portfólio.

A primeira leva foi portada do catálogo de `mycroft440/b3-strategy-lab` preservando parâmetros e regras de sinal. São **40 estratégias** distribuídas em quatro grupos:

- 10 cruzamentos de médias SMA/EMA;
- 10 variações de MACD;
- 10 rompimentos de Donchian;
- 10 reversões por RSI.

## Organização

```text
data/quotes/                         40 séries OHLCV B3
src/b3_backtest/strategies/core.py   motores e validação de sinais
src/b3_backtest/strategies/catalog.py catálogo e parâmetros das estratégias
scripts/run_signal_strategy.py       CLI para gerar BUY/SELL
tests/test_signal_strategies.py      testes do catálogo e dos dados
```

Os motores retornam estado binário (`0 = fora`, `1 = comprado`). A camada de eventos transforma apenas mudanças de estado em `BUY` e `SELL`.

## Uso

Listar estratégias:

```bash
python scripts/run_signal_strategy.py --list
```

Gerar sinais para uma ação:

```bash
python scripts/run_signal_strategy.py \
  --ticker PETR4 \
  --strategy donchian_breakout_20_10 \
  --output results/petr4-donchian.csv
```

O arquivo de saída contém somente:

```text
date,close,event,strategy
```

## Regra temporal

As estratégias calculam o sinal no fechamento do candle atual. Quando adicionarmos a camada de execução/backtest, a ordem deverá ser executada **no próximo candle negociável**, e não retroativamente no mesmo fechamento que produziu o sinal. Isso evita usar informação do fechamento antes de ela existir.

## Dados

Os 40 CSVs em `data/quotes/` permanecem separados do código de estratégia. Eles usam `date,open,high,low,close,volume` e, conforme a etapa anterior deste projeto, os preços não incorporam dividendos/JCP ao retorno.

## Próximas etapas

Depois de validar esta primeira leva, outras famílias do `b3-strategy-lab` podem ser portadas para a mesma interface de sinais. Gestão de carteira será tratada separadamente somente quando a camada de compra/venda estiver validada.
