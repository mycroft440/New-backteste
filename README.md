# New-backteste

Backtest independente para executar o catálogo completo do [`mycroft440/b3-strategy-lab`](https://github.com/mycroft440/b3-strategy-lab) usando os dados e contratos de mercado daquele repositório, sem manter uma cópia divergente das estratégias ou dos indicadores.

## Princípio central

O `b3-strategy-lab` continua sendo a fonte canônica de:

- estratégias e seus parâmetros;
- indicadores usados por essas estratégias;
- configurações de gerenciamento de carteira;
- loaders e verificadores de candles;
- evidências de eventos corporativos;
- construtores do universo point-in-time;
- motor realista e walk-forward certificado.

Em cada execução, o workflow faz checkout do `upstream_ref` solicitado (por padrão `main`) e imediatamente registra o SHA efetivo. Todos os jobs seguintes fazem checkout desse SHA exato. Assim, uma alteração no `main` durante uma rodada não muda o código no meio do backtest.

O catálogo não é copiado para este repositório. `scripts/catalog_manifest.py` importa `available_strategies()` diretamente do upstream e `_configs("adjusted", "all")`, grava todas as estratégias, parâmetros e configurações em `CATALOG.json` e calcula um fingerprint SHA-256 dos fontes relevantes.

## O que é testado

A matriz principal executa o produto cartesiano completo:

```text
TODAS as estratégias disponíveis
        ×
TODAS as configurações de gerenciamento (_configs("adjusted", "all"))
```

As estratégias são distribuídas entre shards. Dentro de cada shard, os sinais de uma estratégia são calculados uma vez e reutilizados ao testar todas as configurações de gerenciamento, preservando o contrato do motor original e evitando recalcular indicadores para cada combinação.

Depois do merge, `scripts/verify_matrix.py` verifica independentemente que:

- o número de linhas é exatamente o número esperado de combinações;
- cada par estratégia × gerenciamento aparece exatamente uma vez;
- não existem combinações inesperadas;
- não existem combinações faltantes;
- as principais métricas numéricas são finitas.

Além disso, o auditor original `audit_matrix_results.py` do `b3-strategy-lab` é executado contra o mesmo snapshot e o mesmo SHA usados nos shards.

## Dados e survivorship bias

A data final padrão não é simplesmente "hoje". `scripts/resolve_cutoff.py` resolve o último cutoff que já pode ser verificado com os dados versionados no `b3-strategy-lab`. Se uma data posterior for solicitada, o workflow falha e exige que o repositório upstream seja atualizado/certificado primeiro.

A partir desse cutoff, o workflow usa os próprios scripts do upstream para construir:

- universo histórico top-liquidity point-in-time;
- continuidade de tickers/classes;
- candles point-in-time;
- ações corporativas point-in-time;
- manifests de verificação;
- dados de execução necessários ao motor realista.

O snapshot gerado é hashado arquivo por arquivo. Os shards, o merge, o replay realista e o walk-forward verificam esses hashes antes de usar os dados.

## Pipeline

```text
prepare
  ├─ congela SHA do b3-strategy-lab
  ├─ roda pytest do upstream
  ├─ resolve cutoff certificado
  ├─ congela catálogo completo
  ├─ valida dados versionados
  ├─ constrói universo survivorship-safe PIT
  ├─ sincroniza dados/evidências PIT
  └─ publica snapshot + hashes

backtest (matrix)
  └─ N shards de estratégias
       └─ cada estratégia × todos os gerenciamentos

merge
  ├─ consolida a matriz completa
  ├─ roda auditor original do upstream
  ├─ prova o produto cartesiano completo
  └─ gera shortlist Top N

realistic_replay
  └─ reexecuta os finalistas no motor realista

walk_forward (habilitado por padrão)
  └─ executa pesquisa causal OOS do catálogo completo
     com conta contínua e gates de certificação

final_status
  └─ falha fechado se uma etapa obrigatória não passar
```

## Executar

Abra **Actions → New backtest — full B3 catalog → Run workflow**.

Parâmetros principais:

- `upstream_ref`: `main` para usar o estado atual do B3 Strategy Lab, ou um SHA para reprodução exata;
- `start`: padrão `2018-01-02`;
- `end`: vazio usa o cutoff verificável do upstream;
- `shards`: padrão `32`;
- `max_parallel`: padrão `20`;
- `initial_cash`: padrão `1000`;
- `cost_bps`: padrão `3.2`;
- `slippage_bps`: padrão `10`;
- `top_n`: padrão `10`;
- `run_walk_forward`: habilitado por padrão;
- `first_test_year`: padrão `2021`.

## Resultados

A execução publica artifacts separados para:

- contrato da rodada (`CATALOG.json`, hashes e auditoria dos inputs);
- shards individuais;
- matriz exaustiva e auditorias;
- replay realista dos finalistas;
- walk-forward OOS certificado;
- `FINAL_STATUS.json`.

A matriz retrospectiva serve para pesquisa/ranking e não deve ser interpretada, por si só, como evidência de desempenho futuro ou recomendação para dinheiro real. O walk-forward existe justamente para separar seleção no treino de avaliação fora da amostra e manter explícitas as limitações estatísticas e de execução.
