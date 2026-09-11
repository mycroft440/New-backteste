from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from b3_backtest.backtest import simulate_from_positions
from b3_backtest.data_quality import apply_known_scale_repairs, assert_trusted_window
from b3_backtest.strategies.catalog import list_original_strategies, run_strategy

INITIAL_CASH = 1000.0
EXPECTED_TICKERS = 40
EXPECTED_STRATEGIES = 40
DATA_DIR = Path("data/quotes")
OUTPUT_DIR = Path("results/backtests")


def _load_quotes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} has no date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{path} dates must be unique and sorted")
    return frame


def _common_window(frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = max(frame.iloc[0]["date"] for frame in frames.values())
    end = min(frame.iloc[-1]["date"] for frame in frames.values())
    if start >= end:
        raise RuntimeError(f"invalid common window: {start} -> {end}")
    return pd.Timestamp(start), pd.Timestamp(end)


def main() -> None:
    quote_paths = sorted(DATA_DIR.glob("*.csv"))
    if len(quote_paths) != EXPECTED_TICKERS:
        raise RuntimeError(f"expected {EXPECTED_TICKERS} quote files, found {len(quote_paths)}")

    specs = list_original_strategies()
    if len(specs) != EXPECTED_STRATEGIES:
        raise RuntimeError(f"expected {EXPECTED_STRATEGIES} original strategies, found {len(specs)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    repairs_applied: list[dict[str, object]] = []
    for quote_path in quote_paths:
        ticker = quote_path.stem
        frame = _load_quotes(quote_path)
        repaired, applied = apply_known_scale_repairs(frame, ticker)
        frames[ticker] = repaired
        repairs_applied.extend(applied)

    common_start, common_end = _common_window(frames)
    full_years = set(range(common_start.year + 1, common_end.year))
    if not full_years:
        raise RuntimeError("common window does not contain a full calendar year")

    # The comparison window must be free of gross discontinuities after the small,
    # documented vendor-scale repair layer. Old pre-window anomalies cannot leak into
    # executions, while the full history remains available to warm indicators.
    for ticker, frame in frames.items():
        test_frame = frame[(frame["date"] >= common_start) & (frame["date"] <= common_end)].copy()
        assert_trusted_window(test_frame, ticker)

    detail_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []

    for ticker, frame in frames.items():
        start_positions = frame.index[frame["date"] >= common_start]
        end_positions = frame.index[frame["date"] <= common_end]
        if len(start_positions) == 0 or len(end_positions) == 0:
            raise RuntimeError(f"{ticker}: missing common-window rows")
        start_pos = int(start_positions[0])
        end_pos = int(end_positions[-1])
        test_frame = frame.iloc[start_pos : end_pos + 1].reset_index(drop=True)

        for spec in specs:
            # Indicators use all history available before common_start as warm-up.
            full_positions = run_strategy(frame, spec.name)
            initial_target = full_positions[start_pos - 1] if start_pos > 0 else 0
            positions = full_positions[start_pos : end_pos + 1]

            result = simulate_from_positions(
                test_frame,
                positions,
                initial_cash=INITIAL_CASH,
                commission_bps=0.0,
                slippage_bps=0.0,
                initial_target=initial_target,
            )
            detail_rows.append(
                {
                    "strategy": spec.name,
                    "family": spec.family,
                    "ticker": ticker,
                    "start_date": common_start.date().isoformat(),
                    "end_date": common_end.date().isoformat(),
                    "initial_cash_brl": INITIAL_CASH,
                    "final_equity_brl": result.final_equity,
                    "final_profit_brl": result.final_profit,
                    "final_return_pct": result.final_return_pct,
                    "completed_trades": len(result.trades),
                    "max_trade_gain_pct": result.max_trade_gain_pct,
                    "max_trade_loss_pct": result.max_trade_loss_pct,
                }
            )
            for row in result.annual_returns.to_dict("records"):
                year = int(row["year"])
                annual_rows.append(
                    {
                        "strategy": spec.name,
                        "family": spec.family,
                        "ticker": ticker,
                        "year": year,
                        "is_full_year": year in full_years,
                        "return_pct": float(row["return_pct"]),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    annual = pd.DataFrame(annual_rows)

    strategy_year = (
        annual.groupby(["strategy", "family", "year", "is_full_year"], as_index=False)
        .agg(
            mean_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            tickers=("ticker", "nunique"),
        )
        .sort_values(["strategy", "year"])
    )

    summary_rows: list[dict[str, object]] = []
    for spec in specs:
        subset = detail[detail["strategy"] == spec.name].copy()
        years = strategy_year[
            (strategy_year["strategy"] == spec.name)
            & strategy_year["is_full_year"]
            & (strategy_year["tickers"] == EXPECTED_TICKERS)
        ].copy()

        if len(years) != len(full_years):
            raise RuntimeError(
                f"{spec.name}: expected {len(full_years)} complete strategy-years, found {len(years)}"
            )

        losing_years = years.loc[years["mean_return_pct"] < 0, "year"].astype(int).tolist()
        best_idx = years["mean_return_pct"].idxmax()
        worst_idx = years["mean_return_pct"].idxmin()
        max_gain_series = subset["max_trade_gain_pct"].dropna()
        max_loss_series = subset["max_trade_loss_pct"].dropna()

        summary_rows.append(
            {
                "strategy": spec.name,
                "family": spec.family,
                "tickers_tested": int(subset["ticker"].nunique()),
                "common_start": common_start.date().isoformat(),
                "common_end": common_end.date().isoformat(),
                "initial_cash_per_ticker_brl": INITIAL_CASH,
                "mean_final_equity_brl": float(subset["final_equity_brl"].mean()),
                "mean_final_profit_brl": float(subset["final_profit_brl"].mean()),
                "mean_final_return_pct": float(subset["final_return_pct"].mean()),
                "median_final_return_pct": float(subset["final_return_pct"].median()),
                "profitable_tickers": int((subset["final_profit_brl"] > 0).sum()),
                "losing_tickers": int((subset["final_profit_brl"] < 0).sum()),
                "losing_years": ";".join(str(year) for year in losing_years),
                "mean_annual_return_pct": float(years["mean_return_pct"].mean()),
                "best_year": int(years.loc[best_idx, "year"]),
                "best_year_return_pct": float(years.loc[best_idx, "mean_return_pct"]),
                "worst_year": int(years.loc[worst_idx, "year"]),
                "worst_year_return_pct": float(years.loc[worst_idx, "mean_return_pct"]),
                "max_trade_gain_pct": float(max_gain_series.max()) if not max_gain_series.empty else None,
                "max_trade_loss_pct": float(max_loss_series.min()) if not max_loss_series.empty else None,
                "completed_trades_total": int(subset["completed_trades"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_final_return_pct", "median_final_return_pct"], ascending=[False, False]
    )

    for table in (summary, detail, strategy_year):
        numeric_cols = table.select_dtypes(include="number").columns
        table[numeric_cols] = table[numeric_cols].round(4)

    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    detail.sort_values(["strategy", "ticker"]).to_csv(
        OUTPUT_DIR / "strategy_ticker_results.csv", index=False
    )
    strategy_year.to_csv(OUTPUT_DIR / "strategy_annual_results.csv", index=False)
    (OUTPUT_DIR / "data_repairs.json").write_text(
        json.dumps(repairs_applied, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    methodology = {
        "initial_cash_brl": INITIAL_CASH,
        "capital_model": "R$ 1.000 em cada backtest independente estrategia x ticker; sem compartilhamento de capital entre ativos",
        "comparison_window": {
            "start": common_start.date().isoformat(),
            "end": common_end.date().isoformat(),
            "rule": "maior primeira data e menor ultima data entre as 40 series, apos reparos de escala documentados",
        },
        "indicator_warmup": "indicadores usam o historico anterior ao inicio comum; capital e posicao so passam a existir na janela de teste",
        "execution": "sinal conhecido no fechamento; execucao no open do proximo candle negociavel",
        "position_sizing": "maximo numero inteiro de acoes que o caixa permite; sobra permanece em caixa",
        "portfolio_management": False,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "open_position_at_end": "marcada a mercado no ultimo close, sem venda forcada",
        "strategy_summary": "media simples dos 40 backtests independentes de R$ 1.000; nao representa uma carteira dividida entre 40 ativos",
        "annual_summary": f"somente anos-calendario completos: {min(full_years)} a {max(full_years)}; todos exigem 40 tickers",
        "losing_year_rule": "ano completo em que a media simples dos 40 retornos anuais e negativa",
        "max_gain_loss_rule": "maior ganho e maior perda percentuais entre trades encerrados nos 40 backtests independentes",
        "dividends_jcp": "nao adicionados pelo motor; reparos documentados alteram somente escala de preco/volume, sem retorno de proventos",
        "data_quality_gate": "falha se qualquer ticker mantiver salto open/close >=2x ou <=0.5x contra o fechamento anterior dentro da janela comum",
    }
    (OUTPUT_DIR / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"common window: {common_start.date()} -> {common_end.date()}")
    print(f"full calendar years: {min(full_years)} -> {max(full_years)}")
    print(f"documented vendor repairs applied: {len(repairs_applied)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
