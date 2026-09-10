from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from b3_backtest.backtest import simulate_from_positions
from b3_backtest.strategies.catalog import list_strategies, run_strategy

INITIAL_CASH = 1000.0
MIN_YEAR_TICKERS = 20
DATA_DIR = Path("data/quotes")
OUTPUT_DIR = Path("results/backtests")


def _load_quotes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path} has no date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def main() -> None:
    quote_paths = sorted(DATA_DIR.glob("*.csv"))
    if len(quote_paths) != 40:
        raise RuntimeError(f"expected 40 quote files, found {len(quote_paths)}")

    specs = list_strategies()
    if len(specs) != 40:
        raise RuntimeError(f"expected 40 strategies, found {len(specs)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, object]] = []
    annual_rows: list[dict[str, object]] = []

    for quote_path in quote_paths:
        ticker = quote_path.stem
        frame = _load_quotes(quote_path)
        for spec in specs:
            positions = run_strategy(frame, spec.name)
            result = simulate_from_positions(
                frame,
                positions,
                initial_cash=INITIAL_CASH,
                commission_bps=0.0,
                slippage_bps=0.0,
            )
            detail_rows.append(
                {
                    "strategy": spec.name,
                    "family": spec.family,
                    "ticker": ticker,
                    "start_date": frame.iloc[0]["date"].date().isoformat(),
                    "end_date": frame.iloc[-1]["date"].date().isoformat(),
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
                annual_rows.append(
                    {
                        "strategy": spec.name,
                        "family": spec.family,
                        "ticker": ticker,
                        "year": int(row["year"]),
                        "return_pct": float(row["return_pct"]),
                    }
                )

    detail = pd.DataFrame(detail_rows)
    annual = pd.DataFrame(annual_rows)

    strategy_year = (
        annual.groupby(["strategy", "family", "year"], as_index=False)
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
            & (strategy_year["tickers"] >= MIN_YEAR_TICKERS)
        ].copy()

        losing_years = years.loc[years["mean_return_pct"] < 0, "year"].astype(int).tolist()
        if years.empty:
            best_year = None
            best_year_return = None
            worst_year = None
            worst_year_return = None
            mean_annual = None
        else:
            best_idx = years["mean_return_pct"].idxmax()
            worst_idx = years["mean_return_pct"].idxmin()
            best_year = int(years.loc[best_idx, "year"])
            best_year_return = float(years.loc[best_idx, "mean_return_pct"])
            worst_year = int(years.loc[worst_idx, "year"])
            worst_year_return = float(years.loc[worst_idx, "mean_return_pct"])
            mean_annual = float(years["mean_return_pct"].mean())

        max_gain_series = subset["max_trade_gain_pct"].dropna()
        max_loss_series = subset["max_trade_loss_pct"].dropna()

        summary_rows.append(
            {
                "strategy": spec.name,
                "family": spec.family,
                "tickers_tested": int(subset["ticker"].nunique()),
                "initial_cash_per_ticker_brl": INITIAL_CASH,
                "mean_final_equity_brl": float(subset["final_equity_brl"].mean()),
                "mean_final_profit_brl": float(subset["final_profit_brl"].mean()),
                "mean_final_return_pct": float(subset["final_return_pct"].mean()),
                "median_final_return_pct": float(subset["final_return_pct"].median()),
                "profitable_tickers": int((subset["final_profit_brl"] > 0).sum()),
                "losing_tickers": int((subset["final_profit_brl"] < 0).sum()),
                "losing_years": ";".join(str(year) for year in losing_years),
                "mean_annual_return_pct": mean_annual,
                "best_year": best_year,
                "best_year_return_pct": best_year_return,
                "worst_year": worst_year,
                "worst_year_return_pct": worst_year_return,
                "max_trade_gain_pct": float(max_gain_series.max()) if not max_gain_series.empty else None,
                "max_trade_loss_pct": float(max_loss_series.min()) if not max_loss_series.empty else None,
                "completed_trades_total": int(subset["completed_trades"].sum()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_final_return_pct", "median_final_return_pct"], ascending=[False, False]
    )

    numeric_cols = summary.select_dtypes(include="number").columns
    summary[numeric_cols] = summary[numeric_cols].round(4)
    detail_numeric = detail.select_dtypes(include="number").columns
    detail[detail_numeric] = detail[detail_numeric].round(4)
    annual_numeric = strategy_year.select_dtypes(include="number").columns
    strategy_year[annual_numeric] = strategy_year[annual_numeric].round(4)

    summary.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    detail.sort_values(["strategy", "ticker"]).to_csv(
        OUTPUT_DIR / "strategy_ticker_results.csv", index=False
    )
    strategy_year.to_csv(OUTPUT_DIR / "strategy_annual_results.csv", index=False)

    methodology = {
        "initial_cash_brl": INITIAL_CASH,
        "capital_model": "R$ 1.000 em cada backtest independente estrategia x ticker; sem compartilhamento de capital entre ativos",
        "execution": "sinal gerado no fechamento do candle; execucao no open do proximo candle negociavel",
        "position_sizing": "maximo numero inteiro de acoes que o caixa permite; sobra permanece em caixa",
        "portfolio_management": False,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "open_position_at_end": "marcada a mercado no ultimo close, sem venda forcada",
        "strategy_summary": "media simples dos 40 backtests independentes; nao representa uma carteira de R$ 1.000 dividida entre 40 ativos",
        "losing_year_rule": f"ano entra no resumo apenas quando possui pelo menos {MIN_YEAR_TICKERS} tickers com observacao; prejuizo quando a media dos retornos anuais desses backtests e negativa",
        "max_gain_loss_rule": "maior ganho e maior perda percentuais entre trades encerrados nos 40 backtests independentes",
        "dividends_jcp": "nao adicionados pelo motor; usa somente os OHLC existentes em data/quotes",
    }
    (OUTPUT_DIR / "methodology.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
