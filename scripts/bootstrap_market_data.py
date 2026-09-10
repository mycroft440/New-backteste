from __future__ import annotations

import shutil
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# 40 ativos B3 líquidos e com histórico longo/estável.
TICKERS = [
    "ABEV3", "BBDC4", "BBAS3", "BBSE3", "BPAC11", "BRAP4", "CMIG4", "CSMG3",
    "CPLE3", "CPFE3", "CSNA3", "CYRE3", "EGIE3", "B3SA3", "ENEV3", "ENGI11",
    "EQTL3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "KLBN11", "LREN3",
    "MGLU3", "MULT3", "PETR4", "PSSA3", "RADL3", "RAIL3", "RENT3", "SANB11",
    "SBSP3", "SMTO3", "SUZB3", "UGPA3", "USIM5", "VALE3", "WEGE3", "TOTS3",
]

START = "2000-01-01"
MIN_ROWS = 1500
OUTPUT_DIR = Path("data/quotes")
STAGING_DIR = Path(".quotes_staging")


def download_adjusted(ticker: str) -> pd.DataFrame:
    symbol = f"{ticker}.SA"
    end = (date.today() + timedelta(days=1)).isoformat()
    errors: list[str] = []

    # Primeiro usa o reparador do yfinance. Em séries em que o reparador falha por
    # dados auxiliares problemáticos (por exemplo, volume NaN histórico), recua
    # para a série Yahoo auto-ajustada sem o reparador experimental. Em ambos os
    # casos o resultado ainda passa integralmente pelo nosso gate estrutural.
    for repair in (True, False):
        for attempt in range(1, 3):
            try:
                df = yf.Ticker(symbol).history(
                    start=START,
                    end=end,
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    repair=repair,
                    keepna=False,
                    timeout=30,
                    raise_errors=True,
                )
                if df.empty:
                    raise RuntimeError("histórico vazio")
                if not repair:
                    print(f"  {ticker}: fallback repair=False", flush=True)
                return df
            except Exception as exc:  # noqa: BLE001
                errors.append(f"repair={repair} attempt={attempt}: {exc}")
                if attempt < 2:
                    time.sleep(attempt * 5)

    raise RuntimeError(f"{ticker}: falha ao baixar dados: {' | '.join(errors)}")


def normalize_and_validate(ticker: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise RuntimeError(f"{ticker}: colunas ausentes: {missing}")

    df = raw[required].copy()
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="raise").round().astype("int64")

    if len(df) < MIN_ROWS:
        raise RuntimeError(f"{ticker}: apenas {len(df)} pregões; mínimo exigido {MIN_ROWS}")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError(f"{ticker}: OHLC não positivo")
    if (df["volume"] < 0).any():
        raise RuntimeError(f"{ticker}: volume negativo")

    # Algumas séries Yahoo ajustadas contêm raros candles históricos em que High/Low
    # não englobam Open/Close. Mantemos Open/Close da fonte e canonizamos somente
    # os extremos, de forma determinística, antes do gate final.
    original_high = df["high"].copy()
    original_low = df["low"].copy()
    extrema = df[["open", "high", "low", "close"]]
    df["high"] = extrema.max(axis=1)
    df["low"] = extrema.min(axis=1)
    repaired_rows = int(((df["high"] != original_high) | (df["low"] != original_low)).sum())

    max_ocl = df[["open", "close", "low"]].max(axis=1)
    min_och = df[["open", "close", "high"]].min(axis=1)
    bad = (df["high"] + 1e-12 < max_ocl) | (df["low"] - 1e-12 > min_och)
    if bad.any():
        first = df.index[bad][0].date().isoformat()
        raise RuntimeError(f"{ticker}: candle OHLC inválido após reparo em {first}")

    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(8)

    # Revalida após arredondamento/serialização.
    if (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    ).any():
        raise RuntimeError(f"{ticker}: OHLC inválido após arredondamento")

    return df, repaired_rows


def main() -> int:
    if len(TICKERS) != 40 or len(set(TICKERS)) != 40:
        raise RuntimeError("a lista deve conter exatamente 40 tickers únicos")

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    staging_quotes = STAGING_DIR / "quotes"
    staging_quotes.mkdir(parents=True, exist_ok=True)

    summaries: list[str] = []
    total_repairs = 0
    for pos, ticker in enumerate(TICKERS, start=1):
        print(f"[{pos:02d}/40] {ticker}", flush=True)
        df, repaired_rows = normalize_and_validate(ticker, download_adjusted(ticker))
        total_repairs += repaired_rows
        target = staging_quotes / f"{ticker}.csv"
        df.to_csv(target, date_format="%Y-%m-%d", lineterminator="\n")
        summaries.append(
            f"{ticker}: {len(df)} rows, {df.index[0].date().isoformat()} -> "
            f"{df.index[-1].date().isoformat()}, canonicalized_candles={repaired_rows}"
        )

    files = sorted(staging_quotes.glob("*.csv"))
    if len(files) != 40:
        raise RuntimeError(f"esperados 40 CSVs, encontrados {len(files)}")

    # Só toca em data/ depois que todas as 40 séries passaram o gate.
    shutil.rmtree(OUTPUT_DIR.parent, ignore_errors=True)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_quotes), str(OUTPUT_DIR))
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

    print("\nVALIDATED ADJUSTED QUOTES")
    for line in summaries:
        print(line)
    print(f"\nCanonicalized OHLC rows: {total_repairs}")
    print(f"OK: {len(files)} séries em {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
