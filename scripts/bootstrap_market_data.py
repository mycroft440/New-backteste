from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

B3_YEARLY_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
TICKER_RE = re.compile(r"^[A-Z]{4}[3456]$")
EQUITY_PREFIXES = ("ON", "PN")


@dataclass(frozen=True)
class Quote:
    date: str
    ticker: str
    company: str
    specification: str
    open: float
    high: float
    low: float
    average: float
    close: float
    trades: int
    quantity: int
    financial_volume: float
    isin: str
    quotation_factor: int
    source_year: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_year(year: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"COTAHIST_A{year}.ZIP"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = B3_YEARLY_URL.format(year=year)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 New-backteste market-data bootstrap",
                    "Accept": "application/zip,application/octet-stream,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise RuntimeError(f"arquivo anual muito pequeno: {len(payload)} bytes")
            target.write_bytes(payload)
            with zipfile.ZipFile(target) as archive:
                members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
                if not members:
                    raise RuntimeError("ZIP anual não contém COTAHIST TXT")
            return target
        except Exception as error:  # noqa: BLE001
            last_error = error
            target.unlink(missing_ok=True)
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(f"falha ao baixar {url}: {last_error}")


def parse_price(line: str, start: int, end: int) -> float:
    return int(line[start:end]) / 100.0


def parse_line(line: str, year: int) -> Quote | None:
    if len(line) < 245 or line[0:2] != "01":
        return None
    if line[24:27] != "010":
        return None
    if line[10:12] != "02":
        return None

    ticker = line[12:24].strip().upper()
    if not TICKER_RE.fullmatch(ticker):
        return None
    specification = " ".join(line[39:49].strip().upper().split())
    if not specification.startswith(EQUITY_PREFIXES):
        return None

    raw_date = line[2:10]
    quote_date = f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    return Quote(
        date=quote_date,
        ticker=ticker,
        company=" ".join(line[27:39].strip().upper().split()),
        specification=specification,
        open=parse_price(line, 56, 69),
        high=parse_price(line, 69, 82),
        low=parse_price(line, 82, 95),
        average=parse_price(line, 95, 108),
        close=parse_price(line, 108, 121),
        trades=int(line[147:152]),
        quantity=int(line[152:170]),
        financial_volume=int(line[170:188]) / 100.0,
        quotation_factor=int(line[210:217]),
        isin=line[230:242].strip().upper(),
        source_year=year,
    )


def iter_quotes(archive_path: Path, year: int):
    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.upper().endswith(".TXT")]
        if len(members) != 1:
            raise RuntimeError(f"{archive_path.name}: esperado 1 TXT, encontrado {len(members)}")
        with archive.open(members[0]) as raw:
            for payload in raw:
                line = payload.decode("latin-1").rstrip("\r\n")
                quote = parse_line(line, year)
                if quote is not None:
                    yield quote


def select_universe(selection_archive: Path, selection_year: int, top_n: int) -> list[dict]:
    stats: dict[str, dict[str, object]] = {}
    for quote in iter_quotes(selection_archive, selection_year):
        item = stats.setdefault(
            quote.ticker,
            {
                "ticker": quote.ticker,
                "root": quote.ticker[:4],
                "company": quote.company,
                "specification": quote.specification,
                "sessions": 0,
                "trades": 0,
                "financial_volume": 0.0,
                "first_date": quote.date,
                "last_date": quote.date,
            },
        )
        item["sessions"] = int(item["sessions"]) + 1
        item["trades"] = int(item["trades"]) + quote.trades
        item["financial_volume"] = float(item["financial_volume"]) + quote.financial_volume
        item["first_date"] = min(str(item["first_date"]), quote.date)
        item["last_date"] = max(str(item["last_date"]), quote.date)

    # Evita duas classes da mesma companhia: para cada raiz, mantém a classe mais líquida.
    best_by_root: dict[str, dict] = {}
    for item in stats.values():
        if int(item["sessions"]) < 180:
            continue
        root = str(item["root"])
        current = best_by_root.get(root)
        if current is None or (
            float(item["financial_volume"]), int(item["trades"]), str(item["ticker"])
        ) > (
            float(current["financial_volume"]), int(current["trades"]), str(current["ticker"])
        ):
            best_by_root[root] = item

    ranked = sorted(
        best_by_root.values(),
        key=lambda item: (-float(item["financial_volume"]), -int(item["trades"]), str(item["ticker"])),
    )
    if len(ranked) < top_n:
        raise RuntimeError(f"apenas {len(ranked)} emissores elegíveis no ano de seleção")
    selected = ranked[:top_n]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    return selected


def validate_rows(rows_by_ticker: dict[str, list[Quote]], selected_tickers: list[str]) -> dict:
    duplicate_count = 0
    invalid_ohlc: list[dict] = []
    nonpositive: list[dict] = []
    ticker_summary: dict[str, dict] = {}

    for ticker in selected_tickers:
        rows = sorted(rows_by_ticker.get(ticker, []), key=lambda row: row.date)
        seen: set[str] = set()
        for row in rows:
            if row.date in seen:
                duplicate_count += 1
            seen.add(row.date)
            if row.high < max(row.open, row.close, row.low) or row.low > min(row.open, row.close, row.high):
                invalid_ohlc.append({"ticker": ticker, "date": row.date})
            if min(row.open, row.high, row.low, row.close) <= 0:
                nonpositive.append({"ticker": ticker, "date": row.date})
        ticker_summary[ticker] = {
            "rows": len(rows),
            "first_date": rows[0].date if rows else None,
            "last_date": rows[-1].date if rows else None,
            "unique_dates": len(seen),
        }

    missing = [ticker for ticker in selected_tickers if not rows_by_ticker.get(ticker)]
    raw_valid = not missing and duplicate_count == 0 and not invalid_ohlc and not nonpositive
    return {
        "raw_ohlcv_valid": raw_valid,
        "duplicate_ticker_dates": duplicate_count,
        "invalid_ohlc_count": len(invalid_ohlc),
        "nonpositive_ohlc_count": len(nonpositive),
        "missing_tickers": missing,
        "invalid_ohlc_examples": invalid_ohlc[:20],
        "nonpositive_examples": nonpositive[:20],
        "tickers": ticker_summary,
        "important": (
            "COTAHIST é preço bruto. Este gate valida integridade dos candles, mas NÃO declara "
            "a série pronta para indicadores atravessarem splits/grupamentos. Ajustes corporativos "
            "e transições de ticker serão uma camada separada e independente."
        ),
        "ready_for_full_strategy_backtest": False,
    }


def write_outputs(
    output_dir: Path,
    selected: list[dict],
    rows_by_ticker: dict[str, list[Quote]],
    archives: dict[int, Path],
    selection_year: int,
    start_year: int,
    end_year: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candles_dir = output_dir / "candles_raw"
    candles_dir.mkdir(parents=True, exist_ok=True)
    selected_tickers = [str(item["ticker"]) for item in selected]

    fields = [
        "date",
        "ticker",
        "company",
        "specification",
        "open",
        "high",
        "low",
        "average",
        "close",
        "trades",
        "quantity",
        "financial_volume",
        "isin",
        "quotation_factor",
        "source_year",
    ]
    for ticker in selected_tickers:
        target = candles_dir / f"{ticker}.csv.gz"
        with gzip.open(target, "wt", encoding="utf-8", newline="", compresslevel=9) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in sorted(rows_by_ticker[ticker], key=lambda item: item.date):
                writer.writerow(row.__dict__)

    validation = validate_rows(rows_by_ticker, selected_tickers)
    if not validation["raw_ohlcv_valid"]:
        raise RuntimeError("dataset bruto falhou validação: " + json.dumps(validation, ensure_ascii=False))

    source_archives = [
        {
            "year": year,
            "url": B3_YEARLY_URL.format(year=year),
            "filename": path.name,
            "sha256": sha256(path),
            "size": path.stat().st_size,
        }
        for year, path in sorted(archives.items())
    ]
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "independent_b3_cotahist_40_raw",
        "market_data_source": "B3 official COTAHIST yearly archives",
        "strategy_lab_market_data_used": False,
        "price_adjustment": "raw_unadjusted",
        "dividends_jcp_included": False,
        "selection": {
            "selection_year": selection_year,
            "test_start_year": start_year,
            "lookahead_into_test_period": False,
            "rule": "top 40 distinct ticker roots by total financial volume in prior calendar year; minimum 180 sessions; market 010; BDI 02; ON/PN only",
            "top_n": len(selected),
            "rows": selected,
            "tickers": selected_tickers,
        },
        "coverage": {"from_year": selection_year, "test_from_year": start_year, "through_year": end_year},
        "sources": source_archives,
        "validation_file": "validation.json",
        "next_required_layer": "independent corporate-action and ticker-transition adjustment before full strategy backtest",
    }
    (output_dir / "universe_40.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap independente de 40 ações B3 via COTAHIST oficial.")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/b3_cotahist"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/market"))
    args = parser.parse_args()

    if args.start_year < 2001 or args.end_year < args.start_year or args.top <= 0:
        parser.error("intervalo/top inválido")
    selection_year = args.start_year - 1

    archives: dict[int, Path] = {}
    for year in range(selection_year, args.end_year + 1):
        print(f"downloading/checking COTAHIST {year}", flush=True)
        archives[year] = download_year(year, args.cache_dir)

    selected = select_universe(archives[selection_year], selection_year, args.top)
    selected_tickers = {str(item["ticker"]) for item in selected}
    print("selected:", ", ".join(sorted(selected_tickers)), flush=True)

    rows_by_ticker: dict[str, list[Quote]] = defaultdict(list)
    for year, archive in sorted(archives.items()):
        for quote in iter_quotes(archive, year):
            if quote.ticker in selected_tickers:
                rows_by_ticker[quote.ticker].append(quote)
        print(f"parsed {year}", flush=True)

    write_outputs(
        args.output_dir,
        selected,
        rows_by_ticker,
        archives,
        selection_year,
        args.start_year,
        args.end_year,
    )
    print(f"dataset written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
