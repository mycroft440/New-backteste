from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


NUMERIC_FIELDS = (
    "initial_equity",
    "final_equity",
    "total_return",
    "cagr",
    "average_annual_return",
    "max_drawdown",
    "annual_volatility",
    "sharpe",
    "turnover",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Confere que a matriz contém exatamente o produto cartesiano esperado."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    strategies = [str(row["name"]) for row in catalog["strategies"]]
    management = [str(row["name"]) for row in catalog["management_configs"]]
    expected = {(strategy, config) for strategy in strategies for config in management}

    seen: set[tuple[str, str]] = set()
    duplicate_pairs: list[tuple[str, str]] = []
    unexpected_pairs: list[tuple[str, str]] = []
    non_finite: list[dict[str, str]] = []
    row_count = 0

    opener = gzip.open if args.results.suffix == ".gz" else open
    with opener(args.results, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"trading_strategy", "management_strategy", *NUMERIC_FIELDS}
        missing_columns = sorted(required - set(reader.fieldnames or []))
        if missing_columns:
            raise SystemExit("colunas ausentes: " + ", ".join(missing_columns))

        for row in reader:
            row_count += 1
            pair = (str(row["trading_strategy"]), str(row["management_strategy"]))
            if pair in seen and len(duplicate_pairs) < 20:
                duplicate_pairs.append(pair)
            seen.add(pair)
            if pair not in expected and len(unexpected_pairs) < 20:
                unexpected_pairs.append(pair)

            for field in NUMERIC_FIELDS:
                try:
                    value = float(row[field])
                except (TypeError, ValueError):
                    value = math.nan
                if not math.isfinite(value):
                    if len(non_finite) < 20:
                        non_finite.append(
                            {
                                "strategy": pair[0],
                                "management": pair[1],
                                "field": field,
                                "value": str(row.get(field)),
                            }
                        )
                    break

    missing_pairs = sorted(expected - seen)
    expected_count = int(catalog["combination_count"])
    failures: list[str] = []
    if expected_count != len(expected):
        failures.append(
            f"manifest combination_count={expected_count}, produto real={len(expected)}"
        )
    if row_count != expected_count:
        failures.append(f"linhas={row_count}, esperado={expected_count}")
    if duplicate_pairs:
        failures.append(f"pares duplicados (amostra): {duplicate_pairs[:10]}")
    if unexpected_pairs:
        failures.append(f"pares inesperados (amostra): {unexpected_pairs[:10]}")
    if missing_pairs:
        failures.append(f"pares ausentes (amostra): {missing_pairs[:10]}")
    if non_finite:
        failures.append(f"métricas não finitas (amostra): {non_finite[:10]}")

    report = {
        "schema_version": 1,
        "upstream_sha": catalog["upstream_sha"],
        "strategy_count": len(strategies),
        "management_count": len(management),
        "expected_combinations": expected_count,
        "rows": row_count,
        "unique_pairs": len(seen),
        "complete_cartesian_product": not missing_pairs and not unexpected_pairs,
        "no_duplicates": not duplicate_pairs,
        "all_numeric_metrics_finite": not non_finite,
        "passed": not failures,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit("matriz exaustiva inválida: " + " | ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
