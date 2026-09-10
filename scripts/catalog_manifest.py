from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def _git_sha(upstream: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _source_fingerprint(upstream: Path) -> tuple[str, list[dict[str, str]]]:
    files = list((upstream / "b3_strategy_lab").rglob("*.py"))
    for relative in (
        "scripts/research_portfolio_allocation.py",
        "scripts/research_portfolio_allocation_core.py",
        "scripts/backtest_strategy_management_combinations.py",
        "scripts/merge_matrix_shards.py",
        "scripts/audit_matrix_results.py",
        "scripts/validate_matrix_top_realistic.py",
        "scripts/walk_forward_certified.py",
    ):
        path = upstream / relative
        if path.exists():
            files.append(path)

    records: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(upstream).as_posix()
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        records.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(payload)
        aggregate.update(b"\0")
    return aggregate.hexdigest(), records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve e registra o catálogo completo do b3-strategy-lab."
    )
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    if args.shards <= 0:
        parser.error("--shards precisa ser maior que zero")
    if not (upstream / "b3_strategy_lab").is_dir():
        parser.error(f"upstream inválido: {upstream}")

    sys.path.insert(0, str(upstream))
    from b3_strategy_lab.strategies import (  # noqa: E402
        available_strategies,
        portfolio_strategies,
        strategy_info,
        strategy_parameters,
    )
    from scripts.research_portfolio_allocation import _configs  # noqa: E402

    # O objetivo é realmente testar TODO o catálogo disponível. O executor de
    # estratégia+gerenciamento do upstream aceita portfolio_strategies(); portanto,
    # qualquer divergência entre os dois catálogos deve interromper a rodada em vez
    # de excluir silenciosamente uma estratégia recém-adicionada.
    available = list(available_strategies())
    engine_supported = list(portfolio_strategies())
    available_set = set(available)
    supported_set = set(engine_supported)
    if available_set != supported_set:
        missing_from_engine = sorted(available_set - supported_set)
        unexpected_in_engine = sorted(supported_set - available_set)
        raise SystemExit(
            "catálogo completo diverge das estratégias aceitas pelo motor de portfólio; "
            f"ausentes_no_motor={missing_from_engine!r} "
            f"extras_no_motor={unexpected_in_engine!r}"
        )

    strategies = available
    configs = list(_configs("adjusted", "all"))

    if not strategies:
        raise SystemExit("catálogo upstream não contém estratégias")
    if len(strategies) != len(set(strategies)):
        raise SystemExit("catálogo upstream contém estratégias duplicadas")
    config_names = [config.name for config in configs]
    if not configs or len(config_names) != len(set(config_names)):
        raise SystemExit("catálogo upstream de gerenciamento está vazio ou duplicado")

    strategy_rows = []
    for name in strategies:
        info = strategy_info(name)
        strategy_rows.append(
            {
                "name": name,
                "family": info.family,
                "description": info.description,
                "parameters": strategy_parameters(name),
            }
        )

    shard_count = min(args.shards, len(strategies))
    shards = []
    assigned: list[str] = []
    for index in range(shard_count):
        subset = strategies[index::shard_count]
        if not subset:
            continue
        assigned.extend(subset)
        shards.append(
            {
                "index": index,
                "strategy_count": len(subset),
                "strategies": " ".join(subset),
            }
        )
    if sorted(assigned) != sorted(strategies) or len(assigned) != len(set(assigned)):
        raise SystemExit("sharding não cobre o catálogo exatamente uma vez")

    source_digest, source_files = _source_fingerprint(upstream)
    upstream_sha = _git_sha(upstream)
    manifest = {
        "schema_version": 1,
        "upstream_repository": "mycroft440/b3-strategy-lab",
        "upstream_sha": upstream_sha,
        "source_fingerprint_sha256": source_digest,
        "source_files": source_files,
        "signal_mode": "adjusted",
        "config_set": "all",
        "full_available_catalog_equals_portfolio_engine_catalog": True,
        "strategy_count": len(strategies),
        "management_count": len(configs),
        "combination_count": len(strategies) * len(configs),
        "shard_count": len(shards),
        "strategies": strategy_rows,
        "management_configs": [asdict(config) for config in configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    matrix = json.dumps({"include": shards}, separators=(",", ":"))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"matrix={matrix}\n")
            output.write(f"upstream_sha={upstream_sha}\n")
            output.write(f"strategy_count={len(strategies)}\n")
            output.write(f"management_count={len(configs)}\n")
            output.write(f"combination_count={len(strategies) * len(configs)}\n")
            output.write(f"shard_count={len(shards)}\n")

    print(
        f"upstream={upstream_sha} strategies={len(strategies)} "
        f"management={len(configs)} combinations={len(strategies) * len(configs)} "
        f"shards={len(shards)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
