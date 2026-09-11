from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.strategies.catalog import NATIVE_SPECS
from b3_backtest.strategies.vendor_registry import discover_vendored_strategies

BATCH_DIR = SRC / "b3_backtest" / "strategies" / "catalog_batches"
INDEX_PATH = SRC / "b3_backtest" / "strategies" / "strategy_catalog_index.json"
MAX_BATCH_SIZE = 40


def slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "outros"


def main() -> int:
    native_names = {spec.name for spec in NATIVE_SPECS}
    discovered = discover_vendored_strategies()
    if not discovered:
        raise SystemExit("no vendored strategies discovered")

    remaining = [item for item in discovered if item.name not in native_names]
    all_names = native_names | {item.name for item in remaining}
    if len(all_names) != len(native_names) + len(remaining):
        raise SystemExit("duplicate strategy names after native de-duplication")

    # The pinned additional_strategies module alone defines 130 variants, of
    # which 40 are the original native benchmark. The full source catalog must
    # therefore add materially more than those 90 remaining variants.
    if len(remaining) <= 90:
        raise SystemExit(f"incomplete full catalog: only {len(remaining)} strategies beyond the original 40")

    by_family: dict[str, list[object]] = defaultdict(list)
    for item in remaining:
        by_family[item.family].append(item)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("*.json"):
        old.unlink()

    batches: list[dict[str, object]] = []
    for family in sorted(by_family, key=lambda value: (slug(value), value)):
        values = sorted(by_family[family], key=lambda item: item.name)
        family_slug = slug(family)
        for offset in range(0, len(values), MAX_BATCH_SIZE):
            chunk = values[offset : offset + MAX_BATCH_SIZE]
            batch_number = offset // MAX_BATCH_SIZE + 1
            batch_name = f"{family_slug}_{batch_number:03d}"
            strategies = [
                {
                    "name": item.name,
                    "family": item.family,
                    "description": item.description,
                    "source_module": item.source_module,
                }
                for item in chunk
            ]
            payload = {
                "batch": batch_name,
                "family": family,
                "max_batch_size": MAX_BATCH_SIZE,
                "strategy_count": len(strategies),
                "strategies": strategies,
            }
            path = BATCH_DIR / f"{batch_name}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            batches.append({"name": batch_name, "family": family, "count": len(strategies), "file": path.name})

    source_counts = Counter(item.source_module for item in remaining)
    family_counts = Counter(item.family for item in remaining)
    index = {
        "source_repository": "mycroft440/b3-strategy-lab",
        "source_commit": "8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d",
        "organization": "family-first, maximum 40 strategies per JSON batch",
        "native_strategy_count": len(native_names),
        "vendored_strategy_count": len(remaining),
        "total_strategy_count": len(all_names),
        "batch_count": len(batches),
        "families": dict(sorted(family_counts.items())),
        "source_modules": dict(sorted(source_counts.items())),
        "batches": batches,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(index, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
