from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3_backtest.strategies.catalog import NATIVE_SPECS, SOURCE_COMMIT, SOURCE_REPOSITORY  # noqa: E402
from b3_backtest.strategies.vendor_registry import discover_vendored_strategies, vendor_available  # noqa: E402

BATCH_SIZE = 40
STRATEGY_DIR = SRC / "b3_backtest" / "strategies"
BATCH_DIR = STRATEGY_DIR / "catalog_batches"
MANIFEST_PATH = STRATEGY_DIR / "catalog_manifest.json"


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.strip().lower()).strip("_")
    return normalized or "outros"


def main() -> int:
    if not vendor_available():
        raise SystemExit("vendored source package is missing; run sync_strategy_vendor.py first")

    discovered = discover_vendored_strategies()
    if not discovered:
        raise SystemExit("no strategies were discovered in the vendored source package")

    source_names = [item.name for item in discovered]
    if len(source_names) != len(set(source_names)):
        duplicates = sorted(name for name, count in Counter(source_names).items() if count > 1)
        raise SystemExit(f"source discovery returned duplicate strategy names: {duplicates[:20]}")

    native_names = {spec.name for spec in NATIVE_SPECS}
    overlapping = sorted(native_names.intersection(source_names))
    vendored = [item for item in discovered if item.name not in native_names]

    if BATCH_DIR.exists():
        for path in BATCH_DIR.glob("*.json"):
            path.unlink()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    by_family: dict[str, list[object]] = defaultdict(list)
    for item in vendored:
        by_family[item.family].append(item)

    batch_records: list[dict[str, object]] = []
    family_counts: dict[str, int] = {}
    source_module_counts: Counter[str] = Counter()

    for family in sorted(by_family, key=lambda value: (_slug(value), value)):
        items = sorted(by_family[family], key=lambda item: item.name)
        family_counts[family] = len(items)
        slug = _slug(family)
        for offset in range(0, len(items), BATCH_SIZE):
            chunk = items[offset : offset + BATCH_SIZE]
            index = offset // BATCH_SIZE + 1
            filename = f"{slug}_{index:03d}.json"
            strategies: list[dict[str, str]] = []
            for item in chunk:
                source_module_counts[item.source_module] += 1
                strategies.append(
                    {
                        "name": item.name,
                        "family": item.family,
                        "description": item.description,
                        "source_module": item.source_module,
                    }
                )
            payload = {
                "source_repository": SOURCE_REPOSITORY,
                "source_commit": SOURCE_COMMIT,
                "family": family,
                "batch": filename.removesuffix(".json"),
                "count": len(strategies),
                "strategies": strategies,
            }
            (BATCH_DIR / filename).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            batch_records.append(
                {
                    "file": filename,
                    "family": family,
                    "count": len(strategies),
                    "first_strategy": strategies[0]["name"],
                    "last_strategy": strategies[-1]["name"],
                }
            )

    batch_names = [item.name for item in vendored]
    if len(batch_names) != len(set(batch_names)):
        raise SystemExit("generated vendored strategy list contains duplicate names")
    if any(record["count"] > BATCH_SIZE for record in batch_records):
        raise SystemExit("a generated strategy batch exceeds the 40-strategy limit")

    manifest = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "batch_size_limit": BATCH_SIZE,
        "native_strategy_count": len(NATIVE_SPECS),
        "source_unique_strategy_count": len(discovered),
        "source_overlap_with_native_count": len(overlapping),
        "source_overlap_with_native": overlapping,
        "vendored_strategy_count": len(vendored),
        "total_catalog_strategy_count": len(NATIVE_SPECS) + len(vendored),
        "batch_count": len(batch_records),
        "families": dict(sorted(family_counts.items(), key=lambda item: (_slug(item[0]), item[0]))),
        "source_modules": dict(sorted(source_module_counts.items())),
        "batches": batch_records,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
