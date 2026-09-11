from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

SOURCE_REPOSITORY = "mycroft440/b3-strategy-lab"
SOURCE_COMMIT = "8cb3a9e906dfae69e74d26d8cd3a9c76c380d55d"
ENTRY_MODULES = (
    "strategies.py",
    "additional_strategies.py",
    "extended_strategies.py",
    "indicator_strategies.py",
    "researched_strategies.py",
    "smi_ergodic_strategies.py",
    "trend_strategies.py",
)


def relative_dependencies(path: Path, package_root: Path) -> set[Path]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[Path] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level <= 0:
            continue
        current_rel = path.relative_to(package_root)
        parent_parts = list(current_rel.parent.parts)
        climb = node.level - 1
        base_parts = parent_parts[: max(0, len(parent_parts) - climb)]
        if node.module:
            module_parts = node.module.split(".")
            candidate = package_root.joinpath(*base_parts, *module_parts).with_suffix(".py")
            package_candidate = package_root.joinpath(*base_parts, *module_parts, "__init__.py")
            if candidate.exists():
                found.add(candidate)
            elif package_candidate.exists():
                found.add(package_candidate)
        else:
            for alias in node.names:
                candidate = package_root.joinpath(*base_parts, alias.name).with_suffix(".py")
                package_candidate = package_root.joinpath(*base_parts, alias.name, "__init__.py")
                if candidate.exists():
                    found.add(candidate)
                elif package_candidate.exists():
                    found.add(package_candidate)
    return found


def collect_files(source_root: Path) -> list[Path]:
    pending = [source_root / name for name in ENTRY_MODULES]
    collected: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in collected:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        collected.add(path)
        if path.suffix == ".py":
            pending.extend(relative_dependencies(path, source_root) - collected)
    return sorted(collected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor the pinned B3 Strategy Lab strategy dependency closure.")
    parser.add_argument("source_root", type=Path)
    parser.add_argument("target_root", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    if not all((source_root / name).exists() for name in ENTRY_MODULES):
        raise SystemExit("source checkout is missing one or more strategy entry modules")

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target_root.parent.mkdir(parents=True, exist_ok=True)

    # Do not copy the source package __init__.py: it imports unrelated application
    # modules. The vendored package intentionally exposes only the strategy subset.
    (target_root.parent / "__init__.py").write_text(
        '"""Vendored strategy packages used for reproducible signal execution."""\n',
        encoding="utf-8",
    )
    (target_root / "__init__.py").write_text(
        '"""Pinned B3 Strategy Lab strategy implementation subset."""\n',
        encoding="utf-8",
    )

    copied: list[str] = []
    for source in collect_files(source_root):
        relative = source.relative_to(source_root)
        if relative.as_posix() == "__init__.py":
            continue
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(relative.as_posix())

    metadata = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "entry_modules": list(ENTRY_MODULES),
        "files": copied,
    }
    (target_root / "_SOURCE.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Vendored {len(copied)} Python files from {SOURCE_REPOSITORY}@{SOURCE_COMMIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
