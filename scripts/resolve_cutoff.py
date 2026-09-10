from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve o último cutoff verificável já presente no b3-strategy-lab."
    )
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--requested-end", default="")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    upstream = args.upstream.resolve()
    sys.path.insert(0, str(upstream))

    from b3_strategy_lab.cotahist import load_verified_candles  # noqa: E402

    universe_path = upstream / "data/universes/fixed_40_2018.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    warmup_start = str(universe["warmup_start"])

    ends: list[str] = []
    for ticker in universe["tickers"]:
        candles, _manifest = load_verified_candles(
            str(ticker),
            "1d",
            require_verified_splits_from=warmup_start,
            data_dir=upstream / "data/candles",
            actions_dir=upstream / "data/corporate_actions",
            manifests_dir=upstream / "data/manifests",
            split_evidence_path=upstream / "data/corporate_actions/split_evidence.json",
        )
        if not candles:
            raise SystemExit(f"{ticker}: série verificada vazia")
        ends.append(candles[-1].date)

    certified_end = min(ends)
    requested = args.requested_end.strip()
    if requested:
        requested_date = date.fromisoformat(requested)
        if requested_date.isoformat() > certified_end:
            raise SystemExit(
                f"--end={requested} excede o cutoff verificável do upstream ({certified_end}). "
                "Atualize/certifique primeiro o b3-strategy-lab."
            )
        resolved_end = requested_date.isoformat()
    else:
        resolved_end = certified_end

    if date.fromisoformat(resolved_end) >= date.today():
        raise SystemExit("cutoff precisa ser anterior ao dia corrente")

    print(f"certified_end={certified_end} resolved_end={resolved_end}")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"certified_end={certified_end}\n")
            output.write(f"resolved_end={resolved_end}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
