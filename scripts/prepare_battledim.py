from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.battledim import prepare_battledim


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge official BattLeDIM pressure, flow, and leakage CSV files.")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/battledim.csv")
    args = parser.parse_args()
    try:
        frame, summary = prepare_battledim(args.raw_dir, args.year)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output)
    summary["output_path"] = str(args.output.resolve())
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
