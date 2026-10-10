from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.validation_report import write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render completed validation artifacts without retraining.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/results/validation_v1")
    args = parser.parse_args()
    write_validation_report(args.output, pd.read_csv(args.output / "summary.csv"), pd.read_csv(args.output / "event_audit.csv"))
    print(f"Saved {args.output / 'report.html'}")


if __name__ == "__main__":
    main()
