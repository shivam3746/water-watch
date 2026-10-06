from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_detectors import run_evaluation
from scripts.phase2_baseline import run as run_baseline
from scripts.train_lightgbm import run as run_classifier
from src.data.loader import load_sensor_dataset
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-train both detectors and evaluate them on the same held-out period.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    root = config_path.parent
    config = load_config(config_path)

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    try:
        data = dict(config["data"])
        dataset = load_sensor_dataset(resolve(data.pop("raw_path")), **data)
        baseline_dir = resolve(config.get("baseline", {}).get("output_dir", "artifacts/baseline"))
        print("Training baseline...", flush=True)
        run_baseline(config, dataset, baseline_dir)
        print("Training LightGBM and selecting threshold on chronological validation...", flush=True)
        classifier_summary = run_classifier(config, root)
        classifier_dir = Path(classifier_summary["model_path"]).parent
        print("Evaluating saved alarms...", flush=True)
        comparison = run_evaluation(config, root, {
            "baseline": baseline_dir / "baseline_alarms.json",
            "lightgbm": classifier_dir / "lightgbm_alarms.json",
        })
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    columns = ["model", "precision", "recall", "f1", "avg_detection_delay_hours", "false_alarms_per_week", "eligible_leak_events", "duplicate_alarms"]
    print(comparison[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
