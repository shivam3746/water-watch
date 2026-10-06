from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import SensorDataset, load_sensor_dataset
from src.data.preprocessing import chronological_split
from src.detection.baseline import BaselineConfig, ResidualBaselineDetector
from src.utils.config import load_config


def demo_dataset() -> SensorDataset:
    """Seeded five-minute fixture for checking the pipeline, not accuracy claims."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2026-01-01", periods=288 * 7, freq="5min", name="timestamp")
    cycle = np.sin(np.arange(len(index)) * 2 * np.pi / 288)
    frame = pd.DataFrame({
        "P1": 40 + 2 * cycle + rng.normal(0, 0.12, len(index)),
        "P2": 38 + 1.5 * cycle + rng.normal(0, 0.12, len(index)),
        "F1": 10 + cycle + rng.normal(0, 0.05, len(index)),
        "leak": 0,
    }, index=index)
    leak = (index >= index[1550]) & (index < index[1750])
    frame.loc[leak, "P1"] -= 5
    frame.loc[leak, "leak"] = 1
    return SensorDataset(frame, ["P1", "P2"], ["F1"], "leak")


def run(config: dict, dataset: SensorDataset, output_dir: Path, demo: bool = False) -> dict:
    settings = config.get("baseline", {})
    model_config = BaselineConfig(**{
        key: settings[key] for key in BaselineConfig.__dataclass_fields__ if key in settings
    })
    split = chronological_split(dataset.frame, float(config.get("split", {}).get("train_fraction", 0.7)))
    # Labels are used only for the output plot and never passed to the detector.
    train = split.train[dataset.sensor_columns]
    test = split.test[dataset.sensor_columns]
    detector = ResidualBaselineDetector(dataset.pressure_sensors, dataset.sensor_columns, model_config).fit(train)
    scores = detector.score(test)
    alarms = detector.create_alarms(scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline_model.joblib"
    joblib.dump(detector, model_path)
    scores.residuals.to_csv(output_dir / "baseline_residuals.csv")
    scores.sensor_scores.to_csv(output_dir / "baseline_sensor_scores.csv")
    pd.DataFrame({"anomaly_score": scores.aggregate_score, "threshold": scores.threshold}).to_csv(output_dir / "baseline_scores.csv")
    alarm_records = [{**asdict(alarm), "timestamp": alarm.timestamp.isoformat()} for alarm in alarms]
    (output_dir / "baseline_alarms.json").write_text(json.dumps(alarm_records, indent=2, allow_nan=False), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(test.index, scores.aggregate_score, label="Rolling anomaly score", linewidth=1)
    axis.axhline(scores.threshold, color="black", linestyle="--", label="Training-derived threshold" if model_config.threshold is None else "Configured threshold")
    labels = split.test[dataset.leak_label_column].astype(bool)
    starts = labels & ~labels.shift(1, fill_value=False)
    step = detector.sample_interval_
    for start in labels.index[starts]:
        subsequent = labels.loc[start:]
        ends = subsequent.index[~subsequent]
        end = ends[0] if len(ends) else labels.index[-1] + step
        axis.axvspan(start, end, color="red", alpha=0.15)
    axis.plot([], [], color="red", alpha=0.3, linewidth=8, label="True leak intervals")
    if alarms:
        axis.scatter([a.timestamp for a in alarms], [a.score for a in alarms], marker="^", color="darkgreen", s=65, label="Baseline alarms", zorder=3)
    axis.set(title="Baseline alarms on the held-out test period", xlabel="Timestamp", ylabel="Mean normalized absolute residual")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    plot_path = output_dir / "baseline_alarms.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    summary = {
        "demo": demo,
        "train_rows": len(train), "test_rows": len(test),
        "train_start": train.index.min().isoformat(), "train_end": train.index.max().isoformat(),
        "test_start": test.index.min().isoformat(), "test_end": test.index.max().isoformat(),
        "pressure_sensors": dataset.pressure_sensors,
        "feature_sensors": dataset.sensor_columns,
        "baseline_config": asdict(model_config),
        "threshold": scores.threshold,
        "threshold_source": "configured" if model_config.threshold is not None else "training_score_quantile",
        "sample_interval_minutes": detector.sample_interval_.total_seconds() / 60,
        "alarm_count": len(alarms),
        "training_labels_used": False,
        "model_path": str(model_path.resolve()), "plot_path": str(plot_path.resolve()),
    }
    (output_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Phase 2 baseline and export test scores, alarms, and plot.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--demo", action="store_true", help="Use seeded synthetic data instead of the configured CSV.")
    parser.add_argument("--output-dir", help="Override baseline.output_dir.")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_dir = Path(args.output_dir or config.get("baseline", {}).get("output_dir", "artifacts/baseline"))
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    if args.demo:
        dataset = demo_dataset()
        if args.output_dir is None:
            output_dir = output_dir / "demo"
    else:
        data_config = dict(config.get("data", {}))
        raw_path = Path(data_config.pop("raw_path", "data/raw/battledim.csv"))
        if not raw_path.is_absolute():
            raw_path = config_path.parent / raw_path
        dataset = load_sensor_dataset(raw_path, **data_config)
    try:
        summary = run(config, dataset, output_dir, demo=args.demo)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
