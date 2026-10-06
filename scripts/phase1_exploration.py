from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_sensor_dataset
from src.data.preprocessing import chronological_split, report_missing_values
from src.utils.config import load_config


def leak_intervals(index, labels):
    intervals = []
    in_leak = False
    start = None
    previous_timestamp = None
    for timestamp, label in zip(index, labels):
        if bool(label) and not in_leak:
            start = timestamp
            in_leak = True
        if not bool(label) and in_leak:
            intervals.append((start, previous_timestamp))
            in_leak = False
        previous_timestamp = timestamp
    if in_leak:
        intervals.append((start, previous_timestamp))
    return intervals


def plot_sensor_group(frame, sensors, label_column, title, output_path):
    fig, axis = plt.subplots(figsize=(12, 5))
    for sensor in sensors:
        axis.plot(frame.index, frame[sensor], label=sensor, linewidth=1)

    for start, end in leak_intervals(frame.index, frame[label_column]):
        axis.axvspan(start, end, color="red", alpha=0.16)

    axis.set_title(title)
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Sensor value")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 1 data exploration.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config.get("data", {})
    split_config = config.get("split", {})
    plot_config = config.get("plots", {})

    dataset = load_sensor_dataset(
        raw_path=data_config.get("raw_path", "data/raw/battledim.csv"),
        timestamp_column=data_config.get("timestamp_column"),
        leak_label_column=data_config.get("leak_label_column"),
        pressure_sensors=data_config.get("pressure_sensors") or [],
        flow_sensors=data_config.get("flow_sensors") or [],
        duplicate_strategy=data_config.get("duplicate_strategy", "mean"),
    )

    missing_report = report_missing_values(dataset.frame)
    split = chronological_split(
        dataset.frame,
        train_fraction=float(split_config.get("train_fraction", 0.7)),
    )

    output_dir = Path(plot_config.get("output_dir", "artifacts/plots"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_pressure = int(plot_config.get("max_pressure_sensors", 3))
    max_flow = int(plot_config.get("max_flow_sensors", 3))
    pressure_plot = output_dir / "phase1_pressure_signals.png"
    flow_plot = output_dir / "phase1_flow_signals.png"

    plot_sensor_group(
        dataset.frame,
        dataset.pressure_sensors[:max_pressure],
        dataset.leak_label_column,
        "Representative pressure signals with leak intervals",
        pressure_plot,
    )
    plot_sensor_group(
        dataset.frame,
        dataset.flow_sensors[:max_flow],
        dataset.leak_label_column,
        "Representative flow signals with leak intervals",
        flow_plot,
    )

    summary = {
        "rows": len(dataset.frame),
        "columns": dataset.frame.columns.tolist(),
        "start": dataset.frame.index.min().isoformat(),
        "end": dataset.frame.index.max().isoformat(),
        "pressure_sensors": dataset.pressure_sensors,
        "flow_sensors": dataset.flow_sensors,
        "leak_label_column": dataset.leak_label_column,
        "missing_values": {
            "total": missing_report.total_missing,
            "by_column": missing_report.missing_by_column,
        },
        "split": {
            "train_rows": len(split.train),
            "test_rows": len(split.test),
            "train_end": split.train.index.max().isoformat(),
            "test_start": split.test.index.min().isoformat(),
        },
        "plots": [str(pressure_plot), str(flow_plot)],
    }

    results_path = Path("artifacts/results/phase1_summary.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
