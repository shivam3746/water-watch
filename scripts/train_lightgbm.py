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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.battledim import read_battledim_csv
from src.data.loader import load_sensor_dataset
from src.data.preprocessing import chronological_split
from src.detection.classifier import ClassifierConfig, LeakClassifier
from src.detection.features import onset_window_labels
from src.evaluation.metrics import extract_leak_events
from src.utils.config import load_config


def run(config: dict, root: Path) -> dict:
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    settings = config.get("classifier", {})
    classifier_config = ClassifierConfig(**{key: settings[key] for key in ClassifierConfig.__dataclass_fields__ if key in settings})
    data_config = dict(config["data"])
    dataset = load_sensor_dataset(resolve(data_config.pop("raw_path")), **data_config)
    split = chronological_split(dataset.frame, float(config.get("split", {}).get("train_fraction", 0.7)))
    leakage = read_battledim_csv(resolve(config.get("evaluation", {}).get("leakage_path", "data/raw/2018_Leakages.csv")))
    if not leakage.index.equals(dataset.frame.index):
        raise ValueError("Leakage timestamps must match the merged sensor dataset.")
    if not leakage.gt(0).any(axis=1).astype(int).equals(dataset.frame[dataset.leak_label_column]):
        raise ValueError("Merged labels disagree with per-pipe ground truth. Re-run data preparation.")
    training_leakage = leakage.loc[split.train.index]
    labels = onset_window_labels(training_leakage, classifier_config.onset_window_hours)
    train = split.train[dataset.sensor_columns]
    test = split.test[dataset.sensor_columns]
    detector = LeakClassifier(dataset.pressure_sensors, dataset.sensor_columns, classifier_config, float(config.get("baseline", {}).get("alpha", 1))).fit(train, labels, extract_leak_events(training_leakage))
    history = train.tail(classifier_config.rolling_window)
    scores = detector.score(test, history=history)
    alarms = detector.create_alarms(scores)
    output_dir = resolve(settings.get("output_dir", "artifacts/lightgbm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lightgbm_model.joblib"
    joblib.dump(detector, model_path)
    detector.model.booster_.save_model(str(output_dir / "lightgbm_booster.txt"))
    history.to_csv(output_dir / "feature_history.csv")
    pd.DataFrame({"probability": scores.probability, "threshold": scores.threshold}).to_csv(output_dir / "lightgbm_scores.csv")
    records = [{**asdict(alarm), "timestamp": alarm.timestamp.isoformat()} for alarm in alarms]
    (output_dir / "lightgbm_alarms.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(detector.validation_results_).to_csv(output_dir / "threshold_validation.csv", index=False)
    pd.DataFrame({"feature": scores.features.columns, "gain": detector.model.booster_.feature_importance(importance_type="gain")}).sort_values("gain", ascending=False).to_csv(output_dir / "feature_importance.csv", index=False)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(test.index, scores.probability, linewidth=1, label="New-leak window probability (uncalibrated)")
    axis.axhline(scores.threshold, color="black", linestyle="--", label="Validation-selected threshold")
    first = True
    for event in extract_leak_events(leakage):
        if test.index[0] <= event.start <= test.index[-1] and not event.left_censored:
            axis.axvspan(event.start, min(event.end, event.start + pd.Timedelta(hours=classifier_config.onset_window_hours)), color="red", alpha=0.2, label="New-leak target windows" if first else None)
            first = False
    if alarms:
        axis.scatter([a.timestamp for a in alarms], [a.score for a in alarms], color="darkgreen", marker="^", s=65, label="LightGBM alarms", zorder=3)
    axis.set(title="LightGBM on the held-out test period", xlabel="Timestamp", ylabel="Probability", ylim=(-0.03, 1.03))
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    plot_path = output_dir / "lightgbm_alarms.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    summary = {
        "train_start": train.index[0].isoformat(), "train_end": train.index[-1].isoformat(),
        "test_start": test.index[0].isoformat(), "test_end": test.index[-1].isoformat(),
        "train_rows": len(train), "test_rows": len(test),
        "partitions": detector.partitions_, "classifier_config": asdict(classifier_config),
        "threshold": detector.threshold_, "threshold_source": "configured" if classifier_config.threshold is not None else "chronological_validation_event_f1",
        "validation_status": "events_detected" if any(row["true_positives"] for row in detector.validation_results_) else "no_events_detected",
        "target": "new pipe leak within onset_window_hours while still active; not general leak presence",
        "feature_count": scores.features.shape[1], "alarm_count": len(alarms),
        "affected_sensor_rule": "largest baseline sensor residual scores; SHAP attribution is a later phase",
        "test_labels_used_for_training_or_calibration": False,
        "model_path": str(model_path.resolve()), "plot_path": str(plot_path.resolve()),
    }
    (output_dir / "lightgbm_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and save the Phase 3 LightGBM detector using chronological training/validation data.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    try:
        summary = run(load_config(config_path), config_path.parent)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
