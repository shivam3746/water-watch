from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.schemas import Alarm
from src.data.battledim import read_battledim_csv
from src.data.loader import load_sensor_dataset
from src.data.preprocessing import chronological_split
from src.evaluation.metrics import evaluate_alarms, extract_leak_events
from src.utils.config import load_config


def load_alarms(path: Path) -> list[Alarm]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [Alarm(**{**record, "timestamp": datetime.fromisoformat(record["timestamp"])}) for record in records]


def run_evaluation(config: dict, root: Path, alarm_paths: dict[str, Path] | None = None) -> pd.DataFrame:
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    settings = config.get("evaluation", {})
    data_settings = dict(config.get("data", {}))
    dataset = load_sensor_dataset(resolve(data_settings.pop("raw_path")), **data_settings)
    split = chronological_split(dataset.frame, float(config.get("split", {}).get("train_fraction", 0.7)))
    leakage = read_battledim_csv(resolve(settings.get("leakage_path", "data/raw/2018_Leakages.csv")))
    if not leakage.index.equals(dataset.frame.index):
        raise ValueError("Ground truth timestamps must exactly match the merged sensor dataset.")
    if not leakage.gt(0).any(axis=1).astype(int).equals(dataset.frame[dataset.leak_label_column]):
        raise ValueError("Merged labels disagree with per-pipe ground truth. Re-run data preparation.")
    events = extract_leak_events(leakage)
    paths = alarm_paths if alarm_paths is not None else {
        name: resolve(path) for name, path in settings.get("alarm_paths", {
            "baseline": str(Path(config.get("baseline", {}).get("output_dir", "artifacts/baseline")) / "baseline_alarms.json")
        }).items()
    }
    if not paths:
        raise ValueError("Configure at least one model alarm file.")
    results = {}
    manifests = {}
    for name, path in paths.items():
        metadata_path = path.with_name(path.stem.removesuffix("_alarms") + "_summary.json")
        if metadata_path.exists():
            manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                "train_end": split.train.index[-1].isoformat(),
                "test_start": split.test.index[0].isoformat(),
                "test_end": split.test.index[-1].isoformat(),
                "test_rows": len(split.test),
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Saved {name} alarms came from a different split. Re-run scripts/run_comparison.py.")
            manifests[name] = manifest
        result = evaluate_alarms(load_alarms(path), events, split.test.index, name, settings.get("max_detection_delay_hours"))
        results[name] = result
    output_dir = resolve(settings.get("output_dir", "artifacts/results/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame([result.metrics for result in results.values()])
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    payload = {name: {"metrics": result.metrics, "events": result.events, "alarms": result.alarms, "run_manifest": manifests.get(name)} for name, result in results.items()}
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for name, result in results.items():
        # A filename index avoids using untrusted model names as paths.
        number = list(results).index(name) + 1
        pd.DataFrame(result.events, columns=["event_id", "pipe", "start", "end", "status", "left_censored", "right_censored_at_test_end", "alarm_id", "delay_hours"]).to_csv(output_dir / f"model_{number}_events.csv", index=False)
        pd.DataFrame(result.alarms, columns=["alarm_id", "timestamp", "status", "event_id", "candidate_event_ids", "ambiguous_match", "delay_hours"]).to_csv(output_dir / f"model_{number}_alarms.csv", index=False)
    report = [
        "# Detector Comparison", "",
        "Event metrics measure new pipe leak onsets in the held-out period. Matching is temporal, not physical pipe localization.", "",
        "| Model | Precision | Recall | F1 | Mean Delay (Hours) | False Alerts/Week | Detected / Eligible Events |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        row = result.metrics
        delay = f'{row["avg_detection_delay_hours"]:.2f}' if row["avg_detection_delay_hours"] is not None else "N/A"
        report.append(f'| {name} | {row["precision"]:.3f} | {row["recall"]:.3f} | {row["f1"]:.3f} | {delay} | {row["false_alarms_per_week"]:.3f} | {row["true_positives"]} / {row["eligible_leak_events"]} |')
    first = next(iter(results.values())).metrics
    report += [
        "", f'Test interval: {first["test_start"]} to {first["test_end_exclusive"]} (exclusive).',
        f'Maximum detection delay: {first["max_detection_delay_hours"]} hours (null means while the event is active).',
        "", "## Matching Rules", "",
        "- An event is a contiguous run of positive leakage on one pipe, with an exclusive end at the first zero sample.",
        "- Each alarm and event can be matched at most once. Chronological alarms select the earliest-ending eligible active event, with deterministic ties. Ambiguous temporal overlaps are recorded.",
        "- Events active before the test split or source beginning are excluded from new-onset recall and delay; their records remain in the event audit.",
        "- An alarm before an event onset cannot match that event. Repeats, alarms after the detection window, and alarms attributable only to carry-in events count as false alerts for the new-event task.",
        "- Eligible events without a match are missed. Their delay is null, never zero. Mean delay includes detected events only.",
        "- Events still active at the test end remain eligible if they started during the test period; their shorter observation horizon is recorded as censored.",
        "- False alerts per week uses the complete regular held-out observation interval, including the final sample duration. Zero-denominator precision, recall, and F1 are defined as zero.",
        "", "## Interpretation", "",
        f'The held-out period contains {first["eligible_leak_events"]} new events and {first["excluded_carry_in_events"]} excluded carry-in events. Small event counts make these figures exploratory. An unmatched alert is not proof that the network is leak-free. Detection credit is based on timing and does not prove the alarm identified the correct pipe.',
        "",
    ]
    for name, manifest in manifests.items():
        if manifest.get("validation_status") == "no_events_detected":
            report.append(f'{name}: no validation event was detected at any configured threshold. Calibration did not establish useful detection; retain this model as an experimental result, not a replacement for the baseline.')
    (output_dir / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved detector alarms against individual BattLeDIM leak events.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    config_path = args.config.resolve()
    try:
        comparison = run_evaluation(load_config(config_path), config_path.parent)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    columns = ["model", "precision", "recall", "f1", "avg_detection_delay_hours", "false_alarms_per_week", "eligible_leak_events", "duplicate_alarms"]
    print(comparison[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
