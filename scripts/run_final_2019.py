from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.battledim import read_battledim_csv
from src.data.loader import load_sensor_dataset
from src.detection.baseline import BaselineConfig, ResidualBaselineDetector
from src.detection.alarms import create_persistent_alarms
from src.evaluation.metrics import evaluate_alarms, extract_leak_events
from src.utils.config import load_config


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def require_year(frame: pd.DataFrame, year: int) -> None:
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:55", freq="5min", name=frame.index.name)
    if not frame.index.equals(expected):
        raise ValueError(f"Expected complete, uninterrupted five-minute {year} observations.")


def freeze(protocol: Path) -> Path:
    config = load_config(protocol)
    output = ROOT / config["output_dir"]
    if (output / "frozen_manifest.json").exists():
        raise ValueError("A frozen model already exists; it will not be silently replaced.")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / config["train_path"]
    data = load_sensor_dataset(source)
    require_year(data.frame, 2018)
    (output / "protocol_snapshot.yaml").write_bytes(protocol.read_bytes())
    model = ResidualBaselineDetector(data.pressure_sensors, data.sensor_columns, BaselineConfig(**config["baseline"])).fit(data.frame[data.sensor_columns])
    joblib.dump(model, output / "baseline_model.joblib")
    code = [ROOT / "scripts/run_final_2019.py", ROOT / "src/detection/baseline.py", ROOT / "src/detection/alarms.py", ROOT / "src/evaluation/metrics.py", ROOT / "src/data/battledim.py", ROOT / "src/data/loader.py", ROOT / "src/utils/config.py"]
    manifest = {"protocol_id": config["protocol_id"], "protocol_sha256": checksum(protocol),
                "train_sha256": checksum(source), "model_sha256": checksum(output / "baseline_model.joblib"),
                "code_sha256": {str(path.relative_to(ROOT)): checksum(path) for path in code},
                "threshold": model.threshold_, "settings": config, "sensors": data.sensor_columns,
                "2019_loaded": False}
    (output / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Frozen 2018-only model and protocol: {output}", flush=True)
    return output


def evaluate(protocol: Path) -> dict:
    config = load_config(protocol)
    output = ROOT / config["output_dir"]
    manifest = json.loads((output / "frozen_manifest.json").read_text(encoding="utf-8"))
    if (output / "evaluation.json").exists():
        raise ValueError("Final test already completed. Existing outcomes cannot be overwritten.")
    checks = [(protocol, manifest["protocol_sha256"]), (ROOT / config["train_path"], manifest["train_sha256"]),
              (output / "baseline_model.joblib", manifest["model_sha256"])]
    checks += [(ROOT / path, sha) for path, sha in manifest["code_sha256"].items()]
    if any(checksum(path) != sha for path, sha in checks):
        raise ValueError("Frozen protocol, inputs, model, or implementation changed; final test refused.")
    raw = ROOT / config["raw_dir"]
    train = load_sensor_dataset(ROOT / config["train_path"]).frame[manifest["sensors"]]
    pressures = read_battledim_csv(raw / "2019_SCADA_Pressures.csv").add_prefix("pressure_")
    flows = read_battledim_csv(raw / "2019_SCADA_Flows.csv").add_prefix("flow_")
    require_year(pressures, 2019)
    require_year(flows, 2019)
    sensors = pd.concat([pressures, flows], axis=1)
    if set(sensors.columns) != set(manifest["sensors"]):
        raise ValueError("2019 sensor names differ from the frozen 2018 model.")
    model = joblib.load(output / "baseline_model.joblib")
    warm_start = sensors.index[0] - pd.Timedelta(days=config["warmup_days"])
    history = train.loc[train.index >= warm_start - pd.Timedelta(minutes=5 * (model.config.rolling_window - 1))]
    scores = model.score(pd.concat([history, sensors[manifest["sensors"]]]))
    notification_index = scores.aggregate_score.index[scores.aggregate_score.index >= warm_start]
    sequence = create_persistent_alarms(scores.aggregate_score.loc[notification_index], scores.threshold,
        model.config.persistence_hours, scores.sensor_scores.loc[notification_index], config["detector"],
        model.sample_interval_, model.config.recovery_hours, model.config.cooldown_hours)
    alarms = [alarm for alarm in sequence if pd.Timestamp(alarm.timestamp).year == 2019]
    pd.DataFrame({"score": scores.aggregate_score.loc[sensors.index], "threshold": scores.threshold}).to_csv(output / "scores.csv.gz", compression="gzip")
    (output / "alarms.json").write_text(json.dumps([{**asdict(a), "timestamp": a.timestamp.isoformat()} for a in alarms], indent=2), encoding="utf-8")
    prediction_manifest = {"labels_loaded": False, "scores_sha256": checksum(output / "scores.csv.gz"), "alarms_sha256": checksum(output / "alarms.json")}
    (output / "prediction_manifest.json").write_text(json.dumps(prediction_manifest, indent=2), encoding="utf-8")
    print("Sensor-only predictions saved. Loading final labels for evaluation now.", flush=True)
    leakage = read_battledim_csv(raw / "2019_Leakages.csv")
    require_year(leakage, 2019)
    result = evaluate_alarms(alarms, extract_leak_events(leakage), sensors.index, config["detector"], config["max_detection_delay_hours"])
    event_frame = pd.DataFrame(result.events)
    detected = event_frame.loc[event_frame.status == "detected", "delay_hours"].astype(float)
    spread = detected.std()
    payload = {"status": "held_out_year_same_simulated_network", "metrics": {**result.metrics,
               "detection_delay_std_hours": float(spread) if pd.notna(spread) else None},
               "events": result.events, "alarms": result.alarms, "frozen_manifest": manifest,
               "test_sha256": {name: checksum(raw / name) for name in ["2019_SCADA_Pressures.csv", "2019_SCADA_Flows.csv", "2019_Leakages.csv"]}}
    (output / "evaluation.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    event_frame.to_csv(output / "event_audit.csv", index=False)
    pd.DataFrame(result.alarms).to_csv(output / "alarm_audit.csv", index=False)
    rows = ["# Held-Out 2019 Test", "", "Same simulated network; no parameters fitted or calibrated on 2019.", "", "| Metric | Value |", "| --- | --- |"]
    rows += [f"| {key} | {value} |" for key, value in payload["metrics"].items() if not isinstance(value, (dict, list))]
    rows += ["", "One-to-one temporal matching is not localization. Unmatched notifications do not imply a leak-free network.", "Missed-event delay is null; reported delay includes detected events only. See event/notification audits and frozen/prediction manifests."]
    (output / "report.md").write_text("\n".join(rows), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freeze the 2018-only baseline, then evaluate 2019 once.")
    parser.add_argument("phase", choices=["freeze", "evaluate"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/final_2019_v1.yaml")
    args = parser.parse_args()
    (freeze if args.phase == "freeze" else evaluate)(args.protocol)
