from dataclasses import asdict
import json

import pandas as pd
import pytest

from scripts.evaluate_detectors import run_evaluation
from src.agent.schemas import Alarm


def test_comparison_json_csv_audits_and_stale_split_detection(tmp_path):
    index = pd.date_range("2018-01-01", periods=12, freq="h", name="timestamp")
    leakage = pd.DataFrame({"pipe1": [0] * 8 + [1, 1, 0, 0]}, index=index)
    sensors = pd.DataFrame({"pressure_n1": 30.0, "flow_p1": 10.0, "leak": leakage.gt(0).any(axis=1).astype(int)}, index=index)
    sensors.to_csv(tmp_path / "merged.csv")
    leakage.rename_axis("Timestamp").to_csv(tmp_path / "leakage.csv", sep=";", decimal=",")
    alarm = Alarm("a", index[9].to_pydatetime(), 1.0, "baseline", ["pressure_n1"])
    record = {**asdict(alarm), "timestamp": alarm.timestamp.isoformat()}
    (tmp_path / "baseline_alarms.json").write_text(json.dumps([record]))
    (tmp_path / "lightgbm_alarms.json").write_text("[]")
    manifest = {"train_end": index[5].isoformat(), "test_start": index[6].isoformat(), "test_end": index[-1].isoformat(), "test_rows": 6}
    manifest_path = tmp_path / "baseline_summary.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "data": {"raw_path": "merged.csv"}, "split": {"train_fraction": 0.5},
        "evaluation": {"leakage_path": "leakage.csv", "alarm_paths": {"baseline": "baseline_alarms.json", "lightgbm": "lightgbm_alarms.json"}, "output_dir": "results"},
    }
    comparison = run_evaluation(config, tmp_path)
    assert comparison["true_positives"].tolist() == [1, 0]
    assert comparison["false_negatives"].tolist() == [0, 1]
    payload = json.loads((tmp_path / "results/evaluation.json").read_text())
    assert payload["lightgbm"]["metrics"]["avg_detection_delay_hours"] is None
    assert (tmp_path / "results/model_1_events.csv").exists()
    assert "Detector Comparison" in (tmp_path / "results/comparison.md").read_text()
    first = (tmp_path / "results/evaluation.json").read_bytes()
    run_evaluation(config, tmp_path)
    assert first == (tmp_path / "results/evaluation.json").read_bytes()
    manifest["train_end"] = index[4].isoformat()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different split"):
        run_evaluation(config, tmp_path)
