import json

import numpy as np
import pandas as pd

from scripts.compare_alarm_policies import run as compare_policies
from scripts.phase2_baseline import run as run_baseline
from src.data.loader import SensorDataset


def test_frozen_score_comparison_preserves_original_artifacts(tmp_path):
    rng = np.random.default_rng(7)
    index = pd.date_range("2018-01-01", periods=72, freq="h", name="timestamp")
    frame = pd.DataFrame({"pressure_n1": 30 + rng.normal(0, 0.1, len(index)), "pressure_n4": 35 + rng.normal(0, 0.1, len(index)), "flow_p1": 10 + rng.normal(0, 0.1, len(index)), "leak": 0}, index=index)
    frame.loc[index[40:60], "leak"] = 1
    frame.loc[index[40:48], "pressure_n1"] -= 5
    frame.loc[index[49:60], "pressure_n1"] -= 5
    frame.to_csv(tmp_path / "merged.csv")
    leakage = pd.DataFrame({"p1": frame["leak"]}, index=index)
    leakage.rename_axis("Timestamp").to_csv(tmp_path / "leakage.csv", sep=";", decimal=",")
    dataset = SensorDataset(frame, ["pressure_n1", "pressure_n4"], ["flow_p1"], "leak")
    config = {
        "data": {"raw_path": "merged.csv"}, "split": {"train_fraction": 0.5},
        "baseline": {"rolling_window": 1, "threshold": 3, "persistence_hours": 1, "output_dir": "original"},
        "evaluation": {"leakage_path": "leakage.csv", "max_detection_delay_hours": 24},
    }
    run_baseline(config, dataset, tmp_path / "original")
    original = {path.name: path.read_bytes() for path in (tmp_path / "original").iterdir()}
    config["baseline"].update(recovery_hours=24, cooldown_hours=24)
    comparison = compare_policies(config, tmp_path)
    assert comparison["alarm_count"].tolist() == [2, 1]
    assert comparison["recall"].tolist() == [1, 1]
    assert original == {path.name: path.read_bytes() for path in (tmp_path / "original").iterdir()}
    summary = json.loads((tmp_path / "artifacts/baseline_deduplicated/baseline_deduplicated_summary.json").read_text())
    assert summary["retrained"] is False
    assert summary["baseline_config"]["recovery_hours"] == 24
    assert "previously inspected" in summary["comparison_status"]
