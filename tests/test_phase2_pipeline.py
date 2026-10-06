import json

import joblib
import numpy as np

from scripts.phase2_baseline import demo_dataset, run


def test_pipeline_exports_reproducible_artifacts_and_reloadable_model(tmp_path):
    dataset = demo_dataset()
    config = {"split": {"train_fraction": 0.7}}
    first = run(config, dataset, tmp_path / "first", demo=True)
    second = run(config, dataset, tmp_path / "second", demo=True)
    assert first["train_end"] < first["test_start"]
    assert first["alarm_count"] > 0
    assert first["threshold"] == second["threshold"]
    for filename in ("baseline_scores.csv", "baseline_residuals.csv", "baseline_sensor_scores.csv", "baseline_alarms.json"):
        assert (tmp_path / "first" / filename).read_bytes() == (tmp_path / "second" / filename).read_bytes()
    alarms = json.loads((tmp_path / "first" / "baseline_alarms.json").read_text())
    assert alarms[0]["detector"] == "baseline"
    assert (tmp_path / "first" / "baseline_alarms.png").stat().st_size > 1000
    model = joblib.load(first["model_path"])
    test = dataset.frame.iloc[int(len(dataset.frame) * 0.7):][dataset.sensor_columns]
    scores = model.score(test)
    import pandas as pd
    exported = pd.read_csv(tmp_path / "first" / "baseline_scores.csv")
    np.testing.assert_allclose(scores.aggregate_score, exported["anomaly_score"])
    # Changing labels cannot change detection or threshold selection.
    dataset.frame["leak"] = 1 - dataset.frame["leak"]
    third = run(config, dataset, tmp_path / "third", demo=True)
    assert third["threshold"] == first["threshold"]
    assert (tmp_path / "third" / "baseline_alarms.json").read_bytes() == (tmp_path / "first" / "baseline_alarms.json").read_bytes()
