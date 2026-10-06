from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.train_lightgbm import run
from src.detection.baseline import BaselineConfig, ResidualBaselineDetector
from src.detection.classifier import ClassifierConfig, ClassifierScores, LeakClassifier
from src.detection.features import onset_window_labels, window_features


def fixture_frames():
    rng = np.random.default_rng(31)
    index = pd.date_range("2018-01-01", periods=24 * 20, freq="h", name="timestamp")
    daily = np.sin(np.arange(len(index)) * np.pi / 12)
    frame = pd.DataFrame({
        "pressure_n1": 30 + daily + rng.normal(0, 0.1, len(index)),
        "pressure_n4": 35 + daily + rng.normal(0, 0.1, len(index)),
        "flow_p1": 10 + daily / 2 + rng.normal(0, 0.1, len(index)),
    }, index=index)
    leakage = pd.DataFrame({"p1": 0.0}, index=index)
    for day in (8, 10, 12, 16, 18):
        rows = slice(day * 24, day * 24 + 12)
        frame.iloc[rows, 0] -= 4
        leakage.iloc[rows, 0] = 2
    return frame, leakage


def test_window_features_are_causal_and_exclude_leak_labels():
    frame, _ = fixture_frames()
    detector = ResidualBaselineDetector(["pressure_n1", "pressure_n4"], list(frame.columns), BaselineConfig(rolling_window=6)).fit(frame.iloc[:100])
    first = window_features(frame, detector.score(frame), list(frame.columns), 6)
    changed = frame.copy()
    changed.iloc[300:, 0] += 10000
    second = window_features(changed, detector.score(changed), list(frame.columns), 6)
    pd.testing.assert_frame_equal(first.iloc[:300], second.iloc[:300])
    assert np.isfinite(first.to_numpy()).all()
    assert "residual_std__pressure_n1" in first
    assert not any("leak" in column for column in first)


def test_onset_target_excludes_carry_in_and_has_exact_window_end():
    index = pd.date_range("2018-01-01", periods=60, freq="h")
    leakage = pd.DataFrame({"old": 1, "new": [0] * 5 + [1] * 55}, index=index)
    labels = onset_window_labels(leakage, 24)
    assert labels.iloc[:5].sum() == 0
    assert labels.iloc[5:29].eq(1).all()
    assert labels.iloc[29:].sum() == 0
    leakage.loc[index[10]:, "new"] = 0
    assert onset_window_labels(leakage, 24).iloc[10:].sum() == 0


def test_training_reload_and_test_labels_do_not_affect_model(tmp_path: Path):
    frame, leakage = fixture_frames()
    merged = frame.assign(leak=leakage.gt(0).any(axis=1).astype(int))
    merged_path = tmp_path / "merged.csv"
    leakage_path = tmp_path / "leakage.csv"

    def write_data():
        merged.to_csv(merged_path)
        leakage.rename_axis("Timestamp").to_csv(leakage_path, sep=";", decimal=",")

    write_data()
    config = {
        "data": {"raw_path": str(merged_path)}, "split": {"train_fraction": 0.7},
        "evaluation": {"leakage_path": str(leakage_path)},
        "classifier": {"n_estimators": 15, "num_leaves": 7, "output_dir": "model"},
    }
    summary = run(config, tmp_path)
    partitions = summary["partitions"]
    assert partitions["ridge_end"] < partitions["classifier_start"]
    assert partitions["classifier_end"] < partitions["validation_start"]
    assert partitions["validation_end"] < summary["test_start"]
    model = joblib.load(summary["model_path"])
    split_pos = int(len(frame) * 0.7)
    test = frame.iloc[split_pos:]
    history = frame.iloc[:split_pos].tail(model.config.rolling_window)
    scores = model.score(test, history)
    exported = pd.read_csv(tmp_path / "model/lightgbm_scores.csv")
    np.testing.assert_allclose(scores.probability, exported["probability"])
    assert scores.probability.between(0, 1).all()
    assert (tmp_path / "model/lightgbm_booster.txt").stat().st_size > 0
    # Alter unseen ground truth and its merged labels, leaving sensors fixed.
    leakage.iloc[split_pos:] = 0
    merged["leak"] = leakage.gt(0).any(axis=1).astype(int)
    write_data()
    second_summary = run(config, tmp_path)
    assert second_summary["threshold"] == summary["threshold"]
    second_model = joblib.load(second_summary["model_path"])
    np.testing.assert_array_equal(scores.probability, second_model.score(test, history).probability)


def test_classifier_uses_standard_alarm_contract():
    index = pd.date_range("2018-01-01", periods=40, freq="5min")
    detector = LeakClassifier(["P1"], ["P1", "F1"], ClassifierConfig(persistence_hours=3))
    sensor_scores = pd.DataFrame({"P1": 1.0}, index=index)
    scores = ClassifierScores(pd.Series(0.8, index=index), pd.DataFrame(index=index), sensor_scores, 0.5)
    alarms = detector.create_alarms(scores)
    assert len(alarms) == 1
    assert alarms[0].detector == "lightgbm"
    assert alarms[0].timestamp == index[36].to_pydatetime()


def test_history_cannot_include_future_samples():
    frame, _ = fixture_frames()
    detector = LeakClassifier(["pressure_n1"], list(frame.columns))
    detector.model = object()
    detector.threshold_ = 0.5
    with pytest.raises(ValueError, match="strictly precede"):
        detector.score(frame.iloc[-10:], frame.iloc[-20:])
