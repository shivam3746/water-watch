from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.detection.baseline import BaselineConfig, BaselineScores, ResidualBaselineDetector


def test_baseline_fit_and_score_uses_chronological_train_data():
    index = pd.date_range("2026-01-01", periods=12, freq="h")
    frame = pd.DataFrame(
        {
            "P1": [10, 10.1, 10.2, 10.1, 10.0, 10.2, 8.0, 8.1, 8.2, 8.0, 8.1, 8.2],
            "P2": [9, 9.1, 9.2, 9.1, 9.0, 9.2, 9.1, 9.0, 9.1, 9.0, 9.1, 9.0],
            "F1": [2, 2, 2.1, 2, 2, 2.1, 3, 3.1, 3, 3.1, 3, 3.1],
        },
        index=index,
    )
    train = frame.iloc[:6]
    test = frame.iloc[6:]

    detector = ResidualBaselineDetector(
        pressure_sensors=["P1", "P2"],
        feature_sensors=["P1", "P2", "F1"],
        config=BaselineConfig(rolling_window=2, threshold=0.5, persistence_hours=2),
    ).fit(train)
    scores = detector.score(test)

    assert scores.residuals.index.min() == test.index.min()
    assert scores.threshold == 0.5
    assert len(scores.aggregate_score) == len(test)


def test_alarm_creation_requires_persistence():
    index = pd.date_range("2026-01-01", periods=5, freq="h")
    scores = BaselineScores(
        residuals=pd.DataFrame({"P1": [0, 0, 0, 0, 0]}, index=index),
        sensor_scores=pd.DataFrame({"P1": [0.1, 0.8, 0.9, 0.2, 0.95]}, index=index),
        aggregate_score=pd.Series([0.1, 0.8, 0.9, 0.2, 0.95], index=index),
        threshold=0.5,
    )
    detector = ResidualBaselineDetector(
        pressure_sensors=["P1"],
        feature_sensors=["P1", "F1"],
        config=BaselineConfig(persistence_hours=1),
    )

    alarms = detector.create_alarms(scores)

    assert len(alarms) == 1
    assert alarms[0].timestamp == index[2].to_pydatetime()
    assert alarms[0].affected_sensors == ["P1"]


def make_scores(index, values):
    sensor = pd.DataFrame({"P1": values}, index=index)
    return BaselineScores(sensor, sensor, sensor["P1"], 0.5)


def test_persistence_uses_elapsed_hours_and_emits_once_per_episode():
    index = pd.date_range("2026-01-01", periods=40, freq="5min")
    detector = ResidualBaselineDetector(["P1"], ["P1", "F1"], BaselineConfig(persistence_hours=3))
    scores = make_scores(index, [1.0] * len(index))
    alarms = detector.create_alarms(scores)
    assert len(alarms) == 1
    assert alarms[0].timestamp == index[36].to_pydatetime()
    assert alarms == detector.create_alarms(scores)


def test_gap_and_threshold_equality_reset_persistence():
    index = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 05:00", "2026-01-01 06:00", "2026-01-01 07:00", "2026-01-01 08:00", "2026-01-01 09:00"])
    detector = ResidualBaselineDetector(["P1"], ["P1", "F1"], BaselineConfig(persistence_hours=2))
    alarms = detector.create_alarms(make_scores(index, [1, 1, 1, 1, 0.5, 1, 1]))
    assert alarms == []


def test_score_never_updates_training_threshold_or_models():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(100, 3)), columns=["P1", "P2", "F1"], index=pd.date_range("2026-01-01", periods=100, freq="5min"))
    detector = ResidualBaselineDetector(["P1", "P2"], list(frame.columns)).fit(frame.iloc[:70])
    threshold = detector.threshold_
    coefficients = detector.models["P1"].coef_.copy()
    test = frame.iloc[70:].copy()
    test["P1"] += 1000
    detector.score(test)
    assert detector.threshold_ == threshold
    np.testing.assert_array_equal(detector.models["P1"].coef_, coefficients)


def test_invalid_sensor_data_fails_clearly():
    frame = pd.DataFrame({"P1": [1, np.nan], "F1": [2, 3]}, index=pd.date_range("2026-01-01", periods=2, freq="h"))
    with pytest.raises(ValueError, match="missing/infinite"):
        ResidualBaselineDetector(["P1"], ["P1", "F1"]).fit(frame)


@pytest.mark.parametrize("kwargs", [{"rolling_window": 0}, {"persistence_hours": -1}, {"threshold": -1}, {"threshold_quantile": 1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BaselineConfig(**kwargs)
