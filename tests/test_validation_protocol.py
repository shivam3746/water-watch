import numpy as np
import pandas as pd
import pytest

from src.detection.baseline import BaselineScores
from src.detection.change import cusum, ewma
from src.detection.compact_features import compact_features, known_onset_labels, predicted_rise
from src.evaluation.validation import chronological_folds
from src.evaluation.validation_report import write_validation_report


def test_monthly_folds_are_complete_and_strictly_chronological():
    index = pd.date_range("2018-01-01", "2018-12-31 23:55", freq="5min")
    folds = chronological_folds(index, ["2018-05-01", "2018-06-01"], 28, 2018)
    assert len(folds) == 2
    assert folds[0].fit_index[-1] == pd.Timestamp("2018-04-02 23:55")
    assert len(folds[0].calibration_index) == 28 * 288
    assert len(folds[0].test_index) == 31 * 288
    assert folds[0].test_index.intersection(folds[1].test_index).empty
    for fold in folds:
        assert fold.fit_index[-1] < fold.calibration_index[0]
        assert fold.calibration_index[-1] < fold.test_index[0]


def test_final_year_is_rejected():
    with pytest.raises(ValueError, match="2019 is reserved"):
        chronological_folds(pd.date_range("2019-01-01", periods=10, freq="5min"), ["2019-05-01"], 28, 2019)


@pytest.mark.parametrize("missing", [100, -1])
def test_missing_observations_are_rejected(missing):
    index = pd.date_range("2018-01-01", "2018-05-31 23:55", freq="5min").delete(missing)
    with pytest.raises(ValueError):
        chronological_folds(index, ["2018-05-01"], 28, 2018)


def test_change_charts_are_causal_and_cusum_resets():
    score = pd.Series([1., 3., 3., 0., 20.])
    assert cusum(score, 1, 1, .5).tolist() == [0., 1.5, 3., 1.5, 20.]
    for transform in (lambda s: ewma(s, .1), lambda s: cusum(s, 1, 1, .5)):
        pd.testing.assert_series_equal(transform(score).iloc[:4], transform(score.iloc[:4]))


@pytest.mark.parametrize("alpha", [0, -1, 2])
def test_invalid_ewma_alpha(alpha):
    with pytest.raises(ValueError):
        ewma(pd.Series([1.]), alpha)


def test_ongoing_leaks_are_unknown_not_negative():
    index = pd.date_range("2018-01-01", periods=7, freq="h")
    leakage = pd.DataFrame({"pipe": [0, 1, 1, 1, 1, 0, 0]}, index=index)
    labels = known_onset_labels(leakage, 2)
    assert labels.iloc[:3].tolist() == [0, 1, 1]
    assert labels.iloc[3:5].isna().all()
    assert labels.iloc[5:].tolist() == [0, 0]
    leakage.iloc[0, 0] = 1
    assert known_onset_labels(leakage, 2).iloc[:5].isna().all()


def test_compact_features_do_not_depend_on_future_rows():
    index = pd.date_range("2018-01-01", periods=100, freq="5min")
    frame = pd.DataFrame({"pressure": np.arange(100.), "flow": np.arange(100.) * 2}, index=index)
    def build(data):
        residual = data[["pressure"]] * .1
        scores = BaselineScores(residual, residual.abs(), residual.abs().mean(axis=1), 1.)
        return compact_features(data, scores, ["pressure"], ["flow"])
    assert build(frame).shape[1] == 21
    pd.testing.assert_frame_equal(build(frame).iloc[:80], build(frame.iloc[:80]))


def test_regression_rise_is_trailing_and_ignores_future():
    index = pd.date_range("2018-01-01", periods=60, freq="h")
    prediction = pd.Series([10.] * 30 + [20.] * 30, index=index)
    rise = predicted_rise(prediction)
    assert rise.iloc[:30].eq(0).all()
    assert rise.iloc[35] == 10
    assert rise.iloc[-1] == 0
    pd.testing.assert_series_equal(rise.iloc[:40], predicted_rise(prediction.iloc[:40]))


def test_portable_report_retains_misses_and_distinct_events(tmp_path):
    (tmp_path / "run_manifest.json").write_text('{"2019_loaded": false}')
    summary = pd.DataFrame([{
        "model": "baseline", "evaluated_folds": 1, "eligible_events": 2,
        "detected_events": 1, "missed_events": 1, "precision": .5, "recall": .5,
        "false_alarms_per_week": .2, "detection_delay_mean_hours": 3.,
        "detection_delay_std_hours": float("nan"), "uninformative_calibration_folds": 1,
    }])
    events = pd.DataFrame([
        {"fold": "2018-05", "model": "baseline", "pipe": "p1", "event_id": "p1-first", "status": "detected", "delay_hours": 3.},
        {"fold": "2018-05", "model": "baseline", "pipe": "p1", "event_id": "p1-second", "status": "missed", "delay_hours": None},
    ])
    write_validation_report(tmp_path, summary, events)
    content = (tmp_path / "report.html").read_text()
    assert "MISSED" in content
    assert "p1-first" in content and "p1-second" in content
    assert "data:image/png;base64," in content
    assert (tmp_path / "comparison.png").stat().st_size > 1000
