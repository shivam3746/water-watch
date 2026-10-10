import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.dashboard import MODEL_NAMES, event_window, load_results, load_timeline, replay_figure


@pytest.fixture
def replay_artifacts(tmp_path):
    index = pd.date_range("2018-10-01", periods=100, freq="5min", name="timestamp")
    timeline = pd.DataFrame({"total_leakage_m3_h": 12., "predicted_leakage_m3_h": 11.}, index=index)
    summaries, folds, events, alarms, calibrations = [], [], [], [], []
    for model in MODEL_NAMES:
        timeline[model] = .8
        timeline[model + "_threshold"] = .5
        summaries.append(dict(model=model, eligible_events=2, detected_events=1, missed_events=1,
                              false_alerts=0, precision=1., recall=.5, f1=2/3,
                              false_alarms_per_week=0., detection_delay_mean_hours=1., detection_delay_std_hours=None))
        folds.append(dict(model=model, fold="2018-10", status="evaluated", true_positives=1, false_negatives=1,
                          false_positives=0, eligible_leak_events=2, alarm_count=1, precision=1., recall=.5,
                          avg_detection_delay_hours=1., calibration_informative=False, excluded_carry_in_events=1))
        for event_id, status, start in [("p1", "detected", index[12]), ("p2", "missed", index[50]), ("p3", "excluded_carry_in", index[0] - pd.Timedelta(days=2))]:
            events.append(dict(model=model, fold="2018-10", pipe=event_id, event_id=event_id, start=start,
                               end=index[-1] + pd.Timedelta(days=1), status=status, delay_hours=1. if status == "detected" else None,
                               alarm_id="a1" if status == "detected" else None, detection_window_truncated=True))
        alarms.append(dict(model=model, fold="2018-10", timestamp=index[24], status="matched", event_id="p1", delay_hours=1., ambiguous_match=False))
        calibrations.append(dict(model=model, fold="2018-10", candidate=0, parameters="{}", threshold=.5,
                                 true_positives=0, false_positives=0, false_negatives=1, f1=0.))
    for filename, rows in [("summary.csv", summaries), ("fold_metrics.csv", folds), ("event_audit.csv", events),
                           ("alarm_audit.csv", alarms), ("calibration_audit.csv", calibrations)]:
        pd.DataFrame(rows).to_csv(tmp_path / filename, index=False)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"protocol_id": "test_protocol", "2019_loaded": False}))
    (tmp_path / "2018-10").mkdir()
    timeline.to_csv(tmp_path / "2018-10/timeline.csv.gz", compression="gzip")
    return tmp_path


def test_artifact_loading_and_path_traversal(replay_artifacts):
    data = load_results(replay_artifacts)
    assert len(data.summary) == 6
    assert isinstance(data.events.iloc[0]["start"], pd.Timestamp)
    with pytest.raises(ValueError, match="Unknown replay month"):
        load_timeline(replay_artifacts, "../2018-10")


def test_chart_cursor_does_not_show_future_observations(replay_artifacts):
    data = load_results(replay_artifacts)
    timeline = load_timeline(replay_artifacts, "2018-10")
    model = "baseline_deduplicated"
    events = data.events.loc[data.events.model == model]
    alarms = data.alarms.loc[data.alarms.model == model]
    end = timeline.index[30]
    figure = replay_figure(timeline, model, events, alarms, timeline.index[0], end, True, True)
    for trace in figure.data:
        assert all(pd.Timestamp(value) <= end for value in trace.x)
    assert len(figure.layout.shapes) == 1
    start, stop = event_window(timeline, events.iloc[1])
    assert start >= timeline.index[0] and stop <= timeline.index[-1]


def test_app_controls_and_downloads(replay_artifacts, monkeypatch):
    monkeypatch.setenv("WATER_WATCH_RESULTS", str(replay_artifacts))
    monkeypatch.setenv("WATER_WATCH_EXPLANATIONS", str(replay_artifacts / "missing_explanations"))
    monkeypatch.setenv("WATER_WATCH_FINAL_RESULTS", str(replay_artifacts / "missing_final"))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "1 / 2"
    app.selectbox[0].select(1).run()
    assert not app.exception
    app.selectbox[1].select("lightgbm_regression").run()
    assert not app.exception
    assert len(app.get("download_button")) == 5
    app.multiselect[0].set_value([]).run()
    assert not app.exception


def test_missing_artifacts_have_an_actionable_state(tmp_path, monkeypatch):
    monkeypatch.setenv("WATER_WATCH_RESULTS", str(tmp_path))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "Replay results unavailable" in app.error[0].value
