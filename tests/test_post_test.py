import numpy as np
import json
from pathlib import Path
import pandas as pd
import pytest

from src.detection.alarms import create_persistent_alarms
from src.evaluation.post_test import compare_sensors, event_size_rows, notification_trace


@pytest.mark.parametrize("recovery,cooldown", [(0, 0), (24, 24), (6, 48)])
def test_diagnostic_state_reproduces_original_notifications(recovery, cooldown):
    index = pd.date_range("2018-12-20", periods=2000, freq="5min")
    values = np.tile(np.r_[np.ones(60) * 2, np.zeros(360)], 5)[:len(index)]
    score = pd.Series(values, index=index)
    sensor_scores = pd.DataFrame({"pressure_a": values}, index=index)
    alarms = create_persistent_alarms(score, 1, 3, sensor_scores, "baseline", index[1] - index[0], recovery, cooldown)
    trace = notification_trace(score, 1, 3, recovery, cooldown)
    assert list(trace.index[trace.notification]) == [pd.Timestamp(a.timestamp) for a in alarms]
    assert not trace.open_incident_blocks_ready_score.loc[trace.notification].any()


def test_open_incident_blocks_later_ready_scores_and_recovery_closes():
    index = pd.date_range("2019-01-01", periods=40, freq="h")
    score = pd.Series([2] * 10 + [0] * 25 + [2] * 5, index=index)
    trace = notification_trace(score, 1, 3, 24, 24)
    assert list(trace.index[trace.notification]) == [index[3], index[38]]
    assert trace.loc[index[9], "open_incident_blocks_ready_score"]
    assert not trace.loc[index[35], "incident_open_before"]


def test_event_size_uses_exclusive_early_window_not_late_peak():
    index = pd.date_range("2019-01-01", periods=60, freq="h")
    leakage = pd.DataFrame({"p1": [1.] * 48 + [100.] * 12}, index=index)
    event = {"event_id": "p1-event", "pipe": "p1", "start": index[0].isoformat(),
             "end": (index[-1] + pd.Timedelta(hours=1)).isoformat(), "status": "missed"}
    rows = event_size_rows(leakage, [event, {**event, "status": "excluded_carry_in"}], 2019)
    assert len(rows) == 1
    assert rows.iloc[0].early_mean_flow_m3h == 1
    assert rows.iloc[0].early_peak_flow_m3h == 1
    assert rows.iloc[0].observed_peak_flow_m3h == 100
    assert rows.iloc[0].early_samples == 48


def test_event_size_respects_short_event_end():
    index = pd.date_range("2019-01-01", periods=60, freq="h")
    leakage = pd.DataFrame({"p1": [2.] * 6 + [0.] * 54}, index=index)
    event = {"event_id": "e", "pipe": "p1", "start": index[0].isoformat(),
             "end": index[6].isoformat(), "status": "missed"}
    assert event_size_rows(leakage, [event], 2019).iloc[0].early_window_hours == 6


def test_sensor_comparison_uses_2018_scale_and_separate_profiles():
    first = pd.DataFrame({"p": [1., 2., 3., 4.]}, index=pd.date_range("2018-01-01", periods=4, freq="h"))
    second = pd.DataFrame({"p": [3., 4., 5., 6.]}, index=pd.date_range("2019-01-01", periods=4, freq="h"))
    audit, profiles = compare_sensors(first, second, "pressure")
    assert audit.iloc[0].mean_shift_2018_sd == pytest.approx(2 / first.p.std())
    assert audit.iloc[0].matched_month_hour_rmse == 2
    assert set(profiles.year) == {2018, 2019}
    with pytest.raises(ValueError, match="names differ"):
        compare_sensors(first, second.rename(columns={"p": "other"}), "pressure")


def test_diagnostic_replay_rejects_missing_scores_and_gaps():
    score = pd.Series([1., np.nan, 2., 3.], index=pd.date_range("2019-01-01", periods=4, freq="h"))
    with pytest.raises(ValueError, match="finite"):
        notification_trace(score, 1, 3, 24, 24)
    with pytest.raises(ValueError, match="gaps"):
        notification_trace(score.dropna(), 1, 3, 24, 24)


def test_published_diagnostics_keep_all_frozen_outcomes_and_provenance():
    root = Path(__file__).resolve().parents[1] / "demo_bundle"
    events = pd.read_csv(root / "diagnostics_2019/event_size_2019.csv")
    original = pd.read_csv(root / "final_2019/event_audit.csv")
    eligible = original.loc[original.status != "excluded_carry_in"]
    assert len(events) == len(eligible) == 19
    assert events.set_index("event_id").status.to_dict() == eligible.set_index("event_id").status.to_dict()
    assert events.status.eq("detected").sum() == 2
    assert events.loc[events.status == "missed", "incident_open_at_onset"].sum() == 14
    manifest = json.loads((root / "diagnostics_2019/manifest.json").read_text())
    assert manifest["protected_final_outputs_unchanged"]
    assert manifest["status"] == "exploratory_post_test_no_retuning"
