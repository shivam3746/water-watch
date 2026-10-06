from datetime import datetime
import json

import pandas as pd
import pytest

from src.agent.schemas import Alarm
from src.evaluation.metrics import LeakEvent, evaluate_alarms, extract_leak_events


def event(pipe, start, end, **kwargs):
    return LeakEvent(pipe, pipe, pd.Timestamp(start), pd.Timestamp(end), **kwargs)


def alarm(name, timestamp):
    return Alarm(name, datetime.fromisoformat(timestamp), 1.0, "baseline", ["P1"])


def test_extracts_individual_overlapping_runs_and_censoring():
    index = pd.date_range("2018-01-01", periods=6, freq="h")
    leakage = pd.DataFrame({"p1": [0, 1, 1, 0, 1, 0], "p2": [1, 1, 0, 0, 0, 1]}, index=index)
    events = extract_leak_events(leakage)
    p1 = [item for item in events if item.pipe == "p1"]
    p2 = [item for item in events if item.pipe == "p2"]
    assert len(p1) == len(p2) == 2
    assert p1[0].start == index[1] and p1[0].end == index[3]
    assert p2[0].left_censored
    assert p2[1].right_censored and p2[1].end == index[-1] + pd.Timedelta(hours=1)
    result = evaluate_alarms([], events, index)
    json.dumps({"metrics": result.metrics, "events": result.events, "alarms": result.alarms}, allow_nan=False)


def test_delay_duplicates_pre_onset_and_missed_events():
    index = pd.date_range("2018-01-01", periods=168, freq="h")
    events = [event("p1", "2018-01-02", "2018-01-03"), event("p2", "2018-01-04", "2018-01-05")]
    alarms = [alarm("early", "2018-01-01T23:00"), alarm("first", "2018-01-02T02:00"), alarm("repeat", "2018-01-02T03:00")]
    result = evaluate_alarms(alarms, events, index)
    assert result.metrics["true_positives"] == 1
    assert result.metrics["false_positives"] == 2
    assert result.metrics["false_negatives"] == 1
    assert result.metrics["precision"] == pytest.approx(1 / 3)
    assert result.metrics["recall"] == 0.5
    assert result.metrics["f1"] == pytest.approx(0.4)
    assert result.metrics["avg_detection_delay_hours"] == 2
    assert result.metrics["false_alarms_per_week"] == 2
    assert [row["status"] for row in result.alarms] == ["unmatched", "matched", "duplicate"]
    assert result.events[1]["delay_hours"] is None


def test_one_alarm_cannot_detect_two_overlapping_events():
    index = pd.date_range("2018-01-01", periods=24, freq="h")
    events = [event("long", "2018-01-01T02:00", "2018-01-01T12:00"), event("short", "2018-01-01T03:00", "2018-01-01T06:00")]
    result = evaluate_alarms([alarm("a", "2018-01-01T04:00")], events, index)
    assert result.metrics["true_positives"] == 1
    assert result.metrics["false_negatives"] == 1
    assert result.alarms[0]["event_id"] == "short"
    assert result.alarms[0]["ambiguous_match"]


def test_carry_in_excluded_from_recall_and_end_boundary_is_exclusive():
    index = pd.date_range("2018-01-02", periods=24, freq="h")
    events = [event("old", "2018-01-01", "2018-01-03"), event("new", "2018-01-02T03:00", "2018-01-02T05:00")]
    result = evaluate_alarms([alarm("old_alarm", "2018-01-02T02:00"), alarm("too_late", "2018-01-02T05:00")], events, index)
    assert result.metrics["excluded_carry_in_events"] == 1
    assert result.metrics["eligible_leak_events"] == 1
    assert result.metrics["false_negatives"] == 1
    assert result.metrics["false_positives"] == 2
    assert result.events[0]["status"] == "excluded_carry_in"


def test_max_delay_is_configurable_and_boundary_is_exclusive():
    index = pd.date_range("2018-01-01", periods=24, freq="h")
    events = [event("p1", "2018-01-01T02:00", "2018-01-01T20:00")]
    alarms = [alarm("a", "2018-01-01T05:00")]
    result = evaluate_alarms(alarms, events, index, max_detection_delay_hours=3)
    assert result.alarms[0]["status"] == "late"
    assert result.metrics["recall"] == 0
    detected = evaluate_alarms([alarm("first", "2018-01-01T03:00"), *alarms], events, index, max_detection_delay_hours=3)
    assert detected.alarms[1]["status"] == "duplicate"


def test_empty_results_are_json_safe_and_repeatable():
    index = pd.date_range("2018-01-01", periods=24, freq="h")
    result = evaluate_alarms([], [], index)
    assert result.metrics["precision"] == result.metrics["recall"] == result.metrics["f1"] == 0
    assert result.metrics["avg_detection_delay_hours"] is None
    assert result == evaluate_alarms([], [], index)


def test_invalid_alarm_ids_and_times_are_rejected():
    index = pd.date_range("2018-01-01", periods=24, freq="h")
    item = alarm("a", "2018-01-01T05:00")
    with pytest.raises(ValueError, match="IDs must be unique"):
        evaluate_alarms([item, item], [], index)
    with pytest.raises(ValueError, match="outside"):
        evaluate_alarms([alarm("a", "2017-12-31T23:55")], [], index)


def test_gaps_are_not_treated_as_observed_leak_or_test_time():
    index = pd.to_datetime(["2018-01-01T00:00", "2018-01-01T01:00", "2018-01-01T03:00"])
    with pytest.raises(ValueError, match="no gaps"):
        extract_leak_events(pd.DataFrame({"p1": [0, 1, 1]}, index=index))
    with pytest.raises(ValueError, match="no gaps"):
        evaluate_alarms([], [], index)
