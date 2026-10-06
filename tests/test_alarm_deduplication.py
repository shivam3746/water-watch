import numpy as np
import pandas as pd
import pytest

from src.detection.alarms import create_persistent_alarms


def alarms_for(values, recovery=24, cooldown=24, index=None):
    if index is None:
        index = pd.date_range("2018-01-01", periods=len(values), freq="h")
    score = pd.Series(values, index=index)
    return create_persistent_alarms(score, 0.5, 3, pd.DataFrame({"P1": score}, index=index), "baseline", pd.Timedelta(hours=1), recovery, cooldown)


def test_short_normal_dips_do_not_create_repeat_notifications():
    values = [1] * 10 + [0] * 2 + [1] * 40 + [0] * 3 + [1] * 10
    legacy = alarms_for(values, recovery=0, cooldown=0)
    dedup = alarms_for(values)
    assert len(legacy) == 3
    assert len(dedup) == 1
    assert dedup[0] == legacy[0]


def test_sustained_recovery_allows_a_new_alarm_with_unchanged_persistence():
    index = pd.date_range("2018-01-01", periods=43, freq="h")
    values = [1] * 10 + [0] * 25 + [1] * 8
    alarms = alarms_for(values, index=index)
    assert [alarm.timestamp for alarm in alarms] == [index[3].to_pydatetime(), index[38].to_pydatetime()]


def test_cooldown_is_elapsed_from_emitted_alarm_and_not_extended_by_suppressed_runs():
    index = pd.date_range("2018-01-01", periods=40, freq="h")
    alarms = alarms_for([1] * 5 + [0] + [1] * 34, recovery=0, cooldown=24, index=index)
    assert [alarm.timestamp for alarm in alarms] == [index[3].to_pydatetime(), index[27].to_pydatetime()]


def test_missing_samples_and_gaps_cannot_close_an_open_incident():
    index = pd.to_datetime(["2018-01-01 00:00", "2018-01-01 01:00", "2018-01-01 02:00", "2018-01-01 03:00", "2018-01-01 04:00", "2018-01-03 04:00", "2018-01-03 05:00", "2018-01-03 06:00", "2018-01-03 07:00", "2018-01-03 08:00"])
    assert len(alarms_for([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], index=index)) == 1
    assert len(alarms_for([1] * 5 + [np.nan] * 25 + [1] * 10)) == 1


def test_batch_replay_is_causal_and_reproducible():
    values = [1] * 10 + [0] * 25 + [1] * 10
    assert alarms_for(values[:20]) == alarms_for(values)[:1]
    assert alarms_for(values) == alarms_for(values)


@pytest.mark.parametrize("recovery,cooldown", [(-1, 24), (24, -1), (np.nan, 24), (24, np.inf)])
def test_invalid_dedup_parameters_are_rejected(recovery, cooldown):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        alarms_for([1] * 10, recovery=recovery, cooldown=cooldown)
