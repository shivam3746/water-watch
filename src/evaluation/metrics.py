from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.agent.schemas import Alarm


@dataclass(frozen=True)
class LeakEvent:
    id: str
    pipe: str
    start: pd.Timestamp
    end: pd.Timestamp
    left_censored: bool = False
    right_censored: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict
    events: list[dict]
    alarms: list[dict]


def extract_leak_events(leakage: pd.DataFrame) -> list[LeakEvent]:
    """Extract per-pipe positive runs as [start, end) observation intervals."""
    index = leakage.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        raise ValueError("Leakage requires at least two timestamped observations.")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Leakage timestamps must be valid, sorted, and unique.")
    deltas = index.to_series().diff().dropna()
    cadence = deltas.iloc[0]
    if not (deltas == cadence).all():
        raise ValueError("Leakage observations must have a regular cadence with no gaps.")
    if not np.isfinite(leakage.to_numpy(dtype=float)).all() or (leakage < 0).any().any():
        raise ValueError("Leakage values must be finite and nonnegative.")
    events = []
    for pipe in leakage.columns:
        positive = leakage[pipe].gt(0)
        starts = np.flatnonzero((positive & ~positive.shift(1, fill_value=False)).to_numpy())
        ends = np.flatnonzero((~positive & positive.shift(1, fill_value=False)).to_numpy())
        for start_pos in starts:
            later_ends = ends[ends > start_pos]
            end_pos = int(later_ends[0]) if len(later_ends) else len(index)
            start = index[start_pos]
            end = index[end_pos] if end_pos < len(index) else index[-1] + cadence
            events.append(LeakEvent(
                id=f"{pipe}-{start.isoformat()}", pipe=str(pipe), start=start, end=end,
                left_censored=bool(start_pos == 0), right_censored=bool(end_pos == len(index)),
            ))
    return sorted(events, key=lambda event: (event.start, event.pipe))


def evaluate_alarms(
    alarms: list[Alarm],
    events: list[LeakEvent],
    test_index: pd.DatetimeIndex,
    model_name: str = "baseline",
    max_detection_delay_hours: float | None = None,
) -> EvaluationResult:
    """One-to-one temporal matching for new leak onsets in the test period.

    All unmatched test alarms, including repeats and carry-in-only alarms, are
    false alerts for the new-event task. They are not proof of leak-free water.
    """
    if not isinstance(test_index, pd.DatetimeIndex) or len(test_index) < 2:
        raise ValueError("Evaluation requires at least two test observations.")
    if test_index.hasnans or test_index.has_duplicates or not test_index.is_monotonic_increasing:
        raise ValueError("Test timestamps must be valid, sorted, and unique.")
    deltas = test_index.to_series().diff().dropna()
    if not (deltas == deltas.iloc[0]).all():
        raise ValueError("Evaluation requires a regular test cadence with no gaps.")
    if max_detection_delay_hours is not None and (
        not np.isfinite(max_detection_delay_hours) or max_detection_delay_hours <= 0
    ):
        raise ValueError("max_detection_delay_hours must be positive and finite, or null.")
    if len({alarm.id for alarm in alarms}) != len(alarms):
        raise ValueError("Alarm IDs must be unique within a model run.")
    if len({event.id for event in events}) != len(events):
        raise ValueError("Leak event IDs must be unique.")
    if any(event.end <= event.start for event in events):
        raise ValueError("Leak events must have end > start.")
    start, end = test_index[0], test_index[-1] + deltas.iloc[0]
    overlapping = [event for event in events if event.start < end and event.end > start]
    eligible = [event for event in overlapping if event.start >= start and not event.left_censored]
    carry_in = [event for event in overlapping if event.start < start or event.left_censored]
    limit = pd.Timedelta(hours=max_detection_delay_hours) if max_detection_delay_hours is not None else None

    def matching_end(event):
        return min(event.end, end, event.start + limit) if limit is not None else min(event.end, end)

    matched: dict[str, Alarm] = {}
    alarm_rows = []
    for alarm in sorted(alarms, key=lambda alarm: (alarm.timestamp, alarm.id)):
        timestamp = pd.Timestamp(alarm.timestamp)
        if not start <= timestamp < end:
            raise ValueError(f"Alarm {alarm.id} is outside the held-out test interval.")
        active = [event for event in eligible if event.start <= timestamp < matching_end(event)]
        candidates = [event for event in active if event.id not in matched]
        # Earliest deadline first, with deterministic tie-breaking for overlaps.
        candidates.sort(key=lambda event: (matching_end(event), event.start, event.pipe, event.id))
        selected = candidates[0] if candidates else None
        if selected:
            matched[selected.id] = alarm
            status = "matched"
        elif any(event.start <= timestamp < event.end and event.id in matched for event in eligible):
            status = "duplicate"
        elif any(event.start <= timestamp < event.end for event in eligible):
            status = "late"
        elif any(event.start <= timestamp < event.end for event in carry_in):
            status = "carry_in_only"
        else:
            status = "unmatched"
        alarm_rows.append({
            "alarm_id": alarm.id, "timestamp": timestamp.isoformat(),
            "status": status, "event_id": selected.id if selected else None,
            "candidate_event_ids": [event.id for event in candidates],
            "ambiguous_match": len(candidates) > 1,
            "delay_hours": (timestamp - selected.start).total_seconds() / 3600 if selected else None,
        })
    event_rows = []
    for event in overlapping:
        alarm = matched.get(event.id)
        excluded = event not in eligible
        event_rows.append({
            "event_id": event.id, "pipe": event.pipe,
            "start": event.start.isoformat(), "end": event.end.isoformat(),
            "status": "excluded_carry_in" if excluded else ("detected" if alarm else "missed"),
            "left_censored": event.left_censored,
            "right_censored_at_test_end": event.right_censored or event.end > end,
            "alarm_id": alarm.id if alarm else None,
            "delay_hours": (pd.Timestamp(alarm.timestamp) - event.start).total_seconds() / 3600 if alarm else None,
        })
    tp, fp, fn = len(matched), len(alarms) - len(matched), len(eligible) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    delays = [row["delay_hours"] for row in event_rows if row["status"] == "detected"]
    weeks = (end - start).total_seconds() / (7 * 24 * 3600)
    metrics = {
        "model": model_name, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "avg_detection_delay_hours": float(np.mean(delays)) if delays else None,
        "false_alarms_per_week": fp / weeks,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "alarm_count": len(alarms), "eligible_leak_events": len(eligible),
        "excluded_carry_in_events": len(carry_in),
        "duplicate_alarms": sum(row["status"] == "duplicate" for row in alarm_rows),
        "carry_in_only_alarms": sum(row["status"] == "carry_in_only" for row in alarm_rows),
        "ambiguous_matches": sum(row["ambiguous_match"] for row in alarm_rows),
        "test_start": start.isoformat(), "test_end_exclusive": end.isoformat(),
        "observed_weeks": weeks, "max_detection_delay_hours": max_detection_delay_hours,
        "matching_rule": "one-to-one temporal matching; earliest active deadline first; [onset, end)",
        "false_alarm_rule": "all test alarms unmatched to a new eligible event, including duplicates and carry-in-only alarms",
        "zero_denominator_rule": "precision/recall/F1 are 0 when their denominator is 0; missing delay is null",
    }
    return EvaluationResult(metrics, event_rows, alarm_rows)
