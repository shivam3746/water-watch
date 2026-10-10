from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def notification_trace(score: pd.Series, threshold: float, persistence_hours: float,
                       recovery_hours: float, cooldown_hours: float) -> pd.DataFrame:
    """Diagnostic replay of the unchanged notification state, not a new detector."""
    if not isinstance(score.index, pd.DatetimeIndex) or len(score) < 2 or not score.index.is_monotonic_increasing or score.index.has_duplicates:
        raise ValueError("A regular chronological score sequence is required.")
    if not np.isfinite(threshold) or any(not np.isfinite(x) or x < 0 for x in
                                       (persistence_hours, recovery_hours, cooldown_hours)):
        raise ValueError("Threshold and nonnegative durations must be finite.")
    cadence = score.index[1] - score.index[0]
    if cadence <= pd.Timedelta(0) or not (score.index.to_series().diff().dropna() == cadence).all():
        raise ValueError("Score timestamps must have no gaps.")
    if not np.isfinite(score.to_numpy()).all():
        raise ValueError("Diagnostic scores must be finite.")
    persistence, recovery, cooldown = [pd.Timedelta(hours=x) for x in
                                        (persistence_hours, recovery_hours, cooldown_hours)]
    run_start, recovery_start, last_alarm = None, None, None
    incident_open, emitted = False, False
    rows = []
    for timestamp, value in score.items():
        before = incident_open
        above, ready, notification = value > threshold, False, False
        cooling = last_alarm is not None and timestamp - last_alarm < cooldown
        if not above:
            run_start, emitted = None, False
            if incident_open:
                if recovery_start is None:
                    recovery_start = timestamp
                if timestamp - recovery_start >= recovery:
                    incident_open, recovery_start = False, None
        else:
            recovery_start = None
            if run_start is None:
                run_start = timestamp
            ready = timestamp - run_start >= persistence
            if not emitted and not incident_open and ready and not cooling:
                emitted, incident_open, last_alarm, notification = True, True, timestamp, True
        rows.append((above, ready, before, before and ready and above, notification))
    return pd.DataFrame(rows, index=score.index, columns=["above_threshold", "persistence_ready",
                        "incident_open_before", "open_incident_blocks_ready_score", "notification"])


def event_size_rows(leakage: pd.DataFrame, events: list[dict], year: int,
                    scores: pd.Series | None = None, gates: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    observation_end = leakage.index[-1] + (leakage.index[1] - leakage.index[0])
    for event in events:
        if event["status"] == "excluded_carry_in":
            continue
        start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
        deadline = min(start + pd.Timedelta(hours=48), end, observation_end)
        early = leakage.loc[(leakage.index >= start) & (leakage.index < deadline), event["pipe"]]
        full = leakage.loc[(leakage.index >= start) & (leakage.index < end), event["pipe"]]
        if early.empty or full.empty or early.min() <= 0:
            raise ValueError("Event labels do not agree with positive pipe leakage.")
        row = {"year": year, **event, "early_window_end_exclusive": deadline.isoformat(),
               "early_samples": len(early), "early_mean_flow_m3h": float(early.mean()),
               "early_peak_flow_m3h": float(early.max()), "observed_peak_flow_m3h": float(full.max()),
               "early_window_hours": len(early) * (leakage.index[1] - leakage.index[0]).total_seconds() / 3600}
        if scores is not None and gates is not None:
            window = gates.loc[early.index]
            row.update({"score_at_onset": float(scores.loc[start]),
                        "score_mean_48h": float(scores.loc[early.index].mean()),
                        "score_peak_48h": float(scores.loc[early.index].max()),
                        "incident_open_at_onset": bool(gates.loc[start, "incident_open_before"]),
                        "above_threshold_fraction": float(window.above_threshold.mean()),
                        "persistence_ready_samples": int(window.persistence_ready.sum()),
                        "blocked_ready_samples": int(window.open_incident_blocks_ready_score.sum()),
                        "notifications_in_window": int(window.notification.sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def compare_sensors(first: pd.DataFrame, second: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(first.columns) != set(second.columns):
        raise ValueError("Yearly sensor names differ.")
    rows, profiles = [], []
    for sensor in first.columns:
        a, b = first[sensor], second[sensor]
        sd = float(a.std())
        profile_a = a.groupby([a.index.month, a.index.hour]).mean()
        profile_b = b.groupby([b.index.month, b.index.hour]).mean()
        change = profile_b - profile_a
        rows.append({"kind": kind, "sensor": sensor, "unit": "m" if kind == "pressure" else "m3/h",
                     "mean_2018": float(a.mean()), "mean_2019": float(b.mean()),
                     "std_2018": sd, "std_2019": float(b.std()),
                     "mean_shift_2018_sd": float((b.mean() - a.mean()) / sd) if sd > 0 else None,
                     "matched_month_hour_rmse": float(np.sqrt(np.mean(change ** 2))),
                     "daily_mean_ks_distance": float(ks_2samp(a.resample("D").mean(), b.resample("D").mean()).statistic),
                     "unchanged_step_fraction_2018": float(a.diff().iloc[1:].eq(0).mean()),
                     "unchanged_step_fraction_2019": float(b.diff().iloc[1:].eq(0).mean())})
        for year, values in [(2018, a), (2019, b)]:
            for month, mean in values.groupby(values.index.month).mean().items():
                profiles.append({"year": year, "kind": kind, "sensor": sensor, "month": int(month), "mean": float(mean)})
    return pd.DataFrame(rows), pd.DataFrame(profiles)
