from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd

from src.agent.schemas import Alarm


def create_persistent_alarms(
    score: pd.Series,
    threshold: float,
    persistence_hours: float,
    sensor_scores: pd.DataFrame,
    detector: str,
    cadence: pd.Timedelta | None = None,
    recovery_hours: float = 0,
    cooldown_hours: float = 0,
) -> list[Alarm]:
    """Emit causal notifications, retaining an incident through short normal dips."""
    for name, value in (("persistence_hours", persistence_hours), ("recovery_hours", recovery_hours), ("cooldown_hours", cooldown_hours)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    persistence = pd.Timedelta(hours=persistence_hours)
    recovery = pd.Timedelta(hours=recovery_hours)
    cooldown = pd.Timedelta(hours=cooldown_hours)
    if cadence is None and len(score) > 1:
        cadence = score.index.to_series().diff().dropna().median()
    run_start, previous, emitted = None, None, False
    incident_open, recovery_start, last_alarm = False, None, None
    alarms = []
    for timestamp, value in score.items():
        if previous is not None and cadence is not None and timestamp - previous > cadence * 1.5:
            run_start, emitted = None, False
            recovery_start = None
        previous = timestamp
        if not np.isfinite(value):
            run_start, recovery_start, emitted = None, None, False
            continue
        if value <= threshold:
            run_start, emitted = None, False
            if incident_open:
                if recovery_start is None:
                    recovery_start = timestamp
                if timestamp - recovery_start >= recovery:
                    incident_open, recovery_start = False, None
            continue
        recovery_start = None
        if run_start is None:
            run_start = timestamp
        if emitted or incident_open or timestamp - run_start < persistence:
            continue
        if last_alarm is not None and timestamp - last_alarm < cooldown:
            continue
        emitted = True
        incident_open, last_alarm = True, timestamp
        affected = sensor_scores.loc[timestamp].sort_values(ascending=False, kind="stable").head(3).index.tolist()
        alarms.append(Alarm(
            id=f"{detector}-{uuid5(NAMESPACE_URL, timestamp.isoformat()).hex[:12]}",
            timestamp=timestamp.to_pydatetime(), score=float(value),
            detector=detector, affected_sensors=affected,
        ))
    return alarms
