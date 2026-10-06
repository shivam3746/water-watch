from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Alarm:
    id: str
    timestamp: datetime
    score: float
    detector: str
    affected_sensors: list[str]
