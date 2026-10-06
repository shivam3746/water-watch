from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.preprocessing import (
    LEAK_LABEL_CANDIDATES,
    deduplicate_timestamps,
    infer_column,
    infer_flow_sensors,
    infer_pressure_sensors,
    parse_timestamp_index,
    select_existing_columns,
)


@dataclass(frozen=True)
class SensorDataset:
    frame: pd.DataFrame
    pressure_sensors: list[str]
    flow_sensors: list[str]
    leak_label_column: str

    @property
    def sensor_columns(self) -> list[str]:
        return [*self.pressure_sensors, *self.flow_sensors]


def load_sensor_dataset(
    raw_path: str | Path,
    timestamp_column: str | None = None,
    leak_label_column: str | None = None,
    pressure_sensors: list[str] | None = None,
    flow_sensors: list[str] | None = None,
    duplicate_strategy: str = "mean",
) -> SensorDataset:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put a CSV there or update config.yaml."
        )
    if path.suffix.lower() != ".csv":
        raise ValueError("Phase 1 loader currently supports CSV files only.")

    raw = pd.read_csv(path)
    frame = parse_timestamp_index(raw, timestamp_column=timestamp_column)

    label_column = leak_label_column or infer_column(frame.columns, LEAK_LABEL_CANDIDATES)
    if label_column is None:
        raise ValueError(
            "Could not infer a leak label column. Set data.leak_label_column in config.yaml."
        )
    if label_column not in frame.columns:
        raise ValueError(f"Leak label column '{label_column}' was not found.")

    frame = deduplicate_timestamps(
        frame,
        leak_label_column=label_column,
        strategy=duplicate_strategy,
    )

    configured_pressure = pressure_sensors or []
    configured_flow = flow_sensors or []
    pressure = (
        select_existing_columns(frame, configured_pressure)
        if configured_pressure
        else infer_pressure_sensors(frame.columns)
    )
    flow = (
        select_existing_columns(frame, configured_flow)
        if configured_flow
        else infer_flow_sensors(frame.columns)
    )

    if not pressure:
        raise ValueError("No pressure sensors found. Configure data.pressure_sensors.")
    if not flow:
        raise ValueError("No flow sensors found. Configure data.flow_sensors.")

    frame[label_column] = frame[label_column].fillna(0).astype(int)
    ordered = frame[[*pressure, *flow, label_column]].sort_index()
    return SensorDataset(
        frame=ordered,
        pressure_sensors=pressure,
        flow_sensors=flow,
        leak_label_column=label_column,
    )
