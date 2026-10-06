from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_battledim_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", decimal=",")
    if "Timestamp" not in frame.columns or len(frame.columns) < 2:
        raise ValueError(f"{path.name}: expected Timestamp and measurement columns separated by semicolons.")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise")
    frame = frame.set_index("Timestamp").sort_index()
    frame.index.name = "timestamp"
    if frame.empty or frame.index.hasnans or frame.index.has_duplicates:
        raise ValueError(f"{path.name}: timestamps must be nonempty, valid, and unique.")
    frame = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path.name}: missing or infinite measurements found.")
    return frame


def prepare_battledim(raw_dir: Path, year: int = 2018) -> tuple[pd.DataFrame, dict]:
    sources = {
        "pressure": raw_dir / f"{year}_SCADA_Pressures.csv",
        "flow": raw_dir / f"{year}_SCADA_Flows.csv",
        "leakage": raw_dir / f"{year}_Leakages.csv",
    }
    frames = {kind: read_battledim_csv(path) for kind, path in sources.items()}
    pressure, flow, leakage = (frames[key] for key in ("pressure", "flow", "leakage"))
    for kind, frame in frames.items():
        if not pressure.index.equals(frame.index):
            raise ValueError(f"{kind} timestamps do not match pressure timestamps. No rows were dropped or filled.")
        if not (frame.index.year == year).all():
            raise ValueError(f"{kind} contains timestamps outside {year}.")
    if len(pressure) < 2 or not (pressure.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all():
        raise ValueError("Expected uninterrupted five-minute observations.")
    if (leakage < 0).any().any():
        raise ValueError("Leakage measurements must be nonnegative.")
    merged = pd.concat([pressure.add_prefix("pressure_"), flow.add_prefix("flow_")], axis=1)
    merged["leak"] = leakage.gt(0).any(axis=1).astype(int)
    labels = merged["leak"].astype(bool)
    summary = {
        "year": year, "rows": len(merged),
        "start": merged.index.min().isoformat(), "end": merged.index.max().isoformat(),
        "sample_interval_minutes": 5,
        "pressure_sensors": list(pressure.add_prefix("pressure_").columns),
        "flow_sensors": list(flow.add_prefix("flow_").columns),
        "sensor_mapping": {
            **{f"pressure_{name}": name for name in pressure.columns},
            **{f"flow_{name}": name for name in flow.columns},
        },
        "source_files": {kind: str(path.resolve()) for kind, path in sources.items()},
        "label_rule": "leak = 1 when any source pipe leakage measurement is strictly positive",
        "leak_rows": int(labels.sum()),
        "leak_fraction": float(labels.mean()),
        "network_leak_episodes": int((labels & ~labels.shift(1, fill_value=False)).sum()),
        "leaking_pipe_count": int(leakage.gt(0).any().sum()),
        "missing_values": int(merged.isna().sum().sum()),
    }
    return merged, summary
