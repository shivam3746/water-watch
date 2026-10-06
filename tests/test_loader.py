from __future__ import annotations

import pandas as pd

from src.data.loader import load_sensor_dataset


def test_loader_returns_timestamp_indexed_sensor_dataset(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00", "2026-01-01 01:00"],
            "P1": [10.0, 9.5],
            "F1": [2.0, 2.2],
            "leak": [0, 1],
        }
    ).to_csv(path, index=False)

    dataset = load_sensor_dataset(path)

    assert dataset.frame.index.name == "timestamp"
    assert dataset.pressure_sensors == ["P1"]
    assert dataset.flow_sensors == ["F1"]
    assert dataset.leak_label_column == "leak"
    assert dataset.frame["leak"].tolist() == [0, 1]
