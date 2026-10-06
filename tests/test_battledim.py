import pandas as pd
import pytest

from src.data.battledim import prepare_battledim
from src.data.loader import load_sensor_dataset


def write_sources(directory):
    index = pd.date_range("2018-01-01", periods=3, freq="5min", name="Timestamp")
    tables = {
        "SCADA_Pressures": {"n1": [28.92, 29.0, 29.1]},
        "SCADA_Flows": {"p227": [77.77, 72.51, 71.54]},
        "Leakages": {"p31": [0, 0.01, 0], "p158": [0, 0, 2.5]},
    }
    for kind, values in tables.items():
        pd.DataFrame(values, index=index).to_csv(directory / f"2018_{kind}.csv", sep=";", decimal=",")


def test_official_format_and_labels_are_loader_compatible(tmp_path):
    write_sources(tmp_path)
    frame, summary = prepare_battledim(tmp_path)
    assert frame["pressure_n1"].iloc[0] == 28.92
    assert frame["flow_p227"].iloc[0] == 77.77
    assert frame["leak"].tolist() == [0, 1, 1]
    assert summary["network_leak_episodes"] == 1
    path = tmp_path / "merged.csv"
    frame.to_csv(path)
    dataset = load_sensor_dataset(path)
    assert dataset.pressure_sensors == ["pressure_n1"]
    assert dataset.flow_sensors == ["flow_p227"]
    assert dataset.sensor_columns == ["pressure_n1", "flow_p227"]


def test_misaligned_sources_fail_without_silently_dropping_rows(tmp_path):
    write_sources(tmp_path)
    path = tmp_path / "2018_SCADA_Flows.csv"
    frame = pd.read_csv(path, sep=";", decimal=",")
    frame.iloc[:-1].to_csv(path, sep=";", decimal=",", index=False)
    with pytest.raises(ValueError, match="timestamps do not match"):
        prepare_battledim(tmp_path)


def test_missing_leakage_does_not_become_a_negative_label(tmp_path):
    write_sources(tmp_path)
    path = tmp_path / "2018_Leakages.csv"
    frame = pd.read_csv(path, sep=";", decimal=",")
    frame.loc[1, "p31"] = float("nan")
    frame.to_csv(path, sep=";", decimal=",", index=False)
    with pytest.raises(ValueError, match="missing or infinite"):
        prepare_battledim(tmp_path)
