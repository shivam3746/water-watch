from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import (
    chronological_split,
    deduplicate_timestamps,
    parse_timestamp_index,
    report_missing_values,
)


def test_chronological_split_does_not_overlap():
    frame = pd.DataFrame(
        {"P1": [1, 2, 3, 4], "leak": [0, 0, 1, 1]},
        index=pd.date_range("2026-01-01", periods=4, freq="h"),
    )

    split = chronological_split(frame, train_fraction=0.5)

    assert split.train.index.max() < split.test.index.min()
    assert split.train.index.tolist() == frame.index[:2].tolist()
    assert split.test.index.tolist() == frame.index[2:].tolist()


def test_parse_timestamp_index_sorts_rows():
    frame = pd.DataFrame(
        {
            "timestamp": ["2026-01-01 02:00", "2026-01-01 01:00"],
            "P1": [2, 1],
            "leak": [0, 0],
        }
    )

    parsed = parse_timestamp_index(frame)

    assert parsed.index.is_monotonic_increasing
    assert parsed.iloc[0]["P1"] == 1


def test_deduplicate_timestamps_preserves_positive_leak_label():
    frame = pd.DataFrame(
        {"P1": [1.0, 3.0], "leak": [0, 1]},
        index=pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
    )

    deduped = deduplicate_timestamps(frame, leak_label_column="leak", strategy="mean")

    assert len(deduped) == 1
    assert deduped.iloc[0]["P1"] == 2.0
    assert deduped.iloc[0]["leak"] == 1


def test_missing_value_report_counts_missing_cells():
    frame = pd.DataFrame({"P1": [1.0, None], "leak": [0, 1]})

    report = report_missing_values(frame)

    assert report.total_missing == 1
    assert report.missing_by_column["P1"] == 1
    assert report.missing_fraction_by_column["P1"] == pytest.approx(0.5)
