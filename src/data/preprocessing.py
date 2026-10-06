from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


TIMESTAMP_CANDIDATES = ("timestamp", "time", "datetime", "date")
LEAK_LABEL_CANDIDATES = ("leak", "leak_label", "is_leak", "label")


@dataclass(frozen=True)
class MissingValueReport:
    total_missing: int
    missing_by_column: dict[str, int]
    missing_fraction_by_column: dict[str, float]


@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def infer_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for column in columns:
        lowered = column.lower()
        if any(candidate.lower() in lowered for candidate in candidates):
            return column
    return None


def parse_timestamp_index(
    df: pd.DataFrame,
    timestamp_column: str | None = None,
) -> pd.DataFrame:
    column = timestamp_column or infer_column(df.columns, TIMESTAMP_CANDIDATES)
    if column is None:
        raise ValueError(
            "Could not infer a timestamp column. Set data.timestamp_column in config.yaml."
        )
    if column not in df.columns:
        raise ValueError(f"Timestamp column '{column}' was not found in the dataset.")

    parsed = df.copy()
    parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
    if parsed[column].isna().any():
        bad_count = int(parsed[column].isna().sum())
        raise ValueError(f"Timestamp parsing failed for {bad_count} rows.")

    parsed = parsed.set_index(column)
    parsed.index.name = "timestamp"
    return parsed.sort_index()


def deduplicate_timestamps(
    df: pd.DataFrame,
    leak_label_column: str,
    strategy: str = "mean",
) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df

    if strategy not in {"mean", "first", "last"}:
        raise ValueError("duplicate_strategy must be one of: mean, first, last.")

    if strategy == "first":
        return df[~df.index.duplicated(keep="first")]
    if strategy == "last":
        return df[~df.index.duplicated(keep="last")]

    numeric = df.select_dtypes(include="number").columns.tolist()
    non_numeric = [column for column in df.columns if column not in numeric]

    aggregated = df[numeric].groupby(level=0).mean()
    if leak_label_column in df.columns:
        aggregated[leak_label_column] = df[leak_label_column].groupby(level=0).max()

    for column in non_numeric:
        if column != leak_label_column:
            aggregated[column] = df[column].groupby(level=0).first()

    return aggregated.sort_index()


def report_missing_values(df: pd.DataFrame) -> MissingValueReport:
    missing = df.isna().sum()
    fractions = df.isna().mean()
    return MissingValueReport(
        total_missing=int(missing.sum()),
        missing_by_column={column: int(value) for column, value in missing.items()},
        missing_fraction_by_column={
            column: float(value) for column, value in fractions.items()
        },
    )


def infer_pressure_sensors(columns: Iterable[str]) -> list[str]:
    sensors: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered.startswith("p") or "pressure" in lowered:
            sensors.append(column)
    return sensors


def infer_flow_sensors(columns: Iterable[str]) -> list[str]:
    sensors: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered.startswith(("f", "q")) or "flow" in lowered:
            sensors.append(column)
    return sensors


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Configured columns were not found: {missing}")
    return columns


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float,
) -> ChronologicalSplit:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dataframe must be sorted by timestamp before splitting.")
    if len(df) < 2:
        raise ValueError("At least two rows are required for a chronological split.")

    split_index = int(len(df) * train_fraction)
    split_index = min(max(split_index, 1), len(df) - 1)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    if not train.index.max() < test.index.min():
        raise ValueError("Chronological split failed: train overlaps test.")

    return ChronologicalSplit(train=train, test=test)
