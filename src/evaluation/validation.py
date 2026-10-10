from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DevelopmentFold:
    name: str
    fit_index: pd.DatetimeIndex
    calibration_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def chronological_folds(index: pd.DatetimeIndex, starts: list[str], calibration_days: int, year: int) -> list[DevelopmentFold]:
    if year != 2018 or not (index.year == year).all():
        raise ValueError("Development validation accepts only 2018; 2019 is reserved for a frozen final test.")
    if len(index) < 2 or index.has_duplicates or index.hasnans or not index.is_monotonic_increasing:
        raise ValueError("Development timestamps must be valid, sorted, unique, and nonempty.")
    delta = index.to_series().diff().dropna()
    if not delta.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("Development validation requires uninterrupted five-minute observations.")
    boundaries = [pd.Timestamp(value) for value in starts]
    if not boundaries or boundaries != sorted(set(boundaries)) or calibration_days < 1:
        raise ValueError("Test starts must be unique and chronological; calibration_days must be positive.")
    folds = []
    previous_end = None
    for start in boundaries:
        if start.year != year or start.day != 1 or start != start.normalize():
            raise ValueError("Each outer block must start at midnight on the first day of a 2018 month.")
        end = start + pd.offsets.MonthBegin(1)
        if previous_end is not None and start < previous_end:
            raise ValueError("Outer blocks must not overlap.")
        calibration_start = start - pd.Timedelta(days=calibration_days)
        fit = index[index < calibration_start]
        calibration = index[(index >= calibration_start) & (index < start)]
        test = index[(index >= start) & (index < end)]
        if min(len(fit), len(calibration), len(test)) < 2:
            raise ValueError("Each fold needs fitting, calibration, and outer observations.")
        if calibration[0] != calibration_start or test[0] != start or test[-1] + delta.iloc[0] != end:
            raise ValueError("Calibration and outer calendar blocks must be complete.")
        if not fit[-1] < calibration[0] < test[0]:
            raise ValueError("Fitting, calibration, and outer periods overlap.")
        folds.append(DevelopmentFold(start.strftime("%Y-%m"), fit, calibration, test))
        previous_end = end
    return folds
