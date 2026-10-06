from __future__ import annotations

import numpy as np
import pandas as pd

from src.detection.baseline import BaselineScores


def window_features(frame: pd.DataFrame, scores: BaselineScores, sensors: list[str], window: int) -> pd.DataFrame:
    if window < 1 or not frame.index.equals(scores.residuals.index):
        raise ValueError("Feature window must be positive and sensor/residual timestamps aligned.")
    residual = scores.residuals
    rolling = residual.rolling(window, min_periods=1)
    parts = [
        residual.add_prefix("residual__"),
        rolling.mean().add_prefix("residual_mean__"),
        rolling.std(ddof=0).add_prefix("residual_std__"),
        rolling.max().add_prefix("residual_max__"),
        residual.abs().rolling(window, min_periods=1).max().add_prefix("residual_abs_max__"),
        frame[sensors].diff().fillna(0).add_prefix("change__"),
    ]
    features = pd.concat(parts, axis=1)
    hour = frame.index.hour + frame.index.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["aggregate_residual"] = residual.abs().mean(axis=1)
    features["aggregate_score"] = scores.aggregate_score
    return features


def onset_window_labels(leakage: pd.DataFrame, window_hours: float) -> pd.Series:
    if not np.isfinite(window_hours) or window_hours <= 0:
        raise ValueError("The onset label window must be positive and finite.")
    positive = leakage.gt(0)
    onsets = positive & ~positive.shift(1, fill_value=False)
    # Existing leaks at the start of the source have an unknown onset.
    onsets.iloc[0] = False
    recent = onsets.astype(int).rolling(pd.Timedelta(hours=window_hours)).max().astype(bool)
    return (recent & positive).any(axis=1).astype(int).rename("new_leak_window")
