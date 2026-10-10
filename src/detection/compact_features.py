from __future__ import annotations

import pandas as pd

from src.detection.baseline import BaselineScores


def compact_features(frame: pd.DataFrame, scores: BaselineScores, pressure: list[str], flow: list[str]) -> pd.DataFrame:
    """Fixed sensor-independent statistics, never selected from outer labels."""
    normalized = scores.sensor_scores
    signed = scores.residuals
    features = pd.DataFrame(index=frame.index)
    for name, series in {
        "residual_abs_mean": normalized.mean(axis=1), "residual_abs_max": normalized.max(axis=1),
        "residual_abs_median": normalized.median(axis=1), "residual_abs_std": normalized.std(axis=1, ddof=0),
        "residual_abs_p90": normalized.quantile(0.9, axis=1), "residual_signed_mean": signed.mean(axis=1),
        "residual_signed_min": signed.min(axis=1), "residual_signed_max": signed.max(axis=1),
        "pressure_change_mean": frame[pressure].diff().fillna(0).mean(axis=1),
        "pressure_change_min": frame[pressure].diff().fillna(0).min(axis=1),
        "pressure_change_max": frame[pressure].diff().fillna(0).max(axis=1),
        "flow_change_mean": frame[flow].diff().fillna(0).mean(axis=1),
        "flow_change_max": frame[flow].diff().fillna(0).max(axis=1),
    }.items():
        features[name] = series
    for samples in (6, 72):
        rolling = features["residual_abs_mean"].rolling(samples, min_periods=1)
        features[f"mean_{samples}"] = rolling.mean()
        features[f"std_{samples}"] = rolling.std(ddof=0)
        features[f"max_{samples}"] = rolling.max()
    features["hour"] = frame.index.hour + frame.index.minute / 60
    features["day_of_week"] = frame.index.dayofweek
    return features


def known_onset_labels(leakage: pd.DataFrame, hours: float) -> pd.Series:
    from src.detection.features import onset_window_labels

    recent = onset_window_labels(leakage, hours).astype(bool)
    labels = pd.Series(float("nan"), index=leakage.index, name="new_leak_window")
    labels.loc[~leakage.gt(0).any(axis=1)] = 0
    labels.loc[recent] = 1
    return labels


def predicted_rise(prediction: pd.Series) -> pd.Series:
    smooth = prediction.rolling("6h", min_periods=1).mean()
    previous = smooth.shift(freq="24h").reindex(smooth.index)
    return (smooth - previous).clip(lower=0).fillna(0).rename("predicted_leakage_rise")
