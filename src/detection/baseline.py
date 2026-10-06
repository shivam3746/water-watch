from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.agent.schemas import Alarm
from src.detection.alarms import create_persistent_alarms


@dataclass(frozen=True)
class BaselineConfig:
    rolling_window: int = 6
    threshold: float | None = None
    persistence_hours: float = 3
    alpha: float = 1.0
    threshold_quantile: float = 0.995
    recovery_hours: float = 0
    cooldown_hours: float = 0

    def __post_init__(self) -> None:
        if not isinstance(self.rolling_window, int) or self.rolling_window < 1:
            raise ValueError("rolling_window must be a positive integer (samples).")
        if not np.isfinite(self.persistence_hours) or self.persistence_hours < 0:
            raise ValueError("persistence_hours must be finite and nonnegative.")
        if any(not np.isfinite(value) or value < 0 for value in (self.recovery_hours, self.cooldown_hours)):
            raise ValueError("recovery_hours and cooldown_hours must be finite and nonnegative.")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must be between 0 and 1.")
        if self.threshold is not None and (not np.isfinite(self.threshold) or self.threshold < 0):
            raise ValueError("threshold must be finite and nonnegative, or null.")


@dataclass(frozen=True)
class BaselineScores:
    residuals: pd.DataFrame
    sensor_scores: pd.DataFrame
    aggregate_score: pd.Series
    threshold: float


class ResidualBaselineDetector:
    def __init__(
        self,
        pressure_sensors: list[str],
        feature_sensors: list[str],
        config: BaselineConfig | None = None,
    ) -> None:
        if not pressure_sensors:
            raise ValueError("At least one pressure sensor is required.")
        if not feature_sensors:
            raise ValueError("At least one feature sensor is required.")

        self.pressure_sensors = pressure_sensors
        self.feature_sensors = feature_sensors
        self.config = config or BaselineConfig()
        self.models: dict[str, Ridge] = {}
        self.residual_scale_: pd.Series | None = None
        self.threshold_: float | None = None
        self.sample_interval_: pd.Timedelta | None = None

    def fit(self, train: pd.DataFrame) -> "ResidualBaselineDetector":
        self._validate_columns(train, [*self.pressure_sensors, *self.feature_sensors])
        if len(train) < 2:
            raise ValueError("At least two training samples are required.")
        self.sample_interval_ = train.index.to_series().diff().dropna().median()
        self.models = {}
        self.threshold_ = None
        residuals = {}

        for sensor in self.pressure_sensors:
            features = self._feature_frame(train, excluded_sensor=sensor)
            target = train[sensor]
            model = Ridge(alpha=self.config.alpha)
            model.fit(features, target)
            self.models[sensor] = model
            residuals[sensor] = target - model.predict(features)

        residual_frame = pd.DataFrame(residuals, index=train.index)
        scale = residual_frame.std().clip(lower=1e-8)
        self.residual_scale_ = scale

        train_scores = self._score_residuals(residual_frame)
        self.threshold_ = (
            self.config.threshold
            if self.config.threshold is not None
            else float(train_scores.aggregate_score.quantile(self.config.threshold_quantile))
        )
        return self

    def score(self, frame: pd.DataFrame) -> BaselineScores:
        if not self.models or self.residual_scale_ is None or self.threshold_ is None:
            raise RuntimeError("Detector must be fitted before scoring.")

        self._validate_columns(frame, [*self.pressure_sensors, *self.feature_sensors])
        residuals = {}
        for sensor, model in self.models.items():
            features = self._feature_frame(frame, excluded_sensor=sensor)
            residuals[sensor] = frame[sensor] - model.predict(features)

        residual_frame = pd.DataFrame(residuals, index=frame.index)
        scores = self._score_residuals(residual_frame)
        return BaselineScores(
            residuals=residual_frame,
            sensor_scores=scores.sensor_scores,
            aggregate_score=scores.aggregate_score,
            threshold=self.threshold_,
        )

    def create_alarms(self, scores: BaselineScores) -> list[Alarm]:
        return create_persistent_alarms(
            scores.aggregate_score, scores.threshold, self.config.persistence_hours,
            scores.sensor_scores, "baseline", self.sample_interval_,
            recovery_hours=self.config.recovery_hours, cooldown_hours=self.config.cooldown_hours,
        )

    def _score_residuals(self, residuals: pd.DataFrame) -> BaselineScores:
        if self.residual_scale_ is None:
            raise RuntimeError("Residual scale is unavailable before fitting.")

        normalized = residuals.abs().divide(self.residual_scale_, axis="columns")
        sensor_scores = normalized.rolling(
            window=self.config.rolling_window,
            min_periods=1,
        ).mean()
        aggregate = sensor_scores.mean(axis=1)
        threshold = self.threshold_ if self.threshold_ is not None else float("nan")
        return BaselineScores(
            residuals=residuals,
            sensor_scores=sensor_scores,
            aggregate_score=aggregate,
            threshold=threshold,
        )

    def _feature_frame(self, frame: pd.DataFrame, excluded_sensor: str) -> pd.DataFrame:
        feature_columns = [column for column in self.feature_sensors if column != excluded_sensor]
        if not feature_columns:
            raise ValueError(
                f"No feature columns remain after excluding target sensor '{excluded_sensor}'."
            )

        features = frame[feature_columns].copy()
        hour = frame.index.hour + frame.index.minute / 60.0
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        return features

    @staticmethod
    def _validate_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
            raise ValueError("Sensor data must have a nonempty DatetimeIndex.")
        if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError("Timestamps must be valid, sorted, and unique.")
        columns = list(dict.fromkeys(columns))
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        try:
            finite = np.isfinite(frame[columns].to_numpy(dtype=float)).all()
        except (ValueError, TypeError) as exc:
            raise ValueError("Sensor columns must be numeric.") from exc
        if not finite:
            raise ValueError("Sensor data contains missing/infinite values. Clean these before running the baseline.")
