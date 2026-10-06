from __future__ import annotations

from dataclasses import dataclass

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd

from src.detection.alarms import create_persistent_alarms
from src.detection.baseline import BaselineConfig, ResidualBaselineDetector
from src.detection.features import window_features
from src.evaluation.metrics import LeakEvent, evaluate_alarms


@dataclass(frozen=True)
class ClassifierConfig:
    rolling_window: int = 6
    onset_window_hours: float = 24
    ridge_fraction: float = 0.5
    validation_fraction: float = 0.2
    persistence_hours: float = 3
    threshold: float | None = None
    threshold_candidates: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    n_estimators: int = 100
    num_leaves: int = 15
    learning_rate: float = 0.05
    random_state: int = 42

    def __post_init__(self):
        if not 0 < self.ridge_fraction < 1 or not 0 < self.validation_fraction < 1 or self.ridge_fraction + self.validation_fraction >= 1:
            raise ValueError("Ridge and validation fractions must leave a separate classifier training period.")
        BaselineConfig(rolling_window=self.rolling_window, persistence_hours=self.persistence_hours)
        if not np.isfinite(self.onset_window_hours) or self.onset_window_hours <= 0:
            raise ValueError("onset_window_hours must be positive and finite.")
        thresholds = [self.threshold] if self.threshold is not None else self.threshold_candidates
        if not thresholds or any(not np.isfinite(t) or not 0 < t < 1 for t in thresholds):
            raise ValueError("Probability thresholds must be between 0 and 1.")
        if self.n_estimators < 1 or self.num_leaves < 2 or not 0 < self.learning_rate <= 1:
            raise ValueError("Invalid LightGBM model parameters.")


@dataclass(frozen=True)
class ClassifierScores:
    probability: pd.Series
    features: pd.DataFrame
    sensor_scores: pd.DataFrame
    threshold: float


class LeakClassifier:
    def __init__(self, pressure_sensors: list[str], sensors: list[str], config: ClassifierConfig | None = None, ridge_alpha: float = 1.0):
        self.config = config or ClassifierConfig()
        self.baseline = ResidualBaselineDetector(pressure_sensors, sensors, BaselineConfig(rolling_window=self.config.rolling_window, alpha=ridge_alpha))
        self.sensors = sensors
        self.model: LGBMClassifier | None = None
        self.threshold_: float | None = None
        self.partitions_: dict = {}
        self.validation_results_: list[dict] = []

    def fit(self, train: pd.DataFrame, labels: pd.Series, events: list[LeakEvent]) -> "LeakClassifier":
        if not train.index.equals(labels.index) or not labels.isin([0, 1]).all():
            raise ValueError("Binary training labels must exactly align with training sensor data.")
        ridge_end = int(len(train) * self.config.ridge_fraction)
        validation_start = int(len(train) * (1 - self.config.validation_fraction))
        if min(ridge_end, validation_start - ridge_end, len(train) - validation_start) < 2:
            raise ValueError("At least two observations are needed in each chronological training section.")
        ridge_train = train.iloc[:ridge_end]
        classifier_index = train.index[ridge_end:validation_start]
        validation_index = train.index[validation_start:]
        self.baseline.fit(ridge_train)
        scores = self.baseline.score(train)
        features = window_features(train, scores, self.sensors, self.config.rolling_window)
        training_labels = labels.loc[classifier_index]
        if training_labels.nunique() != 2:
            raise ValueError("The classifier training period needs both new-leak and normal windows.")
        self.model = LGBMClassifier(
            n_estimators=self.config.n_estimators, num_leaves=self.config.num_leaves,
            learning_rate=self.config.learning_rate, random_state=self.config.random_state,
            deterministic=True, force_col_wise=True, n_jobs=1, verbosity=-1,
        )
        self.model.fit(features.loc[classifier_index], training_labels)
        probability = pd.Series(self.model.predict_proba(features.loc[validation_index])[:, 1], index=validation_index, name="probability")
        self.validation_results_ = []
        thresholds = [self.config.threshold] if self.config.threshold is not None else self.config.threshold_candidates
        for threshold in thresholds:
            alarms = create_persistent_alarms(probability, threshold, self.config.persistence_hours, scores.sensor_scores.loc[validation_index], "lightgbm", self.baseline.sample_interval_)
            result = evaluate_alarms(alarms, events, validation_index, "lightgbm", self.config.onset_window_hours)
            self.validation_results_.append({"threshold": threshold, **result.metrics})
        if self.config.threshold is None and not self.validation_results_[0]["eligible_leak_events"]:
            raise ValueError("Validation needs new leak events for threshold selection; choose a fixed training-derived threshold or revise the chronological split.")
        best = max(self.validation_results_, key=lambda row: (row["f1"], -row["false_positives"], row["threshold"]))
        self.threshold_ = float(best["threshold"])
        self.partitions_ = {
            "ridge_start": ridge_train.index[0].isoformat(), "ridge_end": ridge_train.index[-1].isoformat(),
            "classifier_start": classifier_index[0].isoformat(), "classifier_end": classifier_index[-1].isoformat(),
            "validation_start": validation_index[0].isoformat(), "validation_end": validation_index[-1].isoformat(),
            "ridge_rows": len(ridge_train), "classifier_rows": len(classifier_index), "validation_rows": len(validation_index),
            "classifier_positive_rows": int(training_labels.sum()), "validation_positive_rows": int(labels.loc[validation_index].sum()),
        }
        return self

    def score(self, frame: pd.DataFrame, history: pd.DataFrame | None = None) -> ClassifierScores:
        if self.model is None or self.threshold_ is None:
            raise RuntimeError("Classifier must be fitted before scoring.")
        if history is not None:
            if history.empty or history.index.max() >= frame.index.min():
                raise ValueError("Feature history must strictly precede scoring timestamps.")
            combined = pd.concat([history.tail(self.config.rolling_window), frame])
        else:
            combined = frame
        scores = self.baseline.score(combined)
        features = window_features(combined, scores, self.sensors, self.config.rolling_window).loc[frame.index]
        probability = pd.Series(self.model.predict_proba(features)[:, 1], index=frame.index, name="probability")
        return ClassifierScores(probability, features, scores.sensor_scores.loc[frame.index], self.threshold_)

    def create_alarms(self, scores: ClassifierScores):
        return create_persistent_alarms(scores.probability, scores.threshold, self.config.persistence_hours, scores.sensor_scores, "lightgbm", self.baseline.sample_interval_)
