from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pandas as pd

from src.agent.schemas import Alarm
from src.detection.baseline import ResidualBaselineDetector


def explain_snapshot(model: ResidualBaselineDetector, frame: pd.DataFrame, timestamp: pd.Timestamp,
                     alarm: Alarm | None = None) -> tuple[dict, pd.DataFrame]:
    """Explain exact score components using only measurements at or before the snapshot."""
    timestamp = pd.Timestamp(timestamp)
    if timestamp not in frame.index:
        raise ValueError("Explanation timestamp must be observed.")
    if alarm is not None and pd.Timestamp(alarm.timestamp) != timestamp:
        raise ValueError("Alarm and explanation timestamps must match.")
    context = frame.loc[frame.index <= timestamp].tail(6 * 12 + model.config.rolling_window)
    if len(context) < max(13, model.config.rolling_window):
        raise ValueError("Insufficient causal smoothing and flow-comparison history for an explanation.")
    scores = model.score(context)
    score = float(scores.aggregate_score.loc[timestamp])
    if alarm is not None and abs(alarm.score - score) > 1e-6 * max(1, abs(score)):
        raise ValueError("Recomputed evidence disagrees with the saved alarm score.")
    normalized = scores.sensor_scores.loc[timestamp]
    contributions = normalized / len(normalized)
    rows = []
    for sensor in contributions.sort_values(ascending=False).index:
        residual = float(scores.residuals.loc[timestamp, sensor])
        measured = float(context.loc[timestamp, sensor])
        rows.append({"sensor": sensor, "score_contribution": float(contributions[sensor]),
                     "score_share": float(contributions[sensor] / score) if score else 0.,
                     "smoothed_normalized_residual": float(normalized[sensor]),
                     "observed_pressure_m": measured, "expected_pressure_m": measured - residual,
                     "signed_residual_m": residual})
    trace = pd.DataFrame(index=context.tail(72).index)
    for sensor in [row["sensor"] for row in rows[:3]]:
        trace[sensor + "__observed"] = context.loc[trace.index, sensor]
        trace[sensor + "__expected"] = context.loc[trace.index, sensor] - scores.residuals.loc[trace.index, sensor]
    flow_rows = []
    for sensor in [name for name in model.feature_sensors if name.startswith("flow_")]:
        trace[sensor] = context.loc[trace.index, sensor]
        recent = float(context[sensor].tail(12).mean())
        previous = float(context[sensor].iloc[:-12].mean())
        flow_rows.append({"sensor": sensor, "recent_hour_mean_m3_h": recent,
                          "previous_context_mean_m3_h": previous, "change_m3_h": recent - previous})
    evidence = {"schema_version": 1, "kind": "alarm" if alarm else "diagnostic_snapshot",
                "alarm": {**asdict(alarm), "timestamp": alarm.timestamp.isoformat()} if alarm else None,
                "timestamp": timestamp.isoformat(), "score": score, "threshold": float(scores.threshold),
                "above_threshold": score > scores.threshold, "method": "exact additive residual score decomposition",
                "context_start": trace.index[0].isoformat(), "context_end": timestamp.isoformat(),
                "smoothing_samples": model.config.rolling_window, "sensors": rows, "flows": flow_rows,
                "causal": True, "localization_claim": False,
                "limitations": "Contributions explain this anomaly score, not its physical cause. Expected pressure is a conditional Ridge prediction, not a hydraulic leak-free counterfactual. Alarm score is not a leak probability."}
    evidence["evidence_sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return evidence, trace


def explain(alarm: Alarm, model: ResidualBaselineDetector, frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    return explain_snapshot(model, frame, pd.Timestamp(alarm.timestamp), alarm)
