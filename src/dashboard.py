from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

MODEL_NAMES = {
    "baseline_deduplicated": "Baseline · de-duplicated",
    "baseline_original": "Baseline · original",
    "ewma": "EWMA",
    "cusum": "CUSUM",
    "lightgbm_compact": "LightGBM classifier",
    "lightgbm_regression": "LightGBM regression",
}
SCORE_UNITS = {
    "baseline_original": "Normalized residual",
    "baseline_deduplicated": "Normalized residual",
    "ewma": "Smoothed residual",
    "cusum": "Cumulative standardized shift",
    "lightgbm_compact": "Onset-window probability",
    "lightgbm_regression": "Predicted rise (m³/h)",
}


@dataclass(frozen=True)
class ReplayResults:
    summary: pd.DataFrame
    folds: pd.DataFrame
    events: pd.DataFrame
    alarms: pd.DataFrame
    calibration: pd.DataFrame
    manifest: dict


def load_results(path: Path) -> ReplayResults:
    required = ["summary.csv", "fold_metrics.csv", "event_audit.csv", "alarm_audit.csv",
                "calibration_audit.csv", "run_manifest.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment artifacts: " + ", ".join(missing))
    summary, folds, events, alarms, calibration = [pd.read_csv(path / name) for name in required[:5]]
    for name in ("start", "end"):
        events[name] = pd.to_datetime(events[name], errors="raise")
    alarms["timestamp"] = pd.to_datetime(alarms["timestamp"], errors="raise")
    manifest = json.loads((path / required[-1]).read_text(encoding="utf-8"))
    return ReplayResults(summary, folds, events, alarms, calibration, manifest)


def load_timeline(path: Path, fold: str) -> pd.DataFrame:
    if fold not in {child.name for child in path.iterdir() if child.is_dir()}:
        raise ValueError("Unknown replay month.")
    frame = pd.read_csv(path / fold / "timeline.csv.gz", index_col="timestamp", parse_dates=True)
    if frame.empty or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Replay timeline must be nonempty, sorted and unique.")
    return frame


def event_window(timeline: pd.DataFrame, event: pd.Series | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if event is None:
        return timeline.index[0], timeline.index[-1]
    return max(timeline.index[0], event["start"] - pd.Timedelta(hours=24)), min(timeline.index[-1], event["start"] + pd.Timedelta(hours=48))


def replay_figure(timeline: pd.DataFrame, model: str, events: pd.DataFrame, alarms: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp, show_truth: bool, show_prediction: bool) -> go.Figure:
    frame = timeline.loc[start:end]
    if frame.empty:
        raise ValueError("Replay window has no observations.")
    if model not in MODEL_NAMES or model not in frame or model + "_threshold" not in frame:
        raise ValueError("Detector score is unavailable in this month.")
    fig = make_subplots(rows=2 if show_truth or show_prediction else 1, cols=1,
                        shared_xaxes=True, vertical_spacing=.13)
    fig.add_trace(go.Scatter(x=frame.index, y=frame[model], name="Detector score",
                            line=dict(color="#277d72", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame[model + "_threshold"], name="Alarm threshold",
                            line=dict(color="#b64b65", width=1.4, dash="dash")), row=1, col=1)
    visible_alarms = alarms.loc[alarms["timestamp"].between(start, end)]
    for detected, color, label in [(True, "#277d72", "Matched alarm"), (False, "#b64b65", "Unmatched alarm")]:
        selected = visible_alarms.loc[visible_alarms["status"].eq("matched") == detected]
        if not selected.empty:
            values = frame[model].reindex(pd.DatetimeIndex(selected["timestamp"]))
            fig.add_trace(go.Scatter(x=selected["timestamp"], y=values, mode="markers", name=label,
                                     marker=dict(color=color, size=10, symbol="diamond"),
                                     text=selected["status"], hovertemplate="%{x}<br>%{text}<extra>%{fullData.name}</extra>"), row=1, col=1)
    if show_truth:
        fig.add_trace(go.Scatter(x=frame.index, y=frame["total_leakage_m3_h"], name="Ground-truth total leak flow",
                                line=dict(color="#647387", width=1.6)), row=2, col=1)
        for _, event in events.loc[events["start"].between(start, end)].iterrows():
            if event["status"] != "excluded_carry_in":
                # Shape labels use strings: Plotly's datetime vline annotations can perform invalid arithmetic.
                fig.add_shape(type="line", x0=event["start"], x1=event["start"], y0=0, y1=1,
                              xref="x", yref="y domain", line=dict(color="#9b748d", dash="dot", width=1))
                fig.add_annotation(x=event["start"], y=1, xref="x", yref="y domain",
                                   text=event["pipe"], showarrow=False, yshift=10, font=dict(size=11, color="#74465e"))
    if show_prediction:
        fig.add_trace(go.Scatter(x=frame.index, y=frame["predicted_leakage_m3_h"], name="Regressor prediction",
                                line=dict(color="#aa7d24", width=1.6)), row=2, col=1)
    fig.update_yaxes(title_text=SCORE_UNITS[model], row=1, col=1)
    if show_truth or show_prediction:
        fig.update_yaxes(title_text="Leak flow (m³/h)", row=2, col=1)
    fig.update_layout(height=510 if show_truth or show_prediction else 330, margin=dict(l=15, r=15, t=45, b=15),
                      paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
                      legend=dict(orientation="h", y=-.18), font=dict(family="Arial", color="#20292d"))
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f1")
    fig.update_yaxes(showgrid=True, gridcolor="#edf0f1", zeroline=False)
    return fig


def comparison_figure(summary: pd.DataFrame) -> go.Figure:
    names = summary["model"].map(MODEL_NAMES)
    fig = go.Figure()
    for column, color in [("precision", "#277d72"), ("recall", "#b64b65"), ("f1", "#647387")]:
        fig.add_bar(y=names, x=summary[column], name=column.title(), orientation="h", marker_color=color)
    fig.update_layout(barmode="group", height=460, margin=dict(l=5, r=5, t=5, b=15),
                      paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.1),
                      xaxis=dict(range=[0, 1.02], title="Pooled metric", gridcolor="#edf0f1"),
                      yaxis=dict(autorange="reversed"))
    return fig
