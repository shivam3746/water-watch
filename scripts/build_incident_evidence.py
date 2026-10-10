from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.schemas import Alarm
from src.data.loader import load_sensor_dataset
from src.explanation.residual_explainer import explain_snapshot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_evidence(evidence, trace, path):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), constrained_layout=True)
    top = evidence["sensors"][:5]
    axes[0].barh([row["sensor"] for row in top][::-1], [row["score_contribution"] for row in top][::-1], color="#277d72")
    axes[0].set_title("Exact additive score contributions | Top five sensors")
    axes[0].set_xlabel("Contribution to aggregate normalized residual score")
    for axis, row in zip(axes[1:], top[:3]):
        sensor = row["sensor"]
        axis.plot(trace.index, trace[sensor + "__observed"], label="Measured", color="#277d72")
        axis.plot(trace.index, trace[sensor + "__expected"], label="Ridge expected", color="#b64b65", linestyle="--")
        axis.set_ylabel("Pressure (m)")
        axis.set_title(sensor, fontsize=11)
        axis.legend(frameon=False, loc="best")
        axis.grid(alpha=.15)
    fig.suptitle(f'{evidence["kind"]} | {evidence["timestamp"]}', fontsize=13)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build():
    results = ROOT / "artifacts/results/validation_v1"
    output = ROOT / "artifacts/explanations"
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/processed/battledim.csv"
    sensors = load_sensor_dataset(source).frame.drop(columns=["leak"])
    audits = pd.read_csv(results / "alarm_audit.csv")
    events = pd.read_csv(results / "event_audit.csv")
    selected = audits.loc[audits.model == "baseline_deduplicated"].sort_values("timestamp")
    incidents = []
    cases = []
    models = {}

    def save(fold, timestamp, alarm, identifier):
        model_path = results / fold / "baseline_model.joblib"
        if fold not in models:
            models[fold] = joblib.load(model_path)
        evidence, trace = explain_snapshot(models[fold], sensors, pd.Timestamp(timestamp), alarm)
        (output / f"{identifier}.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
        trace.to_csv(output / f"{identifier}.csv")
        plot_evidence(evidence, trace, output / f"{identifier}.png")
        return {"id": identifier, "fold": fold, "timestamp": evidence["timestamp"],
                "evidence_file": f"{identifier}.json", "trace_file": f"{identifier}.csv", "plot_file": f"{identifier}.png",
                "model_sha256": digest(model_path)}

    for row in selected.itertuples():
        saved = json.loads((results / row.fold / "baseline_deduplicated_alarms.json").read_text())
        raw = next(alarm for alarm in saved if alarm["id"] == row.alarm_id)
        alarm = Alarm(**{**raw, "timestamp": pd.Timestamp(raw["timestamp"]).to_pydatetime()})
        record = save(row.fold, row.timestamp, alarm, row.alarm_id)
        incidents.append(record)
        category = "detected_event" if row.status == "matched" else "unmatched_alert"
        if not any(case["category"] == category for case in cases):
            cases.append({**record, "category": category, "evaluation_status": row.status,
                          "event_id": row.event_id if pd.notna(row.event_id) else None,
                          "interpretation": "Retrospective temporal match only." if row.status == "matched" else "Unmatched new-onset notification, not proof that no leak was present."})
    misses = events.loc[(events.model == "baseline_deduplicated") & (events.status == "missed")].sort_values("start")
    if not misses.empty:
        row = misses.iloc[0]
        # Fixed end-of-window snapshot, not the highest score selected after inspecting the event.
        timestamp = min(pd.Timestamp(row["start"]) + pd.Timedelta(hours=48), pd.Timestamp(row["end"]), pd.Timestamp(row["fold"]) + pd.offsets.MonthBegin(1)) - pd.Timedelta(minutes=5)
        record = save(row["fold"], timestamp, None, "missed_event_snapshot")
        cases.append({**record, "category": "missed_event", "evaluation_status": "missed", "event_id": row["event_id"],
                      "interpretation": "Retrospective diagnostic at the end of the allowed detection window. No alarm existed and this is not an operator notification."})
    (output / "incidents.json").write_text(json.dumps(incidents, indent=2), encoding="utf-8")
    (output / "cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    manifest = {"source_sha256": digest(source), "baseline_only": True, "2019_used": False,
                "case_rule": "First chronological matched notification, unmatched notification, and missed event in development audits.",
                "explanation_sha256": digest(ROOT / "src/explanation/residual_explainer.py"), "incidents": len(incidents), "cases": len(cases)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(incidents)} real notification explanations and {len(cases)} case studies to {output}")


if __name__ == "__main__":
    build()
