from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_detectors import load_alarms, run_evaluation
from src.detection.baseline import BaselineConfig, BaselineScores
from src.evaluation.plots import plot_alarm_policy_comparison
from src.utils.config import load_config


def run(config: dict, root: Path) -> pd.DataFrame:
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    original_dir = resolve(config.get("baseline", {}).get("output_dir", "artifacts/baseline"))
    original_summary = json.loads((original_dir / "baseline_summary.json").read_text(encoding="utf-8"))
    score_frame = pd.read_csv(original_dir / "baseline_scores.csv", index_col="timestamp", parse_dates=True)
    sensor_scores = pd.read_csv(original_dir / "baseline_sensor_scores.csv", index_col="timestamp", parse_dates=True)
    if not sensor_scores.index.equals(score_frame.index):
        raise ValueError("Saved aggregate and sensor scores do not align.")
    if not score_frame["threshold"].eq(original_summary["threshold"]).all():
        raise ValueError("Saved threshold does not match the run manifest.")
    detector = joblib.load(original_dir / "baseline_model.joblib")
    old_config = dict(original_summary["baseline_config"])
    detector.config = BaselineConfig(**old_config)
    scores = BaselineScores(pd.DataFrame(index=score_frame.index), sensor_scores, score_frame["anomaly_score"], original_summary["threshold"])
    replay = detector.create_alarms(scores)
    original_alarms = load_alarms(original_dir / "baseline_alarms.json")
    if [(alarm.id, alarm.timestamp) for alarm in replay] != [(alarm.id, alarm.timestamp) for alarm in original_alarms]:
        raise ValueError("Saved-score replay differs from original alarms. Regenerate the original run before comparing policies.")
    if old_config.get("recovery_hours", 0) or old_config.get("cooldown_hours", 0):
        raise ValueError("The source run already has de-duplication enabled. Preserve a legacy run with both settings zero to compare.")
    settings = config.get("baseline", {})
    policy = {"recovery_hours": settings.get("recovery_hours", 24), "cooldown_hours": settings.get("cooldown_hours", 24)}
    detector.config = BaselineConfig(**{**old_config, **policy})
    alarms = detector.create_alarms(scores)
    output_dir = resolve("artifacts/baseline_deduplicated")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "baseline_deduplicated_model.joblib"
    joblib.dump(detector, model_path)
    records = [{**asdict(alarm), "timestamp": alarm.timestamp.isoformat()} for alarm in alarms]
    alarm_path = output_dir / "baseline_deduplicated_alarms.json"
    alarm_path.write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    summary = {
        **original_summary, "baseline_config": asdict(detector.config), "alarm_count": len(alarms),
        "model_path": str(model_path.resolve()), "source_run": str(original_dir.resolve()),
        "alarm_policy": "one notification until continuous recovery and minimum cooldown",
        "comparison_status": "exploratory replay of previously inspected 2018 holdout; not a fresh test",
        "retrained": False,
    }
    # The source plot contains legacy markers, so it must not be claimed as a new plot.
    summary.pop("plot_path", None)
    (output_dir / "baseline_deduplicated_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    evaluation_config = {**config, "evaluation": {**config.get("evaluation", {}), "output_dir": "artifacts/results/alarm_deduplication"}}
    comparison = run_evaluation(evaluation_config, root, {
        "baseline_original": original_dir / "baseline_alarms.json",
        "baseline_deduplicated": alarm_path,
    })
    report_path = resolve("artifacts/results/alarm_deduplication/comparison.md")
    with report_path.open("a", encoding="utf-8") as report:
        report.write("\nThis is an exploratory policy replay on the previously inspected 2018 holdout. Models, scores, detection thresholds, and initial persistence are unchanged. Only incident recovery and cooldown settings differ. Policy choices were fixed before this replay, but results are not a fresh estimate of generalization.\n")
    evaluation = json.loads((report_path.parent / "evaluation.json").read_text(encoding="utf-8"))
    plot_path = report_path.parent / "alarm_policy_comparison.png"
    plot_alarm_policy_comparison(scores.aggregate_score, scores.threshold, original_alarms, alarms, evaluation["baseline_original"]["events"], plot_path)
    retained_ids = {alarm.id for alarm in alarms}
    audit = evaluation["baseline_original"]["alarms"]
    notifications = pd.DataFrame([
        {"timestamp": alarm.timestamp.isoformat(), "notification": "Retained" if alarm.id in retained_ids else "Suppressed", "original_match": row["status"], "score": round(alarm.score, 3), "sensors": ", ".join(alarm.affected_sensors)}
        for alarm, row in zip(sorted(original_alarms, key=lambda item: (item.timestamp, item.id)), audit)
    ], columns=["timestamp", "notification", "original_match", "score", "sensors"])
    notifications.to_csv(report_path.parent / "notification_audit.csv", index=False)
    image_data = base64.b64encode(plot_path.read_bytes()).decode("ascii")
    display = comparison[["model", "alarm_count", "precision", "recall", "avg_detection_delay_hours", "duplicate_alarms"]].copy()
    display["model"] = ["Original baseline", "De-duplicated baseline"]
    display.columns = ["Detector", "Notifications", "Precision", "Recall", "Mean delay (hours)", "Repeat alerts"]
    notification_table = notifications.rename(columns={"timestamp": "Timestamp", "notification": "Notification", "original_match": "Original match", "score": "Score", "sensors": "Sensor hints"})
    report_html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Water Watch | Alarm Comparison</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:#222;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1200px;margin:auto;padding:24px}}h1{{font-size:26px;margin:0 0 8px}}h2{{font-size:19px;margin:28px 0 12px}}p{{max-width:850px}}img{{display:block;width:100%;height:auto}}.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px 12px;border-bottom:1px solid #ddd;white-space:nowrap}}th{{background:#f1f3f4}}.status{{color:#526056}}@media(max-width:600px){{main{{padding:14px}}h1{{font-size:23px}}}}
</style></head><body><main>
<h1>Water Watch: Alarm Comparison</h1>
<p class="status">Completed replay | {html.escape(scores.aggregate_score.index[0].isoformat())} to {html.escape(scores.aggregate_score.index[-1].isoformat())}</p>
<p>Same model, scores, threshold, and initial persistence. An incident now closes after {policy["recovery_hours"]} continuous normal hours; the minimum notification cooldown is {policy["cooldown_hours"]} hours.</p>
<div class="table">{display.to_html(index=False, border=0, float_format=lambda value: f"{value:.3f}")}</div>
<h2>Score and Notifications</h2>
<img src="data:image/png;base64,{image_data}" alt="Anomaly score with unchanged threshold, new leak intervals, retained notifications, and suppressed notifications">
<h2>Notification Audit</h2><div class="table">{notification_table.to_html(index=False, border=0)}</div>
<p>This replay uses two previously inspected 2018 events. Results are exploratory, not a fresh accuracy estimate. Sensor hints do not prove pipe location. Distinct leaks during an open incident may be suppressed.</p>
<p>EWMA/CUSUM, blocked validation, and revised LightGBM targets remain pending.</p>
</main></body></html>'''
    (report_path.parent / "report.html").write_text(report_html, encoding="utf-8")
    with report_path.open("a", encoding="utf-8") as report:
        report.write(f"\n![Alarm notification comparison]({plot_path.resolve().as_posix()})\n")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare legacy and de-duplicated alarms using frozen baseline scores; preserve the original run.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    path = args.config.resolve()
    try:
        comparison = run(load_config(path), path.parent)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    columns = ["model", "alarm_count", "precision", "recall", "f1", "avg_detection_delay_hours", "false_alarms_per_week", "duplicate_alarms"]
    print(comparison[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
