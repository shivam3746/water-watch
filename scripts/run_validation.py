from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.battledim import read_battledim_csv
from src.data.loader import load_sensor_dataset
from src.detection.alarms import create_persistent_alarms
from src.detection.baseline import BaselineConfig, ResidualBaselineDetector
from src.detection.change import cusum, ewma
from src.detection.compact_features import compact_features, known_onset_labels, predicted_rise
from src.evaluation.metrics import evaluate_alarms, extract_leak_events
from src.evaluation.validation import chronological_folds
from src.evaluation.validation_report import write_validation_report
from src.utils.config import load_config


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def summarize(fold_metrics: pd.DataFrame, event_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, all_folds in fold_metrics.groupby("model", sort=False):
        folds = all_folds.loc[all_folds["status"] == "evaluated"]
        if folds.empty:
            rows.append({"model": model, "evaluated_folds": 0, "unavailable_folds": len(all_folds)})
            continue
        tp, fp, fn = (int(folds[key].sum()) for key in ("true_positives", "false_positives", "false_negatives"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recalls = folds.loc[folds["eligible_leak_events"] > 0, "recall"]
        precisions = folds.loc[folds["alarm_count"] > 0, "precision"]
        delays = event_rows.loc[(event_rows["model"] == model) & (event_rows["status"] == "detected"), "delay_hours"].astype(float)
        rows.append({
            "model": model, "evaluated_folds": len(folds), "unavailable_folds": len(all_folds) - len(folds),
            "eligible_events": tp + fn, "detected_events": tp, "missed_events": fn, "false_alerts": fp,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_alarms_per_week": fp / float(folds["observed_weeks"].sum()),
            "fold_recall_mean": recalls.mean(), "fold_recall_std": recalls.std(),
            "fold_precision_mean": precisions.mean(), "fold_precision_std": precisions.std(),
            "detection_delay_mean_hours": delays.mean(), "detection_delay_std_hours": delays.std(),
            "uninformative_calibration_folds": int((~folds["calibration_informative"].astype(bool)).sum()),
        })
    return pd.DataFrame(rows)


def run(protocol_path: Path, root: Path = PROJECT_ROOT, only_folds: list[str] | None = None) -> pd.DataFrame:
    config = load_config(protocol_path)
    resolve = lambda value: Path(value) if Path(value).is_absolute() else root / value
    folds_subset = set(only_folds or [])
    dataset = load_sensor_dataset(resolve(config["data_path"]))
    leakage = read_battledim_csv(resolve(config["leakage_path"]))
    if not leakage.index.equals(dataset.frame.index):
        raise ValueError("Ground truth and sensor timestamps must match exactly.")
    if not leakage.gt(0).any(axis=1).astype(int).equals(dataset.frame[dataset.leak_label_column]):
        raise ValueError("Merged leak labels disagree with source ground truth.")
    folds = chronological_folds(dataset.frame.index, config["test_starts"], config["calibration_days"], config["year"])
    if folds_subset:
        if not folds_subset.issubset({fold.name for fold in folds}):
            raise ValueError("Requested fold is not in the frozen protocol.")
        folds = [fold for fold in folds if fold.name in folds_subset]
    output = resolve(config["output_dir"])
    if folds_subset:
        output = output / ("subset_" + "_".join(sorted(folds_subset)))
    output.mkdir(parents=True, exist_ok=True)
    protocol_bytes = protocol_path.read_bytes()
    (output / "protocol_snapshot.yaml").write_bytes(protocol_bytes)
    manifest = {
        "protocol_id": config["protocol_id"], "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "status": config["status"], "selected_folds": [fold.name for fold in folds],
        "input_sha256": {key: digest(resolve(config[key])) for key in ("data_path", "leakage_path")},
        "source_sha256": {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in [PROJECT_ROOT / "scripts/run_validation.py", *sorted((PROJECT_ROOT / "src").rglob("*.py"))]},
        "python_version": sys.version.split()[0], "2019_loaded": False,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f'Protocol frozen: {manifest["protocol_sha256"]}', flush=True)
    sensors = dataset.frame[dataset.sensor_columns]
    reference_end = sensors.index[0] + pd.Timedelta(days=config["reference_days"])
    reference = sensors.loc[sensors.index < reference_end]
    if reference.index[-1] >= folds[0].fit_index[-1]:
        raise ValueError("The supervised reference must precede supervised fitting rows.")
    baseline_settings = config["baseline"]
    base_config = BaselineConfig(rolling_window=config["rolling_window"], alpha=baseline_settings["alpha"], threshold_quantile=baseline_settings["threshold_quantile"], persistence_hours=baseline_settings["persistence_hours"])
    print("Preparing fixed causal supervised features...", flush=True)
    feature_reference = ResidualBaselineDetector(dataset.pressure_sensors, dataset.sensor_columns, base_config).fit(reference)
    reference_scores = feature_reference.score(sensors)
    features = compact_features(sensors, reference_scores, dataset.pressure_sensors, dataset.flow_sensors)
    joblib.dump(feature_reference, output / "feature_reference.joblib")
    labels = known_onset_labels(leakage, config["supervised"]["onset_window_hours"])
    target = leakage.sum(axis=1).rename("total_leakage_m3_h")
    events = extract_leak_events(leakage)
    metrics, audit_events, audit_alarms, calibrations = [], [], [], []
    supervised = config["supervised"]
    model_parameters = {key: supervised[key] for key in ("n_estimators", "max_depth", "num_leaves", "min_child_samples", "reg_alpha", "reg_lambda", "learning_rate", "random_state")}
    model_parameters.update(n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)

    for fold in folds:
        print(f"[{fold.name}] Fitting prefix and calibration candidates...", flush=True)
        fold_output = output / fold.name
        fold_output.mkdir(exist_ok=True)
        baseline = ResidualBaselineDetector(dataset.pressure_sensors, dataset.sensor_columns, base_config).fit(sensors.loc[fold.fit_index])
        joblib.dump(baseline, fold_output / "baseline_model.joblib")
        available_index = sensors.index[sensors.index <= fold.test_index[-1]]
        scores = baseline.score(sensors.loc[available_index])
        notification_index = fold.calibration_index.append(fold.test_index)
        metadata = {
            "fold": fold.name, "fit_start": fold.fit_index[0].isoformat(), "fit_end": fold.fit_index[-1].isoformat(),
            "calibration_start": fold.calibration_index[0].isoformat(), "calibration_end": fold.calibration_index[-1].isoformat(),
            "outer_start": fold.test_index[0].isoformat(), "outer_end": fold.test_index[-1].isoformat(),
        }
        timeline = pd.DataFrame(index=fold.test_index)
        timeline["baseline_score"] = scores.aggregate_score.loc[fold.test_index]
        timeline["total_leakage_m3_h"] = target.loc[fold.test_index]

        def alarm_sequence(chart, threshold, model, persistence, dedup, index):
            return create_persistent_alarms(
                chart.loc[index], threshold, persistence, scores.sensor_scores.loc[index], model,
                baseline.sample_interval_, config["recovery_hours"] if dedup else 0,
                config["cooldown_hours"] if dedup else 0,
            )

        def evaluate_candidate(chart, threshold, model, persistence, dedup):
            alarms = alarm_sequence(chart, threshold, model, persistence, dedup, fold.calibration_index)
            return evaluate_alarms(alarms, events, fold.calibration_index, model, config["max_detection_delay_hours"])

        def complete(model, candidates, persistence=0, dedup=True, saved_model=None, extra=None):
            candidates_with_metrics = []
            for number, (parameters, chart, threshold) in enumerate(candidates):
                result = evaluate_candidate(chart, threshold, model, persistence, dedup)
                record = {**metadata, "model": model, "candidate": number, "parameters": parameters, "threshold": float(threshold), **result.metrics}
                calibrations.append(record)
                candidates_with_metrics.append((record, chart, threshold))
            def rank(item):
                row = item[0]
                delay = row["avg_detection_delay_hours"]
                return row["f1"], -row["false_positives"], -(delay if delay is not None else float("inf")), -row["candidate"]
            selected, chart, threshold = max(candidates_with_metrics, key=rank)
            sequence = alarm_sequence(chart, threshold, model, persistence, dedup, notification_index)
            outer_alarms = [alarm for alarm in sequence if pd.Timestamp(alarm.timestamp) in fold.test_index]
            result = evaluate_alarms(outer_alarms, events, fold.test_index, model, config["max_detection_delay_hours"])
            informative = selected["eligible_leak_events"] > 0 and any(row[0]["true_positives"] > 0 for row in candidates_with_metrics)
            row = {**metadata, **result.metrics, "status": "evaluated", "selected_parameters": selected["parameters"], "selected_threshold": float(threshold), "calibration_informative": informative, **(extra or {})}
            metrics.append(row)
            for event in result.events:
                event_start = pd.Timestamp(event["start"])
                event_end = min(pd.Timestamp(event["end"]), event_start + pd.Timedelta(hours=config["max_detection_delay_hours"]))
                audit_events.append({"fold": fold.name, "model": model, **event, "detection_window_truncated": event["status"] != "excluded_carry_in" and event_end > fold.test_index[-1] + baseline.sample_interval_})
            audit_alarms.extend({"fold": fold.name, "model": model, **alarm} for alarm in result.alarms)
            timeline[model] = chart.loc[fold.test_index]
            timeline[model + "_threshold"] = float(threshold)
            alarms_path = fold_output / f"{model}_alarms.json"
            alarms_path.write_text(json.dumps([{**asdict(alarm), "timestamp": alarm.timestamp.isoformat()} for alarm in outer_alarms], indent=2), encoding="utf-8")
            (fold_output / f"{model}_manifest.json").write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
            if saved_model is not None:
                joblib.dump(saved_model, fold_output / f"{model}_model.joblib")
            print(f'  {model}: {row["true_positives"]}/{row["eligible_leak_events"]} new events, {row["false_positives"]} false alerts', flush=True)

        baseline_candidate = [({"quantile": base_config.threshold_quantile}, scores.aggregate_score, baseline.threshold_)]
        complete("baseline_original", baseline_candidate, base_config.persistence_hours, False)
        complete("baseline_deduplicated", baseline_candidate, base_config.persistence_hours)
        ewma_candidates = []
        for alpha in config["ewma"]["alphas"]:
            chart = ewma(scores.aggregate_score, alpha)
            for quantile in config["ewma"]["quantiles"]:
                ewma_candidates.append(({"alpha": alpha, "quantile": quantile}, chart, float(chart.loc[fold.fit_index].quantile(quantile))))
        complete("ewma", ewma_candidates)
        center = float(scores.aggregate_score.loc[fold.fit_index].mean())
        scale = max(float(scores.aggregate_score.loc[fold.fit_index].std()), 1e-8)
        cusum_candidates = []
        for shift in config["cusum"]["reference_shifts"]:
            chart = cusum(scores.aggregate_score, center, scale, shift)
            for limit in config["cusum"]["decision_limits"]:
                cusum_candidates.append(({"reference_shift": shift, "decision_limit": limit, "center": center, "scale": scale}, chart, limit))
        complete("cusum", cusum_candidates)
        supervised_index = fold.fit_index[fold.fit_index >= reference_end]
        training_labels = labels.loc[supervised_index]
        known = training_labels.dropna().astype(int)
        counts = {"known_negative_rows": int(known.eq(0).sum()), "known_positive_rows": int(known.eq(1).sum()), "unknown_rows_excluded": int(training_labels.isna().sum()), "feature_count": features.shape[1]}
        prediction_index = available_index[available_index >= reference_end]
        if known.nunique() == 2:
            model = LGBMClassifier(**model_parameters)
            weights = known.map({value: len(known) / (2 * count) for value, count in known.value_counts().items()})
            model.fit(features.loc[known.index], known, sample_weight=weights)
            probability = pd.Series(model.predict_proba(features.loc[prediction_index])[:, 1], index=prediction_index)
            candidates = [({"probability_threshold": threshold}, probability, threshold) for threshold in supervised["probability_thresholds"]]
            complete("lightgbm_compact", candidates, supervised["persistence_hours"], saved_model=model, extra=counts)
        else:
            metrics.append({**metadata, "model": "lightgbm_compact", "status": "unavailable_training_classes", **counts})
            print("  lightgbm_compact unavailable: usable fitting labels lack a class", flush=True)
        regressor = LGBMRegressor(**model_parameters)
        regressor.fit(features.loc[supervised_index], target.loc[supervised_index])
        prediction = pd.Series(regressor.predict(features.loc[prediction_index]), index=prediction_index).clip(lower=0)
        chart = predicted_rise(prediction)
        positive_training_rise = chart.loc[supervised_index].loc[lambda values: values > 0]
        thresholds = [(q, float(positive_training_rise.quantile(q)) if len(positive_training_rise) else 0.0) for q in supervised["rise_quantiles"]]
        extra = {
            "feature_count": features.shape[1],
            "regression_mae_m3_h": float(mean_absolute_error(target.loc[fold.test_index], prediction.loc[fold.test_index])),
            "regression_rmse_m3_h": float(np.sqrt(mean_squared_error(target.loc[fold.test_index], prediction.loc[fold.test_index]))),
            "constant_rmse_m3_h": float(np.sqrt(mean_squared_error(target.loc[fold.test_index], np.full(len(fold.test_index), target.loc[supervised_index].median())))),
            "regression_target_units": "m3/h",
        }
        complete("lightgbm_regression", [({"rise_quantile": q}, chart, threshold) for q, threshold in thresholds], supervised["persistence_hours"], saved_model=regressor, extra=extra)
        timeline["predicted_leakage_m3_h"] = prediction.loc[fold.test_index]
        timeline.to_csv(fold_output / "timeline.csv.gz", compression="gzip")

    metrics_frame = pd.DataFrame(metrics)
    events_frame = pd.DataFrame(audit_events)
    summary = summarize(metrics_frame, events_frame)
    metrics_frame.to_csv(output / "fold_metrics.csv", index=False)
    events_frame.to_csv(output / "event_audit.csv", index=False)
    pd.DataFrame(audit_alarms).to_csv(output / "alarm_audit.csv", index=False)
    pd.DataFrame(calibrations).to_csv(output / "calibration_audit.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    report = ["# Chronological Detector Experiments", "", "Retrospective 2018 development validation, not a fresh test. The original holdout was already inspected. 2019 was not loaded.", "", f'Protocol SHA-256: `{manifest["protocol_sha256"]}`', "", "| Detector | Evaluated Folds | Events | Detected | Missed | Precision | Recall | F1 | False Alerts/Week | Mean Delay (h) |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in json.loads(summary.to_json(orient="records")):
        if not row["evaluated_folds"]:
            report.append(f'| {row["model"]} | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
            continue
        delay = f'{row["detection_delay_mean_hours"]:.2f}' if row["detection_delay_mean_hours"] is not None else "N/A"
        report.append(f'| {row["model"]} | {row["evaluated_folds"]} | {row["eligible_events"]} | {row["detected_events"]} | {row["missed_events"]} | {row["precision"]:.3f} | {row["recall"]:.3f} | {row["f1"]:.3f} | {row["false_alarms_per_week"]:.3f} | {delay} |')
    report += ["", "## Interpretation", "", "Folds evaluate different months of one simulated network; they are not independent network samples. Carry-in events are excluded from onset recall. Alert state carries from calibration into each outer month; already-open incidents may suppress new events. Detections are timing matches, not proven pipe identification. False alerts mean unmatched new-event notifications, not proven absence of all leakage.", "", "Macro mean/std and pooled detected-event delay spread are in summary.csv; misses remain explicit rather than assigned zero delay. Calibration audits flag no-event or all-zero-detection blocks. Unknown ongoing-leak classifier rows are excluded; usable negative examples can be scarce. Continuous-flow regression error is separate from event-detection quality.", "", "Source models use different Ridge fitting references: the baselines use each fold's fitting prefix, supervised features use the fixed first seven days. Feature count is 21 rather than the original 205. Shallow regularized supervised models are experimental and do not guarantee improved generalization.", "", "See docs/VALIDATION_PROTOCOL_V1.md and the saved protocol/manifest for the complete procedure. No final detector has been approved for deployment from these development scores alone."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_validation_report(output, summary, events_frame)
    print(f"Saved reports to {output}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen chronological development experiments on 2018 only.")
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs/validation_v1.yaml")
    parser.add_argument("--fold", action="append", help="Run a named subset such as 2018-05, in a separate output folder.")
    args = parser.parse_args()
    try:
        summary = run(args.protocol.resolve(), only_folds=args.fold)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
