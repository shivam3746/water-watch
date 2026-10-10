"""Post-test diagnostics: preserve every frozen prediction and evaluation byte."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.battledim import read_battledim_csv
from src.data.loader import load_sensor_dataset
from src.evaluation.metrics import extract_leak_events
from src.evaluation.post_test import compare_sensors, event_size_rows, notification_trace


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(output: Path) -> dict:
    if output.exists():
        raise ValueError("Diagnostic output already exists; select a new --output directory.")
    final = ROOT / "artifacts/results/final_2019"
    protected = {path: sha(path) for path in final.iterdir() if path.is_file()}
    evaluation = json.loads((final / "evaluation.json").read_text())
    frozen = evaluation["frozen_manifest"]
    raw = ROOT / "data/raw"
    train_path = ROOT / frozen["settings"]["train_path"]
    required = {train_path: frozen["train_sha256"], final / "baseline_model.joblib": frozen["model_sha256"]}
    required.update({raw / name: value for name, value in evaluation["test_sha256"].items()})
    prediction = json.loads((final / "prediction_manifest.json").read_text())
    required.update({final / "scores.csv.gz": prediction["scores_sha256"], final / "alarms.json": prediction["alarms_sha256"]})
    if any(sha(path) != expected for path, expected in required.items()):
        raise ValueError("Frozen inputs, model or saved predictions changed.")
    frames = {}
    for year in (2018, 2019):
        for kind, suffix in [("pressure", "SCADA_Pressures"), ("flow", "SCADA_Flows"), ("leakage", "Leakages")]:
            frames[year, kind] = read_battledim_csv(raw / f"{year}_{suffix}.csv")
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:55", freq="5min", name="timestamp")
        if any(not frames[year, kind].index.equals(expected) for kind in ("pressure", "flow", "leakage")):
            raise ValueError("Diagnostics require complete, aligned yearly five-minute data.")
    # Only the locally produced, hash-verified frozen model is deserialized.
    model = joblib.load(final / "baseline_model.joblib")
    train = load_sensor_dataset(train_path).frame[frozen["sensors"]]
    fitting_scores = model.score(train).aggregate_score
    saved = pd.read_csv(final / "scores.csv.gz", index_col="timestamp", parse_dates=True)
    if not saved.index.equals(frames[2019, "pressure"].index) or not saved.threshold.eq(model.threshold_).all():
        raise ValueError("Saved scores do not agree with frozen timestamps/threshold.")
    warm = saved.index[0] - pd.Timedelta(days=frozen["settings"]["warmup_days"])
    combined = pd.concat([fitting_scores.loc[warm:], saved.score])
    gates = notification_trace(combined, model.threshold_, model.config.persistence_hours,
                               model.config.recovery_hours, model.config.cooldown_hours)
    actual = gates.index[gates.notification & (gates.index.year == 2019)]
    expected = pd.DatetimeIndex([row["timestamp"] for row in json.loads((final / "alarms.json").read_text())])
    if list(actual) != list(expected):
        raise ValueError("Diagnostic replay did not exactly reproduce frozen notifications.")
    events = event_size_rows(frames[2019, "leakage"], evaluation["events"], 2019, saved.score, gates)
    if len(events) != 19 or events.status.eq("detected").sum() != 2:
        raise ValueError("Unexpected frozen final-event outcomes.")
    reference_events = []
    for event in extract_leak_events(frames[2018, "leakage"]):
        reference_events.append({**asdict(event), "start": event.start.isoformat(), "end": event.end.isoformat(),
                                 "event_id": event.id, "status": "excluded_carry_in" if event.left_censored else "reference_only"})
    reference = event_size_rows(frames[2018, "leakage"], reference_events, 2018)
    comparisons, profiles = zip(*(compare_sensors(frames[2018, kind], frames[2019, kind], kind)
                                 for kind in ("pressure", "flow")))
    sensors, monthly = pd.concat(comparisons, ignore_index=True), pd.concat(profiles, ignore_index=True)
    detected, missed = events.loc[events.status == "detected"], events.loc[events.status == "missed"]
    summary = {"analysis_status": "exploratory_post_test_no_retuning", "recorded_date": "2026-10-10",
               "window_hours": 48, "frozen_notifications_reproduced": True,
               "new_events_2019": len(events), "detected": len(detected), "missed": len(missed),
               "new_event_size_reference_2018": len(reference),
               "detected_median_early_mean_m3h": float(detected.early_mean_flow_m3h.median()),
               "missed_median_early_mean_m3h": float(missed.early_mean_flow_m3h.median()),
               "largest_detected_early_mean_m3h": float(detected.early_mean_flow_m3h.max()),
               "smallest_detected_early_mean_m3h": float(detected.early_mean_flow_m3h.min()),
               "largest_missed_early_mean_m3h": float(missed.early_mean_flow_m3h.max()),
               "missed_larger_than_largest_detected": int((missed.early_mean_flow_m3h > detected.early_mean_flow_m3h.max()).sum()),
               "missed_larger_than_smallest_detected": int((missed.early_mean_flow_m3h > detected.early_mean_flow_m3h.min()).sum()),
               "missed_incident_open_at_onset": int(missed.incident_open_at_onset.sum()),
               "missed_with_ready_score_blocked": int(missed.blocked_ready_samples.gt(0).sum()),
               "reference_2018_median_early_mean_m3h": float(reference.early_mean_flow_m3h.median()),
               "all_2019_median_early_mean_m3h": float(events.early_mean_flow_m3h.median()),
               "fitting_2018_score_median": float(fitting_scores.median()),
               "test_2019_score_median": float(saved.score.median()),
               "fitting_2018_above_threshold_fraction": float(fitting_scores.gt(model.threshold_).mean()),
               "test_2019_above_threshold_fraction": float(saved.score.gt(model.threshold_).mean()),
               "pressure_sensors_shifted_over_one_2018_sd": int(sensors.loc[sensors.kind == "pressure", "mean_shift_2018_sd"].abs().gt(1).sum()),
               "threshold": model.threshold_}
    if any(sha(path) != expected for path, expected in protected.items()):
        raise ValueError("A frozen output changed during diagnostics.")
    output.mkdir(parents=True)
    events.to_csv(output / "event_size_2019.csv", index=False)
    reference.to_csv(output / "event_size_reference_2018.csv", index=False)
    sensors.to_csv(output / "sensor_comparison.csv", index=False)
    monthly.to_csv(output / "monthly_sensor_means.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    plot_events(events, output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, (year, data) in zip(axes, [(2018, reference), (2019, events)]):
        values = data.early_mean_flow_m3h.sort_values()
        ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", color="#277d72" if year == 2018 else "#b64b65")
        ax.set(title=f"{year}: {len(values)} new events", xlabel="First-48h mean leak flow (m3/h)", ylabel="Cumulative event fraction", ylim=(0, 1.05))
        ax.grid(alpha=.2)
    fig.savefig(output / "yearly_leak_sizes.png", dpi=150)
    plt.close(fig)
    flow_profiles = monthly.loc[monthly.kind == "flow"]
    fig, axes = plt.subplots(len(frames[2018, "flow"].columns), 1, figsize=(10, 7), layout="constrained")
    for ax, sensor in zip(axes, frames[2018, "flow"].columns):
        for year, color in [(2018, "#277d72"), (2019, "#b64b65")]:
            values = flow_profiles.loc[(flow_profiles.sensor == sensor) & (flow_profiles.year == year)]
            ax.plot(values.month, values["mean"], label=str(year), color=color)
        ax.set(title=sensor + " (measured flow, not isolated demand)", ylabel="m3/h", xticks=range(1, 13))
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / "monthly_flow_comparison.png", dpi=150)
    plt.close(fig)
    report = make_report(summary, sensors)
    (output / "report.md").write_text(report, encoding="utf-8")
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in required}
    inputs.update({str(path.relative_to(ROOT)): sha(path) for path in raw.glob("2018_*.csv")})
    inputs["artifacts/results/final_2019/evaluation.json"] = protected[final / "evaluation.json"]
    manifest = {"protocol": "post_test_diagnostics_v1", "status": summary["analysis_status"],
                "input_sha256": inputs, "protected_final_outputs_unchanged": True,
                "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                    [Path(__file__), ROOT / "src/evaluation/post_test.py", ROOT / "docs/POST_TEST_ANALYSIS_V1.md"]},
                "output_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def plot_events(events: pd.DataFrame, output: Path) -> None:
    ordered = events.sort_values("early_mean_flow_m3h").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True, layout="constrained")
    for ax, column, label in zip(axes, ["early_mean_flow_m3h", "early_peak_flow_m3h"], ["Mean", "Peak"]):
        for status, color, marker in [("detected", "#277d72", "o"), ("missed", "#b64b65", "x")]:
            rows = ordered.loc[ordered.status == status]
            ax.scatter(rows[column], rows.index, label=f"{status} (n={len(rows)})", color=color, marker=marker, s=65)
        ax.set(xlabel=f"{label} leak flow in first 48h (m3/h)", yticks=ordered.index,
               yticklabels=[f'{row.pipe} / {str(row.start)[:10]}' for row in ordered.itertuples()])
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Frozen 2019 outcomes: all 19 new events; no detector retuning")
    fig.savefig(output / "detection_by_leak_size.png", dpi=150)
    plt.close(fig)


def make_report(s: dict, sensors: pd.DataFrame) -> str:
    flows = sensors.loc[sensors.kind == "flow"]
    flow_rows = "\n".join(f'| {r.sensor} | {r.mean_2018:.2f} | {r.mean_2019:.2f} | {r.mean_shift_2018_sd:.2f} |'
                          for r in flows.itertuples())
    contradiction = (f'{s["missed_larger_than_largest_detected"]} missed events have a larger first-48h mean '
                     'flow than the largest detected event. A small-leaks-only explanation is not supported.'
                     if s["missed_larger_than_largest_detected"] else
                     'The observed size ordering is descriptive only; two detections cannot establish size sensitivity.')
    return f'''# Frozen 2019 Misses and Year-to-Year Conditions

Exploratory post-test diagnostics, recorded 10 October 2026. No refitting,
recalibration, alarm-policy change or rescoring of outcomes. All frozen final
output hashes are unchanged. Diagnostic notification replay exactly reproduces
the two saved 2019 notification timestamps, including late-2018 warmup state.

## Leak Size

All 19 new events are plotted; four carry-in events are excluded from onset recall.
Size is the mean/peak pipe leakage during [onset, min(onset + 48h, end, year end)).
This uses retrospective labels for diagnosis, never detector inputs. Observed
full-event peaks are secondary, may occur after the matching deadline, and may
be right-censored. No favourable size cutoff was selected.

- Detected: 2 events; median early mean flow {s["detected_median_early_mean_m3h"]:.2f} m3/h.
- Missed: 17 events; median early mean flow {s["missed_median_early_mean_m3h"]:.2f} m3/h.
- Largest detected early mean: {s["largest_detected_early_mean_m3h"]:.2f} m3/h.
- Largest missed early mean: {s["largest_missed_early_mean_m3h"]:.2f} m3/h.

{contradiction}
However, {s["missed_larger_than_smallest_detected"]} missed events exceed the
smaller detected event ({s["smallest_detected_early_mean_m3h"]:.2f} m3/h).
Thus leak size alone does not explain the observed outcomes. In particular,
large missed events must be considered alongside notification-state blocking.

![All event sizes and outcomes](detection_by_leak_size.png)

## Notification State

{s["missed_incident_open_at_onset"]}/17 missed events began while an incident was
already open. In {s["missed_with_ready_score_blocked"]}/17 missed-event windows,
the score reached the frozen three-hour persistence criterion but an open
incident blocked a new notification. Windows can overlap; this does not mean
each eligible score run is caused by that pipe or that removing suppression
would produce a valid localized detection. It identifies a conflict between
incident-level de-duplication and new-onset recall, not a new performance estimate.

## Yearly Conditions

The identical frozen model's median score is {s["fitting_2018_score_median"]:.3f}
on 2018 fitting data and {s["test_2019_score_median"]:.3f} in 2019. Score exceedance
changes from {s["fitting_2018_above_threshold_fraction"]:.1%} to
{s["test_2019_above_threshold_fraction"]:.1%}. The fitting exceedance rate reflects
the quantile chosen on that same data; this is not a fair generalization test.
{s["pressure_sensors_shifted_over_one_2018_sd"]} pressure sensors have an absolute
yearly mean shift exceeding one 2018 standard deviation.

| Flowmeter | 2018 Mean (m3/h) | 2019 Mean (m3/h) | Shift / 2018 SD |
| --- | ---: | ---: | ---: |
{flow_rows}

Each meter is considered separately, not summed (pump flow can duplicate network
movement). Raw inlet flows contain demand, leakage and operational effects.
Without AMRs and operating controls this is not proof of changed consumption.
The sensor audit also reports matched month/hour profile differences, daily-mean
KS distances without IID p-values, and repeated-value fractions (not fault labels).

The {s["new_event_size_reference_2018"]} full-year 2018 new-event size references
have median early mean flow {s["reference_2018_median_early_mean_m3h"]:.2f} m3/h,
versus {s["all_2019_median_early_mean_m3h"]:.2f} for 2019. This 2018 descriptive
reference is not the nine-event development evaluation. Raw condition differences
and fitting-vs-test residual changes do not distinguish distribution shift,
leak contamination, operating changes and overfitting causally.

![Leak-size distributions](yearly_leak_sizes.png)
![Separate measured-flow profiles](monthly_flow_comparison.png)

## Hypothesis and Falsifiable Next Study

Conditional pressure residuals may absorb common leak responses because other
pressure sensors and flows are predictors. Shifted conditions may keep scores
elevated, while recovery-based de-duplication prevents separate onset notifications.
These are competing, potentially interacting hypotheses, not established causes.

Build paired nominal/leak WNTR simulations with controlled leak size/location,
demand uncertainty and pump/valve regimes. Compare hydraulic-model residuals
against the current conditional residuals under the same recorded protocol;
report event recall by predeclared size bands, false alerts and missed-inclusive
delay summaries. Separate ablations for residual model and notification policy.
Hold out locations, demand regimes and an independent network/year; validate
simulation realism against observed sensor statistics. Do not retune on observed
2019 and describe it as an untouched final test.

Sources: [official BattLeDIM dataset](https://zenodo.org/records/4017659),
[official units/conditions](https://battledim.ucy.ac.cy/wp-content/uploads/2020/01/BattLeDIM_Problem_Description_and_Rules-v1.3.pdf),
[WNTR documentation](https://usepa.github.io/WNTR/userguide.html).
Dataset-derived content: CC BY 4.0. No supervisor-specific alignment is claimed
without the actual DC10 project description.
'''


def publish(output: Path) -> None:
    """Add reviewed aggregate diagnostics without replacing frozen bundle files."""
    target = ROOT / "demo_bundle/diagnostics_2019"
    if target.exists():
        raise ValueError("Published diagnostics exist; do not silently overwrite them.")
    names = ["event_size_2019.csv", "event_size_reference_2018.csv", "sensor_comparison.csv",
             "monthly_sensor_means.csv", "summary.json", "report.md", "manifest.json",
             "detection_by_leak_size.png", "yearly_leak_sizes.png", "monthly_flow_comparison.png"]
    diagnostics = json.loads((output / "manifest.json").read_text())
    if any(sha(output / name) != value for name, value in diagnostics["output_sha256"].items()):
        raise ValueError("Diagnostic outputs changed since analysis; publishing refused.")
    if any(not (output / name).is_file() for name in names):
        raise ValueError("Diagnostic outputs are incomplete.")
    manifest_path = ROOT / "demo_bundle/bundle_manifest.json"
    bundle = json.loads(manifest_path.read_text())
    for name, item in bundle["files"].items():
        if sha(ROOT / "demo_bundle" / name) != item["sha256"]:
            raise ValueError("Existing bundle checksum mismatch; publishing refused.")
    target.mkdir()
    for name in names:
        destination = target / name
        shutil.copyfile(output / name, destination)
        bundle["files"][f"diagnostics_2019/{name}"] = {"sha256": sha(destination), "bytes": destination.stat().st_size}
    manifest_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/results/diagnostics_2019")
    parser.add_argument("--publish", action="store_true", help="Add aggregate diagnostics to the public bundle after analysis")
    args = parser.parse_args()
    analyze(args.output)
    if args.publish:
        publish(args.output)
