from __future__ import annotations

import base64
import html
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_validation_report(output: Path, summary: pd.DataFrame, events: pd.DataFrame) -> None:
    """Create a portable, data-free visual summary of the audited experiment."""
    evaluated = summary.loc[summary["evaluated_folds"] > 0]
    labels = evaluated["model"].str.replace("_", " ").tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(evaluated))
    axes[0].bar(x - .18, evaluated["precision"], width=.36, label="Precision", color="#277d72")
    axes[0].bar(x + .18, evaluated["recall"], width=.36, label="Recall", color="#b64b65")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Pooled new-event detection")
    axes[0].legend(frameon=False)
    axes[1].bar(x, evaluated["false_alarms_per_week"], color="#647387")
    axes[1].set_title("Unmatched notifications per week")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("Water Watch | 2018 development experiments", fontsize=15)
    fig.savefig(output / "comparison.png", dpi=150)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    eligible = events.loc[events["status"] != "excluded_carry_in"].copy()
    eligible["event"] = eligible["fold"] + " / " + eligible["event_id"]
    eligible["outcome"] = eligible.apply(
        lambda row: f'Detected ({row["delay_hours"]:.2f} h)' if row["status"] == "detected" else "MISSED", axis=1
    )
    event_table = eligible.pivot(index="event", columns="model", values="outcome")
    columns = ["model", "eligible_events", "detected_events", "missed_events", "precision", "recall",
               "false_alarms_per_week", "detection_delay_mean_hours", "detection_delay_std_hours",
               "uninformative_calibration_folds"]
    manifest = html.escape((output / "run_manifest.json").read_text(encoding="utf-8"))
    content = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Water Watch Development Validation</title>
<style>body{{font:15px/1.55 system-ui,sans-serif;margin:0;color:#20292d;background:#fff}}
main{{max-width:1200px;margin:auto;padding:32px 20px}}h1{{font-size:30px}}h2{{font-size:21px}}
.notice{{border-left:4px solid #b64b65;padding:12px 18px;background:#faf1f3}}
img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{padding:9px;border-bottom:1px solid #dce2e4;text-align:left}}th{{background:#f0f4f4}}
pre{{overflow:auto;font-size:12px}}a{{color:#277d72}}</style><main>
<h1>Water Watch: Detector Experiments</h1>
<p class="notice"><strong>Retrospective development evidence, not an independent accuracy claim.</strong>
The chronological monthly protocol was specified before these comparisons, after the original 2018 holdout had already been inspected.
2019 was not loaded. This is historical replay on one simulated network.</p>
<img src="data:image/png;base64,{image}" alt="Detector precision, recall and unmatched notifications per week">
<h2>Results</h2><div class="scroll">{evaluated[columns].to_html(index=False, float_format=lambda value: f"{value:.3f}", na_rep="N/A", escape=True)}</div>
<p>Delay mean and spread include detected events only. Misses are shown below. Uninformative calibration means no eligible calibration events or no candidate detected any.
Precision uses one-to-one temporal event matching within 48 hours; this does not establish pipe localization.
Unmatched notifications can occur while old leaks remain active.</p>
<h2>Event Outcomes</h2><div class="scroll">{event_table.to_html(escape=True, na_rep="N/A")}</div>
<h2>Limits and Next Check</h2><p>Monthly folds share one network and overlapping fitting prefixes. They are not independent network samples.
Open incidents can suppress notifications for additional leaks. The compact classifier excludes unknown ongoing-leak rows,
leaving few normal examples. Regression fit quality is not event-detection quality. Review fold_metrics.csv, calibration_audit.csv,
event_audit.csv and alarm_audit.csv alongside this summary. Freeze a final detector and test procedure before inspecting 2019 outcomes.</p>
<details><summary>Reproducibility Manifest</summary><pre>{manifest}</pre></details></main></html>'''
    (output / "report.html").write_text(content, encoding="utf-8")
    inputs = [output / name for name in ("summary.csv", "event_audit.csv", "run_manifest.json")]
    report_manifest = {
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.exists()},
        "purpose": "Presentation derived from completed experiment; does not refit or select detectors.",
    }
    (output / "report_manifest.json").write_text(json.dumps(report_manifest, indent=2), encoding="utf-8")
