from __future__ import annotations

import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard import MODEL_NAMES, comparison_figure, event_window, load_results, load_timeline, replay_figure
from src.incident_ui import case_panel, final_test_panel, incident_panel
from src.deployment import public_review_workspace

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("WATER_WATCH_RESULTS", str(ROOT / "artifacts/results/validation_v1")))
EXPLANATIONS = Path(os.environ.get("WATER_WATCH_EXPLANATIONS", str(ROOT / "artifacts/explanations")))
REVIEW_DB = Path(os.environ.get("WATER_WATCH_REVIEW_DB", str(ROOT / "artifacts/incidents/reviews.sqlite")))
FINAL_RESULTS = Path(os.environ.get("WATER_WATCH_FINAL_RESULTS", str(ROOT / "artifacts/results/final_2019")))

st.set_page_config(page_title="Water Watch | Research Replay", page_icon="W", layout="wide")
PUBLIC_DEMO = os.environ.get("WATER_WATCH_PUBLIC_DEMO") == "1"
if PUBLIC_DEMO:
    REVIEW_DB = public_review_workspace(st.session_state)
st.markdown("""<style>
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1400px}
h1{font-size:2rem!important;letter-spacing:0!important}h2{font-size:1.4rem!important}
h3{font-size:1.1rem!important} [data-testid="stMetricValue"]{font-size:1.8rem}
@media(max-width:640px){
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]){display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])>[data-testid="stColumn"]{width:100%!important;min-width:0!important}
}
</style>""", unsafe_allow_html=True)


@st.cache_data
def cached_results(path: str):
    return load_results(Path(path))


@st.cache_data
def cached_timeline(path: str, month: str):
    return load_timeline(Path(path), month)


def csv_download(label: str, frame: pd.DataFrame, filename: str):
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), filename, "text/csv", icon=":material/download:")


try:
    results = cached_results(str(RESULTS))
except (FileNotFoundError, ValueError, KeyError) as exc:
    st.title("Water Watch")
    st.error(f"Replay results unavailable: {exc}")
    st.code(".\\.venv\\Scripts\\python.exe scripts/run_validation.py", language="powershell")
    st.stop()

st.title("Water Watch")
st.caption("L-Town / BattLeDIM · Historical replay · 2018 development validation")
if (FINAL_RESULTS / "evaluation.json").exists():
    final_metrics = json.loads((FINAL_RESULTS / "evaluation.json").read_text())["metrics"]
    st.warning(f'Research demonstration, not live monitoring. The frozen 2019 test detected {final_metrics["true_positives"]}/{final_metrics["eligible_leak_events"]} new events ({final_metrics["recall"]:.1%} recall). Not operationally validated.', icon=":material/science:")
else:
    st.warning("Research demonstration, not live monitoring. The 2018 holdout was previously inspected; independent 2019 validation is outstanding.", icon=":material/science:")

months = sorted(results.folds["fold"].unique())
models = [model for model in MODEL_NAMES if model in results.summary["model"].values]
with st.sidebar:
    st.subheader("Experiment")
    model = st.selectbox("Detector", models, format_func=MODEL_NAMES.get)
    month = st.selectbox("Evaluation month", months, index=months.index("2018-10") if "2018-10" in months else 0)
    st.divider()
    st.caption("PROTOCOL")
    st.write(results.manifest.get("protocol_id", "Unknown"))
    st.caption("May–December 2018 · 8 folds · 48-hour event deadline")
    st.caption("Timing matches are not proven pipe localization.")

selected = results.folds.loc[(results.folds["model"] == model) & (results.folds["fold"] == month)]
if selected.empty or selected.iloc[0]["status"] != "evaluated":
    st.error("This detector has no evaluated model for the selected month.")
    st.stop()
fold = selected.iloc[0]
events = results.events.loc[(results.events["model"] == model) & (results.events["fold"] == month)].copy()
alarms = results.alarms.loc[(results.alarms["model"] == model) & (results.alarms["fold"] == month)].copy()
eligible = events.loc[events["status"] != "excluded_carry_in"].reset_index(drop=True)

replay_tab, comparison_tab, evidence_tab, protocol_tab, incident_tab, cases_tab, final_tab = st.tabs(["Historical Replay", "Detector Comparison", "Event Evidence", "Protocol & Provenance", "Incident Review", "Case Studies", "2019 Final Test"])
with replay_tab:
    st.subheader(f"{pd.Timestamp(month).strftime('%B %Y')} · {MODEL_NAMES[model]}")
    overall = results.summary.loc[results.summary["model"] == model].iloc[0]
    st.caption(f'All development months: {int(overall["detected_events"])}/{int(overall["eligible_events"])} new leaks detected · {overall["precision"]:.1%} precision · {int(overall["missed_events"])} missed events')
    st.caption("Completed-month metrics · Changing the replay window does not recalculate these results.")
    cols = st.columns(4)
    cols[0].metric("New leaks detected", f'{int(fold["true_positives"])} / {int(fold["eligible_leak_events"])}')
    cols[1].metric("Unmatched alerts", str(int(fold["false_positives"])))
    cols[2].metric("Precision", f'{fold["precision"]:.1%}' if fold["alarm_count"] else "N/A")
    delay = fold["avg_detection_delay_hours"]
    cols[3].metric("Detected-event mean delay", f"{delay:.2f} h" if pd.notna(delay) else "N/A")
    if not fold["calibration_informative"]:
        st.caption("Calibration warning: no eligible calibration event, or no candidate detected any. The selected setting is experimental.")
    try:
        timeline = cached_timeline(str(RESULTS), month)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Timeline unavailable: {exc}")
        st.stop()
    controls = st.columns([2, 1, 1])
    options = ["Full month"] + [f'{row.pipe} · {row.start:%d %b %H:%M} · {row.status}' for row in eligible.itertuples()]
    focus = controls[0].selectbox("Event window", range(len(options)), format_func=lambda value: options[value])
    show_truth = controls[1].toggle("Ground truth", value=True)
    show_prediction = controls[2].toggle("Regressor prediction", value=model == "lightgbm_regression")
    start, end = event_window(timeline, eligible.iloc[focus - 1] if focus else None)
    cursor = st.slider("Replay until", min_value=start.to_pydatetime(), max_value=end.to_pydatetime(),
                       value=end.to_pydatetime(), step=pd.Timedelta(minutes=5).to_pytimedelta(), format="DD MMM HH:mm")
    st.plotly_chart(replay_figure(timeline, model, events, alarms, start, pd.Timestamp(cursor), show_truth, show_prediction),
                    width="stretch", key="replay_chart", config={"displaylogo": False})
    st.caption("Ground-truth flow includes existing leaks. Dotted lines mark new pipe-leak onsets; diamonds mark emitted notifications.")
    csv_download("Download visible timeline", timeline.loc[start:cursor].reset_index(), f"water_watch_{month}_replay.csv")

with comparison_tab:
    st.subheader("Across all development months")
    total_events = int(results.summary["eligible_events"].max())
    st.caption(f"{len(months)} chronological folds · {total_events} new events · May–December 2018 · No independent test")
    st.plotly_chart(comparison_figure(results.summary), width="stretch", key="comparison_chart", config={"displaylogo": False})
    summary = results.summary.copy()
    summary["model"] = summary["model"].map(MODEL_NAMES)
    st.dataframe(summary[["model", "detected_events", "missed_events", "false_alerts", "precision", "recall", "f1",
                          "false_alarms_per_week", "detection_delay_mean_hours", "detection_delay_std_hours"]],
                 hide_index=True, width="stretch")
    st.caption("Delay includes detected events only; misses are not assigned zero delay. Precision/recall use zero for empty denominators in stored metrics.")
    st.subheader("Monthly results")
    st.dataframe(results.folds.loc[results.folds["model"] == model, ["fold", "true_positives", "false_negatives", "false_positives",
                   "precision", "recall", "calibration_informative"]], hide_index=True, width="stretch")
    csv_download("Download detector comparison", results.summary, "water_watch_comparison.csv")

with evidence_tab:
    st.subheader("Completed-month event audit")
    carry_in = st.toggle("Include carry-in leaks", value=False)
    status = st.multiselect("Event status", ["detected", "missed"], default=["detected", "missed"])
    filtered = events.loc[events["status"].isin(status + (["excluded_carry_in"] if carry_in else []))]
    st.dataframe(filtered[["pipe", "start", "end", "status", "delay_hours", "alarm_id", "detection_window_truncated"]],
                 hide_index=True, width="stretch")
    st.caption(f'{int(fold["excluded_carry_in_events"])} carry-in events excluded from new-onset recall. Unmatched alarms do not prove absence of leakage.')
    st.subheader("Notification audit")
    if alarms.empty:
        st.info("No notifications were emitted in this month.")
    else:
        st.dataframe(alarms[["timestamp", "status", "event_id", "delay_hours", "ambiguous_match"]], hide_index=True, width="stretch")
    csv_download("Download event audit", filtered, f"water_watch_{month}_{model}_events.csv")
    csv_download("Download notification audit", alarms, f"water_watch_{month}_{model}_alarms.csv")

with protocol_tab:
    st.subheader("Recorded development procedure")
    st.write("Each month uses an earlier fitting prefix and the preceding 28 days for calibration. Parameters are selected using calibration only. Notification state carries into the evaluation month. The original 2018 results were already inspected before this protocol was recorded.")
    st.write("One notification matches at most one new pipe event before its end and the exclusive 48-hour deadline. Carry-in events are excluded from onset recall. Duplicate, late, and other unmatched notifications count as false alerts for the new-onset task.")
    st.write("The compact classifier uses 21 causal features; ongoing leaks outside the onset window are unknown, not normal. Regression predicts total leak flow in cubic metres per hour and alarms on its trailing rise. Neither model has established robust generalization.")
    st.caption("Fold blocks share one simulated network and overlapping training histories; they are not independent network samples. No SHAP explanation, physical localization, or operational safety claim is made.")
    st.subheader("Selected calibration candidates")
    calibration = results.calibration.loc[(results.calibration["model"] == model) & (results.calibration["fold"] == month)]
    st.dataframe(calibration[["candidate", "parameters", "threshold", "true_positives", "false_positives", "false_negatives", "f1"]], hide_index=True, width="stretch")
    with st.expander("Experiment manifest"):
        st.json(results.manifest)
    st.download_button("Download manifest", (RESULTS / "run_manifest.json").read_bytes(), "water_watch_manifest.json", "application/json", icon=":material/download:")

with incident_tab:
    incident_panel(EXPLANATIONS, REVIEW_DB, public_demo=PUBLIC_DEMO)
with cases_tab:
    case_panel(EXPLANATIONS)
with final_tab:
    final_test_panel(FINAL_RESULTS)
if PUBLIC_DEMO:
    st.caption("Data: Vrachimis et al., BattLeDIM (2020), DOI 10.5281/zenodo.4017659. CC BY 4.0. Derived scores, evaluations, and selected sensor traces; no endorsement by the dataset authors.")
    st.markdown("[Dataset and attribution](https://zenodo.org/records/4017659) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)")
