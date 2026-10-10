from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.graph import decisions, review, review_state, summarize


def evidence_view(record: dict, directory: Path) -> dict:
    evidence = json.loads((directory / record["evidence_file"]).read_text(encoding="utf-8"))
    trace = pd.read_csv(directory / record["trace_file"], index_col="timestamp", parse_dates=True)
    st.caption(f'Evidence ends at {evidence["timestamp"]} · Causal sensor-only context · Exact residual decomposition, not SHAP')
    left, right = st.columns([1, 2])
    top = evidence["sensors"][:5]
    chart = go.Figure(go.Bar(x=[row["score_contribution"] for row in top][::-1],
                             y=[row["sensor"] for row in top][::-1], orientation="h", marker_color="#277d72"))
    chart.update_layout(height=260, margin=dict(l=0, r=5, t=5, b=5), xaxis_title="Contribution to score")
    with left:
        st.metric("Residual score", f'{evidence["score"]:.3f}')
        st.caption(f'Threshold {evidence["threshold"]:.3f}')
        st.plotly_chart(chart, width="stretch", key=record["id"] + "_contributions", config={"displaylogo": False})
    with right:
        sensor = st.selectbox("Pressure sensor", [row["sensor"] for row in top[:3]], key=record["id"] + "_sensor")
        pressure = go.Figure()
        for suffix, label, color in [("observed", "Measured pressure", "#277d72"), ("expected", "Ridge expected pressure", "#b64b65")]:
            pressure.add_scatter(x=trace.index, y=trace[sensor + "__" + suffix], name=label, line=dict(color=color))
        pressure.update_layout(height=310, margin=dict(l=5, r=5, t=5, b=5), yaxis_title="Pressure (m)", hovermode="x unified", legend=dict(orientation="h"))
        st.plotly_chart(pressure, width="stretch", key=record["id"] + "_pressure", config={"displaylogo": False})
    st.caption(evidence["limitations"])
    with st.expander("Sensor and flow evidence"):
        st.dataframe(pd.DataFrame(evidence["sensors"]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(evidence["flows"]), hide_index=True, width="stretch")
    st.download_button("Download structured evidence", json.dumps(evidence, indent=2), record["id"] + ".json", "application/json",
                       icon=":material/download:", key=record["id"] + "_download")
    return evidence


def incident_panel(directory: Path, database: Path, public_demo: bool = False):
    st.subheader("Evidence-backed incident review")
    st.caption("2018 de-duplicated baseline notifications · Demonstration only · Reviewer identity is not authenticated")
    if public_demo:
        st.info("Reviews are isolated to this browser session and may be lost on reconnect or app restart. Use a demo alias, not personal details. This is not a permanent operator log.")
    if not (directory / "incidents.json").exists():
        st.info("Incident explanations have not been generated for this run.")
        return
    records = json.loads((directory / "incidents.json").read_text())
    if not records:
        st.info("No explained notifications available.")
        return
    selected = st.selectbox("Notification", range(len(records)), format_func=lambda i: records[i]["timestamp"] + " / " + records[i]["id"], index=len(records) - 1)
    record = records[selected]
    evidence = evidence_view(record, directory)
    st.subheader("Deterministic incident summary")
    st.write(summarize(evidence))
    st.write("Recommendation: review sensor reliability and corroborating evidence; consider field investigation through the normal operator procedure.")
    st.caption("Approval accepts this review recommendation only. No network control, dispatch, or other physical action is executed.")
    state = review_state(evidence, database)
    if not state:
        if st.button("Start human review", icon=":material/rule:"):
            review(evidence, database)
            st.rerun()
    elif state["awaiting_review"]:
        st.info("Awaiting human decision · Workflow checkpoint saved")
        with st.form("review_" + evidence["evidence_sha256"]):
            reviewer = st.text_input("Reviewer alias (demo)" if public_demo else "Reviewer name", max_chars=200)
            choice = st.radio("Review decision", ["Approve review recommendation", "Reject review recommendation"], index=None)
            note = st.text_area("Decision rationale", max_chars=5000)
            submitted = st.form_submit_button("Record decision", icon=":material/save:")
        if submitted:
            if choice is None or not reviewer.strip():
                st.error("A reviewer name and explicit decision are required.")
            else:
                try:
                    review(evidence, database, {"decision": "approve" if choice.startswith("Approve") else "reject", "reviewer": reviewer, "note": note})
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    else:
        st.success(f'Decision recorded: {state["decision"]["decision"]} · {state["decision"]["reviewer"]}')
        st.caption(state["decision"]["decision_timestamp"])
        st.write(state["decision"]["note"] or "No rationale supplied.")
    history = decisions(database)
    if history:
        with st.expander("Decision history"):
            st.dataframe(pd.DataFrame(history)[["alarm_id", "decision", "reviewer", "decision_timestamp", "action_executed"]], hide_index=True, width="stretch")
            st.download_button("Download decision log", json.dumps(history, indent=2), "water_watch_decisions.json", "application/json", icon=":material/download:")


def case_panel(directory: Path):
    st.subheader("Development case studies")
    if not (directory / "cases.json").exists():
        st.info("Case studies have not been generated.")
        return
    cases = json.loads((directory / "cases.json").read_text())
    labels = {"detected_event": "Detected event", "unmatched_alert": "Unmatched notification", "missed_event": "Missed event diagnostic"}
    selected = st.selectbox("Case study", range(len(cases)), format_func=lambda i: labels[cases[i]["category"]])
    case = {**cases[selected], "id": "case_" + cases[selected]["id"]}
    st.write(case["interpretation"])
    st.caption("First chronological example in each category, not the most favourable example. Event labels are retrospective evaluation context only.")
    if case["event_id"]:
        st.caption("Evaluation event: " + case["event_id"])
    evidence_view(case, directory)


def final_test_panel(path: Path):
    st.subheader("Held-out 2019 test")
    evaluation = path / "evaluation.json"
    if not evaluation.exists():
        st.info("Final-year evaluation has not been completed.")
        return
    result = json.loads(evaluation.read_text(encoding="utf-8"))
    metrics = result["metrics"]
    st.caption("Frozen 2018-only residual baseline · No 2019 fitting or calibration · Same simulated network, not external-network validation")
    cols = st.columns(4)
    cols[0].metric("New events detected", f'{metrics["true_positives"]} / {metrics["eligible_leak_events"]}')
    cols[1].metric("Recall", f'{metrics["recall"]:.1%}')
    cols[2].metric("Precision", f'{metrics["precision"]:.1%}')
    cols[3].metric("Missed events", str(metrics["false_negatives"]))
    st.warning("The held-out result shows poor sensitivity. Low alert volume and high precision do not establish a useful operational warning system.")
    st.caption(f'Mean detected-event delay {metrics["avg_detection_delay_hours"]:.2f} h · {metrics["false_positives"]} unmatched notifications · Missed-event delays remain null')
    st.dataframe(pd.DataFrame(result["events"])[["pipe", "start", "end", "status", "delay_hours", "right_censored_at_test_end"]], hide_index=True, width="stretch")
    with st.expander("Frozen final-test manifest"):
        st.json(result["frozen_manifest"])
    st.download_button("Download final evaluation", evaluation.read_bytes(), "water_watch_final_2019.json", "application/json", icon=":material/download:")
