from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def diagnostic_panel(directory: Path) -> None:
    if not (directory / "summary.json").exists():
        return
    summary = json.loads((directory / "summary.json").read_text())
    events = pd.read_csv(directory / "event_size_2019.csv")
    st.subheader("Post-test failure analysis")
    st.caption("Exploratory diagnostics recorded after observing 2019. Frozen predictions, threshold and outcomes unchanged. Not an independent validation.")
    st.write(f'Detected events had median first-48h mean leakage {summary["detected_median_early_mean_m3h"]:.2f} m3/h; '
             f'missed events {summary["missed_median_early_mean_m3h"]:.3f} m3/h. '
             f'However, {summary["missed_larger_than_smallest_detected"]} missed events were larger than the smaller detected event. Size alone does not explain the misses.')
    statistic = st.selectbox("Leak-size statistic", ["First-48h mean", "First-48h peak"])
    column = "early_mean_flow_m3h" if statistic.endswith("mean") else "early_peak_flow_m3h"
    ordered = events.sort_values(column).copy()
    ordered["label"] = ordered["pipe"] + " / " + ordered["start"].str[:10]
    chart = go.Figure()
    for status, color, symbol in [("detected", "#277d72", "circle"), ("missed", "#b64b65", "x")]:
        group = ordered.loc[ordered.status == status]
        chart.add_scatter(x=group[column], y=group.label, mode="markers", name=f"{status} ({len(group)})",
                          marker=dict(color=color, symbol=symbol, size=10))
    chart.update_layout(height=620, xaxis_title="Pipe leak flow (m3/h)", margin=dict(l=5, r=10, t=35, b=35),
                        legend=dict(orientation="h", y=1.06),
                        yaxis=dict(categoryorder="array", categoryarray=ordered.label.tolist()))
    st.plotly_chart(chart, width="stretch", key="post_test_sizes", config={"displaylogo": False, "displayModeBar": False})
    st.caption("All 19 new events. Four carry-in events excluded. Leakage is measured before the exclusive 48-hour deadline (or event end); later full-event peaks are not early detectability evidence.")
    st.write(f'{summary["missed_incident_open_at_onset"]}/17 missed events started with an incident already open. '
             f'In {summary["missed_with_ready_score_blocked"]}/17 missed-event windows, an open incident blocked scores that met the three-hour persistence criterion.')
    st.caption("Diagnostic replay exactly reproduces the two frozen notification timestamps. Ready scores can reflect existing leaks or operating changes; these counts are not localized leak detections or a counterfactual recall estimate.")
    st.dataframe(events[["pipe", "start", "status", "early_mean_flow_m3h", "early_peak_flow_m3h",
                         "incident_open_at_onset", "above_threshold_fraction", "blocked_ready_samples"]], hide_index=True, width="stretch")
    st.download_button("Download missed-event audit", (directory / "event_size_2019.csv").read_bytes(),
                       "water_watch_2019_miss_diagnostics.csv", "text/csv", icon=":material/download:")
    st.subheader("2018 and 2019 conditions")
    st.write(f'The same frozen model exceeded threshold on {summary["fitting_2018_above_threshold_fraction"]:.1%} of '
             f'2018 fitting observations and {summary["test_2019_above_threshold_fraction"]:.1%} of 2019 observations. '
             f'{summary["pressure_sensors_shifted_over_one_2018_sd"]} pressure sensors shifted by more than one 2018 standard deviation in yearly mean.')
    st.caption("Fitting versus test scores confound generalization error and changed conditions; they cannot establish overfitting or a causal distribution-shift explanation. Flowmeters include leakage and operational effects, not just demand.")
    monthly = pd.read_csv(directory / "monthly_sensor_means.csv")
    flows = monthly.loc[monthly.kind == "flow"]
    sensor = st.selectbox("Measured flowmeter", flows.sensor.unique().tolist())
    profile = go.Figure()
    for year, color in [(2018, "#277d72"), (2019, "#b64b65")]:
        group = flows.loc[(flows.sensor == sensor) & (flows.year == year)]
        profile.add_scatter(x=group.month, y=group["mean"], mode="lines+markers", name=str(year), line=dict(color=color))
    profile.update_layout(height=300, xaxis=dict(title="Month", dtick=1), yaxis_title="Measured flow (m3/h)",
                          margin=dict(l=5, r=5, t=15, b=35), legend=dict(orientation="h"))
    st.plotly_chart(profile, width="stretch", key="post_test_flow", config={"displaylogo": False, "displayModeBar": False})
    with st.expander("All sensor diagnostics and leak-size reference"):
        st.dataframe(pd.read_csv(directory / "sensor_comparison.csv"), hide_index=True, width="stretch")
        st.image(str(directory / "yearly_leak_sizes.png"))
        st.caption("2018 uses all 14 raw new events as a size reference, not the nine-event development evaluation. Daily-mean KS distances are descriptive; repeated readings are not sensor-fault labels.")
    st.subheader("Research hypothesis and next experiment")
    st.write("Conditional pressure residuals may absorb shared leak responses. Changed conditions may keep scores high, while long incident recovery suppresses later onsets. These mechanisms remain hypotheses, not established causes.")
    st.write("Next: paired nominal/leak WNTR simulations with controlled size, location, demand uncertainty and pump/valve regimes. Compare hydraulic residuals and notification-policy ablations under a new recorded protocol, holding out locations and operating conditions. Already-observed 2019 cannot serve as an untouched test for retuning.")
    st.download_button("Download post-test report", (directory / "report.md").read_bytes(),
                       "water_watch_post_test_report.md", "text/markdown", icon=":material/download:")
