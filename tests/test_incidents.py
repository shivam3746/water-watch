from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.agent.graph import decisions, review
from src.agent.schemas import Alarm
from src.detection.baseline import BaselineConfig, ResidualBaselineDetector
from src.explanation.residual_explainer import explain_snapshot
from scripts.run_final_2019 import require_year
from streamlit.testing.v1 import AppTest


@pytest.fixture
def context():
    rng = np.random.default_rng(42)
    index = pd.date_range("2018-01-01", periods=160, freq="5min")
    frame = pd.DataFrame({"pressure_a": 30 + rng.normal(size=160), "pressure_b": 40 + rng.normal(size=160),
                          "flow_c": 10 + rng.normal(size=160)}, index=index)
    model = ResidualBaselineDetector(["pressure_a", "pressure_b"], list(frame.columns), BaselineConfig()).fit(frame.iloc[:80])
    timestamp = index[120]
    score = float(model.score(frame.loc[:timestamp]).aggregate_score.iloc[-1])
    alarm = Alarm("alarm-test", timestamp.to_pydatetime(), score, "baseline", ["pressure_a"])
    return model, frame, timestamp, alarm


def test_explanation_is_exact_causal_and_json_serializable(context):
    model, frame, timestamp, alarm = context
    evidence, trace = explain_snapshot(model, frame, timestamp, alarm)
    assert sum(row["score_contribution"] for row in evidence["sensors"]) == pytest.approx(evidence["score"])
    assert sum(row["score_share"] for row in evidence["sensors"]) == pytest.approx(1)
    assert trace.index[-1] == timestamp
    changed = frame.copy()
    changed.loc[changed.index > timestamp] *= 1000
    other, other_trace = explain_snapshot(model, changed, timestamp, alarm)
    assert other == evidence
    pd.testing.assert_frame_equal(trace, other_trace)
    json.dumps(evidence, allow_nan=False)


def test_mismatched_alarm_score_is_rejected(context):
    model, frame, timestamp, alarm = context
    with pytest.raises(ValueError, match="disagrees"):
        explain_snapshot(model, frame, timestamp, replace(alarm, score=alarm.score + 1))


def test_short_context_is_explicitly_unavailable_not_nan(context):
    model, frame, _, _ = context
    with pytest.raises(ValueError, match="Insufficient"):
        explain_snapshot(model, frame.iloc[:8], frame.index[7])


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_interrupt_persists_and_only_explicit_review_logs(context, tmp_path, decision):
    model, frame, timestamp, alarm = context
    evidence, _ = explain_snapshot(model, frame, timestamp, alarm)
    database = tmp_path / "reviews.sqlite"
    state = review(evidence, database)
    assert state["awaiting_review"]
    assert decisions(database) == []
    assert review(evidence, database)["awaiting_review"]
    finished = review(evidence, database, {"decision": decision, "reviewer": "Test reviewer", "note": "Reviewed sensor evidence"})
    assert not finished["awaiting_review"]
    assert finished["logged"]
    logged = decisions(database)
    assert len(logged) == 1 and logged[0]["decision"] == decision
    assert logged[0]["action_executed"] is False
    assert review(evidence, database)["decision"]["decision"] == decision
    with pytest.raises(ValueError, match="already"):
        review(evidence, database, {"decision": decision, "reviewer": "Other reviewer"})


def test_invalid_review_and_tampered_evidence_do_not_log(context, tmp_path):
    model, frame, timestamp, alarm = context
    evidence, _ = explain_snapshot(model, frame, timestamp, alarm)
    database = tmp_path / "reviews.sqlite"
    for response in [{"decision": "approve", "reviewer": ""}, {"decision": "automatic", "reviewer": "Test"}]:
        with pytest.raises(ValueError):
            review(evidence, database, response)
    assert decisions(database) == []
    evidence["score"] += 1
    with pytest.raises(ValueError, match="integrity"):
        review(evidence, database)


def test_missed_event_snapshot_cannot_become_operator_incident(context, tmp_path):
    model, frame, timestamp, _ = context
    evidence, _ = explain_snapshot(model, frame, timestamp)
    with pytest.raises(ValueError, match="real detector"):
        review(evidence, tmp_path / "reviews.sqlite")


def test_final_test_requires_a_complete_year():
    frame = pd.DataFrame(index=pd.date_range("2019-01-01", "2019-12-31 23:55", freq="5min"))
    require_year(frame, 2019)
    with pytest.raises(ValueError, match="complete"):
        require_year(frame.iloc[:-1], 2019)
    with pytest.raises(ValueError):
        require_year(frame, 2018)


def test_incident_ui_requires_explicit_decision_and_persists_it(context, tmp_path):
    model, frame, timestamp, alarm = context
    evidence, trace = explain_snapshot(model, frame, timestamp, alarm)
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    trace.to_csv(tmp_path / "trace.csv", index_label="timestamp")
    (tmp_path / "incidents.json").write_text(json.dumps([{"id": "test", "timestamp": timestamp.isoformat(),
        "evidence_file": "evidence.json", "trace_file": "trace.csv"}]))
    database = tmp_path / "reviews.sqlite"
    source = f"from pathlib import Path\nfrom src.incident_ui import incident_panel\nincident_panel(Path({str(tmp_path)!r}), Path({str(database)!r}))"
    app = AppTest.from_string(source, default_timeout=30).run()
    assert not app.exception
    assert decisions(database) == []
    app.button[0].click().run()
    assert not app.exception
    assert app.radio[0].value is None
    app.button[0].click().run()
    assert app.error and decisions(database) == []
    app.text_input[0].set_value("Synthetic test reviewer")
    app.radio[0].set_value("Reject review recommendation")
    app.text_area[0].set_value("Synthetic test; not an operator decision")
    app.button[0].click().run()
    assert not app.exception
    assert decisions(database)[0]["decision"] == "reject"
    assert app.success
