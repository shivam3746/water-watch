from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class IncidentState(TypedDict, total=False):
    evidence: dict
    incident_summary: str
    suggested_action: str
    decision: dict
    logged: bool


def validate_evidence(evidence: dict) -> None:
    if evidence.get("kind") != "alarm" or not evidence.get("alarm"):
        raise ValueError("Only real detector notifications enter human review, not missed-event snapshots.")
    content = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if evidence.get("evidence_sha256") != digest:
        raise ValueError("Evidence integrity check failed.")
    if not evidence.get("causal") or evidence.get("localization_claim"):
        raise ValueError("Review requires causal evidence without an asserted physical location.")


def summarize(evidence: dict) -> str:
    validate_evidence(evidence)
    top = evidence["sensors"][:3]
    descriptions = [f'{row["sensor"]}: {row["score_share"]:.1%} of the score; measured pressure {row["observed_pressure_m"]:.2f} m versus expected {row["expected_pressure_m"]:.2f} m' for row in top]
    return (f'An anomaly notification was generated at {evidence["timestamp"]}. '
            f'The score was {evidence["score"]:.3f}, against threshold {evidence["threshold"]:.3f}. '
            + "; ".join(descriptions) + ". These are score contributors, not confirmed leak locations. "
            "A leak is not confirmed by this summary. No physical action has been executed.")


def validate_decision(decision: dict) -> dict:
    if not isinstance(decision, dict) or decision.get("decision") not in {"approve", "reject"}:
        raise ValueError("Decision must be explicitly approve or reject.")
    reviewer = decision.get("reviewer", "")
    if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
        raise ValueError("A reviewer name of 1-200 characters is required.")
    note = decision.get("note", "")
    if not isinstance(note, str) or len(note) > 5000:
        raise ValueError("Review note must be text of at most 5000 characters.")
    return {"decision": decision["decision"], "reviewer": reviewer.strip(), "note": note,
            "decision_timestamp": datetime.now(timezone.utc).isoformat()}


@contextmanager
def incident_graph(database: Path):
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=30, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS decisions (evidence_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.commit()

    def context_node(state):
        validate_evidence(state["evidence"])
        return {}

    def summary_node(state):
        return {"incident_summary": summarize(state["evidence"]),
                "suggested_action": "Review sensor reliability and corroborating evidence; consider a field investigation through the normal operator procedure. No automated network control."}

    def human_node(state):
        response = interrupt({"alarm_id": state["evidence"]["alarm"]["id"],
                              "summary": state["incident_summary"], "suggested_action": state["suggested_action"],
                              "approval_scope": "Accept or reject the review recommendation only; no physical action."})
        return {"decision": validate_decision(response)}

    def log_node(state):
        evidence = state["evidence"]
        payload = {"alarm_id": evidence["alarm"]["id"], "evidence_sha256": evidence["evidence_sha256"],
                   "alarm_timestamp": evidence["timestamp"], "score": evidence["score"],
                   "explanation": evidence, "summary": state["incident_summary"],
                   "suggested_action": state["suggested_action"], **state["decision"],
                   "action_executed": False, "reviewer_authenticated": False}
        # Checkpoint writes can run in background threads; isolate the audit transaction.
        with sqlite3.connect(database, timeout=30) as audit:
            audit.execute("BEGIN IMMEDIATE")
            existing = audit.execute("SELECT payload FROM decisions WHERE evidence_id=?", (evidence["evidence_sha256"],)).fetchone()
            if existing and json.loads(existing[0]) != payload:
                raise ValueError("An existing decision cannot be overwritten.")
            audit.execute("INSERT OR IGNORE INTO decisions VALUES (?,?)", (evidence["evidence_sha256"], json.dumps(payload, allow_nan=False)))
        return {"logged": True}

    builder = StateGraph(IncidentState)
    for name, function in [("context", context_node), ("summary", summary_node), ("human_review", human_node), ("decision_log", log_node)]:
        builder.add_node(name, function)
    builder.add_edge(START, "context")
    builder.add_edge("context", "summary")
    builder.add_edge("summary", "human_review")
    builder.add_edge("human_review", "decision_log")
    builder.add_edge("decision_log", END)
    try:
        yield builder.compile(checkpointer=SqliteSaver(connection))
    finally:
        connection.close()


def review(evidence: dict, database: Path, decision: dict | None = None) -> dict:
    validate_evidence(evidence)
    config = {"configurable": {"thread_id": evidence["evidence_sha256"]}}
    with incident_graph(database) as graph:
        state = graph.get_state(config)
        if not state.values:
            graph.invoke({"evidence": evidence}, config)
            state = graph.get_state(config)
        if decision is not None:
            validate_decision(decision)
            if not state.next:
                raise ValueError("This incident already has a recorded decision.")
            graph.invoke(Command(resume=decision), config)
            state = graph.get_state(config)
        return {**state.values, "awaiting_review": bool(state.next)}


def decisions(database: Path) -> list[dict]:
    if not database.exists():
        return []
    with sqlite3.connect(database) as connection:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='decisions'").fetchone():
            return []
        return [json.loads(row[0]) for row in connection.execute("SELECT payload FROM decisions ORDER BY rowid")]


def review_state(evidence: dict, database: Path) -> dict:
    validate_evidence(evidence)
    if not database.exists():
        return {}
    with incident_graph(database) as graph:
        state = graph.get_state({"configurable": {"thread_id": evidence["evidence_sha256"]}})
        return {**state.values, "awaiting_review": bool(state.next)} if state.values else {}
