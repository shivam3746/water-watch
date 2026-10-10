from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.graph import review


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Review a saved detector notification with a persistent LangGraph interrupt.")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--database", type=Path, default=ROOT / "artifacts/incidents/reviews.sqlite")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    state = review(evidence, args.database)
    print(state["incident_summary"])
    print(state["suggested_action"])
    if state["awaiting_review"]:
        decision = input("Decision (approve/reject; blank leaves pending): ").strip().lower()
        if decision:
            reviewer = input("Reviewer name: ").strip()
            note = input("Review note: ")
            state = review(evidence, args.database, {"decision": decision, "reviewer": reviewer, "note": note})
    print("Awaiting human review" if state["awaiting_review"] else "Decision recorded; no physical action executed")
