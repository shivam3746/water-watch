# Supervisor Release Checklist

Target: supervisor handoff Monday, 12 October 2026.
This is a reviewable prototype, not an operational leak-warning service.

## Evidence First

- [x] Finish and test the frozen 2018 chronological experiment runner.
- [x] Run all eight folds; retain input/code hashes and calibration audits.
- [x] Review missed events, false alerts, delay spread, and sparse calibration.
- [x] Download official 2019 sensors/labels alongside 2018 and verify checksums.
- [x] Freeze the 2018-only detector/procedure before loading 2019; complete and
  preserve the final test, including its low recall (2 of 19 new events detected).
- [x] Prepare a concise results document with explicit development/test status
  (`DEVELOPMENT_RESULTS_V1.md`).

## Demonstration

- [x] Build a local replay dashboard with detector selection, score/alarm timelines,
  ground-truth overlays, and event-level results.
- [x] Verify local desktop/mobile charts, event selection, tab navigation, and
  document overflow with browser checks.
- [x] Generate exact baseline-score explanations and three development case studies.
- [x] Implement a persistent LangGraph interrupt and human approval/rejection log;
  no physical action is executed. LightGBM SHAP and LLM summaries are not included.
- [x] Clearly label historical replay, simulated-network provenance, and
  temporal matching rather than proven localization.
- [x] Package only the artifacts needed for a reproducible public demo; verify
  CC BY 4.0 redistribution terms and include dataset attribution.
- [x] Test the public entrypoint in a clean, deployment-only environment;
  verify bundle hashes and session-isolated temporary review storage.
- [x] Deploy the bundled app; verify startup and desktop/mobile usability.
- [x] Test the public URL in fresh signed-out browser sessions.

## Handoff

- [x] Prepare a research brief: question, methods, results, limitations,
  and proposed supervisor discussion topics (`RESEARCH_BRIEF.md`).
- [x] Update README setup and reproduction steps.
- [x] Full-suite/browser verification after incident and final-test integration;
  desktop/mobile checks pass. Recheck again on the public deployment.
- [x] Review release contents for credentials, private files, and unreviewed
  raw data; exclude model pickles and persistent review databases.
- [ ] Tag the reviewed release and provide repository, app, and brief links.

2019 testing is complete and shows poor sensitivity. Share the app as an
explainable decision-support research demonstration, not a validated operational
detector. Further tuning requires another development protocol and a new final
holdout for independent confirmation.
Simulation augmentation and operational deployment are future work, not release
requirements for this research demonstration.
