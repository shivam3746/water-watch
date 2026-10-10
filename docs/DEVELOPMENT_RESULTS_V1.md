# Water Watch: Development Results V1

Recorded 6 October 2026. Research prototype on BattLeDIM L-Town.

## Research Question

Can pressure/flow residuals and lightweight supervised models provide useful
new-leak notifications, with fewer repeated alarms and auditable event metrics?
This prototype detects temporal evidence; it does not establish pipe localization.

## Procedure

Eight chronological outer months, May-December 2018, contain nine new pipe-leak
events. Earlier observations fit each model; the preceding 28 days calibrate
candidates. One notification can match one event within 48 hours. Carry-in leaks
do not enter new-onset recall. Open incident state carries across calibration into
evaluation. The recorded protocol is [VALIDATION_PROTOCOL_V1.md](VALIDATION_PROTOCOL_V1.md).

These are retrospective development results: the original 2018 holdout had
already been inspected. Neither 2019 data nor an independent network was tested.
Protocol SHA-256:
`73bacc6efe77e19e5eeaf1421c98c01cb06a6282afe2b6023ddefe1d786436bb`.

## Results

| Detector | Detected / New Events | Unmatched Alerts | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original residual baseline | 4 / 9 | 15 | 0.211 | 0.444 | 0.286 |
| De-duplicated residual baseline | 4 / 9 | 1 | 0.800 | 0.444 | 0.571 |
| EWMA | 2 / 9 | 17 | 0.105 | 0.222 | 0.143 |
| CUSUM | 0 / 9 | 0 | 0.000 | 0.000 | 0.000 |
| Compact LightGBM classifier | 0 / 9 | 0 | 0.000 | 0.000 | 0.000 |
| Leak-flow LightGBM regression | 1 / 9 | 4 | 0.200 | 0.111 | 0.143 |

The de-duplicated baseline's detected-event mean delay was 6.96 hours, sample
standard deviation 5.95 hours; five missed events have no delay. Its unmatched
alert rate was 0.029 per week over 35 weeks. De-duplication reduced unmatched
notifications without losing detections in this experiment, but this is not a
guarantee for other networks or leak patterns.

Baseline recall across the five event-containing months averaged 0.50, with
sample standard deviation 0.50. The nine eligible events were:

| Pipe | Onset | De-duplicated Baseline Outcome |
| --- | --- | --- |
| p628 | 2 May | Missed |
| p538 | 18 May | Missed |
| p866 | 1 June | Missed |
| p31 | 29 June | Missed |
| p654 | 7 July | Missed |
| p810 | 30 July | Detected after 15 h 40 min |
| p183 | 7 August | Detected after 3 h 15 min |
| p158 | 6 October | Detected after 3 h |
| p369 | 26 October | Detected after 5 h 55 min |

## Interpretation and Limitations

The earlier two-event slice was too narrow to describe overall performance.
The baseline remains the strongest **development candidate**, not a validated
operational detector. Candidate calibration was uninformative in six baseline
folds and all eight classifier/CUSUM folds. Monthly blocks share fitting history
and one simulated network; they are not independent physical-network samples.

The classifier uses 21 fixed features instead of 205, excludes unknown ongoing
leaks, and has only 294 normal fitting rows. It detected no outer events despite
the target and regularization corrections. Limited normal examples and poor
calibration are concrete constraints; the exact failure mechanism remains to be
investigated rather than asserted from these scores alone.

Regression beat the fitting-median constant predictor's flow RMSE in six of eight
months, but produced only one timely event notification. Predicting total flow and
detecting its onset are different tasks. CUSUM and EWMA need careful diagnosis of
chart/recovery state before any further parameter changes. New experiments must
be recorded separately rather than overwriting this protocol to improve scores.

## Reproduce and Share

Run `scripts/run_validation.py` after preparing the 2018 CSVs. Results, source/input
hashes, calibration audits and portable HTML are generated under
`artifacts/results/validation_v1/`; raw data and generated artifacts remain ignored
by Git. `scripts/render_validation_report.py` regenerates presentation only and
records separate renderer/input hashes. The original experiment manifest is kept.
The complete test suite currently passes 55 tests; correctness is not accuracy.

Next: freeze an independent 2019 final-test procedure, build a historical-replay
dashboard, review data redistribution terms, deploy and verify the public demo,
then prepare a supervisor brief. Until independent testing is complete, describe
the release as a research demonstration with outstanding validation.
