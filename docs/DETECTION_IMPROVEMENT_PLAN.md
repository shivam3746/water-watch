# Detection Improvement Protocol

Recorded 2026-10-06 after inspecting the original 2018 holdout. These are follow-up development experiments, not a newly untouched test protocol. Original artifacts and evaluation remain available.

## 1. Alarm Notification Policy

Implemented first. Keep the initial three-hour persistence and frozen Ridge score/threshold. Once an alarm fires, hold its incident open until the score is at or below the threshold for 24 continuously observed hours. Impose a minimum 24-hour cooldown measured from the emitted alarm. Short normal dips, missing values, and observation gaps cannot close an incident. Gaps reset recovery evidence; suppressed candidates do not extend cooldown.

These durations are fixed operator-policy choices requested before running the replay, not values optimized over the inspected leak outcomes. All deduplication uses current/past scores only, never ground truth or future scores. A network-wide open incident can suppress a distinct leak; future evaluation must measure this failure mode, especially overlapping leaks. This is batch replay; persistent incident state across separate calls/UI actions is not yet implemented.

The original 2018 replay has nine alarms, two detections, and seven repeats. The revised policy has two alarms and the same two detections and delays. Precision rises from 0.222 to 1.000 in this small inspected slice. This measures notification reduction, not improved underlying pressure prediction or a fresh generalization claim.

## 2. Validation Before Additional Detector Tuning

Use chronological expanding/blocked validation, fitting all preprocessing and detectors only on preceding data. Select model parameters on inner chronological validation; report event counts, carry-in exclusions, per-event delays, misses, and fold-level metrics with spread. Do not use ordinary leave-one-event-out that trains on future events to predict earlier events. Overlapping ongoing leaks must remain identified; an event split must not silently distribute related windows across train and test.

Because the original holdout was inspected, 2018 can now serve as development data. Freeze exact block boundaries and parameter candidates before running new experiments. Record the protocol change and preserve the original fixed-split result. Full-year blocked folds are not yet implemented or selected in this document; calendar blocks and minimum training requirements must be fixed before their first run.

The local raw folder currently contains only 2018. The official release includes `2019_SCADA_Pressures.csv`, `2019_SCADA_Flows.csv`, and `2019_Leakages.csv`. Reserve 2019 for a final evaluation after development choices are frozen; do not inspect its leak outcomes during tuning.

Source: [official BattLeDIM dataset](https://zenodo.org/records/4017659). Chronological validation reference: [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## 3. Residual Change Detection

Implement one-sided EWMA or CUSUM as an alternative to raw threshold persistence, sharing the notification policy above. Estimate score normalization and choose smoothing/reference/control parameters using training and inner validation only. The residual scores are smoothed and correlated, and training contains leaks, so textbook independent, in-control average-run-length guarantees cannot be assumed. Report an empirical delay/false-alert trade-off instead of claiming guaranteed earlier detection.

References: [NIST EWMA](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm), [NIST CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm).

## 4. Supervised Target and Model Simplicity

Prefer a separate experiment regressing the sum of per-pipe leakage measurements, after verifying their units and interpretation against dataset documentation. A predicted rise rather than positive absolute leakage would drive alarms. Fit transformation, feature selection, and rise thresholds on past training data only. Evaluate both continuous prediction error and event detection; good regression error alone does not establish onset detection.

Alternative: use a 48-72-hour post-onset detection window while marking subsequent ongoing-leak windows unknown instead of negative. "Onset within the next N hours" is a different forecasting task; making its horizon larger does not give the detector more time after a leak starts. Any future-dependent labeling needs a chronological purge at boundaries. Because multiple background leaks are ongoing for most of 2018, excluding every ongoing-leak row from negatives may leave too little data; count usable events and rows before choosing that target.

Reduce sensor-specific features by a training-only selection or sensor-independent residual summaries. Use a predefined small feature set and stronger regularization rather than a large parameter search. Compare the simplified model against the residual baseline on the same folds and new-event definition. Do not select features from the already inspected test leaks.

Reference: [LightGBM parameter tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html).

## 5. Simulation

WNTR/EPANET augmentation remains optional future work. Verify hydraulic realism, demand variation, leak sizes/locations, and sensor noise before treating synthetic examples as transferable evidence. No simulation has been implemented.
