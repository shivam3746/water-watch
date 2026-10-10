# Chronological Development Validation V1

Recorded 2026-10-06 before running this comparative experiment. The original 2018 holdout and event timings were already inspected, so this is retrospective development validation, not a blind preregistration or fresh test. The original fixed-split results remain available. Settings are in `configs/validation_v1.yaml`; each run saves a byte-for-byte protocol snapshot and SHA-256 digest before fitting.

## Fixed Folds

Evaluate the complete calendar months May through December 2018, eight non-overlapping outer blocks. Every fold fits on observations before the preceding 28-day calibration block. Calibration uses only that block, never the following outer month. Each model has its own fold-specific fitting and calibration; no model is selected from the outer-month labels inside a fold. The first four months provide warm-up history rather than evaluated events.

The shared supervised feature reference Ridge model is fitted on January 1-7 sensor data only. Supervised fitting starts January 8 and ends before each calibration block; reference residuals are out-of-sample on supervised fitting rows. Sensor-independent features are fixed before results rather than selected using known outer leaks. Each unsupervised baseline Ridge model is fitted on that fold's entire fitting prefix. These differing Ridge reference periods are an explicit design difference, not a controlled comparison of learner type alone.

## Comparisons and Selection

1. Original baseline: training-score quantile 0.995 and three-hour persistence, no recovery hold or cooldown.
2. De-duplicated baseline: identical Ridge/threshold/persistence, plus 24-hour continuous recovery and cooldown.
3. EWMA: pandas causal EWMA, alpha in {0.02, 0.1}, training-chart quantile in {0.99, 0.995}.
4. One-sided standardized CUSUM: training-score mean/std, reference shift in {0.25, 0.5}, decision limit in {5, 10}. Recurrence is max(0, previous + standardized_score - reference_shift).
5. Simplified classifier: fixed compact causal features, shallow regularized LightGBM, 48-hour post-onset positives while active. Subsequent ongoing-leak rows are unknown and excluded from fitting. Rows with no active leak are negatives. Candidate probability thresholds are {0.1, 0.3, 0.5, 0.7}.
6. Regressor: same compact features and regularization, predicting total per-pipe leakage flow in m^3/h. Alarm score is the positive rise between the current trailing six-hour prediction mean and its value 24 hours earlier. Positive training-rise quantiles {0.95, 0.99, 0.995} provide threshold candidates. Regression quality also reports MAE/RMSE against a fitting-prefix median-flow constant predictor, which uses no test targets.

EWMA/CUSUM alarm immediately on a chart crossing; supervised detectors require one hour of continuous evidence. All alternative detectors share 24-hour recovery/cooldown. These are empirical trade-offs, not identical alarm latencies or textbook independent-process false-alarm guarantees. The scores are autocorrelated and fitting data contain leaks.

Candidate selection maximizes calibration event F1, then minimizes calibration false positives, then detection delay, then chooses the earliest listed candidate. No calibration events or zero detections at all candidates is flagged as uninformative; the rule still produces a deterministic experimental candidate, not a claim of successful calibration. A classifier with no usable positive or negative examples is explicitly unavailable for that fold, never silently evaluated as a trained model.

## Causal State and Event Metrics

Training history warms rolling/chart features. Notification state starts at calibration, continues across calibration into the outer month, and only outer-month alarms enter outer metrics. It is never reset at the outer boundary to create a new notification artificially. Folds are independent replay experiments and not one production model retrained in place.

Ground truth events are individual positive pipe-leak runs. Match an alarm to one new event after onset, before the event ends and before the exclusive 48-hour deadline. Carry-in events are excluded from onset recall. Repeats and otherwise unmatched alerts count as false alerts for the new-onset task. Matching is temporal and does not establish pipe localization. No alarm is credited to two overlapping events. Short follow-up at month end is explicitly flagged; missed-event delays remain null.

Report every outer fold, every new event, calibration candidate results, usable supervised labels, and selected parameters. Pool true/false positives and missed events across folds for micro metrics. Report recall mean/std across event-containing folds, precision mean/std across alarm-containing folds, and detected-event delay mean/std; omit undefined denominators from macro summaries rather than fabricating observations. Quiet months still contribute to false-alert duration. Report untrained folds separately. Do not present eight folds as eight independent physical networks or bootstrap correlated rows as independent events.

## Holdout and Release

Only paths to 2018 data are accepted by this development runner. Do not load or inspect 2019 ground truth. After development choices are finalized, freeze the chosen model, features, notification policy, and metrics before evaluating 2019 once. Claims before that evaluation are exploratory development findings. The release should include an incident-evidence demo, reproducible commands, a brief research note, and an honest limitations section.

Leakage units are documented in the [official dataset metadata](https://zenodo.org/records/4017659/files/README.txt?download=1). Method references: [NIST CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm), [NIST EWMA](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm), [LightGBM regularization](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html).
