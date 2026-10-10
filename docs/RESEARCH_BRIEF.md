# Water Watch: Explainable Leak-Notification Decision Support

Updated 10 October 2026 for supervisor discussion. Target handoff: Monday,
12 October 2026. Research demonstration, not an operational warning service.

## Question and Contribution

Can pressure/flow anomaly detection be combined with auditable evidence and
human review, without allowing a language model to decide whether a leak exists?

The prototype connects chronological data validation, six detector experiments,
event-level metrics, exact baseline-score explanations, a persistent human-review
workflow, and a replay dashboard. Its contribution is a reproducible investigation
and transparent decision-support pipeline, not a novel algorithm or proven
improvement over published BattLeDIM methods.

## Data and Methods

The official BattLeDIM L-Town dataset contains five-minute pressure and flow
observations from one simulated water network. Ridge models predict each pressure
sensor from the other measurements and time of day. Absolute standardized
residuals are smoothed over six samples and averaged. Notifications require three
hours above a fitting-period threshold, with 24-hour recovery and cooldown.

Eight 2018 chronological development folds compare the original baseline,
de-duplication, EWMA, CUSUM, compact classification, and leak-flow regression.
The recorded development protocol followed inspection of the original holdout,
so those findings are explicitly retrospective. Alternatives select parameters
using preceding calibration blocks only; unknown ongoing-leak classifier rows
are excluded rather than labelled normal.

For the held-out 2019 test, the selected baseline is fitted on all 2018 sensor
data, with its threshold and notification settings fixed. A final-test protocol,
input/model/code hashes and fitted threshold are saved before loading 2019.
Sensor-only predictions are persisted before opening labels. Notification state
carries from 28 days of late-2018 replay into the final year.

One notification matches at most one new pipe event before its end and within an
exclusive 48-hour deadline. Carry-in events are excluded from onset recall.
Temporal matching is not leak localization or official competition scoring.

## Results and Failure Analysis

| Evaluation | New Events Detected | Missed | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2018 development, de-duplicated baseline | 4 / 9 | 5 | 0.800 | 0.444 | 0.571 |
| 2019 held-out year, frozen baseline | 2 / 19 | 17 | 1.000 | 0.105 | 0.190 |

In development, de-duplication reduced unmatched notifications from 15 to 1,
with unchanged event recall. Alternative detectors did not outperform it under
the recorded settings. In 2019, only two notifications matched new events and
none were unmatched. Their mean delay was 3.04 hours (sample SD 0.059 hours),
which excludes the seventeen misses and therefore does not describe overall
responsiveness.

The frozen score exceeded threshold for approximately 74.9% of 2019 observations.
This is evidence of sustained abnormal-score periods, not proof of a particular
failure cause. Long open incidents can suppress separate onsets; fitting data
contain leaks, and the threshold/model may not transfer well. Distinguishing
these mechanisms requires a new recorded development study, not silent tuning
against the final-test labels. High precision here must not be interpreted as
operational success.

### Post-Test Diagnostic Finding

An explicitly post-test analysis joins all 19 new onsets to the original frozen
outcomes, measures pipe leakage within the exclusive first-48-hour deadline,
and reproduces the two notification timestamps exactly without changing the
detector. Detected events have median early mean leakage 21.82 m3/h, versus
0.050 m3/h for misses. However, two missed events have early mean flow 26.19
and 26.57 m3/h, larger than the smaller detected event (15.45 m3/h). The detector
does not simply find all large leaks and miss all small ones.
Early size is not ultimate burst size: p800, for example, averaged only 0.094
m3/h in its first 48 hours but reached an observed peak of 21.95 m3/h later.
The strict onset deadline can therefore penalize gradually developing leaks;
future studies should distinguish onset detection from later actionable growth
using rules recorded before evaluation, not relabel this completed result.

Fourteen of seventeen missed events started while an incident was already open;
in those fourteen windows, scores meeting the three-hour persistence criterion
were blocked by open-incident suppression. The other three missed windows never
met that criterion. These are notification-state diagnostics, not localized
leak detections or a counterfactual recall estimate: existing leaks and operating
conditions can contribute to an elevated score.

Between years, 17/33 pressure sensors change in yearly mean by more than one
2018 standard deviation. Inlet-meter means change from 98.20 to 126.68 m3/h
(p227), and 103.50 to 124.14 m3/h (p235); pump flow is kept separate. Monthly
profiles and matched month/hour differences are audited. Flow changes are not
pure demand measurements: leakage and operational regimes also contribute.
The same frozen model's median residual score increases from 0.767 on fitting
data to 1.735 in 2019. This fitting-versus-test comparison cannot distinguish
distribution shift, contaminated fitting data and overfitting causally.

The useful finding is therefore a combined onset-sensitivity and incident-policy
limitation under changed observed conditions, not an improved headline score.
See `POST_TEST_ANALYSIS_V1.md` and the dashboard's post-test diagnostics for
definitions, all-event plots, complete audits and separate provenance.

## Explainability and Human Oversight

Every pressure sensor's smoothed normalized residual divided by the sensor
count is an exact additive contribution to the aggregate baseline score.
The interface shows those components, observed/expected pressure and recent
flow context, using only measurements at or before the evidence timestamp.
Conditional Ridge expectations are not hydraulic leak-free counterfactuals.
Contributors identify influential sensor evidence, not confirmed leak locations.

Three fixed-rule development case studies cover a detected event, unmatched
notification and missed-event diagnostic. Ground truth appears only as
retrospective evaluation context. A missing alarm is never fabricated to support
an operator workflow.

A deterministic summary feeds a real LangGraph human interrupt. SQLite preserves
pending reviews and explicit approval/rejection decisions. Approval records
acceptance of a review recommendation; no control or dispatch action is executed.
No API key or LLM is required. Local reviewer identity is self-reported, so this
is not an authenticated production incident system.

## Research Next Steps

Hypothesis: conditioning each pressure sensor on other pressure/flow measurements
can absorb shared leak responses, while shifted conditions sustain residual
scores and long recovery rules suppress later onsets. These interacting causes
remain hypotheses, not confirmed explanations of each miss.

Next study: use paired nominal/leak WNTR simulations varying size, location,
demand uncertainty and pump/valve regimes. Compare conditional versus hydraulic
residuals, with separate notification-policy ablations. Record size bands and
event matching before evaluation; measure false alerts, recall and missed-inclusive
delay summaries. Validate simulation realism against measured statistics and
hold out leak locations, operating regimes and an independent network/year.
The already-observed 2019 outcomes cannot remain an untouched test for retuning.
LightGBM SHAP explanations and operator studies remain future work.

The licensed, session-isolated public replay demo is deployed; a short
demonstration video remains a handoff task. It is not production streaming.
The post-test diagnostic bundle is packaged separately from the unchanged
frozen evaluation for the hosted demo.

Public demo: https://water-watch-ukddepshzllqushgvdpzvv.streamlit.app/
Hydraulic simulation: [WNTR documentation](https://usepa.github.io/WNTR/userguide.html).
No claim of specific DC10 supervisor alignment is made without the vacancy brief.

Dataset: [Vrachimis et al., BattLeDIM, DOI 10.5281/zenodo.4017659](https://zenodo.org/records/4017659).
Procedures: `VALIDATION_PROTOCOL_V1.md`, `FINAL_2019_PROTOCOL.md`.
