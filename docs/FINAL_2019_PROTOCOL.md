# Final 2019 Test V1

Recorded 8 October 2026 before loading any 2019 measurements or outcomes.
Official downloads may be transferred and checksum-verified without parsing.
This is a held-out year of the same simulated network, not external-network
validation and not the official competition scoring protocol.

The development-selected candidate is the de-duplicated residual baseline.
Fit all Ridge pressure models and residual scales on the complete 2018 sensor
observations. Fix the threshold to the 2018 score quantile 0.995; six-sample
smoothing, three-hour persistence, 24-hour continuous normal recovery, and
24-hour cooldown remain unchanged. No parameter is calibrated on 2019.
Fitting data contain leaks; this is an acknowledged limitation.

Replay the last 28 days of 2018 followed by all 2019 observations, without
resetting notification state at the year boundary. Warm the six-sample residual
smoother with five extra 2018 rows. Filter emitted notifications to 2019 only.
No 2019 sensor normalization, fitting, threshold selection, or imputation occurs.

Freeze the protocol hash, implementation hashes, 2018 input hash, fitted model
hash, sensor names, and fitted threshold before reading 2019 data. Generate and
save 2019 scores and notifications using only sensor files, then load leakage
labels for evaluation. A completed final evaluation cannot be overwritten by the
runner. Corrections require a documented audit, not silent retuning.

Require complete, aligned, uninterrupted five-minute observations for each year.
Source-first-row positive leaks have unknown onset and are excluded as carry-in.
One notification matches one new pipe event before its end and the exclusive
48-hour deadline. Duplicates, late and other unmatched alerts count as false
notifications for new-onset detection, even when old leaks remain active.
Missed events have null delay; report event/notification audits, precision,
recall, F1, false alerts/week, detected-event delay mean and sample standard
deviation, and end-of-year truncated follow-up. Temporal matching does not prove
localization. No LLM or explanation output affects predictions or evaluation.

Report the result whether favourable or unfavourable. Do not make further model
changes against this year's labels while continuing to call it an untouched test.
