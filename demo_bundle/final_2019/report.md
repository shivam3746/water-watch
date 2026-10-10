# Held-Out 2019 Test

Same simulated network; no parameters fitted or calibrated on 2019.

| Metric | Value |
| --- | --- |
| model | baseline_deduplicated |
| precision | 1.0 |
| recall | 0.10526315789473684 |
| f1 | 0.1904761904761905 |
| avg_detection_delay_hours | 3.041666666666667 |
| false_alarms_per_week | 0.0 |
| true_positives | 2 |
| false_positives | 0 |
| false_negatives | 17 |
| alarm_count | 2 |
| eligible_leak_events | 19 |
| excluded_carry_in_events | 4 |
| duplicate_alarms | 0 |
| carry_in_only_alarms | 0 |
| ambiguous_matches | 0 |
| test_start | 2019-01-01T00:00:00 |
| test_end_exclusive | 2020-01-01T00:00:00 |
| observed_weeks | 52.142857142857146 |
| max_detection_delay_hours | 48 |
| matching_rule | one-to-one temporal matching; earliest active deadline first; [onset, end) |
| false_alarm_rule | all test alarms unmatched to a new eligible event, including duplicates and carry-in-only alarms |
| zero_denominator_rule | precision/recall/F1 are 0 when their denominator is 0; missing delay is null |
| detection_delay_std_hours | 0.058925565098879064 |

One-to-one temporal matching is not localization. Unmatched notifications do not imply a leak-free network.
Missed-event delay is null; reported delay includes detected events only. See event/notification audits and frozen/prediction manifests.