# Post-Test Diagnostic Protocol V1

Recorded 10 October 2026, after observing the frozen 2019 result. This is
exploratory explanation, not another independent validation or parameter search.
No detector, threshold, alarm policy, event matching or final-test output changes.

1. Join all 19 eligible 2019 onsets to their original detected/missed outcome.
   Keep four carry-in events separate. Measure each pipe's mean and peak leakage
   in [onset, min(onset + 48 hours, event end, observation end)). Plot every event,
   not selected examples. Full observed-event peaks are secondary and censored
   where the year ends. Do not choose a size cutoff to maximize separation.
2. Replay only the frozen notification state, including the original late-2018
   warmup. Require emitted 2019 timestamps to exactly match saved predictions.
   Record incident-open state at onset, score exceedance, three-hour persistence
   readiness and blocking by an already-open incident in each matching window.
   These describe policy state, not confirmed physical causes of a miss.
3. Compare the complete 2018 and 2019 raw sensors: mean, spread, daily-mean
   distribution, repeated-value fraction and matched month/hour mean profiles.
   Standardize mean shifts using 2018 variability only. Use descriptive KS
   distance on daily means, without IID p-values for autocorrelated observations.
   Show each flowmeter separately: the two inlet meters and pump meter must not
   be summed or equated to consumption. Without AMRs/operating-state controls,
   flow changes cannot be attributed to demand alone or sensor faults.
4. Compare all non-carry-in 2018/2019 leak sizes with identical early windows.
   Full-year 2018 raw descriptors are NOT the nine-event development evaluation.
   Compare the same frozen model's 2018 fitting scores with saved 2019 scores,
   explicitly noting that in-sample versus out-of-sample scores confound
   generalization error, leak contamination and condition changes.
5. Save source/code/output hashes separately from the frozen evaluation. Report
   what supports and contradicts the small-leak hypothesis. With two detections,
   avoid causal or general sensitivity claims. New tuning needs a new protocol
   and a new independent holdout; 2019 is now observed diagnostic material.

Flow units: m3/h; pressure: m. Dataset source:
https://zenodo.org/records/4017659 (CC BY 4.0).
Official units/conditions:
https://battledim.ucy.ac.cy/wp-content/uploads/2020/01/BattLeDIM_Problem_Description_and_Rules-v1.3.pdf

Candidate hypothesis (not a conclusion): conditional pressure residuals can
absorb shared leak responses; shifted conditions can keep the score elevated,
and incident-level suppression can conflict with new-onset recall. Test these
mechanisms using paired nominal/leak hydraulic simulations with uncertain
demands, operating regimes and leak sizes, holding out locations and scenarios.
WNTR: https://usepa.github.io/WNTR/userguide.html
Specific DC10 supervisor alignment requires the actual vacancy/project brief;
it is not asserted from the project number alone.
