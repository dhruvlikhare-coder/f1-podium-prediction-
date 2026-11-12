# Car-health status: thresholds and run-level rules

Covers SRS FR-77 to FR-84 and AC-13/AC-14. Procedure choices are in
`configs/health.yaml` and were fixed before any evaluation.

**What this is.** An anomaly indicator over telemetry windows. The synthetic
generator's sensor values and injected faults are simulation assumptions, so a
good result means "the prototype detects the injected patterns", not "it can
diagnose a real F1 power unit". Every result carries `is_synthetic` and a
disclaimer string.

## Protocol (nothing here touches test runs)

1. **Fit** the baseline on *normal train runs*: per-feature median and robust
   scale (1.4826 x MAD). Window score = largest absolute robust z across
   features; the responsible feature is kept as evidence.
2. **Calibrate** thresholds on *normal validation runs*:
   `warning` = 99th percentile of window scores, `critical` = 99.9th percentile,
   forced to be at least `warning + 0.5`.
3. **Freeze** baseline, thresholds and provenance to JSON with a SHA-256 hash
   (`freeze_model`). It will not overwrite an existing file, and
   `load_frozen_model` rejects any edited file.
4. **Evaluate** on test runs using only the frozen file. Splits are by run:
   `assert_run_disjoint` fails if a run id appears in two splits.

## Window to run status

A window is *elevated* at `score >= warning` and *severe* at `score >= critical`
(inclusive). Invalid windows and time gaps break streaks.

| Status | Rule |
|---|---|
| `critical` | 3 or more consecutive severe windows |
| `warning` | 2 or more consecutive elevated windows, or any severe window that did not persist |
| `normal` | otherwise |
| `unknown` | overrides all of the above (below) |

So one noisy window can raise a warning at most; it can never produce `critical`.

The `SynthetIc/synthetic_telemetry.py` window output is compatible with the shipped
`configs/health.yaml` mapping. Its `is_anomaly`, `anomaly_frac`, and `window_type`
columns are labels for evaluation only; they are not health-model features. Results
on the generator only show behavior on its assumed injected patterns.

## When the answer is `unknown` (never inferred `normal`)

Missing required columns (run id, timestamp, `is_synthetic`, features), null run
id, missing or invalid synthetic flag, no valid timestamps, fewer than
`min_valid_windows` valid windows, coverage below `min_coverage`, or data older
than `max_age_s` at the `as_of` time. Non-finite and physically implausible
values make a window invalid; they are reported in `warnings`, not scored as
anomalies. Duplicate timestamps are dropped and out-of-order rows are sorted,
both with a warning.

## Output

`RunHealth`: `status`, `reasons`, `warnings`, `peak_score`, window counts and
streaks, `coverage`, `last_timestamp`, `age_s`, `stale`, `n_gaps`,
`observed_pattern` (most frequent contributing feature and its optional group),
top-scoring `evidence` windows, `is_synthetic`, the thresholds used, and the
disclaimer. This is separate from podium probabilities and must not feed them
(FR-83). Any pre-race reliability feature should start from the mechanical-DNF
history already in `clean_results.py`, and only after a chronological with/without
backtest (FR-84).