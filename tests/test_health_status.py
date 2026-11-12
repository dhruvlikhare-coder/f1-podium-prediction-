"""Tests for the window-to-status function and run-level assessment."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.health.config import (
    STATUS_CRITICAL,
    STATUS_NORMAL,
    STATUS_UNKNOWN,
    STATUS_WARNING,
    HealthConfig,
    Thresholds,
)
from src.health.status import (
    assess_run,
    assess_runs,
    results_frame,
    window_scores_to_status,
)

NAN = float("nan")


def status_of(scores, thresholds, **kw):
    return window_scores_to_status(scores, thresholds, **kw).status


# --------------------------------------------------------------------------- #
# Status boundaries (pure aggregation function; thresholds warn=3, crit=5)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scores, expected", [
    ([0.0] * 10, STATUS_NORMAL),
    # warning boundary is inclusive
    ([3.0, 3.0], STATUS_WARNING),
    ([2.999, 2.999, 2.999], STATUS_NORMAL),
    # one elevated window is not enough for any alert
    ([0, 0, 3.5, 0, 0], STATUS_NORMAL),
    ([3.5, 0, 3.5, 0, 3.5], STATUS_NORMAL),
    # critical boundary is inclusive and needs 3 consecutive severe windows
    ([5.0, 5.0, 5.0], STATUS_CRITICAL),
    ([4.999, 4.999, 4.999], STATUS_WARNING),
    ([0, 4, 5, 5, 5, 0], STATUS_CRITICAL),
    # severe but not persistent -> warning, never critical
    ([5.0], STATUS_WARNING),
    ([0, 0, 9.0, 0, 0], STATUS_WARNING),
    ([5.0, 5.0], STATUS_WARNING),
    ([5, 5, 0, 5, 5], STATUS_WARNING),
    ([5, 4, 5], STATUS_WARNING),
    # warning persistence
    ([3.5, 4.0], STATUS_WARNING),
])
def test_status_boundaries(thresholds, scores, expected):
    assert status_of(scores, thresholds) == expected


def test_invalid_window_breaks_a_streak(thresholds):
    assert status_of([5, 5, 5, 5, 5], thresholds) == STATUS_CRITICAL
    assert status_of([5, 5, NAN, 5, 5], thresholds) == STATUS_WARNING


def test_time_gap_breaks_a_streak(thresholds):
    assert status_of([5, 5, 5], thresholds) == STATUS_CRITICAL
    assert status_of([5, 5, 5], thresholds, breaks=[False, False, True]) == STATUS_WARNING


def test_no_valid_scores_is_unknown_not_normal(thresholds):
    agg = window_scores_to_status([NAN, NAN], thresholds)
    assert agg.status == STATUS_UNKNOWN
    assert agg.peak_score is None
    assert window_scores_to_status([], thresholds).status == STATUS_UNKNOWN


def test_persistence_parameters_are_respected(thresholds):
    kw = dict(warning_min_consecutive=3, critical_min_consecutive=4)
    assert status_of([4, 4], thresholds, **kw) == STATUS_NORMAL
    assert status_of([4, 4, 4], thresholds, **kw) == STATUS_WARNING
    assert status_of([6, 6, 6], thresholds, **kw) == STATUS_WARNING
    assert status_of([6, 6, 6, 6], thresholds, **kw) == STATUS_CRITICAL


def test_counts_and_peak(thresholds):
    agg = window_scores_to_status([0, 3, 5, 6, 1], thresholds)
    assert (agg.n_valid, agg.n_elevated, agg.n_severe) == (5, 3, 2)
    assert agg.longest_elevated_streak == 3
    assert agg.longest_severe_streak == 2
    assert agg.peak_score == 6.0


def test_bad_aggregation_arguments_raise(thresholds):
    with pytest.raises(ValueError):
        window_scores_to_status([1, 2], thresholds, warning_min_consecutive=3,
                                critical_min_consecutive=2)
    with pytest.raises(ValueError):
        window_scores_to_status([1, 2], thresholds, breaks=[False])


def test_config_and_threshold_validation():
    with pytest.raises(ValueError):
        Thresholds(warning=5.0, critical=3.0)
    with pytest.raises(ValueError):
        Thresholds(warning=3.0, critical=3.0)
    with pytest.raises(ValueError):
        Thresholds(warning=float("nan"), critical=5.0)
    with pytest.raises(ValueError):
        HealthConfig(feature_cols=("a",), warning_min_consecutive=3, critical_min_consecutive=2)
    with pytest.raises(ValueError):
        HealthConfig(feature_cols=())
    with pytest.raises(ValueError):
        HealthConfig(feature_cols=("a",), min_coverage=0.0)
    with pytest.raises(ValueError):
        HealthConfig(feature_cols=("a",), sensor_ranges={"b": (0.0, 1.0)})


# --------------------------------------------------------------------------- #
# Run-level assessment: status from windows
# --------------------------------------------------------------------------- #
def test_normal_run(make_run, baseline, thresholds, config):
    res = assess_run(make_run([0.5] * 20), baseline, thresholds, config)
    assert res.status == STATUS_NORMAL
    assert res.n_valid_windows == 20
    assert res.observed_pattern is None
    assert res.coverage == 1.0


def test_single_noisy_window_never_critical(make_run, baseline, thresholds, config):
    values = [0.1] * 20
    values[10] = 50.0  # one huge window, plausible range is +-100
    res = assess_run(make_run(values), baseline, thresholds, config)
    assert res.status == STATUS_WARNING
    assert res.longest_severe_streak == 1
    assert any("did not persist" in r for r in res.reasons)


def test_single_moderately_elevated_window_is_normal(make_run, baseline, thresholds, config):
    values = [0.1] * 20
    values[10] = 4.0
    assert assess_run(make_run(values), baseline, thresholds, config).status == STATUS_NORMAL


def test_persistent_fault_is_critical_with_evidence(make_run, baseline, thresholds, config):
    values = [0.1] * 16 + [6.0, 7.0, 8.0, 6.5]
    res = assess_run(make_run(values), baseline, thresholds, config)
    assert res.status == STATUS_CRITICAL
    assert res.peak_score == 8.0
    assert res.observed_pattern == {"feature": "f1", "group": "thermal", "n_windows": 4}
    assert len(res.evidence) == config.max_evidence
    scores = [e["score"] for e in res.evidence]
    assert scores == sorted(scores, reverse=True)
    assert res.evidence[0]["level"] == "severe"
    assert res.evidence[0]["top_feature"] == "f1"
    assert res.evidence[0]["timestamp"] == 18.0


def test_pattern_follows_the_dominant_feature(make_run, baseline, thresholds, config):
    f1 = [0.0] * 20
    f2 = [0.0] * 16 + [6.0, 6.0, 6.0, 6.0]
    res = assess_run(make_run(f1, f2=f2), baseline, thresholds, config)
    assert res.status == STATUS_CRITICAL
    assert res.observed_pattern["feature"] == "f2"
    assert res.observed_pattern["group"] == "pressure"


def test_two_consecutive_elevated_windows_warn(make_run, baseline, thresholds, config):
    values = [0.0] * 18 + [4.0, 4.5]
    assert assess_run(make_run(values), baseline, thresholds, config).status == STATUS_WARNING


# --------------------------------------------------------------------------- #
# Missing columns / identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("col", ["f1", "f2", "run_id", "timestamp", "is_synthetic"])
def test_missing_column_is_unknown(make_run, baseline, thresholds, config, col):
    df = make_run([0.0] * 20).drop(columns=[col])
    res = assess_run(df, baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert "missing_columns" in res.reasons
    assert col in res.warnings[0]


def test_empty_input_is_unknown(make_run, baseline, thresholds, config):
    res = assess_run(make_run([0.0] * 5).iloc[0:0], baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN and "no_windows" in res.reasons


def test_null_run_id_is_unknown(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20)
    df.loc[3, "run_id"] = None
    res = assess_run(df, baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN and "missing_run_id" in res.reasons


def test_assess_run_rejects_multiple_runs(make_run, baseline, thresholds, config):
    df = pd.concat([make_run([0.0] * 10, run_id="a"), make_run([0.0] * 10, run_id="b")])
    with pytest.raises(ValueError, match="assess_runs"):
        assess_run(df, baseline, thresholds, config)


def test_baseline_config_mismatch_raises(make_run, baseline, thresholds):
    other = HealthConfig(feature_cols=("f1", "other"))
    with pytest.raises(ValueError):
        assess_run(make_run([0.0] * 10), baseline, thresholds, other)


# --------------------------------------------------------------------------- #
# Invalid, stale and insufficient data
# --------------------------------------------------------------------------- #
def test_few_nan_windows_are_excluded_and_reported(make_run, baseline, thresholds, config):
    values = [0.0] * 20
    values[2] = NAN
    values[7] = np.inf
    res = assess_run(make_run(values), baseline, thresholds, config)
    assert res.status == STATUS_NORMAL
    assert res.n_valid_windows == 18
    assert res.coverage == pytest.approx(0.9)
    assert any("non finite" in w for w in res.warnings)


def test_mostly_invalid_data_is_unknown(make_run, baseline, thresholds, config):
    values = [NAN] * 10 + [0.0] * 10
    res = assess_run(make_run(values), baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert "low_coverage" in res.reasons


def test_out_of_range_value_is_invalid_not_critical(make_run, baseline, thresholds, config):
    values = [0.0] * 20
    values[10] = 1e6
    res = assess_run(make_run(values), baseline, thresholds, config)
    assert res.status == STATUS_NORMAL
    assert res.n_valid_windows == 19
    assert res.peak_score == 0.0
    assert any("out of range f1" in w for w in res.warnings)


def test_all_out_of_range_is_unknown(make_run, baseline, thresholds, config):
    res = assess_run(make_run([1e6] * 20), baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert "insufficient_windows" in res.reasons


def test_non_numeric_values_are_invalid(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20).astype({"f1": object})
    df.loc[4, "f1"] = "bad"
    res = assess_run(df, baseline, thresholds, config)
    assert res.n_valid_windows == 19
    assert res.status == STATUS_NORMAL


@pytest.mark.parametrize("values", [[0.0] * 3, [9.0] * 3, []])
def test_insufficient_windows_is_unknown_even_if_extreme(make_run, baseline, thresholds,
                                                         config, values):
    res = assess_run(make_run(values), baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert res.status != STATUS_CRITICAL


def test_sparse_windows_low_coverage_is_unknown(make_run, baseline, thresholds, config):
    ts = list(range(0, 40, 2))  # every other second: 20 windows over 39 slots
    res = assess_run(make_run([0.0] * 20, timestamps=ts), baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert "low_coverage" in res.reasons
    assert res.coverage == pytest.approx(20 / 39)


def test_gap_splits_a_critical_streak(make_run, baseline, thresholds, config):
    # severe windows at t=8,9 then a 2s hole, then t=11 -> not 3 consecutive in time
    ts = list(range(0, 10)) + list(range(11, 21))
    values = [0.0] * 8 + [6.0, 6.0, 6.0] + [0.0] * 9
    res = assess_run(make_run(values, timestamps=ts), baseline, thresholds, config)
    assert res.n_gaps == 1
    assert res.longest_severe_streak == 2
    assert res.status == STATUS_WARNING
    contiguous = assess_run(make_run(values), baseline, thresholds, config)
    assert contiguous.status == STATUS_CRITICAL


def test_stale_data_is_unknown(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20)  # last timestamp = 19
    stale = assess_run(df, baseline, thresholds, config, as_of=19 + 100)
    assert stale.status == STATUS_UNKNOWN
    assert "stale_data" in stale.reasons
    assert stale.stale is True and stale.age_s == pytest.approx(100)
    fresh = assess_run(df, baseline, thresholds, config, as_of=19 + 5)
    assert fresh.status == STATUS_NORMAL and fresh.stale is False
    boundary = assess_run(df, baseline, thresholds, config, as_of=19 + 10)
    assert boundary.status == STATUS_NORMAL  # age == max_age_s is still fresh


def test_freshness_not_evaluated_without_as_of(make_run, baseline, thresholds, config):
    res = assess_run(make_run([0.0] * 20), baseline, thresholds, config)
    assert res.stale is None and res.age_s is None


def test_datetime_timestamps_and_staleness(make_run, baseline, thresholds, config):
    ts = pd.date_range("2026-03-01 12:00:00", periods=20, freq="s")
    df = make_run([0.0] * 20, timestamps=ts)
    last = ts[-1]
    ok = assess_run(
        df,
        baseline,
        thresholds,
        config,
        as_of=last.to_pydatetime() + timedelta(seconds=3),
    )
    assert ok.status == STATUS_NORMAL and ok.age_s == pytest.approx(3)
    old = assess_run(
        df,
        baseline,
        thresholds,
        config,
        as_of=last.to_pydatetime() + timedelta(minutes=5),
    )
    assert old.status == STATUS_UNKNOWN and "stale_data" in old.reasons
    assert isinstance(ok.last_timestamp, str)  # JSON-friendly


def test_duplicate_timestamps_are_dropped_and_reported(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    res = assess_run(df, baseline, thresholds, config)
    assert res.n_valid_windows == 20
    assert any("duplicate" in w for w in res.warnings)


def test_out_of_order_rows_are_sorted(make_run, baseline, thresholds, config):
    df = make_run([0.1] * 16 + [6.0, 7.0, 8.0, 6.5])
    shuffled = df.sample(frac=1.0, random_state=0)
    res = assess_run(shuffled, baseline, thresholds, config)
    assert res.status == STATUS_CRITICAL
    assert any("out-of-order" in w for w in res.warnings)


def test_bad_timestamps_are_ignored_but_reported(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20)
    df.loc[3, "timestamp"] = NAN
    res = assess_run(df, baseline, thresholds, config)
    assert res.n_windows == 20 and res.n_valid_windows == 19
    assert any("invalid timestamp" in w for w in res.warnings)
    all_bad = make_run([0.0] * 10)
    all_bad["timestamp"] = NAN
    assert assess_run(all_bad, baseline, thresholds, config).status == STATUS_UNKNOWN


@pytest.mark.parametrize("bad", ["missing_col", "nan", "short", "stale", "sparse", "oor"])
def test_unusable_input_is_never_normal(make_run, baseline, thresholds, config, bad):
    df = make_run([0.0] * 20)
    kwargs = {}
    if bad == "missing_col":
        df = df.drop(columns=["f2"])
    elif bad == "nan":
        df["f1"] = NAN
    elif bad == "short":
        df = df.iloc[:2]
    elif bad == "stale":
        kwargs["as_of"] = 10_000
    elif bad == "sparse":
        df["timestamp"] = df["timestamp"] * 5
    elif bad == "oor":
        df["f1"] = 1e9
    res = assess_run(df, baseline, thresholds, config, **kwargs)
    assert res.status == STATUS_UNKNOWN
    assert res.reasons, "unknown must always say why"


# --------------------------------------------------------------------------- #
# Synthetic labelling (FR-81 / NFR-13)
# --------------------------------------------------------------------------- #
def test_synthetic_input_is_labelled(make_run, baseline, thresholds, config):
    res = assess_run(make_run([0.0] * 20, is_synthetic=True), baseline, thresholds, config)
    assert res.is_synthetic is True
    assert "SYNTHETIC" in res.disclaimer
    assert "not a confirmed mechanical diagnosis" in res.disclaimer
    assert res.to_dict()["is_synthetic"] is True


def test_non_synthetic_input_keeps_diagnosis_disclaimer(make_run, baseline, thresholds, config):
    res = assess_run(make_run([0.0] * 20, is_synthetic=False), baseline, thresholds, config)
    assert res.is_synthetic is False
    assert "SYNTHETIC" not in res.disclaimer
    assert "not a confirmed mechanical diagnosis" in res.disclaimer


def test_mixed_synthetic_flags_are_treated_as_synthetic(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20)
    df.loc[0, "is_synthetic"] = False
    res = assess_run(df, baseline, thresholds, config)
    assert res.is_synthetic is True
    assert any("mixed is_synthetic" in w for w in res.warnings)


@pytest.mark.parametrize("bad", [None, "False", "yes", 2])
def test_unusable_synthetic_flag_is_unknown(make_run, baseline, thresholds, config, bad):
    df = make_run([0.0] * 20).astype({"is_synthetic": object})
    df.loc[0, "is_synthetic"] = bad
    res = assess_run(df, baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert "synthetic_flag_missing_or_invalid" in res.reasons
    assert res.is_synthetic is None


def test_unknown_results_still_carry_a_disclaimer(make_run, baseline, thresholds, config):
    res = assess_run(make_run([0.0] * 2), baseline, thresholds, config)
    assert res.status == STATUS_UNKNOWN
    assert "not a confirmed mechanical diagnosis" in res.disclaimer


# --------------------------------------------------------------------------- #
# Run aggregation: runs are assessed independently, never pooled
# --------------------------------------------------------------------------- #
def test_each_run_gets_its_own_status(make_run, baseline, thresholds, config):
    df = pd.concat([
        make_run([0.0] * 20, run_id="normal"),
        make_run([0.0] * 16 + [6.0] * 4, run_id="faulty"),
        make_run([0.0] * 19 + [30.0], run_id="spike"),
        make_run([0.0] * 3, run_id="short"),
    ], ignore_index=True)
    results = assess_runs(df, baseline, thresholds, config)
    by_run = {r.run_id: r.status for r in results}
    assert by_run == {"normal": STATUS_NORMAL, "faulty": STATUS_CRITICAL,
                      "spike": STATUS_WARNING, "short": STATUS_UNKNOWN}


def test_windows_from_different_runs_are_not_pooled(make_run, baseline, thresholds, config):
    # Run A ends with 2 severe windows, run B starts with 2 severe windows at the
    # same times. Pooled they would form a critical streak; per run they must not.
    a = make_run([0.0] * 18 + [6.0, 6.0], run_id="a")
    b = make_run([6.0, 6.0] + [0.0] * 18, run_id="b", start=18.0)
    results = assess_runs(pd.concat([a, b], ignore_index=True), baseline, thresholds, config)
    assert [r.status for r in results] == [STATUS_WARNING, STATUS_WARNING]


def test_one_runs_problem_does_not_change_another(make_run, baseline, thresholds, config):
    clean = make_run([0.0] * 20, run_id="clean")
    alone = assess_run(clean, baseline, thresholds, config)
    mixed = assess_runs(pd.concat([clean, make_run([9.0] * 20, run_id="bad")],
                                  ignore_index=True), baseline, thresholds, config)
    assert mixed[0].status == alone.status == STATUS_NORMAL
    assert mixed[1].status == STATUS_CRITICAL


def test_results_frame_has_one_row_per_run(make_run, baseline, thresholds, config):
    df = pd.concat([make_run([0.0] * 20, run_id=i) for i in range(3)], ignore_index=True)
    frame = results_frame(assess_runs(df, baseline, thresholds, config))
    assert len(frame) == 3
    assert {"run_id", "status", "is_synthetic", "coverage", "disclaimer"} <= set(frame.columns)


def test_assess_runs_without_run_column_is_unknown(make_run, baseline, thresholds, config):
    df = make_run([0.0] * 20).drop(columns=["run_id"])
    results = assess_runs(df, baseline, thresholds, config)
    assert len(results) == 1 and results[0].status == STATUS_UNKNOWN


def test_null_run_ids_do_not_leak_into_other_runs(make_run, baseline, thresholds, config):
    df = pd.concat([make_run([0.0] * 20, run_id="ok"),
                    make_run([0.0] * 20, run_id=None)], ignore_index=True)
    statuses = {str(r.run_id): r.status for r in assess_runs(df, baseline, thresholds, config)}
    assert statuses == {"ok": STATUS_NORMAL, "None": STATUS_UNKNOWN}


def test_results_are_json_serialisable(make_run, baseline, thresholds, config):
    import json

    ts = pd.date_range("2026-03-01 12:00:00", periods=20, freq="s")
    for df in (make_run([0.0] * 16 + [6.0] * 4), make_run([0.0] * 16 + [6.0] * 4, timestamps=ts),
               make_run([0.0] * 3)):
        res = assess_run(df, baseline, thresholds, config, as_of=df["timestamp"].iloc[-1])
        payload = json.loads(json.dumps(res.to_dict()))
        assert payload["status"] in {"normal", "warning", "critical", "unknown"}
        assert payload["disclaimer"] and payload["thresholds"] == thresholds.to_dict()
