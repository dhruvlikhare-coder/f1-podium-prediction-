"""Tests for the baseline, threshold protocol, freezing and run-level split handling."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.health.config import (
    STATUS_CRITICAL,
    CalibrationSpec,
    HealthConfig,
    Thresholds,
    load_health_config,
)
from src.health.scoring import NormalBaseline, invalid_row_mask
from src.health.status import assess_runs
from src.health.thresholds import (
    assert_run_disjoint,
    calibrate_thresholds,
    fit_and_calibrate,
    freeze_model,
    load_frozen_model,
)

SPEC = CalibrationSpec(warning_quantile=0.99, critical_quantile=0.999,
                       min_critical_margin=0.5, min_windows=100)


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def _normal_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"f1": rng.normal(10.0, 2.0, n), "f2": rng.normal(0.0, 1.0, n)})


def test_baseline_learns_center_and_scale(config):
    base = NormalBaseline.fit(_normal_frame(), config)
    assert base.center[0] == pytest.approx(10.0, abs=0.3)
    assert base.scale[0] == pytest.approx(2.0, abs=0.3)
    row = pd.DataFrame({"f1": [16.0], "f2": [0.0]})
    out = base.score(row)
    assert out["anomaly_score"].iloc[0] == pytest.approx(3.0, abs=0.5)
    assert out["top_feature"].iloc[0] == "f1"


def test_baseline_is_robust_to_a_few_outliers(config):
    df = _normal_frame()
    df.loc[:3, "f1"] = 90.0  # 4 wild windows must barely move median/MAD
    base = NormalBaseline.fit(df, config)
    assert base.center[0] == pytest.approx(10.0, abs=0.5)


def test_baseline_fit_ignores_invalid_rows(config):
    df = _normal_frame()
    clean = NormalBaseline.fit(df, config)
    dirty = df.copy()
    dirty.loc[0, "f1"] = np.nan
    dirty.loc[1, "f2"] = 1e6  # outside plausibility range
    refit = NormalBaseline.fit(dirty, config)
    assert refit.center[0] == pytest.approx(clean.center[0], abs=0.05)


def test_baseline_fit_failures(config):
    with pytest.raises(ValueError, match="at least"):
        NormalBaseline.fit(_normal_frame(n=10), config)
    with pytest.raises(ValueError, match="missing columns"):
        NormalBaseline.fit(_normal_frame().drop(columns=["f2"]), config)
    constant = _normal_frame().assign(f2=1.0)
    with pytest.raises(ValueError, match="constant"):
        NormalBaseline.fit(constant, config)


def test_score_marks_unusable_rows(config):
    base = NormalBaseline.fit(_normal_frame(), config)
    rows = pd.DataFrame({"f1": [10.0, np.nan, np.inf], "f2": [0.0, 0.0, 0.0]})
    out = base.score(rows)
    assert out["anomaly_score"].notna().tolist() == [True, False, False]
    assert out["top_feature"].isna().tolist() == [False, True, True]


def test_baseline_roundtrip(config):
    base = NormalBaseline.fit(_normal_frame(), config)
    assert NormalBaseline.from_dict(base.to_dict()) == base


def test_invalid_row_mask_counts(config):
    df = pd.DataFrame({"f1": [0.0, np.nan, 500.0, 1.0], "f2": [0.0, 0.0, 0.0, "x"]})
    valid, counts = invalid_row_mask(df, config)
    assert valid.tolist() == [True, False, False, False]
    assert counts == {"non_finite:f1": 1, "non_finite:f2": 1, "out_of_range:f1": 1}


# --------------------------------------------------------------------------- #
# Threshold calibration
# --------------------------------------------------------------------------- #
def test_thresholds_are_quantiles_of_normal_validation_scores():
    # wide enough that the 0.99 and 0.999 quantiles differ by more than the minimum margin
    scores = np.linspace(0.0, 100.0, 2001)
    t = calibrate_thresholds(scores, SPEC)
    assert t.warning == pytest.approx(np.quantile(scores, 0.99))
    assert t.critical == pytest.approx(np.quantile(scores, 0.999))
    assert t.warning < t.critical


def test_critical_keeps_a_minimum_margin_above_warning():
    t = calibrate_thresholds(np.full(500, 2.0) + np.linspace(0, 1e-6, 500), SPEC)
    assert t.critical >= t.warning + SPEC.min_critical_margin


def test_calibration_needs_enough_windows_and_ignores_nan():
    with pytest.raises(ValueError, match="at least"):
        calibrate_thresholds(np.linspace(0, 5, 50), SPEC)
    scores = np.concatenate([np.linspace(0, 5, 200), [np.nan] * 500])
    assert calibrate_thresholds(scores, SPEC).warning < 5.0


def test_calibration_spec_validation():
    with pytest.raises(ValueError):
        CalibrationSpec(warning_quantile=0.999, critical_quantile=0.99)
    with pytest.raises(ValueError):
        CalibrationSpec(min_critical_margin=0.0)


# --------------------------------------------------------------------------- #
# Run-level splits
# --------------------------------------------------------------------------- #
def test_assert_run_disjoint_accepts_disjoint_splits():
    assert_run_disjoint({"train": ["a", "b"], "val": ["c"], "test": ["d", "e"]})


def test_assert_run_disjoint_names_the_offending_run():
    with pytest.raises(ValueError, match="r2"):
        assert_run_disjoint({"train": ["r1", "r2"], "val": ["r3"], "test": ["r2", "r4"]})


def test_fit_rejects_a_run_shared_by_train_and_validation(config, toy_runs):
    rng = np.random.default_rng(1)
    train = toy_runs(rng, ["t0", "t1", "shared"])
    val = toy_runs(rng, ["v0", "shared"])
    with pytest.raises(ValueError, match="shared"):
        fit_and_calibrate(train, val, config, SPEC, train_normal_runs=["t0", "t1"],
                          val_normal_runs=["v0"])


def test_fit_uses_only_the_named_normal_runs(config, toy_runs):
    rng = np.random.default_rng(2)
    train = toy_runs(rng, [f"t{i}" for i in range(6)])
    faulty = toy_runs(rng, ["t_fault"], drift_from=0, drift=50.0)
    val = toy_runs(rng, [f"v{i}" for i in range(4)])
    normal_ids = [f"t{i}" for i in range(6)]
    clean = fit_and_calibrate(train, val, config, SPEC, train_normal_runs=normal_ids,
                              val_normal_runs=[f"v{i}" for i in range(4)])
    with_fault = fit_and_calibrate(pd.concat([train, faulty], ignore_index=True), val, config,
                                   SPEC, train_normal_runs=normal_ids,
                                   val_normal_runs=[f"v{i}" for i in range(4)])
    assert with_fault.baseline == clean.baseline
    assert with_fault.provenance["train_normal_runs"] == sorted(normal_ids)


# --------------------------------------------------------------------------- #
# Freezing
# --------------------------------------------------------------------------- #
@pytest.fixture
def fitted(config, toy_runs):
    rng = np.random.default_rng(3)
    train = toy_runs(rng, [f"t{i}" for i in range(6)])
    val = toy_runs(rng, [f"v{i}" for i in range(4)])
    return fit_and_calibrate(train, val, config, SPEC,
                             train_normal_runs=[f"t{i}" for i in range(6)],
                             val_normal_runs=[f"v{i}" for i in range(4)])


def test_freeze_and_load_roundtrip(fitted, tmp_path):
    path = tmp_path / "frozen.json"
    digest = freeze_model(fitted, path)
    loaded = load_frozen_model(path)
    assert loaded.thresholds == fitted.thresholds
    assert loaded.baseline == fitted.baseline
    assert loaded.provenance == fitted.provenance
    assert json.loads(path.read_text())["sha256"] == digest


def test_freeze_refuses_to_overwrite(fitted, tmp_path):
    path = tmp_path / "frozen.json"
    freeze_model(fitted, path)
    with pytest.raises(FileExistsError):
        freeze_model(fitted, path)
    freeze_model(fitted, path, overwrite=True)  # explicit opt-in still works


def test_tampered_thresholds_are_detected(fitted, tmp_path):
    path = tmp_path / "frozen.json"
    freeze_model(fitted, path)
    raw = json.loads(path.read_text())
    raw["thresholds"]["critical"] += 1.0  # "tuned" after seeing test results
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="integrity"):
        load_frozen_model(path)


def test_missing_hash_is_rejected(fitted, tmp_path):
    path = tmp_path / "frozen.json"
    freeze_model(fitted, path)
    raw = json.loads(path.read_text())
    raw.pop("sha256")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="integrity"):
        load_frozen_model(path)


# --------------------------------------------------------------------------- #
# Toy end-to-end, evaluated by run (not a claim about the real generator)
# --------------------------------------------------------------------------- #
def test_toy_run_level_evaluation(config, toy_runs, tmp_path):
    rng = np.random.default_rng(42)
    train_ids = [f"train_{i}" for i in range(8)]
    val_ids = [f"val_{i}" for i in range(4)]
    test_normal = [f"test_ok_{i}" for i in range(6)]
    test_fault = [f"test_fault_{i}" for i in range(4)]
    train = toy_runs(rng, train_ids)
    val = toy_runs(rng, val_ids)
    test = pd.concat([toy_runs(rng, test_normal),
                      toy_runs(rng, test_fault, drift_from=30, drift=8.0)], ignore_index=True)
    assert_run_disjoint({"train": train_ids, "val": val_ids, "test": test_normal + test_fault})

    model = fit_and_calibrate(train, val, config, SPEC, train_normal_runs=train_ids,
                              val_normal_runs=val_ids)
    path = tmp_path / "frozen.json"
    freeze_model(model, path)
    frozen = load_frozen_model(path)  # evaluation only ever uses the frozen copy

    results = assess_runs(test, frozen.baseline, frozen.thresholds, config)
    by_run = {r.run_id: r for r in results}
    assert len(by_run) == len(test_normal) + len(test_fault)
    for rid in test_fault:
        assert by_run[rid].status == STATUS_CRITICAL
        assert by_run[rid].is_synthetic is True
    for rid in test_normal:
        assert by_run[rid].status != STATUS_CRITICAL  # one noisy window can't do it


# --------------------------------------------------------------------------- #
# YAML procedure config
# --------------------------------------------------------------------------- #
def test_health_yaml_loads(tmp_path):
    path = tmp_path / "health.yaml"
    path.write_text(
        "features: [a, b]\n"
        "sensor_ranges: {a: [0, 10]}\n"
        "feature_groups: {a: thermal}\n"
        "data_quality: {window_step_s: 2, min_valid_windows: 7}\n"
        "run_status: {warning_min_consecutive: 2, critical_min_consecutive: 4}\n"
        "calibration: {warning_quantile: 0.95, critical_quantile: 0.99}\n",
        encoding="utf-8")
    cfg, spec = load_health_config(path)
    assert isinstance(cfg, HealthConfig) and cfg.feature_cols == ("a", "b")
    assert cfg.sensor_ranges["a"] == (0.0, 10.0)
    assert cfg.effective_max_gap_s == 3.0
    assert cfg.critical_min_consecutive == 4 and cfg.min_valid_windows == 7
    assert spec.warning_quantile == 0.95
    assert not hasattr(cfg, "thresholds"), "thresholds are fitted and frozen, never configured"
    assert Thresholds(1.0, 2.0).to_dict() == {"warning": 1.0, "critical": 2.0}


def test_shipped_health_yaml_parses():
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "configs" / "health.yaml"
    cfg, spec = load_health_config(path)
    assert cfg.feature_cols and cfg.critical_min_consecutive >= cfg.warning_min_consecutive
    assert spec.warning_quantile < spec.critical_quantile
