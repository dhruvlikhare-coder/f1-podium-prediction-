from pathlib import Path

import numpy as np
import pandas as pd

from src.health import (
    HealthConfig,
    NormalBaseline,
    Thresholds,
    assess_run,
    fit_and_calibrate,
    load_health_config,
)
from src.health.scoring import invalid_rows_mask
from src.health.status import window_scores_to_status
from SynthetIc.synthetic_telemetry import Config as SyntheticConfig
from SynthetIc.synthetic_telemetry import make_windows, simulate_dataset


def test_window_status_resets_streak_after_gap():
    result = window_scores_to_status(
        [5.0, 5.0, 5.0],
        Thresholds(warning=1.0, critical=4.0),
        warning_min_consecutive=2,
        critical_min_consecutive=3,
        breaks=[False, False, True],
    )

    assert result.status == "warning"
    assert result.n_elevated == 3
    assert result.n_severe == 3
    assert result.longest_severe_streak == 2


def test_invalid_rows_mask_checks_sensor_bounds_for_finite_values():
    config = HealthConfig(
        feature_cols=("temperature",),
        sensor_ranges={"temperature": (-10.0, 10.0)},
    )
    frame = pd.DataFrame({"temperature": [0.0, 12.0, np.nan]})

    valid, counts = invalid_rows_mask(frame, config)

    assert valid.tolist() == [True, False, False]
    assert counts == {
        "out_of_range:temperature": 1,
        "non_finite:temperature": 1,
    }


def test_baseline_serialization_round_trip_accepts_legacy_center_key():
    baseline = NormalBaseline(features=("temperature",), center=(1.5,), scale=(2.0,))

    assert NormalBaseline.from_dict(baseline.to_dict()) == baseline
    assert (
        NormalBaseline.from_dict(
            {"feature": ["temperature"], "centere": [1.5], "scale": [2.0]}
        )
        == baseline
    )


def test_assess_run_returns_labeled_warning_with_evidence():
    config = HealthConfig(
        feature_cols=("temperature",),
        window_step_s=1.0,
        min_valid_windows=4,
        min_coverage=0.8,
        max_age_s=10.0,
        warning_min_consecutive=2,
        critical_min_consecutive=3,
    )
    normal = pd.DataFrame({"temperature": np.arange(-5.0, 5.0)})
    baseline = NormalBaseline.fit(normal, config, min_rows=5)
    windows = pd.DataFrame(
        {
            "run_id": ["run-1"] * 5,
            "timestamp": [0, 1, 2, 3, 4],
            "is_synthetic": [True] * 5,
            "temperature": [20.0] * 5,
        }
    )

    result = assess_run(
        windows,
        baseline,
        Thresholds(warning=1.0, critical=100.0),
        config,
        as_of=5.0,
    )

    assert result.status == "warning"
    assert result.is_synthetic is True
    assert result.observed_pattern == {
        "feature": "temperature",
        "group": None,
        "n_windows": 5,
    }
    assert len(result.evidence) == 3
    assert "synthetic" in result.disclaimer.lower()


def test_assess_run_returns_unknown_when_quality_gates_fail():
    config = HealthConfig(
        feature_cols=("temperature",),
        sensor_ranges={"temperature": (-10.0, 10.0)},
        min_valid_windows=3,
        min_coverage=0.8,
    )
    baseline = NormalBaseline.fit(
        pd.DataFrame({"temperature": np.arange(-5.0, 5.0)}),
        config,
        min_rows=5,
    )
    windows = pd.DataFrame(
        {
            "run_id": ["run-1"] * 4,
            "timestamp": [0, 1, 2, 3],
            "is_synthetic": [False] * 4,
            "temperature": [0.0, 0.0, 0.0, 12.0],
        }
    )

    result = assess_run(windows, baseline, Thresholds(1.0, 100.0), config)

    assert result.status == "unknown"
    assert "low_coverage" in result.reasons
    assert any("out of range" in warning for warning in result.warnings)


def test_synthetic_generator_windows_fit_health_monitor():
    synthetic_config = SyntheticConfig(n_runs=50, laps_per_run=3, seed=4)
    telemetry, _ = simulate_dataset(synthetic_config)
    windows = make_windows(telemetry, synthetic_config.window_s, synthetic_config.hz)

    health_config, calibration = load_health_config(
        Path(__file__).resolve().parents[1] / "configs" / "health.yaml"
    )
    assert windows["is_synthetic"].all()
    assert health_config.timestamp_col == "time_s"
    assert set(health_config.feature_cols) <= set(windows.columns)

    train = windows[windows["split"] == "train"]
    validation = windows[windows["split"] == "val"]
    test = windows[windows["split"] == "test"]
    model = fit_and_calibrate(
        train,
        validation,
        health_config,
        calibration,
        train_normal_runs=train.loc[train["is_anomaly"] == 0, "run_id"].unique(),
        val_normal_runs=validation.loc[
            validation["is_anomaly"] == 0, "run_id"
        ].unique(),
    )

    test_run_id = test["run_id"].iloc[0]
    test_run = test.loc[test["run_id"] == test_run_id]
    result = assess_run(test_run, model.baseline, model.thresholds, health_config)

    assert result.status != "unknown"
    assert result.is_synthetic is True
