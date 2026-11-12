"""Tests for the interactive engine health check feature.

These tests verify:
  1. get_health_model() returns a valid FittedHealthModel.
  2. check_car_by_id() returns a correctly structured DataFrame.
  3. A manually constructed windows DataFrame can be assessed by assess_run().
  4. load_health_from_csv() loads and validates a CSV correctly.
  5. display_health_result() runs without error and prints output.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# Project root (two levels up: tests/ -> repo root)
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def health_model_and_config():
    """Fit and cache the health model once per test session (slow ~2 s)."""
    from src.health.loader import get_health_model

    return get_health_model(ROOT)


@pytest.fixture(scope="module")
def fitted_model(health_model_and_config):
    model, _ = health_model_and_config
    return model


@pytest.fixture(scope="module")
def health_config(health_model_and_config):
    _, cfg = health_model_and_config
    return cfg


# ---------------------------------------------------------------------------
# Test 1: get_health_model returns a valid FittedHealthModel
# ---------------------------------------------------------------------------

def test_get_health_model_returns_fitted_model(fitted_model, health_config):
    """Model baseline must cover the 3 expected feature columns."""
    from src.health.thresholds import FittedHealthModel

    assert isinstance(fitted_model, FittedHealthModel)

    # Baseline should have been fitted on 3 features
    baseline = fitted_model.baseline
    # NormalBaseline stores feature names in the .features attribute (a tuple)
    feature_attr = None
    for attr in ("features", "feature_names_", "feature_names"):
        if hasattr(baseline, attr):
            feature_attr = attr
            break
    assert feature_attr is not None, (
        f"NormalBaseline must expose a feature name attribute; got: {dir(baseline)}"
    )
    feature_names = getattr(baseline, feature_attr)
    assert len(feature_names) == 3, (
        f"Expected 3 features, got {feature_names}"
    )

    expected_features = {
        "coolant_minus_ambient",
        "oil_pressure_mean",
        "vibration_mean",
    }
    assert set(feature_names) == expected_features

    # Thresholds: warning < critical
    thr = fitted_model.thresholds
    assert thr.warning < thr.critical, (
        f"warning threshold ({thr.warning}) must be < critical ({thr.critical})"
    )


# ---------------------------------------------------------------------------
# Test 2: check_car_by_id returns a correctly structured DataFrame
# ---------------------------------------------------------------------------

def test_check_car_by_id_returns_windows():
    """Requesting run_id=0 should return ≥5 windows with the required columns."""
    from src.health.check_health import check_car_by_id

    windows, car_name = check_car_by_id(0, ROOT)

    assert isinstance(windows, pd.DataFrame)
    assert len(windows) >= 5, f"Expected ≥5 windows, got {len(windows)}"
    assert isinstance(car_name, str)
    assert len(car_name) > 0

    required = {
        "run_id",
        "time_s",
        "is_synthetic",
        "coolant_minus_ambient",
        "oil_pressure_mean",
        "vibration_mean",
    }
    missing = required - set(windows.columns)
    assert not missing, f"Windows DataFrame missing columns: {missing}"


def test_check_car_by_id_mid_range():
    """run_id=100 (mid-range) should also return valid windows."""
    from src.health.check_health import check_car_by_id

    windows, car_name = check_car_by_id(100, ROOT)
    assert len(windows) >= 5
    assert "coolant_minus_ambient" in windows.columns


def test_check_car_by_id_invalid_raises():
    """Requesting a non-existent car_id should raise ValueError."""
    from src.health.check_health import check_car_by_id

    with pytest.raises(ValueError, match="No data found"):
        check_car_by_id(99999, ROOT)


# ---------------------------------------------------------------------------
# Test 3: Manually constructed windows can be assessed
# ---------------------------------------------------------------------------

def _make_manual_windows(car_name: str = "TEST_CAR", n: int = 5) -> pd.DataFrame:
    """Build a minimal windows DataFrame with normal sensor readings."""
    return pd.DataFrame({
        "run_id": [car_name] * n,
        "time_s": [i * 10.0 for i in range(n)],
        "is_synthetic": [True] * n,
        "coolant_minus_ambient": [80.0] * n,   # typical normal value
        "oil_pressure_mean": [4.5] * n,         # normal
        "vibration_mean": [1.2] * n,            # normal
    })


def test_manual_input_windows_structure(fitted_model, health_config):
    """Manually built windows must yield a non-unknown RunHealth result."""
    from src.health.status import assess_run

    windows = _make_manual_windows()
    result = assess_run(
        windows,
        fitted_model.baseline,
        fitted_model.thresholds,
        health_config,
    )

    assert result is not None
    assert result.status in ("normal", "warning", "critical", "unknown"), (
        f"Unexpected status: {result.status!r}"
    )
    # Normal readings should almost certainly be healthy or at worst warning
    assert result.status != "unknown", (
        f"5 valid windows with normal readings should not produce 'unknown'; "
        f"got status={result.status!r}, n_valid={result.n_valid_windows}"
    )


def test_manual_input_critical_readings(fitted_model, health_config):
    """Extreme sensor readings should trigger warning or critical status."""
    from src.health.status import assess_run

    # Massively elevated coolant and vibration — should score very high
    windows = pd.DataFrame({
        "run_id": ["CRITICAL_CAR"] * 5,
        "time_s": [i * 10.0 for i in range(5)],
        "is_synthetic": [True] * 5,
        "coolant_minus_ambient": [195.0] * 5,  # near max = very hot
        "oil_pressure_mean": [0.5] * 5,         # near min = pressure loss
        "vibration_mean": [9.5] * 5,            # near max = severe vibration
    })

    result = assess_run(
        windows,
        fitted_model.baseline,
        fitted_model.thresholds,
        health_config,
    )
    assert result.status in ("warning", "critical"), (
        f"Extreme readings should not be 'normal'; got {result.status!r}"
    )
    assert result.peak_score > 0, "Peak score must be positive for extreme readings"


# ---------------------------------------------------------------------------
# Test 4: load_health_from_csv validates and loads CSV
# ---------------------------------------------------------------------------

def test_health_csv_loader_valid():
    """A valid CSV with required columns should load correctly."""
    from src.health.check_health import load_health_from_csv

    content = (
        "run_id,time_s,coolant_minus_ambient,oil_pressure_mean,vibration_mean\n"
        "my_car,0.0,80.0,4.5,1.2\n"
        "my_car,10.0,82.0,4.4,1.3\n"
        "my_car,20.0,81.0,4.6,1.1\n"
        "my_car,30.0,80.5,4.5,1.2\n"
        "my_car,40.0,79.0,4.7,1.0\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp = f.name

    try:
        df = load_health_from_csv(tmp)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "is_synthetic" in df.columns
        assert df["is_synthetic"].all(), "is_synthetic should be True when not in CSV"
    finally:
        os.unlink(tmp)


def test_health_csv_loader_missing_columns():
    """CSV missing required columns should raise ValueError."""
    from src.health.check_health import load_health_from_csv

    content = "run_id,time_s\nmy_car,0.0\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        tmp = f.name

    try:
        with pytest.raises(ValueError, match="missing required columns"):
            load_health_from_csv(tmp)
    finally:
        os.unlink(tmp)


def test_health_csv_loader_file_not_found():
    """Non-existent CSV path should raise FileNotFoundError."""
    from src.health.check_health import load_health_from_csv

    with pytest.raises(FileNotFoundError):
        load_health_from_csv("/nonexistent/path/data.csv")


# ---------------------------------------------------------------------------
# Test 5: display_health_result runs without error
# ---------------------------------------------------------------------------

def test_display_health_result_runs_without_error(fitted_model, health_config, capsys):
    """display_health_result() must print to stdout without raising."""
    from src.health.check_health import display_health_result
    from src.health.status import assess_run

    windows = _make_manual_windows(car_name="DISPLAY_TEST")
    result = assess_run(
        windows,
        fitted_model.baseline,
        fitted_model.thresholds,
        health_config,
    )

    # Should not raise
    display_health_result(result, "DISPLAY_TEST")

    captured = capsys.readouterr()
    out = captured.out

    assert "DISPLAY_TEST" in out, "Car name should appear in health report"
    assert any(
        badge in out for badge in ("NORMAL", "WARNING", "CRITICAL", "UNKNOWN")
    ), "Status badge should appear in health report"
    assert "DISCLAIMER" in out, "Disclaimer must always be shown"
