"""Shared fixtures for the car-health tests.

The fixture config uses a baseline with center 0 and scale 1, so a window's
anomaly score is simply max(|f1|, |f2|). That keeps expected statuses obvious.
Column names here are test-local; they are NOT the synthetic generator's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.health.config import HealthConfig, Thresholds
from src.health.scoring import NormalBaseline


@pytest.fixture
def config() -> HealthConfig:
    return HealthConfig(
        feature_cols=("f1", "f2"),
        sensor_ranges={"f1": (-100.0, 100.0), "f2": (-100.0, 100.0)},
        feature_groups={"f1": "thermal", "f2": "pressure"},
        window_step_s=1.0,
        max_age_s=10.0,
        min_valid_windows=5,
        min_coverage=0.8,
        warning_min_consecutive=2,
        critical_min_consecutive=3,
        max_evidence=3,
    )


@pytest.fixture
def baseline() -> NormalBaseline:
    return NormalBaseline(features=("f1", "f2"), center=(0.0, 0.0), scale=(1.0, 1.0))


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds(warning=3.0, critical=5.0)


@pytest.fixture
def make_run():
    """Build one run's windows. Score of each window = max(|f1|, |f2|)."""
    def _make(f1, *, run_id="run_a", f2=None, is_synthetic=True, start=0.0, step=1.0,
              timestamps=None) -> pd.DataFrame:
        f1 = list(f1)
        n = len(f1)
        ts = list(timestamps) if timestamps is not None else [start + step * i for i in range(n)]
        return pd.DataFrame({
            "run_id": run_id,
            "timestamp": ts,
            "is_synthetic": is_synthetic,
            "f1": f1,
            "f2": list(f2) if f2 is not None else [0.0] * n,
        })
    return _make


@pytest.fixture
def toy_runs():
    """Tiny in-test stand-in for generated telemetry (NOT the real generator)."""
    def _make(rng: np.random.Generator, run_ids, *, n=60, drift_from=None, drift=8.0):
        frames = []
        for rid in run_ids:
            f1 = rng.normal(0.0, 1.0, n)
            f2 = rng.normal(0.0, 1.0, n)
            if drift_from is not None:
                f1[drift_from:] += drift
            frames.append(pd.DataFrame({
                "run_id": rid, "timestamp": np.arange(n, dtype=float),
                "is_synthetic": True, "f1": f1, "f2": f2,
            }))
        return pd.concat(frames, ignore_index=True)
    return _make
