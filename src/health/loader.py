"""Load or fit the engine-health model from the SynthetIc telemetry dataset.

This module is intentionally lightweight: it only reads parquet data and
delegates calibration to :mod:`src.health.thresholds`.  The result is a
:class:`~src.health.thresholds.FittedHealthModel` that can be passed directly
to :func:`~src.health.status.assess_run`.

Usage::

    from src.health.loader import get_health_model
    model, cfg = get_health_model()
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.health.config import CalibrationSpec, HealthConfig, load_health_config
from src.health.thresholds import FittedHealthModel, fit_and_calibrate

# Project root (three levels up from this file)
ROOT = Path(__file__).resolve().parent.parent.parent


def get_health_model(
    root: Path | None = None,
) -> tuple[FittedHealthModel, HealthConfig]:
    """Fit baseline + calibrate thresholds from SynthetIc/windows.parquet.

    Parameters
    ----------
    root:
        Project root directory.  Defaults to the repository root inferred
        from this file's location.

    Returns
    -------
    model:
        A :class:`~src.health.thresholds.FittedHealthModel` with a fitted
        :class:`~src.health.scoring.NormalBaseline` and calibrated
        :class:`~src.health.config.Thresholds`.
    health_config:
        The parsed :class:`~src.health.config.HealthConfig` from
        ``configs/health.yaml``.

    Raises
    ------
    FileNotFoundError
        If ``configs/health.yaml`` or ``SynthetIc/windows.parquet`` are
        missing.
    ValueError
        If the parquet file lacks a ``split`` or ``is_anomaly`` column.
    """
    if root is None:
        root = ROOT

    config_path = root / "configs" / "health.yaml"
    windows_path = root / "SynthetIc" / "windows.parquet"

    if not config_path.exists():
        raise FileNotFoundError(f"Health config not found: {config_path}")
    if not windows_path.exists():
        raise FileNotFoundError(f"Windows parquet not found: {windows_path}")

    health_config, calibration_spec = load_health_config(config_path)

    df = pd.read_parquet(windows_path)

    for col in ("split", "is_anomaly", "run_id"):
        if col not in df.columns:
            raise ValueError(
                f"windows.parquet is missing required column: '{col}'"
            )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()

    train_normal = train.loc[train["is_anomaly"] == 0, "run_id"].unique()
    val_normal = val.loc[val["is_anomaly"] == 0, "run_id"].unique()

    model = fit_and_calibrate(
        train,
        val,
        health_config,
        calibration_spec,
        train_normal_runs=train_normal,
        val_normal_runs=val_normal,
    )

    return model, health_config
