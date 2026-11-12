"""Interactive / batch car engine-health check CLI helper.

This module is called by :mod:`src.prediction.predict` when the user passes
``--health``, ``--car <id>``, or ``--health-file <path>``.

It is **entirely separate** from the race-podium prediction pipeline — health
results do not feed into race outcome models (SRS FR-83).

Typical usage (from predict.py)::

    from src.health.check_health import run_health_check
    run_health_check(args, ROOT)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Minimum number of windows required by the health config (min_valid_windows=5)
_MIN_WINDOWS = 5
_WINDOW_STEP_S = 10.0

# Sensor input ranges shown to the user during interactive prompting
_SENSOR_RANGES: dict[str, tuple[float, float, str]] = {
    "ambient_temp_c": (0.0, 60.0, "°C"),
    "coolant_temp_c": (0.0, 260.0, "°C"),
    "oil_pressure_mean": (0.0, 10.0, "bar"),
    "vibration_mean": (0.0, 10.0, "g"),
}

# Status badge mapping
_STATUS_BADGE: dict[str, str] = {
    "normal": "🟢 NORMAL",
    "warning": "🟡 WARNING",
    "critical": "🔴 CRITICAL",
    "unknown": "⚪ UNKNOWN",
}


def _safe_print(text: str = "") -> None:
    """Print with UTF-8 safe fallback for Windows consoles."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode())


def _prompt_float(
    label: str,
    low: float,
    high: float,
    unit: str,
    *,
    default: float | None = None,
) -> float:
    """Prompt the user for a float value within [low, high]."""
    hint = f"[{low:.1f} – {high:.1f} {unit}]"
    if default is not None:
        hint += f", default={default:.1f}"
    while True:
        raw = input(f"  {label} {hint}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            print(f"    ! Enter a number between {low} and {high}.")
            continue
        if not (low <= value <= high):
            print(f"    ! Value must be between {low} and {high}.")
            continue
        return value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prompt_engine_health(car_name: str) -> pd.DataFrame:
    """Interactively prompt the user for sensor readings and build a windows DataFrame.

    The user is asked whether they want to enter 5 sequential readings (one per
    10-second window) or a single reading that is replicated across all 5
    windows.  Either way the resulting DataFrame satisfies the minimum
    *min_valid_windows* = 5 constraint required by :func:`~src.health.status.assess_run`.

    Parameters
    ----------
    car_name:
        Identifier label for the car (used as ``run_id`` in the DataFrame).

    Returns
    -------
    pd.DataFrame
        Columns: ``run_id``, ``time_s``, ``is_synthetic``,
        ``coolant_minus_ambient``, ``oil_pressure_mean``, ``vibration_mean``.
    """
    _safe_print()
    _safe_print(f"  Enter sensor readings for: {car_name}")
    _safe_print("  ─" * 25)

    _safe_print()
    _safe_print("  How many readings would you like to enter?")
    _safe_print("    [1] Single reading (replicated across 5 windows)")
    _safe_print("    [5] Five sequential readings (one per 10-second window)")
    while True:
        choice = input("  Choice [1/5]: ").strip()
        if choice in ("1", "5"):
            n_readings = int(choice)
            break
        print("    ! Enter 1 or 5.")

    readings: list[dict[str, float]] = []
    for i in range(n_readings):
        if n_readings > 1:
            _safe_print()
            _safe_print(f"  --- Reading {i + 1} of {n_readings} ---")
        else:
            _safe_print()

        ambient = _prompt_float("Ambient temperature", 0.0, 60.0, "°C", default=25.0)
        coolant = _prompt_float("Coolant temperature", ambient, 260.0, "°C")
        oil_p = _prompt_float("Oil pressure (mean)", 0.0, 10.0, "bar")
        vib = _prompt_float("Vibration (mean)", 0.0, 10.0, "g")
        readings.append(
            {
                "ambient": ambient,
                "coolant_minus_ambient": max(0.0, coolant - ambient),
                "oil_pressure_mean": oil_p,
                "vibration_mean": vib,
            }
        )

    # Replicate single reading into 5 windows
    if n_readings == 1:
        readings = readings * _MIN_WINDOWS

    rows = []
    for idx, r in enumerate(readings):
        rows.append(
            {
                "run_id": car_name,
                "time_s": float(idx) * _WINDOW_STEP_S,
                "is_synthetic": True,
                "coolant_minus_ambient": r["coolant_minus_ambient"],
                "oil_pressure_mean": r["oil_pressure_mean"],
                "vibration_mean": r["vibration_mean"],
            }
        )

    df = pd.DataFrame(rows)
    return df


def check_car_by_id(car_id: int | str, root: Path) -> tuple[pd.DataFrame, str]:
    """Load windows for an existing car/run from SynthetIc telemetry.

    Looks up ``run_id`` first; falls back to ``unit_id`` if not found.

    Parameters
    ----------
    car_id:
        Run ID (0-239) or unit ID (0-7).
    root:
        Project root directory.

    Returns
    -------
    windows:
        DataFrame slice containing the matching run's windows.
    car_name:
        A human-readable label for the car (e.g. ``"run_42 (unit 3)"``).

    Raises
    ------
    ValueError
        If no matching run or unit is found.
    """
    windows_path = root / "SynthetIc" / "windows.parquet"
    df = pd.read_parquet(windows_path)

    try:
        id_int = int(car_id)
    except (ValueError, TypeError):
        raise ValueError(f"car_id must be an integer, got: {car_id!r}")

    def _ensure_is_synthetic(frame: pd.DataFrame) -> pd.DataFrame:
        """Add is_synthetic=True if the column is absent (SynthetIc parquet lacks it)."""
        frame = frame.copy()
        if "is_synthetic" not in frame.columns:
            frame["is_synthetic"] = True
        return frame

    # Try run_id first
    subset = df[df["run_id"] == id_int]
    if not subset.empty:
        # Pick the first (and usually only) run with this run_id
        run_id_val = subset["run_id"].iloc[0]
        unit_id_val = subset["unit_id"].iloc[0] if "unit_id" in subset.columns else "?"
        car_name = f"run_{run_id_val} (unit {unit_id_val})"
        return _ensure_is_synthetic(subset), car_name

    # Fall back to unit_id — pick the first run for that unit
    if "unit_id" in df.columns:
        unit_subset = df[df["unit_id"] == id_int]
        if not unit_subset.empty:
            first_run = unit_subset["run_id"].iloc[0]
            subset = unit_subset[unit_subset["run_id"] == first_run]
            car_name = f"run_{first_run} (unit {id_int})"
            return _ensure_is_synthetic(subset), car_name

    max_run = int(df["run_id"].max()) if "run_id" in df.columns else "?"
    max_unit = int(df["unit_id"].max()) if "unit_id" in df.columns else "?"
    raise ValueError(
        f"No data found for car_id={id_int}. "
        f"Valid run_id range: 0-{max_run}, unit_id range: 0-{max_unit}."
    )


def load_health_from_csv(path: str) -> pd.DataFrame:
    """Load a health-check windows CSV file and validate required columns.

    Parameters
    ----------
    path:
        Path to a CSV file.  Required columns:
        ``run_id``, ``time_s``, ``coolant_minus_ambient``,
        ``oil_pressure_mean``, ``vibration_mean``.

    Returns
    -------
    pd.DataFrame
        The loaded (and lightly normalised) DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"run_id", "time_s", "coolant_minus_ambient", "oil_pressure_mean", "vibration_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Found: {list(df.columns)}"
        )

    if "is_synthetic" not in df.columns:
        df = df.copy()
        df["is_synthetic"] = True

    return df


def display_health_result(result: Any, car_name: str) -> None:  # noqa: ANN401
    """Pretty-print a :class:`~src.health.status.RunHealth` result.

    Parameters
    ----------
    result:
        A :class:`~src.health.status.RunHealth` instance.
    car_name:
        Human-readable car identifier shown in the header.
    """
    badge = _STATUS_BADGE.get(str(result.status), f"? {result.status}")
    _safe_print()
    _safe_print("=" * 56)
    _safe_print(f"  CAR / ENGINE HEALTH REPORT")
    _safe_print(f"  Car : {car_name}")
    _safe_print("=" * 56)
    _safe_print(f"  Status        : {badge}")
    _safe_print(f"  Peak score    : {result.peak_score:.3f}")
    _safe_print(f"  Windows       : {result.n_valid_windows} valid / {result.n_windows} total")
    _safe_print(f"  Elevated      : {result.n_elevated}  |  Severe: {result.n_severe}")
    _safe_print(
        f"  Longest streak: elevated={result.longest_elevated_streak}  "
        f"severe={result.longest_severe_streak}"
    )

    if result.observed_pattern:
        _safe_print()
        _safe_print("  Dominant pattern:")
        for key, val in result.observed_pattern.items():
            _safe_print(f"    {key}: {val}")

    if result.reasons:
        _safe_print()
        _safe_print("  Reasons:")
        for r in result.reasons:
            _safe_print(f"    • {r}")

    if result.warnings:
        _safe_print()
        _safe_print("  Warnings:")
        for w in result.warnings:
            _safe_print(f"    ! {w}")

    if result.evidence:
        _safe_print()
        _safe_print("  Top anomalous windows:")
        for ev in result.evidence[:5]:
            t = ev.get("time_s", ev.get("win", "?"))
            sc = ev.get("score", "?")
            _safe_print(f"    t={t}s  score={sc:.3f}" if isinstance(sc, float) else f"    t={t}s  score={sc}")

    _safe_print()
    _safe_print("  DISCLAIMER")
    _safe_print("  " + "-" * 52)
    if result.disclaimer:
        # Wrap disclaimer at ~52 chars
        words = result.disclaimer.split()
        line = "  "
        for word in words:
            if len(line) + len(word) + 1 > 54:
                _safe_print(line)
                line = "  " + word
            else:
                line += " " + word
        if line.strip():
            _safe_print(line)
    else:
        _safe_print(
            "  Results are based on simulated/synthetic data and "
            "anomaly-detection heuristics only. This is NOT a real "
            "mechanical diagnosis."
        )
    _safe_print("=" * 56)


def run_health_check(args: argparse.Namespace, root: Path) -> None:  # noqa: C901
    """Entry point called from :func:`src.prediction.predict.main`.

    Dispatches to the appropriate data source (interactive prompt, stored
    telemetry, or CSV file), runs the health model, displays results, and
    offers to check additional cars.

    Parameters
    ----------
    args:
        Parsed CLI arguments (expects ``args.health``, ``args.car``,
        ``args.health_file``).
    root:
        Project root directory.
    """
    # Reconfigure stdout for UTF-8 on Windows (emoji safety)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    _safe_print()
    _safe_print("  Loading engine-health model (calibrating from telemetry)...")

    from src.health.loader import get_health_model
    from src.health.status import assess_run

    model, health_config = get_health_model(root)
    _safe_print("  Model ready.")

    while True:
        # ----------------------------------------------------------------
        # Determine data source
        # ----------------------------------------------------------------
        if getattr(args, "health_file", None):
            # CSV file mode
            _safe_print(f"\n  Loading windows from: {args.health_file}")
            try:
                windows = load_health_from_csv(args.health_file)
            except (FileNotFoundError, ValueError) as exc:
                _safe_print(f"  ERROR: {exc}")
                return
            car_name = Path(args.health_file).stem
            args.health_file = None  # only use once; fall to interactive after

        elif getattr(args, "car", None) is not None:
            # Stored telemetry mode
            car_id = args.car
            _safe_print(f"\n  Looking up car/unit: {car_id}")
            try:
                windows, car_name = check_car_by_id(car_id, root)
            except ValueError as exc:
                _safe_print(f"  ERROR: {exc}")
                return
            args.car = None  # only use once

        else:
            # Interactive mode
            _safe_print()
            _safe_print("  ┌─────────────────────────────────────────────┐")
            _safe_print("  │        F1 ENGINE HEALTH CHECK               │")
            _safe_print("  └─────────────────────────────────────────────┘")
            car_name = input("\n  Enter a name or ID for this car (e.g. HAM, CAR-44): ").strip()
            if not car_name:
                car_name = "unnamed_car"
            windows = prompt_engine_health(car_name)

        # ----------------------------------------------------------------
        # Assess health
        # ----------------------------------------------------------------
        _safe_print(f"\n  Assessing health for: {car_name}  ({len(windows)} windows)...")

        try:
            result = assess_run(
                windows,
                model.baseline,
                model.thresholds,
                health_config,
                # Omit as_of — skips the staleness gate (not meaningful for user input)
            )
        except Exception as exc:  # noqa: BLE001
            _safe_print(f"  ERROR during health assessment: {exc}")
            return

        display_health_result(result, car_name)

        # ----------------------------------------------------------------
        # Loop prompt
        # ----------------------------------------------------------------
        _safe_print()
        again = input("  Check another car? (y/n) [n]: ").strip().lower()
        if again not in ("y", "yes"):
            break
        # Reset so next iteration goes to interactive mode
        args.car = None
        args.health_file = None

    _safe_print("\n  Health check complete. Goodbye!\n")
