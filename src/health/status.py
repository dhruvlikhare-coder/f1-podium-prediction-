"""Run-level health status aggregation for telemetry windows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    STATUS_CRITICAL,
    STATUS_NORMAL,
    STATUS_UNKNOWN,
    STATUS_WARNING,
    HealthConfig,
    Thresholds,
    build_disclaimer,
)
from .scoring import NormalBaseline, invalid_rows_mask


@dataclass(frozen=True)
class Aggregation:
    status: str
    peak_score: float | None
    n_valid: int
    n_elevated: int
    n_severe: int
    longest_elevated_streak: int
    longest_severe_streak: int


def _longest_streak(flags: np.ndarray, breaks: np.ndarray) -> int:
    current = best = 0
    for flag, is_break in zip(flags, breaks):
        if is_break:
            current = 0
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def window_scores_to_status(
    score: Sequence[float] | np.ndarray | pd.Series,
    thresholds: Thresholds,
    *,
    warning_min_consecutive: int = 2,
    critical_min_consecutive: int = 3,
    breaks: Sequence[bool] | np.ndarray | pd.Series | None = None,
) -> Aggregation:
    """Apply persistence rules to window scores; time gaps break streaks."""
    if warning_min_consecutive < 1 or critical_min_consecutive < warning_min_consecutive:
        raise ValueError(
            "Need 1 <= warning_min_consecutive <= critical_min_consecutive"
        )
    values = np.asarray(score, dtype=float)
    break_flags = (
        np.zeros(len(values), dtype=bool)
        if breaks is None
        else np.asarray(breaks, dtype=bool)
    )
    if len(break_flags) != len(values):
        raise ValueError("breaks must have the same length as score")

    valid = np.isfinite(values)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return Aggregation(STATUS_UNKNOWN, None, 0, 0, 0, 0, 0)

    elevated = valid & (values >= thresholds.warning)
    severe = valid & (values >= thresholds.critical)
    elevated_streak = _longest_streak(elevated, break_flags)
    severe_streak = _longest_streak(severe, break_flags)

    if severe_streak >= critical_min_consecutive:
        status = STATUS_CRITICAL
    elif elevated_streak >= warning_min_consecutive or severe.any():
        status = STATUS_WARNING
    else:
        status = STATUS_NORMAL

    return Aggregation(
        status=status,
        peak_score=float(np.max(values[valid])),
        n_valid=n_valid,
        n_elevated=int(elevated.sum()),
        n_severe=int(severe.sum()),
        longest_elevated_streak=elevated_streak,
        longest_severe_streak=severe_streak,
    )


@dataclass
class RunHealth:
    run_id: object
    status: str
    is_synthetic: bool | None
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    peak_score: float | None = None
    n_windows: int = 0
    n_valid_windows: int = 0
    n_elevated: int = 0
    n_severe: int = 0
    longest_elevated_streak: int = 0
    longest_severe_streak: int = 0
    coverage: float | None = None
    last_timestamp: object = None
    age_s: float | None = None
    stale: bool | None = None
    n_gaps: int = 0
    observed_pattern: dict[str, object] | None = None
    evidence: list[dict[str, object]] = field(default_factory=list)
    thresholds: dict[str, float] | None = None
    disclaimer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _py(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _to_seconds(series: pd.Series) -> pd.Series:
    """Convert numeric or datetime-like timestamps to epoch seconds."""
    if not pd.api.types.is_datetime64_any_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == series.notna().sum():
            return numeric.astype(float)

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    seconds = parsed.astype("int64").astype(float) / 1_000_000_000
    return seconds.where(parsed.notna(), np.nan)


def _scalar_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        seconds = float(value)
    else:
        numeric = pd.to_numeric(value, errors="coerce")
        if isinstance(value, str) and not pd.isna(numeric):
            seconds = float(numeric)
        else:
            timestamp = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(timestamp):
                raise ValueError("as_of must be a valid timestamp or epoch-seconds value")
            seconds = float(timestamp.timestamp())
    if not np.isfinite(seconds):
        raise ValueError("as_of must be finite")
    return seconds


def _unknown(
    run_id: object,
    is_synthetic: bool | None,
    reasons: list[str],
    warnings: list[str],
    thresholds: Thresholds,
    **details: Any,
) -> RunHealth:
    return RunHealth(
        run_id=_py(run_id),
        status=STATUS_UNKNOWN,
        is_synthetic=is_synthetic,
        reasons=reasons,
        warnings=warnings,
        thresholds=thresholds.to_dict(),
        disclaimer=build_disclaimer(is_synthetic),
        **details,
    )


def assess_run(
    windows: pd.DataFrame,
    baseline: NormalBaseline,
    thresholds: Thresholds,
    config: HealthConfig,
    *,
    as_of: Any = None,
) -> RunHealth:
    """Assess one run; data-quality failures produce an ``unknown`` result."""
    if tuple(baseline.features) != tuple(config.feature_cols):
        raise ValueError("baseline features do not match config.feature_cols")

    required = [
        config.run_col,
        config.timestamp_col,
        config.synthetic_col,
        *config.feature_cols,
    ]
    missing = [column for column in required if column not in windows.columns]
    if missing:
        return _unknown(
            None,
            None,
            ["missing_columns"],
            [f"missing columns: {missing}"],
            thresholds,
            n_windows=len(windows),
        )
    if windows.empty:
        return _unknown(None, None, ["no_windows"], [], thresholds)

    run_ids = windows[config.run_col]
    if run_ids.isna().any():
        return _unknown(
            None,
            None,
            ["missing_run_id"],
            ["some rows have no run id"],
            thresholds,
            n_windows=len(windows),
        )
    unique_run_ids = run_ids.unique()
    if len(unique_run_ids) != 1:
        raise ValueError(
            f"assess_run expects one run, got {len(unique_run_ids)} run ids; use assess_runs"
        )
    run_id = unique_run_ids[0]

    synthetic = windows[config.synthetic_col]
    if synthetic.isna().any() or not synthetic.isin([True, False, 0, 1]).all():
        return _unknown(
            run_id,
            None,
            ["synthetic_flag_missing_or_invalid"],
            [],
            thresholds,
            n_windows=len(windows),
        )
    synthetic_flags = synthetic.astype(bool)
    is_synthetic = bool(synthetic_flags.any())
    warnings: list[str] = []
    if synthetic_flags.nunique() > 1:
        warnings.append("mixed is_synthetic flag within one run; treated as synthetic")

    frame = windows.reset_index(drop=True).copy()
    frame["_timestamp_seconds"] = _to_seconds(frame[config.timestamp_col])
    bad_timestamps = ~np.isfinite(frame["_timestamp_seconds"].to_numpy(dtype=float))
    n_bad_timestamps = int(bad_timestamps.sum())
    if n_bad_timestamps:
        warnings.append(f"{n_bad_timestamps} rows with missing/invalid timestamp ignored")
    frame = frame.loc[~bad_timestamps]
    if frame.empty:
        return _unknown(
            run_id,
            is_synthetic,
            ["no_valid_timestamps"],
            warnings,
            thresholds,
            n_windows=len(windows),
        )
    if not frame["_timestamp_seconds"].is_monotonic_increasing:
        warnings.append("out-of-order timestamps; sorted by time")
        frame = frame.sort_values("_timestamp_seconds", kind="stable")
    duplicate_timestamps = frame["_timestamp_seconds"].duplicated()
    if duplicate_timestamps.any():
        warnings.append(f"{int(duplicate_timestamps.sum())} duplicate timestamps dropped")
        frame = frame.loc[~duplicate_timestamps]
    frame = frame.reset_index(drop=True)
    timestamps = frame["_timestamp_seconds"].to_numpy(dtype=float)

    valid_rows, invalid_counts = invalid_rows_mask(frame, config)
    for reason, count in sorted(invalid_counts.items()):
        warnings.append(
            f"{count} windows with {reason.replace(':', ' ').replace('_', ' ')}"
        )
    scored = baseline.score(frame)
    scores = np.where(
        valid_rows.to_numpy(),
        scored["anomaly_score"].to_numpy(dtype=float),
        np.nan,
    )

    n_valid = int(np.isfinite(scores).sum())
    n_expected = len(frame) + n_bad_timestamps
    if config.window_step_s is not None:
        n_expected = max(
            n_expected,
            int(round((timestamps[-1] - timestamps[0]) / config.window_step_s)) + 1,
        )
    coverage = n_valid / n_expected if n_expected else 0.0

    max_gap = config.effective_max_gap_s
    breaks = np.zeros(len(timestamps), dtype=bool)
    if max_gap is not None and len(timestamps) > 1:
        breaks[1:] = np.diff(timestamps) > max_gap
    n_gaps = int(breaks.sum())
    if n_gaps:
        warnings.append(f"{n_gaps} gaps longer than {max_gap:g}s between windows")

    age_s: float | None = None
    stale: bool | None = None
    if as_of is not None:
        reference = (
            float(timestamps[np.isfinite(scores)].max())
            if n_valid
            else float(timestamps[-1])
        )
        age_s = _scalar_to_seconds(as_of) - reference
        stale = config.max_age_s is not None and age_s > config.max_age_s

    common = {
        "peak_score": float(np.max(scores[np.isfinite(scores)])) if n_valid else None,
        "n_windows": len(windows),
        "n_valid_windows": n_valid,
        "coverage": coverage,
        "last_timestamp": _py(frame[config.timestamp_col].iloc[-1]),
        "age_s": age_s,
        "stale": stale,
        "n_gaps": n_gaps,
    }

    gate: list[str] = []
    if n_valid < config.min_valid_windows:
        gate.append("insufficient_windows")
    if coverage < config.min_coverage:
        gate.append("low_coverage")
    if stale:
        gate.append("stale_data")
    if gate:
        return _unknown(run_id, is_synthetic, gate, warnings, thresholds, **common)

    aggregate = window_scores_to_status(
        scores,
        thresholds,
        warning_min_consecutive=config.warning_min_consecutive,
        critical_min_consecutive=config.critical_min_consecutive,
        breaks=breaks,
    )
    common.update(
        n_elevated=aggregate.n_elevated,
        n_severe=aggregate.n_severe,
        longest_elevated_streak=aggregate.longest_elevated_streak,
        longest_severe_streak=aggregate.longest_severe_streak,
    )

    if aggregate.status == STATUS_CRITICAL:
        reasons = [
            f"{aggregate.longest_severe_streak} consecutive windows at or above the "
            f"critical threshold (needs {config.critical_min_consecutive})"
        ]
    elif aggregate.status == STATUS_WARNING and aggregate.n_severe:
        reasons = [
            "isolated window(s) at or above the critical threshold did not persist; "
            "not enough for critical"
        ]
    elif aggregate.status == STATUS_WARNING:
        reasons = [
            f"{aggregate.longest_elevated_streak} consecutive windows at or above the "
            f"warning threshold (needs {config.warning_min_consecutive})"
        ]
    else:
        reasons = ["no persistent windows above the warning threshold"]

    evidence: list[dict[str, object]] = []
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    for index in order[: config.max_evidence]:
        if not np.isfinite(scores[index]):
            break
        level = (
            "severe"
            if scores[index] >= thresholds.critical
            else "elevated"
            if scores[index] >= thresholds.warning
            else "below_warning"
        )
        evidence.append(
            {
                "timestamp": _py(frame[config.timestamp_col].iloc[index]),
                "score": float(scores[index]),
                "top_feature": scored["top_feature"].iloc[index],
                "level": level,
            }
        )

    observed_pattern = None
    if aggregate.status in (STATUS_WARNING, STATUS_CRITICAL):
        hot_features = [
            scored["top_feature"].iloc[index]
            for index in range(len(frame))
            if np.isfinite(scores[index]) and scores[index] >= thresholds.warning
        ]
        if hot_features:
            feature, count = Counter(hot_features).most_common(1)[0]
            observed_pattern = {
                "feature": feature,
                "group": config.feature_groups.get(feature),
                "n_windows": count,
            }

    return RunHealth(
        run_id=_py(run_id),
        status=aggregate.status,
        is_synthetic=is_synthetic,
        reasons=reasons,
        warnings=warnings,
        observed_pattern=observed_pattern,
        evidence=evidence,
        thresholds=thresholds.to_dict(),
        disclaimer=build_disclaimer(is_synthetic),
        **common,
    )


def assess_runs(
    windows: pd.DataFrame,
    baseline: NormalBaseline,
    thresholds: Thresholds,
    config: HealthConfig,
    *,
    as_of: Any = None,
) -> list[RunHealth]:
    """Assess each run independently; windows from separate runs are never pooled."""
    if config.run_col not in windows.columns or windows.empty:
        return [assess_run(windows, baseline, thresholds, config, as_of=as_of)]
    return [
        assess_run(group, baseline, thresholds, config, as_of=as_of)
        for _, group in windows.groupby(config.run_col, dropna=False, sort=False)
    ]


def results_frame(results: Sequence[RunHealth]) -> pd.DataFrame:
    """Return one reporting row per run."""
    return pd.DataFrame([result.to_dict() for result in results])


__all__ = [
    "Aggregation",
    "RunHealth",
    "assess_run",
    "assess_runs",
    "results_frame",
    "window_scores_to_status",
]
