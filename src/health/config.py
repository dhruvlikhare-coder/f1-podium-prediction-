"""Configuration and threshold objects for telemetry health assessment."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATUS_NORMAL = "normal"
STATUS_CRITICAL = "critical"
STATUS_UNKNOWN = "unknown"
STATUS_WARNING = "warning"
STATUS = (STATUS_NORMAL, STATUS_WARNING, STATUS_CRITICAL, STATUS_UNKNOWN)

_BASE_DISCLAIMER = (
    "This is an anomaly indicator computed from telemetry windows, not a confirmed "
    "mechanical diagnosis."
)
_SYNTHETIC_DISCLAIMER = (
    " Input is SYNTHETIC: sensor values and injected faults are simulated, "
    "not measurements from a real car."
)


def build_disclaimer(is_synthetic: bool | None) -> str:
    """Build the user-facing disclaimer for a health result."""
    if is_synthetic:
        return _BASE_DISCLAIMER + _SYNTHETIC_DISCLAIMER
    return _BASE_DISCLAIMER


@dataclass(frozen=True)
class Thresholds:
    """Frozen anomaly-score thresholds fitted on normal validation runs."""

    warning: float
    critical: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.warning) and math.isfinite(self.critical)):
            raise ValueError("Thresholds must be finite")
        if not 0 <= self.warning < self.critical:
            raise ValueError("Thresholds must satisfy 0 <= warning < critical")

    @property
    def warnings(self) -> float:
        """Compatibility alias for the original misspelled field."""
        return self.warning

    def to_dict(self) -> dict[str, float]:
        return {"warning": float(self.warning), "critical": float(self.critical)}


Threeshold = Thresholds


@dataclass(frozen=True)
class CalibrationSpec:
    """Choices for deriving score thresholds from normal validation windows."""

    warning_quantile: float = 0.99
    critical_quantile: float = 0.9999
    min_critical_margin: float = 0.5
    min_windows: int = 100

    def __post_init__(self) -> None:
        if not 0 < self.warning_quantile < self.critical_quantile < 1:
            raise ValueError(
                "Calibration quantiles must satisfy "
                "0 < warning_quantile < critical_quantile < 1"
            )
        if self.min_critical_margin <= 0 or not math.isfinite(self.min_critical_margin):
            raise ValueError("min_critical_margin must be finite and positive")
        if self.min_windows < 1:
            raise ValueError("min_windows must be at least 1")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "warning_quantile": self.warning_quantile,
            "critical_quantile": self.critical_quantile,
            "min_critical_margin": self.min_critical_margin,
            "min_windows": self.min_windows,
        }


Calibrations = CalibrationSpec


@dataclass(frozen=True)
class HealthConfig:
    """Column mapping, data-quality limits, and run-level aggregation rules."""

    feature_cols: tuple[str, ...]
    run_col: str = "run_id"
    timestamp_col: str = "timestamp"
    synthetic_col: str = "is_synthetic"
    sensor_ranges: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    feature_groups: Mapping[str, str] = field(default_factory=dict)
    window_step_s: float | None = None
    max_gap_s: float | None = None
    max_age_s: float | None = None
    min_valid_windows: int = 5
    min_coverage: float = 0.8
    warning_min_consecutive: int = 2
    critical_min_consecutive: int = 3
    max_evidence: int = 3

    def __post_init__(self) -> None:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty")
        if len(set(self.feature_cols)) != len(self.feature_cols):
            raise ValueError("feature_cols must not contain duplicates")

        unknown_ranges = set(self.sensor_ranges) - set(self.feature_cols)
        if unknown_ranges:
            raise ValueError(f"sensor_ranges refer to unknown features: {sorted(unknown_ranges)}")
        for name, (low, high) in self.sensor_ranges.items():
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(
                    f"sensor range for {name!r} must be finite and satisfy low < high"
                )

        for name, value in (
            ("window_step_s", self.window_step_s),
            ("max_gap_s", self.max_gap_s),
            ("max_age_s", self.max_age_s),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when set")
        if self.min_valid_windows < 1:
            raise ValueError("min_valid_windows must be at least 1")
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        if self.warning_min_consecutive < 1:
            raise ValueError("warning_min_consecutive must be at least 1")
        if self.critical_min_consecutive < self.warning_min_consecutive:
            raise ValueError(
                "critical_min_consecutive must be at least warning_min_consecutive"
            )
        if self.max_evidence < 1:
            raise ValueError("max_evidence must be at least 1")

    @property
    def windows_step_s(self) -> float | None:
        """Compatibility alias for the original plural field name."""
        return self.window_step_s

    @property
    def feature_column(self) -> tuple[str, ...]:
        """Compatibility alias for the original singular field name."""
        return self.feature_cols

    @property
    def effective_max_gap_s(self) -> float | None:
        """Return the configured gap limit or a window-sized default."""
        if self.max_gap_s is not None:
            return self.max_gap_s
        if self.window_step_s is not None:
            return 1.5 * self.window_step_s
        return None


Healthconfig = HealthConfig


def load_health_config(path: str | Path) -> tuple[HealthConfig, CalibrationSpec]:
    """Load health procedure choices and calibration settings from YAML."""
    raw: Any = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Health configuration must contain a YAML mapping")

    columns = raw.get("columns", {})
    quality = raw.get("data_quality", raw.get("quality", {}))
    rules = raw.get("run_status", raw.get("rules", {}))
    calibration = raw.get("calibration", {})
    mappings = (columns, quality, rules, calibration)
    if any(not isinstance(section, Mapping) for section in mappings):
        raise ValueError("columns, quality, rules, and calibration must be mappings")

    features = raw.get("features", raw.get("feature", ()))
    if isinstance(features, str):
        features = (features,)
    elif not isinstance(features, (list, tuple)):
        raise ValueError("feature must be a string or a sequence of column names")

    sensor_ranges_raw = raw.get("sensor_ranges", {})
    feature_groups_raw = raw.get("feature_groups", {})
    if not isinstance(sensor_ranges_raw, Mapping):
        raise ValueError("sensor_ranges must be a mapping")
    if not isinstance(feature_groups_raw, Mapping):
        raise ValueError("feature_groups must be a mapping")
    sensor_ranges: dict[str, tuple[float, float]] = {}
    for name, bounds in sensor_ranges_raw.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"sensor range for {name!r} must contain two values")
        sensor_ranges[str(name)] = (float(bounds[0]), float(bounds[1]))

    health_config = HealthConfig(
        feature_cols=tuple(str(name) for name in features),
        run_col=str(columns.get("run", "run_id")),
        timestamp_col=str(columns.get("timestamp", "timestamp")),
        synthetic_col=str(columns.get("synthetic", "is_synthetic")),
        sensor_ranges=sensor_ranges,
        feature_groups={str(name): str(label) for name, label in feature_groups_raw.items()},
        window_step_s=quality.get("window_step_s", quality.get("windows_step_s")),
        max_gap_s=quality.get("max_gap_s"),
        max_age_s=quality.get("max_age_s"),
        min_valid_windows=int(quality.get("min_valid_windows", 5)),
        min_coverage=float(quality.get("min_coverage", 0.8)),
        warning_min_consecutive=int(rules.get("warning_min_consecutive", 2)),
        critical_min_consecutive=int(rules.get("critical_min_consecutive", 3)),
        max_evidence=int(rules.get("max_evidence", 3)),
    )
    calibrations = CalibrationSpec(
        warning_quantile=float(
            calibration.get("warning_quantile", calibration.get("warnings_quantile", 0.99))
        ),
        critical_quantile=float(calibration.get("critical_quantile", 0.9999)),
        min_critical_margin=float(calibration.get("min_critical_margin", 0.5)),
        min_windows=int(calibration.get("min_windows", 100)),
    )
    return health_config, calibrations


load_heath_config = load_health_config
