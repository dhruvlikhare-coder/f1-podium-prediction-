"""Robust anomaly scores relative to normal training telemetry windows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HealthConfig

_MAD_SCALE = 1.4826
_MIN_SCALE = 1e-9


def _numeric_matrix(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Return feature columns as floats, coercing non-numeric cells to NaN."""
    return df[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def invalid_rows_mask(
    df: pd.DataFrame, config: HealthConfig
) -> tuple[pd.Series, dict[str, int]]:
    """Identify rows with non-finite or out-of-range feature values."""
    values = _numeric_matrix(df, config.feature_cols)
    finite = np.isfinite(values)
    valid = finite.all(axis=1)
    counts: dict[str, int] = {}

    for index, name in enumerate(config.feature_cols):
        non_finite = ~finite[:, index]
        if non_finite.any():
            counts[f"non_finite:{name}"] = int(non_finite.sum())

        if name in config.sensor_ranges:
            low, high = config.sensor_ranges[name]
            out_of_range = finite[:, index] & (
                (values[:, index] < low) | (values[:, index] > high)
            )
            if out_of_range.any():
                counts[f"out_of_range:{name}"] = int(out_of_range.sum())
            valid &= ~out_of_range

    return pd.Series(valid.tolist(), index=df.index), counts


invalid_row_mask = invalid_rows_mask


@dataclass(frozen=True)
class NormalBaseline:
    """Per-feature median and robust scale learned from normal training windows."""

    features: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.features
            or len(self.features) != len(self.center)
            or len(self.features) != len(self.scale)
        ):
            raise ValueError("features, center, and scale must have the same non-zero length")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must not contain duplicates")
        if any(not np.isfinite(value) for value in (*self.center, *self.scale)):
            raise ValueError("baseline center and scale values must be finite")
        if any(value <= _MIN_SCALE for value in self.scale):
            raise ValueError("baseline scale values must be positive")

    @property
    def feature(self) -> tuple[str, ...]:
        """Compatibility alias for the original singular field name."""
        return self.features

    @property
    def centere(self) -> tuple[float, ...]:
        """Compatibility alias for the original misspelled attribute."""
        return self.center

    @classmethod
    def fit(
        cls,
        normal_windows: pd.DataFrame,
        config: HealthConfig,
        *,
        min_rows: int = 30,
    ) -> NormalBaseline:
        missing = [name for name in config.feature_cols if name not in normal_windows.columns]
        if missing:
            raise ValueError(f"Cannot fit baseline; missing columns: {missing}")
        if min_rows < 1:
            raise ValueError("min_rows must be at least 1")

        valid, _ = invalid_rows_mask(normal_windows, config)
        values = _numeric_matrix(normal_windows.loc[valid], config.feature_cols)
        if len(values) < min_rows:
            raise ValueError(
                f"Need at least {min_rows} valid normal windows to fit baseline; got {len(values)}"
            )

        center = np.median(values, axis=0)
        scale = _MAD_SCALE * np.median(np.abs(values - center), axis=0)
        fallback = np.std(values, axis=0, ddof=1) if len(values) > 1 else np.zeros(values.shape[1])
        scale = np.where(scale > _MIN_SCALE, scale, fallback)
        degenerate = [
            name
            for name, value in zip(config.feature_cols, scale)
            if not np.isfinite(value) or value <= _MIN_SCALE
        ]
        if degenerate:
            raise ValueError(
                f"Baseline scale is constant or degenerate for features: {degenerate}; "
                "normal training windows need variation"
            )

        return cls(
            features=config.feature_cols,
            center=tuple(float(value) for value in center),
            scale=tuple(float(value) for value in scale),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> NormalBaseline:
        features = data.get("features", data.get("feature"))
        center = data.get("center", data.get("centere"))
        scale = data.get("scale")
        if not isinstance(features, (list, tuple)):
            raise ValueError("Baseline features must be a sequence")
        if not isinstance(center, (list, tuple)):
            raise ValueError("Baseline center must be a sequence")
        if not isinstance(scale, (list, tuple)):
            raise ValueError("Baseline scale must be a sequence")
        return cls(
            features=tuple(str(value) for value in features),
            center=tuple(float(value) for value in center),
            scale=tuple(float(value) for value in scale),
        )

    def to_dict(self) -> dict[str, object]:
        return {"features": self.features, "center": self.center, "scale": self.scale}

    def score(self, windows: pd.DataFrame) -> pd.DataFrame:
        """Score each window and report the feature with the largest robust z-score."""
        missing = [name for name in self.features if name not in windows.columns]
        if missing:
            raise ValueError(f"Cannot score windows; missing columns: {missing}")

        values = _numeric_matrix(windows, self.features)
        z_scores = np.abs(
            (values - np.asarray(self.center, dtype=float))
            / np.asarray(self.scale, dtype=float)
        )
        row_valid = np.isfinite(z_scores).all(axis=1)
        scores = np.full(len(windows), np.nan, dtype=float)
        top_features = np.full(len(windows), None, dtype=object)
        if row_valid.any():
            valid_z_scores = z_scores[row_valid]
            top_indices = np.argmax(valid_z_scores, axis=1)
            scores[row_valid] = valid_z_scores.max(axis=1)
            top_features[row_valid] = np.asarray(self.features, dtype=object)[top_indices]

        return pd.DataFrame(
            {"anomaly_score": scores, "top_feature": top_features},
            index=windows.index,
        )
