"""Threshold protocol: fit on normal train runs, calibrate on validation, freeze.

Order of operations (nothing here ever sees test runs):

1. ``fit_and_calibrate`` fits the baseline on NORMAL TRAIN runs and sets alert
   thresholds from the window-score quantiles of NORMAL VALIDATION runs.
2. ``freeze_model`` writes baseline + thresholds + provenance to a JSON file
   with a SHA-256 hash. It refuses to overwrite an existing file, so thresholds
   cannot be quietly re-tuned after a look at test results.
3. Evaluation loads the file with ``load_frozen_model``, which re-checks the hash.

Splits are by run, never by window: ``assert_run_disjoint`` fails if any run id
appears in more than one split.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CalibrationSpec, HealthConfig, Thresholds
from .scoring import NormalBaseline, invalid_row_mask

FORMAT_VERSION = 1


def assert_run_disjoint(splits: Mapping[str, Iterable]) -> None:
    """Raise if any run id appears in more than one split (window leakage guard)."""
    seen: dict[str, str] = {}
    clashes: dict[str, list[str]] = {}
    for split, runs in splits.items():
        for run in set(runs):
            key = str(run)
            if key in seen:
                clashes.setdefault(key, [seen[key]]).append(split)
            else:
                seen[key] = split
    if clashes:
        detail = "; ".join(f"{run} in {sorted(set(s))}" for run, s in sorted(clashes.items()))
        raise ValueError(f"Runs appear in more than one split: {detail}")


def calibrate_thresholds(normal_val_scores: Iterable[float], spec: CalibrationSpec) -> Thresholds:
    """Quantile thresholds from window scores of normal validation runs."""
    scores = np.asarray(list(normal_val_scores), dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < spec.min_windows:
        raise ValueError(
            f"Need at least {spec.min_windows} normal validation windows to set "
            f"thresholds (got {len(scores)})."
        )
    warning = float(np.quantile(scores, spec.warning_quantile))
    critical = float(np.quantile(scores, spec.critical_quantile))
    critical = max(critical, warning + spec.min_critical_margin)
    return Thresholds(warning=warning, critical=critical)


@dataclass(frozen=True)
class FittedHealthModel:
    baseline: NormalBaseline
    thresholds: Thresholds
    spec: CalibrationSpec
    provenance: dict


def fit_and_calibrate(train_windows: pd.DataFrame, val_windows: pd.DataFrame,
                      config: HealthConfig, spec: CalibrationSpec,
                      *, train_normal_runs: Iterable, val_normal_runs: Iterable
                      ) -> FittedHealthModel:
    """Fit on normal train runs, set thresholds on normal validation runs.

    ``train_normal_runs`` / ``val_normal_runs`` are run ids the caller knows to be
    fault-free (for example from the generator's labels). Test runs are not
    accepted by design.
    """
    run_col = config.run_col
    for name, df in (("train", train_windows), ("validation", val_windows)):
        if run_col not in df.columns:
            raise ValueError(f"{name} windows are missing the run column '{run_col}'.")
    assert_run_disjoint({"train": train_windows[run_col].dropna().unique(),
                         "validation": val_windows[run_col].dropna().unique()})

    train_ids, val_ids = set(train_normal_runs), set(val_normal_runs)
    train_normal = train_windows[train_windows[run_col].isin(train_ids)]
    val_normal = val_windows[val_windows[run_col].isin(val_ids)]
    if train_normal.empty or val_normal.empty:
        raise ValueError("No windows found for the given normal train/validation runs.")

    baseline = NormalBaseline.fit(train_normal, config)
    valid, _ = invalid_row_mask(val_normal, config)
    val_scores = baseline.score(val_normal.loc[valid])["anomaly_score"]
    thresholds = calibrate_thresholds(val_scores, spec)

    provenance = {
        "train_normal_runs": sorted(map(str, train_ids)),
        "val_normal_runs": sorted(map(str, val_ids)),
        "n_train_windows": int(len(train_normal)),
        "n_val_windows": int(len(val_scores)),
        "feature_cols": list(config.feature_column),
        "note": "Thresholds set on validation runs only; test runs were never used.",
    }
    return FittedHealthModel(baseline, thresholds, spec, provenance)


def _payload(model: FittedHealthModel) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "baseline": model.baseline.to_dict(),
        "thresholds": model.thresholds.to_dict(),
        "calibration": model.spec.to_dict(),
        "provenance": model.provenance,
    }


def _digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_model(model: FittedHealthModel, path: str | Path, *, overwrite: bool = False) -> str:
    """Write the frozen model. Returns its SHA-256. Refuses to overwrite by default."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Thresholds are frozen; re-tuning needs an explicit "
            f"overwrite=True and should happen before looking at test results."
        )
    payload = _payload(model)
    digest = _digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**payload, "sha256": digest}, indent=2, sort_keys=True),
                    encoding="utf-8")
    return digest


def load_frozen_model(path: str | Path) -> FittedHealthModel:
    """Load a frozen model and verify its hash; raises ValueError if it was altered."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = raw.pop("sha256", None)
    if stored is None or _digest(raw) != stored:
        raise ValueError(f"Frozen model {path} failed its integrity check (hash mismatch).")
    return FittedHealthModel(
        baseline=NormalBaseline.from_dict(raw["baseline"]),
        thresholds=Thresholds(**raw["thresholds"]),
        spec=CalibrationSpec(**raw["calibration"]),
        provenance=raw["provenance"],
    )
