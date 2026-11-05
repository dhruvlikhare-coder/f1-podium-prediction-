from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ensemble import evaluate, walk_forward

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "backtest.yaml"
REQUIRED_IDENTIFIERS = {"season", "round", "driver", "is_podium", "is_winner", "grid_clean"}


def _resolve_path(value: str) -> Path:
    return (ROOT / value).resolve()


def _calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, n_bins: int
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if in_bin.any():
            error += in_bin.mean() * abs(
                probabilities[in_bin].mean() - labels[in_bin].mean()
            )
    return float(error)


def _make_base_models(config: dict[str, Any], seed: int) -> list[Any]:
    return [
        LogisticRegression(
            **config["logistic_regression"],
            random_state=seed,
            solver="liblinear",
        ),
        RandomForestClassifier(
            **config["random_forest"],
            random_state=seed,
        ),
        XGBClassifier(
            **config["xgboost"],
            random_state=seed,
            eval_metric="logloss",
        ),
    ]


def _validate_config(config: dict[str, Any], features: pd.DataFrame) -> None:
    split = config["split"]
    model = config["model"]
    train_years = set(range(split["train_start"], split["train_end"] + 1))
    expected_years = train_years | {split["validation_season"]} | set(split["test_seasons"])
    present_years = set(features["season"].astype(int).unique())
    missing_years = expected_years - present_years
    if missing_years:
        raise ValueError(f"Feature table is missing configured seasons: {sorted(missing_years)}")
    if split["validation_season"] != split["train_end"] + 1:
        raise ValueError("Validation must immediately follow the configured training period.")
    if min(split["test_seasons"]) != split["validation_season"] + 1:
        raise ValueError("Test seasons must immediately follow the validation season.")
    if not split["test_seasons"] or split["train_start"] > split["train_end"]:
        raise ValueError("Configured train and test season ranges must be non-empty.")
    if not REQUIRED_IDENTIFIERS.issubset(features.columns):
        raise ValueError("Feature table is missing required identifiers and labels.")

    required = REQUIRED_IDENTIFIERS | set(model["feature_columns"]) | set(
        model["context_columns"]
    )
    missing_columns = required - set(features.columns)
    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {sorted(missing_columns)}")
    if features[list(required)].isna().any().any():
        raise ValueError("Feature table has missing values in required columns.")


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    data_config = config["data"]
    split = config["split"]
    model_config = config["model"]

    feature_path = _resolve_path(data_config["features_path"])
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature table not found: {feature_path}. Run build_dataset.py first."
        )
    features = pd.read_parquet(feature_path)
    _validate_config(config, features)

    test_seasons = set(split["test_seasons"])
    validation_season = split["validation_season"]
    test_rows = features.loc[features["season"].isin(test_seasons)].copy()
    validation_rows = features.loc[features["season"] == validation_season].copy()
    if test_rows.empty or validation_rows.empty:
        raise ValueError("No rows found for the configured test seasons.")

    metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    config_json = json.dumps(config, sort_keys=True)

    def record_scores(frame: pd.DataFrame, split_name: str, method: str, seed: int | str) -> None:
        frame = frame.copy()
        frame["split"] = split_name
        frame["method"] = method
        frame["seed"] = seed
        prediction_frames.append(frame)
        scopes = [("all", frame)]
        if split_name == "test":
            scopes.extend(
                (str(int(season)), season_frame)
                for season, season_frame in frame.groupby("season", sort=True)
            )
        for season_scope, scored_frame in scopes:
            metrics.append(
                {
                    "split": split_name,
                    "season_scope": season_scope,
                    "method": method,
                    "seed": seed,
                    **evaluate(scored_frame, target=model_config["target"]),
                    "ece": _calibration_error(
                        scored_frame[model_config["target"]].to_numpy(),
                        scored_frame["probability"].to_numpy(),
                        model_config["calibration_bins"],
                    ),
                    "config_json": config_json,
                }
            )

    for split_name, frame in (("validation", validation_rows), ("test", test_rows)):
        baseline = frame.copy()
        baseline["probability"] = baseline["grid_clean"].between(1, 3).astype(float)
        record_scores(baseline, split_name, "top3_grid_baseline", "baseline")

    backtest_data = features.loc[
        features["season"] <= max(test_seasons)
    ].copy()
    for seed in model_config["random_seeds"]:
        base_models = _make_base_models(model_config, seed)
        for method in model_config["methods"]:
            for split_name, start_season, split_data, expected_seasons in (
                (
                    "validation",
                    validation_season,
                    features.loc[features["season"] <= validation_season].copy(),
                    {validation_season},
                ),
                ("test", min(test_seasons), backtest_data, test_seasons),
            ):
                predictions = walk_forward(
                    split_data,
                    feature_cols=model_config["feature_columns"],
                    start_season=start_season,
                    target=model_config["target"],
                    method=method,
                    min_train_seasons=model_config["min_train_seasons"],
                    context_cols=model_config["context_columns"],
                    base_models=base_models,
                    replay_window=model_config["replay_window"],
                    learning_rate=model_config["learning_rate"],
                    update_epochs=model_config["update_epochs"],
                    hidden=model_config["hidden"],
                    mlp_alpha=model_config["mlp_alpha"],
                    mlp_max_iter=model_config["mlp_max_iter"],
                    random_state=seed,
                )
                predictions = predictions.loc[
                    predictions["season"].isin(expected_seasons)
                ].copy()
                if set(predictions["season"].unique()) != expected_seasons:
                    raise ValueError(
                        f"{split_name} backtest for {method}, seed {seed} did not "
                        f"cover all configured seasons."
                    )
                record_scores(predictions, split_name, method, seed)
                scores = evaluate(predictions, target=model_config["target"])
                print(
                    f"Scored {split_name}: {method}, seed {seed}, "
                    f"log_loss={scores['log_loss']:.4f}"
                )

    metrics_path = _resolve_path(data_config["metrics_path"])
    predictions_path = _resolve_path(data_config["predictions_path"])
    experiments_path = _resolve_path(data_config["experiments_path"])
    timestamp = datetime.now(UTC)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    recorded_at = timestamp.isoformat()
    for row in metrics:
        row["run_id"] = run_id
        row["recorded_at"] = recorded_at
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    experiment_frame = pd.DataFrame(metrics)
    if experiments_path.exists():
        experiment_frame = pd.concat(
            [pd.read_csv(experiments_path), experiment_frame],
            ignore_index=True,
        )
    experiment_frame.to_csv(experiments_path, index=False)
    print("\nMetrics by split, method, and seed")
    print(pd.DataFrame(metrics).drop(columns="config_json").to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Appended experiment log: {experiments_path}")


if __name__ == "__main__":
    main()
