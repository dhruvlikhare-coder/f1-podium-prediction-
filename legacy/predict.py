"""
Interactive F1 Podium Predictor — takes user input and returns predictions.

Usage
-----
    python predict.py                        # interactive mode (prompts for input)
    python predict.py --demo                 # run a quick demo with synthetic data
    python predict.py --file race_input.csv  # batch mode from a CSV file

The script trains the AdaptiveStack ensemble on the historical feature table
(data/features/features.parquet) and then lets the user enter race details to
get podium probability predictions for each driver.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

# Fix encoding for Windows consoles that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ensemble import AdaptiveStack, evaluate

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
FEATURE_PATH = ROOT / "data" / "features" / "features.parquet"

# ─── Feature columns (must match backtest.yaml / feature.py) ────────────────
FEATURE_COLS = [
    "grid_clean",
    "pit_lane_start",
    "driver_avg_finish",
    "driver_podium_rate",
    "driver_dnf_rate",
    "driver_experience",
    "team_avg_finish",
    "team_podium_rate",
    "team_mech_dnf_rate",
    "circuit_avg_finish",
    "circuit_races",
    "era",
]

TARGET = "is_podium"
ERA_STARTS = (2017, 2022)

# ─── Defaults for neutral / unknown drivers ─────────────────────────────────
NEUTRAL = {
    "driver_avg_finish": 10.5,
    "driver_podium_rate": 0.15,
    "driver_dnf_rate": 0.10,
    "driver_experience": 0,
    "team_avg_finish": 10.5,
    "team_podium_rate": 0.30,
    "team_mech_dnf_rate": 0.05,
    "circuit_avg_finish": 10.5,
    "circuit_races": 0,
}


# ─── Model training ────────────────────────────────────────────────────────
def _default_base_models(seed: int = 42) -> list[Any]:
    """Build the default base models (LR + RF + XGB)."""
    models: list[Any] = [
        LogisticRegression(
            random_state=seed, solver="liblinear", max_iter=1000, C=0.5
        ),
        RandomForestClassifier(
            random_state=seed, n_estimators=100, max_depth=5, n_jobs=-1
        ),
    ]
    try:
        from xgboost import XGBClassifier

        models.append(
            XGBClassifier(
                random_state=seed,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                eval_metric="logloss",
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
            )
        )
    except ImportError:
        print("[warn] xgboost not installed — using LR + RF only.")
    return models


def train_model(features: pd.DataFrame, seed: int = 42) -> AdaptiveStack:
    """Train the full AdaptiveStack on historical features."""
    print("Training ensemble on historical data …")
    model = AdaptiveStack(
        base_model=_default_base_models(seed),
        replay_window=40,
        learning_rate=1e-4,
        update_epochs=3,
        hidden=(16,),
        mlp_alpha=1e-4,
        mlp_max_iter=800,
        random_state=seed,
    )
    model.fit(
        features,
        feature_cols=FEATURE_COLS,
        target=TARGET,
        min_train_seasons=3,
    )
    print(f"  ✓ Trained on {len(features):,} rows "
          f"(seasons {int(features['season'].min())}–{int(features['season'].max())})\n")
    return model


# ─── Helpers to compute era ─────────────────────────────────────────────────
def _compute_era(season: int) -> int:
    return int(np.searchsorted(ERA_STARTS, season, side="right"))


# ─── User input helpers ─────────────────────────────────────────────────────
def _prompt(msg: str, default: Any = None, cast: type = str) -> Any:
    """Prompt the user; return *default* when they press Enter on an empty line."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {msg}{suffix}: ").strip()
        if not raw and default is not None:
            return cast(default)
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"    ⚠  Please enter a valid {cast.__name__}.")


def _prompt_driver(idx: int) -> dict[str, Any]:
    """Interactively collect feature values for one driver."""
    print(f"\n  ── Driver {idx + 1} ──")
    name = _prompt("Driver name (or id)", cast=str)
    grid = _prompt("Grid position (1–20, 0 = pit-lane start)", default=10, cast=int)
    driver_avg_finish = _prompt(
        "Driver avg finish (recent)", default=NEUTRAL["driver_avg_finish"], cast=float
    )
    driver_podium_rate = _prompt(
        "Driver podium rate (0–1)", default=NEUTRAL["driver_podium_rate"], cast=float
    )
    driver_dnf_rate = _prompt(
        "Driver DNF rate (0–1)", default=NEUTRAL["driver_dnf_rate"], cast=float
    )
    driver_experience = _prompt(
        "Driver experience (# past races)", default=NEUTRAL["driver_experience"], cast=int
    )
    team_avg_finish = _prompt(
        "Team avg finish", default=NEUTRAL["team_avg_finish"], cast=float
    )
    team_podium_rate = _prompt(
        "Team podium rate (0–1)", default=NEUTRAL["team_podium_rate"], cast=float
    )
    team_mech_dnf_rate = _prompt(
        "Team mechanical DNF rate (0–1)",
        default=NEUTRAL["team_mech_dnf_rate"],
        cast=float,
    )
    circuit_avg_finish = _prompt(
        "Circuit avg finish (driver at this track)",
        default=NEUTRAL["circuit_avg_finish"],
        cast=float,
    )
    circuit_races = _prompt(
        "Circuit races (driver's past races here)", default=NEUTRAL["circuit_races"], cast=int
    )

    return {
        "driver": name,
        "grid_clean": grid,
        "pit_lane_start": int(grid == 0),
        "driver_avg_finish": driver_avg_finish,
        "driver_podium_rate": driver_podium_rate,
        "driver_dnf_rate": driver_dnf_rate,
        "driver_experience": driver_experience,
        "team_avg_finish": team_avg_finish,
        "team_podium_rate": team_podium_rate,
        "team_mech_dnf_rate": team_mech_dnf_rate,
        "circuit_avg_finish": circuit_avg_finish,
        "circuit_races": circuit_races,
    }


def collect_race_input() -> pd.DataFrame:
    """Interactively collect race + driver data from the user."""
    print("\n╔══════════════════════════════════════════════════╗")
    print("║       F1 Podium Predictor — Enter Race Info      ║")
    print("╚══════════════════════════════════════════════════╝\n")

    season = _prompt("Season (year)", default=2025, cast=int)
    race_round = _prompt("Round number", default=1, cast=int)
    n_drivers = _prompt("Number of drivers to enter", default=20, cast=int)

    era = _compute_era(season)

    drivers = []
    for i in range(n_drivers):
        d = _prompt_driver(i)
        d["season"] = season
        d["round"] = race_round
        d["era"] = era
        # Dummy target columns required by the model schema
        d["is_podium"] = 0
        d["is_winner"] = 0
        drivers.append(d)

    return pd.DataFrame(drivers)


def load_race_from_csv(path: str) -> pd.DataFrame:
    """
    Load race input from a CSV file.

    Required columns: driver, grid_clean, season, round
    Optional columns: all other features (defaults applied if missing).
    """
    df = pd.read_csv(path)
    required = {"driver", "grid_clean", "season", "round"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Fill missing feature columns with neutral defaults
    if "pit_lane_start" not in df.columns:
        df["pit_lane_start"] = (df["grid_clean"] == 0).astype(int)
    if "era" not in df.columns:
        df["era"] = df["season"].apply(_compute_era)

    for col, default_val in NEUTRAL.items():
        if col not in df.columns:
            df[col] = default_val

    # Dummy labels
    if "is_podium" not in df.columns:
        df["is_podium"] = 0
    if "is_winner" not in df.columns:
        df["is_winner"] = 0

    return df


# ─── Prediction display ────────────────────────────────────────────────────
def display_predictions(race_df: pd.DataFrame, model: AdaptiveStack) -> None:
    """Run predictions and display results in a pretty table."""
    methods = ["soft", "stack_lr", "mlp_static", "mlp_adaptive"]

    print("\n" + "=" * 72)
    print("  🏎️  PODIUM PROBABILITY PREDICTIONS")
    print("=" * 72)
    season = int(race_df["season"].iloc[0])
    race_round = int(race_df["round"].iloc[0])
    print(f"  Season {season}, Round {race_round}  |  {len(race_df)} drivers\n")

    results = race_df[["driver", "grid_clean"]].copy()

    for method in methods:
        try:
            proba = model.predict_proba(race_df, methods=method)
            results[method] = proba.values
        except Exception as e:
            print(f"  [warn] Method '{method}' failed: {e}")
            results[method] = np.nan

    # Use the best available method for ranking
    rank_col = "mlp_adaptive" if "mlp_adaptive" in results.columns and results["mlp_adaptive"].notna().all() else "soft"
    results = results.sort_values(rank_col, ascending=False).reset_index(drop=True)
    results.index = results.index + 1  # 1-based ranking
    results.index.name = "Rank"

    # Format probabilities as percentages
    display_df = results.copy()
    for method in methods:
        if method in display_df.columns:
            display_df[method] = display_df[method].apply(
                lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "N/A"
            )

    display_df = display_df.rename(columns={
        "driver": "Driver",
        "grid_clean": "Grid",
        "soft": "Soft Vote",
        "stack_lr": "Stacking (LR)",
        "mlp_static": "MLP Static",
        "mlp_adaptive": "MLP Adaptive",
    })

    print(display_df.to_string())

    # Predicted podium
    top3 = results.head(3)
    print("\n  🏆 Predicted Podium:")
    for pos, (_, row) in enumerate(top3.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}[pos]
        prob = row[rank_col]
        print(f"    {medal}  P{pos}: {row['driver']:<20s}  "
              f"({prob * 100:.1f}% podium probability)")

    print("\n" + "=" * 72 + "\n")


# ─── Demo mode using synthetic data ────────────────────────────────────────
def run_demo() -> None:
    """Train on synthetic data and predict for a demo race."""
    from ensemble import make_synthethic_f1

    print("\n🏁 Running demo with synthetic data …\n")
    synthetic = make_synthethic_f1(seed=0)
    demo_features = ["grid", "quali_gap", "circuit_type", "driver_form", "team_form"]

    # Rename columns to match FEATURE_COLS expectations
    # For the demo we use the synthetic feature set directly
    base_models = [
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42),
    ]

    train = synthetic[synthetic["season"] < 2025].copy()
    model = AdaptiveStack(
        base_model=base_models,
        replay_window=40,
        learning_rate=1e-4,
        update_epochs=3,
        hidden=(16,),
        random_state=42,
    )
    model.fit(train, feature_cols=demo_features, target="is_podium", min_train_seasons=3)

    # Pick a single test race
    test_race = synthetic[(synthetic["season"] == 2025) & (synthetic["round"] == 1)].copy()

    print("=" * 72)
    print("  🏎️  DEMO PREDICTIONS — Season 2025, Round 1")
    print("=" * 72)

    methods = ["soft", "stack_lr", "mlp_static", "mlp_adaptive"]
    results = test_race[["driver", "grid", "is_podium"]].copy()
    for method in methods:
        results[method] = model.predict_proba(test_race, methods=method).values

    results = results.sort_values("mlp_adaptive", ascending=False).reset_index(drop=True)
    results.index = results.index + 1
    results.index.name = "Rank"

    display = results.copy()
    for m in methods:
        display[m] = display[m].apply(lambda x: f"{x * 100:.1f}%")
    display = display.rename(columns={
        "driver": "Driver",
        "grid": "Grid",
        "is_podium": "Actual Podium",
        "soft": "Soft Vote",
        "stack_lr": "Stacking (LR)",
        "mlp_static": "MLP Static",
        "mlp_adaptive": "MLP Adaptive",
    })
    print(display.to_string())

    top3 = results.head(3)
    print("\n  🏆 Predicted Podium:")
    for pos, (_, row) in enumerate(top3.iterrows(), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}[pos]
        actual = "✅" if row["is_podium"] == 1 else "❌"
        print(f"    {medal}  P{pos}: Driver {int(row['driver']):<5d}  "
              f"({row['mlp_adaptive'] * 100:.1f}%)  Actual: {actual}")

    actual_podium = results[results["is_podium"] == 1]["driver"].tolist()
    predicted_podium = top3["driver"].tolist()
    hits = len(set(predicted_podium) & set(actual_podium))
    print(f"\n  📊 Podium hits: {hits}/3")
    print("=" * 72 + "\n")


# ─── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1 Podium Predictor — Interactive predictions from user input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python predict.py              # interactive mode
              python predict.py --demo       # quick demo with synthetic data
              python predict.py --file input.csv  # batch prediction from CSV

            CSV format (minimum columns):
              driver, grid_clean, season, round

            All other feature columns are optional and will default to neutral
            values if not provided.
        """),
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run a demo with synthetic data."
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to a CSV file with race input data.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mlp_adaptive",
        choices=["soft", "stack_lr", "mlp_static", "mlp_adaptive"],
        help="Primary ensemble method for ranking (default: mlp_adaptive).",
    )
    args = parser.parse_args()

    # Demo mode
    if args.demo:
        run_demo()
        return

    # Load or train the model
    if FEATURE_PATH.exists():
        features = pd.read_parquet(FEATURE_PATH)
        model = train_model(features)
    else:
        print(f"[info] Feature file not found at {FEATURE_PATH}")
        print("[info] Falling back to synthetic training data …\n")
        from ensemble import make_synthethic_f1

        synthetic = make_synthethic_f1(seed=0)
        demo_features = ["grid", "quali_gap", "circuit_type", "driver_form", "team_form"]
        base_models = [
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
            RandomForestClassifier(
                n_estimators=50, max_depth=5, n_jobs=-1, random_state=42
            ),
        ]
        model = AdaptiveStack(base_model=base_models, random_state=42)
        model.fit(synthetic, feature_cols=demo_features, target="is_podium", min_train_seasons=3)
        # Override FEATURE_COLS so prediction uses the synthetic columns
        global FEATURE_COLS
        FEATURE_COLS = demo_features
        print("  ✓ Trained on synthetic data.\n")

    # Batch mode from CSV
    if args.file:
        race_df = load_race_from_csv(args.file)
        print(f"Loaded {len(race_df)} drivers from {args.file}")
        display_predictions(race_df, model)
        return

    # Interactive mode — loop
    while True:
        race_df = collect_race_input()
        display_predictions(race_df, model)

        again = input("  Predict another race? (y/n) [y]: ").strip().lower()
        if again and again != "y":
            print("\n  Goodbye! 🏁\n")
            break


if __name__ == "__main__":
    main()
