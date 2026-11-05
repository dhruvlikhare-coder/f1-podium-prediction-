# building the feature for the given architecture
from __future__ import annotations

import numpy as np
import pandas as pd

ERA_STARTS = (2017, 2022)
# neutral fill values for drivers/team with no history yet
NEUTRAL = {
    "driver_avg_finish": 10.5,
    "driver_podium_rate": 0.15,
    "driver_dnf_rate": 0.10,
    "team_avg_finish": 10.5,
    "team_podium_rate": 0.30,
    "team_mech_dnf_rate": 0.05,
}

ID_COLUMNS = ["season", "round", "race_id", "date", "circuit", "driver", "team"]
LABEL_COLUMNS = ["position", "is_podium", "is_winner"]

BASE_FEATURES = [
    "grid_clean", "pit_lane_start",
    "driver_avg_finish", "driver_podium_rate", "driver_dnf_rate", "driver_experience",
    "team_avg_finish", "team_podium_rate", "team_mech_dnf_rate",
    "circuit_avg_finish", "circuit_races",
    "era",
]

QUALI_FEATURES = ["quali_gap_to_pole"]


# starting to build the features defining the features
def features_column(df: pd.DataFrame) -> list[str]:
    return [c for c in BASE_FEATURES + QUALI_FEATURES if c in df.columns]


def build_features(results: pd.DataFrame,
                   window: int = 5,
                   qualifying: pd.DataFrame | None = None) -> pd.DataFrame:
    df = results.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "race_id" not in df.columns:
        df["race_id"] = df["season"] * 100 + df["round"]

    # Sorting by race id and driver keeps the feature construction leak-free.
    df = df.sort_values(["race_id", "driver"]).reset_index(drop=True)

    if df.duplicated(["race_id", "driver"]).any():
        raise ValueError("Duplicate rows are present in the column. Please run the clean step first.")
    if "grid_clean" in df.columns:
        df["pit_lane_start"] = df["grid_clean"].eq(0).astype(int)
    if "season" in df.columns:
        df["era"] = np.searchsorted(ERA_STARTS, df["season"].to_numpy(), side="right")

    def roll_feature(s: pd.Series) -> pd.Series:
        """Mean of the previous 'window' values, excluding the current one."""
        return s.shift(1).rolling(window, min_periods=1).mean()

    # Driver form from only the previous races for each driver.
    g = df.groupby("driver", sort=False)
    df["driver_avg_finish"] = g["position"].transform(roll_feature)
    df["driver_podium_rate"] = g["is_podium"].transform(roll_feature)
    df["driver_dnf_rate"] = g["dnf"].transform(roll_feature)
    df["driver_experience"] = g.cumcount()

    # Circuit history for each driver/circuit pair.
    gc = df.groupby(["driver", "circuit"], sort=False)
    df["circuit_races"] = gc.cumcount()
    df["circuit_avg_finish"] = gc["position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Team form aggregated per race, then rolled over earlier races.
    team_race = (
        df.groupby(["team", "race_id"], as_index=False)
        .agg(
            t_pos=("position", "mean"),
            t_podium=("is_podium", "max"),
            t_mech=("dnf_mechanical", "mean"),
        )
        .sort_values(["team", "race_id"])
    )
    gt = team_race.groupby("team", sort=False)
    team_race["team_avg_finish"] = gt["t_pos"].transform(roll_feature)
    team_race["team_podium_rate"] = gt["t_podium"].transform(roll_feature)
    team_race["team_mech_dnf_rate"] = gt["t_mech"].transform(roll_feature)

    df = df.merge(
        team_race[["team", "race_id", "team_avg_finish", "team_podium_rate",
                    "team_mech_dnf_rate"]],
        on=["team", "race_id"],
        how="left",
    )

    # Fill missing values in the history with neutral constants.
    for col, val in NEUTRAL.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    df["circuit_avg_finish"] = df["circuit_avg_finish"].fillna(df["driver_avg_finish"])

    # Optional qualifying data if provided.
    if qualifying is not None and "quali_gap_to_pole" in qualifying.columns:
        quali = qualifying[["race_id", "driver", "quali_gap_to_pole"]].drop_duplicates()
        df = df.merge(quali, on=["race_id", "driver"], how="left")

    # columns id and label + feature column
    cols = ID_COLUMNS + LABEL_COLUMNS + features_column(df)
    out = df[cols].copy()

    feat = features_column(out)
    if out[feat].isnull().any().any():
        bad = out[feat].isnull().sum()
        raise ValueError(f"Missing values in features:\n{bad[bad > 0]}")

    return out
