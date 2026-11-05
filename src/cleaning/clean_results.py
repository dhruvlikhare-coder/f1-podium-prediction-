# cleaning the results from the dataset
"""
Cleaning for Jolpica-F1 race results (one row per driver per race).

Expected raw columns (as produced by the download function):
    season, round, circuit, date, driver, team, grid, position, status

Usage
-----
    python -m src.cleaning.clean_results
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

Required = ["season", "round", "circuit", "date", "driver", "team", "grid", "position", "status"]
# team ids that actually appear on your data
TEAM_MAP = {
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "renault": "alpine",
    "lotus_f1": "alpine",
    "alfa": "sauber",
}

MECHANICAL = (
    "engine", "gearbox", "transmission", "hydraulics", "brakes", "suspension",
    "electrical", "power", "turbo", "mechanical", "oil", "water", "cooling",
    "overheating", "clutch", "exhaust", "fuel", "driveshaft", "differential",
    "battery", "ers", "mgu", "energy store", "control electronics",
)
ACCIDENT = ("accident", "collision", "spun off", "damage")
# to find the missing values from the key
KEY = ["season", "round", "driver", "position", "date"]


def _dnf_status(status: str) -> str:
    s = status.strip().lower()
    if s == "finished" or (s.startswith("+") and "lap" in s):
        return " finished"
    if s == "disqualified":
        return " disqualified"
    if any(k in s for k in ACCIDENT):
        return "accident"
    if any(k in s for k in MECHANICAL):
        return "mechanical"
    return s.strip() or "unknown"


def get_result(raw: pd.DataFrame, team_map: dict | None = None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in Required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["season", "round", "grid", "position"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["driver", "team", "circuit", "status"]:
        df[c] = df[c].astype("string").str.strip()
    for c in ["driver", "team", "circuit"]:
        df[c] = df[c].str.lower()
    df["team"] = df["team"].replace(TEAM_MAP if team_map is None else team_map)

    # remove unusable rows and duplicates
    before = len(df)
    df = df.dropna(subset=KEY).copy()

    if before - len(df):
        warnings.warn(f"Dropped {before - len(df)} rows with missing key fields.")
    before = len(df)
    df = df.drop_duplicates(subset=["season", "round", "driver", "position", "date"])
    if before - len(df):
        warnings.warn(f"Dropped {before - len(df)} duplicate driver race-rows.")

    # integer type
    df["grid"] = df["grid"].fillna(0).astype(int)
    df = df.reset_index(drop=True)
    # grid: 0 means pit_lane start or no time set -> treat as last
    df["pit_lane"] = (df["grid"] == 0).astype(int)
    df["grid_clean"] = df["grid"].fillna(0).astype(int)

    # Status -> finished / DNF / reason (used for reliability features)
    df["dnf_reason"] = df["status"].fillna("").map(_dnf_status)
    df["finished"] = (df["dnf_reason"] == "finished").astype(int)
    df["disqualified"] = (df["dnf_reason"] == "disqualified").astype(int)
    df["dnf"] = (~df["dnf_reason"].isin(["finished", "disqualified"])).astype(int)
    df["dnf_mechanical"] = (df["dnf_reason"] == "mechanical").astype(int)
    df["dnf_accident"] = (df["dnf_reason"] == "accident").astype(int)
    # labels to the podium
    df["is_podium"] = (df["position"] <= 3).astype(int)
    df["is_winner"] = (df["position"] == 1).astype(int)
    return df


def clean_results(raw: pd.DataFrame, team_map: dict | None = None) -> pd.DataFrame:
    """Clean raw race results and derive labels and past-race reliability fields."""
    return get_result(raw, team_map=team_map)


def validate_result(df: pd.DataFrame) -> dict:
    """Quick sanity report. Anything in the 'problems' list should be investigated."""
    race_key = ["season", "round"]
    per_race = df.groupby(race_key, dropna=False)
    n_winner = per_race["is_winner"].sum()
    n_podium = per_race["is_podium"].sum()
    n_drivers = per_race.size()

    problems = []
    bad_win = n_winner[n_winner != 1]
    bad_pod = n_podium[n_podium != 3]
    small = n_drivers[n_drivers < 15]
    dups = int(df.duplicated(subset=["season", "round", "driver"]).sum())
    if len(bad_win):
        problems.append(f"{len(bad_win)} races without exactly one winner")
    if len(bad_pod):
        problems.append(f"{len(bad_pod)} races without exactly three podium finishers")
    if len(small):
        problems.append(f"{len(small)} races with fewer than 15 drivers")
    if dups:
        problems.append(f"{dups} duplicate (season, round, driver) rows")
    return {
        "ok": not problems,
        "problems": problems,
        "n_races": int(per_race.ngroups),
        "n_drivers": int(df.shape[0]),
    }


def validate_results(df: pd.DataFrame) -> dict:
    """Backward-compatible alias."""
    return validate_result(df)


if __name__ == "__main__":
    raw_path = ROOT / "data" / "raw" / "results.parquet"
    clean_path = ROOT / "data" / "clean" / "results.parquet"
    clean_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(raw_path)
    clean = clean_results(raw)
    report = validate_results(clean)
    print(report)
    clean.to_parquet(clean_path, index=False)
    print(f"Saved {len(clean):,} cleaned rows to {clean_path}")
