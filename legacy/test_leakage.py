from pathlib import Path

import numpy as np
import pandas as pd

from clean import clean_results, validate_results
from feature import build_features, features_column

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "results.parquet"


def test_real_results_are_complete_and_clean():
    raw = pd.read_parquet(RAW_PATH)
    clean = clean_results(raw)
    report = validate_results(clean)

    assert report["ok"], report["problems"]
    assert report["n_races"] > 100
    assert set(range(2014, 2026)).issubset(set(clean["season"].unique()))


def test_real_feature_builder_has_no_future_outcome_leakage():
    raw = pd.read_parquet(RAW_PATH)
    clean = clean_results(raw)
    original = build_features(clean)
    changed = clean.copy()
    cutoff = int(changed.loc[changed["season"] == 2024, "season"].min() * 100 + 1)
    future = changed["season"] * 100 + changed["round"] >= cutoff

    changed.loc[future, "position"] = 21 - changed.loc[future, "position"]
    changed.loc[future, "is_podium"] = (changed.loc[future, "position"] <= 3).astype(int)
    changed.loc[future, "is_winner"] = (changed.loc[future, "position"] == 1).astype(int)
    changed.loc[future, "dnf"] = 1 - changed.loc[future, "dnf"]
    changed.loc[future, "dnf_mechanical"] = 1 - changed.loc[future, "dnf_mechanical"]
    perturbed = build_features(changed)

    feature_cols = features_column(original)
    past = original["race_id"] <= cutoff
    np.testing.assert_allclose(
        original.loc[past, feature_cols].to_numpy(),
        perturbed.loc[past, feature_cols].to_numpy(),
    )


def test_real_feature_table_has_one_complete_row_per_driver_race():
    clean = clean_results(pd.read_parquet(RAW_PATH))
    features = build_features(clean)
    feature_cols = features_column(features)

    assert not features.duplicated(["season", "round", "driver"]).any()
    assert features[feature_cols].notna().all().all()
    assert {"pit_lane_start", "driver_dnf_rate", "team_mech_dnf_rate"} <= set(feature_cols)
