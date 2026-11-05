from __future__ import annotations

from pathlib import Path

import pandas as pd

from clean import clean_results, validate_results
from feature import build_features, features_column

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "results.parquet"
CLEAN_PATH = ROOT / "data" / "clean" / "results.parquet"
FEATURE_PATH = ROOT / "data" / "features" / "features.parquet"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw results not found: {RAW_PATH}. Run src.py first.")

    raw = pd.read_parquet(RAW_PATH)
    clean = clean_results(raw)
    report = validate_results(clean)
    print(f"Clean validation: {report}")
    if not report["ok"]:
        raise ValueError(f"Clean results validation failed: {report['problems']}")

    features = build_features(clean)
    feature_cols = features_column(features)
    if features.duplicated(["season", "round", "driver"]).any():
        raise ValueError("Feature data contains duplicate driver-race rows.")
    if features[feature_cols].isna().any().any():
        raise ValueError("Feature data contains missing feature values.")

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PATH, index=False)
    features.to_parquet(FEATURE_PATH, index=False)
    print(
        f"Saved {len(clean):,} cleaned rows and {len(features):,} feature rows "
        f"({len(feature_cols)} features) to {FEATURE_PATH}"
    )


if __name__ == "__main__":
    main()
