"""
Download race results from the Jolpica-F1 (Ergast successor) API.

Usage
-----
    python -m src.ingestion.jolpica          # downloads 2014–2025 to data/raw/
"""
import os
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1" 
PAGE_SIZE = 100
ROOT = Path(__file__).resolve().parent.parent.parent


def get_season_result(season: int) -> pd.DataFrame:
    rows = []
    offset = 0
    total = None
    while total is None or offset < total:
        response = requests.get(
            f"{BASE_URL}/{season}/results.json",
            params={"limit": PAGE_SIZE, "offset": offset},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()["MRData"]
        total = int(data["total"])
        races = data["RaceTable"]["Races"]

        for race in races:
            for result in race["Results"]:
                rows.append(
                    {
                        "season": int(race["season"]),
                        "round": int(race["round"]),
                        "circuit": race["Circuit"]["circuitId"],
                        "date": race["date"],
                        "driver": result["Driver"]["driverId"],
                        "team": result["Constructor"]["constructorId"],
                        "grid": int(result["grid"]),
                        "position": int(result["position"]),
                        "status": result["status"],
                    }
                )

        if not races:
            raise RuntimeError(
                f"Jolpica returned no results at offset {offset} of {total} "
                f"for season {season}."
            )
        offset += int(data["limit"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [get_season_result(year) for year in range(2014, 2026)], ignore_index=True
    )
    df.to_parquet(raw_dir / "results.parquet")
    print(f"Saved {len(df):,} rows to {raw_dir / 'results.parquet'}")
