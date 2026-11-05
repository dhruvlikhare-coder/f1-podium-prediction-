"""
FastF1 session loader — caches lap, tyre, and weather data.

Usage
-----
    python -m src.ingestion.fastf1_ingest
"""
import os
from pathlib import Path

import fastf1

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = str(ROOT / "data" / "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_race_session(year: int, round_number: int):
    """Load a race session and return laps with weather columns."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, messages=False)

    laps = session.laps.reset_index(drop=True)
    weather = laps.get_weather_data().reset_index(drop=True)
    laps[["AirTemp", "TrackTemp", "Humidity", "Rainfall"]] = weather[
        ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
    ]

    cols = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "TyreLife",
        "Stint",
        "AirTemp",
        "TrackTemp",
        "Rainfall",
    ]
    return laps[cols]


if __name__ == "__main__":
    laps = load_race_session(2024, 1)
    print(laps.head())
