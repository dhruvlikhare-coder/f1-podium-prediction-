# fast f1 caching the dataset fast f1
import os

import fastf1

CACHE_DIR = "data/f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

schedule = fastf1.get_event_schedule(2024)
session = fastf1.get_session(2024, 1, "R")
session.load(telemetry=False, messages=False)

laps = session.laps
weather = laps.get_weather_data().reset_index(drop=True)

laps = laps.reset_index(drop=True)
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
print(laps[cols].head())