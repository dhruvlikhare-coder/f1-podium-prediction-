"""
Synthetic power-unit telemetry with labelled anomalies, for unsupervised anomaly detection.

Suggested location in the repo: src/synthetic/telemetry.py

IMPORTANT - what is real and what is assumed
--------------------------------------------
* Baseline channels (rpm, throttle, speed, gear, brake) follow the structure of FastF1 car data.
  FastF1 needs network access, so by default this module builds a *parametric lap model*
  (straights, braking zones, corners). To use real laps, convert them with
  `baseline_from_fastf1()` and pass them as `baseline_laps=` (see `load_fastf1_baseline_laps`).
* Sensor channels (coolant/oil temperature, oil and fuel pressure, vibration) are NOT in public
  F1 data. They are simulated from simple physics-like relationships with PLAUSIBLE but
  ASSUMED numbers. They are not those of any real power unit.
* Anomalies (thermal drift, pressure drop, sensor noise, stuck sensor, vibration rise) are
  injected by rules written here, so a detector's score on this data shows it can find THESE
  patterns, not that it would find real faults.

Outputs
-------
telemetry : one row per sample (default 4 Hz) with sensors, labels (`is_anomaly`, `anomaly_type`,
            `severity`) and the run's `split`
runs      : one row per run (a stint on one power unit) with ambient temperature, anomaly type,
            onset time and split
windows   : one row per 10 s window with engineered features and window labels

Splits are by RUN. `train` contains only normal runs (unsupervised training); `val` and `test`
mix normal and anomalous runs.

CLI:  python synthetic_telemetry.py --out data/synthetic_telemetry --runs 240 --seed 0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

RPM_MAX = 13000.0
ANOMALY_TYPES = ["thermal_drift", "pressure_drop", "sensor_noise", "sensor_stuck", "vibration_rise"]
SENSORS = ["coolant_temp_c", "oil_temp_c", "oil_pressure_bar", "fuel_pressure_bar", "vibration_g"]
NOISE_STD = {"coolant_temp_c": 0.25, "oil_temp_c": 0.30, "oil_pressure_bar": 0.07,
             "fuel_pressure_bar": 3.5, "vibration_g": 0.12}
BASELINE_COLS = ["time_s", "rpm", "throttle", "speed_kmh", "gear", "brake"]
UPSHIFT_KMH = np.array([75.0, 105.0, 140.0, 175.0, 215.0, 255.0, 295.0])


@dataclass
class Config:
    n_runs: int = 240
    laps_per_run: int = 8
    hz: int = 4                       # FastF1 car data is roughly 4 Hz
    anomaly_run_share: float = 0.40   # share of runs that contain an injected anomaly
    severity_threshold: float = 0.10  # rows below this severity are NOT labelled anomalous
    window_s: int = 10
    seed: int = 0


# --------------------------------------------------------------------------- #
# 1. Baseline lap (structure of FastF1 car data)
# --------------------------------------------------------------------------- #
def make_circuit(rng: np.random.Generator, n_corners: int | None = None) -> list[dict]:
    n = n_corners or int(rng.integers(10, 17))
    circuit = []
    for _ in range(n):
        kind = rng.choice(["slow", "medium", "fast"], p=[0.35, 0.40, 0.25])
        lo, hi = {"slow": (65, 95), "medium": (110, 160), "fast": (190, 260)}[kind]
        circuit.append(dict(straight_m=float(rng.uniform(80, 420)),
                            v_corner_kmh=float(rng.uniform(lo, hi)),
                            corner_m=float(rng.uniform(30, 90))))
    return circuit


def gear_and_rpm(speed_kmh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gear = 1 + np.searchsorted(UPSHIFT_KMH, speed_kmh)
    lo = np.concatenate(([0.0], UPSHIFT_KMH))[gear - 1]
    hi = np.concatenate((UPSHIFT_KMH, [345.0]))[gear - 1]
    frac = np.clip((speed_kmh - lo) / (hi - lo), 0, 1)
    rpm_lo = np.where(gear == 1, 4500.0, 8800.0)
    return gear, rpm_lo + frac * (12600.0 - rpm_lo)


def simulate_baseline_lap(rng: np.random.Generator, circuit: Sequence[dict], hz: int, car: dict) -> pd.DataFrame:
    dt = 1.0 / hz
    vmax, a_max, a_brk = car["vmax"], car["a_max"], car["a_brake"]
    v = circuit[-1]["v_corner_kmh"] / 3.6
    speed, throttle, brake = [], [], []
    for c in circuit:
        vc = c["v_corner_kmh"] / 3.6 * rng.uniform(0.97, 1.03)
        remaining = c["straight_m"]
        while remaining > 0:
            braking_dist = max(0.0, (v * v - vc * vc) / (2 * a_brk))
            if v > vc and remaining <= braking_dist:
                v, thr, brk = max(vc, v - a_brk * dt), 0.0, 1
            else:
                v = min(vmax, v + a_max * max(0.05, 1 - (v / vmax) ** 2) * dt)
                thr, brk = 100.0 - abs(rng.normal(0, 0.6)), 0
            remaining -= v * dt
            speed.append(v), throttle.append(thr), brake.append(brk)
        remaining, base_thr = c["corner_m"], rng.uniform(30, 65)
        while remaining > 0:
            v = vc + rng.normal(0, 0.3)
            remaining -= v * dt
            speed.append(v), throttle.append(float(np.clip(base_thr + rng.normal(0, 3), 0, 100))), brake.append(0)
    speed_kmh = np.asarray(speed) * 3.6
    gear, rpm = gear_and_rpm(speed_kmh)
    return pd.DataFrame({
        "time_s": np.arange(len(speed)) * dt,
        "rpm": rpm + rng.normal(0, 40, len(speed)),
        "throttle": np.clip(throttle, 0, 100),
        "speed_kmh": speed_kmh,
        "gear": gear.astype(int),
        "brake": np.asarray(brake, dtype=int),
    })


def baseline_from_fastf1(car_data: pd.DataFrame, hz: int = 4) -> pd.DataFrame:
    """Convert FastF1 car data (columns RPM, Speed, Throttle, nGear, Brake, Time) to the baseline schema."""
    missing = {"RPM", "Speed", "Throttle", "nGear", "Brake"} - set(car_data.columns)
    if missing:
        raise ValueError(f"car_data is missing columns: {sorted(missing)}")
    time_col = next((c for c in ("Time", "SessionTime") if c in car_data.columns), None)
    if time_col is None:
        raise ValueError("car_data needs a 'Time' or 'SessionTime' column")
    t = car_data[time_col]
    t = t.dt.total_seconds().to_numpy() if pd.api.types.is_timedelta64_dtype(t) else t.to_numpy(float)
    t = t - t[0]
    grid = np.arange(0.0, t[-1], 1.0 / hz)
    interp = lambda col: np.interp(grid, t, car_data[col].astype(float).to_numpy())
    return pd.DataFrame({
        "time_s": grid, "rpm": interp("RPM"), "throttle": interp("Throttle"),
        "speed_kmh": interp("Speed"), "gear": np.rint(interp("nGear")).astype(int),
        "brake": (interp("Brake") > 0.5).astype(int),
    })


def load_fastf1_baseline_laps(year: int, gp: str | int, driver: str = "VER", session: str = "R",
                              n_laps: int = 8, hz: int = 4) -> list[pd.DataFrame]:
    """Real baseline laps from FastF1. Needs network access (NOT run in the environment this was built in)."""
    import fastf1
    s = fastf1.get_session(year, gp, session)
    s.load(telemetry=True, weather=False, messages=False)
    laps = s.laps.pick_drivers(driver).pick_quicklaps().head(n_laps)
    return [baseline_from_fastf1(lap.get_car_data(), hz) for _, lap in laps.iterlaps()]


# --------------------------------------------------------------------------- #
# 2. Sensor simulation (assumed, physics-like)
# --------------------------------------------------------------------------- #
def unit_effects(seed: int, unit: int) -> dict:
    """Engine-to-engine differences: each power unit has its own calibration offsets."""
    r = np.random.default_rng([seed, 10_000 + unit])
    return dict(coolant_off=r.normal(0, 1.5), oil_off=r.normal(0, 2.0), oilp_off=r.normal(0, 0.10),
                fuelp_off=r.normal(0, 6.0), vib_off=r.normal(0, 0.05), noise_mult=float(r.lognormal(0, 0.15)),
                tau_c=r.uniform(20, 32), tau_o=r.uniform(32, 48))


def simulate_sensors(rng: np.random.Generator, base: pd.DataFrame, hz: int, ambient: float, u: dict) -> dict:
    rpm, thr = base["rpm"].to_numpy(), base["throttle"].to_numpy()
    load = (thr / 100.0) * (np.clip(rpm, 0, RPM_MAX) / RPM_MAX) ** 1.5
    n, dt = len(base), 1.0 / hz
    coolant, oil = np.empty(n), np.empty(n)
    tc = ambient + 66.0 + 28.0 * load[0] - 3.0
    to = tc + 12.0
    for i in range(n):                                   # first-order thermal response
        tc += dt / u["tau_c"] * (ambient + 66.0 + 28.0 * load[i] - tc)
        to += dt / u["tau_o"] * (tc + 12.0 + 9.0 * load[i] - to)
        coolant[i], oil[i] = tc, to
    nm = u["noise_mult"]
    smooth_load = np.convolve(load, np.ones(4) / 4, mode="same")
    sens = {
        "coolant_temp_c": coolant + u["coolant_off"] + rng.normal(0, NOISE_STD["coolant_temp_c"] * nm, n),
        "oil_temp_c": oil + u["oil_off"] + rng.normal(0, NOISE_STD["oil_temp_c"] * nm, n),
        "oil_pressure_bar": 1.8 + 3.4 * np.clip(rpm, 0, RPM_MAX) / RPM_MAX - 0.015 * (oil - 115.0)
                            + u["oilp_off"] + rng.normal(0, NOISE_STD["oil_pressure_bar"] * nm, n),
        "fuel_pressure_bar": 240.0 + 210.0 * smooth_load + u["fuelp_off"]
                             + rng.normal(0, NOISE_STD["fuel_pressure_bar"] * nm, n),
        "vibration_g": 0.7 + 3.2 * (np.clip(rpm, 0, RPM_MAX) / RPM_MAX) ** 2 + u["vib_off"]
                       + rng.normal(0, NOISE_STD["vibration_g"] * nm, n),
    }
    sens["load"] = load
    return sens


# --------------------------------------------------------------------------- #
# 3. Anomaly injection
# --------------------------------------------------------------------------- #
def inject_anomaly(rng: np.random.Generator, sens: dict, rpm: np.ndarray, kind: str, onset_idx: int,
                   ramp_n: int, sev_max: float) -> np.ndarray:
    """Modify `sens` in place; return the per-sample severity (0 = healthy, up to sev_max)."""
    n = len(rpm)
    idx = np.arange(n)
    step = kind == "sensor_stuck"
    ramp = (idx >= onset_idx).astype(float) if step else np.clip((idx - onset_idx) / max(ramp_n, 1), 0, 1)
    sev = sev_max * ramp if not step else ramp
    load, rpm_n = sens["load"], np.clip(rpm, 0, RPM_MAX) / RPM_MAX

    if kind == "thermal_drift":                      # cooling degrades; worse under load
        extra = sev * (0.4 + 0.6 * load)
        sens["coolant_temp_c"] += 20.0 * extra
        sens["oil_temp_c"] += 26.0 * extra
    elif kind == "pressure_drop":                    # oil pump wear / leak, fuel pressure sag
        sens["oil_pressure_bar"] *= 1.0 - 0.45 * sev
        sens["fuel_pressure_bar"] -= 70.0 * sev * load
    elif kind == "sensor_noise":                     # one sensor gets noisy, with occasional spikes
        ch = SENSORS[int(rng.integers(len(SENSORS)))]
        sens[ch] += rng.normal(0, 1, n) * NOISE_STD[ch] * 6.0 * sev
        spikes = (rng.random(n) < 0.004) & (sev > 0)
        sens[ch] += spikes * rng.choice([-1, 1], n) * NOISE_STD[ch] * 10.0
    elif kind == "sensor_stuck":                     # one sensor flatlines
        ch = SENSORS[int(rng.integers(len(SENSORS)))]
        sens[ch][onset_idx:] = sens[ch][onset_idx]
    elif kind == "vibration_rise":                   # bearing wear: stronger vibration at high rpm
        sens["vibration_g"] += sev * (2.4 * rpm_n ** 2 + 0.4)
    else:
        raise ValueError(f"Unknown anomaly type: {kind}")
    return sev


# --------------------------------------------------------------------------- #
# 4. One run and the whole dataset
# --------------------------------------------------------------------------- #
def simulate_run(run_id: int, cfg: Config, circuits: Sequence[list], anomaly_type: str | None,
                 baseline_laps: Sequence[pd.DataFrame] | None = None) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng([cfg.seed, run_id])
    car = dict(vmax=rng.uniform(88, 95), a_max=rng.uniform(10.0, 12.0), a_brake=rng.uniform(34, 42))
    circuit = circuits[run_id % len(circuits)]
    laps = []
    for lap in range(cfg.laps_per_run):
        b = (baseline_laps[lap % len(baseline_laps)].copy() if baseline_laps
             else simulate_baseline_lap(rng, circuit, cfg.hz, car))
        b["lap"] = lap + 1
        laps.append(b)
    base = pd.concat(laps, ignore_index=True)
    base["time_s"] = np.arange(len(base)) / cfg.hz

    unit = run_id % 8
    ambient = float(rng.uniform(14, 38))
    sens = simulate_sensors(rng, base, cfg.hz, ambient, unit_effects(cfg.seed, unit))

    n, duration = len(base), len(base) / cfg.hz
    sev = np.zeros(n)
    onset_s, sev_max = np.nan, 0.0
    if anomaly_type:
        onset_idx = int(rng.uniform(0.30, 0.70) * n)          # onset sits exactly on the sample grid
        onset_s = onset_idx / cfg.hz
        sev_max = float(rng.uniform(0.5, 1.0))
        ramp_n = int(rng.uniform(60, 240) * cfg.hz)
        sev = inject_anomaly(rng, sens, base["rpm"].to_numpy(), anomaly_type, onset_idx, ramp_n, sev_max)

    is_anom = (sev >= cfg.severity_threshold).astype(int)
    tel = base.copy()
    tel.insert(0, "run_id", run_id)
    tel["unit_id"] = unit
    tel["ambient_c"] = ambient
    tel["load"] = sens["load"]
    for s in SENSORS:
        tel[s] = sens[s]
    tel["severity"] = sev
    tel["is_anomaly"] = is_anom
    tel["anomaly_type"] = np.where(is_anom == 1, anomaly_type or "normal", "normal")
    meta = dict(run_id=run_id, unit_id=unit, ambient_c=round(ambient, 2), circuit_id=run_id % len(circuits),
                anomaly_type=anomaly_type or "normal", onset_s=None if not anomaly_type else round(onset_s, 2),
                max_severity=round(sev_max, 3), duration_s=round(duration, 1), n_samples=n)
    return tel, meta


def assign_splits(runs: pd.DataFrame, seed: int) -> pd.Series:
    """Run-level splits. `train` holds only normal runs; anomalous runs go to val/test."""
    r = np.random.default_rng([seed, 99])
    split = pd.Series("test", index=runs.index)
    normal = np.array(runs.index[runs["anomaly_type"] == "normal"])
    anom = np.array(runs.index[runs["anomaly_type"] != "normal"])
    r.shuffle(normal), r.shuffle(anom)
    n_tr, n_va = int(0.6 * len(normal)), int(0.2 * len(normal))
    split.loc[normal[:n_tr]] = "train"
    split.loc[normal[n_tr:n_tr + n_va]] = "val"
    split.loc[anom[:len(anom) // 2]] = "val"
    return split


def simulate_dataset(cfg: Config | None = None, baseline_laps: Sequence[pd.DataFrame] | None = None):
    cfg = cfg or Config()
    rng = np.random.default_rng([cfg.seed, 1])
    circuits = [make_circuit(np.random.default_rng([cfg.seed, 500 + k])) for k in range(6)]
    n_anom = int(round(cfg.anomaly_run_share * cfg.n_runs))
    kinds = [None] * (cfg.n_runs - n_anom) + [ANOMALY_TYPES[i % len(ANOMALY_TYPES)] for i in range(n_anom)]
    rng.shuffle(kinds)
    parts, metas = [], []
    for run_id, kind in enumerate(kinds):
        tel, meta = simulate_run(run_id, cfg, circuits, kind, baseline_laps)
        parts.append(tel), metas.append(meta)
    runs = pd.DataFrame(metas)
    runs["split"] = assign_splits(runs, cfg.seed)
    telemetry = pd.concat(parts, ignore_index=True).merge(runs[["run_id", "split"]], on="run_id")
    float_cols = telemetry.select_dtypes("float64").columns
    telemetry[float_cols] = telemetry[float_cols].astype("float32")
    return telemetry, runs


# --------------------------------------------------------------------------- #
# 5. Windowed feature table
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    "rpm_mean", "throttle_mean", "speed_mean", "load_mean", "ambient_c",
    "coolant_mean", "coolant_std", "coolant_delta", "oil_temp_mean", "oil_temp_std", "oil_temp_delta",
    "oil_pressure_mean", "oil_pressure_std", "fuel_pressure_mean", "fuel_pressure_std",
    "vibration_mean", "vibration_std", "coolant_minus_ambient", "oil_minus_coolant",
    "oilp_per_krpm", "vib_per_rpm2",
    "coolant_dstd", "oil_temp_dstd", "oil_pressure_dstd", "fuel_pressure_dstd", "vibration_dstd",
]


def make_windows(tel: pd.DataFrame, window_s: int = 10, hz: int = 4) -> pd.DataFrame:
    t = tel.copy()
    t["win"] = (t["time_s"] // window_s).astype(int)
    for s in SENSORS:
        t[f"d_{s}"] = t.groupby("run_id")[s].diff()
    g = t.groupby(["run_id", "win"])
    w = g.agg(
        n=("rpm", "size"), split=("split", "first"), time_s=("time_s", "first"),
        rpm_mean=("rpm", "mean"), throttle_mean=("throttle", "mean"), speed_mean=("speed_kmh", "mean"),
        load_mean=("load", "mean"), ambient_c=("ambient_c", "first"),
        coolant_mean=("coolant_temp_c", "mean"), coolant_std=("coolant_temp_c", "std"),
        coolant_first=("coolant_temp_c", "first"), coolant_last=("coolant_temp_c", "last"),
        oil_temp_mean=("oil_temp_c", "mean"), oil_temp_std=("oil_temp_c", "std"),
        oil_temp_first=("oil_temp_c", "first"), oil_temp_last=("oil_temp_c", "last"),
        oil_pressure_mean=("oil_pressure_bar", "mean"), oil_pressure_std=("oil_pressure_bar", "std"),
        fuel_pressure_mean=("fuel_pressure_bar", "mean"), fuel_pressure_std=("fuel_pressure_bar", "std"),
        vibration_mean=("vibration_g", "mean"), vibration_std=("vibration_g", "std"),
        coolant_dstd=("d_coolant_temp_c", "std"), oil_temp_dstd=("d_oil_temp_c", "std"),
        oil_pressure_dstd=("d_oil_pressure_bar", "std"), fuel_pressure_dstd=("d_fuel_pressure_bar", "std"),
        vibration_dstd=("d_vibration_g", "std"),
        severity_mean=("severity", "mean"), anomaly_frac=("is_anomaly", "mean"),
    ).reset_index()
    w = w[w["n"] >= 0.8 * window_s * hz].copy()
    w["coolant_delta"] = w["coolant_last"] - w["coolant_first"]
    w["oil_temp_delta"] = w["oil_temp_last"] - w["oil_temp_first"]
    w["coolant_minus_ambient"] = w["coolant_mean"] - w["ambient_c"]
    w["oil_minus_coolant"] = w["oil_temp_mean"] - w["coolant_mean"]
    w["oilp_per_krpm"] = w["oil_pressure_mean"] / (w["rpm_mean"] / 1000.0)
    w["vib_per_rpm2"] = w["vibration_mean"] / (w["rpm_mean"] / 1e4) ** 2
    w["is_anomaly"] = (w["anomaly_frac"] >= 0.5).astype(int)
    kinds = (t[t["anomaly_type"] != "normal"].groupby(["run_id", "win"])["anomaly_type"]
             .agg(lambda s: s.mode().iat[0]).rename("window_type").reset_index())
    w = w.merge(kinds, on=["run_id", "win"], how="left")
    w["window_type"] = np.where(w["is_anomaly"] == 1, w["window_type"].fillna("normal"), "normal")
    return w.drop(columns=["coolant_first", "coolant_last", "oil_temp_first", "oil_temp_last"])


# --------------------------------------------------------------------------- #
# 6. Sanity check: is the dataset learnable but not trivial?
# --------------------------------------------------------------------------- #
def sanity_check(windows: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Isolation Forest trained on normal `train` windows only; AUC per anomaly type on `test`."""
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    feats = FEATURE_COLUMNS
    tr, va, te = (windows[windows["split"] == s] for s in ("train", "val", "test"))
    scaler = StandardScaler().fit(tr[feats])
    model = IsolationForest(n_estimators=300, random_state=seed).fit(scaler.transform(tr[feats]))
    score = lambda d: -model.score_samples(scaler.transform(d[feats]))      # higher = more anomalous
    thr = np.quantile(score(va[va["is_anomaly"] == 0]), 0.95)               # 5% false-alarm rate on val
    te = te.assign(score=score(te), naive=te["coolant_mean"])               # naive: absolute coolant temp
    normal = te[te["is_anomaly"] == 0]
    rows = []
    for kind in ["ALL"] + ANOMALY_TYPES:
        a = te[te["is_anomaly"] == 1] if kind == "ALL" else te[te["window_type"] == kind]
        if len(a) == 0:
            continue
        both = pd.concat([normal, a])
        rows.append(dict(anomaly=kind, windows=len(a),
                         iforest_auc=roc_auc_score(both["is_anomaly"], both["score"]),
                         naive_temp_auc=roc_auc_score(both["is_anomaly"], both["naive"]),
                         iforest_recall_at_5pct_fpr=float((a["score"] > thr).mean())))
    return pd.DataFrame(rows).set_index("anomaly").round(3)


# --------------------------------------------------------------------------- #
# 7. Save + CLI
# --------------------------------------------------------------------------- #
def write_dataset_card(path: Path, cfg: Config, runs: pd.DataFrame, tel: pd.DataFrame, win: pd.DataFrame) -> None:
    counts = runs.groupby(["split", "anomaly_type"]).size().unstack(fill_value=0)
    path.write_text(
        "# Synthetic power-unit telemetry (dataset card)\n\n"
        "**Synthetic data. Not real telemetry.** Baseline channels follow the structure of FastF1 car data "
        "(parametric lap model unless real laps were supplied); sensor channels and anomalies are simulated "
        "with assumed numbers. Scores on this data do not describe performance on real power units.\n\n"
        f"## Configuration\n```json\n{json.dumps(asdict(cfg), indent=2)}\n```\n\n"
        f"## Size\n- telemetry rows: {len(tel):,}\n- runs: {len(runs)}\n- 10 s windows: {len(win):,}\n\n"
        f"## Runs per split and anomaly type\n```\n{counts.to_string()}\n```\n\n"
        "## Files\n- `telemetry.parquet`: one row per sample\n- `runs.csv`: one row per run\n"
        "- `windows.parquet`: one row per window with engineered features\n\n"
        "## Columns (telemetry)\n"
        "- baseline: `time_s, rpm, throttle, speed_kmh, gear, brake, lap`\n"
        "- context: `run_id, unit_id, ambient_c, load, split`\n"
        "- simulated sensors: `coolant_temp_c, oil_temp_c, oil_pressure_bar, fuel_pressure_bar, vibration_g`\n"
        "- labels: `severity` (0-1), `is_anomaly` (severity >= threshold), `anomaly_type`\n\n"
        "## Anomaly types\n- `thermal_drift`: coolant and oil temperature drift upward, more under load\n"
        "- `pressure_drop`: oil pressure sags, fuel pressure droops under load\n"
        "- `sensor_noise`: one sensor's noise grows, with spikes\n- `sensor_stuck`: one sensor flatlines\n"
        "- `vibration_rise`: vibration grows at high rpm\n\n"
        "## Built-in difficulty\nAmbient temperature changes the normal temperature level between runs, and each of 8 "
        "power units has its own offsets, so a fixed absolute threshold is not enough.\n\n"
        "## Splits\nBy run. `train` has normal runs only (for unsupervised training); `val`/`test` mix normal and "
        "anomalous runs. Windows inherit their run's split.\n\n"
        "## Known limits\nThe injection rules are mine, so detecting them is partly circular. Degradation curves are "
        "assumed, not physical. Real faults are rarer, messier, and multi-causal.\n")


def save_dataset(out: str | Path, cfg: Config, tel: pd.DataFrame, runs: pd.DataFrame, win: pd.DataFrame) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tel.to_parquet(out / "telemetry.parquet", index=False, compression="zstd")
    win.to_parquet(out / "windows.parquet", index=False, compression="zstd")
    runs.to_csv(out / "runs.csv", index=False)
    write_dataset_card(out / "DATASET_CARD.md", cfg, runs, tel, win)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="data/synthetic_telemetry")
    ap.add_argument("--runs", type=int, default=240)
    ap.add_argument("--laps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = Config(n_runs=a.runs, laps_per_run=a.laps, seed=a.seed)
    tel, runs = simulate_dataset(cfg)
    win = make_windows(tel, cfg.window_s, cfg.hz)
    save_dataset(a.out, cfg, tel, runs, win)
    print(f"{len(tel):,} samples | {len(runs)} runs | {len(win):,} windows -> {a.out}")
    print(sanity_check(win, cfg.seed))


if __name__ == "__main__":
    main()
