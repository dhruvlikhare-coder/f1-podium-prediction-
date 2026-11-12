"""Tests for synthetic_telemetry.py. Run: pytest -q test_synthetic_telemetry.py"""
import numpy as np
import pandas as pd
import pytest
import synthetic_telemetry as st


@pytest.fixture(scope="module")
def small():
    cfg = st.Config(n_runs=24, laps_per_run=4, seed=1)
    tel, runs = st.simulate_dataset(cfg)
    return cfg, tel, runs, st.make_windows(tel, cfg.window_s, cfg.hz)


@pytest.fixture(scope="module")
def medium():
    cfg = st.Config(n_runs=100, laps_per_run=6, seed=0)
    tel, runs = st.simulate_dataset(cfg)
    return cfg, tel, runs, st.make_windows(tel, cfg.window_s, cfg.hz)


def base_frame(laps=3, seed=0):
    rng = np.random.default_rng(seed)
    circ = st.make_circuit(rng)
    car = dict(vmax=92.0, a_max=11.0, a_brake=38.0)
    b = pd.concat([st.simulate_baseline_lap(rng, circ, 4, car) for _ in range(laps)], ignore_index=True)
    b["time_s"] = np.arange(len(b)) / 4
    return b, rng


# ------------------------------------------------------------------ baseline
def test_baseline_lap_is_plausible():
    rng = np.random.default_rng(0)
    lap = st.simulate_baseline_lap(rng, st.make_circuit(rng), 4, dict(vmax=92.0, a_max=11.0, a_brake=38.0))
    assert 50 <= len(lap) / 4 <= 130                                   # lap time in seconds
    assert lap["speed_kmh"].between(0, 345).all() and lap["throttle"].between(0, 100).all()
    assert lap["gear"].between(1, 8).all() and set(lap["brake"]) <= {0, 1}
    assert lap["rpm"].between(4000, 13500).all()
    assert (lap.loc[lap["brake"] == 1, "throttle"] == 0).all()         # no throttle while braking


# ------------------------------------------------------------------ reproducibility and ranges
def test_same_seed_gives_identical_data(small):
    cfg, tel, runs, _ = small
    tel2, runs2 = st.simulate_dataset(cfg)
    pd.testing.assert_frame_equal(tel, tel2)
    pd.testing.assert_frame_equal(runs, runs2)


def test_normal_sensor_values_are_in_plausible_ranges(medium):
    _, tel, _, _ = medium
    n = tel[tel["is_anomaly"] == 0]
    assert n["coolant_temp_c"].between(80, 128).all()
    assert n["oil_temp_c"].between(95, 145).all()
    assert n["oil_pressure_bar"].between(2.0, 6.5).all()
    assert n["fuel_pressure_bar"].between(180, 500).all()
    assert n["vibration_g"].between(0.5, 5.0).all()


# ------------------------------------------------------------------ labels
def test_labels_respect_onset_and_run_type(medium):
    cfg, tel, runs, _ = medium
    for _, r in runs.iterrows():
        d = tel[tel["run_id"] == r["run_id"]]
        if r["anomaly_type"] == "normal":
            assert d["is_anomaly"].sum() == 0 and d["severity"].max() == 0
        else:
            assert d.loc[d["time_s"] < r["onset_s"], "is_anomaly"].sum() == 0   # healthy before onset
            assert d.loc[d["time_s"] >= r["onset_s"], "severity"].max() > 0     # severity starts at onset
            assert d["is_anomaly"].sum() > 0                                   # but it does develop
            assert set(d.loc[d["is_anomaly"] == 1, "anomaly_type"]) == {r["anomaly_type"]}


def test_all_anomaly_types_are_present(medium):
    assert set(medium[2]["anomaly_type"]) == {"normal", *st.ANOMALY_TYPES}


# ------------------------------------------------------------------ splits
def test_splits_are_by_run_and_train_is_normal_only(small):
    _, tel, runs, win = small
    assert runs["split"].isin(["train", "val", "test"]).all()
    assert (runs.loc[runs["split"] == "train", "anomaly_type"] == "normal").all()
    assert tel.groupby("run_id")["split"].nunique().eq(1).all()          # a run never straddles splits
    merged = win.merge(runs[["run_id", "split"]], on="run_id", suffixes=("", "_run"))
    assert (merged["split"] == merged["split_run"]).all()


# ------------------------------------------------------------------ injection effects
def test_each_injection_does_what_it_says():
    b, rng = base_frame()
    rpm, n = b["rpm"].to_numpy(), len(b)
    onset = n // 3

    def run(kind):
        sens = st.simulate_sensors(np.random.default_rng(5), b, 4, 25.0, st.unit_effects(0, 0))
        clean = {k: v.copy() for k, v in sens.items()}
        sev = st.inject_anomaly(np.random.default_rng(6), sens, rpm, kind, onset, 200, 1.0)
        return clean, sens, sev

    late = slice(n - 100, n)
    c, s, sev = run("thermal_drift")
    assert (s["coolant_temp_c"][late] - c["coolant_temp_c"][late]).mean() > 5
    assert (s["oil_temp_c"][late] - c["oil_temp_c"][late]).mean() > 5
    c, s, _ = run("pressure_drop")
    assert (s["oil_pressure_bar"][late] / c["oil_pressure_bar"][late]).mean() < 0.8
    c, s, _ = run("vibration_rise")
    assert (s["vibration_g"][late] - c["vibration_g"][late]).mean() > 0.5
    c, s, _ = run("sensor_noise")
    changed = [k for k in st.SENSORS if not np.allclose(c[k], s[k])]
    assert len(changed) == 1
    assert np.std(np.diff(s[changed[0]][onset:])) > 1.5 * np.std(np.diff(c[changed[0]][onset:]))
    c, s, _ = run("sensor_stuck")
    stuck = [k for k in st.SENSORS if not np.allclose(c[k], s[k])]
    assert len(stuck) == 1 and np.ptp(s[stuck[0]][onset:]) == 0           # flatline from onset
    assert np.all(sev[:onset] == 0)                                       # healthy before onset


def test_unknown_anomaly_type_raises():
    b, rng = base_frame(laps=1)
    sens = st.simulate_sensors(rng, b, 4, 25.0, st.unit_effects(0, 0))
    with pytest.raises(ValueError):
        st.inject_anomaly(rng, sens, b["rpm"].to_numpy(), "not_a_type", 10, 10, 1.0)


# ------------------------------------------------------------------ windows
def test_windows_are_clean_and_consistent(small):
    cfg, _, _, win = small
    assert not win[st.FEATURE_COLUMNS].isna().any().any()
    assert not win.duplicated(["run_id", "win"]).any()
    assert win["is_synthetic"].all()
    assert ((win["anomaly_frac"] >= 0.5).astype(int) == win["is_anomaly"]).all()
    assert (win.loc[win["is_anomaly"] == 0, "window_type"] == "normal").all()
    assert (win["n"] >= 0.8 * cfg.window_s * cfg.hz).all()                # partial windows dropped


def test_features_contain_no_label_information():
    banned = ("severity", "anomaly", "split", "window_type", "run_id")
    assert not any(b in f for f in st.FEATURE_COLUMNS for b in banned)


# ------------------------------------------------------------------ built-in difficulty
def test_ambient_temperature_is_a_nuisance_variable(medium):
    _, _, _, win = medium
    normal = win[win["is_anomaly"] == 0]
    assert normal["ambient_c"].corr(normal["coolant_mean"]) > 0.4         # hot day = hotter coolant, still normal


def test_detector_beats_a_fixed_temperature_threshold(medium):
    """Overall only: per-type numbers swing a lot with which few runs land in the test split."""
    _, _, _, win = medium
    res = st.sanity_check(win, seed=0)
    assert res.loc["ALL", "iforest_auc"] > 0.7
    assert res.loc["ALL", "iforest_auc"] > res.loc["ALL", "naive_temp_auc"] + 0.15


# ------------------------------------------------------------------ FastF1 adapter
def fake_fastf1(n=400, seed=0):
    r = np.random.default_rng(seed)
    t = np.cumsum(r.uniform(0.18, 0.30, n))                              # irregular ~4 Hz sampling
    return pd.DataFrame({
        "Time": pd.to_timedelta(t, unit="s"), "RPM": r.uniform(8000, 12500, n),
        "Speed": r.uniform(80, 320, n), "Throttle": r.uniform(0, 100, n),
        "nGear": r.integers(2, 9, n).astype(float), "Brake": r.random(n) < 0.2,
    })


def test_fastf1_adapter_returns_the_baseline_schema():
    out = st.baseline_from_fastf1(fake_fastf1(), hz=4)
    assert list(out.columns) == st.BASELINE_COLS
    assert np.allclose(np.diff(out["time_s"]), 0.25)                     # regular 4 Hz grid
    assert out["gear"].between(1, 8).all() and set(out["brake"]) <= {0, 1}


def test_fastf1_adapter_validates_input():
    with pytest.raises(ValueError):
        st.baseline_from_fastf1(fake_fastf1().drop(columns=["RPM"]))
    with pytest.raises(ValueError):
        st.baseline_from_fastf1(fake_fastf1().drop(columns=["Time"]))


def test_real_style_laps_can_drive_the_simulator():
    laps = [st.baseline_from_fastf1(fake_fastf1(seed=s)) for s in range(2)]
    cfg = st.Config(n_runs=4, laps_per_run=3, anomaly_run_share=0.5, seed=2)
    tel, runs = st.simulate_dataset(cfg, baseline_laps=laps)
    assert len(tel) > 0 and tel["coolant_temp_c"].notna().all() and len(runs) == 4
