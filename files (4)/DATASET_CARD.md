# Synthetic power-unit telemetry (dataset card)

**Synthetic data. Not real telemetry.** Baseline channels follow the structure of FastF1 car data (parametric lap model unless real laps were supplied); sensor channels and anomalies are simulated with assumed numbers. Scores on this data do not describe performance on real power units.

## Configuration
```json
{
  "n_runs": 240,
  "laps_per_run": 8,
  "hz": 4,
  "anomaly_run_share": 0.4,
  "severity_threshold": 0.1,
  "window_s": 10,
  "seed": 0
}
```

## Size
- telemetry rows: 600,753
- runs: 240
- 10 s windows: 14,952

## Runs per split and anomaly type
```
anomaly_type  normal  pressure_drop  sensor_noise  sensor_stuck  thermal_drift  vibration_rise
split                                                                                         
test              30              8             8            10             11              11
train             86              0             0             0              0               0
val               28             11            11             9              9               8
```

## Files
- `telemetry.parquet`: one row per sample
- `runs.csv`: one row per run
- `windows.parquet`: one row per window with engineered features

## Columns (telemetry)
- baseline: `time_s, rpm, throttle, speed_kmh, gear, brake, lap`
- context: `run_id, unit_id, ambient_c, load, split`
- simulated sensors: `coolant_temp_c, oil_temp_c, oil_pressure_bar, fuel_pressure_bar, vibration_g`
- labels: `severity` (0-1), `is_anomaly` (severity >= threshold), `anomaly_type`

## Anomaly types
- `thermal_drift`: coolant and oil temperature drift upward, more under load
- `pressure_drop`: oil pressure sags, fuel pressure droops under load
- `sensor_noise`: one sensor's noise grows, with spikes
- `sensor_stuck`: one sensor flatlines
- `vibration_rise`: vibration grows at high rpm

## Built-in difficulty
Ambient temperature changes the normal temperature level between runs, and each of 8 power units has its own offsets, so a fixed absolute threshold is not enough.

## Splits
By run. `train` has normal runs only (for unsupervised training); `val`/`test` mix normal and anomalous runs. Windows inherit their run's split.

## Known limits
The injection rules are mine, so detecting them is partly circular. Degradation curves are assumed, not physical. Real faults are rarer, messier, and multi-causal.
