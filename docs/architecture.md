# Architecture Overview

## System Architecture (from SRS Section 4)

```
 Data Sources          Data Layer                Modeling Layer                    Serving & Ops
+-------------+   +-----------------+   +----------------------------+   +----------------------+
| Jolpica-F1  |-->| Ingestion/cache |   | Base models (LR, RF, XGB)  |   | FastAPI prediction   |
| FastF1      |-->| Validation      |-->| Pairwise / ranking models  |-->| Streamlit dashboard  |
+-------------+   | Raw->Clean->Feat|   | Monte Carlo simulator      |   | Prediction log       |
                  +-----------------+   | Synthetic data (train only)|   | Monitoring & alerts  |
                                        | Stacking (OOF, time folds) |   +----------------------+
                  Backtest harness ---->| Neural layer + adaptation  |        ^
                  MLflow tracking  ---->| Calibration & normalization|   Scheduled refresh (CI/CD)
                                        +----------------------------+
```

## Project Structure

```
f1_racing/
├── src/
│   ├── ingestion/       # Jolpica-F1 API + FastF1 data download
│   ├── cleaning/        # Raw data -> cleaned Parquet
│   ├── features/        # Leak-free feature engineering
│   ├── models/          # AdaptiveStack ensemble, walk-forward backtest
│   ├── evaluation/      # Metrics, calibration
│   ├── prediction/      # Interactive + batch prediction CLI
│   └── pipeline/        # Orchestration: build_dataset, run_backtest
├── tests/               # pytest test suite
├── configs/             # YAML configuration (backtest.yaml)
├── notebooks/           # EDA scripts
├── docs/                # SRS and architecture docs
├── data/                # Raw, clean, features, backtest results (gitignored)
└── docker/              # Dockerfile
```

## Data Flow

1. **Ingest** -> `src/ingestion/jolpica.py` downloads race results -> `data/raw/`
2. **Clean** -> `src/cleaning/clean_results.py` standardizes names, derives DNF flags -> `data/clean/`
3. **Features** -> `src/features/build_features.py` computes leak-free features -> `data/features/`
4. **Train & Backtest** -> `src/models/ensemble.py` trains AdaptiveStack, `src/pipeline/run_backtest.py` runs walk-forward evaluation
5. **Predict** -> `src/prediction/predict.py` takes user input and outputs podium probabilities
