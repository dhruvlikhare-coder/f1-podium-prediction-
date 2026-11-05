# 🏎️ F1 Race Winner & Podium Prediction System

A machine learning system that predicts Formula 1 race podium finishes using historical data, ensemble models, and adaptive learning.

## Features

- **Data Pipeline**: Automated ingestion from Jolpica-F1 API, cleaning, and leak-free feature engineering
- **Ensemble Models**: Logistic Regression, Random Forest, XGBoost with soft voting, stacking, and neural meta-layer
- **Adaptive Learning**: MLP meta-layer that fine-tunes after each race result
- **Walk-Forward Backtest**: Chronologically correct evaluation — no future data leakage
- **Interactive Predictions**: CLI tool for real-time predictions from user input
- **Batch Predictions**: CSV-based bulk prediction mode

## Quick Start

### 1. Setup

```bash
# Clone and create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### 2. Download Data

```bash
python -m src.ingestion.jolpica
```

### 3. Build Features

```bash
python -m src.pipeline.build_dataset
```

### 4. Run Backtest

```bash
python -m src.pipeline.run_backtest
```

### 5. Make Predictions

```bash
# Interactive mode — enter driver details manually
python -m src.prediction.predict

# Demo mode — quick demo with synthetic data
python -m src.prediction.predict --demo

# Batch mode — predict from a CSV file
python -m src.prediction.predict --file sample_race_input.csv
```

## Project Structure

```
f1_racing/
├── src/
│   ├── ingestion/       # Jolpica-F1 API + FastF1 data download
│   ├── cleaning/        # Raw data → cleaned Parquet
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

## Ensemble Methods

| Method | Description |
|--------|-------------|
| **Soft Voting** | Mean of base model probabilities |
| **Stacking (LR)** | Logistic Regression meta-model on OOF predictions |
| **MLP Static** | Neural meta-layer trained on OOF predictions |
| **MLP Adaptive** | Neural meta-layer with race-by-race fine-tuning |

## Testing

```bash
python -m pytest tests/ -v
```

## Configuration

All backtest parameters are in `configs/backtest.yaml`:
- Train/validation/test season splits
- Model hyperparameters
- Feature columns
- Random seeds for reproducibility

## Tech Stack

- **Python 3.10+**
- **pandas** / **NumPy** — data manipulation
- **scikit-learn** — base models, metrics, preprocessing
- **XGBoost** — gradient-boosted trees
- **FastF1** — practice and lap data (2018+)
- **PyYAML** — configuration
- **pytest** — testing

## License

MIT
