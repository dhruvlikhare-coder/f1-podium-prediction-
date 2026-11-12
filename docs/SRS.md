# Software Requirements Specification (SRS)

## F1 Race Winner & Podium Prediction System

| | |
|---|---|
| **Version** | 1.2 (Draft) |
| **Date** | 6 November 2026 |
| **Project type** | Machine learning / MLOps (portfolio and industry-level system) |
| **Status** | Planning |

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a machine learning system that predicts, before a Formula 1 race starts, which drivers are most likely to finish on the podium (top 3) and which driver is most likely to win. The system is intended both as a portfolio project and as an industry-grade, deployable and monitored ML product.

### 1.2 Scope
The system will:
- collect and clean historical F1 data from public sources,
- build leak-free pre-race features for every driver in every race,
- train and compare single models, ensembles, a stacking ensemble, and an optional adaptive neural layer,
- evaluate a pairwise ranking formulation as an alternative to per-driver classification,
- run controlled synthetic-data experiments and a Monte Carlo race simulator,
- provide a separate car-health status prototype based on available telemetry, clearly distinguishing synthetic signals from verified real data,
- track experiments, serve predictions through an API and dashboard, and monitor live performance,
- be containerized, tested, and automated through CI/CD.

The system will **not** use proprietary team data (for example engine temperatures or internal telemetry), place bets, or provide real-time in-race podium predictions in this version. The car-health prototype is a separate monitoring output: synthetic telemetry may be used for development and demonstration, but it must not be presented as real F1 engine data or as a confirmed mechanical diagnosis. Health or reliability information may affect podium predictions only when it is publicly sourced, available before the race being predicted, and validated in a separate backtest.

### 1.3 Definitions and Abbreviations
| Term | Meaning |
|---|---|
| F1 | Formula 1 |
| Podium | Finishing position 1, 2, or 3 |
| FP1/FP2/FP3 | Free practice sessions |
| DNF | Did Not Finish |
| ROC-AUC | Area under the Receiver Operating Characteristic curve |
| Brier score | Mean squared error of predicted probabilities |
| SHAP | SHapley Additive exPlanations (model interpretability) |
| Stacking | Ensemble where a meta-model learns to combine base-model predictions |
| OOF | Out-of-fold prediction (made on data a model was not trained on) |
| Pairwise ranking | Predicting whether driver A finishes ahead of driver B |
| Car-health status | A data-supported indicator of observed telemetry anomalies; it is not a mechanical diagnosis |
| Synthetic telemetry | Simulated sensor readings and injected patterns, not measurements from a real car |
| Concept drift | Change in data patterns over time, such as after regulation changes |
| Ablation | Experiment removing one component to measure its contribution |
| MLOps | Practices for deploying, tracking, and monitoring ML systems |
| SRS | Software Requirements Specification |

### 1.4 References
- Jolpica-F1 API (Ergast successor): historical results, qualifying, standings, pit stops
- FastF1 Python library: lap times, tyre compounds, weather, session data (2018 onward)
- scikit-learn, XGBoost, LightGBM, PyTorch, SHAP, MLflow, FastAPI, Streamlit, Docker documentation

### 1.5 Document Overview
Section 2 describes the product and the phased development plan. Section 3 lists specific requirements. Section 4 covers the architecture, Section 5 the data, Section 6 evaluation and acceptance criteria, Section 7 risks, and Section 8 future work.

---

## 2. Overall Description

### 2.1 Product Perspective
A Python-based ML system consisting of a data pipeline, a modeling and evaluation framework, a prediction API, a dashboard, and automation and monitoring components. It depends on public F1 data sources and has no connection to team or FIA systems.

### 2.2 Product Functions (Summary)
1. Data ingestion, validation, and storage
2. Leak-free feature engineering
3. Walk-forward backtesting framework
4. Model training: baseline, single models, soft voting, stacking
5. Pairwise ranking experiments
6. Synthetic-data experiments and Monte Carlo race simulation
7. Neural meta-layer with adaptive (race-by-race) updating
8. Separate telemetry-based car-health status prototype
9. Experiment tracking and model registry
10. Prediction API and dashboard
11. CI/CD, scheduled refresh, and live monitoring

### 2.3 Development Phases
Each phase must produce a usable result, so the project remains complete if stopped after any phase. Complexity is added only when it measurably improves results in the backtest.

| Phase | Goal | Output |
|---|---|---|
| 1 | Data pipeline, leak-free features, backtest harness, baseline | Clean dataset, baseline results |
| 2 | Single models, soft voting, stacking with time-ordered OOF folds | Model comparison table |
| 3 | Pairwise ranking, synthetic-data experiments, Monte Carlo simulator, and a separate synthetic telemetry-health prototype | Experiment report with ablations and a clearly labeled health-status demo |
| 4 | Neural meta-layer and adaptive updates | Ablation: does adaptation help? |
| 5 | FastAPI service, dashboard, Docker | Working demo |
| 6 | CI/CD, scheduled refresh, monitoring | Live, self-tracking system |
| 7 | Documentation, model card, write-up | Portfolio-ready release |

Recommended priority if time is limited: Phase 1 → 2 → 5 → 6 → 3 → 4 → 7.

### 2.4 User Classes
| User | Description |
|---|---|
| Developer / Analyst | Runs the pipeline, trains models, reviews experiments |
| Viewer (F1 fan) | Reads predictions through the dashboard |
| Reviewer (recruiter / engineer) | Inspects code, documentation, results, and live accuracy |

### 2.5 Operating Environment
- Python 3.10+
- Windows, macOS, or Linux; Docker for deployment
- Libraries: pandas, NumPy, scikit-learn, XGBoost, LightGBM, PyTorch, FastF1, SHAP, MLflow, FastAPI, Pydantic, Streamlit, pytest
- Internet access for data download; local or cloud storage for cached data
- No GPU required

### 2.6 Design and Implementation Constraints
- Only publicly available data may be used.
- Training, validation, and testing must be split chronologically.
- Only information known before the race start may be used as input features.
- Test and validation data must always be real races, never synthetic.
- The system must run on a standard laptop.

### 2.7 Assumptions and Dependencies
- Jolpica-F1 and FastF1 remain available with stable formats.
- Historical performance is a useful but imperfect signal for future results.
- Regulation changes (such as the 2022 and 2026 rules) reduce the relevance of older seasons, so recency weighting or era features are needed.
- F1 outcomes contain substantial randomness, so a performance ceiling exists.

---

## 3. Specific Requirements

### 3.1 Functional Requirements

#### 3.1.1 Data Collection and Validation
| ID | Requirement |
|---|---|
| FR-1 | The system shall download race results, qualifying results, standings, and pit stop data from Jolpica-F1. |
| FR-2 | The system shall download practice, lap, tyre, and weather data from FastF1 where available. |
| FR-3 | The system shall cache downloaded data locally and retry failed requests. |
| FR-4 | The system shall allow the user to choose the range of seasons to load. |
| FR-5 | The system shall validate data schemas and value ranges and report violations. |
| FR-6 | The system shall store data in separate raw, cleaned, and feature layers (for example Parquet files or DuckDB/PostgreSQL). |

#### 3.1.2 Data Cleaning and Preparation
| ID | Requirement |
|---|---|
| FR-7 | The system shall merge all sources into one table with one row per driver per race. |
| FR-8 | The system shall handle missing values using documented rules. |
| FR-9 | The system shall standardize driver, team, and circuit names across seasons and sources. |
| FR-10 | The system shall create labels `is_podium` (finish ≤ 3) and `is_winner` (finish = 1). |
| FR-11 | The system shall add a regulation-era feature to mark major rule changes. |

#### 3.1.3 Feature Engineering
| ID | Requirement |
|---|---|
| FR-12 | The system shall compute starting-position features (grid position, qualifying gap to pole). |
| FR-13 | The system shall compute driver form features (average finish and podium rate over the last N races, N configurable). |
| FR-14 | The system shall compute constructor strength features (recent average finish and points). |
| FR-15 | The system shall compute circuit-specific driver history. |
| FR-16 | The system shall compute reliability features (recent DNF rates for driver and team). |
| FR-17 | The system shall compute practice-pace features from FP sessions where available. |
| FR-18 | The system shall include context features such as circuit type and weather. |
| FR-19 | The system shall ensure no feature uses information from the race being predicted or any later race, and shall include an automated test that verifies this. |

#### 3.1.4 Backtesting Framework
| ID | Requirement |
|---|---|
| FR-20 | The system shall provide a walk-forward backtest in which, for each race, models are trained only on earlier races. |
| FR-21 | The system shall support fixed chronological train/validation/test splits as well as the walk-forward mode. |
| FR-22 | All model comparisons shall use the same backtest harness and the same test seasons. |

#### 3.1.5 Baseline and Single-Model Training
| ID | Requirement |
|---|---|
| FR-23 | The system shall implement a baseline that predicts a podium for drivers starting in the top 3. |
| FR-24 | The system shall train a Logistic Regression model. |
| FR-25 | The system shall train at least one tree-based model (Random Forest, XGBoost, or LightGBM). |
| FR-26 | The system shall support hyperparameter tuning with time-aware cross-validation. |
| FR-27 | The system shall support recency weighting of training samples. |

#### 3.1.6 Ensemble and Stacking
| ID | Requirement |
|---|---|
| FR-28 | The system shall implement a soft-voting ensemble of the base models. |
| FR-29 | The system shall implement a stacking ensemble with at least three diverse base models (for example Logistic Regression, Random Forest, XGBoost/LightGBM). |
| FR-30 | The stacking meta-model shall be trained only on out-of-fold base-model predictions generated with time-ordered (expanding-window, season-grouped) folds. |
| FR-31 | The meta-model shall be a simple model (default: Logistic Regression) to limit overfitting. |
| FR-32 | The system shall normalize predicted probabilities within each race (podium probabilities sum to 3, win probabilities sum to 1). |
| FR-33 | The system shall support probability calibration (Platt or isotonic) and report calibration curves. |

#### 3.1.7 Pairwise Ranking Experiments
| ID | Requirement |
|---|---|
| FR-34 | The system shall construct a pairwise dataset in which each row compares two drivers in the same race and the label states whether the first finished ahead of the second. |
| FR-35 | Pairwise features shall be differences (or ratios) of the two drivers' features, and pair construction shall use training-period races only for training. |
| FR-36 | The system shall train at least one pairwise model (for example a gradient-boosted classifier on pair differences) and at least one learning-to-rank model (for example LambdaMART / XGBoost ranker). |
| FR-37 | The system shall convert pairwise predictions into a per-race driver ranking and into podium and win probabilities (for example via win counts or a Bradley-Terry style aggregation). |
| FR-38 | The system shall compare pairwise and ranking models against the per-driver classification models on the same backtest. |

#### 3.1.8 Synthetic Data and Simulation Experiments
| ID | Requirement |
|---|---|
| FR-39 | Synthetic data generation shall be performed only after the chronological split and only from training-period data. |
| FR-40 | Validation and test sets shall contain only real race data. |
| FR-41 | The system shall implement at least two augmentation methods from: whole-race bootstrap resampling, Gaussian jitter on continuous features, constrained oversampling of winners, and CTGAN/TVAE tabular synthesis. |
| FR-42 | Augmentation shall preserve race structure (exactly three podium finishers and one winner per race) where applicable. |
| FR-43 | Synthetic rows shall be flagged in the data and excluded from rolling-form feature calculations. |
| FR-44 | The system shall run an ablation comparing each model with and without synthetic data on identical real test seasons, and shall keep synthetic data only if it improves log loss consistently across test seasons. |
| FR-45 | The system shall implement a Monte Carlo race simulator that samples each driver's pace from a distribution fitted to recent form, simulates many races, and outputs simulated podium and win frequencies. |
| FR-46 | Simulator outputs shall be usable both as a standalone model and as an input feature to the stacking ensemble. |
| FR-47 | The system shall provide schema-compatible fake data generation for pipeline, API, and CI testing, clearly labeled as synthetic. |
| FR-48 | The system shall include a drift stress test that simulates changes such as a top team's pace drop or a new regulation era, and reports model behavior. |

#### 3.1.9 Neural Layer and Adaptive Learning
| ID | Requirement |
|---|---|
| FR-49 | The system shall implement a neural meta-layer (small MLP, 1-2 hidden layers) that takes OOF base-model probabilities and context features and outputs podium and win probabilities. |
| FR-50 | The neural layer shall use regularization (dropout, weight decay) and early stopping on a time-ordered validation set. |
| FR-51 | The system shall optionally implement a sequence model (GRU/LSTM) over each driver's and team's last N races to produce a form embedding for the neural layer. |
| FR-52 | The system shall implement adaptive updating: after each race, only the neural layer is fine-tuned on recent results with a small learning rate, while base models are retrained on a slower schedule (configurable). |
| FR-53 | Adaptive updating shall use a replay window of older races and recency weighting to limit catastrophic forgetting and noise chasing. |
| FR-54 | Neural training shall be repeated over multiple random seeds and report mean and spread of results. |
| FR-55 | The adaptive layer shall be evaluated by walk-forward backtest, including the seasons following regulation changes. |

#### 3.1.10 Prediction
| ID | Requirement |
|---|---|
| FR-56 | The system shall output podium and win probabilities for every driver in a selected race. |
| FR-57 | The system shall rank drivers and display the predicted top 3. |
| FR-58 | The system shall allow predictions for past races (backtesting) and for the next race once pre-race data is available. |
| FR-59 | The system shall log every live prediction with a timestamp before the race starts, so later accuracy can be verified. |

#### 3.1.11 Evaluation, Ablation, and Explanation
| ID | Requirement |
|---|---|
| FR-60 | The system shall evaluate models using ROC-AUC, precision, recall, F1-score, log loss, and Brier score. |
| FR-61 | The system shall report per-race metrics: whether the predicted winner was correct and how many actual podium drivers were in the predicted top 3. |
| FR-62 | The system shall produce a results table comparing, on the same test seasons: (1) baseline, (2) best single model, (3) soft voting, (4) stacking with logistic meta-model, (5) pairwise/ranking model, (6) stacking with synthetic augmentation, (7) Monte Carlo simulator, (8) static neural layer, (9) adaptive neural layer. |
| FR-63 | The system shall report results separately by regulation era. |
| FR-64 | The system shall produce feature-importance and SHAP explanations for the best model. |

#### 3.1.12 Experiment Tracking and Model Management
| ID | Requirement |
|---|---|
| FR-65 | The system shall log parameters, metrics, data version, and artifacts of every experiment (for example in MLflow). |
| FR-66 | The system shall maintain a model registry with versioning and a clearly marked production model. |
| FR-67 | The system shall save trained models and configuration to disk. |

#### 3.1.13 API and Dashboard
| ID | Requirement |
|---|---|
| FR-68 | The system shall expose a FastAPI service with endpoints for predictions by season and round, and a health check. |
| FR-69 | API inputs and outputs shall be validated with typed schemas. |
| FR-70 | The dashboard shall let the user select a season and race and view predicted podium and win probabilities as a table and chart. |
| FR-71 | The dashboard shall show the main factors behind a prediction (SHAP). |
| FR-72 | The dashboard shall show the live track record of past predictions versus actual results. |

#### 3.1.14 Automation and Monitoring
| ID | Requirement |
|---|---|
| FR-73 | The system shall run a scheduled job that pulls new race data and refreshes predictions after qualifying and after each race. |
| FR-74 | The system shall track live performance (log loss, calibration, per-race accuracy) race by race. |
| FR-75 | The system shall monitor feature distributions for drift and flag significant changes. |
| FR-76 | The system shall raise a visible alert when live performance or drift crosses configured thresholds. |

#### 3.1.15 Car Health Status
| ID | Requirement |
|---|---|
| FR-77 | The system shall provide car-health status as a separate output from pre-race podium and win predictions. |
| FR-78 | The health monitor shall accept time-stamped telemetry samples or telemetry windows associated with an identifiable run/session and car or driver where that information is available. |
| FR-79 | The health monitor shall report a status of `normal`, `warning`, `critical`, or `unknown`, together with the observed anomaly type, severity or score, and the supporting sensor/window evidence where available. Status thresholds shall be configurable and documented. |
| FR-80 | Missing, stale, invalid, or insufficient telemetry shall produce an `unknown` status (or a clear data-quality warning), never an inferred `normal` status. |
| FR-81 | Synthetic telemetry and injected anomaly labels shall be visibly identified as synthetic in stored data and user-facing outputs. Synthetic health results shall not be represented as measurements or diagnoses of real F1 cars. |
| FR-82 | The health monitor shall support testing with the synthetic telemetry generator; performance shall be reported by anomaly type and at run/session level using held-out runs. |
| FR-83 | Health status and in-race telemetry shall not be used as inputs to a pre-race podium or win prediction. A reliability feature may be added to that model only when its source is public and its value is available before the prediction cutoff. |
| FR-84 | Before any pre-race reliability feature is enabled, the system shall compare models with and without that feature using the same chronological real-race backtest and shall report the results and data provenance. |

### 3.2 Non-Functional Requirements
| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Performance | Training the full model set (excluding CTGAN experiments) shall complete within 60 minutes on a standard laptop. |
| NFR-2 | Performance | A prediction for one race shall return within 5 seconds once data is cached. |
| NFR-3 | Reproducibility | Results shall be reproducible using fixed seeds, pinned dependencies, and logged configuration and data versions. |
| NFR-4 | Maintainability | Code shall be modular (ingestion, features, models, evaluation, serving) and driven by configuration files. |
| NFR-5 | Code quality | The project shall use type hints, linting, formatting, and pre-commit checks. |
| NFR-6 | Testing | Unit tests shall cover feature logic, leakage checks, pair construction, and the API; the test suite shall run in CI. |
| NFR-7 | Reliability | The pipeline shall handle API failures with retries and clear error messages. |
| NFR-8 | Portability | The system shall run in a Docker container and on Windows, macOS, and Linux. |
| NFR-9 | Usability | The dashboard shall be usable without programming knowledge. |
| NFR-10 | Data integrity | Chronological splitting shall be enforced in code so that test data never appears in training. |
| NFR-11 | Transparency | Synthetic data, limitations, and negative results shall be clearly documented. |
| NFR-12 | Observability | The service shall produce structured logs for ingestion, training, and prediction. |
| NFR-13 | Transparency | Car-health outputs shall distinguish observed data from synthetic data, expose data freshness/quality, and state that anomaly status is not a confirmed mechanical diagnosis. |

### 3.3 External Interface Requirements
- **Software interfaces:** HTTP requests to Jolpica-F1; FastF1 library calls; MLflow tracking; REST API exposed by the system.
- **User interface:** Streamlit (or equivalent) web dashboard; command-line scripts for training and backtesting.
- **Hardware interfaces:** None.

---

## 4. System Architecture

```
 Data Sources          Data Layer                Modeling Layer                    Serving & Ops
┌─────────────┐   ┌─────────────────┐   ┌────────────────────────────┐   ┌──────────────────────┐
│ Jolpica-F1  │──▶│ Ingestion/cache │   │ Base models (LR, RF, XGB)  │   │ FastAPI prediction   │
│ FastF1      │──▶│ Validation      │──▶│ Pairwise / ranking models  │──▶│ Streamlit dashboard  │
└─────────────┘   │ Raw→Clean→Feat. │   │ Monte Carlo simulator      │   │ Prediction log       │
                  └─────────────────┘   │ Synthetic data (train only)│   │ Monitoring & alerts  │
                                        │ Stacking (OOF, time folds) │   └──────────────────────┘
                  Backtest harness ────▶│ Neural layer + adaptation  │        ▲
                  MLflow tracking  ────▶│ Calibration & normalization│   Scheduled refresh (CI/CD)
                                        └────────────────────────────┘
```

Suggested project structure: `data/`, `src/ingestion/`, `src/features/`, `src/models/`, `src/ranking/`, `src/synthetic/`, `src/simulation/`, `src/neural/`, `src/evaluation/`, `src/serving/`, `app/`, `configs/`, `notebooks/`, `tests/`, `docker/`.

The telemetry-health monitor is a separate path from the pre-race prediction model. It may consume public telemetry when available or synthetic telemetry for a labeled prototype; it must not imply access to confidential power-unit sensors.

---

## 5. Data Requirements

### 5.1 Units of Observation
- **Classification:** one row per driver per race.
- **Pairwise ranking:** one row per ordered driver pair per race (about 190 unordered pairs for 20 drivers).
- **Telemetry health:** one row per telemetry sample or time window per run/session and car/driver, where identifiers are available; health records are separate from the driver-race prediction table.

### 5.2 Targets
- `is_podium` (binary), `is_winner` (binary, highly imbalanced; use class weights, calibration, or ranking formulations).
- `a_beats_b` (binary) for pairwise models.

### 5.3 Data Splits (Example)
| Set | Seasons |
|---|---|
| Train | 2014-2022 |
| Validation | 2023 |
| Test | 2024-2025 and later live races |

Seasons may be adjusted, but splits must always be chronological and identical across all compared models.

### 5.4 Synthetic Data Policy
- Generated from training-period data only.
- Never used for validation or testing.
- Flagged with an `is_synthetic` column and excluded from rolling-form features.
- Reported separately in every results table.

### 5.5 Telemetry and Health Data Policy
- Telemetry sources, units, sampling frequency, time coverage, and any transformations shall be recorded where known.
- The synthetic power-unit telemetry dataset is for software development and controlled anomaly-detection experiments only; its sensor values and injected faults are assumptions, not real-car measurements.
- Telemetry-health evaluation shall split by run/session (not by individual windows) to prevent samples from the same run leaking across splits.
- Synthetic telemetry shall not be mixed into real-race validation/test data or used to claim real-world engine-health performance.
- Any reliability input to podium prediction shall be derived only from public information available before the race cutoff and shall have documented provenance.

### 5.6 Data Quality Rules
- Remove or flag races with missing results.
- Treat disqualifications and non-starts consistently.
- Document every imputation rule.
- Validate telemetry timestamps, units, sensor values, and run/session identity; flag duplicate, out-of-order, or missing samples and report coverage.

---

## 6. Evaluation and Acceptance Criteria

| ID | Criterion |
|---|---|
| AC-1 | The best model shall outperform the "top-3 grid position" baseline on ROC-AUC and log loss on the test seasons. |
| AC-2 | The pipeline shall run end-to-end from raw data download to predictions without manual steps. |
| AC-3 | The automated leakage test shall pass, and no feature shall use post-race information. |
| AC-4 | The stacking ensemble shall be compared with the best single model and soft voting; results shall be reported whether or not stacking improves them. |
| AC-5 | Pairwise/ranking models shall be compared with per-driver classification on the same test seasons. |
| AC-6 | The synthetic-data ablation shall be completed and reported with and without augmentation. |
| AC-7 | The neural layer (static and adaptive) shall be compared with the plain stack across multiple seeds and seasons, including post-regulation-change seasons. |
| AC-8 | Probabilities shall be calibrated and calibration curves reported. |
| AC-9 | The results table in FR-62 shall be produced for all implemented components, with negative results reported. |
| AC-10 | The API and dashboard shall run from a Docker container, and CI shall pass (tests, linting). |
| AC-11 | At least one live race shall have predictions logged before the race and compared with the actual result. |
| AC-12 | Documentation shall include setup instructions, data description, architecture diagram, results summary, model card, and limitations. |
| AC-13 | The health monitor shall return `unknown` for missing or insufficient telemetry and shall identify synthetic input in its outputs. |
| AC-14 | The synthetic telemetry-health prototype shall be evaluated on held-out runs, with results reported separately by anomaly type and without claiming real-car diagnostic accuracy. |
| AC-15 | No health or telemetry feature shall enter pre-race podium predictions unless its pre-race availability is verified and the with/without-feature real-race ablation is reported. |

---

## 7. Risks and Limitations

| Risk | Impact | Mitigation |
|---|---|---|
| Small data (about 20-24 races per season; drivers within a race are not independent) | Overfitting, high variance between seasons | Simple models, regularization, pairwise reformulation, multi-season evaluation |
| Data leakage (random folds in stacking, rolling features including the current race, post-qualifying information) | Inflated metrics | Time-ordered folds, strict "before this race" feature rule, automated leakage test |
| Regulation changes (concept drift) | Older patterns mislead the model; slow adaptation early in a new era | Recency weighting, era feature, per-era evaluation, drift monitoring, adaptive layer |
| Adaptive learning instability (forgetting, chasing noise from a single odd race) | Degraded predictions | Small learning rate, update last layer only, replay window, conservative updates |
| Stacking adds little when base models are highly correlated | Wasted complexity | Diverse base models; compare with soft voting; accept null results |
| Synthetic data adds no new information and may add overconfidence or unrealistic samples | Misleading gains | Generate from training data only, never test on synthetic data, ablation, label clearly |
| Predictability ceiling (crashes, safety cars, weather, strategy luck) | Accuracy cannot be very high | Report probabilities, calibration, and log loss, not only accuracy |
| Class imbalance and linked outcomes within a race | Wrong number of predicted podiums | Within-race normalization, ranking formulations, class weights |
| Missing data (FastF1 starts in 2018) and name inconsistencies | Gaps in features | Imputation rules, name mapping, era-aware feature availability |
| Synthetic or sparse telemetry mistaken for actual engine health | Misleading status or unsupported diagnostic claims | Label synthetic data, expose coverage and freshness, use `unknown` when evidence is insufficient, and describe outputs as anomaly indicators rather than diagnoses |
| API or library changes | Pipeline breaks | Caching, pinned versions, schema validation |
| Project scope too large | Unfinished system | Phased plan; each phase is independently complete |
| Evaluation traps (single-season results, tuning on test data, weak baselines) | False conclusions | Fixed test seasons, walk-forward backtest, strong baseline, logged experiments |

---

## 8. Future Enhancements
- Live in-race prediction using lap-by-lap data
- Car-health monitoring using suitable public telemetry, after the synthetic prototype and data-availability review
- Tyre degradation and pit strategy modules
- Driver-versus-car performance separation
- Multi-task learning (finishing position, podium, and points jointly)
- Use of F2/F3 data for driver-ability features
- Public season-long accuracy leaderboard for the system

---

## Appendix A: Requirement Priority

| Priority | Requirements |
|---|---|
| Must have | FR-1 to FR-33 (data, features, backtest, baseline, ensembles, stacking), FR-56 to FR-58, FR-60 to FR-62, FR-64 to FR-69, NFR-1 to NFR-10 |
| Should have | FR-34 to FR-48 (pairwise ranking, synthetic experiments, Monte Carlo), FR-59, FR-63, FR-70 to FR-84 (including car-health status), NFR-13 |
| Could have | FR-49 to FR-55 (neural layer and adaptive learning), sequence model (FR-51) |

## Appendix B: Change Log

| Version | Changes |
|---|---|
| 1.0 | Initial SRS: data pipeline, baseline, single models, prediction, evaluation, optional dashboard |
| 1.1 | Added stacking with time-ordered OOF folds, pairwise ranking, synthetic-data and Monte Carlo experiments, neural meta-layer with adaptive updates, backtesting framework, experiment tracking, API, CI/CD, monitoring, phased plan, expanded risks and acceptance criteria |
| 1.2 | Added a separate telemetry-based car-health status prototype, synthetic telemetry safeguards, data-quality requirements, and criteria for any future pre-race reliability feature |
