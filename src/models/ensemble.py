"""
F1 podium prediction: soft voting, stacking, neural meta-layer, adaptive updating.
 
Suggested location in the repo: src/models/ensemble.py
 
Pieces
------
1. make_base_models()      Logistic Regression, Random Forest, XGBoost
2. time_ordered_oof()      Out-of-fold base predictions using expanding season windows
                           (no future races ever used to predict an earlier one)
3. AdaptiveStack           - soft voting (mean of base probabilities)
                           - stacking with a Logistic Regression meta-model
                           - neural meta-layer (small MLP) on the base probabilities
                           - adaptive updating: after each race, fine-tune the MLP
                             with a small learning rate on a replay window
4. Probability processing - Platt calibration on later validation races and optional
                             race-level normalization
5. walk_forward()          Race-by-race backtest: predict, then update after the result
6. evaluate()              Log loss, Brier, ROC-AUC, per-race top-3 and winner hit rate
 
Required columns in the DataFrame: season, round, the target (default `is_podium`),
and the feature columns you pass in. For winner hit rate also include `is_winner`.
Features must already be leak-free (built only from data known before the race).
"""
from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _core(model: Any) -> Any:
    """Final estimator of a Pipeline (or the model itself)."""
    return model.steps[-1][1] if hasattr(model, "steps") else model
 
 
def _model_names(models: Sequence[Any]) -> list[str]:
    """Unique, readable names: base_0_LogisticRegression, base_1_RandomForestClassifier, ..."""
    return [f"base_{i}_{_core(m).__class__.__name__}" for i, m in enumerate(models)]
 
 
def _race_id(df: pd.DataFrame) -> np.ndarray:
    """One number per race that increases with time (season, then round)."""
    return (df["season"] * 100 + df["round"]).to_numpy()
 
 
def _positive_col(model: Any) -> int:
    return list(model.classes_).index(1)

class AdaptiveStack:
    """
    Adaptive stacking for ensemble predictions.
    """

    def __init__(
        self,
        base_model: list | None = None,
        meta_model: Any | None = None,
        replay_window: int = 40,
        learning_rate: float = 1e-4,
        update_epochs: int = 3,
        hidden: Sequence[int] = (16,),
        mlp_alpha: float = 1e-4,
        mlp_max_iter: int = 800,
        random_state: int = 42,
    ):
        self.base_model = base_model
        self.meta_model = (
            meta_model if meta_model is not None
            else LogisticRegression(random_state=random_state, solver="liblinear")
        )
        self.replay_window = replay_window
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs
        self.hidden = tuple(hidden)
        self.random_state = random_state
        self.mlp_alpha = mlp_alpha
        self.mlp_max_iter = mlp_max_iter

    # making the base models for the ensemble and stacking the models are used logistic regression, random forest, xgboost.
    def make_base_models(self, base_models: Sequence[Any] | None = None) -> list[Any]:
        """
        Return the ensemble base models.
        """
        if base_models is not None:
            return list(base_models)
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "The default ensemble includes XGBoost; install xgboost or pass custom base_models."
            ) from exc

        return [
            LogisticRegression(random_state=self.random_state, solver="liblinear", max_iter=1000, C=0.5),
            RandomForestClassifier(random_state=self.random_state, n_estimators=200, max_depth=5, n_jobs=-1),
            XGBClassifier(
                random_state=self.random_state,
                n_estimators=200,
                max_depth=3,
                learning_rate=0.5,
                eval_metric="logloss",
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
            ),
        ]

    # time-ordered out-of-fold predictions for base models using expanding seasons.
    def time_ordered_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seasons: Iterable[int] | None,
        feature_cols: Sequence[str],
        base_models: Sequence[Any] | None = None,
        min_train_seasons: int = 3,
    ) -> pd.DataFrame:
        """
        Out-of-fold predictions for base models using expanding seasons.
        Each season is predicted using only data from earlier seasons.
        """
        if feature_cols is None or len(feature_cols) == 0:
            raise ValueError("feature_cols must contain at least one column.")

        if base_models is None:
            base_models = self.make_base_models()

        for model in base_models:
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"{_core(model).__class__.__name__} does not support predict_proba")

        if seasons is None:
            seasons = sorted(X["season"].dropna().unique())
        else:
            seasons = sorted(pd.Series(list(seasons)).dropna().astype(int).unique())

        cols = [f"oof_{n}" for n in _model_names(base_models)]
        oof_preds = pd.DataFrame(np.nan, index=X.index, columns=cols)

        for season in seasons:
            train_idx = X["season"] < season
            test_idx = X["season"] == season
            if not train_idx.any() or not test_idx.any():
                continue
            if X.loc[train_idx, "season"].nunique() < min_train_seasons:
                continue
            if y.loc[train_idx].nunique() < 2:
                continue

            for model, model_name in zip(base_models, cols):
                fitted_model = clone(model)
                fitted_model.fit(X.loc[train_idx, feature_cols], y.loc[train_idx])
                proba = fitted_model.predict_proba(X.loc[test_idx, feature_cols])
                oof_preds.loc[test_idx, model_name] = proba[:, _positive_col(fitted_model)]

        if oof_preds.isna().all().all():
            raise ValueError("No out-of-fold predictions were generated. Check the season and feature columns.")

        return oof_preds

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
        target: str = "is_podium",
        min_train_seasons: int = 3,
        context_cols: Sequence[str] = (),
    ):
        """Fit the stack using strictly past-season OOF predictions for its meta-model."""
        if "season" not in train or target not in train:
            raise ValueError("train must include 'season' and the target column.")
        if feature_cols is None:
            feature_cols = list(getattr(self, "X_meta", []))
        if not feature_cols:
            raise ValueError("Pass feature_cols (the leak-free base-model features).")

        missing = set(feature_cols) | set(context_cols)
        missing -= set(train.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")

        self.X_meta = list(feature_cols)
        self.target = target
        self.context_cols = list(context_cols)
        prototypes = self.base_model if self.base_model else self.make_base_models()

        oof = self.time_ordered_oof(
            train,
            train[target],
            train["season"].dropna().unique(),
            self.X_meta,
            base_models=prototypes,
            min_train_seasons=min_train_seasons,
        )

        honest_rows = oof.notna().all(axis=1)
        if not honest_rows.any():
            raise ValueError("No honest OOF rows; lower min_train_seasons or provide more seasons.")

        self.meta_feature_names_ = list(oof.columns)
        oof_h = oof.loc[honest_rows, self.meta_feature_names_]
        ctx = train.loc[honest_rows]
        y_h = ctx[target].to_numpy()

        self.meta_model.fit(oof_h, y_h)

        Xm = self._meta_X(oof_h, ctx)
        self.scaler = StandardScaler()
        Xm_scaled = self.scaler.fit_transform(Xm)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden,
            alpha=self.mlp_alpha,
            learning_rate_init=self.learning_rate,
            max_iter=self.mlp_max_iter,
            random_state=self.random_state,
        ).fit(Xm_scaled, y_h)

        self.mlp_static = copy.deepcopy(self.mlp)
        self._buf_X, self._buf_y, self._buf_race = Xm, y_h, _race_id(ctx)
        self._trim_buffer()

        self.base_model = [clone(model).fit(train[self.X_meta], train[target]) for model in prototypes]
        self.base_names_ = _model_names(self.base_model)
        self.available_methods = self.base_names_ + ["soft", "stack_lr", "mlp_static", "mlp_adaptive"]
        self._train_max_race_id = _race_id(train).max()
        self._last_updated_race_id = int(self._train_max_race_id)
        return self

    def _meta_X(self, base_prob: pd.DataFrame | np.ndarray, df: pd.DataFrame) -> np.ndarray:
        base_arr = base_prob.to_numpy() if isinstance(base_prob, pd.DataFrame) else np.asarray(base_prob)
        parts = [base_arr]
        if self.context_cols:
            parts.append(df[self.context_cols].to_numpy(dtype=float))
        return np.hstack(parts) if len(parts) > 1 else base_arr

    def _trim_buffer(self) -> None:
        if not hasattr(self, "_buf_X") or self._buf_X is None:
            return
        if self.replay_window is None or self.replay_window <= 0:
            return
        races = np.unique(self._buf_race)
        if len(races) <= self.replay_window:
            return
        keep_races = races[-self.replay_window:]
        keep = np.isin(self._buf_race, keep_races)
        self._buf_X = self._buf_X[keep]
        self._buf_y = self._buf_y[keep]
        self._buf_race = self._buf_race[keep]

    def base_prob(self, df: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "base_model") or self.base_model is None:
            raise ValueError("Call fit() before base_prob().")

        prob_cols = []
        for model in self.base_model:
            proba = model.predict_proba(df[self.X_meta])
            prob_cols.append(proba[:, _positive_col(model)])

        return pd.DataFrame(
            np.column_stack(prob_cols),
            columns=[f"oof_{name}" for name in self.base_names_],
            index=df.index,
        )

    def predict_proba(self, df: pd.DataFrame, methods: str = "mlp_adaptive") -> pd.Series:
        if not hasattr(self, "base_model") or self.base_model is None:
            raise ValueError("Call fit() before predict_proba().")

        bp = self.base_prob(df)
        methods = methods.strip()

        if methods in self.base_names_:
            p = bp[f"oof_{methods}"].to_numpy()
        elif methods == "soft":
            p = bp.mean(axis=1).to_numpy()
        elif methods == "stack_lr":
            if not hasattr(self, "meta_model") or self.meta_model is None:
                raise ValueError("Call fit() before predict_proba().")
            p = self.meta_model.predict_proba(bp)[:, -1]
        elif methods == "mlp_static":
            if not hasattr(self, "mlp_static") or self.mlp_static is None:
                raise ValueError("Call fit() before predict_proba().")
            Xm = self._meta_X(bp, df)
            p = self.mlp_static.predict_proba(self.scaler.transform(Xm))[:, -1]
        elif methods == "mlp_adaptive":
            if not hasattr(self, "mlp") or self.mlp is None:
                raise ValueError("Call fit() before predict_proba().")
            Xm = self._meta_X(bp, df)
            p = self.mlp.predict_proba(self.scaler.transform(Xm))[:, -1]
        else:
            raise ValueError(f"Unknown ensemble method '{methods}'. Available: {self.available_methods}")

        return pd.Series(p, index=df.index, name=methods)

    def update(self, race_df: pd.DataFrame) -> None:
        """
        Call AFTER a race result is known. Fine-tunes only the neural meta-layer: small
        learning rate, a few epochs, mixed with a window of older races (replay) so one
        odd race cannot overwrite what was learned. Base models are not changed here.
        """
        if not hasattr(self, "mlp"):
            raise ValueError("Call fit() before update().")
        required = {"season", "round", self.target}
        missing = required - set(race_df.columns)
        if missing:
            raise ValueError(f"race_df is missing required columns: {sorted(missing)}")
        if race_df.empty:
            raise ValueError("race_df must contain at least one driver result.")
        if race_df[self.target].isna().any() or not np.isin(race_df[self.target], [0, 1]).all():
            raise ValueError(f"{self.target} must contain binary labels encoded as 0 and 1.")
        race_ids = _race_id(race_df)
        if np.unique(race_ids).size != 1:
            raise ValueError("update() expects results from exactly one race.")
        if race_ids.min() <= self._last_updated_race_id:
            raise ValueError("Updates must be provided in strictly chronological order.")

        bp = self.base_prob(race_df)
        self._buf_X = np.vstack([self._buf_X, self._meta_X(bp, race_df)])
        self._buf_y = np.concatenate([self._buf_y, race_df[self.target].to_numpy()])
        self._buf_race = np.concatenate([self._buf_race, race_ids])
        self._trim_buffer()

        for _ in range(self.update_epochs):
            self.mlp.partial_fit(
                self.scaler.transform(self._buf_X),
                self._buf_y,
                classes=np.array([0, 1]),
            )
        self._last_updated_race_id = int(race_ids[0])

    def fit_probability_calibrator(
        self,
        validation_df: pd.DataFrame,
        methods: str = "mlp_static",
    ) -> AdaptiveStack:
        """Fit Platt calibration on held-out races later than the training period."""
        if not hasattr(self, "_train_max_race_id"):
            raise ValueError("Call fit() before fitting a probability calibrator.")
        required = {"season", "round", self.target}
        missing = required - set(validation_df.columns)
        if missing:
            raise ValueError(f"validation_df is missing required columns: {sorted(missing)}")
        if validation_df.empty or _race_id(validation_df).min() <= self._train_max_race_id:
            raise ValueError("Calibration races must be chronologically later than all training races.")
        y = validation_df[self.target].to_numpy()
        if pd.isna(y).any() or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
            raise ValueError("Calibration labels must contain both classes, encoded as 0 and 1.")

        probabilities = self.predict_proba(validation_df, methods=methods).to_numpy()
        eps = np.finfo(float).eps
        logits = np.log(np.clip(probabilities, eps, 1 - eps) /
                        np.clip(1 - probabilities, eps, 1))
        self.probability_calibrator_ = LogisticRegression(solver="liblinear")
        self.probability_calibrator_.fit(logits.reshape(-1, 1), y)
        self.calibrator_method_ = methods
        self._calibration_max_race_id = _race_id(validation_df).max()
        return self

    @staticmethod
    def normalize_race_probabilities(
        probabilities: Sequence[float] | np.ndarray,
        df: pd.DataFrame,
        expected_positives: int,
        race_cols: Sequence[str] = ("season", "round"),
    ) -> np.ndarray:
        """Rescale each race's probabilities to sum to the expected positive count."""
        if isinstance(expected_positives, bool) or not isinstance(expected_positives, (int, np.integer)):
            raise TypeError("expected_positives must be an integer.")
        if expected_positives < 0:
            raise ValueError("expected_positives must be non-negative.")
        if not race_cols:
            raise ValueError("race_cols must contain at least one column.")
        missing = set(race_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Race columns are missing: {sorted(missing)}")
        result = np.asarray(probabilities, dtype=float).copy()
        if result.ndim != 1 or len(result) != len(df):
            raise ValueError("probabilities must be a one-dimensional array aligned with df.")
        if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise ValueError("probabilities must be finite values between 0 and 1.")

        for positions in df.groupby(list(race_cols), sort=False, dropna=False).indices.values():
            race_p = result[positions]
            if len(race_p) < expected_positives:
                raise ValueError("expected_positives cannot exceed the number of rows in a race.")
            if expected_positives == 0:
                result[positions] = 0.0
                continue
            if expected_positives == len(race_p):
                result[positions] = 1.0
                continue
            if race_p.sum() == 0:
                result[positions] = expected_positives / len(race_p)
                continue
            low, high = 0.0, max(1.0, expected_positives / race_p.max())
            while np.minimum(1.0, high * race_p).sum() < expected_positives:
                high *= 2.0
            for _ in range(60):
                scale = (low + high) / 2.0
                if np.minimum(1.0, scale * race_p).sum() < expected_positives:
                    low = scale
                else:
                    high = scale
            result[positions] = np.minimum(1.0, high * race_p)
        return result

    def postprocess_probabilities(
        self,
        df: pd.DataFrame,
        methods: str = "mlp_static",
        calibrate: bool = False,
        normalize: bool = False,
        expected_positives: int = 3,
        race_cols: Sequence[str] = ("season", "round"),
    ) -> pd.Series:
        """Predict, optionally calibrate, and optionally normalize race probabilities."""
        probabilities = self.predict_proba(df, methods=methods)
        values = probabilities.to_numpy()
        if calibrate:
            if not hasattr(self, "probability_calibrator_"):
                raise ValueError("Fit a calibrator with fit_probability_calibrator() first.")
            if methods != self.calibrator_method_:
                raise ValueError(
                    f"The calibrator was fitted for '{self.calibrator_method_}', not '{methods}'."
                )
            if df.empty or _race_id(df).min() <= self._calibration_max_race_id:
                raise ValueError(
                    "Calibrated predictions must be for races later than the calibration period."
                )
            eps = np.finfo(float).eps
            logits = np.log(np.clip(values, eps, 1 - eps) /
                            np.clip(1 - values, eps, 1))
            values = self.probability_calibrator_.predict_proba(logits.reshape(-1, 1))[:, 1]

        if normalize:
            values = self.normalize_race_probabilities(
                values, df, expected_positives, race_cols
            )
        return pd.Series(values, index=df.index, name=methods)


def walk_forward(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    start_season: int,
    target: str = "is_podium",
    method: str = "mlp_adaptive",
    min_train_seasons: int = 3,
    context_cols: Sequence[str] = (),
    base_models: Sequence[Any] | None = None,
    replay_window: int = 40,
    learning_rate: float = 1e-4,
    update_epochs: int = 3,
    hidden: Sequence[int] = (16,),
    mlp_alpha: float = 1e-4,
    mlp_max_iter: int = 800,
    random_state: int = 42,
) -> pd.DataFrame:
    """Backtest each race in and after start_season, updating only after its result."""
    required = {"season", "round", target, *feature_cols, *context_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("df must contain race data.")

    train = df.loc[df["season"] < start_season].copy()
    test = df.loc[df["season"] >= start_season].copy()
    if train.empty or test.empty:
        raise ValueError("start_season must leave both earlier training data and races to test.")

    model = AdaptiveStack(
        base_model=list(base_models) if base_models is not None else None,
        replay_window=replay_window,
        learning_rate=learning_rate,
        update_epochs=update_epochs,
        hidden=hidden,
        mlp_alpha=mlp_alpha,
        mlp_max_iter=mlp_max_iter,
        random_state=random_state,
    ).fit(
        train,
        feature_cols=feature_cols,
        target=target,
        min_train_seasons=min_train_seasons,
        context_cols=context_cols,
    )

    predictions = []
    ordered_races = test.sort_values(["season", "round"], kind="stable")
    for _, race in ordered_races.groupby(["season", "round"], sort=False):
        race = race.copy()
        race["probability"] = model.predict_proba(race, methods=method).to_numpy()
        race["method"] = method
        predictions.append(race)
        model.update(race)
    return pd.concat(predictions).sort_values(["season", "round"], kind="stable")


def evaluate(
    results: pd.DataFrame,
    target: str = "is_podium",
    probability_col: str = "probability",
    race_cols: Sequence[str] = ("season", "round"),
    winner_col: str = "is_winner",
    top_k: int = 3,
) -> dict[str, float | int | None]:
    """Evaluate probability quality and race-level podium/winner ranking."""
    required = {target, probability_col, *race_cols}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing required columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("results must contain at least one row.")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    y = results[target].to_numpy()
    probabilities = results[probability_col].to_numpy(dtype=float)
    if pd.isna(y).any() or not np.isin(y, [0, 1]).all():
        raise ValueError(f"{target} must contain binary labels encoded as 0 and 1.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{probability_col} must contain finite values between 0 and 1.")
    if winner_col in results.columns:
        winner_values = results[winner_col].to_numpy()
        if pd.isna(winner_values).any() or not np.isin(winner_values, [0, 1]).all():
            raise ValueError(f"{winner_col} must contain binary labels encoded as 0 and 1.")
    else:
        winner_values = None

    auc = (
        float(roc_auc_score(y, probabilities))
        if np.unique(y).size == 2
        else float("nan")
    )
    podium_recall = []
    winner_hits = []
    groups = results.groupby(list(race_cols), sort=False, dropna=False).indices
    for group_positions in groups.values():
        positions = np.asarray(group_positions, dtype=int)
        if len(positions) < top_k:
            raise ValueError(f"Each race must have at least top_k={top_k} entries.")
        selected = positions[np.argsort(-probabilities[positions], kind="stable")[:top_k]]
        actual_podium = y[positions] == 1
        if actual_podium.any():
            podium_recall.append(float(actual_podium[np.isin(positions, selected)].sum()
                                       / actual_podium.sum()))
        if winner_values is not None:
            winner_hits.append(float(winner_values[selected[0]] == 1))

    return {
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "roc_auc": auc,
        "top3_hit_rate": float(np.mean(podium_recall)) if podium_recall else float("nan"),
        "winner_hit_rate": float(np.mean(winner_hits)) if winner_hits else None,
        "n_races": int(results.groupby(list(race_cols), dropna=False).ngroups),
    }
def make_synthethic_f1( seed : int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows  , n_drivers , n_teams = [] , 20 , 10
    skill = rng.normal(0, 1, n_drivers)
    team_of = np.arange(n_drivers) % n_teams
    for season in range(2014, 2026):
        team_pace = rng.normal(0, 1, n_teams)
        for rnd in range(1, 23):
            circuit_type = rng.integers(0, 2)
            pace = skill * 0.5 + team_pace[team_of] + rng.normal(0, 0.3, n_drivers)
            quali = pace + rng.normal(0, 0.6, n_drivers)
            race = pace + rng.normal(0, 1.2, n_drivers)
            grid = (-quali).argsort().argsort() + 1
            finish = (-race).argsort().argsort() + 1
            for d in range(n_drivers):
                rows.append(dict(season=season, round=rnd, driver=d, team=team_of[d],
                                 grid=grid[d], quali_gap=quali.max() - quali[d],
                                 circuit_type=circuit_type, finish=finish[d]))
    df = pd.DataFrame(rows).sort_values(["season", "round", "driver"]).reset_index(drop=True)
    df["driver_form"] = df.groupby("driver")["finish"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(10.5)
    tf = df.groupby(["team", "season", "round"])["finish"].mean().reset_index()
    tf["team_form"] = tf.groupby("team")["finish"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(10.5)
    df = df.merge(tf[["team", "season", "round", "team_form"]], on=["team", "season", "round"])
    df["is_podium"] = (df["finish"] <= 3).astype(int)
    df["is_winner"] = (df["finish"] == 1).astype(int)
    return df


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    synthetic_data = make_synthethic_f1(seed=0)
    feature_cols = ["grid", "quali_gap", "circuit_type", "driver_form", "team_form"]
    base_models = [
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        RandomForestClassifier(
            n_estimators=50, max_depth=5, n_jobs=-1, random_state=42
        ),
    ]
    results = walk_forward(
        synthetic_data,
        feature_cols=feature_cols,
        start_season=2022,
        base_models=base_models,
    )
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Predictions shape: {results.shape}")
    print(f"Test seasons: {sorted(results['season'].unique())}")
    print(f"Races: {results[['season', 'round']].drop_duplicates().shape[0]}")
    print(f"Metrics: {evaluate(results)}")
    print("First 10 predictions:")
    print(
        results[
            ["season", "round", "driver", "is_podium", "probability", "method"]
        ]
        .head(10)
        .to_string(index=False)
    )
            