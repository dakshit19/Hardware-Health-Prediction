"""Regression model selection for health score and RUL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RegressionTrainingResult:
    """Result returned by regression model selection."""

    best_name: str
    best_model: Any
    metrics: dict[str, dict[str, float | str]]


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _candidate_regressors(random_state: int) -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to train regressors.") from exc

    candidates: dict[str, Any] = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
    }

    try:
        from xgboost import XGBRegressor

        candidates["XGBoostRegressor"] = XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    return candidates


def train_regressors(x_train, x_test, y_train, y_test, random_state: int = 42) -> RegressionTrainingResult:
    """Train candidate regressors and select the lowest-RMSE model."""

    try:
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to evaluate regressors.") from exc

    metrics: dict[str, dict[str, float | str]] = {}
    best_name = ""
    best_model = None
    best_rmse = float("inf")

    for name, model in _candidate_regressors(random_state).items():
        try:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            rmse = _rmse(y_test, predictions)
            metrics[name] = {
                "rmse": round(float(rmse), 4),
                "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
                "r2": round(float(r2_score(y_test, predictions)), 4),
            }
            if rmse < best_rmse:
                best_name = name
                best_model = model
                best_rmse = rmse
        except Exception as exc:
            metrics[name] = {"error": str(exc)}

    if best_model is None:
        raise RuntimeError(f"No regressor could be trained. Candidate results: {metrics}")

    return RegressionTrainingResult(best_name=best_name, best_model=best_model, metrics=metrics)
