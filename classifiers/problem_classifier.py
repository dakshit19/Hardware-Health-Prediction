"""Multiclass problem detection model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClassifierTrainingResult:
    """Result returned by classifier model selection."""

    best_name: str
    best_model: Any
    metrics: dict[str, dict[str, float | str]]
    confusion_matrix: np.ndarray
    test_probabilities: np.ndarray | None


def _candidate_classifiers(random_state: int, n_classes: int) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to train classifiers.") from exc

    candidates: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        candidates["CatBoost"] = CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=random_state,
            allow_writing_files=False,
            verbose=False,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=350,
            learning_rate=0.05,
            objective="multiclass",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    return candidates


def train_problem_classifiers(x_train, x_test, y_train, y_test, random_state: int = 42) -> ClassifierTrainingResult:
    """Train candidate classifiers and return the best weighted-F1 model."""

    try:
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to evaluate classifiers.") from exc

    n_classes = len(np.unique(y_train))
    metrics: dict[str, dict[str, float | str]] = {}
    best_name = ""
    best_model = None
    best_score = -1.0
    best_cm = None
    best_probabilities = None

    for name, model in _candidate_classifiers(random_state, n_classes).items():
        try:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
            probabilities = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
            roc_auc = np.nan
            if probabilities is not None:
                try:
                    roc_auc = roc_auc_score(y_test, probabilities, multi_class="ovr", average="weighted")
                except ValueError:
                    roc_auc = np.nan

            accuracy = accuracy_score(y_test, predictions)
            metrics[name] = {
                "accuracy": round(float(accuracy), 4),
                "precision_weighted": round(float(precision), 4),
                "recall_weighted": round(float(recall), 4),
                "f1_weighted": round(float(f1), 4),
                "roc_auc_ovr_weighted": None if np.isnan(roc_auc) else round(float(roc_auc), 4),
            }

            if f1 > best_score:
                best_name = name
                best_model = model
                best_score = float(f1)
                best_cm = confusion_matrix(y_test, predictions)
                best_probabilities = probabilities
        except Exception as exc:
            metrics[name] = {"error": str(exc)}

    if best_model is None or best_cm is None:
        raise RuntimeError(f"No classifier could be trained. Candidate results: {metrics}")

    return ClassifierTrainingResult(
        best_name=best_name,
        best_model=best_model,
        metrics=metrics,
        confusion_matrix=best_cm,
        test_probabilities=best_probabilities,
    )
