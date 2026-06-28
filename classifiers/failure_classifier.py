"""Binary failure probability model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FailureClassifierResult:
    """Result returned by failure classifier training."""

    best_name: str
    best_model: Any
    metrics: dict[str, dict[str, float | str]]


def _candidate_failure_models(random_state: int) -> dict[str, Any]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to train failure probability models.") from exc

    candidates: dict[str, Any] = {
        "RandomForestFailure": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoostingFailure": GradientBoostingClassifier(random_state=random_state),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoostFailure"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        pass

    return candidates


def train_failure_classifier(x_train, x_test, y_train, y_test, random_state: int = 42) -> FailureClassifierResult:
    """Train binary classifiers and select the best weighted-F1 model."""

    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to evaluate failure classifiers.") from exc

    metrics: dict[str, dict[str, float | str]] = {}
    best_name = ""
    best_model = None
    best_score = -1.0

    for name, model in _candidate_failure_models(random_state).items():
        try:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
            roc_auc = None
            if hasattr(model, "predict_proba"):
                try:
                    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
                except ValueError:
                    roc_auc = None
            metrics[name] = {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "precision_weighted": round(float(precision), 4),
                "recall_weighted": round(float(recall), 4),
                "f1_weighted": round(float(f1), 4),
                "roc_auc": None if roc_auc is None else round(float(roc_auc), 4),
            }
            if f1 > best_score:
                best_name = name
                best_model = model
                best_score = float(f1)
        except Exception as exc:
            metrics[name] = {"error": str(exc)}

    if best_model is None:
        raise RuntimeError(f"No failure classifier could be trained. Candidate results: {metrics}")

    return FailureClassifierResult(best_name=best_name, best_model=best_model, metrics=metrics)
