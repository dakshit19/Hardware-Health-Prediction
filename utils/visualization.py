"""Visualization helpers used during training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_plotting_stack():
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return None, None
    return plt, sns


def save_correlation_heatmap(frame: pd.DataFrame, output_path: Path) -> bool:
    """Save a numeric correlation heatmap if plotting dependencies exist."""

    plt, sns = _load_plotting_stack()
    if plt is None or sns is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = frame.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0.0, linewidths=0.2)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> bool:
    """Save a confusion matrix plot."""

    plt, sns = _load_plotting_stack()
    if plt is None or sns is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Problem Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_distribution(values: np.ndarray | pd.Series, title: str, xlabel: str, output_path: Path) -> bool:
    """Save a histogram distribution."""

    plt, sns = _load_plotting_stack()
    if plt is None or sns is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.histplot(values, bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: Path,
    title: str = "Feature Importance",
    top_n: int = 20,
) -> bool:
    """Save a horizontal feature importance chart."""

    plt, sns = _load_plotting_stack()
    if plt is None or sns is None:
        return False

    if len(feature_names) != len(importances):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(importances)[-top_n:]
    names = [feature_names[index] for index in order]
    scores = importances[order]

    plt.figure(figsize=(9, 7))
    sns.barplot(x=scores, y=names, orient="h")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_roc_curve(model: Any, x_test: Any, y_test: Any, output_path: Path) -> bool:
    """Save a multiclass ROC curve when the model supports probabilities."""

    plt, _ = _load_plotting_stack()
    if plt is None or not hasattr(model, "predict_proba"):
        return False

    try:
        from sklearn.metrics import RocCurveDisplay
        from sklearn.preprocessing import label_binarize
    except ImportError:
        return False

    try:
        probabilities = model.predict_proba(x_test)
        classes = np.unique(y_test)
        y_binary = label_binarize(y_test, classes=classes)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for index, class_value in enumerate(classes):
            RocCurveDisplay.from_predictions(
                y_binary[:, index],
                probabilities[:, index],
                name=f"Class {class_value}",
                ax=plt.gca(),
            )
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return True
    except Exception:
        plt.close()
        return False
