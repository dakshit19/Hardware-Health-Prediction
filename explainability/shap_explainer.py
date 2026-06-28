"""SHAP integration with safe fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from explainability.local_importance import readable_feature_name


def generate_shap_summary_plot(model: Any, x_sample: Any, feature_names: list[str], output_path: Path) -> bool:
    """Generate a SHAP summary plot when shap and matplotlib are installed."""

    catboost_result = _generate_catboost_shap_summary(model, x_sample, feature_names, output_path)
    if catboost_result:
        return True

    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import shap
    except ImportError:
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        display_names = [readable_feature_name(name) for name in feature_names]
        x_display = pd.DataFrame(x_sample, columns=display_names)
        explainer = shap.Explainer(model, x_sample)
        shap_values = explainer(x_sample)
        shap.summary_plot(shap_values, x_display, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()
        return True
    except BaseException:
        try:
            plt.close()
        except BaseException:
            pass
        return False


def shap_top_features(model: Any, x_row: Any, feature_names: list[str], top_n: int = 3) -> list[str] | None:
    """Return SHAP top features for a single row, or None if unavailable."""

    catboost_top = _catboost_shap_top_features(model, x_row, feature_names, top_n)
    if catboost_top:
        return catboost_top

    try:
        import shap
    except ImportError:
        return None

    try:
        explainer = shap.Explainer(model, x_row)
        values = explainer(x_row).values
        values_array = np.asarray(values)
        if values_array.ndim == 3:
            values_array = np.max(np.abs(values_array[0]), axis=1)
        elif values_array.ndim == 2:
            values_array = np.abs(values_array[0])
        else:
            values_array = np.abs(values_array)

        order = np.argsort(values_array)[::-1][:top_n]
        return [readable_feature_name(feature_names[index]) for index in order]
    except BaseException:
        return None


def _is_catboost_model(model: Any) -> bool:
    return model.__class__.__module__.startswith("catboost")


def _catboost_shap_values(model: Any, x_values: Any, feature_names: list[str]) -> np.ndarray | None:
    if not _is_catboost_model(model):
        return None
    try:
        from catboost import Pool

        pool = Pool(x_values, feature_names=feature_names)
        raw_values = np.asarray(model.get_feature_importance(data=pool, type="ShapValues"))
        if raw_values.ndim == 3:
            return raw_values[:, :, :-1]
        if raw_values.ndim == 2:
            return raw_values[:, :-1]
        return None
    except BaseException:
        return None


def _generate_catboost_shap_summary(model: Any, x_sample: Any, feature_names: list[str], output_path: Path) -> bool:
    shap_values = _catboost_shap_values(model, x_sample, feature_names)
    if shap_values is None:
        return False

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return False

    if shap_values.ndim == 3:
        importances = np.mean(np.abs(shap_values), axis=(0, 1))
    else:
        importances = np.mean(np.abs(shap_values), axis=0)

    order = np.argsort(importances)[-20:]
    names = [readable_feature_name(feature_names[index]) for index in order]
    scores = importances[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    sns.barplot(x=scores, y=names, orient="h")
    plt.title("SHAP Summary - Mean Absolute Impact")
    plt.xlabel("Mean absolute SHAP value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def _catboost_shap_top_features(model: Any, x_row: Any, feature_names: list[str], top_n: int) -> list[str] | None:
    shap_values = _catboost_shap_values(model, x_row, feature_names)
    if shap_values is None:
        return None

    if shap_values.ndim == 3:
        row_values = np.max(np.abs(shap_values[0]), axis=0)
    else:
        row_values = np.abs(shap_values[0])

    order = np.argsort(row_values)[::-1][:top_n]
    return [readable_feature_name(feature_names[index]) for index in order]
