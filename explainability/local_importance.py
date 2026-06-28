"""Local explanation fallbacks for report generation."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd


def readable_feature_name(feature_name: str) -> str:
    """Convert transformed sklearn feature names into report-friendly labels."""

    clean = re.sub(r"^(num|cat)__", "", feature_name)
    clean = re.sub(r"^ModelName_", "Model Name: ", clean)
    aliases = {
        "CPUUsage": "CPU Usage",
        "RAMUsage": "RAM Usage",
        "DiskUsage": "Disk Usage",
        "FanSpeed": "Fan Speed",
        "PowerStability": "Voltage Stability",
        "VoltageDeviation": "Voltage",
        "CoolingEfficiency": "Cooling Efficiency",
        "MemoryPressure": "Memory Pressure",
        "ThermalStress": "Thermal Stress",
        "DiskLoadFactor": "Disk Load Factor",
        "ResourcePressure": "Resource Pressure",
        "FanStress": "Fan Stress",
        "VoltageThermalInteraction": "Voltage Thermal Interaction",
    }
    return aliases.get(clean, clean)


def collapse_encoded_importances(feature_names: list[str], importances: np.ndarray) -> dict[str, float]:
    """Collapse one-hot encoded and numeric importances back to readable feature names."""

    grouped: defaultdict[str, float] = defaultdict(float)
    for feature, importance in zip(feature_names, importances):
        grouped[readable_feature_name(feature)] += float(abs(importance))
    return dict(grouped)


def sensor_severity_scores(frame: pd.DataFrame) -> dict[str, float]:
    """Score raw sensor readings by operational severity for local explanations."""

    row = frame.iloc[0]
    scores = {
        "Temperature": max(0.0, min(1.0, (float(row["Temperature"]) - 45.0) / 50.0)),
        "Voltage": max(0.0, min(1.0, abs(float(row["Voltage"]) - 12.0) / 6.0)),
        "CPU Usage": max(0.0, min(1.0, float(row["CPUUsage"]) / 100.0)),
        "RAM Usage": max(0.0, min(1.0, float(row["RAMUsage"]) / 100.0)),
        "Disk Usage": max(0.0, min(1.0, float(row["DiskUsage"]) / 100.0)),
        "Fan Speed": max(0.0, min(1.0, (3000.0 - float(row["FanSpeed"])) / 2500.0)),
    }
    return scores


def top_contributing_features(
    raw_frame: pd.DataFrame,
    model,
    feature_names: list[str],
    top_n: int = 3,
) -> list[str]:
    """Return local top features using model importances plus sensor severity."""

    severity = sensor_severity_scores(raw_frame)
    model_scores: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        model_scores = collapse_encoded_importances(feature_names, np.asarray(model.feature_importances_))

    combined: dict[str, float] = {}
    for feature, score in severity.items():
        combined[feature] = 0.65 * score + 0.35 * model_scores.get(feature, 0.0)

    for feature, score in model_scores.items():
        combined.setdefault(feature, 0.35 * score)

    ordered = sorted(combined.items(), key=lambda item: item[1], reverse=True)
    return [feature for feature, _ in ordered[:top_n]]
