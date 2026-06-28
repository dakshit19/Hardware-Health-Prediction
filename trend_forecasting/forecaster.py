"""Linear Regression baseline forecaster for future sensor values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrendForecastArtifact:
    """Serializable trend forecasting artifact."""

    models: dict[str, Any]
    feature_columns: list[str]

    def forecast(self, raw_frame: pd.DataFrame, engineered_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Forecast current, next-hour, and next-day values."""

        x = engineered_frame.loc[:, self.feature_columns]
        forecast: dict[str, dict[str, float]] = {}
        for sensor in ("Temperature", "Voltage", "CPUUsage"):
            forecast[sensor] = {"current": round(float(raw_frame.iloc[0][sensor]), 2)}
            for horizon in ("next_hour", "next_day"):
                key = f"{sensor}_{horizon}"
                prediction = self.models[key].predict(x)[0]
                forecast[sensor][horizon] = round(float(prediction), 2)
        forecast["early_warning"] = _early_warning(forecast)
        return forecast


def train_trend_forecaster(raw_frame: pd.DataFrame, engineered_frame: pd.DataFrame, feature_columns: list[str]) -> TrendForecastArtifact:
    """Train Linear Regression models on deterministic synthetic future states."""

    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to train trend forecasters.") from exc

    targets = _build_forecast_targets(raw_frame)
    x = engineered_frame.loc[:, feature_columns]
    models: dict[str, Any] = {}
    for column in targets.columns:
        model = LinearRegression()
        model.fit(x, targets[column])
        models[column] = model
    return TrendForecastArtifact(models=models, feature_columns=feature_columns)


def _build_forecast_targets(frame: pd.DataFrame) -> pd.DataFrame:
    voltage_deviation = frame["Voltage"] - 12.0
    fan_factor = (2500.0 - frame["FanSpeed"]) / 2500.0
    thermal_drift = 0.04 * (frame["CPUUsage"] - 50.0) + 0.03 * (frame["Temperature"] - 60.0) + 1.5 * fan_factor
    cpu_drift = 0.03 * (frame["RAMUsage"] - 50.0) + 0.02 * (frame["DiskUsage"] - 50.0)
    voltage_drift = 0.06 * voltage_deviation + 0.005 * (frame["Temperature"] - 65.0)

    targets = pd.DataFrame(index=frame.index)
    targets["Temperature_next_hour"] = (frame["Temperature"] + thermal_drift).clip(20, 110)
    targets["Temperature_next_day"] = (frame["Temperature"] + 6.0 * thermal_drift).clip(20, 115)
    targets["Voltage_next_hour"] = (frame["Voltage"] + voltage_drift).clip(8, 18)
    targets["Voltage_next_day"] = (frame["Voltage"] + 5.0 * voltage_drift).clip(7, 19)
    targets["CPUUsage_next_hour"] = (frame["CPUUsage"] + cpu_drift).clip(0, 100)
    targets["CPUUsage_next_day"] = (frame["CPUUsage"] + 4.0 * cpu_drift).clip(0, 100)
    return targets


def _early_warning(forecast: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    warnings: list[str] = []
    if forecast["Temperature"]["next_hour"] >= 85 or forecast["Temperature"]["next_day"] >= 90:
        warnings.append("Temperature forecast exceeds safe operating limit")
    if abs(forecast["Voltage"]["next_hour"] - 12.0) >= 3.0 or abs(forecast["Voltage"]["next_day"] - 12.0) >= 3.5:
        warnings.append("Voltage forecast indicates unsafe power drift")
    if forecast["CPUUsage"]["next_hour"] >= 90 or forecast["CPUUsage"]["next_day"] >= 95:
        warnings.append("CPU usage forecast indicates sustained saturation")
    return {"warnings": warnings, "status": "Warning" if warnings else "Normal"}
