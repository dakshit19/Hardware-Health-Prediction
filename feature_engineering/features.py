"""Feature engineering for motherboard sensor telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, ProjectConfig


class MotherboardFeatureEngineer:
    """Generate physically meaningful derived features from sensor readings."""

    engineered_columns: tuple[str, ...] = (
        "CoolingEfficiency",
        "PowerStability",
        "VoltageDeviation",
        "MemoryPressure",
        "ThermalStress",
        "DiskLoadFactor",
        "ResourcePressure",
        "FanStress",
        "VoltageThermalInteraction",
    )

    def __init__(self, config: ProjectConfig = CONFIG) -> None:
        self.config = config

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "MotherboardFeatureEngineer":
        """Fit-compatible no-op for sklearn Pipeline support."""

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features and sanitize invalid values."""

        frame = x.copy()
        fan_speed = frame["FanSpeed"].clip(lower=1)
        voltage_deviation = (frame["Voltage"] - self.config.ideal_voltage).abs()

        frame["CoolingEfficiency"] = frame["Temperature"] / fan_speed
        frame["PowerStability"] = voltage_deviation
        frame["VoltageDeviation"] = voltage_deviation
        frame["MemoryPressure"] = (frame["CPUUsage"] * frame["RAMUsage"]) / 100.0
        frame["ThermalStress"] = (frame["Temperature"] * frame["CPUUsage"]) / 100.0
        frame["DiskLoadFactor"] = (frame["DiskUsage"] * frame["CPUUsage"]) / 100.0
        frame["ResourcePressure"] = (frame["CPUUsage"] + frame["RAMUsage"]) / 2.0
        frame["FanStress"] = frame["Temperature"] / (fan_speed / 1000.0)
        frame["VoltageThermalInteraction"] = voltage_deviation * frame["Temperature"]

        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        for column in self.engineered_columns:
            frame[column] = frame[column].fillna(frame[column].median())

        return frame

    def fit_transform(self, x: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform in one call."""

        return self.fit(x, y).transform(x)
