"""Synthetic health, failure, and RUL targets derived from degradation physics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG, ProjectConfig


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def calculate_degradation_score(frame: pd.DataFrame, config: ProjectConfig = CONFIG) -> pd.Series:
    """Calculate a 0-1 degradation score from sensor behavior."""

    temp_severity = _clip01((frame["Temperature"] - 45.0) / 50.0)
    voltage_severity = _clip01((frame["Voltage"] - config.ideal_voltage).abs() / 6.0)
    cpu_severity = _clip01(frame["CPUUsage"] / 100.0)
    ram_severity = _clip01(frame["RAMUsage"] / 100.0)
    disk_severity = _clip01(frame["DiskUsage"] / 100.0)
    cooling_severity = _clip01((frame["Temperature"] / frame["FanSpeed"].clip(lower=1) - 0.018) / 0.045)

    degradation = (
        0.26 * temp_severity
        + 0.22 * voltage_severity
        + 0.16 * cpu_severity
        + 0.14 * ram_severity
        + 0.10 * disk_severity
        + 0.12 * cooling_severity
    )
    return degradation.clip(lower=0.0, upper=1.0)


def build_synthetic_targets(frame: pd.DataFrame, config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Add deterministic targets for health score, failure probability, and RUL."""

    enriched = frame.copy()
    degradation = calculate_degradation_score(enriched, config)

    problem_penalty = enriched[config.target_column].map(
        {
            "No Problem": 0.00,
            "Memory Leak": 0.06,
            "Disk Failure": 0.10,
            "Power Issue": 0.12,
            "Overheating": 0.14,
        }
    ).fillna(0.08)

    adjusted_degradation = (degradation + problem_penalty).clip(lower=0.0, upper=1.0)
    health_score = (100.0 - 100.0 * adjusted_degradation).clip(lower=0.0, upper=100.0)

    failure_rule = (
        (enriched["Temperature"] >= 85.0).astype(int)
        + ((enriched["Voltage"] - config.ideal_voltage).abs() >= 3.5).astype(int)
        + (enriched["CPUUsage"] >= 85.0).astype(int)
        + (enriched["RAMUsage"] >= 88.0).astype(int)
        + (enriched["DiskUsage"] >= 90.0).astype(int)
        + (health_score <= 45.0).astype(int)
    )

    will_fail_soon = ((adjusted_degradation >= 0.62) | (failure_rule >= 3)).astype(int)
    rul_days = 5.0 + ((1.0 - adjusted_degradation) ** 2.2) * 275.0
    rul_days = np.where(will_fail_soon == 1, np.minimum(rul_days, 45.0), rul_days)

    enriched[config.health_target_column] = health_score.round(2)
    enriched[config.failure_target_column] = will_fail_soon.astype(int)
    enriched[config.rul_target_column] = np.clip(np.round(rul_days).astype(int), 1, 300)
    return enriched
