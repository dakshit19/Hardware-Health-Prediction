"""Risk index calculations."""

from __future__ import annotations


def clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    """Clamp a value into a numeric range."""

    return max(lower, min(upper, float(value)))


def calculate_risk_index(
    failure_probability: float,
    anomaly_score: float,
    health_score: float,
) -> float:
    """Calculate the enterprise risk index specified by the project brief."""

    risk = (
        0.35 * failure_probability
        + 0.30 * anomaly_score * 100.0
        + 0.35 * (100.0 - health_score)
    )
    return round(clamp(risk), 2)


def classify_risk_level(risk_index: float) -> str:
    """Map a numeric risk index to an operational severity band."""

    if risk_index >= 80:
        return "Critical"
    if risk_index >= 60:
        return "High"
    if risk_index >= 35:
        return "Medium"
    return "Low"
