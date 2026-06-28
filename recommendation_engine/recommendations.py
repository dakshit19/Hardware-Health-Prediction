"""Context-aware recommendation engine."""

from __future__ import annotations

import pandas as pd


class RecommendationEngine:
    """Generate prioritized operational guidance from combined AI signals."""

    def generate(
        self,
        frame: pd.DataFrame,
        prediction: str,
        top_features: list[str],
        primary_root_cause: str,
        health_score: float,
        failure_probability: float,
        anomaly_score: float,
        risk_level: str,
    ) -> dict[str, list[str]]:
        """Return immediate, preventive, and monitoring recommendations."""

        row = frame.iloc[0]
        immediate: list[str] = []
        preventive: list[str] = []
        monitoring: list[str] = []

        if risk_level in {"Critical", "High"}:
            immediate.append("Run full motherboard diagnostics")
            immediate.append("Back up critical data before further stress testing")

        if "Voltage" in top_features or "Voltage" in primary_root_cause or prediction == "Power Issue":
            immediate.append("Inspect voltage regulator and power rails")
            immediate.append("Test with a known-good power adapter")
            preventive.append("Replace unstable adapter or charging circuit components")
            monitoring.append("Track voltage stability every 10 minutes")

        if "Temperature" in top_features or prediction == "Overheating" or row["Temperature"] >= 80:
            immediate.append("Inspect fan operation and airflow path")
            preventive.append("Clean cooling system and renew thermal interface material")
            monitoring.append("Track CPU temperature under idle and load conditions")

        if "RAM Usage" in top_features or prediction == "Memory Leak":
            immediate.append("Identify runaway processes and memory allocation spikes")
            preventive.append("Schedule memory diagnostics and firmware updates")
            monitoring.append("Monitor RAM usage trend and paging activity")

        if "Disk Usage" in top_features or prediction == "Disk Failure":
            immediate.append("Run SMART and storage controller diagnostics")
            preventive.append("Plan storage replacement if error counts increase")
            monitoring.append("Monitor disk utilization and reallocated-sector indicators")

        if "Fan Speed" in top_features or row["FanSpeed"] < 1800:
            immediate.append("Verify fan tachometer response")
            preventive.append("Replace fan assembly if RPM remains below safe range")
            monitoring.append("Alert when fan speed drops during thermal load")

        if health_score < 50:
            immediate.append("Reduce workload until health score recovers")
            preventive.append("Schedule board-level inspection")

        if failure_probability >= 70:
            immediate.append("Prepare service ticket for likely near-term failure")
            monitoring.append("Increase telemetry sampling frequency")

        if anomaly_score >= 0.60:
            immediate.append("Validate sensor readings with hardware diagnostics")
            monitoring.append("Compare anomaly score against the next maintenance window")

        if not immediate:
            immediate.append("Continue normal operation")
        if not preventive:
            preventive.append("Keep BIOS, drivers, and diagnostics tooling up to date")
        if not monitoring:
            monitoring.append("Continue routine health monitoring")

        return {
            "immediate": _dedupe(immediate)[:6],
            "preventive": _dedupe(preventive)[:6],
            "monitoring": _dedupe(monitoring)[:6],
        }


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
