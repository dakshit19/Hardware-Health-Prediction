"""Hybrid rule engine for root-cause analysis."""

from __future__ import annotations

import pandas as pd


class RootCauseAnalyzer:
    """Infer root causes from sensor values, prediction, and explanation context."""

    def analyze(
        self,
        frame: pd.DataFrame,
        prediction: str,
        top_features: list[str],
        anomaly_score: float,
    ) -> tuple[str, list[str]]:
        """Return primary and secondary root causes."""

        row = frame.iloc[0]
        temperature = float(row["Temperature"])
        voltage = float(row["Voltage"])
        ram = float(row["RAMUsage"])
        disk = float(row["DiskUsage"])
        fan = float(row["FanSpeed"])
        cpu = float(row["CPUUsage"])
        voltage_deviation = abs(voltage - 12.0)
        secondary: list[str] = []

        if temperature >= 85 and fan < 1800:
            primary = "Cooling System Failure"
            secondary.extend(["Fan underperformance", "Thermal saturation"])
        elif temperature >= 85 and voltage_deviation >= 2.5:
            primary = "Voltage Regulator Instability"
            secondary.extend(["Power fluctuation", "VRM overheating"])
        elif prediction == "Power Issue" or voltage_deviation >= 3.0:
            primary = "Voltage Regulator Instability"
            secondary.extend(["Adapter instability", "Power rail deviation"])
        elif prediction == "Memory Leak" or (ram >= 85 and temperature < 80):
            primary = "Memory Pressure Escalation"
            secondary.extend(["Runaway process memory", "Insufficient memory reclamation"])
        elif prediction == "Disk Failure" or disk >= 90:
            primary = "Storage Subsystem Degradation"
            secondary.extend(["High disk utilization", "Potential controller or drive wear"])
        elif prediction == "Overheating" or temperature >= 80:
            primary = "Thermal Management Degradation"
            secondary.extend(["Dust accumulation", "Thermal paste aging"])
        elif cpu >= 90 and ram >= 80:
            primary = "Sustained Resource Saturation"
            secondary.extend(["High CPU load", "Memory pressure"])
        elif anomaly_score >= 0.60:
            primary = "Intermittent Sensor Anomaly"
            secondary.extend(["Telemetry drift", "Emerging hardware instability"])
        else:
            primary = "No Critical Root Cause Detected"
            secondary.extend(["System operating within expected range"])

        for feature in top_features:
            if feature not in secondary:
                if feature == "Voltage":
                    secondary.append("Voltage fluctuation")
                elif feature == "Temperature":
                    secondary.append("Elevated thermal stress")
                elif feature == "Fan Speed":
                    secondary.append("Cooling response degradation")
                elif feature == "Disk Usage":
                    secondary.append("Storage load pressure")
                elif feature == "RAM Usage":
                    secondary.append("Memory load pressure")

        return primary, _dedupe(secondary)[:5]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
