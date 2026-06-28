"""Central configuration for the Motherboard Health AI project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime and training configuration.

    Keep paths relative to the project root so the package can be moved without
    editing source code.
    """

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_file: str = "Laptop_Motherboard_Health_Monitoring_Dataset.csv"
    random_state: int = 42
    test_size: float = 0.2
    ideal_voltage: float = 12.0
    anomaly_threshold: float = 0.60

    target_column: str = "ProblemDetected"
    health_target_column: str = "HealthScore"
    failure_target_column: str = "WillFailSoon"
    rul_target_column: str = "RemainingUsefulLifeDays"

    raw_feature_columns: tuple[str, ...] = (
        "CPUUsage",
        "RAMUsage",
        "Temperature",
        "Voltage",
        "DiskUsage",
        "FanSpeed",
        "ModelName",
    )
    numeric_columns: tuple[str, ...] = (
        "CPUUsage",
        "RAMUsage",
        "Temperature",
        "Voltage",
        "DiskUsage",
        "FanSpeed",
    )
    categorical_columns: tuple[str, ...] = ("ModelName",)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def visualizations_dir(self) -> Path:
        return self.project_root / "visualizations"

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.data_file

    def ensure_directories(self) -> None:
        """Create required runtime directories."""

        for path in (self.data_dir, self.models_dir, self.visualizations_dir):
            path.mkdir(parents=True, exist_ok=True)


CONFIG = ProjectConfig()
