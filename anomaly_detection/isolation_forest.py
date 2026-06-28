"""Isolation Forest anomaly detector with calibrated 0-1 scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IsolationForestArtifact:
    """Serializable anomaly detector artifact."""

    model: object
    raw_min: float
    raw_max: float
    threshold: float = 0.60

    def score(self, x) -> np.ndarray:
        """Return normalized anomaly scores, where 1 means most anomalous."""

        raw = -self.model.decision_function(x)
        denominator = max(self.raw_max - self.raw_min, 1e-9)
        return np.clip((raw - self.raw_min) / denominator, 0.0, 1.0)

    def label(self, x) -> list[str]:
        """Return Normal or Abnormal labels."""

        return ["Abnormal" if value >= self.threshold else "Normal" for value in self.score(x)]


def train_isolation_forest(x_train, random_state: int = 42, threshold: float = 0.60) -> IsolationForestArtifact:
    """Train an Isolation Forest model and calibrate score normalization."""

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to train Isolation Forest.") from exc

    model = IsolationForest(
        n_estimators=250,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train)
    raw_scores = -model.decision_function(x_train)
    return IsolationForestArtifact(
        model=model,
        raw_min=float(np.min(raw_scores)),
        raw_max=float(np.max(raw_scores)),
        threshold=threshold,
    )
