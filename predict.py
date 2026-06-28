"""Inference pipeline for AI-generated motherboard health reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG, ProjectConfig
from explainability.local_importance import top_contributing_features
from explainability.shap_explainer import shap_top_features
from recommendation_engine.recommendations import RecommendationEngine
from recommendation_engine.root_cause import RootCauseAnalyzer
from utils.io import load_joblib, load_json
from utils.risk import calculate_risk_index, classify_risk_level, clamp
from utils.validation import coerce_to_dataframe


class MotherboardHealthPredictor:
    """Load trained artifacts and generate a structured health report."""

    def __init__(self, model_dir: Path | None = None, config: ProjectConfig = CONFIG) -> None:
        self.config = config
        self.model_dir = model_dir or config.models_dir
        self.metadata = self._load_metadata()
        self.preprocessor = self._load_artifact("preprocessor.pkl")
        self.label_encoder = self._load_artifact("encoder.pkl")
        self.feature_engineer = self._load_artifact("feature_engineer.pkl")
        self.problem_classifier = self._load_artifact("problem_classifier.pkl")
        self.health_regressor = self._load_artifact("health_regressor.pkl")
        self.failure_classifier = self._load_artifact("failure_probability.pkl")
        self.rul_regressor = self._load_artifact("rul_regressor.pkl")
        self.anomaly_detector = self._load_artifact("isolation_forest.pkl")
        self.trend_forecaster = self._load_artifact("trend_forecaster.pkl", required=False)
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.recommendation_engine = RecommendationEngine()

    def predict(self, payload: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Return the complete AI-powered motherboard health report."""

        raw_frame = coerce_to_dataframe(payload, self.config)
        engineered_frame = self.feature_engineer.transform(raw_frame)
        model_frame = engineered_frame.loc[:, self.metadata["model_feature_columns"]]
        x_transformed = self.preprocessor.transform(model_frame)

        prediction, confidence = self._predict_problem(x_transformed)
        health_score = round(clamp(float(self.health_regressor.predict(x_transformed)[0])), 2)
        failure_probability = round(clamp(self._predict_failure_probability(x_transformed)), 2)
        rul_days = max(1, int(round(float(self.rul_regressor.predict(x_transformed)[0]))))

        anomaly_score = round(float(self.anomaly_detector.score(x_transformed)[0]), 4)
        anomaly_label = self.anomaly_detector.label(x_transformed)[0]
        risk_index = calculate_risk_index(failure_probability, anomaly_score, health_score)
        risk_level = classify_risk_level(risk_index)

        top_features = shap_top_features(
            self.problem_classifier,
            x_transformed,
            self.metadata["transformed_feature_names"],
            top_n=3,
        )
        if top_features is None:
            top_features = top_contributing_features(
                raw_frame,
                self.problem_classifier,
                self.metadata["transformed_feature_names"],
                top_n=3,
            )

        primary_root_cause, secondary_root_causes = self.root_cause_analyzer.analyze(
            raw_frame,
            prediction,
            top_features,
            anomaly_score,
        )
        recommendations = self.recommendation_engine.generate(
            raw_frame,
            prediction,
            top_features,
            primary_root_cause,
            health_score,
            failure_probability,
            anomaly_score,
            risk_level,
        )

        report: dict[str, Any] = {
            "prediction": prediction,
            "confidence": confidence,
            "health_score": health_score,
            "anomaly_label": anomaly_label,
            "anomaly_score": anomaly_score,
            "failure_probability": failure_probability,
            "remaining_useful_life": f"{rul_days} Days",
            "risk_index": risk_index,
            "risk_level": risk_level,
            "primary_root_cause": primary_root_cause,
            "secondary_root_causes": secondary_root_causes,
            "top_contributing_features": top_features,
            "recommendations": recommendations,
        }

        if self.trend_forecaster is not None:
            report["trend_forecast"] = self.trend_forecaster.forecast(raw_frame, engineered_frame)

        return report

    def _predict_problem(self, x_transformed) -> tuple[str, float]:
        if hasattr(self.problem_classifier, "predict_proba"):
            probabilities = self.problem_classifier.predict_proba(x_transformed)[0]
            best_probability_index = int(np.argmax(probabilities))
            model_classes = getattr(self.problem_classifier, "classes_", np.arange(len(probabilities)))
            encoded_class = int(model_classes[best_probability_index])
            prediction = str(self.label_encoder.inverse_transform([encoded_class])[0])
            confidence = round(float(probabilities[best_probability_index] * 100.0), 2)
            return prediction, confidence

        encoded_prediction = int(self.problem_classifier.predict(x_transformed)[0])
        prediction = str(self.label_encoder.inverse_transform([encoded_prediction])[0])
        return prediction, 0.0

    def _predict_failure_probability(self, x_transformed) -> float:
        if hasattr(self.failure_classifier, "predict_proba"):
            probabilities = self.failure_classifier.predict_proba(x_transformed)[0]
            classes = list(getattr(self.failure_classifier, "classes_", [0, 1]))
            positive_index = classes.index(1) if 1 in classes else -1
            return float(probabilities[positive_index] * 100.0)
        return float(self.failure_classifier.predict(x_transformed)[0] * 100.0)

    def _load_artifact(self, filename: str, required: bool = True):
        path = self.model_dir / filename
        if not path.exists() and not required:
            return None
        try:
            return load_joblib(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing artifact `{filename}` in {self.model_dir}. Run `python train.py` first."
            ) from exc

    def _load_metadata(self) -> dict[str, Any]:
        try:
            return load_json(self.model_dir / "metadata.json")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Missing metadata.json in {self.model_dir}. Run `python train.py` first."
            ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a motherboard health report.")
    parser.add_argument("--model-dir", type=Path, default=CONFIG.models_dir, help="Directory containing model artifacts.")
    parser.add_argument(
        "--sample-json",
        type=str,
        default=None,
        help="JSON object with CPUUsage, RAMUsage, Temperature, Voltage, DiskUsage, FanSpeed, ModelName.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = json.loads(args.sample_json) if args.sample_json else _default_sample()
    predictor = MotherboardHealthPredictor(model_dir=args.model_dir)
    print(json.dumps(predictor.predict(sample), indent=2))


def _default_sample() -> dict[str, Any]:
    return {
        "ModelName": "Dell Inspiron 6880",
        "CPUUsage": 82.0,
        "RAMUsage": 76.0,
        "Temperature": 91.0,
        "Voltage": 10.2,
        "DiskUsage": 67.0,
        "FanSpeed": 1450,
    }


if __name__ == "__main__":
    main()
