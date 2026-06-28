"""Train all Motherboard Health AI models and save production artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_detection.isolation_forest import train_isolation_forest
from classifiers.failure_classifier import train_failure_classifier
from classifiers.problem_classifier import train_problem_classifiers
from config import CONFIG, ProjectConfig
from explainability.shap_explainer import generate_shap_summary_plot
from feature_engineering.analysis import correlation_report
from feature_engineering.features import MotherboardFeatureEngineer
from feature_engineering.synthetic_targets import build_synthetic_targets
from preprocessing.preprocessor import MotherboardPreprocessor
from regressors.regression_trainer import train_regressors
from trend_forecasting.forecaster import train_trend_forecaster
from utils.io import load_dataset, save_joblib, save_json
from utils.logger import setup_logger
from utils.visualization import (
    save_confusion_matrix,
    save_correlation_heatmap,
    save_distribution,
    save_feature_importance,
    save_roc_curve,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Train Motherboard Health AI.")
    parser.add_argument("--data-path", type=Path, default=CONFIG.dataset_path, help="Path to the training CSV.")
    parser.add_argument("--model-dir", type=Path, default=CONFIG.models_dir, help="Directory for model artifacts.")
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=CONFIG.visualizations_dir,
        help="Directory for generated plots.",
    )
    parser.add_argument("--test-size", type=float, default=CONFIG.test_size, help="Test split fraction.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def main() -> None:
    """Execute the full training pipeline."""

    args = parse_args()
    config = replace(CONFIG, test_size=args.test_size)
    config.ensure_directories()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.visualization_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_file=args.model_dir / "training.log")
    logger.info("Loading dataset from %s", args.data_path)
    raw_frame = load_dataset(args.data_path)

    preprocessor_manager = MotherboardPreprocessor(config)
    raw_frame = preprocessor_manager.clean(raw_frame)
    logger.info("Dataset shape after duplicate removal: %s", raw_frame.shape)

    feature_engineer = MotherboardFeatureEngineer(config)
    engineered_features = feature_engineer.fit_transform(raw_frame.loc[:, list(config.raw_feature_columns)])
    training_frame = engineered_features.copy()
    training_frame[config.target_column] = raw_frame[config.target_column].values
    training_frame = build_synthetic_targets(training_frame, config)

    model_numeric_columns = list(config.numeric_columns) + list(feature_engineer.engineered_columns)
    model_categorical_columns = list(config.categorical_columns)
    model_feature_columns = model_numeric_columns + model_categorical_columns

    logger.info("Splitting dataset with %s model features.", len(model_feature_columns))
    split = preprocessor_manager.train_test_split(training_frame, model_feature_columns)
    sklearn_preprocessor = preprocessor_manager.build_preprocessor(model_numeric_columns, model_categorical_columns)
    x_train = sklearn_preprocessor.fit_transform(split.x_train)
    x_test = sklearn_preprocessor.transform(split.x_test)
    feature_names = list(sklearn_preprocessor.get_feature_names_out())

    y_train_problem, y_test_problem, label_encoder = preprocessor_manager.encode_target(
        split.y_train_problem,
        split.y_test_problem,
    )

    logger.info("Training multiclass problem classifiers.")
    problem_result = train_problem_classifiers(
        x_train,
        x_test,
        y_train_problem,
        y_test_problem,
        random_state=config.random_state,
    )

    logger.info("Training health score regressors.")
    health_result = train_regressors(
        x_train,
        x_test,
        split.y_train_health,
        split.y_test_health,
        random_state=config.random_state,
    )

    logger.info("Training failure probability classifiers.")
    failure_result = train_failure_classifier(
        x_train,
        x_test,
        split.y_train_failure,
        split.y_test_failure,
        random_state=config.random_state,
    )

    logger.info("Training Remaining Useful Life regressors.")
    rul_result = train_regressors(
        x_train,
        x_test,
        split.y_train_rul,
        split.y_test_rul,
        random_state=config.random_state,
    )

    logger.info("Training Isolation Forest anomaly detector.")
    anomaly_detector = train_isolation_forest(
        x_train,
        random_state=config.random_state,
        threshold=config.anomaly_threshold,
    )
    anomaly_scores = anomaly_detector.score(x_test)

    logger.info("Training trend forecaster.")
    trend_forecaster = train_trend_forecaster(
        raw_frame=raw_frame.loc[split.x_train.index, list(config.raw_feature_columns)],
        engineered_frame=training_frame.loc[split.x_train.index],
        feature_columns=model_numeric_columns,
    )

    logger.info("Saving model artifacts.")
    save_joblib(sklearn_preprocessor, args.model_dir / "preprocessor.pkl")
    save_joblib(label_encoder, args.model_dir / "encoder.pkl")
    save_joblib(problem_result.best_model, args.model_dir / "problem_classifier.pkl")
    save_joblib(health_result.best_model, args.model_dir / "health_regressor.pkl")
    save_joblib(failure_result.best_model, args.model_dir / "failure_probability.pkl")
    save_joblib(rul_result.best_model, args.model_dir / "rul_regressor.pkl")
    save_joblib(anomaly_detector, args.model_dir / "isolation_forest.pkl")
    save_joblib(trend_forecaster, args.model_dir / "trend_forecaster.pkl")

    numeric_pipeline = sklearn_preprocessor.named_transformers_["num"]
    categorical_pipeline = sklearn_preprocessor.named_transformers_["cat"]
    save_joblib(numeric_pipeline.named_steps["scaler"], args.model_dir / "scaler.pkl")
    save_joblib(categorical_pipeline.named_steps["onehot"], args.model_dir / "onehot_encoder.pkl")

    correlations = correlation_report(training_frame)
    if not correlations.empty:
        correlations.to_csv(args.model_dir / "high_correlation_report.csv", index=False)

    visualization_results: dict[str, bool] = {}
    if not args.skip_plots:
        logger.info("Generating visualizations where plotting dependencies are available.")
        visualization_results = _generate_visualizations(
            training_frame=training_frame,
            problem_result=problem_result,
            label_classes=list(label_encoder.classes_),
            feature_names=feature_names,
            x_train=x_train,
            x_test=x_test,
            y_test_problem=y_test_problem,
            anomaly_scores=anomaly_scores,
            health_scores=training_frame[config.health_target_column].to_numpy(),
            output_dir=args.visualization_dir,
        )

    metadata: dict[str, Any] = {
        "project": "MotherboardHealthAI",
        "random_state": config.random_state,
        "dataset_rows": int(len(raw_frame)),
        "raw_feature_columns": list(config.raw_feature_columns),
        "model_feature_columns": model_feature_columns,
        "model_numeric_columns": model_numeric_columns,
        "model_categorical_columns": model_categorical_columns,
        "transformed_feature_names": feature_names,
        "problem_classes": list(label_encoder.classes_),
        "best_models": {
            "problem_classifier": problem_result.best_name,
            "health_regressor": health_result.best_name,
            "failure_probability": failure_result.best_name,
            "rul_regressor": rul_result.best_name,
            "anomaly_detector": "IsolationForest",
            "trend_forecaster": "LinearRegression",
        },
        "metrics": {
            "problem_classifier": problem_result.metrics,
            "health_regressor": health_result.metrics,
            "failure_probability": failure_result.metrics,
            "rul_regressor": rul_result.metrics,
        },
        "visualizations": visualization_results,
    }
    save_json(metadata, args.model_dir / "metadata.json")
    logger.info("Training complete. Artifacts saved to %s", args.model_dir)


def _generate_visualizations(
    training_frame,
    problem_result,
    label_classes: list[str],
    feature_names: list[str],
    x_train,
    x_test,
    y_test_problem,
    anomaly_scores: np.ndarray,
    health_scores: np.ndarray,
    output_dir: Path,
) -> dict[str, bool]:
    """Create requested model and data visualizations."""

    importances = getattr(problem_result.best_model, "feature_importances_", None)
    confidence = None
    if problem_result.test_probabilities is not None:
        confidence = np.max(problem_result.test_probabilities, axis=1) * 100.0

    results = {
        "correlation_heatmap": save_correlation_heatmap(training_frame, output_dir / "correlation_heatmap.png"),
        "confusion_matrix": save_confusion_matrix(
            problem_result.confusion_matrix,
            label_classes,
            output_dir / "confusion_matrix.png",
        ),
        "roc_curve": save_roc_curve(problem_result.best_model, x_test, y_test_problem, output_dir / "roc_curve.png"),
        "health_score_distribution": save_distribution(
            health_scores,
            "Health Score Distribution",
            "Health Score",
            output_dir / "health_score_distribution.png",
        ),
        "anomaly_distribution": save_distribution(
            anomaly_scores,
            "Anomaly Score Distribution",
            "Anomaly Score",
            output_dir / "anomaly_distribution.png",
        ),
        "prediction_confidence_histogram": False
        if confidence is None
        else save_distribution(
            confidence,
            "Prediction Confidence Histogram",
            "Confidence (%)",
            output_dir / "prediction_confidence_histogram.png",
        ),
        "feature_importance": False
        if importances is None
        else save_feature_importance(
            feature_names,
            np.asarray(importances),
            output_dir / "feature_importance.png",
            title="Problem Classifier Feature Importance",
        ),
        "shap_summary": generate_shap_summary_plot(
            problem_result.best_model,
            x_train[: min(300, x_train.shape[0])],
            feature_names,
            output_dir / "shap_summary.png",
        ),
    }
    return results


if __name__ == "__main__":
    main()
