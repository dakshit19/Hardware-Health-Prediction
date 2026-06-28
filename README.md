# MotherboardHealthAI

Production-style AI system for laptop motherboard health monitoring and predictive maintenance. The project trains separate models for problem detection, anomaly detection, health score prediction, failure probability, Remaining Useful Life, and short-horizon sensor forecasting, then combines those outputs with root-cause rules and a recommendation engine.

## Dataset

The project includes the provided CSV at:

```text
data/Laptop_Motherboard_Health_Monitoring_Dataset.csv
```

Expected input columns:

- `CPUUsage`
- `RAMUsage`
- `Temperature`
- `Voltage`
- `DiskUsage`
- `FanSpeed`
- `ModelName`

Target column:

- `ProblemDetected`

## Setup

Use Python 3.12 or newer. Python 3.12 is recommended because enterprise ML packages usually publish wheels there first.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`XGBoost`, `CatBoost`, `LightGBM`, and `SHAP` are integrated when installed. The training pipeline skips unavailable optional model families only when imports fail; `scikit-learn`, `pandas`, `numpy`, and `joblib` are required.

## Train

```bash
python train.py
```

Artifacts are saved in `models/`:

- `problem_classifier.pkl`
- `health_regressor.pkl`
- `failure_probability.pkl`
- `rul_regressor.pkl`
- `isolation_forest.pkl`
- `trend_forecaster.pkl`
- `preprocessor.pkl`
- `encoder.pkl`
- `scaler.pkl`
- `onehot_encoder.pkl`
- `feature_engineer.pkl`
- `metadata.json`

Plots are saved in `visualizations/` when plotting dependencies are available:

- Correlation heatmap
- Feature importance plot
- Confusion matrix
- ROC curve
- SHAP summary plot
- Health score distribution
- Anomaly score distribution
- Prediction confidence histogram

## Predict

After training:

```bash
python predict.py --sample-json "{\"ModelName\":\"Dell Inspiron 6880\",\"CPUUsage\":82,\"RAMUsage\":76,\"Temperature\":91,\"Voltage\":10.2,\"DiskUsage\":67,\"FanSpeed\":1450}"
```

Example response shape:

```json
{
  "prediction": "Power Issue",
  "confidence": 96.4,
  "health_score": 81.0,
  "anomaly_label": "Abnormal",
  "anomaly_score": 0.91,
  "failure_probability": 87.0,
  "remaining_useful_life": "5 Days",
  "risk_index": 91.0,
  "risk_level": "Critical",
  "primary_root_cause": "Voltage Regulator Instability",
  "secondary_root_causes": ["Power fluctuation", "VRM overheating"],
  "top_contributing_features": ["Temperature", "Voltage", "CPU Usage"],
  "recommendations": {
    "immediate": ["Inspect voltage regulator and power rails"],
    "preventive": ["Replace unstable adapter or charging circuit components"],
    "monitoring": ["Track voltage stability every 10 minutes"]
  }
}
```

## API

After training and installing FastAPI:

```bash
python app.py
```

Then call:

```text
POST http://localhost:8000/predict
```

## Architecture

- `preprocessing/`: duplicate removal, missing value handling, label encoding, one-hot encoding, scaling, train/test split, sklearn pipeline support.
- `feature_engineering/`: engineered degradation features, correlation report, synthetic health/failure/RUL targets.
- `anomaly_detection/`: Isolation Forest with calibrated anomaly scores from 0 to 1.
- `classifiers/`: multiclass problem detection and binary failure probability model selection.
- `regressors/`: health score and RUL regression model selection.
- `explainability/`: SHAP summary plotting and local fallback explanations.
- `recommendation_engine/`: root-cause inference and prioritized maintenance recommendations.
- `trend_forecasting/`: Linear Regression baseline forecasting for Temperature, Voltage, and CPU usage.
- `utils/`: logging, validation, persistence, risk scoring, and visualization helpers.

## Risk Index

The operational risk score uses the requested formula:

```text
RiskIndex = 0.35 * FailureProbability + 0.30 * AnomalyScore * 100 + 0.35 * (100 - HealthScore)
```

Risk bands:

- `Low`: less than 35
- `Medium`: 35 to 59.99
- `High`: 60 to 79.99
- `Critical`: 80 or above

## Notes

The original dataset does not include `HealthScore`, `WillFailSoon`, or `RemainingUsefulLifeDays`. This project generates those targets deterministically from sensor degradation patterns instead of random labels, so training remains reproducible and auditable.

The current CSV's `ProblemDetected` labels appear to have weak learnable signal relative to the supplied sensor columns. On the included train/test split, the best multiclass classifier is only slightly above random baseline for five balanced classes. The health score, failure probability, and RUL models perform much better because their targets are derived from deterministic degradation rules. For a real deployment, improve the multiclass problem model by collecting incident labels tied to confirmed diagnostics, timestamps, firmware data, adapter details, SMART telemetry, and motherboard revision metadata.
