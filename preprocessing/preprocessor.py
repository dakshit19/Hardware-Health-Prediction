"""Data cleaning, encoding, scaling, and split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import CONFIG, ProjectConfig


@dataclass
class SplitData:
    """Container for train/test data splits."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train_problem: pd.Series
    y_test_problem: pd.Series
    y_train_health: pd.Series
    y_test_health: pd.Series
    y_train_failure: pd.Series
    y_test_failure: pd.Series
    y_train_rul: pd.Series
    y_test_rul: pd.Series


class MotherboardPreprocessor:
    """Preprocessing orchestration for training and inference."""

    def __init__(self, config: ProjectConfig = CONFIG) -> None:
        self.config = config
        self.preprocessor = None
        self.label_encoder = None

    def clean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and normalize string labels."""

        cleaned = frame.drop_duplicates().copy()
        if "ModelName" in cleaned:
            cleaned["ModelName"] = cleaned["ModelName"].astype(str).str.strip()
        if self.config.target_column in cleaned:
            cleaned[self.config.target_column] = cleaned[self.config.target_column].astype(str).str.strip()
        return cleaned

    def build_preprocessor(self, numeric_columns: list[str], categorical_columns: list[str]):
        """Build an sklearn ColumnTransformer with imputation, scaling, and one-hot encoding."""

        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for preprocessing. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

    def encode_target(self, y_train: pd.Series, y_test: pd.Series):
        """Label-encode the problem target."""

        try:
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for label encoding.") from exc

        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        self.label_encoder = encoder
        return y_train_encoded, y_test_encoded, encoder

    def train_test_split(self, frame: pd.DataFrame, feature_columns: list[str]) -> SplitData:
        """Create stratified train/test splits for all model targets."""

        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise RuntimeError("scikit-learn is required for train/test split.") from exc

        x = frame.loc[:, feature_columns].copy()
        y_problem = frame[self.config.target_column].copy()
        y_health = frame[self.config.health_target_column].copy()
        y_failure = frame[self.config.failure_target_column].copy()
        y_rul = frame[self.config.rul_target_column].copy()

        split = train_test_split(
            x,
            y_problem,
            y_health,
            y_failure,
            y_rul,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_problem,
        )

        return SplitData(
            x_train=split[0],
            x_test=split[1],
            y_train_problem=split[2],
            y_test_problem=split[3],
            y_train_health=split[4],
            y_test_health=split[5],
            y_train_failure=split[6],
            y_test_failure=split[7],
            y_train_rul=split[8],
            y_test_rul=split[9],
        )
