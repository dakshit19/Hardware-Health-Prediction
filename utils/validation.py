"""Prediction input validation."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from config import CONFIG, ProjectConfig
from utils.exceptions import InputValidationError


def coerce_to_dataframe(payload: Mapping[str, object] | pd.DataFrame, config: ProjectConfig = CONFIG) -> pd.DataFrame:
    """Validate an inference payload and return a one-row dataframe."""

    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([dict(payload)])

    missing = [column for column in config.raw_feature_columns if column not in frame.columns]
    if missing:
        raise InputValidationError(f"Missing required prediction fields: {missing}")

    frame = frame.loc[:, list(config.raw_feature_columns)].copy()

    for column in config.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise InputValidationError(f"Column `{column}` must contain numeric values.")

    frame["ModelName"] = frame["ModelName"].astype(str).str.strip()
    if frame["ModelName"].eq("").any():
        raise InputValidationError("Column `ModelName` cannot be empty.")

    return frame
