"""I/O helpers for datasets, model artifacts, and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def require_joblib():
    """Import joblib with a clear error if the dependency is missing."""

    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "joblib is required for model persistence. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return joblib


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def save_joblib(obj: Any, path: Path) -> None:
    """Persist an object with joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib = require_joblib()
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load a joblib artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    joblib = require_joblib()
    return joblib.load(path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Persist metadata as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Load formatted JSON metadata."""

    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
