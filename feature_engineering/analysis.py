"""Feature analysis helpers."""

from __future__ import annotations

import pandas as pd


def correlation_report(frame: pd.DataFrame, min_abs_correlation: float = 0.85) -> pd.DataFrame:
    """Return highly correlated numeric feature pairs."""

    numeric = frame.select_dtypes(include="number")
    corr = numeric.corr().abs()
    pairs: list[dict[str, object]] = []
    columns = list(corr.columns)

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if value >= min_abs_correlation:
                pairs.append({"feature_a": left, "feature_b": right, "abs_correlation": round(float(value), 4)})

    return pd.DataFrame(pairs).sort_values("abs_correlation", ascending=False) if pairs else pd.DataFrame()
