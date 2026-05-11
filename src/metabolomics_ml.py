"""Egyszerű felügyelt modellek (RF fontosság) — NMR notebook RF-blokk egyszerűsítése."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def random_forest_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    positive_class: str,
    *,
    n_estimators: int = 300,
    max_depth: int | None = 12,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.Series:
    """
    Bináris címke: RF illesztés, ``feature_importances_`` csökkenő sorrendben.
    """
    y = y.reindex(X.index)
    y01 = (y.astype(str) == str(positive_class)).astype(int).to_numpy()
    xm = X.astype(float).fillna(0).to_numpy()
    xm = np.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    rf.fit(xm, y01)
    return pd.Series(rf.feature_importances_, index=X.columns, name="importance").sort_values(ascending=False)
