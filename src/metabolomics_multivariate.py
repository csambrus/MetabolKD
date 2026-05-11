"""PCA és kétosztályos PLS (PLS-DA-szerű score tér)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_pca(
    X: pd.DataFrame,
    *,
    n_components: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Z-score (oszlop) → PCA.

    Returns
    -------
    dict kulcsok: scores (DataFrame index=minta), loadings, explained_variance_ratio, model, scaler
    """
    Z = StandardScaler().fit_transform(X.astype(float).fillna(0).to_numpy())
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=n_components, random_state=random_state)
    T = pca.fit_transform(Z)
    scores = pd.DataFrame(
        T,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(T.shape[1])],
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
    )
    return {
        "scores": scores,
        "loadings": loadings,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "model": pca,
        "Z": Z,
    }


def fit_pls_binary(
    X: pd.DataFrame,
    y: pd.Series,
    positive_class: str,
    *,
    n_components: int = 2,
    scale: bool = True,
) -> dict[str, Any]:
    """
    Y bináris: 1 ha y == positive_class, különben 0. PLSRegression score plot első komponensekhez.
    """
    y = y.reindex(X.index)
    y01 = (y.astype(str) == str(positive_class)).astype(float).to_numpy().reshape(-1, 1)
    xm = X.astype(float).fillna(0).to_numpy()
    if scale:
        xm = StandardScaler().fit_transform(xm)
        xm = np.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(xm, y01)
    # X_scores_: minták × komponensek
    scores = pd.DataFrame(
        pls.x_scores_,
        index=X.index,
        columns=[f"LV{i+1}" for i in range(pls.x_scores_.shape[1])],
    )
    return {"scores": scores, "model": pls, "y_used": y01.ravel(), "X_matrix": xm}
