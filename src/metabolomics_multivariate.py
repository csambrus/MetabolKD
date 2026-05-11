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
    standardize: bool = True,
) -> dict[str, Any]:
    """
    PCA. Alapértelmezés: oszloponkénti Z-score, majd PCA (LC–MS általános).

    ``standardize=False``: a bemenet már Pareto-skálázott (NMR ``X_PARETO`` minta),
    csak ``nan_to_num`` + PCA — illeszkedik a ``NMR_Metabolomika_Pipeline_v2`` PCA lépéshez.
    """
    xm = X.astype(float).fillna(0).to_numpy()
    if standardize:
        Z = StandardScaler().fit_transform(xm)
    else:
        Z = xm
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


class PLSDAClassifier:
    """
    PLS-DA többosztályos címkével + VIP (``NMR_Metabolomika_Pipeline_v2`` mintájára).
    """

    def __init__(self, n_components: int = 2, scale: bool = True) -> None:
        self.n_components = n_components
        self.scale = scale
        self.pls = PLSRegression(n_components=n_components, scale=scale)
        self.classes_: np.ndarray | None = None
        self.vip_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PLSDAClassifier:
        self.classes_ = np.unique(y)
        y = np.asarray(y)
        n = len(y)
        k = len(self.classes_)
        y_oh = np.zeros((n, k))
        for i, c in enumerate(self.classes_):
            y_oh[y == c, i] = 1.0
        self.pls.fit(X, y_oh)
        self.vip_ = self._compute_vip()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.pls.predict(X)
        assert self.classes_ is not None
        return self.classes_[np.asarray(scores).argmax(axis=1)]

    def _compute_vip(self) -> np.ndarray:
        t = self.pls.x_scores_
        w = self.pls.x_weights_
        q = self.pls.y_loadings_
        _p, h = w.shape
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = float(np.sum(s))
        if total_s <= 0:
            return np.ones(_p)
        w_norm = w / np.linalg.norm(w, axis=0)
        return np.sqrt(_p * (np.sum(s.T * w_norm**2, axis=1) / total_s))


def fit_plsda_multiclass(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_components: int = 3,
    scale: bool = True,
) -> dict[str, Any]:
    """PLS-DA + VIP vektor; ``y`` string vagy kategorikus címkék."""
    y = y.reindex(X.index)
    xm = X.astype(float).fillna(0).to_numpy()
    xm = np.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
    plsda = PLSDAClassifier(n_components=n_components, scale=scale)
    plsda.fit(xm, y.to_numpy())
    scores = pd.DataFrame(
        plsda.pls.x_scores_,
        index=X.index,
        columns=[f"LV{i+1}" for i in range(plsda.pls.x_scores_.shape[1])],
    )
    vip = pd.Series(plsda.vip_, index=X.columns, name="VIP")
    return {"model": plsda, "scores": scores, "vip": vip, "classes": plsda.classes_}
