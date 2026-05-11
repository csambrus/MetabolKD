"""Előfeldolgozás: normalizálás, imputáció, log, Pareto-skálázás (NMR/LC-MS pipeline mintára)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def median_normalise_samples(X: pd.DataFrame) -> pd.DataFrame:
    """Mintánkénti medián-normalizálás (PQN-szerű, egyszerűsített): minden sor / sor medián (pozitív értékekből)."""
    Xf = X.astype(float)
    med = Xf.where(Xf > 0).median(axis=1)
    med = med.replace(0, np.nan)
    out = Xf.div(med, axis=0)
    return out


def impute_half_minimum(X: pd.DataFrame) -> pd.DataFrame:
    """NaN helyett fele annyi, mint az adott feature legkisebb pozitív értéke (ha nincs, 0)."""
    out = X.astype(float).copy()
    for c in out.columns:
        s = out[c]
        pos = s[s > 0]
        m = float(pos.min()) if len(pos) else 0.0
        fill = 0.5 * m if m > 0 else 0.0
        out[c] = s.fillna(fill)
    return out


def log1p_df(X: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(X.astype(float).clip(lower=0).fillna(0))


def pqn_normalize(
    X: pd.DataFrame,
    *,
    reference: str = "median",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Probabilistic Quotient Normalization (Dieterle et al., 2006) — mint a
    ``NMR_Metabolomika_Pipeline_v2`` referencia cellában.
    """
    arr = np.maximum(X.astype(float).to_numpy(), eps)
    integrals = arr.sum(axis=1, keepdims=True)
    xn = arr / integrals
    ref = np.median(xn, axis=0) if reference == "median" else xn.mean(axis=0)
    ref = np.maximum(ref, eps)
    quotients = xn / ref
    dilution = np.median(quotients, axis=1, keepdims=True)
    out = xn / dilution
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def nmr_style_matrices(
    X_raw: pd.DataFrame,
    *,
    impute_first: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Ugyanaz a logika, mint az NMR notebook 2.4 fejezetében:

    - ``X_raw`` : imputált (opcionális) numerikus mátrix
    - ``X_pqn`` : PQN a (nem negatív) értékeken
    - ``X_log`` : ``log1p(X_raw)`` — vulkán / univariáns gyakran ezen
    - ``X_pareto`` : Pareto-skálázás a ``log1p`` mátrixon — PCA / PLS-DA
    """
    x0 = impute_half_minimum(X_raw) if impute_first else X_raw.astype(float).copy()
    x_pqn = pqn_normalize(x0.clip(lower=0).fillna(0))
    x_log = log1p_df(x0)
    x_pareto = pareto_scale_columns(x_log)
    return {
        "X_raw": x0,
        "X_pqn": x_pqn,
        "X_log": x_log,
        "X_pareto": x_pareto,
    }


def pareto_scale_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Oszloponként középre igazítás, majd osztás sqrt(std)-del (Pareto-skálázás)."""
    Xf = X.astype(float)
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0, ddof=1)
    sd = sd.replace(0, np.nan)
    scaled = (Xf - mu) / np.sqrt(sd)
    return scaled.fillna(0.0)


def preprocess_feature_matrix(
    X: pd.DataFrame,
    *,
    median_norm: bool = True,
    impute: bool = True,
    log1p: bool = True,
    pareto: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Sorrend: medián norm → imputáció → log1p → Pareto.

    Returns
    -------
    X_proc : előfeldolgozott mátrix (minták × jellemzők)
    steps : rövid napló a reprodukálhatósághoz
    """
    steps: list[str] = []
    Y = X.astype(float).copy()
    if median_norm:
        Y = median_normalise_samples(Y)
        steps.append("median_sample_norm")
    if impute:
        Y = impute_half_minimum(Y)
        steps.append("impute_half_min")
    if log1p:
        Y = log1p_df(Y)
        steps.append("log1p")
    if pareto:
        Y = pareto_scale_columns(Y)
        steps.append("pareto_columns")
    return Y, steps
