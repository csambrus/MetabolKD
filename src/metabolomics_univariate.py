"""Két csoport összehasonlítása: Welch t-próba, log2 fold-change, FDR (BH)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cohens_d(va: np.ndarray, vb: np.ndarray) -> float:
    """Két minta közötti Cohen *d* (párosítatlan, pooled std)."""
    va = np.asarray(va, dtype=float)
    vb = np.asarray(vb, dtype=float)
    na, nb = int(va.size), int(vb.size)
    if na < 2 or nb < 2:
        return float("nan")
    sa = float(np.var(va, ddof=1))
    sb = float(np.var(vb, ddof=1))
    denom = (na - 1) * sa + (nb - 1) * sb
    pooled = np.sqrt(denom / max(na + nb - 2, 1)) if denom > 0 else 0.0
    if pooled == 0.0 or not np.isfinite(pooled):
        return 0.0
    return float((np.mean(va) - np.mean(vb)) / pooled)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR; p értékek 0..1."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ps = p[order]
    adj = np.zeros(n)
    adj[-1] = min(ps[-1] * n / n, 1.0)
    for i in range(n - 2, -1, -1):
        adj[i] = min(ps[i] * n / (i + 1), adj[i + 1], 1.0)
    out = np.zeros(n)
    out[order] = adj
    return np.clip(out, 0, 1)


def differential_analysis(
    X: pd.DataFrame,
    groups: pd.Series,
    group_a: str,
    group_b: str,
    *,
    eps: float = 1e-12,
    include_cohen_d: bool = True,
    include_mann_whitney: bool = False,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Minden oszlopra: átlag A, átlag B, log2FC (A vs B), Welch t-próba p-érték, FDR.

    ``fdr_method``: ``statsmodels.stats.multitest.multipletests`` metódus
    (alap: ``fdr_bh`` — ugyanaz az NMR notebookban használt BH-FDR).

    log2FC = log2((mean_a + eps) / (mean_b + eps)) az eredeti (nem log) skálán
    ha X már log1p: interpretáció „log1p térben” átlagokból számolt hányados.
    """
    groups = groups.reindex(X.index)
    ma = groups == group_a
    mb = groups == group_b
    rows = []
    for col in X.columns:
        va = X.loc[ma, col].astype(float).to_numpy()
        vb = X.loc[mb, col].astype(float).to_numpy()
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        if va.size < 2 or vb.size < 2:
            continue
        m1, m0 = float(np.mean(va)), float(np.mean(vb))
        log2fc = float(np.log2((m1 + eps) / (m0 + eps)))
        _, pval = stats.ttest_ind(va, vb, equal_var=False)
        row: dict[str, object] = {
            "feature": col,
            "mean_" + str(group_a): m1,
            "mean_" + str(group_b): m0,
            "log2fc": log2fc,
            "pvalue": float(pval) if np.isfinite(pval) else 1.0,
        }
        if include_cohen_d:
            row["cohens_d"] = cohens_d(va, vb)
        if include_mann_whitney:
            try:
                _, mw = stats.mannwhitneyu(va, vb, alternative="two-sided")
                row["mw_pvalue"] = float(mw) if np.isfinite(mw) else 1.0
            except ValueError:
                row["mw_pvalue"] = 1.0
        rows.append(row)
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    _, padj, _, _ = multipletests(
        res["pvalue"].to_numpy(dtype=float),
        method=fdr_method,
    )
    res["padj"] = np.clip(padj, 0.0, 1.0)
    res = res.sort_values("pvalue")
    return res.reset_index(drop=True)
