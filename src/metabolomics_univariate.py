"""Két csoport összehasonlítása: Welch t-próba, log2 fold-change, BH-FDR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


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
) -> pd.DataFrame:
    """
    Minden oszlopra: átlag A, átlag B, log2FC (A vs B), Welch t-próba p-érték, FDR.

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
        rows.append(
            {
                "feature": col,
                "mean_" + str(group_a): m1,
                "mean_" + str(group_b): m0,
                "log2fc": log2fc,
                "pvalue": float(pval) if np.isfinite(pval) else 1.0,
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["padj"] = benjamini_hochberg(res["pvalue"].to_numpy())
    res = res.sort_values("pvalue")
    return res.reset_index(drop=True)
