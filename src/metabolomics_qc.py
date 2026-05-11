"""Mintánkénti QC mutatók (összjel, hiányzás) — pipeline QC lépés."""

from __future__ import annotations

import numpy as np
import pandas as pd


def qc_sample_table(X: pd.DataFrame) -> pd.DataFrame:
    """
    Mintánként: teljes jel (összeg), hiányzó cellák száma és aránya.

    Parameters
    ----------
    X:
        minták × jellemzők (eredeti vagy log előtti numerikus mátrix)
    """
    Xf = X.astype(float)
    n_miss = Xf.isna().sum(axis=1)
    n_tot = Xf.shape[1]
    total = Xf.fillna(0).sum(axis=1)
    return pd.DataFrame(
        {
            "sample_id": Xf.index.astype(str),
            "total_signal": total,
            "n_missing": n_miss,
            "missing_frac": n_miss / max(n_tot, 1),
        },
        index=Xf.index,
    )
