"""Ábrák: QC, PCA/PLS score, vulkán, top feature heatmap."""

from __future__ import annotations

from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_qc_bars(
    qc: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    hue_col: str = "progressor_status",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Teljes jel és hiányzás arány oszlopdiagram mintánként, színezés meta szerint."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    m = meta.reindex(qc.index)
    hues = m[hue_col] if hue_col in m.columns else pd.Series("NA", index=qc.index)
    categories = list(hues.astype(str).unique())
    cmap = matplotlib.colormaps["tab10"]
    color_map = {c: cmap(i % 10) for i, c in enumerate(categories)}
    colors = [color_map[str(hues.loc[i])] for i in qc.index]

    x = np.arange(len(qc))
    axes[0].bar(x, qc["total_signal"].to_numpy(), color=colors, edgecolor="white", linewidth=0.3)
    axes[0].set_title("Teljes jel (összeg) mintánként")
    axes[0].set_ylabel("Összeg")
    axes[0].set_xticks([])

    axes[1].bar(x, qc["missing_frac"].to_numpy(), color=colors, edgecolor="white", linewidth=0.3)
    axes[1].set_title("Hiányzó jellemzők aránya")
    axes[1].set_ylabel("Arány")
    axes[1].set_xticks([])

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[c]) for c in categories]
    fig.legend(handles, categories, loc="upper center", ncol=min(4, len(categories)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig


def plot_pca_scores(
    scores: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    hue_col: str = "progressor_status",
    explained: np.ndarray | None = None,
    title: str = "PCA score plot",
    figsize: tuple[float, float] = (7, 6),
) -> plt.Figure:
    meta = meta.reindex(scores.index)
    fig, ax = plt.subplots(figsize=figsize)
    if hue_col not in meta.columns:
        ax.scatter(scores[pc_x], scores[pc_y], s=45, alpha=0.85, edgecolors="white", linewidths=0.3)
    else:
        for lab in meta[hue_col].dropna().unique():
            m = meta[hue_col] == lab
            ax.scatter(
                scores.loc[m, pc_x],
                scores.loc[m, pc_y],
                s=45,
                alpha=0.85,
                label=str(lab),
                edgecolors="white",
                linewidths=0.3,
            )
    xl = f"{pc_x}"
    yl = f"{pc_y}"
    if explained is not None and pc_x.startswith("PC") and pc_y.startswith("PC"):
        i = int(pc_x.replace("PC", "")) - 1
        j = int(pc_y.replace("PC", "")) - 1
        if 0 <= i < len(explained) and 0 <= j < len(explained):
            xl += f" ({100*explained[i]:.1f} %)"
            yl += f" ({100*explained[j]:.1f} %)"
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    if hue_col in meta.columns:
        ax.legend(loc="best", title=hue_col)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    fig.tight_layout()
    return fig


def plot_volcano(
    diff: pd.DataFrame,
    *,
    log2fc_col: str = "log2fc",
    pval_col: str = "pvalue",
    padj_col: str = "padj",
    alpha: float = 0.05,
    fc_thresh: float = 0.5,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Vulkán: log2FC vs -log10(p); FDR és |FC| küszöb kiemelve."""
    fig, ax = plt.subplots(figsize=figsize)
    x = diff[log2fc_col].to_numpy()
    p = diff[pval_col].to_numpy()
    y = -np.log10(np.clip(p, 1e-300, None))
    sig = diff[padj_col].to_numpy() < alpha
    large = np.abs(x) >= fc_thresh
    highlight = sig & large
    ax.scatter(x[~highlight], y[~highlight], s=18, alpha=0.35, c="gray", edgecolors="none")
    ax.scatter(x[highlight], y[highlight], s=28, alpha=0.85, c="crimson", edgecolors="white", linewidths=0.2)
    ax.axhline(-np.log10(alpha), color="steelblue", ls="--", lw=1, label=f"padj={alpha}")
    ax.axvline(fc_thresh, color="green", ls=":", lw=0.8)
    ax.axvline(-fc_thresh, color="green", ls=":", lw=0.8)
    ax.set_xlabel("log2 fold-change")
    ax.set_ylabel("-log10 p-value")
    ax.set_title("Volcano (kihangsúlyozott: padj & |log2FC| küszöb)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_top_features_heatmap(
    X: pd.DataFrame,
    features: Sequence[str],
    meta: pd.DataFrame,
    *,
    group_col: str = "progressor_status",
    max_samples: int = 60,
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Z-score oszloponként a kiválasztott feature-ökre, minták rendezése csoport szerint."""
    feats = [f for f in features if f in X.columns]
    if not feats:
        raise ValueError("Nincs egyező feature név a mátrixban.")
    sub = X[feats].astype(float)
    m = meta.reindex(sub.index)
    if group_col in m.columns:
        idx = m.sort_values(group_col).index
        sub = sub.reindex(idx)
        m = m.reindex(idx)
    if len(sub) > max_samples:
        sub = sub.iloc[:max_samples]
        m = m.iloc[:max_samples]
    Z = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=1).replace(0, np.nan)
    Z = Z.fillna(0.0).to_numpy().T
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([str(f)[:60] for f in feats], fontsize=7)
    ax.set_xticks(np.arange(sub.shape[0]))
    ax.set_xticklabels(sub.index.astype(str), rotation=90, fontsize=6)
    ax.set_title("Top jellemzők (oszlop Z-score) — minták")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Z (oszlop)")
    fig.tight_layout()
    return fig
