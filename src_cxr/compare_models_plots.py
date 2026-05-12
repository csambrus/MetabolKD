from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ensure_dir
from src.plot_utils import save_show_close_figure

DEFAULT_METRICS = ["accuracy", "recall_macro", "f1_macro", "roc_auc_macro_ovr", "loss"]


def _candidate_history_paths(model_dir: Path) -> list[Path]:
    return [
        model_dir / "history.csv",
        model_dir / "history_full.csv",
        model_dir / "training_history.csv",
        model_dir / "history.json",
        model_dir / "training_history.json",
        model_dir / "history_head.csv",
        model_dir / "history_finetune.csv",
    ]


def _read_history_file(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "history" in data:
                data = data["history"]
            df = pd.DataFrame(data)
        else:
            return None
    except Exception as e:
        print(f"[WARN] Nem sikerult history-t olvasni: {path} | {e}")
        return None

    if len(df) == 0:
        return None

    if "epoch" not in df.columns:
        if "epoch_global" in df.columns:
            df.insert(0, "epoch", df["epoch_global"].astype(int))
        else:
            df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def find_training_history(model_dir: str | Path) -> pd.DataFrame | None:
    model_dir = Path(model_dir)
    for path in _candidate_history_paths(model_dir):
        df = _read_history_file(path)
        if df is not None:
            return df
    for path in sorted(model_dir.rglob("*history*.csv")) + sorted(model_dir.rglob("*history*.json")):
        df = _read_history_file(path)
        if df is not None:
            return df
    return None


def _infer_model_dir(row: pd.Series) -> Path | None:
    for col in ["out_dir", "model_dir"]:
        if col in row and pd.notna(row[col]):
            return Path(row[col])
    if "model_path" in row and pd.notna(row["model_path"]):
        return Path(row["model_path"]).parent
    return None


def _find_metric_pair(hist: pd.DataFrame, base_names: list[str]) -> tuple[str | None, str | None, str]:
    for name in base_names:
        train_col = name if name in hist.columns else None
        val_col = f"val_{name}" if f"val_{name}" in hist.columns else None
        if train_col is not None or val_col is not None:
            return train_col, val_col, name
    return None, None, base_names[0]


def plot_metric_bars(comparison_df: pd.DataFrame, metric: str, save_path: str | Path | None = None, title: str | None = None, show: bool = False):
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")
    df = comparison_df.copy()
    df["label"] = df["model"].astype(str) + "\n(" + df["data_variant"].astype(str) + ")"
    df = df.sort_values(metric, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["label"], df[metric])
    ax.set_title(title or f"Model comparison - {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model / variant")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    save_show_close_figure(fig, save_path=save_path, show=show)


def plot_metric_by_variant(comparison_df: pd.DataFrame, metric: str, save_path: str | Path | None = None, title: str | None = None, show: bool = False):
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in comparison dataframe.")
    pivot_df = comparison_df.pivot(index="model", columns="data_variant", values=metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_df.plot(kind="bar", ax=ax)
    ax.set_title(title or f"{metric} by model and data variant")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="data_variant")
    fig.tight_layout()
    save_show_close_figure(fig, save_path=save_path, show=show)


def plot_models_by_variant_one_row(comparison_df: pd.DataFrame, out_dir: str | Path, metrics: Iterable[str] | None = None, show: bool = False):
    out_dir = ensure_dir(Path(out_dir) / "models_by_variant_one_row")
    if metrics is None:
        metrics = DEFAULT_METRICS
    variants = list(comparison_df["data_variant"].dropna().unique())
    for metric in metrics:
        if metric not in comparison_df.columns:
            continue
        fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 5), sharey=True)
        if len(variants) == 1:
            axes = [axes]
        for ax, variant in zip(axes, variants):
            sub = comparison_df[comparison_df["data_variant"] == variant].copy()
            if sub.empty:
                ax.set_title(str(variant))
                ax.text(0.5, 0.5, "Nincs adat", ha="center", va="center")
                continue
            sub = sub.sort_values(metric, ascending=(metric == "loss"))
            ax.bar(sub["model"].astype(str), sub[metric])
            ax.set_title(str(variant))
            ax.set_xlabel("Model")
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
            if metric != "loss":
                ax.set_ylim(0, 1)
        axes[0].set_ylabel(metric)
        fig.suptitle(f"Modellek osszehasonlitasa variansonkent - {metric}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        save_path = out_dir / f"models_by_variant_one_row_{metric}.png"
        save_show_close_figure(fig, save_path=save_path, show=show)


def plot_metric_heatmaps(comparison_df: pd.DataFrame, out_dir: str | Path, metrics: Iterable[str] | None = None, show: bool = False):
    out_dir = ensure_dir(Path(out_dir) / "heatmaps")
    if metrics is None:
        metrics = DEFAULT_METRICS
    df = comparison_df.dropna(subset=["model", "data_variant"]).copy()
    for metric in metrics:
        if metric not in df.columns:
            continue
        pivot_df = df.dropna(subset=[metric]).pivot_table(index="model", columns="data_variant", values=metric, aggfunc="mean")
        if pivot_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(pivot_df.columns)), max(5, 0.6 * len(pivot_df.index))))
        im = ax.imshow(pivot_df.values, aspect="auto")
        ax.set_title(f"Model x varians heatmap - {metric}")
        ax.set_xlabel("Data variant")
        ax.set_ylabel("Model")
        ax.set_xticks(range(len(pivot_df.columns)))
        ax.set_xticklabels(pivot_df.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot_df.index)))
        ax.set_yticklabels(pivot_df.index)
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                val = pivot_df.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label=metric)
        fig.tight_layout()
        save_show_close_figure(fig, save_path=out_dir / f"heatmap_{metric}.png", show=show)


def plot_all_main_metrics(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False):
    out_dir = ensure_dir(out_dir)
    for metric in DEFAULT_METRICS:
        if metric not in comparison_df.columns:
            continue
        plot_metric_bars(comparison_df, metric, save_path=Path(out_dir) / f"bar_{metric}.png", title=f"Comparison - {metric}", show=show)
        plot_metric_by_variant(comparison_df, metric, save_path=Path(out_dir) / f"grouped_{metric}.png", title=f"{metric} by model / variant", show=show)
    plot_models_by_variant_one_row(comparison_df, out_dir=out_dir, metrics=DEFAULT_METRICS, show=show)
    plot_metric_heatmaps(comparison_df, out_dir, show=show)


def plot_training_history_for_row(row: pd.Series, out_dir: str | Path, show: bool = False) -> Path | None:
    model = str(row["model"])
    variant = str(row["data_variant"])
    model_dir = _infer_model_dir(row)
    if model_dir is None:
        return None
    hist = find_training_history(model_dir)
    if hist is None:
        return None
    out_dir = ensure_dir(Path(out_dir) / "training_curves")
    metric_specs: list[tuple[str | None, str | None, str]] = []
    for names, title in [
        (["loss"], "Loss"),
        (["accuracy", "sparse_categorical_accuracy"], "Accuracy"),
        (["recall_macro", "macro_recall", "recall"], "Macro recall"),
        (["roc_auc_macro_ovr", "macro_auc", "auc"], "Macro ROC-AUC"),
    ]:
        train_col, val_col, _ = _find_metric_pair(hist, names)
        if train_col or val_col:
            metric_specs.append((train_col, val_col, title))
    if not metric_specs:
        return None
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5 * len(metric_specs), 4), squeeze=False)
    axes = axes[0]
    for ax, (train_metric, val_metric, title) in zip(axes, metric_specs):
        if train_metric and train_metric in hist.columns:
            ax.plot(hist["epoch"], hist[train_metric], marker="o", label=train_metric)
        if val_metric and val_metric in hist.columns:
            ax.plot(hist["epoch"], hist[val_metric], marker="o", label=val_metric)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Training curves - {model} / {variant}")
    fig.tight_layout()
    save_path = out_dir / f"{model}_{variant}_training_row.png"
    save_show_close_figure(fig, save_path=save_path, show=show)
    return save_path


def plot_all_training_histories(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False):
    for _, row in comparison_df.iterrows():
        plot_training_history_for_row(row, out_dir=out_dir, show=show)


def plot_history_comparison_by_variant(comparison_df: pd.DataFrame, out_dir: str | Path, metric: str = "val_accuracy", show: bool = False):
    out_dir = ensure_dir(Path(out_dir) / "epoch_comparison_by_variant")
    variants = list(comparison_df["data_variant"].dropna().unique())
    if not variants:
        return None
    fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 5), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        sub = comparison_df[comparison_df["data_variant"] == variant]
        plotted = False
        for _, row in sub.iterrows():
            model_dir = _infer_model_dir(row)
            if model_dir is None:
                continue
            hist = find_training_history(model_dir)
            if hist is None or metric not in hist.columns:
                continue
            x = hist["epoch"] if "epoch" in hist.columns else range(1, len(hist) + 1)
            ax.plot(x, hist[metric], marker="o", label=str(row["model"]))
            plotted = True
        ax.set_title(str(variant))
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
    axes[0].set_ylabel(metric)
    fig.suptitle(f"Modellek epochonkénti osszehasonlitasa variansonkent\nmetric={metric}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save_path = out_dir / f"combined_by_variant_{metric}.png"
    save_show_close_figure(fig, save_path=save_path, show=show)
    return save_path


def plot_history_comparison_by_model(comparison_df: pd.DataFrame, out_dir: str | Path, metric: str = "val_accuracy", show: bool = False):
    out_dir = ensure_dir(Path(out_dir) / "epoch_comparison_by_model")
    for model, sub in comparison_df.groupby("model"):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for _, row in sub.iterrows():
            model_dir = _infer_model_dir(row)
            if model_dir is None:
                continue
            hist = find_training_history(model_dir)
            if hist is None or metric not in hist.columns:
                continue
            ax.plot(hist["epoch"], hist[metric], marker="o", label=str(row["data_variant"]))
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Variansok epochonkenti osszehasonlitasa\nmodel={model} | metric={metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_show_close_figure(fig, save_path=out_dir / f"{model}_{metric}.png", show=show)


def plot_epoch_comparisons(comparison_df: pd.DataFrame, out_dir: str | Path, show: bool = False):
    for metric in ["val_accuracy", "val_recall_macro", "val_roc_auc_macro_ovr", "val_loss", "accuracy", "recall_macro", "roc_auc_macro_ovr", "loss"]:
        plot_history_comparison_by_variant(comparison_df, out_dir, metric=metric, show=show)
        plot_history_comparison_by_model(comparison_df, out_dir, metric=metric, show=show)
