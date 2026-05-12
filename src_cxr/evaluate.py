from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_CLASSES,
    PLOT_DPI,
    ensure_dir,
    get_class_names,
    get_data_root,
    save_json,
)
from src.dataloader import build_datasets_from_split_csvs


# =========================================================
# Core prediction helpers
# =========================================================

def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_all = []
    y_prob_all = []

    for x_batch, y_batch in dataset:
        prob = model.predict(x_batch, verbose=0)
        y_true_all.append(y_batch.numpy())
        y_prob_all.append(prob)

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)

    if not np.all(np.isfinite(y_prob)):
        raise ValueError(
            "[ERROR] A modell predikciója NaN vagy inf értéket tartalmaz. "
            "Ellenőrizd a tanítást, preprocessinget és input skálázást."
        )

    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob


def compute_metrics(
    loss: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    roc_auc_macro_ovr = np.nan
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        roc_auc_macro_ovr = roc_auc_score(
            y_true_bin,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except Exception:
        pass

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_macro_ovr": float(roc_auc_macro_ovr),
    }


# =========================================================
# Plot helpers
# =========================================================

def plot_confusion_and_roc_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_path: str | Path,
):
    labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -----------------------------------------------------
    # confusion matrix
    # -----------------------------------------------------
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha="right")
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center"
            )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # -----------------------------------------------------
    # normalized confusion matrix
    # -----------------------------------------------------
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha="right")
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            axes[1].text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center"
            )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # -----------------------------------------------------
    # ROC
    # -----------------------------------------------------
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))

    for i, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            axes[2].plot(fpr, tpr, label=f"{class_name} (AUC={auc_i:.3f})")
        except Exception:
            continue

    axes[2].plot([0, 1], [0, 1], linestyle="--")
    axes[2].set_title("ROC Curves")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, loc="lower right")

    fig.suptitle(f"===== EVALUATION ===== {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Evaluation
# =========================================================

def run_evaluation(
    model_path: str | Path,
    split_dir: str | Path,
    out_dir: str | Path,
    model_name: str,
    data_root: str | Path | None = None,
    data_variant: str = "raw",
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> dict[str, Any]:
    if data_root is None:
        data_root = get_data_root(data_variant)
    data_root = Path(data_root)

    eval_out_dir = ensure_dir(Path(out_dir) / f"{model_name}_{data_variant}")
    class_names = get_class_names()

    print("=" * 72)
    print(f"===== EVALUATION ===== {model_name}")
    print("=" * 72)
    print("model_path   :", Path(model_path))
    print("split_dir    :", Path(split_dir))
    print("data_root    :", data_root)
    print("data_variant :", data_variant)
    print("out_dir      :", eval_out_dir)
    print("=" * 72)

    _, _, test_ds = build_datasets_from_split_csvs(
        split_dir=split_dir,
        data_root=data_root,
        batch_size=batch_size,
        augment_fn=None,
        cache=False,
        image_size=image_size,
        channels=1,
    )

    model = tf.keras.models.load_model(model_path, safe_mode=False)

    eval_result = model.evaluate(test_ds, verbose=1)

    if isinstance(eval_result, list):
        keras_eval_metrics = {
            name: float(value)
            for name, value in zip(model.metrics_names, eval_result)
        }
        loss = float(keras_eval_metrics.get("loss", eval_result[0]))
    else:
        keras_eval_metrics = {"loss": float(eval_result)}
        loss = float(eval_result)

    y_true, y_pred, y_prob = collect_predictions(model, test_ds)
    metrics = compute_metrics(
        loss=loss,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=NUM_CLASSES,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(eval_out_dir / "classification_report.csv", index=True)

    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            **{f"prob_{class_names[i]}": y_prob[:, i] for i in range(len(class_names))},
        }
    )
    pred_df.to_csv(eval_out_dir / "predictions.csv", index=False)

    plot_confusion_and_roc_row(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        model_name=model_name,
        save_path=eval_out_dir / "evaluation_row.png",
    )

    summary = {
        "model_name": model_name,
        "data_variant": data_variant,
        "model_path": str(Path(model_path)),
        "split_dir": str(Path(split_dir)),
        "data_root": str(data_root),
        "out_dir": str(eval_out_dir),
        "keras_eval_metrics": keras_eval_metrics,
        "metrics": metrics,
    }
    save_json(summary, eval_out_dir / "metrics.json")

    print("\nMetrics")
    print("-" * 72)
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.6f}" if isinstance(v, float) and not np.isnan(v) else f"{k:<20}: {v}")
    print("-" * 72)

    return summary
