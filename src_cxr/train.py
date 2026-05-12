# train.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

try:
    from IPython.display import Image as IPyImage, display
except Exception:  # notebookon kívül is működjön
    IPyImage = None
    display = None

from src.runtime import set_seed

from src.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
    EPOCHS,
    PLOT_DPI,
    SEED,
    ensure_dir,
    get_class_names,
    get_data_root,
    save_json,
)
from src.dataloader import build_datasets_from_split_csvs, build_default_augmentation


# =========================================================
# Display helpers
# =========================================================

def _display_png(path: str | Path) -> None:
    """PNG megjelenítése notebookban, ha IPython környezetben fut."""
    if display is None or IPyImage is None:
        return

    path = Path(path)
    if path.exists():
        display(IPyImage(filename=str(path)))


def _save_show_close(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

    if show:
        plt.show()
        if save_path is not None:
            _display_png(save_path)
    else:
        plt.close(fig)


# =========================================================
# Model builders
# =========================================================

def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Transfer model grayscale -> RGB.

    Fontos:
    Nem használunk Lambda(preprocess_input) réteget, mert Keras 3 / TF 2.19 alatt
    .keras mentés-betöltéskor deszerializációs hibát okozhat.
    """
    name = model_name.lower()
    base_weights = "imagenet" if pretrained else None

    inputs = tf.keras.Input(shape=input_shape, name="grayscale_input")

    x = tf.keras.layers.Concatenate(name="gray_to_rgb")(
        [inputs, inputs, inputs]
    )

    if name == "resnet50":
        # A dataloader 0..1 közé skáláz. ResNet/VGG ImageNet súlyokhoz
        # legalább 0..255 tartományra visszaskálázzuk, Lambda nélkül.
        x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(x)
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

    elif name == "vgg16":
        x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(x)
        base_model = tf.keras.applications.VGG16(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

    elif name == "efficientnetb0":
      # EfficientNetB0 előtt vissza 0..255-re
        x = tf.keras.layers.Rescaling(255.0, name="scale_to_255")(x)
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=base_weights,
            input_shape=(input_shape[0], input_shape[1], 3),
        )

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs, outputs, name=name)
    return model, base_model


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model | None]:
    name = model_name.lower()

    if name == "baseline_cnn":
        model = build_baseline_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
        )
        return model, None

    return build_transfer_model(
        model_name=name,
        input_shape=input_shape,
        num_classes=num_classes,
        pretrained=pretrained,
    )


# =========================================================
# Compile / callbacks
# =========================================================

def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """
    Sparse multiclass setup:

    - labels: integer class ids, shape: (batch,)
    - predictions: softmax probabilities, shape: (batch, NUM_CLASSES)

    A Keras beépített Recall / AUC metrikái sparse multiclass esetben könnyen
    shape hibát okoznak, ezért training közben csak accuracy-t mérünk közvetlenül.
    A macro recall és macro ROC-AUC értékeket külön callback számolja epoch végén
    a validációs halmazon, és beírja a history / CSVLogger logjaiba.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )


def _collect_probs_for_metrics(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    for x_batch, y_batch in dataset:
        probs = model.predict(x_batch, verbose=0)
        y_true_all.append(y_batch.numpy())
        y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)

    return y_true, y_prob


def _safe_macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    except Exception:
        return float("nan")


def _safe_macro_auc_ovr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> float:
    try:
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        return float(
            roc_auc_score(
                y_true_bin,
                y_prob,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return float("nan")


class ValidationClassificationMetrics(tf.keras.callbacks.Callback):
    """
    Epoch végén teljes validációs halmazon számol:
    - val_recall_macro
    - val_auc_macro_ovr

    Aliasokat is ír:
    - val_recall
    - val_auc

    Így a compare_models.py régebbi és újabb plotting logikája is megtalálja.
    """

    def __init__(
        self,
        val_ds: tf.data.Dataset,
        num_classes: int = NUM_CLASSES,
        prefix: str = "val",
    ):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = int(num_classes)
        self.prefix = str(prefix)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is None:
            logs = {}

        y_true, y_prob = _collect_probs_for_metrics(self.model, self.val_ds)
        y_pred = np.argmax(y_prob, axis=1)

        recall_macro = _safe_macro_recall(y_true, y_pred)
        auc_macro_ovr = _safe_macro_auc_ovr(
            y_true=y_true,
            y_prob=y_prob,
            num_classes=self.num_classes,
        )

        logs[f"{self.prefix}_recall_macro"] = recall_macro
        logs[f"{self.prefix}_auc_macro_ovr"] = auc_macro_ovr

        # Rövidebb aliasok, hogy a history plotok egyszerűbben megtalálják.
        logs[f"{self.prefix}_recall"] = recall_macro
        logs[f"{self.prefix}_auc"] = auc_macro_ovr

        print(
            f" - {self.prefix}_recall_macro: {recall_macro:.4f}"
            f" - {self.prefix}_auc_macro_ovr: {auc_macro_ovr:.4f}"
        )


def build_callbacks(
    out_dir: str | Path,
    val_ds: tf.data.Dataset | None = None,
) -> list[tf.keras.callbacks.Callback]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # Fontos: CSVLogger ELÉ kerüljön, hogy a callback által hozzáadott
    # val_recall_macro / val_auc_macro_ovr is bekerüljön a history.csv-be.
    if val_ds is not None:
        callbacks.append(
            ValidationClassificationMetrics(
                val_ds=val_ds,
                num_classes=NUM_CLASSES,
                prefix="val",
            )
        )

    callbacks.append(tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")))

    return callbacks


# =========================================================
# Plot helpers
# =========================================================

def _metric_pair_available(df: pd.DataFrame, train_metric: str, val_metric: str) -> bool:
    return train_metric in df.columns or val_metric in df.columns


def _plot_metric_pair(
    ax,
    df: pd.DataFrame,
    train_metric: str,
    val_metric: str,
    title: str,
    y_label: str,
) -> None:
    x_col = "epoch_global" if "epoch_global" in df.columns else "epoch"

    if x_col not in df.columns:
        x = np.arange(1, len(df) + 1)
    else:
        x = df[x_col]

    if train_metric in df.columns:
        ax.plot(x, df[train_metric], marker="o", label=train_metric)

    if val_metric in df.columns:
        ax.plot(x, df[val_metric], marker="o", label=val_metric)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def _add_phase_separator(axes, history_df: pd.DataFrame) -> None:
    if "phase" not in history_df.columns or "epoch_global" not in history_df.columns:
        return

    phases = history_df["phase"].astype(str).tolist()
    if len(phases) < 2:
        return

    change_positions = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            change_positions.append(float(history_df["epoch_global"].iloc[i]) - 0.5)

    for ax in axes:
        for pos in change_positions:
            ax.axvline(pos, linestyle="--", alpha=0.5)


def _add_best_epoch_marker(axes, history_df: pd.DataFrame) -> None:
    if "val_accuracy" not in history_df.columns:
        return

    x_col = "epoch_global" if "epoch_global" in history_df.columns else "epoch"
    if x_col not in history_df.columns:
        return

    try:
        idx = history_df["val_accuracy"].astype(float).idxmax()
        best_epoch = float(history_df.loc[idx, x_col])
    except Exception:
        return

    for ax in axes:
        ax.axvline(best_epoch, linestyle=":", alpha=0.5)


def plot_training_history(
    history_df: pd.DataFrame,
    save_path: str | Path,
    title: str,
    show: bool = False,
) -> None:
    """
    Egy training futáshoz tartozó görbék egy sorban.

    Panelek:
      1. loss / val_loss
      2. accuracy / val_accuracy
      3. recall_macro / val_recall_macro, ha elérhető
      4. auc_macro_ovr / val_auc_macro_ovr, ha elérhető

    Ha csak loss + accuracy van, akkor csak 2 panel készül.
    """
    history_df = history_df.copy()

    if "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    metric_specs = [
        ("loss", "val_loss", "Loss", "Loss"),
        ("accuracy", "val_accuracy", "Accuracy", "Accuracy"),
        ("recall_macro", "val_recall_macro", "Macro recall", "Recall"),
        ("auc_macro_ovr", "val_auc_macro_ovr", "Macro ROC-AUC", "AUC"),
    ]

    # Alias fallbackok régebbi history fájlokhoz.
    if "val_recall_macro" not in history_df.columns and "val_recall" in history_df.columns:
        history_df["val_recall_macro"] = history_df["val_recall"]
    if "val_auc_macro_ovr" not in history_df.columns and "val_auc" in history_df.columns:
        history_df["val_auc_macro_ovr"] = history_df["val_auc"]

    available_specs = [
        spec for spec in metric_specs
        if _metric_pair_available(history_df, spec[0], spec[1])
    ]

    if len(available_specs) == 0:
        print("[WARN] No plottable training metrics found.")
        return

    fig, axes = plt.subplots(
        1,
        len(available_specs),
        figsize=(5 * len(available_specs), 4),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (train_metric, val_metric, panel_title, y_label) in zip(axes, available_specs):
        _plot_metric_pair(
            ax=ax,
            df=history_df,
            train_metric=train_metric,
            val_metric=val_metric,
            title=panel_title,
            y_label=y_label,
        )

    _add_phase_separator(axes, history_df)
    _add_best_epoch_marker(axes, history_df)

    fig.suptitle(title)
    fig.tight_layout()

    _save_show_close(fig, save_path=save_path, show=show)


# =========================================================
# Training
# =========================================================

def run_training(
    split_dir: str | Path,
    out_dir: str | Path = MODELS_DIR,
    model_name: str = "baseline_cnn",
    pretrained: bool = True,
    do_fine_tuning: bool = False,
    epochs_head: int = EPOCHS,
    epochs_finetune: int = 5,
    learning_rate_head: float = 1e-3,
    learning_rate_finetune: float = 1e-5,
    data_root: str | Path | None = None,
    data_variant: str = "raw",
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
    show_plots: bool = True,
) -> dict[str, Any]:
    set_seed(SEED)

    if data_root is None:
        data_root = get_data_root(data_variant)

    data_root = Path(data_root)
    split_dir = Path(split_dir)

    model_out_dir = ensure_dir(Path(out_dir) / f"{model_name}_{data_variant}")

    print("=" * 72)
    print("TRAINING")
    print("=" * 72)
    print("model_name   :", model_name)
    print("data_variant :", data_variant)
    print("data_root    :", data_root)
    print("split_dir    :", split_dir)
    print("out_dir      :", model_out_dir)
    print("pretrained   :", pretrained)
    print("fine_tuning  :", do_fine_tuning)
    print("show_plots   :", show_plots)
    print("=" * 72)

    augmentation = build_default_augmentation()

    train_ds, val_ds, test_ds = build_datasets_from_split_csvs(
        split_dir=split_dir,
        data_root=data_root,
        batch_size=batch_size,
        augment_fn=augmentation,
        cache=False,
        image_size=image_size,
        channels=1,
    )

    model, base_model = build_model(
        model_name=model_name,
        input_shape=(image_size[0], image_size[1], 1),
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
    )

    if base_model is not None:
        base_model.trainable = False

    compile_model(model, learning_rate=learning_rate_head)
    callbacks = build_callbacks(model_out_dir, val_ds=val_ds)

    history_frames: list[pd.DataFrame] = []

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    head_df = pd.DataFrame(history_head.history)
    head_df["phase"] = "head"
    head_df["epoch_global"] = range(1, len(head_df) + 1)
    history_frames.append(head_df)

    if do_fine_tuning and base_model is not None and epochs_finetune > 0:
        print("=" * 72)
        print("FINE-TUNING")
        print("=" * 72)

        base_model.trainable = True

        if model_name.lower() == "efficientnetb0":
            for layer in base_model.layers[:-30]:
                layer.trainable = False

            for layer in base_model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

        compile_model(model, learning_rate=learning_rate_finetune)
        callbacks = build_callbacks(model_out_dir, val_ds=val_ds)

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_head + epochs_finetune,
            initial_epoch=epochs_head,
            callbacks=callbacks,
            verbose=1,
        )

        ft_df = pd.DataFrame(history_ft.history)
        ft_df["phase"] = "finetune"
        ft_df["epoch_global"] = range(
            len(head_df) + 1,
            len(head_df) + len(ft_df) + 1,
        )
        history_frames.append(ft_df)

    history_df = pd.concat(history_frames, ignore_index=True)

    if "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))

    # Egységes history fájlok. A compare_models.py elsőként history.csv-t keres,
    # ezért ide is a teljes, head + finetune history kerüljön.
    history_df.to_csv(model_out_dir / "history.csv", index=False)
    history_df.to_csv(model_out_dir / "history_full.csv", index=False)

    plot_training_history(
        history_df=history_df,
        save_path=model_out_dir / "training_history.png",
        title=f"Training history - {model_name} ({data_variant})",
        show=show_plots,
    )

    best_model_path = model_out_dir / "best_model.keras"
    final_model_path = model_out_dir / "last_model.keras"

    model.save(final_model_path)

    summary = {
        "model_name": model_name,
        "data_variant": data_variant,
        "data_root": str(data_root),
        "split_dir": str(split_dir),
        "out_dir": str(model_out_dir),
        "best_model_path": str(best_model_path),
        "last_model_path": str(final_model_path),
        "class_names": get_class_names(),
        "num_classes": int(NUM_CLASSES),
        "input_shape": [int(image_size[0]), int(image_size[1]), 1],
        "pretrained": bool(pretrained),
        "do_fine_tuning": bool(do_fine_tuning),
        "epochs_head": int(epochs_head),
        "epochs_finetune": int(epochs_finetune),
        "learning_rate_head": float(learning_rate_head),
        "learning_rate_finetune": float(learning_rate_finetune),
        "batch_size": int(batch_size),
        "metrics": [
            "accuracy",
            "val_recall_macro",
            "val_auc_macro_ovr",
        ],
        "loss": "sparse_categorical_crossentropy",
        "history_csv": str(model_out_dir / "history.csv"),
        "history_full_csv": str(model_out_dir / "history_full.csv"),
        "training_history_png": str(model_out_dir / "training_history.png"),
    }

    save_json(summary, model_out_dir / "train_summary.json")

    print("=" * 72)
    print("[OK] Training finished")
    print("=" * 72)
    print("model_name      :", model_name)
    print("data_variant    :", data_variant)
    print("best_model_path :", best_model_path)
    print("last_model_path :", final_model_path)
    print("history_csv     :", model_out_dir / "history.csv")
    print("history_png     :", model_out_dir / "training_history.png")
    print("=" * 72)

    return summary
