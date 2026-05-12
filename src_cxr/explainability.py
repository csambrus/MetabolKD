# src/explainability.py

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    CLASS_INFOS,
    IMAGE_SIZE,
    MODELS_DIR,
    OUTPUT_DIR,
    SEED,
    SPLITS_DIR,
    ensure_dir,
    save_json,
    get_class_names,
    get_class_name
)
from src.dataloader import build_datasets_from_split_csvs
from src.preprocessing import XrayPreprocessLayer, decode_xray_image

# =========================================================
# Modell path helper
# =========================================================

def resolve_model_path(
    model_name: str | None = None,
    model_path: str | Path | None = None,
    prefer_best: bool = True,
) -> Path:
    """
    Vagy közvetlen model_path-et adsz meg, vagy model_name-et.
    Ha model_name van, akkor:
        MODELS_DIR / model_name / best_model.keras
        MODELS_DIR / model_name / final_model.keras
    """
    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    if model_name is None:
        raise ValueError("Adj meg vagy model_name-et, vagy model_path-et.")

    model_dir = Path(MODELS_DIR) / model_name

    candidates = []
    if prefer_best:
        candidates = [
            model_dir / "best_model.keras",
            model_dir / "final_model.keras",
        ]
    else:
        candidates = [
            model_dir / "final_model.keras",
            model_dir / "best_model.keras",
        ]

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Nem találtam modellt itt: {model_dir}\n"
        f"Várt fájlok: best_model.keras vagy final_model.keras"
    )


# =========================================================
# Kép betöltés / preprocess
# =========================================================

def load_raw_image(path: str | Path) -> np.ndarray:
    """
    Raw grayscale kép [H, W, 1] float32 alakban, [0,1] tartományban.
    """
    img = decode_xray_image(tf.convert_to_tensor(str(path)))
    return img.numpy()


def load_processed_input(
    path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    A modell inputjához illeszkedő preprocesszált kép [H, W, 1] float32 alakban.
    """
    preprocess = XrayPreprocessLayer(image_size=image_size)
    raw = decode_xray_image(tf.convert_to_tensor(str(path)))
    proc = preprocess(raw)
    return proc.numpy()


# =========================================================
# Predikciók összegyűjtése a teljes test setre
# =========================================================

def collect_predictions(model, dataset):
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []

    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# =========================================================
# Last conv layer keresés
# =========================================================

def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    """
    Megpróbálja megtalálni az utolsó 4D kimenetű réteget.
    Ez Grad-CAM-hoz általában megfelelő.
    """
    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
        except Exception:
            continue

        if output_shape is not None and len(output_shape) == 4:
            return layer.name

    raise ValueError("Nem találtam Grad-CAM-kompatibilis 4D réteget a modellben.")


# =========================================================
# Grad-CAM
# =========================================================

def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    last_conv_layer_name: str | None = None,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    image_tensor: [1, H, W, C]
    Visszaad: [Hf, Wf] normalizált heatmap [0,1]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_tensor, training=False)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if float(max_val) > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy().astype(np.float32)


# =========================================================
# Saliency
# =========================================================

def make_saliency_map(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    pred_index: int | None = None,
) -> np.ndarray:
    """
    image_tensor: [1, H, W, C]
    Visszaad: [H, W] normalizált saliency [0,1]
    """
    image_tensor = tf.cast(image_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        preds = model(image_tensor, training=False)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))

        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, image_tensor)
    grads = tf.abs(grads)

    saliency = tf.reduce_max(grads, axis=-1)[0]
    saliency = saliency - tf.reduce_min(saliency)

    max_val = tf.reduce_max(saliency)
    if float(max_val) > 0:
        saliency = saliency / max_val

    return saliency.numpy().astype(np.float32)


# =========================================================
# Vizualizációs utilok
# =========================================================

def resize_heatmap_to_image(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    heatmap_tf = tf.convert_to_tensor(heatmap[..., np.newaxis], dtype=tf.float32)
    heatmap_tf = tf.image.resize(heatmap_tf, target_hw, method="bilinear")
    heatmap = tf.squeeze(heatmap_tf, axis=-1).numpy()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap


def gray_to_rgb(gray_image: np.ndarray) -> np.ndarray:
    """
    [H,W] vagy [H,W,1] -> [H,W,3]
    """
    if gray_image.ndim == 3 and gray_image.shape[-1] == 1:
        gray_image = gray_image[..., 0]
    rgb = np.stack([gray_image, gray_image, gray_image], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def overlay_heatmap_on_image(
    base_gray_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    """
    base_gray_image: [H,W] vagy [H,W,1], [0,1]
    heatmap: [H,W], [0,1]
    """
    base_rgb = gray_to_rgb(base_gray_image)

    colored = cm.get_cmap("jet")(heatmap)[..., :3]
    overlay = (1 - alpha) * base_rgb + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def save_explainability_panel(
    raw_image: np.ndarray,
    processed_image: np.ndarray,
    gradcam_overlay: np.ndarray,
    saliency_map: np.ndarray,
    title: str,
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(raw_image[..., 0], cmap="gray")
    axes[0].set_title("Raw")
    axes[0].axis("off")

    axes[1].imshow(processed_image[..., 0], cmap="gray")
    axes[1].set_title("Processed")
    axes[1].axis("off")

    axes[2].imshow(gradcam_overlay)
    axes[2].set_title("Grad-CAM")
    axes[2].axis("off")

    axes[3].imshow(saliency_map, cmap="gray")
    axes[3].set_title("Saliency")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


def save_summary_grid(
    panel_infos: list[dict[str, Any]],
    save_path: str | Path,
    ncols: int = 3,
) -> None:
    """
    Egy összesítő grid: raw / gradcam / saliency.
    """
    if len(panel_infos) == 0:
        return

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    n = len(panel_infos)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, info in enumerate(panel_infos):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        ax.imshow(info["gradcam_overlay"])
        ax.set_title(info["short_title"], fontsize=9)
        ax.axis("off")

    # üres panelek kikapcsolása
    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.show()


# =========================================================
# Minta kiválasztás
# =========================================================

def select_example_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_correct: int = 4,
    n_incorrect: int = 4,
    n_random: int = 4,
    seed: int = SEED,
) -> list[int]:
    rng = np.random.default_rng(seed)

    correct_idx = np.where(y_true == y_pred)[0]
    incorrect_idx = np.where(y_true != y_pred)[0]
    all_idx = np.arange(len(y_true))

    selected: list[int] = []

    if len(correct_idx) > 0 and n_correct > 0:
        chosen = rng.choice(correct_idx, size=min(n_correct, len(correct_idx)), replace=False)
        selected.extend(chosen.tolist())

    if len(incorrect_idx) > 0 and n_incorrect > 0:
        chosen = rng.choice(incorrect_idx, size=min(n_incorrect, len(incorrect_idx)), replace=False)
        selected.extend(chosen.tolist())

    remaining_pool = np.array([i for i in all_idx if i not in selected], dtype=int)
    if len(remaining_pool) > 0 and n_random > 0:
        chosen = rng.choice(remaining_pool, size=min(n_random, len(remaining_pool)), replace=False)
        selected.extend(chosen.tolist())

    # duplikáció kizárása, sorrend megőrzése
    selected_unique = []
    seen = set()
    for i in selected:
        if i not in seen:
            selected_unique.append(int(i))
            seen.add(int(i))

    return selected_unique


# =========================================================
# Fő futtatás
# =========================================================

def run_explainability_qc(
    model_name: str | None = None,
    model_path: str | Path | None = None,
    split_dir: str | Path = SPLITS_DIR,
    out_dir: str | Path | None = None,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = 16,
    seed: int = SEED,
    last_conv_layer_name: str | None = None,
    n_correct: int = 4,
    n_incorrect: int = 4,
    n_random: int = 4,
    prefer_best_model: bool = True,
):
    """
    Fő explainability pipeline.

    Ha model_name van:
        MODELS_DIR / model_name / best_model.keras
    Ha out_dir nincs megadva:
        OUTPUT_DIR / "figures" / "explainability" / model_name
    """
    resolved_model_path = resolve_model_path(
        model_name=model_name,
        model_path=model_path,
        prefer_best=prefer_best_model,
    )

    if model_name is None:
        model_name = resolved_model_path.parent.name

    if out_dir is None:
        out_dir = Path(OUTPUT_DIR) / "figures" / "explainability" / model_name

    out_dir = ensure_dir(out_dir)
    panels_dir = ensure_dir(Path(out_dir) / "panels")

    print("=" * 80)
    print("EXPLAINABILITY QC")
    print("=" * 80)
    print(f"Model:      {resolved_model_path}")
    print(f"Model name: {model_name}")
    print(f"Split dir:  {split_dir}")
    print(f"Output dir: {out_dir}")
    print("=" * 80)

    # -----------------------------------------------------
    # Modell + dataset
    # -----------------------------------------------------
    model = tf.keras.models.load_model(resolved_model_path, safe_mode = False)

    _, _, test_ds, _, _, test_df = build_datasets_from_split_csvs(
        split_dir=split_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    y_true, y_pred, y_prob = collect_predictions(model, test_ds)

    if len(test_df) != len(y_true):
        raise ValueError(
            f"A test_df ({len(test_df)}) és a predikciók száma ({len(y_true)}) eltér."
        )

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    print(f"[INFO] Last conv layer for Grad-CAM: {last_conv_layer_name}")

    selected_indices = select_example_indices(
        y_true=y_true,
        y_pred=y_pred,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        n_random=n_random,
        seed=seed,
    )

    print(f"[INFO] Selected examples: {len(selected_indices)}")

    panel_rows: list[dict[str, Any]] = []
    summary_panels: list[dict[str, Any]] = []

    for rank, idx in enumerate(selected_indices):
        row = test_df.iloc[idx]
        filepath = row["filepath"]

        true_idx = int(y_true[idx])
        pred_idx = int(y_pred[idx])
        confidence = float(np.max(y_prob[idx]))

        raw_image = load_raw_image(filepath)
        processed_image = load_processed_input(filepath, image_size=image_size)

        input_tensor = tf.convert_to_tensor(processed_image[np.newaxis, ...], dtype=tf.float32)

        gradcam_small = make_gradcam_heatmap(
            model=model,
            image_tensor=input_tensor,
            last_conv_layer_name=last_conv_layer_name,
            pred_index=pred_idx,
        )
        gradcam = resize_heatmap_to_image(gradcam_small, target_hw=image_size)

        saliency = make_saliency_map(
            model=model,
            image_tensor=input_tensor,
            pred_index=pred_idx,
        )

        gradcam_overlay = overlay_heatmap_on_image(processed_image, gradcam, alpha=0.40)

        status = "correct" if true_idx == pred_idx else "incorrect"
        short_title = (
            f"{status} | T:{get_class_name(true_idx)} | "
            f"P:{get_class_name(pred_idx)} | conf={confidence:.3f}"
        )

        long_title = (
            f"{model_name} | #{rank:02d} | {status}\n"
            f"True: {get_class_name(true_idx)} | "
            f"Pred: {get_class_name(pred_idx)} | "
            f"Conf: {confidence:.4f}\n"
            f"{Path(filepath).name}"
        )

        save_path = panels_dir / f"{rank:02d}_{status}_{Path(filepath).stem}.png"
        save_explainability_panel(
            raw_image=raw_image,
            processed_image=processed_image,
            gradcam_overlay=gradcam_overlay,
            saliency_map=saliency,
            title=long_title,
            save_path=save_path,
        )

        panel_rows.append(
            {
                "rank": rank,
                "test_index": int(idx),
                "filepath": filepath,
                "filename": Path(filepath).name,
                "status": status,
                "true_label": true_idx,
                "pred_label": pred_idx,
                "true_name": get_class_name(true_idx),
                "pred_name": get_class_name(pred_idx),
                "confidence": confidence,
                "panel_path": str(save_path),
            }
        )

        summary_panels.append(
            {
                "gradcam_overlay": gradcam_overlay,
                "short_title": short_title,
            }
        )

    # -----------------------------------------------------
    # Mentések
    # -----------------------------------------------------
    panels_df = pd.DataFrame(panel_rows)
    panels_csv = Path(out_dir) / "explainability_examples.csv"
    panels_df.to_csv(panels_csv, index=False)

    summary_grid_path = Path(out_dir) / "gradcam_summary_grid.png"
    save_summary_grid(
        panel_infos=summary_panels,
        save_path=summary_grid_path,
        ncols=3,
    )

    metadata = {
        "model_name": model_name,
        "model_path": str(resolved_model_path),
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "last_conv_layer_name": last_conv_layer_name,
        "n_selected": int(len(selected_indices)),
        "selection": {
            "n_correct": int(n_correct),
            "n_incorrect": int(n_incorrect),
            "n_random": int(n_random),
        },
    }
    save_json(metadata, Path(out_dir) / "explainability_config.json")

    print(f"[INFO] Saved CSV: {panels_csv}")
    print(f"[INFO] Saved summary grid: {summary_grid_path}")
    print(f"[INFO] Saved config: {Path(out_dir) / 'explainability_config.json'}")

    return {
        "model_path": resolved_model_path,
        "out_dir": Path(out_dir),
        "panels_dir": panels_dir,
        "examples_csv": panels_csv,
        "summary_grid_png": summary_grid_path,
        "examples_df": panels_df,
        "last_conv_layer_name": last_conv_layer_name,
    }


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    run_explainability_qc(
        model_name="resnet50",
        split_dir=SPLITS_DIR,
    )