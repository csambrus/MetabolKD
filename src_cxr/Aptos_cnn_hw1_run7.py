#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
APTOS19 DR vs No DR — TensorFlow/Keras CNN pipeline

A script célja:
retina képek alapján eldönteni, hogy van-e diabéteszes retinopathia.

Két modell összehasonlítása:
1) "pretrained" : EfficientNetB0 - ImageNet-en előtanított modell
2) "scratch"    : saját CNN architektúra nulláról

Fő funkciók:
- 4-fold rétegzett keresztvalidáció
- retina képek előfeldolgozása
- training képek augmentációja
- ROC és confusion matrix számítás
- modellek összehasonlítása
- hibás klasszifikációk mentése átnézéshez
"""

import os
import glob
import json
import math
import shutil
import random

import numpy as np
import pandas as pd
import seaborn as sns
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, friedmanchisquare

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)


# ============================================================
# KONFIGURÁCIÓ
# ============================================================

RUN_IMAGE_CHECK = True
USE_AUGMENTATION = True
RESUME_IF_SAVED = True

RUN_PREPROCESS = True
RUN_PREPROCESS_CHECK = True
USE_PREPROCESS = True

SAVE_PLOTS = True
SHOW_PLOTS = True

IMG_SIZE   = (224, 224)
BATCH_SIZE = 8
EPOCHS     = 6
N_SPLITS   = 4
THRESHOLD  = 0.5

MODEL_KINDS = ["pretrained", "scratch"]
CLASS_NAMES = ("No DR", "DR")

PROJECT_DIR = "/media/acsaba/OS/Aptos_19_diabretinopathy/"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
PREP_DIR    = os.path.join(PROJECT_DIR, "preprocessed")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results_v_final")

MAX_COPY_MISCLASS_PER_FOLD = 8
MISCLASS_MONTAGE_N         = 8

SEED       = 124

# ============================================================
# ÁLTALÁNOS SEGÉDFÜGGVÉNYEK
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    """
    Training history és metrikák mentése JSON formátumba
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history, path_json, path_csv=None):
    """
    A Keras training history mentése: loss, accuracy, val_loss, val_accuracy, stb.
    """
    hist = history.history if hasattr(history, "history") else history
    save_json(hist, path_json)
    if path_csv is not None:
        pd.DataFrame(hist).to_csv(path_csv, index=False)


def load_history(path_json):
    return load_json(path_json)


def save_roc_curve_npz(fpr, tpr, roc_auc, fold, path):
    """
    ROC görbe adatok mentése: fpr, tpr, auc
    to NumPy NPZ file
    """
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(
        path,
        fpr=np.asarray(fpr, dtype=float),
        tpr=np.asarray(tpr, dtype=float),
        auc=float(roc_auc),
        fold=int(fold),
    )


def load_roc_curve_npz(path):
    z = np.load(path, allow_pickle=False)
    return {
        "fpr": z["fpr"],
        "tpr": z["tpr"],
        "auc": float(z["auc"]),
        "fold": int(z["fold"]),
    }


def save_or_show(fig, out_path=None, dpi=150):
    if SAVE_PLOTS and out_path is not None:
        ensure_dir(os.path.dirname(out_path))
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ============================================================
# KÖRNYEZET BEÁLLÍTÁSA
# ============================================================

def setup_environment(seed=SEED):
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU-k száma:", len(gpus))  # csak az egyik gépemen :(
    
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    ensure_dir(RESULTS_DIR)


# ============================================================
# OUTPUT DIRz - modell függő !
# ============================================================

def make_output_dirs(model_kind: str):
    out = {}
    out["ROOT"]         = os.path.join(RESULTS_DIR, model_kind)
    out["MODEL_DIR"]    = os.path.join(out["ROOT"], "models")
    out["ROC_DIR"]      = os.path.join(out["ROOT"], "roc_curves")
    out["CM_DIR"]       = os.path.join(out["ROOT"], "confusion_matrices")
    out["CURVES_DIR"]   = os.path.join(out["ROOT"], "training_curves")
    out["MISCLASS_DIR"] = os.path.join(out["ROOT"], "misclassified")
    out["TABLES_DIR"]   = os.path.join(out["ROOT"], "tables")
    out["FOLD_DIR"]     = os.path.join(out["ROOT"], "fold_state")

    for d in out.values():
        ensure_dir(d)
    return out


# ============================================================
# SAVE & RESTORE CHECKPOINT
# ============================================================

def get_fold_ckpt_dir(out_dirs, model_kind, fold):
    return os.path.join(out_dirs["MODEL_DIR"], f"ckpt_fold_{fold}_{model_kind}")

def check_fold_done(out_dirs, model_kind, fold):
    path = os.path.join(out_dirs["FOLD_DIR"], f"fold_{fold}_{model_kind}_done.txt")
    return os.path.exists(path)

def mark_fold_done(out_dirs, model_kind, fold):
    path = os.path.join(out_dirs["FOLD_DIR"], f"fold_{fold}_{model_kind}_done.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("done\n")

def restore_or_init_checkpoint(model, optimizer, ckpt_dir):
    """
    Visszatölt checkpointot, ha van.
    Visszaad:
      - initial_epoch
      - CheckpointManager
    """
    ensure_dir(ckpt_dir)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2)

    initial_epoch = 0

    if RESUME_IF_SAVED and manager.latest_checkpoint:
        print("Checkpoint visszatöltése:", manager.latest_checkpoint)
        ckpt.restore(manager.latest_checkpoint).expect_partial()

        epoch_file = os.path.join(ckpt_dir, "last_epoch.txt")
        if os.path.exists(epoch_file):
            try:
                initial_epoch = int(open(epoch_file, "r", encoding="utf-8").read().strip())
                print("Folytatás ettől az epochtól:", initial_epoch)
            except Exception:
                initial_epoch = 0

    return initial_epoch, manager


class SaveTFCheckpointCallback(keras.callbacks.Callback):
    """
    Minden epoch végén ment:
      - TensorFlow checkpoint
      - utolsó epoch száma
    """
    def __init__(self, manager, ckpt_dir):
        super().__init__()
        self.manager = manager
        self.ckpt_dir = ckpt_dir
        self.epoch_file = os.path.join(ckpt_dir, "last_epoch.txt")

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save()
        with open(self.epoch_file, "w", encoding="utf-8") as f:
            f.write(str(epoch + 1))


# ============================================================
# RETINA PREPROCESSING
# ============================================================
def preprocess_retina(img_rgb_uint8, tol=10):
    """
        1: fekete keret levágása
        2: kör alakú maszk
        3: kontraszt korrekció
    """
    img = img_rgb_uint8

    # Szürkeárnyalatos kép a háttér / fekete keret felismeréséhez
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    # Fekete keret levágása a nem-fekete pixelek alapján
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img[y0:y1, x0:x1]

    # Kör alakú maszk a retina középső részére
    h, w = img.shape[:2]
    r = min(h, w) // 2
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    circle = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

    masked = np.zeros_like(img)
    masked[circle] = img[circle]
    img = masked

    # Illumination / kontraszt korrekció
    blur = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blur, -4, 128)

    # Biztonsági levágás 0..255 tartományra
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def preprocess_and_save(src_path, dst_path, out_size=(224, 224), overwrite=False):
    """
    Egy kép beolvasása, retina preprocesszálása, átméretezése és mentése.
    """
    if (not overwrite) and os.path.exists(dst_path):
        return "skip"

    out_dir = os.path.dirname(dst_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return "error_read"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if USE_PREPROCESS:
        img_rgb = preprocess_retina(img_rgb)

    if img_rgb is None or img_rgb.size == 0:
        return "error_preprocess"

    img_rgb = cv2.resize(img_rgb, out_size, interpolation=cv2.INTER_AREA)
    img_bgr_out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(dst_path, img_bgr_out)

    if not ok:
        return "error_write"

    if not os.path.exists(dst_path):
        return "error_missing"

    return "ok"

def preprocess_dataset_to_disk(
    src_root,
    dst_root,
    out_size=(224, 224),
    overwrite=False
):
    class_names = ["DR", "No DR"]

    n_ok = 0
    n_skip = 0
    n_error = 0

    for cls in class_names:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_dir, exist_ok=True)

        files = []
        files.extend(glob.glob(os.path.join(src_dir, "*.png")))
        files = sorted(files)

        print(f"\n[{cls}] képek száma: {len(files)}")

        for src_path in tqdm(files, desc=f"Preprocess {cls}"):
            base = os.path.basename(src_path)

            # Célszerű egységesen PNG-be menteni
            stem = os.path.splitext(base)[0]
            dst_path = os.path.join(dst_dir, stem + ".png")

            status = preprocess_and_save(
                src_path=src_path,
                dst_path=dst_path,
                out_size=out_size,
                overwrite=overwrite
            )

            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_error += 1

    print("\nKész.")
    print(f"Mentve:   {n_ok}")
    print(f"Kihagyva: {n_skip}")
    print(f"Hibás:    {n_error}")
    
    
# ============================================================
# KÉPBEOLVASÁS
# ============================================================

def list_dataset(data_dir: str):
    """
    Képfájlok listázása könyvtárból.
        data/DR/*
        data/No DR/*
    """

    dr = glob.glob(os.path.join(data_dir, "DR", "*.png"))
    nodr = glob.glob(os.path.join(data_dir, "No DR", "*.png"))

    items = np.array(nodr + dr)
    labels = np.array([0]*len(nodr) + [1]*len(dr), dtype=np.int32)

    print(f"Képek száma: összes={len(items)} | No DR={len(nodr)} | DR={len(dr)}")

    return items, labels


def load_image_preview(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3) # channels: x, y, rgb
    return img.numpy()

# ============================================================
# Retina nézegető
# ============================================================
def check_images(data_dir: str, n_examples=4, seed=SEED):
    rng = np.random.default_rng(seed)

    items, labels = list_dataset(data_dir)

    dr_items = items[labels == 1]
    nodr_items = items[labels == 0]

    dr_pick = rng.choice(dr_items, size=min(n_examples, len(dr_items)), replace=False)
    nodr_pick = rng.choice(nodr_items, size=min(n_examples, len(nodr_items)), replace=False)

    fig, axes = plt.subplots(4, n_examples, figsize=(3 * n_examples, 10))

    for i, it in enumerate(dr_pick):
        img = load_image_preview(str(it))
        img_p = preprocess_retina(img) if USE_PREPROCESS else img

        axes[0, i].imshow(img)
        axes[0, i].set_title("DR eredeti")
        axes[0, i].axis("off")

        axes[1, i].imshow(img_p)
        axes[1, i].set_title("DR preprocess")
        axes[1, i].axis("off")

    for i, it in enumerate(nodr_pick):
        img = load_image_preview(str(it))
        img_p = preprocess_retina(img) if USE_PREPROCESS else img

        axes[2, i].imshow(img)
        axes[2, i].set_title("No DR eredeti")
        axes[2, i].axis("off")

        axes[3, i].imshow(img_p)
        axes[3, i].set_title("No DR preprocess")
        axes[3, i].axis("off")

    plt.suptitle(
        f"Ellenőrző képek — preprocess={USE_PREPROCESS}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def compare_original_vs_preprocessed(
    src_root,
    prep_root,
    n_examples=6,
    seed=42
):
    rng = np.random.default_rng(seed)

    classes = ["DR", "No DR"]

    fig, axes = plt.subplots(
        len(classes) * 2,
        n_examples,
        figsize=(3 * n_examples, 6)
    )

    for ci, cls in enumerate(classes):

        src_dir = os.path.join(src_root, cls)
        prep_dir = os.path.join(prep_root, cls)

        files = sorted(glob.glob(os.path.join(src_dir, "*")))

        if len(files) == 0:
            print(f"Nincs kép: {src_dir}")
            continue

        picks = rng.choice(files, size=min(n_examples, len(files)), replace=False)

        for i, src_path in enumerate(picks):

            base = os.path.basename(src_path)
            stem = os.path.splitext(base)[0]

            prep_path = os.path.join(prep_dir, stem + ".png")

            img_orig = cv2.imread(src_path)
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

            if os.path.exists(prep_path):
                img_prep = cv2.imread(prep_path)
                img_prep = cv2.cvtColor(img_prep, cv2.COLOR_BGR2RGB)
            else:
                img_prep = np.zeros_like(img_orig)

            r0 = ci * 2
            r1 = ci * 2 + 1

            axes[r0, i].imshow(img_orig)
            axes[r0, i].set_title(f"{cls} eredeti")
            axes[r0, i].axis("off")

            axes[r1, i].imshow(img_prep)
            axes[r1, i].set_title("preprocess")
            axes[r1, i].axis("off")

    plt.suptitle("Eredeti vs preprocesszált retina képek", fontsize=14)
    plt.tight_layout()
    plt.show()
    
# ============================================================
# Tensorflow pipeline
# ============================================================
def create_data_pipeline(img_size, batch_size):

    def tf_image_loader(path, label):
        img = tf.io.read_file(path)

        # PNG / JPG / JPEG automatikus dekódolás
        img = tf.io.decode_image(img, channels=3, expand_animations=False)

        # decode_image után állítsuk be a shape-et
        img.set_shape([None, None, 3])

        img = tf.image.resize(img, img_size, method="bilinear")
        img = tf.cast(img, tf.float32)
        label = tf.cast(label, tf.float32)

        return img, label

    def make_ds(items, y, training=False):
        items = np.asarray(items).astype(str)
        y = np.asarray(y).astype(np.float32)

        ds = tf.data.Dataset.from_tensor_slices((items, y))

        if training:
            ds = ds.shuffle(len(items), seed=SEED, reshuffle_each_iteration=True)

        ds = ds.map(tf_image_loader, num_parallel_calls=tf.data.AUTOTUNE)

        if not training:
            ds = ds.cache()
            #ds = ds.cache("/tmp/tf_val_cache")

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    return make_ds
# ============================================================
# ***** MODELLz *****
# ============================================================

def build_augment():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="augment")
    
def build_model_pretrained(img_size):

    # Az augmentáció csak training alatt lesz aktív !
    augment = build_augment()

    # EfficientNet preprocess - kötelező normalizáció !
    preprocess = tf.keras.applications.efficientnet.preprocess_input

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=img_size + (3,)
    )

    # Ne legyen újra tanítva!
    base.trainable = False

    model = keras.Sequential([
        layers.Input(shape=img_size + (3,)),
        augment,
        layers.Lambda(preprocess),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.30),
        layers.Dense(1, activation="sigmoid")
    ], name="EfficientNetB0_pretrained")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def build_model_from_scratch(img_size):

    # Az augmentáció csak training alatt lesz aktív !
    augment = build_augment()

    model = keras.Sequential([
        layers.Input(shape=img_size + (3,)),
        augment,

        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.40),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(1, activation="sigmoid"),
    ], name="CNN_from_scratch")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_model(img_size, model_kind: str):
    if model_kind == "pretrained":
        return build_model_pretrained(img_size)
    elif model_kind == "scratch":
        return build_model_from_scratch(img_size)
    else:
        raise ValueError("model_kind csak 'scratch' vagy 'pretrained' lehet.")


# ============================================================
# *** Plotz (3) ***
# ============================================================

def plot_training_curves(history, fold, out_dirs, model_kind):
    """
    Egyetlen közös 3 paneles ábra:
      1) Accuracy
      2) Loss
      3) AUC
    """
    h = history.history if hasattr(history, "history") else history
    n_epochs = len(h.get("loss", []))
    epochs = range(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # --------------------------------------------------------
    # 1. Accuracy
    # --------------------------------------------------------
    ax = axes[0]
    has_acc = False

    if "acc" in h:
        ax.plot(epochs, h["acc"], label="Training", linewidth=2)
        has_acc = True
    if "val_acc" in h:
        ax.plot(epochs, h["val_acc"], label="Validation", linewidth=2)
        has_acc = True

    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)
    if has_acc:
        ax.legend()
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "Nincs accuracy adat", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # --------------------------------------------------------
    # 2. Loss
    # --------------------------------------------------------
    ax = axes[1]
    has_loss = False

    if "loss" in h:
        ax.plot(epochs, h["loss"], label="Training", linewidth=2)
        has_loss = True
    if "val_loss" in h:
        ax.plot(epochs, h["val_loss"], label="Validation", linewidth=2)
        has_loss = True

    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    if has_loss:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Nincs loss adat", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # --------------------------------------------------------
    # 3. AUC
    # --------------------------------------------------------
    ax = axes[2]
    has_auc = False

    if "auc" in h:
        ax.plot(epochs, h["auc"], label="Training", linewidth=2)
        has_auc = True
    if "val_auc" in h:
        ax.plot(epochs, h["val_auc"], label="Validation", linewidth=2)
        has_auc = True

    ax.set_title("AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.25)
    if has_auc:
        ax.legend()
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "Nincs AUC adat", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(f"Tanulási görbék — fold {fold} ({model_kind})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_or_show(
        fig,
        os.path.join(out_dirs["CURVES_DIR"], f"fold_{fold}_training_curves_3panel_{model_kind}.png")
    )

def plot_confusion_matrix(cm, fold, out_dirs, model_kind, class_names=CLASS_NAMES):
    fig, ax = plt.subplots(figsize=(6, 5))

    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    annot = np.empty_like(cm).astype(str)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f"{count}\n{pct:.1f}%"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )

    ax.set_title(f"Konfúziós mátrix — fold {fold} ({model_kind})", fontsize=14)
    ax.set_ylabel("Valódi osztály")
    ax.set_xlabel("Becsült osztály")

    save_or_show(fig, os.path.join(out_dirs["CM_DIR"], f"cm_fold_{fold}.png"))


def plot_roc_per_fold(fpr, tpr, fold_auc, fold, out_dirs, model_kind):
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"Fold {fold} AUC={fold_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Hamis pozitív arány")
    plt.ylabel("Valódi pozitív arány")
    plt.title(f"ROC — fold {fold} ({model_kind})")
    plt.legend(loc="lower right")
    save_or_show(fig, os.path.join(out_dirs["ROC_DIR"], f"roc_fold_{fold}.png"))


def plot_roc_all_folds_with_mean(roc_data, out_dirs, model_kind):
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    aucs = []

    fig = plt.figure()

    for rd in (roc_data or []):
        fpr = np.asarray(rd.get("fpr", []), dtype=float)
        tpr = np.asarray(rd.get("tpr", []), dtype=float)
        fa = rd.get("auc", np.nan)
        fold = rd.get("fold", "?")

        if fpr.ndim != 1 or tpr.ndim != 1 or len(fpr) < 2 or len(tpr) < 2:
            continue

        plt.plot(fpr, tpr, alpha=0.35, label=f"Fold {fold} AUC={fa:.3f}")
        if np.isfinite(fa):
            aucs.append(float(fa))

        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

    if len(tprs) == 0:
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Hamis pozitív arány")
        plt.ylabel("Valódi pozitív arány")
        plt.title(f"ROC — ({model_kind}) [nincs mentett ROC görbe]")
        save_or_show(fig, os.path.join(out_dirs["ROC_DIR"], f"roc_all_folds_mean_{model_kind}.png"))
        print(f"[WARN] Nincs érvényes mentett ROC görbe ({model_kind})")
        return

    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = float(np.std(aucs)) if len(aucs) else float("nan")

    plt.plot(
        mean_fpr,
        mean_tpr,
        linewidth=2.0,
        label=f"Átlag ROC AUC={mean_auc:.3f} (fold szórás={std_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Hamis pozitív arány")
    plt.ylabel("Valódi pozitív arány")
    plt.title(f"ROC — összes fold + átlag ({model_kind})")
    plt.legend(loc="lower right", fontsize=9)

    save_or_show(fig, os.path.join(out_dirs["ROC_DIR"], f"roc_all_folds_mean_{model_kind}.png"))

def make_misclassified_montage(mis_records, fold, out_dirs, model_kind, max_n=36):
    if len(mis_records) == 0:
        return

    use = mis_records[:min(max_n, len(mis_records))]
    n = len(use)
    cols = max(1, int(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.atleast_2d(axes)

    for idx, rec in enumerate(use):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        img = load_image_preview(rec["item"])

        ax.imshow(img)
        ax.set_title(
            f'{rec["tag"]} V{rec["true"]}→B{rec["pred"]}\n{rec["prob"]:.2f}',
            fontsize=8
        )
        ax.axis("off")

    for idx in range(len(use), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle(f"Hibás klasszifikációk — fold {fold} ({model_kind})", fontsize=14)
    save_or_show(fig, os.path.join(out_dirs["MISCLASS_DIR"], f"fold_{fold}_montage_{model_kind}.png"))

# ============================================================
# Sikertelen predikciók képei
# ============================================================

def collect_misclassified(x_te, y_te, y_pred, y_prob):
    mis_idx = np.where(y_pred != y_te)[0]
    fold_mis = []

    for i in mis_idx:
        true = int(y_te[i])
        pred = int(y_pred[i])
        prob = float(y_prob[i])
        item = str(x_te[i])
        tag = "FP" if (true == 0 and pred == 1) else "FN"

        fold_mis.append({
            "item": item,
            "true": true,
            "pred": pred,
            "prob": prob,
            "tag": tag,
        })
    return fold_mis

def save_misclassified_files(fold_mis, fold, out_dirs):
    if MAX_COPY_MISCLASS_PER_FOLD <= 0 or len(fold_mis) == 0:
        return

    out_dir = os.path.join(out_dirs["MISCLASS_DIR"], f"fold_{fold}")
    ensure_dir(out_dir)

    for rec in fold_mis[:MAX_COPY_MISCLASS_PER_FOLD]:
        base = os.path.basename(rec["item"])
        newname = f'{rec["tag"]}_true{rec["true"]}_pred{rec["pred"]}_p{rec["prob"]:.2f}_{base}'
        out_path = os.path.join(out_dir, newname)
        shutil.copy2(rec["item"], out_path)

# ============================================================
# EGY FOLD FUTTATÁSA
# ============================================================

def run_one_fold(fold, x_tr, y_tr, x_te, y_te, make_ds_fn, out_dirs, model_kind):
    train_ds = make_ds_fn(x_tr, y_tr, training=True)
    test_ds = make_ds_fn(x_te, y_te, training=False)

    model = build_model(IMG_SIZE, model_kind)
    optimizer = model.optimizer

    ckpt_dir = get_fold_ckpt_dir(out_dirs, model_kind, fold)
    initial_epoch, manager = restore_or_init_checkpoint(model, optimizer, ckpt_dir)

    fold_model_path = os.path.join(out_dirs["MODEL_DIR"], f"model_fold_{fold}_{model_kind}.keras")

    callbacks = [
        SaveTFCheckpointCallback(manager, ckpt_dir),
        keras.callbacks.ModelCheckpoint(
            filepath=fold_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # History mentése
    hist_json_path = os.path.join(out_dirs["FOLD_DIR"], f"history_fold_{fold}_{model_kind}.json")
    hist_csv_path = os.path.join(out_dirs["FOLD_DIR"], f"history_fold_{fold}_{model_kind}.csv")
    save_history(history, hist_json_path, hist_csv_path)

    plot_training_curves(history, fold, out_dirs, model_kind)

    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_te, y_pred)
    plot_confusion_matrix(cm, fold, out_dirs, model_kind)

    acc = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, zero_division=0)
    recall = recall_score(y_te, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_te, y_prob)
    fold_auc = auc(fpr, tpr)
    plot_roc_per_fold(fpr, tpr, fold_auc, fold, out_dirs, model_kind)

    roc_npz_path = os.path.join(out_dirs["FOLD_DIR"], f"roc_fold_{fold}_{model_kind}.npz")
    save_roc_curve_npz(fpr, tpr, fold_auc, fold, roc_npz_path)

    print(f"\nOsztályozási riport — fold {fold}, modell: {model_kind}")
    print(classification_report(y_te, y_pred, target_names=list(CLASS_NAMES), digits=3))

    fold_mis = collect_misclassified(x_te, y_te, y_pred, y_prob)
    save_misclassified_files(fold_mis, fold, out_dirs)
    make_misclassified_montage(fold_mis, fold, out_dirs, model_kind, max_n=MISCLASS_MONTAGE_N)

    mis_csv_path = os.path.join(out_dirs["FOLD_DIR"], f"misclassified_fold_{fold}_{model_kind}.csv")
    pd.DataFrame(fold_mis).to_csv(mis_csv_path, index=False)

    fold_metrics = {
        "model_kind": model_kind,
        "fold": fold,
        "auc": float(fold_auc),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "model_path": fold_model_path,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    metrics_json_path = os.path.join(out_dirs["FOLD_DIR"], f"metrics_fold_{fold}_{model_kind}.json")
    save_json(fold_metrics, metrics_json_path)

    return {
        "fold_metrics": fold_metrics,
        "roc": {"fold": fold, "fpr": fpr, "tpr": tpr, "auc": float(fold_auc)},
        "misclassified": fold_mis,
    }


# ============================================================
# CROSS-VALIDATION EGY MODELLRE
# ============================================================

def run_cv_for_model(items, labels, model_kind):
    out_dirs = make_output_dirs(model_kind)

    make_ds_fn = create_data_pipeline(IMG_SIZE, BATCH_SIZE)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_rows = []
    roc_data = []
    all_mis = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(items, labels), start=1):
        print(f"\n===== MODELL: {model_kind} | FOLD {fold}/{N_SPLITS} =====")

        x_tr, y_tr = items[tr_idx], labels[tr_idx]
        x_te, y_te = items[te_idx], labels[te_idx]

        metrics_json_path = os.path.join(out_dirs["FOLD_DIR"], f"metrics_fold_{fold}_{model_kind}.json")
        hist_json_path = os.path.join(out_dirs["FOLD_DIR"], f"history_fold_{fold}_{model_kind}.json")
        roc_npz_path = os.path.join(out_dirs["FOLD_DIR"], f"roc_fold_{fold}_{model_kind}.npz")
        mis_csv_path = os.path.join(out_dirs["FOLD_DIR"], f"misclassified_fold_{fold}_{model_kind}.csv")

        if check_fold_done(out_dirs, model_kind, fold):
            print("⏭ Fold már kész — mentett eredmények visszatöltése")

            if os.path.exists(metrics_json_path):
                fold_metrics = load_json(metrics_json_path)
                fold_rows.append(fold_metrics)
            
                # --- Confusion matrix újrarajzolása a mentett TN/FP/FN/TP alapján ---
                cm_keys = {"tn", "fp", "fn", "tp"}
                if cm_keys.issubset(fold_metrics.keys()):
                    cm = np.array([
                        [int(fold_metrics["tn"]), int(fold_metrics["fp"])],
                        [int(fold_metrics["fn"]), int(fold_metrics["tp"])],
                    ])
                    plot_confusion_matrix(cm, fold, out_dirs, model_kind)
            
            if os.path.exists(hist_json_path):
                hist = load_history(hist_json_path)
                plot_training_curves(hist, fold, out_dirs, model_kind)

            if os.path.exists(roc_npz_path):
                rd = load_roc_curve_npz(roc_npz_path)
                roc_data.append(rd)

            if os.path.exists(mis_csv_path):
                mis_df = pd.read_csv(mis_csv_path)
                if len(mis_df) > 0:
                    for _, r in mis_df.iterrows():
                        all_mis.append({
                            "fold": fold,
                            "item": r["item"],
                            "true": int(r["true"]),
                            "pred": int(r["pred"]),
                            "prob": float(r["prob"]),
                            "tag": r["tag"],
                            "model_kind": model_kind,
                        })
            continue

        fr = run_one_fold(fold, x_tr, y_tr, x_te, y_te, make_ds_fn, out_dirs, model_kind)
        mark_fold_done(out_dirs, model_kind, fold)

        fold_rows.append(fr["fold_metrics"])
        roc_data.append(fr["roc"])
        all_mis.extend([{**m, "model_kind": model_kind, "fold": fold} for m in fr["misclassified"]])

    plot_roc_all_folds_with_mean(roc_data, out_dirs, model_kind)

    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(out_dirs["TABLES_DIR"], f"fold_metrics_{model_kind}.csv"), index=False)

    pd.DataFrame(all_mis).to_csv(
        os.path.join(out_dirs["TABLES_DIR"], f"misclassified_{model_kind}.csv"),
        index=False
    )

    return df_folds


# ============================================================
# MODELLEK ÖSSZEHASONLÍTÁSA
# ============================================================

def compare_models(df_all_folds, out_root=RESULTS_DIR):
    """
    Modellek összehasonlítása cross-validation fold eredmények alapján.
    - 2 modell esetén: páros t-próba
    - 3 vagy több modell esetén: Friedman-próba
    """

    ensure_dir(out_root)

    required_cols = {"fold", "model_kind", "auc", "acc", "precision", "recall"}
    missing = required_cols - set(df_all_folds.columns)
    if missing:
        raise ValueError(f"Hiányzó oszlopok a df_all_folds táblából: {sorted(missing)}")

    metrics = [
        ("auc", "AUC"),
        ("acc", "Accuracy"),
        ("precision", "Precizitás"),
        ("recall", "Recall / szenzitivitás"),
    ]

    # ------------------------------------------------------------
    # Összefoglaló tábla: átlag ± szórás
    # ------------------------------------------------------------

    sum_tbl = (
        df_all_folds
        .groupby("model_kind")
        .agg(
            auc_mean=("auc","mean"),
            auc_std=("auc","std"),
            acc_mean=("acc","mean"),
            acc_std=("acc","std"),
            precision_mean=("precision","mean"),
            precision_std=("precision","std"),
            recall_mean=("recall","mean"),
            recall_std=("recall","std"),
        )
        .reset_index()
    )

    out_csv = os.path.join(out_root, "compare_models.csv")
    sum_tbl.to_csv(out_csv, index=False)

    print("\n=== MODELLEK ÖSSZEHASONLÍTÁSA (átlag ± szórás) ===")
    # print(sum_tbl)
    print(sum_tbl.to_string(index=False))
    print("\nMentve:", out_csv)

    # ------------------------------------------------------------
    # Segédfüggvények
    # ------------------------------------------------------------
    def _mean_std(metric):
        g = df_all_folds.groupby("model_kind")[metric]
        return g.mean(), g.std()

    def _p_to_text(p):
        if pd.isna(p):
            return "p = NA"
        if p < 0.001:
            return "p < 0.001"
        return f"p = {p:.3f}"

    def _run_stat_test(metric):
        tmp = df_all_folds[["fold", "model_kind", metric]].dropna().copy()

        pivot = tmp.pivot(index="fold", columns="model_kind", values=metric)

        # Csak teljes párosítású foldok maradjanak
        pivot = pivot.dropna(axis=0, how="any")

        n_models = pivot.shape[1]
        n_folds = pivot.shape[0]

        if n_models < 2:
            return {
                "test_name": "nincs elég modell",
                "stat": np.nan,
                "p": np.nan,
                #"stat_name": None,
                "note": "Nincs elég modell a statisztikai összehasonlításhoz."
            }

        if n_folds < 2:
            return {
                "test_name": "nincs elég fold",
                "stat": np.nan,
                "p": np.nan,
                #"stat_name": None,
                "note": "Nincs elég fold a statisztikai teszthez."
            }

        if n_models == 2:
            m1, m2 = pivot.columns.tolist()
            stat, p = ttest_rel(pivot[m1], pivot[m2])

            return {
                "test_name": "páros t-próba",
                #"stat_name": "t",
                "stat": stat,
                "p": p,
                "note": (
                  f"Teszt: páros t-próba "
                  f"(azonos foldokon, {m1} vs. {m2}). {_p_to_text(p)}."
                )
            }

        # 3 vagy több modell esetén
        arrays = [pivot[col].values for col in pivot.columns]
        stat, p = friedmanchisquare(*arrays)

        return {
            "test_name": "Friedman-próba",
            #"stat_name": "χ²",
            "stat": stat,
            "p": p,
            "note": (
              f"Teszt: Friedman-próba "
              f"({n_models} modell, azonos {n_folds} fold). {_p_to_text(p)}."
            )
        }

    model_order = df_all_folds["model_kind"].dropna().unique().tolist()
    cmap = plt.get_cmap("tab10")
    color_map = {mk: cmap(i % cmap.N) for i, mk in enumerate(model_order)}

    def _bar(ax, means, stds, title, stat_res, ylim=(0, 1.05)):
        x = np.arange(len(means.index))
        colors = [color_map[m] for m in means.index]

        bars = ax.bar(
            x,
            means.values,
            yerr=stds.values,
            capsize=6,
            color=colors,
            width = 0.6,
            edgecolor="black",
            linewidth=0.5
        )

        ax.set_xticks(x)
        ax.set_xticklabels(means.index.tolist(), rotation=0, ha="center")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)

        for rect, val in zip(bars, means.values):
            y = min(rect.get_height() + 0.015, ylim[1] - 0.02)
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                y,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ptxt = _p_to_text(stat_res["p"])
        test_name = stat_res["test_name"]
        ax.text(
            0.02, 0.02,
            f"{test_name}\n{ptxt}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray")
        )

    # ------------------------------------------------------------
    # Statisztikai tesztek
    # ------------------------------------------------------------
    stat_rows = []
    stat_results = {}

    for mkey, mtitle in metrics:
        res = _run_stat_test(mkey)
        stat_results[mkey] = res

        stat_rows.append({
            #"metric": mkey,
            "metric_label": mtitle,
            "stat": res["stat"],
            "p_value": res["p"],
            "test_name": res["test_name"],
            #"stat_name": res["stat_name"],
            #"note": res["note"],
        })

    stat_tbl = pd.DataFrame(stat_rows)
    out_stats_csv = os.path.join(out_root, "compare_models_stats.csv")
    stat_tbl.to_csv(out_stats_csv, index=False)

    print("\n=== STATISZTIKAI ÖSSZEHASONLÍTÁS ===")
    print(stat_tbl)
    print("\nMentve:", out_stats_csv)

    # ------------------------------------------------------------
    # 4 paneles ábra
    # ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()

    for ax, (mkey, mtitle) in zip(axes, metrics):
        means, stds = _mean_std(mkey)
        _bar(
            ax,
            means,
            stds,
            f"{mtitle} (átlag ± szórás)",
            stat_res=stat_results[mkey],
            ylim=(0, 1.05)
        )
        ax.set_ylabel(mtitle)

    fig.suptitle("Modellek összehasonlítása (4-fold CV)", fontsize=14)

    notes = " | ".join(
        [f"{mtitle}: {stat_results[mkey]['test_name']}, {_p_to_text(stat_results[mkey]['p'])}"
         for mkey, mtitle in metrics]
    )

    fig.text(
        0.01, -0.03,
        f"Megjegyzés: {notes}",
        ha="left", va="bottom", fontsize=9
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_or_show(fig, os.path.join(out_root, "compare_models_4metrics.png"))


# ============================================================
# ***** MAIN *****
# ============================================================

def main():
    setup_environment(SEED)

    # Ellenőrző mintaképek
    if RUN_IMAGE_CHECK:
        check_images(DATA_DIR)

    if RUN_PREPROCESS:
        print("Dataset preprocess indul...")

        preprocess_dataset_to_disk(
            src_root=DATA_DIR,
            dst_root=PREP_DIR,
            out_size=IMG_SIZE,
            overwrite=False
        )
    
        if RUN_PREPROCESS_CHECK:
            print("Preprocess ellenőrzés...")
    
            compare_original_vs_preprocessed(
                src_root=DATA_DIR,
                prep_root=PREP_DIR,
                n_examples=6
            )

    if RUN_PREPROCESS or USE_PREPROCESS:
        if not os.path.exists(PREP_DIR):
            raise RuntimeError("Preprocessed dataset nem létezik. Futtasd a RUN_PREPROCESS lépést!")        
        imgdir = PREP_DIR
    else:
        imgdir = DATA_DIR
    
    print("Dataset betöltése...")
    items, labels = list_dataset(imgdir)

    # Modellek futtatása
    all_folds = []
    for mk in MODEL_KINDS:
        df_folds = run_cv_for_model(items, labels, mk)
        all_folds.append(df_folds)

    df_all = pd.concat(all_folds, ignore_index=True)
    df_all.to_csv(os.path.join(RESULTS_DIR, "all_fold_metrics_both_models.csv"), index=False)

    compare_models(df_all, out_root=RESULTS_DIR)

    print("\nKész. Az eredmények itt vannak:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()


# In[ ]:




