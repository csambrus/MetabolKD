# src/config.py
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import json
import os
import sys


# -------------------------------------------------
# Környezet felismerés
# -------------------------------------------------

IS_COLAB = "google.colab" in sys.modules
IS_RUNPOD = "RUNPOD_POD_ID" in os.environ
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# -------------------------------------------------
# Projekt gyökér
# -------------------------------------------------

if IS_COLAB:
    PROJECT_ROOT = Path("/content/CXR")

elif IS_RUNPOD:
    PROJECT_ROOT = Path("/workspace/CXR")

else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================
# Könyvtárak
# =========================================================
if IS_COLAB:
    DATA_DIR = Path("/content/drive/MyDrive/CXR/data")
    OUTPUT_DIR = Path("/content/drive/MyDrive/CXR/outputs")
else:
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
SPLITS_DIR = INTERIM_DIR / "splits"
MODELS_DIR = INTERIM_DIR / "models"

FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LOGS_DIR = OUTPUT_DIR / "logs"

# segmentation related dirs
SEGMENTATION_RAW_DIR = DATA_DIR / "segment_raw"
SEGMENTATION_DATA_DIR = INTERIM_DIR / "segmentation"
SEGMENTATION_MODELS_DIR = INTERIM_DIR / "segment_models"
SEGMENTATION_SPLITS_DIR = INTERIM_DIR / "segment_splits"
SEGMENTATION_OUTPUT_DIR = OUTPUT_DIR  / "segmentation"

LUNG_MASK_DIR   = SEGMENTATION_DATA_DIR / "lung_masks"
LUNG_MASKED_DIR = SEGMENTATION_DATA_DIR / "lung_masked"
LUNG_CROP_DIR   = SEGMENTATION_DATA_DIR / "lung_crop"

# -------------------------------------------------
# Info
# -------------------------------------------------

print("IS_COLAB:", IS_COLAB)
print("IS_RUNPOD:", IS_RUNPOD)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)


# =========================================================
# Dataset / osztályok
# =========================================================

@dataclass(frozen=True)
class ClassInfo:
    key: str
    raw_dir: str
    display_name: str
    idx: int

CLASS_INFOS = [
    ClassInfo("normal", "Normal", "Normal", 0),
    ClassInfo("pneumonia_viral", "Pneumonia-Viral", "Pneumonia-Viral", 1),
    ClassInfo("pneumonia_bacterial", "Pneumonia-Bacterial", "Pneumonia-Bacterial", 2),
    ClassInfo("covid19", "COVID-19", "COVID-19", 3),
]

NUM_CLASSES = len(CLASS_INFOS)
CLASS_BY_KEY = {c.key: c for c in CLASS_INFOS}
CLASS_BY_IDX = {c.idx: c for c in CLASS_INFOS}

# =========================================================
# Split CSV-k
# =========================================================
TRAIN_CSV = SPLITS_DIR / "train.csv"
VAL_CSV = SPLITS_DIR / "val.csv"
TEST_CSV = SPLITS_DIR / "test.csv"

# =========================================================
# Input image paraméterek
# =========================================================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# =========================================================
# Split arányok
# =========================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42
SHUFFLE_DATA = True

# =========================================================
# DataLoader / tf.data
# =========================================================
BATCH_SIZE =128
CACHE_DATASET = True
PREFETCH_DATASET = True

# =========================================================
# Preprocessing / augmentáció
# =========================================================
USE_MODEL_PREPROCESSING = True
NORMALIZE_TO_0_1 = True

# Később bővíthető:
USE_CLAHE = False
USE_GAUSSIAN_BLUR = False
USE_HIST_EQ = False

# =========================================================
# Training baseline
# =========================================================
EPOCHS = 15
LEARNING_RATE = 1e-4

# =========================================================
# Modellek
# =========================================================
BACKBONE_NAME = "DenseNet121"
USE_IMAGENET_WEIGHTS = True
FREEZE_BACKBONE = True

PLOT_DPI = 150
# =========================================================
# Utilities
# =========================================================
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Main project dirs:
for d in [DATA_DIR, RAW_DIR, INTERIM_DIR, OUTPUT_DIR, SPLITS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    ensure_dir(d)

def get_data_root(variant: str = "raw") -> Path:
    variant = variant.lower()
    if variant == "raw":
        return RAW_DIR
    if variant == "lung_mask":
        return LUNG_MASK_DIR
    if variant == "lung_masked":
        return LUNG_MASKED_DIR
    if variant == "lung_crop":
        return LUNG_CROP_DIR
    raise ValueError(f"Unknown data variant: {variant}")

def get_class_names() -> list[str]:
    return [c.display_name for c in CLASS_INFOS]

def get_class_name(idx: int) -> str:
    class_names = get_class_names()
    if 0 <= idx < len(class_names):
        return class_names[idx]
    return f"class_{idx}"

def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

