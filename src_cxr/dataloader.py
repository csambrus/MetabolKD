# dataloader.py

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    CACHE_DATASET,
    CLASS_INFOS,
    IMAGE_SIZE,
    RAW_DIR,
    SEED,
    SPLITS_DIR,
    ensure_dir,
)

AUTOTUNE = tf.data.AUTOTUNE

SPLIT_COLUMNS = ["relative_path", "filename", "class_key", "class_name", "label"]
REQUIRED_SPLIT_COLUMNS = set(SPLIT_COLUMNS)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# =========================================================
# Path helpers
# =========================================================

def resolve_image_path(root_dir: str | Path, relative_path: str) -> str:
    return str(Path(root_dir) / str(relative_path))


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# =========================================================
# CLASS_INFOS compatibility helpers
# =========================================================

def _class_info_to_dict(class_info: Any) -> dict[str, Any]:
    """
    CLASS_INFOS többféle alakját is kezeli:
    - dataclass
    - namedtuple
    - egyszerű objektum attribútumokkal
    - tuple/list: (class_key, raw_dir, display_name/class_name, label)

    A korábbi hiba oka az volt, hogy a kód csak néhány mezőnevet ismert
    (class_name/name, label/label_index). Ha a config-ban display_name vagy
    más név szerepelt, akkor üres class_name és -1 label került a CSV-be.
    """
    if is_dataclass(class_info):
        return asdict(class_info)

    if hasattr(class_info, "_asdict"):
        return dict(class_info._asdict())

    if isinstance(class_info, dict):
        return dict(class_info)

    if isinstance(class_info, (tuple, list)):
        keys = ["class_key", "raw_dir", "display_name", "label"]
        return {k: class_info[i] for i, k in enumerate(keys) if i < len(class_info)}

    # Általános objektum: csak a publikus attribútumokat vesszük.
    return {
        k: getattr(class_info, k)
        for k in dir(class_info)
        if not k.startswith("_") and not callable(getattr(class_info, k))
    }


def _first_present(data: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return default


def get_class_key(class_info: Any) -> str:
    data = _class_info_to_dict(class_info)
    value = _first_present(
        data,
        ["class_key", "key", "slug", "id"],
        default=None,
    )
    if value is None:
        raise ValueError(f"Cannot determine class_key from CLASS_INFOS item: {class_info!r}")
    return str(value)


def get_class_name(class_info: Any) -> str:
    data = _class_info_to_dict(class_info)
    value = _first_present(
        data,
        ["class_name", "display_name", "name", "label_name", "title"],
        default=None,
    )
    if value is None:
        raise ValueError(f"Cannot determine class_name/display_name from CLASS_INFOS item: {class_info!r}")
    return str(value)


def get_class_label(class_info: Any) -> int:
    data = _class_info_to_dict(class_info)
    value = _first_present(
        data,
        ["label", "label_index", "label_id", "idx", "index"],
        default=None,
    )
    if value is None:
        raise ValueError(f"Cannot determine label from CLASS_INFOS item: {class_info!r}")
    return int(value)


def get_class_raw_dir(class_info: Any) -> Path:
    data = _class_info_to_dict(class_info)
    value = _first_present(
        data,
        ["raw_dir", "dir", "directory", "folder", "path"],
        default=None,
    )
    if value is None:
        raise ValueError(f"Cannot determine raw_dir from CLASS_INFOS item: {class_info!r}")
    return Path(value)


def validate_class_infos() -> pd.DataFrame:
    """Ellenőrzi, hogy a config.CLASS_INFOS minden szükséges mezőt kiad-e."""
    rows: list[dict[str, Any]] = []

    for class_info in CLASS_INFOS:
        rows.append(
            {
                "class_key": get_class_key(class_info),
                "class_name": get_class_name(class_info),
                "label": get_class_label(class_info),
                "raw_dir": str(get_class_raw_dir(class_info)),
            }
        )

    df = pd.DataFrame(rows)

    if df["class_key"].isna().any() or (df["class_key"].astype(str).str.len() == 0).any():
        raise ValueError("CLASS_INFOS contains empty class_key values.")

    if df["class_name"].isna().any() or (df["class_name"].astype(str).str.len() == 0).any():
        raise ValueError("CLASS_INFOS contains empty class_name/display_name values.")

    if df["label"].isna().any() or (df["label"].astype(int) < 0).any():
        raise ValueError("CLASS_INFOS contains invalid negative/empty label values.")

    if df["class_key"].duplicated().any():
        dup = df.loc[df["class_key"].duplicated(), "class_key"].tolist()
        raise ValueError(f"Duplicate class_key values in CLASS_INFOS: {dup}")

    if df["label"].duplicated().any():
        dup = df.loc[df["label"].duplicated(), "label"].tolist()
        raise ValueError(f"Duplicate label values in CLASS_INFOS: {dup}")

    return df.sort_values("label").reset_index(drop=True)


# =========================================================
# Split CSV handling
# =========================================================

def read_split_csv(split_csv_path: str | Path) -> pd.DataFrame:
    split_csv_path = Path(split_csv_path)
    df = pd.read_csv(split_csv_path, keep_default_na=False)

    missing = REQUIRED_SPLIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {split_csv_path}: {sorted(missing)}\n"
            f"Found columns: {df.columns.tolist()}\n"
            f"Required columns: {SPLIT_COLUMNS}"
        )

    df = df[SPLIT_COLUMNS].copy()
    df["relative_path"] = df["relative_path"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["class_key"] = df["class_key"].astype(str)
    df["class_name"] = df["class_name"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    bad = df[(df["label"] < 0) | (df["class_key"].str.len() == 0) | (df["class_name"].str.len() == 0)]
    if len(bad) > 0:
        raise ValueError(
            f"Invalid rows in split CSV: {split_csv_path}\n"
            f"Examples:\n{bad.head(10).to_string(index=False)}"
        )

    return df


def save_split_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    missing = REQUIRED_SPLIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot save split CSV. Missing columns: {sorted(missing)}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    df = df[SPLIT_COLUMNS].copy()

    if df["label"].isna().any() or (df["label"].astype(int) < 0).any():
        raise ValueError("Cannot save split CSV with NaN or negative labels.")
    if df["class_name"].isna().any() or (df["class_name"].astype(str).str.len() == 0).any():
        raise ValueError("Cannot save split CSV with empty class_name values.")
    if df["class_key"].isna().any() or (df["class_key"].astype(str).str.len() == 0).any():
        raise ValueError("Cannot save split CSV with empty class_key values.")

    df.to_csv(path, index=False)


# =========================================================
# Split creation
# =========================================================

def build_metadata_dataframe(source_root: str | Path = RAW_DIR) -> pd.DataFrame:
    """
    Képlistát készít a kiválasztott adatgyökérből.

    Fontos: ez nem csak RAW_DIR-re használható. Ha a lung_masked vagy lung_crop
    variant ugyanazokat az osztálymappákat tartalmazza, akkor azokhoz is jó:
        create_splits(source_root=LUNG_MASKED_DIR, ...)

    A legtöbb esetben elég egyszer, RAW_DIR alapján splitet készíteni, és a
    többi variant ugyanazokat a relative_path értékeket használja.
    """
    source_root = Path(source_root)

    if not source_root.exists():
        raise RuntimeError(f"[ERROR] Dataset root does not exist: {source_root}")

    class_info_df = validate_class_infos()
    rows: list[dict[str, Any]] = []

    print("[INFO] CLASS_INFOS mapping:")
    print(class_info_df[["label", "class_key", "class_name", "raw_dir"]].to_string(index=False))

    for class_info in CLASS_INFOS:
        raw_dir = get_class_raw_dir(class_info)
        raw_dir_name = raw_dir.name
        class_dir = source_root / raw_dir_name

        if not class_dir.exists():
            raise RuntimeError(
                f"[ERROR] Missing class directory: {class_dir}\n"
                f"source_root={source_root}\n"
                f"raw_dir from CLASS_INFOS={raw_dir}"
            )

        files = list_images(class_dir)
        if len(files) == 0:
            raise RuntimeError(f"[ERROR] No images found in class directory: {class_dir}")

        class_key = get_class_key(class_info)
        class_name = get_class_name(class_info)
        label = get_class_label(class_info)

        for path in files:
            rows.append(
                {
                    "relative_path": str(path.relative_to(source_root)),
                    "filename": path.name,
                    "class_key": class_key,
                    "class_name": class_name,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(f"[ERROR] No images found under: {source_root}")

    df = df[SPLIT_COLUMNS].copy()

    # Kemény validáció: itt már soha nem engedünk -1/nan értéket tovább.
    if df["label"].isna().any() or (df["label"].astype(int) < 0).any():
        raise RuntimeError("[ERROR] Invalid labels generated in metadata dataframe.")
    if df["class_name"].isna().any() or (df["class_name"].astype(str).str.len() == 0).any():
        raise RuntimeError("[ERROR] Empty class_name generated in metadata dataframe.")
    if df["class_key"].isna().any() or (df["class_key"].astype(str).str.len() == 0).any():
        raise RuntimeError("[ERROR] Empty class_key generated in metadata dataframe.")

    return df


def create_splits(
    source_root: str | Path = RAW_DIR,
    split_dir: str | Path = SPLITS_DIR,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = SEED,
    overwrite: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Létrehozza a train.csv / val.csv / test.csv fájlokat egységes oszlopokkal:

        relative_path, filename, class_key, class_name, label

    Raw variant esetén tipikusan:
        create_splits(source_root=RAW_DIR, split_dir=SPLITS_DIR)

    Ha minden data_variant ugyanazt a mappaszerkezetet használja, akkor a raw
    alapján készült split újrahasználható lung_masked/lung_crop esetén is.
    """
    split_dir = Path(split_dir)
    ensure_dir(split_dir)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"

    if (
        not overwrite
        and train_csv.exists()
        and val_csv.exists()
        and test_csv.exists()
    ):
        print("[SKIP] Split files already exist.")
        return {
            "train": read_split_csv(train_csv),
            "val": read_split_csv(val_csv),
            "test": read_split_csv(test_csv),
        }

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_size + val_size + test_size must be 1.0, got {total}"
        )

    df = build_metadata_dataframe(source_root)

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df["label"],
    )

    relative_val_size = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.sort_values(["label", "relative_path"]).reset_index(drop=True)
    val_df = val_df.sort_values(["label", "relative_path"]).reset_index(drop=True)
    test_df = test_df.sort_values(["label", "relative_path"]).reset_index(drop=True)

    save_split_csv(train_df, train_csv)
    save_split_csv(val_df, val_csv)
    save_split_csv(test_df, test_csv)

    print("\n[OK] Created split CSV files:")
    print("train:", train_csv, len(train_df))
    print("val  :", val_csv, len(val_df))
    print("test :", test_csv, len(test_df))

    print_split_summary(split_dir)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


# =========================================================
# Image loading
# =========================================================

def load_image_from_path(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = 1,
):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=channels,
        expand_animations=False,
    )
    image.set_shape([None, None, channels])

    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0

    return image, tf.cast(label, tf.int32)


# =========================================================
# Augmentation
# =========================================================

def build_default_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(height_factor=0.05, width_factor=0.05),
            tf.keras.layers.RandomTranslation(height_factor=0.03, width_factor=0.03),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="cxr_augmentation",
    )


# =========================================================
# Dataset builders
# =========================================================

def build_dataset_from_dataframe(
    df: pd.DataFrame,
    root_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    augment_fn: Callable | None = None,
    cache: bool = CACHE_DATASET,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = 1,
):
    if len(df) == 0:
        raise ValueError("Received empty dataframe.")

    missing = REQUIRED_SPLIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    paths = [
        resolve_image_path(root_dir, p)
        for p in df["relative_path"].astype(str).tolist()
    ]
    labels = df["label"].astype(int).tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, y: load_image_from_path(
            p,
            y,
            image_size=image_size,
            channels=channels,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    # Cache szándékosan az augmentáció ELŐTT van.
    # Így a dekódolt/átméretezett képeket cache-eljük, de az augmentáció
    # minden epochban újra random lehet.
    if cache:
        ds = ds.cache()

    if augment_fn is not None:
        ds = ds.map(
            lambda x, y: (augment_fn(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_datasets_from_split_csvs(
    split_dir: str | Path,
    data_root: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
    augment_fn: Callable | None = None,
    cache: bool = CACHE_DATASET,
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = 1,
):
    split_dir = Path(split_dir)
    data_root = Path(data_root) if data_root is not None else RAW_DIR

    train_df = read_split_csv(split_dir / "train.csv")
    val_df = read_split_csv(split_dir / "val.csv")
    test_df = read_split_csv(split_dir / "test.csv")

    train_ds = build_dataset_from_dataframe(
        train_df,
        root_dir=data_root,
        batch_size=batch_size,
        shuffle=True,
        augment_fn=augment_fn,
        cache=cache,
        image_size=image_size,
        channels=channels,
    )

    val_ds = build_dataset_from_dataframe(
        val_df,
        root_dir=data_root,
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None,
        cache=cache,
        image_size=image_size,
        channels=channels,
    )

    test_ds = build_dataset_from_dataframe(
        test_df,
        root_dir=data_root,
        batch_size=batch_size,
        shuffle=False,
        augment_fn=None,
        cache=cache,
        image_size=image_size,
        channels=channels,
    )

    return train_ds, val_ds, test_ds


# =========================================================
# Diagnostics
# =========================================================

def summarize_split(split_dir: str | Path) -> dict[str, pd.DataFrame]:
    split_dir = Path(split_dir)

    result = {}

    for split_name in ["train", "val", "test"]:
        df = read_split_csv(split_dir / f"{split_name}.csv")

        summary = (
            df.groupby(["label", "class_key", "class_name"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("label")
            .reset_index(drop=True)
        )

        result[split_name] = summary

    return result


def print_split_summary(split_dir: str | Path) -> None:
    summaries = summarize_split(split_dir)

    for split_name, summary_df in summaries.items():
        print(f"\n{split_name.title()} class distribution")
        print("-" * 72)

        total = int(summary_df["count"].sum())

        for _, row in summary_df.iterrows():
            pct = 100.0 * row["count"] / total if total > 0 else 0.0
            print(
                f"{int(row['label']):>2} | "
                f"{str(row['class_key']):<22} | "
                f"{str(row['class_name']):<22} | "
                f"{int(row['count']):>6} | "
                f"{pct:6.2f}%"
            )

        print("-" * 72)
        print(f"Total: {total}")


def inspect_split_files(split_dir: str | Path) -> None:
    split_dir = Path(split_dir)

    for split_name in ["train", "val", "test"]:
        path = split_dir / f"{split_name}.csv"
        df = read_split_csv(path)

        print("=" * 72)
        print(split_name.upper())
        print("=" * 72)
        print("path   :", path)
        print("rows   :", len(df))
        print("columns:", df.columns.tolist())
        print(df.head())
