# preprocessing.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import tensorflow as tf

from src.config import CLASS_INFOS, ClassInfo, IMAGE_SIZE, RAW_DIR, SEED, ensure_dir, BATCH_SIZE

AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# Alap utilok
# =========================================================

def get_class_dir(root_dir: str | Path, class_info: ClassInfo) -> Path:
    return Path(root_dir) / class_info.raw_dir


def get_all_image_files(
    directory: str | Path,
    allowed_exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> list[Path]:
    directory = Path(directory)
    allowed_exts = {e.lower() for e in allowed_exts}

    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in allowed_exts
    )


# =========================================================
# Kép alapműveletek
# =========================================================

def _ensure_3d_gray(image: tf.Tensor) -> tf.Tensor:
    image = tf.convert_to_tensor(image)

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)

    if image.shape.rank == 2:
        image = tf.expand_dims(image, axis=-1)

    if image.shape.rank == 3 and image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)

    return image


def minmax_normalize(image: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)
    return (image - min_val) / (max_val - min_val + eps)


# =========================================================
# Opcionális, lassabb preprocess elemek
# =========================================================

def tf_percentile(x: tf.Tensor, q: float) -> tf.Tensor:
    x = tf.reshape(tf.cast(x, tf.float32), [-1])
    x = tf.sort(x)
    n = tf.shape(x)[0]

    idx = tf.cast(
        tf.round((q / 100.0) * tf.cast(n - 1, tf.float32)),
        tf.int32,
    )
    return x[idx]


def smart_border_crop(
    image: tf.Tensor,
    threshold: float = 0.02,
    min_keep_ratio: float = 0.60,
) -> tf.Tensor:
    image = _ensure_3d_gray(image)
    image = minmax_normalize(image)

    mask = image[..., 0] > threshold
    coords = tf.where(mask)

    def _no_crop():
        return image

    def _do_crop():
        y_min = tf.reduce_min(coords[:, 0])
        y_max = tf.reduce_max(coords[:, 0])
        x_min = tf.reduce_min(coords[:, 1])
        x_max = tf.reduce_max(coords[:, 1])

        cropped = image[y_min:y_max + 1, x_min:x_max + 1, :]

        orig_h = tf.cast(tf.shape(image)[0], tf.float32)
        orig_w = tf.cast(tf.shape(image)[1], tf.float32)
        crop_h = tf.cast(tf.shape(cropped)[0], tf.float32)
        crop_w = tf.cast(tf.shape(cropped)[1], tf.float32)

        keep_h = crop_h / (orig_h + 1e-6)
        keep_w = crop_w / (orig_w + 1e-6)

        return tf.cond(
            tf.logical_and(keep_h >= min_keep_ratio, keep_w >= min_keep_ratio),
            lambda: cropped,
            lambda: image,
        )

    return tf.cond(tf.shape(coords)[0] > 0, _do_crop, _no_crop)


def contrast_normalize_percentile(
    image: tf.Tensor,
    clip_limit: float = 0.01,
) -> tf.Tensor:
    image = _ensure_3d_gray(image)
    image = minmax_normalize(image)

    flat = tf.reshape(image, [-1])
    lo = tf_percentile(flat, clip_limit * 100.0)
    hi = tf_percentile(flat, (1.0 - clip_limit) * 100.0)

    image = tf.clip_by_value(image, lo, hi)
    image = minmax_normalize(image)
    return image


def contrast_normalize_fast(
    image: tf.Tensor,
    contrast_factor: float = 1.15,
) -> tf.Tensor:
    image = _ensure_3d_gray(image)
    image = minmax_normalize(image)
    image = tf.image.adjust_contrast(image, contrast_factor=contrast_factor)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


# =========================================================
# Preprocess layer
# =========================================================

class XrayPreprocessLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        image_size: tuple[int, int] = IMAGE_SIZE,
        apply_crop: bool = False,
        apply_contrast_norm: bool = False,
        use_fast_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.apply_crop = apply_crop
        self.apply_contrast_norm = apply_contrast_norm
        self.use_fast_contrast = use_fast_contrast

    def call(self, image: tf.Tensor) -> tf.Tensor:
        image = _ensure_3d_gray(image)
        image = minmax_normalize(image)

        if self.apply_crop:
            image = smart_border_crop(image)

        image = tf.image.resize(
            image,
            self.image_size,
            method="bilinear",
            antialias=True,
        )

        if self.apply_contrast_norm:
            if self.use_fast_contrast:
                image = contrast_normalize_fast(image)
            else:
                image = contrast_normalize_percentile(image)

        return tf.clip_by_value(image, 0.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "apply_crop": self.apply_crop,
                "apply_contrast_norm": self.apply_contrast_norm,
                "use_fast_contrast": self.use_fast_contrast,
            }
        )
        return config


# =========================================================
# Augmentáció
# =========================================================

class RandomXrayAugment(tf.keras.layers.Layer):
    def __init__(
        self,
        rotation_factor: float = 0.03,
        zoom_factor: float = 0.08,
        contrast_factor: float = 0.10,
        brightness_delta: float = 0.05,
        translate_ratio: float = 0.03,
        enable_flip: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rotation = tf.keras.layers.RandomRotation(rotation_factor)
        self.zoom = tf.keras.layers.RandomZoom((-zoom_factor, zoom_factor))
        self.translate = tf.keras.layers.RandomTranslation(
            translate_ratio,
            translate_ratio,
        )
        self.contrast = tf.keras.layers.RandomContrast(contrast_factor)

        self.enable_flip = enable_flip
        if enable_flip:
            self.flip = tf.keras.layers.RandomFlip("horizontal")

        self.brightness_delta = brightness_delta

    def call(self, images, training=None):
        if not training:
            return images

        x = self.rotation(images, training=training)
        x = self.zoom(x, training=training)
        x = self.translate(x, training=training)
        x = self.contrast(x, training=training)

        if self.enable_flip:
            x = self.flip(x, training=training)

        delta = tf.random.uniform([], -self.brightness_delta, self.brightness_delta)
        x = tf.clip_by_value(x + delta, 0.0, 1.0)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "enable_flip": self.enable_flip,
                "brightness_delta": self.brightness_delta,
            }
        )
        return config


# =========================================================
# Decode
# =========================================================

def decode_xray_image(path: tf.Tensor) -> tf.Tensor:
    path = tf.convert_to_tensor(path)
    img_bytes = tf.io.read_file(path)
    lower = tf.strings.lower(path)

    def _decode_png():
        return tf.io.decode_png(img_bytes, channels=1)

    def _decode_jpeg():
        return tf.io.decode_jpeg(img_bytes, channels=1)

    img = tf.cond(
        tf.strings.regex_full_match(lower, r".*\.png"),
        _decode_png,
        _decode_jpeg,
    )

    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


# =========================================================
# Dataset builder
# =========================================================

def build_classification_dataset(
    filepaths: Sequence[str] | Sequence[Path],
    labels: Sequence[int],
    image_size: tuple[int, int] = IMAGE_SIZE,
    training: bool = False,
    shuffle: bool = True,
    seed: int = SEED,
    apply_crop: bool = False,
    apply_contrast_norm: bool = False,
    use_fast_contrast: bool = True,
    cache: bool = False,
):
    preprocess = XrayPreprocessLayer(
        image_size=image_size,
        apply_crop=apply_crop,
        apply_contrast_norm=apply_contrast_norm,
        use_fast_contrast=use_fast_contrast,
    )

    ds = tf.data.Dataset.from_tensor_slices((list(filepaths), list(labels)))

    if training and shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(filepaths), 1000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map(path, label):
        img = decode_xray_image(path)
        img = preprocess(img)
        label = tf.cast(label, tf.int32)
        return img, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)

    return ds


def apply_batch_augmentation(ds):
    augment = RandomXrayAugment()
    return ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )


# =========================================================
# Preview
# =========================================================

def plot_random_pre_post_samples_per_class(
    root_dir: str | Path = RAW_DIR,
    image_size: tuple[int, int] = IMAGE_SIZE,
    n_per_class: int = 3,
    seed: int = SEED,
    augment_preview: bool = False,
    apply_crop: bool = False,
    apply_contrast_norm: bool = False,
    use_fast_contrast: bool = True,
    save_path: str | Path | None = None,
):
    rng = random.Random(seed)

    rows = len(CLASS_INFOS)
    cols = n_per_class * (3 if augment_preview else 2)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    if rows == 1:
        axes = [axes]

    preprocess = XrayPreprocessLayer(
        image_size=image_size,
        apply_crop=apply_crop,
        apply_contrast_norm=apply_contrast_norm,
        use_fast_contrast=use_fast_contrast,
    )
    augment = RandomXrayAugment()

    for row, c in enumerate(CLASS_INFOS):
        files = get_all_image_files(get_class_dir(root_dir, c))

        if not files:
            continue

        chosen = rng.sample(files, min(n_per_class, len(files)))

        for i, f in enumerate(chosen):
            raw = decode_xray_image(str(f)).numpy()
            proc = preprocess(raw).numpy()

            col_base = i * (3 if augment_preview else 2)

            axes[row][col_base].imshow(raw[..., 0], cmap="gray")
            axes[row][col_base].set_title(f"{c.display_name}\nRAW")
            axes[row][col_base].axis("off")

            axes[row][col_base + 1].imshow(proc[..., 0], cmap="gray")
            axes[row][col_base + 1].set_title(f"{c.display_name}\nPROC")
            axes[row][col_base + 1].axis("off")

            if augment_preview:
                aug = augment(tf.expand_dims(proc, 0), training=True)[0].numpy()
                axes[row][col_base + 2].imshow(aug[..., 0], cmap="gray")
                axes[row][col_base + 2].set_title(f"{c.display_name}\nAUG")
                axes[row][col_base + 2].axis("off")

    plt.tight_layout()

    if save_path:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=180)

    plt.show()
