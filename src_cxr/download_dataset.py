from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import kagglehub

from src.config import RAW_DIR, SEGMENTATION_RAW_DIR, ensure_dir


# =========================================================
# Kaggle dataset azonosítók
# =========================================================

COVID_CRD_SLUG = "unaissait/curated-chest-xray-image-dataset-for-covid19"
CRD_SEG_SLUG = "mrunalnshah/crd-chest-x-ray-images-with-lung-segmented-masks"


# =========================================================
# Marker fájlok
# =========================================================

COVID_READY_MARKER = RAW_DIR / ".dataset_ready"
SEG_READY_MARKER = SEGMENTATION_RAW_DIR / ".dataset_ready"


# =========================================================
# Általános utilok
# =========================================================

def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def copytree_merge(src: Path, dst: Path) -> None:
    """
    Python 3.8+ kompatibilis merge-szerű másolás.
    Ha a cél nem létezik, simán másol.
    Ha létezik, a hiányzó fájlokat/mappákat belemozgatja.
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copytree_merge(item, target)
        else:
            if target.exists():
                continue
            shutil.copy2(item, target)


# =========================================================
# Classifier dataset mozgatása
# =========================================================

def move_classifier_dataset(tmp_root: Path) -> None:
    """
    Kaggle dataset tipikusan így bontódik ki:

    tmp_root/
        Curated X-Ray Dataset/
            Normal/
            COVID-19/
            Pneumonia-Bacterial/
            Pneumonia-Viral/

    vagy esetenként közvetlenül a tmp_root alá.
    """
    possible_roots = [
        tmp_root / "Curated X-Ray Dataset",
        tmp_root,
    ]

    src_root = None
    for p in possible_roots:
        if p.exists():
            src_root = p
            break

    if src_root is None:
        raise RuntimeError(
            f"[ERROR] Could not locate extracted classifier dataset in: {tmp_root}"
        )

    print("[INFO] Moving classifier dataset into RAW_DIR...")

    moved_any = False

    for item in src_root.iterdir():
        if not item.is_dir():
            continue

        target = RAW_DIR / item.name

        if target.exists():
            print(f"[SKIP] Already exists: {target}")
            continue

        shutil.move(str(item), str(target))
        print(f"[OK] Moved: {item.name}")
        moved_any = True

    if not moved_any:
        print("[WARN] No new classifier folders were moved.")

    touch(COVID_READY_MARKER)


# =========================================================
# Segmentation dataset mozgatása
# =========================================================

def move_segmentation_dataset(
    tmp_root: Path,
) -> None:
    """
    Várt forrásstruktúra például:

    tmp_root/
        crd_lung_masks/
            CXR_Combined/
                images/
                masks/
            CXR_RadioLucent/
            CXR_RadioLucent_masks/
            CXR_RadioOpaque/
            CXR_RadioOpaque_masks/

    A cél:
        SEGMENTATION_RAW_DIR/
            images/
            masks/
    """
    possible_roots = [
        tmp_root / "crd_lung_masks",
        tmp_root,
    ]

    src_root = None
    for p in possible_roots:
        if p.exists():
            src_root = p
            break

    if src_root is None:
        raise RuntimeError(
            f"[ERROR] Could not locate extracted segmentation dataset in: {tmp_root}"
        )

    print(f"[INFO] Segmentation source root: {src_root}")
    print("[INFO] Moving segmentation dataset into SEGMENTATION_RAW_DIR...")

    combined_dir = src_root / "CXR_Combined"
    src_images = combined_dir / "images"
    src_masks = combined_dir / "masks"

    if not combined_dir.exists():
        raise RuntimeError(f"[ERROR] Missing folder: {combined_dir}")
    if not src_images.exists():
        raise RuntimeError(f"[ERROR] Missing folder: {src_images}")
    if not src_masks.exists():
        raise RuntimeError(f"[ERROR] Missing folder: {src_masks}")

    dst_images = SEGMENTATION_RAW_DIR / "images"
    dst_masks = SEGMENTATION_RAW_DIR / "masks"

    if dst_images.exists():
        print(f"[SKIP] Already exists: {dst_images}")
    else:
        shutil.move(str(src_images), str(dst_images))
        print(f"[OK] Moved: {src_images} -> {dst_images}")

    if dst_masks.exists():
        print(f"[SKIP] Already exists: {dst_masks}")
    else:
        shutil.move(str(src_masks), str(dst_masks))
        print(f"[OK] Moved: {src_masks} -> {dst_masks}")

    if not dst_images.exists():
        raise RuntimeError(f"[ERROR] Target images dir missing: {dst_images}")
    if not dst_masks.exists():
        raise RuntimeError(f"[ERROR] Target masks dir missing: {dst_masks}")

    touch(SEG_READY_MARKER)
    print("[OK] Segmentation dataset structure verified.")
# =========================================================
# Letöltés
# =========================================================

def _download_to_temp(slug: str) -> Path:
    """
    KaggleHub letöltés temp helyre.
    A kagglehub.dataset_download egy cache-elt lokációt ad vissza.
    Innen egy ideiglenes munkakönyvtárba másolunk, hogy biztonságosan
    tudjunk move-olni.
    """
    downloaded_path = Path(kagglehub.dataset_download(slug))
    print(f"[INFO] KaggleHub downloaded/cached at: {downloaded_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="cxr_download_"))
    print(f"[INFO] Temporary working dir: {tmp_dir}")

    if downloaded_path.is_dir():
        copytree_merge(downloaded_path, tmp_dir)
    else:
        raise RuntimeError(f"[ERROR] Downloaded path is not a directory: {downloaded_path}")

    return tmp_dir


def download_classifier_dataset(force: bool = False) -> None:
    ensure_dir(RAW_DIR)

    if COVID_READY_MARKER.exists() and not force:
        print("[SKIP] Classifier dataset already exists, no download needed.")
        return

    if force:
        print("[INFO] Force download requested for classifier dataset.")
        remove_if_exists(COVID_READY_MARKER)

    tmp_dir = None
    try:
        tmp_dir = _download_to_temp(COVID_CRD_SLUG)
        move_classifier_dataset(tmp_dir)
        print("[OK] Classifier dataset ready.")
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def download_segmentation_dataset(force: bool = False) -> None:
    ensure_dir(SEGMENTATION_RAW_DIR)

    if SEG_READY_MARKER.exists() and not force:
        print("[SKIP] Segmentation dataset already exists, no download needed.")
        return

    if force:
        print("[INFO] Force download requested for segmentation dataset.")
        remove_if_exists(SEG_READY_MARKER)

    tmp_dir = None
    try:
        tmp_dir = _download_to_temp(CRD_SEG_SLUG)
        move_segmentation_dataset(tmp_root=tmp_dir)
        print("[OK] Segmentation dataset ready.")
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def download_all_datasets() -> None:
    download_classifier_dataset()
    download_segmentation_dataset()


# =========================================================
# CLI futtatás
# =========================================================

if __name__ == "__main__":
    download_all_datasets()
