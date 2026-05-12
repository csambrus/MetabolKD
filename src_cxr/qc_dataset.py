# src/qc_dataset.py

# qc_dataset.py

from __future__ import annotations

import json
from pathlib import Path

from src.config import BATCH_SIZE, CLASS_INFOS, IMAGE_SIZE, RAW_DIR, SEED, SPLITS_DIR, OUTPUT_DIR, ensure_dir
from src.dataloader import (
    build_datasets_from_root,
    build_metadata_dataframe,
    get_class_distribution,
    inspect_batch,
    print_class_distribution,
)
from src.preprocessing import plot_random_pre_post_samples_per_class

def distribution_df_to_records(dist_df):
    """
    Pandas DataFrame -> JSON-kompatibilis lista.
    """
    if dist_df.empty:
        return []

    records = []
    for _, row in dist_df.iterrows():
        records.append(
            {
                "label": int(row["label"]),
                "class_key": str(row["class_key"]),
                "class_name": str(row["class_name"]),
                "count": int(row["count"]),
                "ratio": float(row["ratio"]),
            }
        )
    return records


# =========================================================
# Fő QC pipeline
# =========================================================

def run_dataset_qc(
    root_dir: str | Path = RAW_DIR,
    out_dir: str | Path = None,
    splits_dir: str | Path = SPLITS_DIR,
    test_size: float = 0.15,
    val_size: float = 0.15,
    image_size: tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    n_preview_per_class: int = 4,
):
    root_dir = Path(root_dir)

    out_dir = out_dir or OUTPUT_DIR / "dataset_qc"
    out_dir = ensure_dir(out_dir)

    preview_dir = ensure_dir(out_dir / "previews")
    split_dir = ensure_dir(splits_dir)
    
    print("=" * 80)
    print("DATASET QC START")
    print("=" * 80)
    print(f"Root dir: {root_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Split dir: {split_dir}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    print(f"Classes: {[c.display_name for c in CLASS_INFOS]}")
    print("=" * 80)

    # -----------------------------------------------------
    # 1. Metadata build
    # -----------------------------------------------------
    print("\n[1/6] Building metadata dataframe...")
    full_df = build_metadata_dataframe(root_dir=root_dir)

    if full_df.empty:
        raise ValueError(f"Nem találtam képeket itt: {root_dir}")

    print_class_distribution(full_df, title="Full dataset class distribution")

    full_df.to_csv(out_dir / "full_metadata.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'full_metadata.csv'}")

    full_dist_df = get_class_distribution(full_df)
    full_dist_df.to_csv(out_dir / "full_distribution.csv", index=False)
    print(f"[INFO] Saved: {out_dir / 'full_distribution.csv'}")

    # -----------------------------------------------------
    # 2. Split + tf.data datasets
    # -----------------------------------------------------
    print("\n[2/6] Building train/val/test splits and tf.data datasets...")
    train_ds, val_ds, test_ds, train_df, val_df, test_df = build_datasets_from_root(
        root_dir=root_dir,
        test_size=test_size,
        val_size=val_size,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        stratify=True,
        export_dir=split_dir,
    )

    # -----------------------------------------------------
    # 3. Preview képek mentése
    # -----------------------------------------------------
    print("\n[3/6] Saving raw vs processed previews...")
    raw_proc_path = preview_dir / "pre_post_examples.png"
    plot_random_pre_post_samples_per_class(
        root_dir=root_dir,
        image_size=image_size,
        n_per_class=n_preview_per_class,
        seed=seed,
        augment_preview=False,
        save_path=raw_proc_path,
    )
    print(f"[INFO] Saved: {raw_proc_path}")

    print("\n[4/6] Saving raw vs processed+augmented previews...")
    raw_proc_aug_path = preview_dir / "pre_post_aug_examples.png"
    plot_random_pre_post_samples_per_class(
        root_dir=root_dir,
        image_size=image_size,
        n_per_class=n_preview_per_class,
        seed=seed,
        augment_preview=True,
        save_path=raw_proc_aug_path,
    )
    print(f"[INFO] Saved: {raw_proc_aug_path}")

    # -----------------------------------------------------
    # 4. Batch sanity check
    # -----------------------------------------------------
    print("\n[5/6] Batch sanity check...")
    print("\nTrain batch:")
    inspect_batch(train_ds, n_classes=len(CLASS_INFOS))

    print("\nValidation batch:")
    inspect_batch(val_ds, n_classes=len(CLASS_INFOS))

    print("\nTest batch:")
    inspect_batch(test_ds, n_classes=len(CLASS_INFOS))

    # -----------------------------------------------------
    # 5. Summary JSON
    # -----------------------------------------------------
    print("\n[6/6] Saving dataset summary JSON...")
    train_dist_df = get_class_distribution(train_df)
    val_dist_df = get_class_distribution(val_df)
    test_dist_df = get_class_distribution(test_df)

    summary = {
        "root_dir": str(root_dir),
        "out_dir": str(out_dir),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "batch_size": int(batch_size),
        "seed": int(seed),
        "n_classes": int(len(CLASS_INFOS)),
        "classes": [
            {
                "key": c.key,
                "raw_dir": c.raw_dir,
                "display_name": c.display_name,
                "idx": int(c.idx),
            }
            for c in CLASS_INFOS
        ],
        "split": {
            "test_size": float(test_size),
            "val_size": float(val_size),
            "full_count": int(len(full_df)),
            "train_count": int(len(train_df)),
            "val_count": int(len(val_df)),
            "test_count": int(len(test_df)),
        },
        "distributions": {
            "full": distribution_df_to_records(full_dist_df),
            "train": distribution_df_to_records(train_dist_df),
            "val": distribution_df_to_records(val_dist_df),
            "test": distribution_df_to_records(test_dist_df),
        },
        "artifacts": {
            "full_metadata_csv": str(out_dir / "full_metadata.csv"),
            "full_distribution_csv": str(out_dir / "full_distribution.csv"),
            "split_train_csv": str(split_dir / "split_train.csv"),
            "split_val_csv": str(split_dir / "split_val.csv"),
            "split_test_csv": str(split_dir / "split_test.csv"),
            "distribution_train_csv": str(out_dir / "distribution_train.csv"),
            "distribution_val_csv": str(out_dir / "distribution_val.csv"),
            "distribution_test_csv": str(out_dir / "distribution_test.csv"),
            "preview_raw_processed": str(raw_proc_path),
            "preview_raw_processed_augmented": str(raw_proc_aug_path),
        },
    }

    summary_path = out_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved: {summary_path}")

    print("\n" + "=" * 80)
    print("DATASET QC DONE")
    print("=" * 80)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "summary_path": summary_path,
        "preview_raw_processed": raw_proc_path,
        "preview_raw_processed_augmented": raw_proc_aug_path,
    }


# =========================================================
# Main
# =========================================================

def main():
    run_dataset_qc(
        root_dir=RAW_DIR,
        out_dir=OUTPUT_DIR / "dataset_qc",
        test_size=0.15,
        val_size=0.15,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        n_preview_per_class=4,
    )


if __name__ == "__main__":
    main()