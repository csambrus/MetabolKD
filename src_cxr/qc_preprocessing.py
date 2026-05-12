# qc_preprocessing.py

from pathlib import Path

from src.config import RAW_DIR, IMAGE_SIZE, SEED
from src.preprocessing import plot_random_pre_post_samples_per_class


def run_preprocessing_qc(
    root_dir=RAW_DIR,
    out_dir="outputs/preprocessing",
    n_per_class=4,
    seed=SEED,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPROCESSING QC")
    print("=" * 80)
    print(f"Root dir: {root_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Samples per class: {n_per_class}")
    print("=" * 80)

    # -----------------------------------------------------
    # RAW vs PROCESSED
    # -----------------------------------------------------
    print("\n[1/2] RAW vs PROCESSED preview")

    raw_proc_path = out_dir / "pre_post_examples.png"

    plot_random_pre_post_samples_per_class(
        root_dir=root_dir,
        image_size=IMAGE_SIZE,
        n_per_class=n_per_class,
        seed=seed,
        augment_preview=False,
        save_path=raw_proc_path,
    )

    print(f"[INFO] Saved: {raw_proc_path}")

    # -----------------------------------------------------
    # RAW vs PROCESSED + AUGMENTED
    # -----------------------------------------------------
    print("\n[2/2] RAW vs PROCESSED + AUGMENTED preview")

    raw_proc_aug_path = out_dir / "pre_post_aug_examples.png"

    plot_random_pre_post_samples_per_class(
        root_dir=root_dir,
        image_size=IMAGE_SIZE,
        n_per_class=n_per_class,
        seed=seed,
        augment_preview=True,
        save_path=raw_proc_aug_path,
    )

    print(f"[INFO] Saved: {raw_proc_aug_path}")

    print("\n" + "=" * 80)
    print("PREPROCESSING QC DONE")
    print("=" * 80)

    return {
        "raw_processed": raw_proc_path,
        "raw_processed_augmented": raw_proc_aug_path,
    }


def main():
    run_preprocessing_qc()


if __name__ == "__main__":
    main()