from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import DATA_DIR, GDRIVE_DATA


STUDY_ID = "MTBLS28"

REQUIRED_FILES = {
    "sample": "s_MTBLS28.txt",
    "assay_pos": "a_MTBLS28_POS.txt",
    "assay_neg": "a_MTBLS28_NEG.txt",
    "maf_pos": "m_MTBLS28_POS_v2_maf.tsv",
    "maf_neg": "m_MTBLS28_NEG_v2_maf.tsv",
}

MODE_SAMPLE_RE = re.compile(r"_(?P<mode>POS|NEG)_(?P<key>\d{6})$", re.IGNORECASE)
ANY_SAMPLE_KEY_RE = re.compile(r"_(?P<key>\d{6})$")


def _read_tsv(path: Path, *, dtype: object = str) -> pd.DataFrame:
    """Read a MetaboLights ISA/MAF TSV file."""
    df = pd.read_csv(path, sep="\t", dtype=dtype, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _study_dir_has_required_files(study_dir: Path) -> bool:
    return all(_is_nonempty_file(study_dir / name) for name in REQUIRED_FILES.values())


def _copy_required_files(src_study_dir: Path, dst_study_dir: Path) -> None:
    """Copy only the files needed for the ML feature matrix."""
    dst_study_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES.values():
        src = src_study_dir / name
        dst = dst_study_dir / name
        if not _is_nonempty_file(src):
            raise FileNotFoundError(f"Hiányzó forrásfájl: {src}")
        shutil.copy2(src, dst)
        print(f"Másolva → {dst}")


def _ensure_local_study_dir(
    *,
    gdrive_data: str | Path | None = None,
    data_dir: str | Path | None = None,
    study_id: str = STUDY_ID,
) -> Path:
    """
    Ensure DATA_DIR/<study_id> contains the required files.

    Priority (ugyanaz, mint a ``download_dataset.py``):
    1. Már kész DATA_DIR/<study_id>
    2. Már kész GDRIVE_DATA/<study_id> → másolás DATA_DIR alá
    3. MetaboLights nyilvános FTP HTTP tükör: ISA-tab + hivatkozott táblák letöltése
       GDRIVE_DATA/<study_id>-be, majd másolás DATA_DIR/<study_id>-be
    """
    from src.download_dataset import _copy_tree_files, _download_selected_files_to_gdrive

    gdrive_root = Path(gdrive_data or GDRIVE_DATA)
    data_root = Path(data_dir or DATA_DIR)

    gdrive_study_dir = gdrive_root / study_id
    work_study_dir = data_root / study_id

    if _study_dir_has_required_files(work_study_dir):
        print(f"DATA_DIR: megvan {work_study_dir}")
        return work_study_dir

    if _study_dir_has_required_files(gdrive_study_dir):
        print(f"GDRIVE_DATA: megvan {gdrive_study_dir}")
        _copy_required_files(gdrive_study_dir, work_study_dir)
        return work_study_dir

    downloaded = _download_selected_files_to_gdrive(study_id, gdrive_root)
    _copy_tree_files(gdrive_study_dir, work_study_dir, downloaded)

    if not _study_dir_has_required_files(work_study_dir):
        missing = [
            name for name in REQUIRED_FILES.values()
            if not _is_nonempty_file(work_study_dir / name)
        ]
        raise RuntimeError(
            f"Letöltés után hiányoznak a kötelező fájlok ({study_id}): {missing}"
        )

    return work_study_dir


def _sample_key(sample_name: object) -> str:
    """
    Convert full MetaboLights sample names to a common biological sample key.

    POS example: 20080809_NIH_EWY_POS_000801 → 000801
    NEG example: 20081009_NIH_EWY_NEG_000801 → 000801
    """
    s = str(sample_name).strip()
    m = MODE_SAMPLE_RE.search(s)
    if m:
        return m.group("key")
    m = ANY_SAMPLE_KEY_RE.search(s)
    if m:
        return m.group("key")
    return s


def _sample_mode(sample_name: object) -> str | None:
    s = str(sample_name).strip()
    m = MODE_SAMPLE_RE.search(s)
    if not m:
        return None
    return m.group("mode").upper()


def _clean_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "na", "n/a"}:
        return pd.NA
    return s


def _first_non_null(values: pd.Series) -> object:
    vals = [v for v in values.map(_clean_value).tolist() if not pd.isna(v)]
    if not vals:
        return pd.NA
    unique = list(dict.fromkeys(vals))
    if len(unique) > 1:
        # Keep the first value but make the conflict visible in the console.
        print(f"Figyelem: eltérő metaértékek ugyanazon mintakulcshoz: {unique[:5]}")
    return unique[0]


def _build_sample_metadata(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the ISA sample file from 2010 POS/NEG rows to 1005 biological samples.

    Returns metadata indexed by sample_id = six-digit sample key.
    """
    df = sample_df.copy()
    if "Sample Name" not in df.columns:
        raise ValueError("Az s_MTBLS28.txt fájlban nincs 'Sample Name' oszlop.")

    df["sample_id"] = df["Sample Name"].map(_sample_key)
    df["ion_mode"] = df["Sample Name"].map(_sample_mode)

    pos_map = (
        df[df["ion_mode"].eq("POS")]
        .drop_duplicates("sample_id")
        .set_index("sample_id")["Sample Name"]
        .rename("pos_sample_name")
    )
    neg_map = (
        df[df["ion_mode"].eq("NEG")]
        .drop_duplicates("sample_id")
        .set_index("sample_id")["Sample Name"]
        .rename("neg_sample_name")
    )

    useful_cols = [
        c for c in df.columns
        if c.startswith("Factor Value[") or c.startswith("Characteristics[")
    ]
    grouped = df.groupby("sample_id", sort=True)[useful_cols].agg(_first_non_null)

    meta = grouped.join(pos_map, how="left").join(neg_map, how="left")
    meta.index.name = "sample_id"

    # Friendly standardized columns for ML.
    rename_map = {
        "Factor Value[Sample Type]": "sample_type",
        "Factor Value[Smoking]": "smoking",
        "Factor Value[Race]": "race",
        "Factor Value[Gender]": "gender",
        "Characteristics[Organism part]": "organism_part",
    }
    for old, new in rename_map.items():
        if old in meta.columns and new not in meta.columns:
            meta[new] = meta[old]

    if "sample_type" in meta.columns:
        meta["class_label"] = meta["sample_type"].map({
            "Case": "Lung cancer",
            "Control": "Control",
        }).fillna(meta["sample_type"])
        meta["target"] = meta["sample_type"].map({"Case": 1, "Control": 0}).astype("Int64")

    return meta


def _format_number(value: object, ndigits: int) -> str:
    if pd.isna(value):
        return "NA"
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return re.sub(r"\s+", "_", str(value).strip())


def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        base = re.sub(r"\s+", " ", str(name).strip()) or "feature"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    return out


def _feature_names_from_maf(maf_df: pd.DataFrame, *, prefix: Literal["pos", "neg"]) -> list[str]:
    """
    Build stable feature names from MAF metadata.

    Most MTBLS28 metabolite_identification values are empty, so m/z + retention time
    is the safest feature identifier.
    """
    names: list[str] = []
    for i, row in maf_df.iterrows():
        ident = _clean_value(row.get("metabolite_identification"))
        mz = _format_number(row.get("mass_to_charge"), 5)
        rt = _format_number(row.get("retention_time"), 2)

        if ident is not pd.NA and not pd.isna(ident):
            base = f"{ident}; mz={mz}; rt={rt}"
        else:
            base = f"mz={mz}; rt={rt}"

        names.append(f"{prefix}__{base}")

    return _make_unique(names)


def _sample_columns_for_mode(maf_df: pd.DataFrame, mode: Literal["pos", "neg"]) -> list[str]:
    mode_upper = mode.upper()
    pat = re.compile(fr"_{mode_upper}_\d{{6}}$", re.IGNORECASE)
    cols = [c for c in maf_df.columns if pat.search(str(c).strip())]
    if not cols:
        raise ValueError(f"Nem találtam {mode_upper} mintaoszlopokat a MAF fájlban.")
    return cols


def _make_one_mode_feature_matrix(maf_df: pd.DataFrame, *, mode: Literal["pos", "neg"]) -> pd.DataFrame:
    """
    Convert one MTBLS28 MAF table to samples x features.

    Input:
        rows = metabolite features
        columns = MAF metadata + full POS/NEG sample names
    Output:
        rows = biological samples indexed by common six-digit sample_id
        columns = prefixed feature names
    """
    sample_cols = _sample_columns_for_mode(maf_df, mode)
    feature_names = _feature_names_from_maf(maf_df, prefix=mode)

    values = maf_df[sample_cols].apply(pd.to_numeric, errors="coerce")
    values.index = feature_names

    X = values.T
    X.index = [_sample_key(c) for c in X.index]
    X.index.name = "sample_id"

    if X.index.duplicated().any():
        dup = X.index[X.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplikált sample_id a {mode} MAF-ban: {dup}")

    return X


def download_dataset(
    gdrive_data: str | Path | None = None,
    data_dir: str | Path | None = None,
    *,
    study_id: str = STUDY_ID,
) -> dict[str, pd.DataFrame | Path | str]:
    """
    MTBLS28: helyi cache vagy MetaboLights nyilvános FTP-ről letöltés
    (``download_dataset.py``-vel azonos mechanizmus), majd ISA/MAF beolvasás.

    Returns raw ISA/MAF tables. Use combine_pos_neg_to_feature_matrix() to get X/meta.
    """
    study_dir = _ensure_local_study_dir(
        gdrive_data=gdrive_data,
        data_dir=data_dir,
        study_id=study_id,
    )

    sample = _read_tsv(study_dir / REQUIRED_FILES["sample"])
    assay_pos = _read_tsv(study_dir / REQUIRED_FILES["assay_pos"])
    assay_neg = _read_tsv(study_dir / REQUIRED_FILES["assay_neg"])
    maf_pos = _read_tsv(study_dir / REQUIRED_FILES["maf_pos"])
    maf_neg = _read_tsv(study_dir / REQUIRED_FILES["maf_neg"])

    print("MTBLS28 raw:")
    print(f"  sample:    {sample.shape}")
    print(f"  assay_pos: {assay_pos.shape}")
    print(f"  assay_neg: {assay_neg.shape}")
    print(f"  maf_pos:   {maf_pos.shape}")
    print(f"  maf_neg:   {maf_neg.shape}")

    return {
        "study_id": study_id,
        "study_dir": study_dir,
        "sample": sample,
        "assay_pos": assay_pos,
        "assay_neg": assay_neg,
        "pos": maf_pos,
        "neg": maf_neg,
    }


def combine_pos_neg_to_feature_matrix(
    metabol_data: dict[str, pd.DataFrame],
    *,
    join: Literal["inner", "outer"] = "inner",
    return_metadata: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine MTBLS28 POS and NEG MAF tables into one samples x features matrix.

    Returns
    -------
    X:
        1005 x 3166 expected for the provided MTBLS28 files
        = 1807 POS features + 1359 NEG features.
    meta:
        1005 sample-level metadata rows, indexed like X.
        Important columns: sample_type, class_label, target, smoking, race, gender.
    """
    pos_raw = metabol_data["pos"]
    neg_raw = metabol_data["neg"]
    sample_raw = metabol_data["sample"]

    if not isinstance(pos_raw, pd.DataFrame) or not isinstance(neg_raw, pd.DataFrame):
        raise TypeError("metabol_data['pos'] és metabol_data['neg'] pandas DataFrame kell legyen.")
    if not isinstance(sample_raw, pd.DataFrame):
        raise TypeError("metabol_data['sample'] pandas DataFrame kell legyen.")

    X_pos = _make_one_mode_feature_matrix(pos_raw, mode="pos")
    X_neg = _make_one_mode_feature_matrix(neg_raw, mode="neg")

    X = X_pos.join(X_neg, how=join)
    X = X.dropna(axis=1, how="all")

    print(f"POS feature matrix: {X_pos.shape}")
    print(f"NEG feature matrix: {X_neg.shape}")
    print(f"Combined feature matrix: {X.shape}")

    if not return_metadata:
        return X

    meta = _build_sample_metadata(sample_raw).reindex(X.index)

    if "class_label" in meta.columns:
        print("Csoportok:")
        print(meta["class_label"].value_counts(dropna=False))

    return X, meta

