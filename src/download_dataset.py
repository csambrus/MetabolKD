from __future__ import annotations

import html.parser
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import DATA_DIR, GDRIVE_DATA


STUDY_ID = "MTBLS28"

# MetaboLights public FTP HTTP mirror
MTBLS_PUBLIC_BASE = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"

# Csak a kis/processed/ISA-tab fájlokat töltjük automatikusan.
# A nyers LC-MS fájlok több GB-osak lehetnek, ML-projekthez első körben nem kellenek.
ISA_FILE_RE = re.compile(r"^[isam]_.*\.(txt|tsv|csv)$", re.IGNORECASE)
TABLE_FILE_RE = re.compile(r".*\.(txt|tsv|csv|xlsx)$", re.IGNORECASE)

RAW_FILE_EXTENSIONS = {
    ".raw", ".mzml", ".mzxml", ".cdf", ".netcdf", ".wiff", ".d", ".zip", ".gz",
}

class _HrefParser(html.parser.HTMLParser):
    """Minimal HTML directory-listing parser for FTP-style index pages."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.hrefs.append(href)


def _download_if_missing(url: str, out_path: Path, *, timeout: int = 300) -> Path:
    """Download url to out_path only if out_path does not exist or is empty."""
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Már megvan, nem töltöm újra: {out_path}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Letöltés: {url}")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        out_path.write_bytes(response.read())

    return out_path


def _read_url_text(url: str, *, timeout: int = 120) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _study_base_url(study_id: str = STUDY_ID) -> str:
    return f"{MTBLS_PUBLIC_BASE.rstrip('/')}/{study_id.strip('/')}/"


def _quote_url_path_part(name: str) -> str:
    """
    Quote a possibly spaced file name while keeping subdirectory slashes.
    """
    return urllib.parse.quote(name, safe="/")


def list_mtbls_study_files(study_id: str = STUDY_ID) -> list[str]:
    """
    List top-level files in a public MetaboLights study directory.

    This relies on the public FTP HTTP mirror directory listing.
    """
    base_url = _study_base_url(study_id)
    html_text = _read_url_text(base_url)

    parser = _HrefParser()
    parser.feed(html_text)

    files: list[str] = []
    for href in parser.hrefs:
        href = urllib.parse.unquote(href)

        if not href or href.startswith("?") or href in {"../", "/"}:
            continue

        # Csak top-level fájlok; könyvtárakat most nem járunk be rekurzívan.
        if href.endswith("/"):
            continue

        name = Path(href).name
        if name and name not in files:
            files.append(name)

    if not files:
        raise RuntimeError(
            f"Nem sikerült fájllistát olvasni a MetaboLights FTP mappából: {base_url}"
        )

    return sorted(files)


def _is_relevant_initial_file(name: str) -> bool:
    """
    ISA-tab és metabolite-assignment fájlok.
    """
    base = Path(name).name
    return bool(ISA_FILE_RE.match(base))


def _is_small_table_reference(name: str) -> bool:
    """
    Assay fájlokban hivatkozott táblázatos fájlok.
    Nyers MS fájlokat és zip/gz archívumokat nem töltünk automatikusan.
    """
    suffix = Path(str(name)).suffix.lower()
    if suffix in RAW_FILE_EXTENSIONS:
        return False
    return bool(TABLE_FILE_RE.match(str(name)))


def _read_any_table(path: Path) -> pd.DataFrame:
    """
    Read txt/tsv/csv/xlsx table with conservative defaults.
    """
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        return pd.read_excel(path, dtype=str)

    if suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def _copy_tree_files(src_dir: Path, dst_dir: Path, filenames: list[str]) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        src = src_dir / name
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Másolva → {dst}")


def _download_selected_files_to_gdrive(
    study_id: str,
    gdrive_dir: Path,
) -> list[str]:
    """
    Download ISA-tab and processed table files into GDRIVE_DATA/MTBLS28.

    First downloads i_/s_/a_/m_ files. Then parses assay files and downloads
    referenced small table files if present.
    """
    study_dir = gdrive_dir / study_id
    study_dir.mkdir(parents=True, exist_ok=True)

    base_url = _study_base_url(study_id)
    all_top_files = list_mtbls_study_files(study_id)

    selected = [f for f in all_top_files if _is_relevant_initial_file(f)]

    if not selected:
        raise RuntimeError(
            f"Nem találtam ISA-tab fájlokat ({study_id}) alatt. "
            f"Ellenőrizd kézzel a MetaboLights FTP mappát."
        )

    downloaded: set[str] = set()

    for name in selected:
        url = base_url + _quote_url_path_part(name)
        _download_if_missing(url, study_dir / name)
        downloaded.add(name)

    # Assay fájlokból kinyerjük az esetleges hivatkozott processed matrix fájlokat.
    referenced_tables: set[str] = set()
    for name in list(downloaded):
        if not Path(name).name.lower().startswith("a_"):
            continue

        assay_path = study_dir / name
        try:
            assay = _read_any_table(assay_path)
        except Exception as exc:
            print(f"Figyelem: nem tudtam olvasni az assay fájlt: {assay_path} ({exc})")
            continue

        for value in assay.astype(str).to_numpy().ravel():
            value = str(value).strip()
            if not value or value.lower() in {"nan", "none"}:
                continue

            # Gyakori eset: csak a fájlnév szerepel a cellában.
            # Ha útvonallal van megadva, megtartjuk.
            if _is_small_table_reference(value):
                referenced_tables.add(value)

    for ref in sorted(referenced_tables):
        # Ha már letöltött ISA fájl, kihagyható.
        if ref in downloaded:
            continue

        # Top-level fájlokra stabil. Ha almapában van, a ref tartalmazhatja a relatív utat.
        url = base_url + _quote_url_path_part(ref)
        try:
            _download_if_missing(url, study_dir / ref)
            downloaded.add(ref)
        except Exception as exc:
            print(f"Figyelem: hivatkozott fájl nem tölthető: {ref} ({exc})")

    return sorted(downloaded)


def download_dataset(
    gdrive_data: str | Path | None = None,
    data_dir: str | Path | None = None,
    study_id: str = STUDY_ID,
) -> dict[str, object]:
    """
    Download/load MTBLS28-like MetaboLights dataset.

    1) GDRIVE_DATA/study_id: hiányzó ISA-tab/processed fájlok letöltése.
    2) Másolás DATA_DIR/study_id alá.
    3) Beolvasás pandas DataFrame-ekbe.

    Returns
    -------
    dict with keys:
        study_id
        study_dir
        files
        sample_tables
        assay_tables
        matrix_tables
        meta
    """
    gdrive_root = Path(gdrive_data or GDRIVE_DATA)
    work_study_dir = Path(data_dir or DATA_DIR)

    gdrive_study_dir = gdrive_root / study_id

    downloaded = _download_selected_files_to_gdrive(study_id, gdrive_root)

    _copy_tree_files(gdrive_study_dir, work_study_dir, downloaded)

    sample_tables: dict[str, pd.DataFrame] = {}
    assay_tables: dict[str, pd.DataFrame] = {}
    matrix_tables: dict[str, pd.DataFrame] = {}

    for name in downloaded:
        path = work_study_dir / name
        base = Path(name).name.lower()

        try:
            df = _read_any_table(path)
        except Exception as exc:
            print(f"Figyelem: nem olvasható táblázatként: {path} ({exc})")
            continue

        if base.startswith("s_"):
            sample_tables[name] = df
        elif base.startswith("a_"):
            assay_tables[name] = df
        elif base.startswith("m_"):
            matrix_tables[name] = df
        elif not base.startswith("i_"):
            # Hivatkozott processed tábla, ha nem ISA i/s/a/m.
            matrix_tables[name] = df

    meta = _build_sample_metadata(sample_tables, assay_tables)

    return {
        "study_id": study_id,
        "study_dir": work_study_dir,
        "files": downloaded,
        "sample_tables": sample_tables,
        "assay_tables": assay_tables,
        "matrix_tables": matrix_tables,
        "meta": meta,
    }


def _normalize_colname(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).strip())


def _find_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for c in df.columns:
        c_norm = _normalize_colname(c).lower()
        for pat in patterns:
            if re.search(pat, c_norm):
                return c
    return None


def _standardize_lung_cancer_label(value: object) -> str | None:
    """
    Convert likely MTBLS28 disease labels into a simple binary label.
    """
    if pd.isna(value):
        return None

    s = str(value).strip().lower()
    if not s or s in {"nan", "none", "na", "n/a"}:
        return None

    if any(x in s for x in ["healthy", "control", "normal"]):
        return "Control"

    if any(x in s for x in ["lung", "cancer", "tumor", "tumour", "case", "patient", "nsclc"]):
        return "Lung cancer"

    return str(value).strip()


def _infer_target_column(meta: pd.DataFrame) -> str | None:
    """
    Find a likely disease/class column in ISA metadata.
    """
    priority_patterns = [
        r"factor value.*disease",
        r"factor value.*diagn",
        r"factor value.*status",
        r"disease",
        r"diagn",
        r"status",
        r"phenotype",
        r"class",
        r"group",
    ]

    candidates: list[str] = []
    for c in meta.columns:
        c_norm = _normalize_colname(c).lower()
        if any(re.search(pat, c_norm) for pat in priority_patterns):
            candidates.append(c)

    # Olyan oszlopot keresünk, amelyben kevés, de legalább 2 kategória van.
    for c in candidates:
        n = meta[c].nunique(dropna=True)
        if 2 <= n <= 10:
            return c

    return None


def _build_sample_metadata(
    sample_tables: dict[str, pd.DataFrame],
    assay_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build sample metadata from s_ and a_ ISA-tab files.
    """
    meta_parts: list[pd.DataFrame] = []

    for name, df in sample_tables.items():
        sample_col = _find_column(df, [r"^sample name$", r"sample"])
        if sample_col is None:
            sample_col = df.columns[0]

        tmp = df.copy()
        tmp.columns = [_normalize_colname(c) for c in tmp.columns]
        sample_col_norm = _normalize_colname(sample_col)

        tmp = tmp.rename(columns={sample_col_norm: "sample_id"})
        if "sample_id" not in tmp.columns:
            tmp = tmp.rename(columns={tmp.columns[0]: "sample_id"})

        tmp["sample_id"] = tmp["sample_id"].astype(str).str.strip()
        tmp = tmp[tmp["sample_id"].notna() & (tmp["sample_id"] != "")]
        meta_parts.append(tmp)

    if meta_parts:
        meta = pd.concat(meta_parts, axis=0, ignore_index=True)
        meta = meta.drop_duplicates(subset=["sample_id"])
    else:
        meta = pd.DataFrame(columns=["sample_id"])

    # Assay fájlokból további oszlopokat is hozzáfűzünk, ha Sample Name alapján lehet.
    for name, assay in assay_tables.items():
        sample_col = _find_column(assay, [r"^sample name$", r"sample"])
        if sample_col is None:
            continue

        tmp = assay.copy()
        tmp.columns = [_normalize_colname(c) for c in tmp.columns]
        sample_col_norm = _normalize_colname(sample_col)
        tmp = tmp.rename(columns={sample_col_norm: "sample_id"})
        tmp["sample_id"] = tmp["sample_id"].astype(str).str.strip()
        tmp = tmp.drop_duplicates(subset=["sample_id"])

        if meta.empty:
            meta = tmp
        else:
            add_cols = [c for c in tmp.columns if c != "sample_id" and c not in meta.columns]
            if add_cols:
                meta = meta.merge(tmp[["sample_id"] + add_cols], on="sample_id", how="left")

    if meta.empty:
        return meta

    target_col = _infer_target_column(meta)
    if target_col is not None:
        meta["lung_cancer_status"] = meta[target_col].map(_standardize_lung_cancer_label)
        meta["class_label"] = meta["lung_cancer_status"]
        print(f"Feltételezett target oszlop: {target_col}")
        print(meta["class_label"].value_counts(dropna=False))
    else:
        print("Figyelem: nem találtam automatikusan target/class oszlopot.")
        print("Elérhető meta oszlopok:")
        print(list(meta.columns))

    meta = meta.set_index("sample_id", drop=True)
    return meta


def _make_unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []

    for name in names:
        clean = re.sub(r"\s+", " ", str(name).strip())
        if not clean:
            clean = "unknown_feature"

        if clean not in seen:
            seen[clean] = 0
            out.append(clean)
        else:
            seen[clean] += 1
            out.append(f"{clean}__dup{seen[clean]}")

    return out


def _feature_name_from_rows(df: pd.DataFrame, non_sample_cols: list[str]) -> list[str]:
    """
    Construct usable feature names from metabolite/feature columns.
    """
    preferred = [
        r"metabolite identification",
        r"metabolite",
        r"compound",
        r"feature",
        r"mass.?to.?charge",
        r"m/z",
        r"mz",
        r"retention",
    ]

    chosen: list[str] = []
    for pat in preferred:
        c = _find_column(df[non_sample_cols], [pat])
        if c is not None and c not in chosen:
            chosen.append(c)

    if not chosen:
        chosen = non_sample_cols[:1]

    # Ha van m/z és retention time, kombináljuk őket.
    values: list[str] = []
    for _, row in df.iterrows():
        parts = []
        for c in chosen[:3]:
            v = row.get(c)
            if pd.notna(v) and str(v).strip() and str(v).strip().lower() != "nan":
                parts.append(f"{c}={str(v).strip()}")
        values.append(" | ".join(parts) if parts else "feature")

    return _make_unique(values)


def _table_to_feature_matrix(
    raw_df: pd.DataFrame,
    sample_ids: list[str],
    *,
    prefix: str,
) -> pd.DataFrame:
    """
    Convert a MetaboLights processed/m_ table to samples x features.

    Supports two common layouts:
    A) rows = features, columns = samples
    B) rows = samples, columns = features
    """
    df = raw_df.copy()
    df.columns = [_normalize_colname(c) for c in df.columns]

    sample_id_set = set(map(str, sample_ids))

    # A) feature rows, sample columns
    sample_cols = [c for c in df.columns if str(c).strip() in sample_id_set]
    if len(sample_cols) >= 10:
        non_sample_cols = [c for c in df.columns if c not in sample_cols]
        feature_names = _feature_name_from_rows(df, non_sample_cols)
        feature_names = [f"{prefix}__{x}" for x in feature_names]

        values = df[sample_cols].apply(pd.to_numeric, errors="coerce")
        values.index = _make_unique(feature_names)

        X = values.T
        X.index.name = "sample_id"
        return X

    # B) sample rows, feature columns
    sample_col = _find_column(df, [r"^sample name$", r"sample_id", r"sample"])
    if sample_col is not None:
        overlap = df[sample_col].astype(str).isin(sample_id_set).sum()
        if overlap >= 10:
            tmp = df.copy()
            tmp[sample_col] = tmp[sample_col].astype(str).str.strip()
            tmp = tmp[tmp[sample_col].isin(sample_id_set)]

            numeric_cols = []
            for c in tmp.columns:
                if c == sample_col:
                    continue
                converted = pd.to_numeric(tmp[c], errors="coerce")
                if converted.notna().sum() >= max(5, int(0.2 * len(tmp))):
                    numeric_cols.append(c)

            if not numeric_cols:
                raise ValueError(f"Nincs elég numerikus feature oszlop ebben a táblában: {prefix}")

            X = tmp.set_index(sample_col)[numeric_cols].apply(pd.to_numeric, errors="coerce")
            X.columns = _make_unique([f"{prefix}__{c}" for c in X.columns])
            X.index.name = "sample_id"
            return X

    raise ValueError(
        f"Nem ismertem fel feature-mátrixként ezt a táblát: {prefix}. "
        f"Oszlopok első 20 eleme: {list(df.columns[:20])}"
    )


def create_metabolomics_feature_matrix(
    metabol_data: dict[str, object],
    *,
    join: Literal["inner", "outer"] = "inner",
    return_metadata: bool = True,
    min_samples: int = 50,
    min_features: int = 5,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    MetaboLights/MTBLS28 processed tables → one samples x features matrix.

    A név szándékosan megmaradhat a régi notebook-kompatibilitás miatt,
    de itt nem feltétlenül 'pos' és 'neg' Workbench Results.txt fájlokat kombinálunk,
    hanem az elérhető m_/processed MetaboLights táblákat.

    Returns
    -------
    X:
        samples x features matrix
    meta:
        sample metadata aligned to X.index
    """
    meta = metabol_data["meta"]
    if not isinstance(meta, pd.DataFrame) or meta.empty:
        raise ValueError("Nincs használható sample metadata. Ellenőrizd az s_/a_ fájlokat.")

    sample_ids = list(meta.index.astype(str))
    matrix_tables = metabol_data["matrix_tables"]
    if not isinstance(matrix_tables, dict) or not matrix_tables:
        raise ValueError("Nem találtam m_/processed matrix táblákat.")

    matrices: list[pd.DataFrame] = []

    for name, raw_df in matrix_tables.items():
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
            continue

        prefix = Path(name).stem.replace(" ", "_")
        try:
            X_one = _table_to_feature_matrix(raw_df, sample_ids, prefix=prefix)
        except Exception as exc:
            print(f"Kihagyva, nem feature-mátrix: {name} ({exc})")
            continue

        if X_one.shape[0] < min_samples or X_one.shape[1] < min_features:
            print(f"Kihagyva, túl kicsi mátrix: {name} {X_one.shape}")
            continue

        print(f"Feature matrix felismerve: {name} {X_one.shape}")
        matrices.append(X_one)

    if not matrices:
        raise RuntimeError(
            "Nem sikerült automatikusan feature-mátrixot készíteni. "
            "Nyomtasd ki a metabol_data['matrix_tables'].keys() listát, "
            "és nézd meg az egyes táblák első sorait."
        )

    X = matrices[0]
    for X_next in matrices[1:]:
        X = X.join(X_next, how=join)

    # Teljesen üres feature-ök törlése
    X = X.dropna(axis=1, how="all")

    if not return_metadata:
        return X

    meta_aligned = meta.reindex(X.index)
    return X, meta_aligned