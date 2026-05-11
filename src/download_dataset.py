from __future__ import annotations

import re
import shutil
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd

from src.config import DATA_DIR, GDRIVE_DATA

SAMPLE_COL_RE = re.compile(r"^S\d+$")

RESULT_TXT = {
    "neg": "ST000816_AN001293_Results.txt",
    "pos": "ST000816_AN001294_Results.txt",
}
MWB_DOWNLOAD = "https://www.metabolomicsworkbench.org/studydownload/"


def _download_if_missing(url: str, out_path: Path, *, timeout: int = 120) -> Path:
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


def _ensure_result_in_gdrive(gdrive_dir: Path, mode: Literal["neg", "pos"]) -> Path:
    """GDRIVE_DATA-ban lévő .txt, ha üres/hiányzik → letöltés."""
    name = RESULT_TXT[mode]
    path = gdrive_dir / name
    if path.exists() and path.stat().st_size > 0:
        print(f"GDRIVE_DATA: megvan {path}")
        return path
    url = f"{MWB_DOWNLOAD}{name}"
    return _download_if_missing(url, path)


def _copy_txt_results_to_data_dir(gdrive_dir: Path, data_dir: Path) -> None:
    """A két Results.txt mindig DATA_DIR-be (felülírás)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in RESULT_TXT.values():
        src = gdrive_dir / name
        dst = data_dir / name
        shutil.copy2(src, dst)
        print(f"Másolva → {dst}")


def _read_result_table(path: Path) -> pd.DataFrame:
    """
    Read one ST000816 Results file as a raw Workbench table.

    Note: the file is tab-separated (.txt from Workbench).
    Rows are lipids/features; columns are samples. The first row is usually 'Factors'.
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def download_dataset(
    gdrive_data: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    1) GDRIVE_DATA: hiányzó/üres .txt → letöltés a Workbench-ről.
    2) A két fájl mindig átmásolódik DATA_DIR-be (meglévő felülírva).
    3) Beolvasás DATA_DIR-ből.
    """
    gdrive_dir = Path(gdrive_data or GDRIVE_DATA)
    work_dir = Path(data_dir or DATA_DIR)
    gdrive_dir.mkdir(parents=True, exist_ok=True)

    _ensure_result_in_gdrive(gdrive_dir, "neg")
    _ensure_result_in_gdrive(gdrive_dir, "pos")
    _copy_txt_results_to_data_dir(gdrive_dir, work_dir)

    neg_path = work_dir / RESULT_TXT["neg"]
    pos_path = work_dir / RESULT_TXT["pos"]
    return {
        "neg": _read_result_table(neg_path),
        "pos": _read_result_table(pos_path),
    }


def _parse_factor_string(factor_text: str) -> dict[str, str]:
    """Parse strings like 'Progressors:Non-progressor | Visit:Baseline'."""
    out = {}
    if pd.isna(factor_text):
        return out

    for item in str(factor_text).split("|"):
        item = item.strip()
        if ":" in item:
            key, value = item.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _extract_sample_metadata(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sample_id, progressor_status and visit from the 'Factors' row.
    This works directly from the Results file; no separate JSON metadata is required.
    """
    feature_col = raw_df.columns[0]
    sample_cols = [c for c in raw_df.columns if SAMPLE_COL_RE.match(str(c))]

    factor_rows = raw_df[raw_df[feature_col].astype(str).str.strip().eq("Factors")]
    if factor_rows.empty:
        return pd.DataFrame({"sample_id": sample_cols})

    factor_row = factor_rows.iloc[0]
    rows = []
    for sample_id in sample_cols:
        factors = _parse_factor_string(factor_row[sample_id])
        rows.append(
            {
                "sample_id": sample_id,
                "progressor_status": factors.get("Progressors"),
                "visit": factors.get("Visit"),
            }
        )
    return pd.DataFrame(rows)


def _make_one_mode_feature_matrix(raw_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Convert one raw Workbench result table to a samples x features matrix.

    Input shape:
        rows = features/lipids + one 'Factors' row
        columns = Sample, S00017985, S00017986, ...

    Output shape:
        rows = samples
        columns = prefixed lipid features, e.g. neg__CL 70:5; [M-2H](2-)@6.12
    """
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    feature_col = df.columns[0]
    sample_cols = [c for c in df.columns if SAMPLE_COL_RE.match(str(c))]
    if not sample_cols:
        raise ValueError("Nem találtam S000... formátumú mintaoszlopokat.")

    # Remove the metadata/factors row before numeric conversion.
    df = df[~df[feature_col].astype(str).str.strip().eq("Factors")].copy()

    feature_names = (
        df[feature_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
    feature_names = [f"{prefix}__{name}" for name in feature_names]

    # Ensure column names are unique even if the same lipid name appears twice.
    seen: dict[str, int] = {}
    unique_names = []
    for name in feature_names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}__dup{seen[name]}")

    values = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    values.index = unique_names

    X = values.T
    X.index.name = "sample_id"
    return X


def combine_pos_neg_to_feature_matrix(
    metabol_data: dict[str, pd.DataFrame],
    *,
    join: Literal["inner", "outer"] = "inner",
    return_metadata: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine positive and negative ion mode tables into one feature matrix.

    Parameters
    ----------
    metabol_data:
        Output of download_dataset(), with keys 'pos' and 'neg'.
    join:
        'inner' keeps only samples present in both modes. For ST000816 this should be all 100.
        'outer' keeps the union of samples.
    return_metadata:
        If True, also return metadata extracted from the Factors row.

    Returns
    -------
    X : pd.DataFrame
        samples x features, raw peak-area values.
    meta : pd.DataFrame, optional
        sample metadata aligned to X.index: sample_id, progressor_status, visit.
    """
    neg_raw = metabol_data["neg"]
    pos_raw = metabol_data["pos"]

    X_neg = _make_one_mode_feature_matrix(neg_raw, prefix="neg")
    X_pos = _make_one_mode_feature_matrix(pos_raw, prefix="pos")

    X = X_neg.join(X_pos, how=join)

    if not return_metadata:
        return X

    # The factors are the same in both files, so the negative-mode file is enough.
    meta = _extract_sample_metadata(neg_raw).set_index("sample_id")
    meta = meta.reindex(X.index)

    return X, meta
