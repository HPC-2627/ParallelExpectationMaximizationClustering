#!/usr/bin/env python3
"""Verify the downloaded datasets: size, empty cells, masked missing values.

Reads data/real/<name>/raw/{features,targets}.csv and writes
doc/real-datasets-info.md. Verdict per dataset:
  COMPLETE  - no empty cell in the numeric columns used for clustering
  DISCARD  - empty cells (or masked ones) present in those columns

Empty = NaN, "", whitespace, "?", "NA", "N/A", "null" (case-insensitive).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The script is in <repo>/src/dataset-managed: reads <repo>/data/real, writes <repo>/doc
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "real"
REPORT = REPO / "doc" / "real-datasets-info.md"
EMPTY_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none"}

# Columns that are identifiers, not data: never used for clustering.
ID_COLUMNS = {"encounter_id", "patient_nbr", "id", "ID", "subject#", "SegFile",
              "FileName", "Date"}

DATASETS = ["diabetes130", "cdc_diabetes", "cardiovascular", "parkinson",
            "eeg_eye_state", "gene_rnaseq", "wdbc", "heart_failure",
            "cardiotocography", "statlog_heart", "cleveland", "brainweb"]


def empty_mask(df):
    """Boolean frame: True where the cell is empty (NaN or an empty token)."""
    mask = df.isna()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            mask[col] |= df[col].astype(str).str.strip().str.lower().isin(EMPTY_TOKENS)
    return mask


def verify(name):
    raw = ROOT / name / "raw"
    X = pd.read_csv(raw / "features.csv", low_memory=False,
                    index_col=0 if name == "gene_rnaseq" else None)
    y = pd.read_csv(raw / "targets.csv", index_col=0 if name == "gene_rnaseq" else None) \
        if (raw / "targets.csv").exists() else None

    n, d = X.shape
    empties = empty_mask(X)
    empty_by_col = empties.sum()
    empty_by_col = empty_by_col[empty_by_col > 0].sort_values(ascending=False)

    # Numeric columns usable by a Gaussian mixture (ids removed).
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c]) and c not in ID_COLUMNS]
    num_empty = int(empties[num].sum().sum())
    incomplete_rows = int(empties[num].any(axis=1).sum())

    # Masked missing: non-binary numeric columns that contain exact zeros.
    zero_cols = {c: int((X[c] == 0).sum()) for c in num
                 if X[c].nunique() > 2 and (X[c] == 0).any()}
    # Only suspicious if few zeros (a real zero count is typical in sparse data).
    suspicious = {c: z for c, z in zero_cols.items() if z / n < 0.2}

    const_cols = [c for c in num if X[c].nunique(dropna=True) <= 1]
    n_dup = int(X.duplicated().sum())
    binary_cols = [c for c in num if X[c].dropna().nunique() == 2]

    verdict = "COMPLETE" if num_empty == 0 else "DISCARD"
    return dict(name=name, n=n, d=d, n_num=len(num), n_binary=len(binary_cols),
                empty_total=int(empties.sum().sum()), empty_by_col=empty_by_col,
                num_empty=num_empty, incomplete_rows=incomplete_rows,
                suspicious=suspicious, const_cols=const_cols, n_dup=n_dup,
                label=None if y is None else
                {c: y[c].value_counts().head(8).to_dict() for c in y.columns},
                verdict=verdict)


def render(r):
    lines = [f"## {r['name']} — {r['verdict']}", "",
             f"- Size: **{r['n']} x {r['d']}** ({r['n_num']} numeric columns, "
             f"{r['n_binary']} of them binary)",
             f"- Empty cells (all columns): {r['empty_total']}",
             f"- Empty cells in numeric columns: **{r['num_empty']}** "
             f"({r['incomplete_rows']} incomplete rows)"]
    if len(r["empty_by_col"]):
        top = ", ".join(f"`{c}`={v}" for c, v in r["empty_by_col"].head(8).items())
        lines.append(f"- Columns with empties: {top}")
    if r["suspicious"]:
        top = ", ".join(f"`{c}`={v}" for c, v in list(r["suspicious"].items())[:8])
        lines.append(f"- Few exact zeros (possible masked missing): {top}")
    if r["const_cols"]:
        lines.append(f"- Constant columns: {r['const_cols'][:8]}")
    lines.append(f"- Duplicate rows: {r['n_dup']}")
    if r["label"]:
        for col, dist in r["label"].items():
            lines.append(f"- Label `{col}`: {dist}")
    return "\n".join(lines) + "\n"


def verify_brainweb():
    """BrainWeb subject 44 (T1 only): voxel count, NaN, range, ground-truth labels."""
    import nibabel as nib
    raw = ROOT / "brainweb" / "raw"
    t1 = np.asarray(nib.load(raw / "subject44_t1w.nii.gz").dataobj)
    lab = np.asarray(nib.load(raw / "brainweb_s44_crisp.nii.gz").dataobj)
    nan_t1 = int(np.isnan(t1.astype(float)).sum())
    n_vox = int(t1.size)
    verdict = "COMPLETE" if nan_t1 == 0 else "DISCARD"
    md = [f"## brainweb (subject 44, T1 only) — {verdict}", "",
          f"- `subject44_t1w.nii.gz` (data): shape {' x '.join(map(str, t1.shape))} "
          f"(1 mm), {t1.dtype}, **{n_vox:,} voxels** with D = 1, {nan_t1} NaN, "
          f"range {t1.min()}-{t1.max()}",
          f"- `brainweb_s44_crisp.nii.gz` (ground truth, labels {int(lab.min())}-{int(lab.max())}): "
          f"shape {' x '.join(map(str, lab.shape))} (0.5 mm), {lab.dtype}",
          "- The two volumes are on different grids: the labels must be resampled "
          "to the T1 grid before computing ARI/Dice.",
          "- Only T1 is used (D = 1). T2/T2* were dropped: the BrainWeb server "
          "download did not complete.", ""]
    row = dict(name="brainweb", n=n_vox, d=1, n_num=1, num_empty=nan_t1,
               verdict=verdict)
    return row, "\n".join(md)


def main():
    names = sys.argv[1:] or DATASETS
    out = ["# Dataset verification", "",
           "Generated by `src/dataset-managed/verify-real-datasets.py` on the raw files in `data/real/*/raw/`.",
           ""]
    rows = []
    extra = []
    if "brainweb" in names:                      # handled separately (3D volumes)
        names = [n for n in names if n != "brainweb"]
        try:
            brow, bmd = verify_brainweb()
            rows.append(brow)
            extra.append(bmd)
            print(f"[{brow['verdict']}] brainweb: {brow['n']:,} voxels, D=1, "
                  f"NaN={brow['num_empty']}")
        except Exception as exc:
            extra.append(f"## brainweb — ERROR\n\n- {exc}\n")
            print(f"[ERR] brainweb: {exc}")
    for name in names:
        try:
            r = verify(name)
        except Exception as exc:
            out += [f"## {name} — ERROR", "", f"- {exc}", ""]
            print(f"[ERR] {name}: {exc}")
            continue
        out += [render(r)]
        rows.append(r)
        print(f"[{r['verdict']}] {name}: {r['n']}x{r['d']}, numeric={r['n_num']}, "
              f"empty(numeric)={r['num_empty']}, dup={r['n_dup']}")
    out += extra
    rows.sort(key=lambda r: r["name"] == "brainweb")   # brainweb last in the table
    out.insert(4, "| Dataset | N x D | Numeric | Empty (numeric) | Verdict |\n|---|---|---|---|---|\n"
               + "\n".join(f"| {r['name']} | {r['n']} x {r['d']} | {r['n_num']} | "
                           f"{r['num_empty']} | {r['verdict']} |" for r in rows) + "\n")
    REPORT.write_text("\n".join(out))


if __name__ == "__main__":
    main()
