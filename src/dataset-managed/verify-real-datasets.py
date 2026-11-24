#!/usr/bin/env python3
"""Verify the downloaded datasets: size, empty cells, masked missing values.

Reads data/real/<name>/raw/{features,targets}.csv and writes the report into
doc/datasets.md (section 3.2), between the markers BEGIN/END verify-real-datasets:
the rest of that file is written by hand and is left untouched. Verdict per dataset:
  COMPLETE  - no empty cell in the numeric columns used for clustering
  DISCARD  - empty cells (or masked ones) present in those columns

Empty = NaN, "", whitespace, "?", "NA", "N/A", "null" (case-insensitive).

WHY THIS CHECK
The EM (as implemented here) needs a complete numeric matrix: a missing value cannot enter the
Gaussian density, and imputing it would invent data and bias the clusters. So before using a
dataset it is checked whether it has empty cells in the columns that would be clustered; the
ones that do are DISCARDED, not repaired. The same rule is applied by convert-real-to-bin.py.
Missing values are often hidden: they come as "?" or as a 0 in a variable where 0 is
impossible. The script looks for the text tokens and reports the suspicious zeros, but a zero
is only reported (never counted as empty): deciding if it is a real value needs domain knowledge.
The report also gives size, constant columns (zero variance, useless for a Gaussian), duplicate
rows and the distribution of the labels, to choose and describe the datasets in the report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The script is in <repo>/src/dataset-managed: reads <repo>/data/real, writes <repo>/doc
REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "real"
REPORT = REPO / "doc" / "datasets.md"
BEGIN = "<!-- BEGIN verify-real-datasets -->"     # the report is written between these markers
END = "<!-- END verify-real-datasets -->"
# Strings that mean "missing" in the sources (compared after strip() and lower())
EMPTY_TOKENS = {"", "?", "na", "n/a", "nan", "null", "none"}

# Columns that are identifiers, not data: never used for clustering.
ID_COLUMNS = {"encounter_id", "patient_nbr", "id", "ID", "subject#", "SegFile",
              "FileName", "Date"}

# Datasets verified when no name is given on the command line
DATASETS = ["diabetes130", "cdc_diabetes", "cardiovascular", "parkinson",
            "eeg_eye_state", "gene_rnaseq", "heart_failure",
            "cardiotocography", "statlog_heart", "brainweb"]


def empty_mask(df):
    """Boolean frame: True where the cell is empty (NaN or an empty token)."""
    mask = df.isna()
    # Text columns only: a numeric column cannot contain "?" (it would not be numeric)
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            mask[col] |= df[col].astype(str).str.strip().str.lower().isin(EMPTY_TOKENS)
    return mask


def verify(name):
    """Analyses data/real/<name>/raw and returns the statistics (dict) used by render()."""
    raw = ROOT / name / "raw"
    # gene_rnaseq has the sample id in the first column: used as index, so it is not data
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

    # Only the empties in the numeric columns matter: the others are not used for clustering
    verdict = "COMPLETE" if num_empty == 0 else "DISCARD"
    return dict(name=name, n=n, d=d, n_num=len(num), n_binary=len(binary_cols),
                empty_total=int(empties.sum().sum()), empty_by_col=empty_by_col,
                num_empty=num_empty, incomplete_rows=incomplete_rows,
                suspicious=suspicious, const_cols=const_cols, n_dup=n_dup,
                label=None if y is None else
                {c: y[c].value_counts().head(8).to_dict() for c in y.columns},
                verdict=verdict)


def note(r):
    """One short line with what is worth knowing about a dataset (r = result of verify()), or ""."""
    parts = []
    if len(r["empty_by_col"]):                      # empties only in non-numeric columns (not used by EM)
        top = ", ".join(f"`{c}`" for c in list(r["empty_by_col"].index)[:3])
        parts.append(f"empty cells only in non-numeric columns ({top}, ...)")
    if r["suspicious"]:                             # exact zeros that could be masked missing values
        top = ", ".join(f"`{c}`" for c in list(r["suspicious"])[:3])
        parts.append(f"few exact zeros ({top}, ...)")
    if r["const_cols"]:
        parts.append(f"{len(r['const_cols'])} constant columns")
    return f"- **{r['name']}**: " + "; ".join(parts) if parts else ""


def verify_brainweb():
    """BrainWeb subject 44 (T1 only): voxel count, NaN, range, ground-truth labels."""
    import nibabel as nib                           # imported here: only this dataset needs it
    raw = ROOT / "brainweb" / "raw"
    t1 = np.asarray(nib.load(raw / "subject44_t1w.nii.gz").dataobj)
    lab = np.asarray(nib.load(raw / "brainweb_s44_crisp.nii.gz").dataobj)
    nan_t1 = int(np.isnan(t1.astype(float)).sum())
    n_vox = int(t1.size)
    verdict = "COMPLETE" if nan_t1 == 0 else "DISCARD"
    md = [f"- **brainweb**: T1 only (D = 1), range {t1.min()}-{t1.max()}; the ground-truth labels "
          f"({int(lab.min())}-{int(lab.max())}, {' x '.join(map(str, lab.shape))}, 0.5 mm) are on a finer grid "
          "than the T1 volume: `convert-brainweb-to-bin.py` subsamples them and places them on the "
          "T1 grid."]
    # Same keys used by the summary table in main() (the other fields do not apply to a volume)
    row = dict(name="brainweb", n=n_vox, d=1, n_num=1, num_empty=nan_t1,
               verdict=verdict, n_dup="n/a")   # duplicates not computed for a volume
    return row, "\n".join(md)


def main():
    names = sys.argv[1:] or DATASETS
    out = []
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
            extra.append(f"- **brainweb**: ERROR, {exc}")
            print(f"[ERR] brainweb: {exc}")
    for name in names:
        try:
            r = verify(name)
        except Exception as exc:
            out += [f"- **{name}**: ERROR, {exc}"]
            print(f"[ERR] {name}: {exc}")
            continue
        out += [note(r)] if note(r) else []
        rows.append(r)                              # kept for the summary table
        print(f"[{r['verdict']}] {name}: {r['n']}x{r['d']}, numeric={r['n_num']}, "
              f"empty(numeric)={r['num_empty']}, dup={r['n_dup']}")
    out += extra
    rows.sort(key=lambda r: r["name"] == "brainweb")   # brainweb last in the table
    # Summary table first, then the short notes
    table = ("| Dataset | Raw N x D | Numeric | Empty (numeric) | Duplicate rows | Verdict |\n|---|---|---|---|---|---|\n"
             + "\n".join(f"| `{r['name']}` | "
                         f"{r['n']} x {r['d']} | {r['n_num']} | {r['num_empty']} | {r['n_dup']} | {r['verdict']} |"
                         for r in rows))
    out = [table, "", "Notes:", ""] + out
    write_report("\n".join(out).rstrip("\n"))


def write_report(body):                         # body: in, markdown of the report
    """Replace the text between BEGIN and END in doc/datasets.md with body."""
    text = REPORT.read_text()
    i, j = text.find(BEGIN), text.find(END)
    if i < 0 or j < i:
        sys.exit(f"markers {BEGIN} / {END} not found in {REPORT}")
    REPORT.write_text(text[:i + len(BEGIN)] + "\n" + body + "\n" + text[j:])


if __name__ == "__main__":
    main()
