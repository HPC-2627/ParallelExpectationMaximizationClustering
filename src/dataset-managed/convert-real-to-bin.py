#!/usr/bin/env python3
"""Convert the real datasets (data/real/<name>/raw/) to the binary format of the synthetic ones.

Output, in data/real/<name>/:
  <name>.bin         8-byte header (int32 N, int32 D, little-endian) + N*D float64, row-major
  <name>_labels.bin  N int32 (only when a usable label exists)
  <name>.json        columns used, dropped columns, mean/std used for the standardization

Preprocessing (the same used to decide COMPLETE/DISCARD in verify-real-datasets.py):
  - only numeric columns are kept, identifiers (ID_COLUMNS) are removed;
  - a dataset with empty cells in those columns is skipped (DISCARD), nothing is imputed;
  - constant columns are dropped (zero variance makes the covariance singular);
  - every column is standardized (z-score), as for the synthetic datasets.
Labels: the first column of targets.csv, encoded as 0..C-1 (sorted values). A numeric target
with more than MAX_CLASSES distinct values is continuous and is not exported.
brainweb: NIfTI volumes, not CSV: converted by convert-brainweb-to-bin.py.

WHY THIS FORMAT
The same binary layout as the synthetic datasets, so the C/MPI program reads real and synthetic
data with the same code: a single fread (or MPI-IO read) of contiguous row-major doubles, with
no text parsing. float64 and little-endian for the same reasons given in
generate-synthetic-dataset.py (accuracy of the sums/exp/log, x86 cluster nodes). The labels are
in a separate file so the EM cannot "see" the truth; they are used only to check the results.
The z-score constants (mean/std) are saved in the JSON to be able to go back to the original units.

Usage:  python3 src/dataset-managed/convert-real-to-bin.py [name ...]     (no names = all)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The script is in <repo>/src/dataset-managed: reads and writes <repo>/data/real
ROOT = Path(__file__).resolve().parents[2] / "data" / "real"

# Columns that are identifiers, not data (same list as verify-real-datasets.py).
ID_COLUMNS = {"encounter_id", "patient_nbr", "id", "ID", "subject#", "SegFile",
              "FileName", "Date"}
# A numeric target with more distinct values than this is treated as a continuous quantity
# (a regression target), not as a cluster label.
MAX_CLASSES = 20

# Datasets converted when no name is given on the command line (same list as the verify script).
DATASETS = ["diabetes130", "cdc_diabetes", "cardiovascular", "parkinson",
            "eeg_eye_state", "gene_rnaseq", "wdbc", "heart_failure",
            "cardiotocography", "statlog_heart", "cleveland"]


def write_bin(path, X):
    """Header (int32 N, int32 D) + float64 row-major, little-endian."""
    # Contiguous row-major copy: tofile() writes the memory as it is, so the layout must be
    # the one expected by the reader (row 0, row 1, ...).
    X = np.ascontiguousarray(X, dtype="<f8")
    with open(path, "wb") as f:
        np.array(X.shape, dtype="<i4").tofile(f)    # header: N, D
        X.tofile(f)


def standardize(X):
    """Z-score per column; returns (Z, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


def encode_labels(y):
    """First target column -> int32 0..C-1, or None when it is continuous."""
    col = y.iloc[:, 0]
    # Many distinct numeric values = continuous target: it is not a grouping, skip it
    if pd.api.types.is_numeric_dtype(col) and col.nunique() > MAX_CLASSES:
        return None, col.name
    # sort=True makes the encoding deterministic (the smallest value gets class 0)
    codes, uniques = pd.factorize(col, sort=True)
    if (codes < 0).any():                           # -1 = missing label: cannot be encoded
        return None, col.name
    return codes.astype("<i4"), {str(u): int(i) for i, u in enumerate(uniques)}


def save(name, X, cols, dropped, labels, label_info, extra=None):
    """Standardizes X and writes <name>.bin, <name>_labels.bin (if any) and <name>.json."""
    out = ROOT / name
    Z, mean, std = standardize(X)
    write_bin(out / f"{name}.bin", Z)
    if labels is not None:
        labels.astype("<i4").tofile(out / f"{name}_labels.bin")
    # The JSON documents the dataset: which columns were used/dropped, the constants of the
    # standardization (to invert it) and how the labels were encoded.
    info = {"name": name, "N": int(Z.shape[0]), "D": int(Z.shape[1]),
            "columns": cols, "dropped_columns": dropped,
            "mean": mean.tolist(), "std": std.tolist(),
            "labels": label_info, "format": "int32 N, int32 D, float64[N*D] row-major, little-endian"}
    if extra:
        info.update(extra)
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    lab = "no labels" if labels is None else f"{len(set(labels.tolist()))} classes"
    print(f"[ok]   {name}: {Z.shape[0]} x {Z.shape[1]}, {lab}, "
          f"{dropped and len(dropped) or 0} columns dropped")


def convert_csv(name):
    """Converts a dataset stored as raw/features.csv (+ raw/targets.csv)."""
    raw = ROOT / name / "raw"
    idx = 0 if name == "gene_rnaseq" else None      # first column = sample id
    X = pd.read_csv(raw / "features.csv", low_memory=False, index_col=idx)
    y = pd.read_csv(raw / "targets.csv", index_col=idx) if (raw / "targets.csv").exists() else None

    # Columns usable by a Gaussian mixture: numeric and not identifiers
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c]) and c not in ID_COLUMNS]
    non_num = [c for c in X.columns if c not in num]
    # No imputation: filling the gaps would invent data, so an incomplete dataset is skipped
    n_empty = int(X[num].isna().sum().sum())
    if n_empty:
        print(f"[skip] {name}: {n_empty} empty cells in the numeric columns (DISCARD)")
        return

    A = X[num].to_numpy(dtype=np.float64)
    # A constant column has zero variance: it would give a singular covariance (and a
    # division by zero in the z-score), so it is dropped
    constant = A.std(axis=0) == 0
    cols = [c for c, k in zip(num, constant) if not k]
    dropped = non_num + [c for c, k in zip(num, constant) if k]
    A = A[:, ~constant]

    labels, label_info = (None, None)
    if y is not None:
        labels, label_info = encode_labels(y)
        if labels is None:
            print(f"[note] {name}: target '{y.columns[0]}' is continuous, labels not exported")
            label_info = {"column": y.columns[0], "exported": False}
        else:
            label_info = {"column": y.columns[0], "exported": True, "classes": label_info}
    save(name, A, [str(c) for c in cols], [str(c) for c in dropped], labels, label_info)


def main():
    names = sys.argv[1:] or DATASETS
    for name in names:
        try:
            if name == "brainweb":
                print("[skip] brainweb: use convert-brainweb-to-bin.py (NIfTI volumes, not CSV)")
                continue
            convert_csv(name)
        except Exception as exc:                    # one failing dataset must not stop the others
            print(f"[ERR]  {name}: {exc}")


if __name__ == "__main__":
    main()
