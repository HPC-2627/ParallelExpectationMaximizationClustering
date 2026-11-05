#!/usr/bin/env python3
"""Convert the BrainWeb MRI volumes (.nii.gz) to the binary format used by the C/MPI program.

Pipeline (three steps, each one visible on disk in data/real/brainweb/):
  1. raw/*.nii.gz  --gunzip-->  nifti/*.nii      (plain NIfTI-1 volumes, kept for inspection)
  2. nifti/*.nii   --nibabel--> in-memory arrays (T1 = data, crisp = ground-truth labels)
  3. arrays        --------->   brainweb.bin, brainweb_labels.bin, brainweb.json

Output, in data/real/brainweb/ (same layout as convert-real-to-bin.py and the synthetic data):
  brainweb.bin         8-byte header (int32 N, int32 D, little-endian) + N*D float64, row-major
  brainweb_labels.bin  N int32, one tissue label (0..11) per T1 voxel
  brainweb.json        columns, mean/std of the z-score, label info, volume shape

How the matrix is built:
  - N = number of voxels of the T1 volume (181*256*256 = 11,862,016), D = 1 (T1 intensity);
    voxel order = C order of the volume (the last NIfTI axis varies fastest);
  - the intensity is standardized (z-score); mean/std are saved in the JSON to invert it.

How the labels are built:
  The crisp labels live on a finer grid (0.5 mm, 362x434x362) than the T1 (1 mm, 181x256x256).
  They are subsampled by 2 (every 2nd voxel -> 181x217x181) and placed inside the T1 grid at the
  offset found by cross-correlating the two brain masks (FFT). The offset is NOT taken from the
  NIfTI affines: in these files they are inconsistent (using them, the labels explain 18% of the
  T1 variance; with the estimated offset 95%). The script prints this share (eta^2) as a check.
  T1 voxels outside the labels get label 0 (background). The labels are in a separate file so
  that the EM cannot "see" the truth: they are used only to check the clustering (ARI / Dice).

Requirements: Python 3 with numpy and nibabel (pip install numpy nibabel).

Usage:  python3 src/dataset-managed/convert-brainweb-to-bin.py [--t1 FILE] [--labels FILE] [--force]
  no arguments  = uses the files downloaded by generate-real-datasets.py in data/real/brainweb/raw/
  --force       = regenerates the .nii and the .bin even if they already exist
"""
import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy and nibabel are missing: pip install numpy nibabel")

# The script is in <repo>/src/dataset-managed: reads and writes <repo>/data/real/brainweb
BRAIN = Path(__file__).resolve().parents[2] / "data" / "real" / "brainweb"
RAW = BRAIN / "raw"
NIFTI = BRAIN / "nifti"
T1_NAME = "subject44_t1w"             # file names produced by generate-real-datasets.py
LABELS_NAME = "brainweb_s44_crisp"


def gunzip(src, dst, force):
    """Step 1: decompress a .nii.gz into a .nii (skipped if dst already exists)."""
    # src, dst: Path /* in */ ; force: bool /* in */
    if dst.exists() and not force:
        print(f"[skip] {dst.name} already exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)          # streamed: no need to hold the volume in memory
    print(f"[ok]   {src.name} -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


def standardize(X):
    """Z-score per column; returns (Z, mean, std)."""
    # X: ndarray (N, D) /* in */
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std, mean, std


def write_bin(path, X):
    """Header (int32 N, int32 D) + float64 row-major, little-endian."""
    # path: Path /* in */ ; X: ndarray (N, D) /* in */
    X = np.ascontiguousarray(X, dtype="<f8")   # tofile() writes the memory as it is
    with open(path, "wb") as f:
        np.array(X.shape, dtype="<i4").tofile(f)
        X.tofile(f)


def find_offset(t1, lab_small):
    """Position of the (downsampled) label volume inside the T1 grid, in voxels."""
    # t1: ndarray /* in */ ; lab_small: ndarray, labels on the 1 mm grid /* in */
    # Brain masks: T1 above its median vs label > 0. The offset that maximizes their
    # cross-correlation (computed with the FFT) is where the labels sit inside the T1.
    mt = (t1 > np.median(t1)).astype(np.float32)
    mc = (lab_small > 0).astype(np.float32)
    mt -= mt.mean()
    mc -= mc.mean()
    shape = t1.shape
    cor = np.fft.irfftn(np.fft.rfftn(mt) * np.conj(np.fft.rfftn(mc, s=shape, axes=(0, 1, 2))),
                        s=shape, axes=(0, 1, 2))
    # Only offsets that keep the whole label volume inside the T1 are valid.
    valid = np.zeros(shape, dtype=bool)
    valid[tuple(slice(0, t - c + 1) for t, c in zip(shape, lab_small.shape))] = True
    cor[~valid] = -np.inf
    return np.unravel_index(np.argmax(cor), shape)


def resample_labels(lab_img, t1_img, t1):
    """Step 2b: labels of the crisp volume on the T1 grid (one label per T1 voxel)."""
    # lab_img, t1_img: nibabel images /* in */ ; t1: ndarray (T1 voxels) /* in */
    # returns int32 array with the T1 shape
    lab = np.asarray(lab_img.dataobj)
    # 0.5 mm -> 1 mm: integer ratio between the voxel sizes (2), take every s-th label voxel
    s = int(round(float(t1_img.header.get_zooms()[0]) / float(lab_img.header.get_zooms()[0])))
    lab_small = lab[::s, ::s, ::s]
    if any(c > t for c, t in zip(lab_small.shape, t1.shape)):
        raise ValueError(f"labels {lab_small.shape} do not fit in the T1 {t1.shape}")
    # The NIfTI affines of these files are NOT reliable: mapping the voxels with them leaves
    # the labels shifted (eta^2 = 0.18 below), so the position is estimated from the images.
    off = find_offset(t1, lab_small)
    out = np.zeros(t1.shape, dtype=np.int32)         # outside the labels = 0 (background)
    sl = tuple(slice(o, o + c) for o, c in zip(off, lab_small.shape))
    out[sl] = lab_small
    # Quality check: eta^2 = share of the T1 variance explained by the labels (1 = perfect).
    # A wrong alignment gives ~0.2, a correct one ~0.95.
    m = t1.mean()
    between = sum((out == l).sum() * (t1[out == l].mean() - m) ** 2 for l in np.unique(out))
    eta2 = float(between / t1.size / t1.var())
    print(f"[ok]   labels step {s}, offset (voxels) {tuple(int(o) for o in off)}, "
          f"eta^2 = {eta2:.3f}")
    if eta2 < 0.8:
        print("[warn] low eta^2: labels and T1 are probably not aligned, check the offset")
    return out


def main():
    ap = argparse.ArgumentParser(description="BrainWeb .nii.gz -> .nii -> .bin")
    ap.add_argument("--t1", type=Path, default=RAW / f"{T1_NAME}.nii.gz", help="T1 volume (.nii.gz)")
    ap.add_argument("--labels", type=Path, default=RAW / f"{LABELS_NAME}.nii.gz",
                    help="crisp labels (.nii.gz)")
    ap.add_argument("--force", action="store_true", help="redo every step")
    args = ap.parse_args()

    try:
        import nibabel as nib
    except ImportError:
        sys.exit("nibabel is missing: pip install numpy nibabel")
    for f in (args.t1, args.labels):
        if not f.exists():
            sys.exit(f"missing {f}: run  python3 src/dataset-managed/generate-real-datasets.py brainweb")

    # Step 1: .nii.gz -> .nii
    t1_nii = NIFTI / f"{T1_NAME}.nii"
    lab_nii = NIFTI / f"{LABELS_NAME}.nii"
    gunzip(args.t1, t1_nii, args.force)
    gunzip(args.labels, lab_nii, args.force)

    # Step 2: .nii -> arrays
    t1_img, lab_img = nib.load(t1_nii), nib.load(lab_nii)
    t1 = np.asarray(t1_img.dataobj)
    if np.isnan(t1.astype(float)).any():
        sys.exit("NaN in the T1 volume: dataset discarded")
    X = t1.reshape(-1, 1).astype(np.float64)       # one row per voxel, D = 1
    labels = resample_labels(lab_img, t1_img, t1.astype(np.float64)).reshape(-1)
    if labels.size != X.shape[0]:                   # safety check: one label per row
        sys.exit("labels size does not match the T1 size")

    # Step 3: arrays -> .bin / _labels.bin / .json
    Z, mean, std = standardize(X)
    write_bin(BRAIN / "brainweb.bin", Z)
    labels.astype("<i4").tofile(BRAIN / "brainweb_labels.bin")
    info = {"name": "brainweb", "N": int(Z.shape[0]), "D": int(Z.shape[1]),
            "columns": ["t1"], "dropped_columns": [],
            "mean": mean.tolist(), "std": std.tolist(),
            "labels": {"exported": True, "classes": sorted(np.unique(labels).tolist()),
                       "note": "crisp labels subsampled by 2 and placed on the T1 grid at the offset "
                               "found by mask cross-correlation; voxels outside = 0"},
            "format": "int32 N, int32 D, float64[N*D] row-major, little-endian",
            "volume_shape": list(t1.shape)}
    (BRAIN / "brainweb.json").write_text(json.dumps(info, indent=1))
    print(f"[ok]   brainweb: {Z.shape[0]} x {Z.shape[1]}, {len(info['labels']['classes'])} classes "
          f"-> {BRAIN}")


if __name__ == "__main__":
    main()
