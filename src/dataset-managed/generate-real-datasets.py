#!/usr/bin/env python3
"""Download the real (medical) datasets into data/real/<name>/raw/.

Each dataset is saved as raw/features.csv (+ raw/targets.csv when a label
exists), exactly as served by the source, with NO preprocessing.
The script is idempotent: a dataset already downloaded is skipped.

WHY REAL DATASETS
The synthetic datasets (generate-synthetic-dataset.py) are used for the weak scaling, because
N can grow at will. The real ones are small, but they are needed to show that the EM works on
data that do not come from a Gaussian mixture. They are medical, like the synthetic profiles,
and cover different shapes: many rows (diabetes130, cardiovascular), high D (gene_rnaseq,
20531 variables), tiny sets (statlog_heart, heart_failure) and an image volume (brainweb, D = 1).
The raw files are kept untouched: all the choices (which columns, what to discard) are made
later and are documented by verify-real-datasets.py and convert-real-to-bin.py.

Sources: UCI Machine Learning Repository (through the ucimlrepo package), OpenML (through
scikit-learn), a direct zip for the UCI dataset that ucimlrepo cannot import, and the BrainWeb
simulator (through the brainweb-dl command line tool).

Usage:  python3 src/dataset-managed/generate-real-datasets.py [name ...]     (no names = all)
"""
import sys
from pathlib import Path

# The script is in <repo>/src/dataset-managed: the data goes to <repo>/data/real
ROOT = Path(__file__).resolve().parents[2] / "data" / "real"

# name -> (source, id). source "uci" = UCI ML Repository, "openml" = OpenML.
DATASETS = {
    "diabetes130":      ("uci", 296),    # Diabetes 130-US hospitals 1999-2008
    "cdc_diabetes":     ("uci", 891),    # CDC Diabetes Health Indicators
    "cardiovascular":   ("openml", 45547),  # Kaggle cardiovascular disease (70k)
    # TCGA RNA-Seq cancer dataset (high D, complete).
    "gene_rnaseq":      ("uci_zip", 401),  # gene expression cancer RNA-Seq (801 x 20531)
    "parkinson":        ("uci", 189),    # Parkinsons Telemonitoring
    "eeg_eye_state":    ("uci", 264),    # EEG Eye State
    "heart_failure":    ("uci", 519),    # Heart Failure Clinical Records
    "cardiotocography": ("uci", 193),    # Cardiotocography
    "statlog_heart":    ("uci", 145),    # Statlog (Heart)
}

# Direct zip URLs for UCI datasets that ucimlrepo cannot import.
UCI_ZIP_URLS = {
    "gene_rnaseq": "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip",
}


def fetch_uci(ident):
    """Return (features, targets, metadata) from the UCI repository."""
    from ucimlrepo import fetch_ucirepo             # imported here: needed only for UCI datasets
    d = fetch_ucirepo(id=ident)
    return d.data.features, d.data.targets, d.metadata


def fetch_openml_ds(ident):
    """Return (features, targets, metadata) from OpenML."""
    from sklearn.datasets import fetch_openml
    # as_frame=True: pandas DataFrames, so they can be saved as CSV as they are.
    # The "liac-arff" parser is pure Python (the default one needs extra dependencies).
    d = fetch_openml(data_id=ident, as_frame=True, parser="liac-arff")
    # d.target is a Series: converted to a DataFrame to have the same shape as the UCI targets
    target = d.target.to_frame() if d.target is not None else None
    return d.data, target, {"name": d.details.get("name"), "url": d.details.get("url")}


def download_uci_zip(ident, url, raw):
    """UCI datasets not importable via ucimlrepo: fetch the zip, unpack data.csv/labels.csv.

    Used for gene_rnaseq (zip containing a tar.gz with data.csv and labels.csv).
    The files are saved as features.csv / targets.csv (first column = sample id).
    """
    import io, shutil, tarfile, tempfile, urllib.request, zipfile
    with tempfile.TemporaryDirectory() as tmp:
        # timeout = max seconds of silence on the socket: fail instead of hanging forever
        data = urllib.request.urlopen(url, timeout=60).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        tar_path = next(Path(tmp).glob("*.tar.gz"))
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp)
        src = next(Path(tmp).rglob("data.csv")).parent
        shutil.move(str(src / "data.csv"), raw / "features.csv")
        shutil.move(str(src / "labels.csv"), raw / "targets.csv")
    print(f"[ok]   {name_of(raw)}: unpacked from {url} (uci {ident})")


def name_of(raw):
    """Dataset name from its raw folder (data/real/<name>/raw)."""
    return raw.parent.name


def download(name):
    """Downloads one CSV dataset into data/real/<name>/raw/, unless already there."""
    source, ident = DATASETS[name]
    raw = ROOT / name / "raw"
    # features.csv is written last, so its presence means the download was completed
    if (raw / "features.csv").exists():
        print(f"[skip] {name}: already downloaded")
        return
    raw.mkdir(parents=True, exist_ok=True)
    if source == "uci_zip":
        download_uci_zip(ident, UCI_ZIP_URLS[name], raw)
        return
    X, y, meta = fetch_uci(ident) if source == "uci" else fetch_openml_ds(ident)
    # index=False: the row number is not data. Nothing else is touched (no preprocessing).
    X.to_csv(raw / "features.csv", index=False)
    if y is not None:      # some datasets have no label
        y.to_csv(raw / "targets.csv", index=False)
    print(f"[ok]   {name}: features {X.shape}, targets "
          f"{None if y is None else y.shape} ({source} {ident})")


def download_brainweb():
    """BrainWeb subject 44: T1 volume (the data) + crisp labels (ground truth).

    Only T1 is used (D = 1): the T2/T2* downloads from the BrainWeb server were
    too slow and were dropped. Saved in brainweb/raw/ as .nii.gz volumes.
    """
    import subprocess
    raw = ROOT / "brainweb" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    cli = Path(sys.executable).parent / "brainweb-dl"   # installed in the same venv
    # "T1" is the MRI intensity (the data, one value per voxel); "crisp" is the discrete tissue
    # segmentation (background, CSF, grey matter, white matter...) used as ground truth
    for contrast, fname in (("T1", "subject44_t1w.nii.gz"),
                            ("crisp", "brainweb_s44_crisp.nii.gz")):
        if (raw / fname).exists():
            print(f"[skip] brainweb {contrast}: already downloaded")
            continue
        subprocess.run([str(cli), "44", "--contrast", contrast,
                        "--brainweb-dir", str(raw), "--extension", "nii.gz"],
                       check=True)
        print(f"[ok]   brainweb {contrast}: {fname}")


def main():
    # No names on the command line = all the datasets, brainweb included (it is not in DATASETS
    # because it is not a CSV and has its own download function)
    names = sys.argv[1:] or list(DATASETS) + ["brainweb"]
    for name in names:
        try:
            if name == "brainweb":
                download_brainweb()
                continue
            download(name)
        except Exception as exc:  # keep going: one failure must not stop the rest
            print(f"[FAIL] {name}: {exc}")


if __name__ == "__main__":
    main()
