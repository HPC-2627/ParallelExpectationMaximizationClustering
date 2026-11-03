#!/usr/bin/env python3
"""Download the real (medical) datasets into data/real/<name>/raw/.

Each dataset is saved as raw/features.csv (+ raw/targets.csv when a label
exists), exactly as served by the source, with NO preprocessing.
The script is idempotent: a dataset already downloaded is skipped.

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
    # Epileptic Seizure Recognition was removed from UCI at the curators' request:
    # replaced by the TCGA RNA-Seq cancer dataset (high D, complete).
    "gene_rnaseq":      ("uci_zip", 401),  # gene expression cancer RNA-Seq (801 x 20531)
    "parkinson":        ("uci", 189),    # Parkinsons Telemonitoring
    "eeg_eye_state":    ("uci", 264),    # EEG Eye State
    "wdbc":             ("uci", 17),     # Breast Cancer Wisconsin Diagnostic
    "heart_failure":    ("uci", 519),    # Heart Failure Clinical Records
    "cardiotocography": ("uci", 193),    # Cardiotocography
    "statlog_heart":    ("uci", 145),    # Statlog (Heart)
    "cleveland":        ("uci", 45),     # Heart Disease (Cleveland), for the missing-row count
}

# Direct zip URLs for UCI datasets that ucimlrepo cannot import.
UCI_ZIP_URLS = {
    "gene_rnaseq": "https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip",
}


def fetch_uci(ident):
    """Return (features, targets, metadata) from the UCI repository."""
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=ident)
    return d.data.features, d.data.targets, d.metadata


def fetch_openml_ds(ident):
    """Return (features, targets, metadata) from OpenML."""
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=ident, as_frame=True, parser="liac-arff")
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
    return raw.parent.name


def download(name):
    source, ident = DATASETS[name]
    raw = ROOT / name / "raw"
    if (raw / "features.csv").exists():
        print(f"[skip] {name}: already downloaded")
        return
    raw.mkdir(parents=True, exist_ok=True)
    if source == "uci_zip":
        download_uci_zip(ident, UCI_ZIP_URLS[name], raw)
        return
    X, y, meta = fetch_uci(ident) if source == "uci" else fetch_openml_ds(ident)
    X.to_csv(raw / "features.csv", index=False)
    if y is not None:
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
