# Data sources

Raw files are in `data/real/<name>/raw/` (not versioned, re-created by `src/dataset-managed/generate-real-datasets.py`).
Downloaded 2026-11-02. Verification results: see `doc/real-datasets-info.md`.

| Folder | Dataset | N x D (verified) | License | URL |
|---|---|---|---|---|
| `diabetes130` | Diabetes 130-US hospitals 1999-2008 | 101766 x 47 (11 numeric) | CC BY 4.0 | https://archive.ics.uci.edu/dataset/296 |
| `cdc_diabetes` | CDC Diabetes Health Indicators | 253680 x 21 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/891 (DOI 10.24432/C53919) |
| `cardiovascular` | Cardiovascular Disease (Kaggle, via OpenML) | 70000 x 11 | see source | https://www.openml.org/d/45547 |
| `parkinson` | Parkinsons Telemonitoring | 5875 x 19 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/189 |
| `eeg_eye_state` | EEG Eye State | 14980 x 14 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/264 |
| `gene_rnaseq` | Gene expression cancer RNA-Seq (TCGA PANCAN HiSeq) | 801 x 20531 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/401 |
| `heart_failure` | Heart Failure Clinical Records | 299 x 12 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/519 |
| `cardiotocography` | Cardiotocography | 2126 x 21 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/193 |
| `statlog_heart` | Statlog (Heart) | 270 x 13 | CC BY 4.0 | https://archive.ics.uci.edu/dataset/145 |
| `brainweb` | BrainWeb simulated brain MRI (subject 44) | 11,862,016 voxels x 1 (T1 only; T2/T2* dropped) | academic use | https://brainweb.bic.mni.mcgill.ca/ |

Licenses marked CC BY 4.0 are the ones stated by the UCI pages; re-check each page before the final report.

## Citations (to complete in the report)
- BrainWeb (T1 volume + crisp labels as ground truth): Cocosco et al. 1997; Kwan, Evans, Pike 1999 (IEEE TMI); Aubert-Broche et al. 2006 for the 20 anatomical models.
- UCI datasets: cite the DOI/page above for each one.
- Epileptic Seizure Recognition was planned but removed from UCI at the curators' request; replaced by `gene_rnaseq`.
