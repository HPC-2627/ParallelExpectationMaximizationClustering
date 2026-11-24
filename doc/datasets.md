# Datasets

3 **synthetic** profiles, 5 synthetic **sweep** files and 10 **real** datasets. `data/` is not versioned: everything is generated or downloaded by the scripts in `src/dataset-managed/` (procedure in [exec-cluster.md](exec-cluster.md), which plot uses which dataset in [plots.md](plots.md)).

Plot codes in the tables: **P** = performance plots, **C** = clustering plots. $p$ = number of cores: every synthetic profile has one file per $p \in \{1,2,4,8,16\}$, suffix `_pNN`, with $250\,000 \cdot p$ rows (weak scaling on $p$ cores; the same file is reused for the strong scaling on any number of cores).

## 1. Overview

| Dataset | Type | N x D | True labels (K of the runs) | Used for |
|---|---|---|---|---|
| `synth_clean` | synthetic | 250 000·p x 10 | yes (4) | correctness, K sweep (P1), control case of C1/C2 |
| `synth_overlap` | synthetic | 250 000·p x 20 | yes (5) | weak scaling (P5), scaling plots of the diagonal model (P2-P4, P8, P10), C1 |
| `synth_rare` | synthetic | 250 000·p x 50 | yes (6, + outliers) | scaling plots of the full model (P2-P4, P8, P10), numerical stability |
| `synth_sweep_dDDD` | synthetic | 250 000 x D, D ∈ {5, 10, 20, 35, 50} | yes (4) | cost of an iteration vs D (P1) |
| `diabetes130` | real | 101 766 x 11 (numeric columns of 47) | `readmitted` (3) | clustering quality |
| `cdc_diabetes` | real | 253 680 x 21 | `Diabetes_binary` (2) | clustering quality |
| `cardiovascular` | real | 70 000 x 11 | `cardio` (2) | clustering quality |
| `parkinson` | real | 5 875 x 19 | none (K = 2, arbitrary) | clustering quality |
| `eeg_eye_state` | real | 14 980 x 14 | `eyeDetection` (2) | clustering quality |
| `gene_rnaseq` | real | 801 x 20 264 (267 constant columns dropped) | `Class` (5) | high D (diagonal model only) |
| `heart_failure` | real | 299 x 12 | `death_event` (2) | clustering quality |
| `cardiotocography` | real | 2 126 x 21 | `CLASS` (10) | clustering quality |
| `statlog_heart` | real | 270 x 13 | `heart-disease` (2) | clustering quality |
| `brainweb` | real (MRI) | 11 862 016 x 1 | tissue classes (12) | large N: scaling plots of both models (P2-P4, P8, P10), clustering quality |

## 2. Synthetic datasets

### 2.1 Why synthetic data

- The real datasets are small: 9 of 10 have at most 254 000 rows and run in milliseconds. `brainweb` is large, but its $N$ is fixed and cannot grow with $p$, as the weak scaling needs.
- The truth is known: the versions can be checked against each other and the clustering quality measured.
- 3 profiles (easy, realistic, hard) with different $K$ and $D$, so different cost per iteration.

### 2.2 The three profiles

| | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| Role | easy | realistic | hard |
| $K$ / $D$ | 4 / 10 | 5 / 20 | 6 / 50 |
| Cluster weights | 25% each | 30, 25, 20, 15, 10% | 40, 25, 18, 12, 4, 1% |
| Separation between cluster means (in std. dev.) | 6.0 | 2.5 (overlapping) | 4.0 |
| Covariance in a cluster | diagonal | correlated | diagonal |
| Outliers | 0% | 0% | 2% (label $-1$) |
| What it tests | correctness | slow convergence, correlations the diagonal model cannot capture | rare group, outliers, high $D$ |

### 2.3 Sizes

| $p$ (file `_pNN`) | Rows | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|---|
| 1 | 250 000 | 20 MB | 40 MB | 100 MB |
| 2 | 500 000 | 40 MB | 80 MB | 200 MB |
| 4 | 1 000 000 | 80 MB | 160 MB | 400 MB |
| 8 | 2 000 000 | 160 MB | 320 MB | 800 MB |
| 16 | 4 000 000 | 320 MB | 640 MB | 1 600 MB |

### 2.4 K and D sweeps

They change one parameter at a time to measure the time per iteration vs $K$ and vs $D$ (P1).

- **K sweep:** `synth_clean_p01` run with $K \in \{2, 4, 8, 16, 32, 64\}$ (same file, only $K$ changes).
- **D sweep:** one file per $D$, $K = 4$, parameters of `synth_clean`:

| File | $D$ | Size |
|---|---|---|
| `synth_sweep_d005_p01.bin` | 5 | 10 MB |
| `synth_sweep_d010_p01.bin` | 10 | 20 MB |
| `synth_sweep_d020_p01.bin` | 20 | 40 MB |
| `synth_sweep_d035_p01.bin` | 35 | 70 MB |
| `synth_sweep_d050_p01.bin` | 50 | 100 MB |

## 3. Real datasets

### 3.1 Sources and licenses

| Folder | Dataset | License | Source | DOI |
|---|---|---|---|---|
| `diabetes130` | Diabetes 130-US hospitals 1999-2008 | CC BY 4.0 | [UCI 296](https://archive.ics.uci.edu/dataset/296) | [10.24432/C5230J](https://doi.org/10.24432/C5230J) |
| `cdc_diabetes` | CDC Diabetes Health Indicators | not stated | [UCI 891](https://archive.ics.uci.edu/dataset/891) | [10.24432/C53919](https://doi.org/10.24432/C53919) |
| `cardiovascular` | Cardiovascular Disease ([Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)) | unknown | [OpenML 45547](https://www.openml.org/d/45547) | — |
| `parkinson` | Parkinsons Telemonitoring | CC BY 4.0 | [UCI 189](https://archive.ics.uci.edu/dataset/189) | [10.24432/C5ZS3N](https://doi.org/10.24432/C5ZS3N) |
| `eeg_eye_state` | EEG Eye State | CC BY 4.0 | [UCI 264](https://archive.ics.uci.edu/dataset/264) | [10.24432/C57G7J](https://doi.org/10.24432/C57G7J) |
| `gene_rnaseq` | Gene expression cancer RNA-Seq | CC BY 4.0 | [UCI 401](https://archive.ics.uci.edu/dataset/401) | [10.24432/C5R88H](https://doi.org/10.24432/C5R88H) |
| `heart_failure` | Heart Failure Clinical Records | CC BY 4.0 | [UCI 519](https://archive.ics.uci.edu/dataset/519) | [10.24432/C5Z89R](https://doi.org/10.24432/C5Z89R) |
| `cardiotocography` | Cardiotocography | CC BY 4.0 | [UCI 193](https://archive.ics.uci.edu/dataset/193) | [10.24432/C51S4N](https://doi.org/10.24432/C51S4N) |
| `statlog_heart` | Statlog (Heart) | CC BY 4.0 | [UCI 145](https://archive.ics.uci.edu/dataset/145) | [10.24432/C57303](https://doi.org/10.24432/C57303) |
| `brainweb` | BrainWeb simulated brain MRI (subject 44) | none stated (citation requested) | [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) | — |

BrainWeb citations: Cocosco et al. 1997; Kwan, Evans, Pike 1996 and 1999; Collins et al. 1998; Aubert-Broche et al. 2006.

### 3.2 Verification

Generated by `src/dataset-managed/verify-real-datasets.py` between the markers. 

Raw N x D = downloaded file, before the conversion (converted sizes in section 1); sources in 3.1.

<!-- BEGIN verify-real-datasets -->
| Dataset | Raw N x D | Numeric | Empty (numeric) | Duplicate rows | Verdict |
|---|---|---|---|---|---|
| `diabetes130` | 101766 x 47 | 11 | 0 | 0 | COMPLETE |
| `cdc_diabetes` | 253680 x 21 | 21 | 0 | 25772 | COMPLETE |
| `cardiovascular` | 70000 x 11 | 11 | 0 | 41 | COMPLETE |
| `parkinson` | 5875 x 19 | 19 | 0 | 0 | COMPLETE |
| `eeg_eye_state` | 14980 x 14 | 14 | 0 | 0 | COMPLETE |
| `gene_rnaseq` | 801 x 20531 | 20531 | 0 | 0 | COMPLETE |
| `heart_failure` | 299 x 12 | 12 | 0 | 0 | COMPLETE |
| `cardiotocography` | 2126 x 21 | 21 | 0 | 15 | COMPLETE |
| `statlog_heart` | 270 x 13 | 13 | 0 | 0 | COMPLETE |
| `brainweb` | 11862016 x 1 | 1 | 0 | n/a | COMPLETE |

Notes:

- **diabetes130**: empty cells only in non-numeric columns (`weight`, `max_glu_serum`, `A1Cresult`, ...)
- **cardiovascular**: few exact zeros (`ap_lo`, ...)
- **gene_rnaseq**: few exact zeros (`gene_1`, `gene_2`, `gene_12`, ...); 267 constant columns
- **cardiotocography**: few exact zeros (`UC`, `MLTV`, `Nmax`, ...)
- **brainweb**: T1 only (D = 1), range 0-4095; the ground-truth labels (0-11, 362 x 434 x 362, 0.5 mm) are on a finer grid than the T1 volume: `convert-brainweb-to-bin.py` subsamples them and places them on the T1 grid.
<!-- END verify-real-datasets -->
