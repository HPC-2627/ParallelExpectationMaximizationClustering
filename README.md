# ParallelExpectationMaximizationClustering

High Performance Computing (HPC) project (University of Trento): parallel Expectation-Maximization (EM) clustering.

**Authors:** 
- Michael Bernasconi
- Gabriele Fronzoni

## Repository structure

```
.
├── src/
│   ├── PEMC-baseline-sequential-diagonal.c    # Sequential version (correctness reference and serial time)
│   ├── PEMC-baseline-MPI-OpenMP-diagonal.c    # Hybrid MPI + OpenMP baseline
│   ├── PEMC-opt1-MPI-OpenMP-diagonal.c        # Optimization 1 of the baseline
│   ├── PEMC-opt2-MPI-OpenMP-diagonal.c        # Optimization 2 of the baseline
│   ├── PEMC-opt3-MPI-OpenMP-diagonal.c        # Optimization 3 of the baseline
│   ├── PEMC-baseline-MPI-OpenMP-full.c        # Baseline variant with full covariance (extension)
│   └── common/
│       └── my-time-lib.c                      # Shared code: statistics for the benchmarks (mean, median, standard deviation)
├── include/
│   └── my-time-lib.h                          # Timer macros and statistics prototypes (compile with -Iinclude and -lm)
├── data/
│   ├── real/
│   │   ├── download_datasets.py   # Downloads the real datasets into <name>/raw/
│   │   ├── verify_datasets.py     # Checks size and missing values, writes doc/VERIFICATION.md
│   │   └── <name>/raw/            # Raw files (not versioned, listed in .gitignore)
│   └── synthetic/   # Synthetically generated datasets
├── results/
│   ├── logs/        # Raw execution logs
│   ├── tables/      # Benchmark tables
│   ├── plots/       # Generated plots
│   ├── plots-performance.py  # Performance plots (time, speedup, efficiency, scaling, ...)
│   └── plots-cluster.py      # Clustering plots (PCA, centroids, cluster sizes, ...)
├── doc/
│   ├── latex/
│   │   ├── Group6-Report.tex         # IEEE-style report
│   │   └── theory/em-theory.tex      # EM theory notes (Gaussian Mixture Model, algorithm, sequential cost)
│   ├── em-explained.md   # Study guide on the EM algorithm
│   ├── papers.md         # Free papers for the state of the art (theory, single node, multiple nodes, 5 candidate optimizations)
│   ├── papers/           # Local copies of the papers, numbered like papers.md (not versioned, listed in .gitignore)
│   ├── project-guide.md  # Project requirements, report outline, evaluation criteria, data tips
│   ├── question.md       # Open questions for the instructors
│   ├── SOURCES.md        # Source, license and citation of each real dataset
│   ├── VERIFICATION.md   # Verification report of the real datasets (size, missing values, labels)
│   └── todo.md           # Task list and workflow
├── makefile         # Build rules
└── README.md
```

The `-diagonal` files use a Gaussian mixture with diagonal covariance (the main line of the project: sequential reference, baseline and 3 optimizations). The `-full` file is a variant with full covariance: it changes the model, so its results are compared only with runs of the same model.

## Documentation

- [doc/todo.md](doc/todo.md): task list and workflow
- [doc/question.md](doc/question.md): open questions for the instructors
- [doc/project-guide.md](doc/project-guide.md): project requirements, report outline, evaluation criteria, data tips
- [doc/em-explained.md](doc/em-explained.md): study guide on the EM algorithm (formulas, notation, data, parallelism)
- [doc/SOURCES.md](doc/SOURCES.md): source, license and citation of each real dataset
- [doc/VERIFICATION.md](doc/VERIFICATION.md): verification report of the real datasets (size, missing values, labels)
- [doc/papers.md](doc/papers.md): free papers for the state of the art (theory, parallel EM on one and several nodes, candidate optimizations); the PDFs are in `doc/papers/`, kept only locally
- [doc/latex/theory/em-theory.tex](doc/latex/theory/em-theory.tex): LaTeX notes on the EM theory
- [doc/latex/Group6-Report.tex](doc/latex/Group6-Report.tex): IEEE-style report
- [Presentation](https://docs.google.com/presentation/d/1nnzwo61ZBHoQlmlYfGVbIb39rVTOexQlnH01uqBfev0/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45): project presentation slides (Google Slides)

## Real datasets

The real datasets are public and are not versioned: `data/real/*/raw/` is listed in `.gitignore`. They are downloaded by `data/real/download_datasets.py` and checked by `data/real/verify_datasets.py`. Sources, licenses and citations are in [doc/SOURCES.md](doc/SOURCES.md).

### Datasets

| Folder | Dataset | N x D | Source |
|---|---|---|---|
| `diabetes130` | Diabetes 130-US hospitals | 101,766 x 11 numeric (of 47) | UCI 296 |
| `cdc_diabetes` | CDC Diabetes Health Indicators | 253,680 x 21 | UCI 891 |
| `cardiovascular` | Cardiovascular Disease | 70,000 x 11 | OpenML 45547 |
| `parkinson` | Parkinsons Telemonitoring | 5,875 x 19 | UCI 189 |
| `eeg_eye_state` | EEG Eye State | 14,980 x 14 | UCI 264 |
| `gene_rnaseq` | Gene expression cancer RNA-Seq (TCGA) | 801 x 20,531 | UCI 401 |
| `wdbc` | Breast Cancer Wisconsin (Diagnostic) | 569 x 30 | UCI 17 |
| `heart_failure` | Heart Failure Clinical Records | 299 x 12 | UCI 519 |
| `cardiotocography` | Cardiotocography | 2,126 x 21 | UCI 193 |
| `statlog_heart` | Statlog (Heart) | 270 x 13 | UCI 145 |
| `brainweb` | BrainWeb simulated brain MRI, subject 44 (T1 + ground-truth labels) | 11,862,016 voxels x 1 | BrainWeb |
| `cleveland` | Heart Disease (Cleveland) | 303 x 13 | UCI 45 |

`cleveland` is downloaded only to document why it is discarded: it has 6 incomplete rows. The other datasets have no missing values in the numeric columns (see [doc/VERIFICATION.md](doc/VERIFICATION.md)).

### Download

Requirements: Python 3 and internet access. Use a virtual environment, since the libraries are not installed by default:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn ucimlrepo brainweb-dl
```

(`brainweb-dl` also installs `nibabel`, used to read the MRI volumes.)

Download everything (about 250 MB; the largest file is `gene_rnaseq`, about 200 MB):

```bash
python3 data/real/download_datasets.py
```

Download only some datasets, by folder name:

```bash
python3 data/real/download_datasets.py wdbc heart_failure brainweb
```

Each dataset is saved in `data/real/<name>/raw/` as `features.csv` and, when a label exists, `targets.csv`, exactly as served by the source and without preprocessing. `brainweb` is saved as two `.nii.gz` volumes: the T1 image (the data) and the `crisp` labels (the ground truth). The script is idempotent: a dataset already downloaded is skipped.

Notes:
- The T1 and `crisp` volumes of BrainWeb are on different grids (1 mm and 0.5 mm): resample the labels to the T1 grid before comparing the clustering with them.
- The T2 and T2* images of BrainWeb are not used: the download from the BrainWeb server did not complete, so BrainWeb has one feature (D = 1).
- `epileptic seizure recognition` was planned but is no longer on UCI (removed at the curators' request): it was replaced by `gene_rnaseq`.

### Verification

```bash
python3 data/real/verify_datasets.py
```

It reports, for each dataset, the size, the number of numeric columns, the empty cells (NaN, empty strings, `?`, `NA`), the duplicate rows and the label distribution. The report is written to [doc/VERIFICATION.md](doc/VERIFICATION.md). A dataset is `COMPLETE` if its numeric columns have no empty cell, otherwise `DISCARD`.

## Build

TODO

## Usage

TODO

## Results

TODO
