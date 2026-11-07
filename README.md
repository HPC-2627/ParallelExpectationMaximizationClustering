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
│   ├── PEMC-baseline-sequential-full.c        # Sequential version with full covariance (Cholesky; reference for the full model)
│   ├── PEMC-baseline-MPI-OpenMP-diagonal.c    # Hybrid MPI + OpenMP baseline
│   ├── PEMC-opt1-MPI-OpenMP-diagonal.c        # Optimization 1 of the baseline
│   ├── PEMC-opt2-MPI-OpenMP-diagonal.c        # Optimization 2 of the baseline
│   ├── PEMC-opt3-MPI-OpenMP-diagonal.c        # Optimization 3 of the baseline
│   ├── PEMC-baseline-MPI-OpenMP-full.c        # Baseline variant with full covariance (extension, to be written)
│   ├── dataset-managed/                       # Scripts that create and check the datasets
│   │   ├── generate-synthetic-dataset.py      # Generator of the synthetic datasets (3 profiles, one file per number of processes) -> data/synthetic/
│   │   ├── generate-real-datasets.py          # Downloads the real datasets -> data/real/<name>/raw/
│   │   ├── verify-real-datasets.py            # Checks size and missing values, writes doc/real-datasets-info.md
│   │   ├── convert-real-to-bin.py             # Converts the real CSV datasets to the binary format of the synthetic ones -> data/real/<name>/<name>.bin
│   │   └── convert-brainweb-to-bin.py         # BrainWeb: .nii.gz -> .nii -> .bin (+ labels aligned to the T1 grid) -> data/real/brainweb/
│   ├── plots-generate/                        # Scripts that generate the plots -> results/plots/
│   │   ├── plots-performance.py               # Performance plots (time, speedup, efficiency, scaling, ...)
│   │   └── plots-cluster.py                   # Clustering plots (PCA, centroids, cluster sizes, ...)
│   ├── visual-sequential-diagonal/            # Step-by-step web visualization of the EM algorithm
│   │   ├── server.py                          #   local server: runs EM on a dataset, records every iteration, checks it against the C binary
│   │   └── index.html                         #   page that plays the iterations back (E-step, convergence test, M-step)
│   └── common/                                # Shared code (no MPI): compiled with every version
│       ├── dataset.c                          #   reading of the .bin datasets (whole file or a block of rows) and of the labels
│       ├── gmm.c                              #   parameters of the mixture, identical initialization, log-sum-exp, convergence test
│       ├── args.c                             #   command-line options, the same in all the versions
│       ├── logger.c                           #   .log text file of the runs (one block per run, with hostname and CPU model)
│       ├── eval.c                             #   adjusted Rand index and purity against the true labels
│       └── my-time-lib.c                      #   statistics for the benchmarks (mean, median, standard deviation)
├── include/                                   # Headers of src/common (compile with -Iinclude and -lm)
│   ├── common.h  dataset.h  gmm.h  args.h  logger.h  eval.h
│   └── my-time-lib.h                          #   timer macros and statistics prototypes
├── data/            # NOT versioned (listed in .gitignore): it is filled locally by the scripts in src/dataset-managed/
│   ├── real/        # Real (medical) datasets: <name>/raw/features.csv (+ targets.csv), converted to <name>/<name>.bin (+ labels, .json)
│   └── synthetic/   # Synthetic datasets, one folder per profile
│       ├── synth_clean/     # K=4, D=10: files synth_clean_p01.bin ... p16.bin, labels, .json
│       ├── synth_overlap/   # K=5, D=20: overlapping clusters, correlated variables
│       └── synth_rare/      # K=6, D=50: rare group (1%), 2% outliers
├── results/
│   ├── logs/        # Raw execution logs
│   ├── tables/      # Benchmark tables
│   └── plots/       # Generated plots
├── doc/
│   ├── latex/
│   │   └── Group6-Report.tex         # IEEE-style report
│   ├── em-explained.md   # Study guide on the EM algorithm
│   ├── papers.md         # Free papers for the state of the art (theory, single node, multiple nodes, 5 candidate optimizations)
│   ├── papers/           # Local copies of the papers, numbered like papers.md (not versioned, listed in .gitignore)
│   ├── project-guide.md  # Project requirements, report outline, evaluation criteria, data tips
│   ├── question.md       # Open questions for the instructors
│   ├── synthetic-datasets-info.md  # Characteristics of the synthetic datasets (profiles, variables, sizes, file format), ready for the report
│   ├── real-datasets-source.md     # Source, license and citation of each real dataset
│   ├── real-datasets-info.md       # Verification report of the real datasets (size, missing values, labels), generated by verify-real-datasets.py
│   └── todo.md           # Task list and workflow
├── makefile         # Build rules (make builds bin/, which is not versioned)
└── README.md
```

The `-diagonal` files use a Gaussian mixture with diagonal covariance (the main line of the project: sequential reference, baseline and 3 optimizations). The `-full` files are a variant with full covariance (each cluster has a complete $D \times D$ matrix, so correlated variables are modelled): it changes the model, so its results are compared only with runs of the same model (`-full` with `-full`, `-diagonal` with `-diagonal`). A full iteration costs $O(NKD^2)$ instead of $O(NKD)$.

## Data (required step before running the experiments)

The whole `data/` folder is **not versioned** (it is listed in `.gitignore`), because the datasets are large. After cloning the repository the folder is empty, so **it is necessary to run the generator of the synthetic datasets** before any experiment:

```
python3 src/dataset-managed/generate-synthetic-dataset.py
```

It needs Python 3 with NumPy and writes about 5 GB in `data/synthetic/`, one folder per profile (`synth_clean/`, `synth_overlap/`, `synth_rare/`), with one file for each number of processes of the weak scaling. The generation is deterministic (default seed 42), so everyone obtains identical files. Options (for example a smaller size) are listed with `python3 src/dataset-managed/generate-synthetic-dataset.py --help`.

The real datasets also have to be downloaded locally into `data/real/` (see below).

### Synthetic datasets

Three profiles, generated from a known Gaussian mixture (true labels available), so that all the versions can be checked against each other and the clustering quality can be measured. Real medical datasets are too small to show any scaling, so these are used for the weak and strong scaling.

| | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| Role | easy, satisfies the model | realistic | hard |
| Clusters $K$ | 4 | 5 | 6 |
| Variables $D$ | 10 | 20 | 50 |
| Cluster weights | 25% each | 30, 25, 20, 15, 10% | 40, 25, 18, 12, 4, 1% |
| Separation $\delta$ (in standard deviations) | 6.0 | 2.5 | 4.0 |
| Covariance inside a cluster | diagonal | correlated (factor model) | diagonal |
| Outliers | 0% | 0% | 2% (label -1) |
| Skewed variables / clipping | no / no | yes / yes | no / yes |
| What it tests | correctness, reference for scaling | slow convergence, diagonal model mismatch | numerical stability: rare group, outliers, high $D$ |

- **Variables:** the first 20 are clinical (age, bmi, sys_bp, glucose, hba1c, cholesterol, ...), the others (up to 50) are generic biomarkers. The values are generic adult values chosen by us, not from a real cohort. Every column is standardized (z-score).
- **Weak scaling:** one file per number of processes $p \in \{1,2,4,8,16\}$, with $N = 250\,000 \cdot p$ rows (so each rank always has 250 000 rows). The files of a profile are nested (the file for $2p$ contains the one for $p$). Strong scaling uses a fixed file (e.g. `_p16`) with more processes.
- **Size on disk:** `synth_clean` 620 MB, `synth_overlap` 1 240 MB, `synth_rare` 3 100 MB (about 5 GB in total, labels included). The largest file is 4 000 000 x 50 doubles (1.6 GB).
- **File format:** binary, an 8-byte header with two `int` (N, D) followed by $N \cdot D$ `double` in row-major order; the labels are in a separate file (`int`, one per row). Reading in C: `fread` of the header, then of the matrix `X[i*D + d]`.
- **Limits:** `synth_overlap` is correlated while the baseline is diagonal on purpose; nested files are not independent samples.

Full details (variables with units and limits, generation procedure, work per rank, C reading code) are in [doc/synthetic-datasets-info.md](doc/synthetic-datasets-info.md).

## Real datasets

The real datasets are public and are not versioned: `data/real/*/raw/` is listed in `.gitignore`. They are downloaded by `src/dataset-managed/generate-real-datasets.py` into `data/real/` and checked by `src/dataset-managed/verify-real-datasets.py`. Sources, licenses and citations are in [doc/real-datasets-source.md](doc/real-datasets-source.md).

### Datasets

| Folder | Dataset | N x D | Source |
|---|---|---|---|
| `diabetes130` | Diabetes 130-US hospitals | 101,766 x 11 numeric (of 47) | [UCI 296](https://archive.ics.uci.edu/dataset/296) |
| `cdc_diabetes` | CDC Diabetes Health Indicators | 253,680 x 21 | [UCI 891](https://archive.ics.uci.edu/dataset/891) |
| `cardiovascular` | Cardiovascular Disease | 70,000 x 11 | [OpenML 45547](https://www.openml.org/d/45547) |
| `parkinson` | Parkinsons Telemonitoring | 5,875 x 19 | [UCI 189](https://archive.ics.uci.edu/dataset/189) |
| `eeg_eye_state` | EEG Eye State | 14,980 x 14 | [UCI 264](https://archive.ics.uci.edu/dataset/264) |
| `gene_rnaseq` | Gene expression cancer RNA-Seq (TCGA) | 801 x 20,531 | [UCI 401](https://archive.ics.uci.edu/dataset/401) |
| `heart_failure` | Heart Failure Clinical Records | 299 x 12 | [UCI 519](https://archive.ics.uci.edu/dataset/519) |
| `cardiotocography` | Cardiotocography | 2,126 x 21 | [UCI 193](https://archive.ics.uci.edu/dataset/193) |
| `statlog_heart` | Statlog (Heart) | 270 x 13 | [UCI 145](https://archive.ics.uci.edu/dataset/145) |
| `brainweb` | BrainWeb simulated brain MRI, subject 44 (T1 + ground-truth labels) | 11,862,016 voxels x 1 | [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) |

`cleveland` is downloaded only to document why it is discarded: it has 6 incomplete rows. The other datasets have no missing values in the numeric columns (see [doc/real-datasets-info.md](doc/real-datasets-info.md)).

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
python3 src/dataset-managed/generate-real-datasets.py
```

Download only some datasets, by folder name:

```bash
python3 src/dataset-managed/generate-real-datasets.py heart_failure brainweb
```

Each dataset is saved in `data/real/<name>/raw/` as `features.csv` and, when a label exists, `targets.csv`, exactly as served by the source and without preprocessing. `brainweb` is saved as two `.nii.gz` volumes: the T1 image (the data) and the `crisp` labels (the ground truth). The script is idempotent: a dataset already downloaded is skipped.

Notes:
- The T1 and `crisp` volumes of BrainWeb are on different grids (1 mm and 0.5 mm): `convert-brainweb-to-bin.py` resamples the labels to the T1 grid (see Conversion to binary).
- The T2 and T2* images of BrainWeb are not used: the download from the BrainWeb server did not complete, so BrainWeb has one feature (D = 1).
- `epileptic seizure recognition` was planned but is no longer on UCI (removed at the curators' request): it was replaced by `gene_rnaseq`.

### Verification

```bash
python3 src/dataset-managed/verify-real-datasets.py
```

It reports, for each dataset, the size, the number of numeric columns, the empty cells (NaN, empty strings, `?`, `NA`), the duplicate rows and the label distribution. The report is written to [doc/real-datasets-info.md](doc/real-datasets-info.md). A dataset is `COMPLETE` if its numeric columns have no empty cell, otherwise `DISCARD`.

### Conversion to binary

The raw CSV files are kept as served by the source. To use the real datasets with the same C loader of the synthetic ones, convert them:

```bash
python3 src/dataset-managed/convert-real-to-bin.py            # all the datasets
python3 src/dataset-managed/convert-real-to-bin.py wdbc       # only some
```

For each dataset it writes in `data/real/<name>/`:
- `<name>.bin`: same format as the synthetic data (8-byte header with two `int` N and D, then $N \cdot D$ `double` row-major);
- `<name>_labels.bin`: one `int` per row (classes encoded as 0..C-1), only if the label is categorical (`parkinson` has continuous targets, so no labels);
- `<name>.json`: columns used and dropped, mean and standard deviation of the standardization, class mapping.

Preprocessing: only the numeric columns are kept (identifiers removed), constant columns are dropped, every column is standardized (z-score). The datasets marked `DISCARD` (`cleveland`) are skipped: nothing is imputed. `brainweb` is not CSV and has its own script (below).

#### BrainWeb (NIfTI volumes)

`brainweb` is converted by a dedicated script, because its raw files are `.nii.gz` volumes, not CSV. It needs `nibabel` (already installed with `brainweb-dl`, otherwise `pip install numpy nibabel`) and the download of the previous step (`generate-real-datasets.py brainweb`):

```bash
python3 src/dataset-managed/convert-brainweb-to-bin.py            # --force to redo every step
```

Three steps, each visible on disk in `data/real/brainweb/`:

```
raw/*.nii.gz  --gunzip-->  nifti/*.nii  --nibabel + numpy-->  brainweb.bin, brainweb_labels.bin, brainweb.json
```

- **Data:** the T1 volume (181 x 256 x 256) becomes a matrix with $N$ = 11,862,016 voxels (C order) and $D = 1$, standardized (z-score). Same `.bin` format as every other dataset (about 95 MB).
- **Labels:** the `crisp` labels (0.5 mm, tissue classes 0-11) are subsampled by 2 and placed on the T1 grid at the offset found by cross-correlating the two brain masks; voxels outside get label 0. The NIfTI affines of these files are inconsistent, so they are not used. The script prints $\eta^2$, the share of T1 variance explained by the labels, as a check (0.95 with the estimated offset, 0.18 with the affines). The labels are in `brainweb_labels.bin` (one `int` per voxel), used only to measure ARI/Dice.

## Build

```
make              # builds bin/PEMC-baseline-sequential-diagonal and bin/PEMC-baseline-sequential-full (gcc, -O3)
make clean
```

The shared code in `src/common/` does not use MPI: the MPI + OpenMP versions will compile it with the same rule (`mpicc -fopenmp`). The `bin/` folder is not versioned.

## Usage

Sequential version (correctness reference and serial time). It needs the dataset (`-d`) and the number of clusters (`-k`):

```
./bin/PEMC-baseline-sequential-diagonal -d data/synthetic/synth_clean/synth_clean_p01.bin -k 4
./bin/PEMC-baseline-sequential-diagonal -d data/real/heart_failure/heart_failure.bin -k 3
```

Sequential version with **full covariance**: same options and same log format, only the executable changes (the log goes to `results/logs/baseline-sequential-full.log`):

```
./bin/PEMC-baseline-sequential-full -d data/synthetic/synth_overlap/synth_overlap_p01.bin -k 5
./bin/PEMC-baseline-sequential-full -d data/real/heart_failure/heart_failure.bin -k 2
```

It is much slower than the diagonal one when $D$ is large (cost $O(NKD^2)$ per iteration; `gene_rnaseq`, $D = 20\,531$, is not feasible: $K D^2$ doubles would not fit in memory), so for the big files start with `-n ROWS`. Its log-likelihood is comparable only with other `-full` runs, and on correlated data (`synth_overlap`) it is clearly higher than the diagonal one.

**Run on all the synthetic datasets.** $K$ is the number of true clusters of each profile (`synth_clean` 4, `synth_overlap` 5, `synth_rare` 6); `p01 … p16` are the five sizes of the scaling tests:

```
for p in 01 02 04 08 16; do
  ./bin/PEMC-baseline-sequential-diagonal -d data/synthetic/synth_clean/synth_clean_p$p.bin -k 4
  ./bin/PEMC-baseline-sequential-diagonal -d data/synthetic/synth_overlap/synth_overlap_p$p.bin -k 5
  ./bin/PEMC-baseline-sequential-diagonal -d data/synthetic/synth_rare/synth_rare_p$p.bin -k 6
done
```

**Run on all the real datasets.** Each entry is `name:K`, with $K$ equal to the number of classes in the labels file:

```
for x in brainweb:12 cardiotocography:10 cardiovascular:2 cdc_diabetes:2 \
         diabetes130:3 eeg_eye_state:2 gene_rnaseq:5 heart_failure:2 \
         parkinson:2 statlog_heart:2; do
  ./bin/PEMC-baseline-sequential-diagonal -d data/real/${x%%:*}/${x%%:*}.bin -k ${x##*:}
done
```

`parkinson` has no labels (continuous targets), so $K = 2$ is an arbitrary choice there. The biggest files (`synth_rare_p16.bin` 1.6 GB, `synth_overlap_p16.bin` 640 MB, `brainweb.bin` 95 MB) take the longest and need enough RAM; add `-n ROWS` to use only the first rows. The same two loops work for the other versions (including `bin/PEMC-baseline-sequential-full`, except `gene_rnaseq`): change only the executable name.

Run it from the root of the repository: the only file written is the log `results/logs/baseline-sequential-diagonal.log` (a block is appended at every run; the folder is created if missing).

| Option | Meaning |
|---|---|
| `-d, --dataset FILE` | dataset in binary format (header N, D + float64 row-major) |
| `-k, --clusters K` | number of clusters |
| `-n, --rows N` | use only the first N rows of the file (default: all) |
| `-i, --max-iter N` | maximum number of iterations (default 200) |
| `-t, --tol X` | relative tolerance on the log-likelihood (default 1e-6) |
| `-s, --seed S` | seed of the initialization (default 42): the same seed gives the same initialization in every version |
| `-r, --restarts R` | number of initializations, the best one is kept (default 1); the times are summed over all the restarts |
| `-l, --log FILE.log` | log file (default `results/logs/<name>.log`) |
| `--no-log` | do not write the log |
| `-v, --verbose` | print the log-likelihood at every iteration |

**Log format** (`.log`, plain text, the same for all the versions). Every run appends a block of `key : value` lines, separated from the next one by an empty line:

```
==== run ====
timestamp      : 2026-11-05T09:53:41Z
version        : baseline-sequential-diagonal
dataset        : data/synthetic/synth_clean/synth_clean_p01.bin
hostname       : HP-Victus
cpu_model      : 12th Gen Intel(R) Core(TM) i5-12450H
ranks          : 1
threads        : 1
N              : 250000
D              : 10
K              : 4
restarts       : 1
iterations     : 12
converged      : 1
loglik         : -3087811.89369
ari            : 0.914663
t_total        : 0.990871
t_read         : 0.031663
t_init         : 2e-06
t_estep        : 0.793119
t_mstep        : 0.162592
t_comm         : 0
t_iteration    : 0.079643
```

Times are in seconds. `hostname` and `cpu_model` identify the node (the nodes of the cluster are not all equal). `ari` (adjusted Rand index against the true labels) is `n/a` if the dataset has no labels. `t_iteration` is `(t_estep + t_mstep + t_comm) / iterations` (only with one restart). To read it in Python, split the file on `==== run ====` and every line on the first `:`.

**Correctness check between versions:** run them with the same dataset, `-k` and `-s`: they must end with the same log-likelihood (up to rounding) and the same number of iterations. Compare only versions of the same model: `-diagonal` with `-diagonal`, `-full` with `-full`.

## Visualization of the algorithm

A small web page shows how EM works step by step on any dataset of `data/`: the initial centroids, then for every iteration the E-step (points coloured by their responsibilities), the convergence test on the log-likelihood and the M-step (means and ellipses moving to their new values). It also shows the formulas with the real numbers of the current iteration, the log-likelihood curve, and the calculation of any point you click.

```
make                              # builds bin/PEMC-baseline-sequential-diagonal
python3 src/visual-sequential-diagonal/server.py   # then open http://127.0.0.1:8000
```

Choose the dataset, `K`, the number of rows (at most 3000), the seed, `max_iter` and `tol`, then press **Run**. Controls: play/pause, back/next phase (also with the arrow keys and space), speed slider, clickable timeline, and a checkbox that draws the true labels as rings.

How it relates to the C program:
- The initial rows are chosen with the **same libc `srand()`/`rand()`** as `gmm_init_indices()` (called through `ctypes`), so the starting point is the same.
- The server repeats the same EM as `PEMC-baseline-sequential-diagonal.c` (diagonal covariance, same floors and convergence test) in NumPy, recording every iteration.
- It also runs the **real binary** with `-v` on the same data and the page compares the log-likelihood of every iteration (expandable section under the chart). A mismatch would mean that the page and the C code have diverged.
- The data have D dimensions: the plot is the projection on the first two principal components (the percentage of variance kept is shown), while EM uses all dimensions.

Requirements: Python 3 with NumPy; the browser only needs to reach `127.0.0.1`.

## Results

TODO

## Documentation

- [doc/todo.md](doc/todo.md): task list and workflow
- [doc/question.md](doc/question.md): open questions for the instructors
- [doc/project-guide.md](doc/project-guide.md): project requirements, report outline, evaluation criteria, data tips
- [doc/em-explained.md](doc/em-explained.md): study guide on the EM algorithm (formulas, notation, data, parallelism)
- [doc/synthetic-datasets-info.md](doc/synthetic-datasets-info.md): characteristics of the three synthetic datasets (profiles, variables, weak scaling files and sizes, file format), ready to be reused in the report
- [doc/real-datasets-source.md](doc/real-datasets-source.md): source, license and citation of each real dataset
- [doc/real-datasets-info.md](doc/real-datasets-info.md): verification report of the real datasets (size, missing values, labels)
- [doc/papers.md](doc/papers.md): free papers for the state of the art (theory, parallel EM on one and several nodes, candidate optimizations); the PDFs are in `doc/papers/`, kept only locally
- [doc/latex/Group6-Report.tex](doc/latex/Group6-Report.tex): IEEE-style report
- [Presentation](https://docs.google.com/presentation/d/1nnzwo61ZBHoQlmlYfGVbIb39rVTOexQlnH01uqBfev0/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45): project presentation slides (Google Slides)
