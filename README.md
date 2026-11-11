# ParallelExpectationMaximizationClustering

### How to run

[![Run on the cluster](https://img.shields.io/badge/Run_on_the-Cluster-blue?style=for-the-badge&logo=gnubash&logoColor=white)](doc/exec-cluster.md)
[![Run locally](https://img.shields.io/badge/Run-Locally-green?style=for-the-badge&logo=linux&logoColor=white)](doc/exec-local.md)

---

### Technologies

![C](https://img.shields.io/badge/C-A8B9CC?style=flat&logo=c&logoColor=black)
![MPI](https://img.shields.io/badge/MPI-OpenMPI-0066B3?style=flat)
![OpenMP](https://img.shields.io/badge/OpenMP-hybrid-orange?style=flat)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![Bash](https://img.shields.io/badge/Bash-4EAA25?style=flat&logo=gnubash&logoColor=white)
![Make](https://img.shields.io/badge/Make-6D00CC?style=flat&logo=gnu&logoColor=white)
![PBS](https://img.shields.io/badge/PBS-job_scheduler-lightgrey?style=flat)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat&logo=latex&logoColor=white)

---

### About

High Performance Computing (HPC) project (University of Trento): parallel Expectation-Maximization (EM) clustering.

**Authors:** 
- Michael Bernasconi
- Gabriele Fronzoni

## AI statement

In this project we used an AI assistant (Claude, by Anthropic) for three purposes: (i) support in writing and debugging the code; (ii) searching for sources and papers for the state of the art and for the datasets; (iii) mathematical and grammatical correction of the report. The design choices, the experiments and the analysis of the results are our own, and we reviewed the output of the assistant.

## Objective

The goal of the project is to **parallelize the Expectation-Maximization (EM) algorithm for clustering with Gaussian mixture models (GMM)** with a hybrid **MPI + OpenMP** implementation in C, and to measure how it scales on the cluster of the University of Trento (speedup, efficiency, strong and weak scaling, single and multiple nodes).

EM alternates two steps until the log-likelihood converges: the **E-step** computes for every point the responsibility of each of the $K$ clusters, the **M-step** recomputes weights, means and covariances from those responsibilities. The cost is dominated by the E-step, which is independent for every point, so the rows of the dataset can be split among MPI ranks and OpenMP threads; only the sufficient statistics have to be combined at every iteration. The explanation of the algorithm (formulas, notation, parallelism) is in [doc/em-explained.md](doc/em-explained.md).

### Models

Two Gaussian mixture models, with the same importance:
- **diagonal** covariance: one variance per variable. An iteration costs $O(NKD)$.
- **full** covariance: each cluster has a complete $D \times D$ matrix, so correlated variables are modelled. An iteration costs $O(NKD^2)$.

The model is different, so the results are compared only with runs of the same model (`-full` with `-full`, `-diagonal` with `-diagonal`), and every optimization is evaluated against the baseline of its own model.

### Versions

Each model has four versions, for a total of eight programs:

| Version | What it is | State |
|---|---|---|
| `sequential` | single process, no parallelism: correctness reference and serial time for the speedup | done |
| `baseline-MPI-OpenMP` | first hybrid parallelization (rows split among MPI ranks, OpenMP threads inside each rank): the reference of the parallel tests | done |
| `opt1-MPI-OpenMP` | first optimization of the baseline | **empty, not written yet** |
| `opt2-MPI-OpenMP` | second optimization of the baseline | **empty, not written yet** |

All the versions of the same model must end with the same log-likelihood and the same number of iterations (same dataset, `-k` and `-s`).

## Repository structure

```
.
├── src/
│   ├── diagonal/                              # Diagonal covariance: sequential, baseline, 2 optimizations
│   │   ├── PEMC-baseline-sequential-diagonal.c    #   Sequential version (correctness reference and serial time)
│   │   ├── PEMC-baseline-MPI-OpenMP-diagonal.c    #   Hybrid MPI + OpenMP baseline
│   │   ├── PEMC-opt1-MPI-OpenMP-diagonal.c        #   Optimization 1 of the baseline (to be written)
│   │   └── PEMC-opt2-MPI-OpenMP-diagonal.c        #   Optimization 2 of the baseline (to be written)
│   ├── full/                                  # Full covariance: sequential, baseline, 2 optimizations
│   │   ├── PEMC-baseline-sequential-full.c        #   Sequential version (Cholesky; reference for the full model)
│   │   ├── PEMC-baseline-MPI-OpenMP-full.c        #   Hybrid MPI + OpenMP baseline with full covariance
│   │   ├── PEMC-opt1-MPI-OpenMP-full.c            #   Optimization 1 with full covariance (to be written)
│   │   └── PEMC-opt2-MPI-OpenMP-full.c            #   Optimization 2 with full covariance (to be written)
│   ├── script/                                # Launchers for the cluster (PBS)
│   │   ├── pemc-submit.sh                         #   submits the 4 baselines (sequential and MPI + OpenMP, diagonal and full) in all the layouts and placements; `sweep` submits the K and D sweeps (P1)
│   │   ├── pemc-job.sh                            #   PBS job body: one version, one layout, ALL the datasets (real and synthetic); SWEEP=K|D for plot P1
│   │   ├── pemc-cluster.sh                        #   runs the sequential baseline for the clustering plots (K range, saves labels and means at the true K)
│   │   └── mpi_wrapper.sh                         #   forces UCX on the Ethernet interface of the cluster (used by mpiexec in multi-node runs)
│   ├── dataset-managed/                       # Scripts that create and check the datasets
│   │   ├── generate-synthetic-dataset.py      # Generator of the synthetic datasets (3 profiles, one file per number of processes; --sweep adds the D sweep for P1) -> data/synthetic/
│   │   ├── generate-real-datasets.py          # Downloads the real datasets -> data/real/<name>/raw/
│   │   ├── verify-real-datasets.py            # Checks size and missing values, writes doc/real-datasets-info.md
│   │   ├── convert-real-to-bin.py             # Converts the real CSV datasets to the binary format of the synthetic ones -> data/real/<name>/<name>.bin
│   │   └── convert-brainweb-to-bin.py         # BrainWeb: .nii.gz -> .nii -> .bin (+ labels aligned to the T1 grid) -> data/real/brainweb/
│   ├── plots-generate/                        # Scripts that generate the plots -> results/plots/
│   │   ├── plots-performance.py               # Performance plots P1-P11 (speedup, efficiency, scaling, placement, roofline, load imbalance, ...)
│   │   └── plots-cluster.py                   # Clustering plots C1-C6 (convergence, BIC/AIC vs K, PCA, centroids, cluster sizes) + summary table
│   ├── visual-sequential/                     # Step-by-step web visualization of the EM algorithm (diagonal or full covariance)
│   │   ├── server.py                          #   local server: runs EM on a dataset, records every iteration, checks it against the C binary
│   │   └── index.html                         #   page that plays the iterations back (E-step, convergence test, M-step)
│   └── common/                                # Shared code (no MPI): compiled with every version
│       ├── dataset.c                          #   reading of the .bin datasets (whole file or a block of rows) and of the labels
│       ├── gmm.c                              #   parameters of the mixture, initialization, log-sum-exp, convergence test, assignment of the rows; Cholesky for the full covariance (sections: shared / diagonal / full)
│       ├── args.c                             #   command-line options, the same in all the versions
│       ├── logger.c                           #   .log text file of the runs (one block per run, with hostname and CPU model)
│       ├── perfmodel.c                        #   analytic FLOP and memory-traffic count of a run (keys flops, bytes of the log; roofline P10)
│       ├── cluster-out.c                      #   saves the clustering result (labels, means, log-likelihood curve) with --save, for the clustering plots
│       ├── eval.c                             #   adjusted Rand index and purity against the true labels
│       └── my-time-lib.c                      #   statistics for the benchmarks (mean, median, standard deviation)
├── include/                                   # Headers of src/common (compile with -Iinclude and -lm)
│   ├── common.h  dataset.h  gmm.h  args.h  logger.h  eval.h  perfmodel.h  cluster-out.h
│   ├── mpi-common.h                           #   helpers shared by the MPI versions only (header-only): ABORT, block_of_rank, max_over_ranks, gather_over_ranks
│   └── my-time-lib.h                          #   timer macros and statistics prototypes
├── data/            # NOT versioned (listed in .gitignore): it is filled locally by the scripts in src/dataset-managed/
│   ├── real/        # Real (medical) datasets: <name>/raw/features.csv (+ targets.csv), converted to <name>/<name>.bin (+ labels, .json)
│   └── synthetic/   # Synthetic datasets, one folder per profile
│       ├── synth_clean/     # K=4, D=10: files synth_clean_p01.bin ... p16.bin, labels, .json
│       ├── synth_overlap/   # K=5, D=20: overlapping clusters, correlated variables
│       └── synth_rare/      # K=6, D=50: rare group (1%), 2% outliers
├── results/
│   ├── logs/        # Raw execution logs of the scaling runs (one file per job); sweep/ = K and D sweeps (P1); cluster/ = runs of the clustering plots; pbs/ = job output
│   ├── clusters/    # Clustering results saved with --save (labels, means, log-likelihood curve), used by plots-cluster.py
│   ├── tables/      # Tables (clustering-summary.csv)
│   └── plots/       # Generated plots (PDF)
├── doc/
│   ├── latex/
│   │   └── Group6-Report.tex         # IEEE-style report (AI statement, model, dataset and cost sections written; design and results to do)
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

## Data

The whole `data/` folder is **not versioned** (it is listed in `.gitignore`), because the datasets are large: they are generated (synthetic) or downloaded (real) on the machine where the experiments run. The procedure is in [doc/exec-cluster.md](doc/exec-cluster.md) (cluster) and [doc/exec-local.md](doc/exec-local.md) (local). The generation of the synthetic datasets is deterministic (default seed 42), so everyone obtains identical files.

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
- **Limits:** `synth_overlap` is correlated: it is the profile where the diagonal model is mismatched and the full one is not; nested files are not independent samples.

Full details (variables with units and limits, generation procedure, work per rank, C reading code) are in [doc/synthetic-datasets-info.md](doc/synthetic-datasets-info.md).

## Real datasets

The real datasets are public and are not versioned: `data/real/*/raw/` is listed in `.gitignore`. They are downloaded on the cluster or locally (see [doc/exec-cluster.md](doc/exec-cluster.md) and [doc/exec-local.md](doc/exec-local.md)). Sources, licenses and citations are in [doc/real-datasets-source.md](doc/real-datasets-source.md).

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

None of the datasets has missing values in the numeric columns (see [doc/real-datasets-info.md](doc/real-datasets-info.md)).

**Preprocessing and format of the real datasets.** The raw files are kept as served by the source (`data/real/<name>/raw/`, not versioned) and converted to the same binary format of the synthetic ones (`<name>.bin`, plus `<name>_labels.bin` when the label is categorical and `<name>.json` with the columns and the standardization). Only the numeric columns are kept (identifiers removed), constant columns are dropped, every column is standardized (z-score); a dataset with empty cells would be discarded, nothing is imputed. `brainweb` is a NIfTI volume: the T1 image (181 x 256 x 256) becomes $N$ = 11,862,016 voxels with $D = 1$ (the T2 and T2* images are not used), and the `crisp` labels are resampled to the T1 grid. The labels are used only to measure ARI/Dice.

## Running the experiments

- **On the cluster:** [doc/exec-cluster.md](doc/exec-cluster.md) (VPN, clone, datasets, modules, build, PBS jobs, plots).
- **Locally:** [doc/exec-local.md](doc/exec-local.md) (datasets, build, usage and options, log format, plots). The visualization of the algorithm is described [below](#visualization-of-the-algorithm).

## Visualization of the algorithm

A small web page shows how EM works step by step on any dataset of `data/`: the initial centroids, then for every iteration the E-step (points coloured by their responsibilities), the convergence test on the log-likelihood and the M-step (means and ellipses moving to their new values). It also shows the formulas with the real numbers of the current iteration, the log-likelihood curve, and the calculation of any point you click.

It runs locally. The page is [src/visual-sequential/index.html](src/visual-sequential/index.html), served by a small Python server ([src/visual-sequential/server.py](src/visual-sequential/server.py)) that runs the EM and the C binary:

```bash
make                                       # builds the two sequential C binaries (diagonal and full)
python3 src/visual-sequential/server.py    # then open http://127.0.0.1:8000
```

Requirements: Python 3 with NumPy, a C compiler (`gcc`) and at least one dataset in `data/`; the browser only needs to reach `127.0.0.1`.

Choose the dataset, the covariance (**diagonal** or **full**), `K`, the number of rows (at most 3000), the seed, `max_iter` and `tol`, then press **Run**. Controls: play/pause, back/next phase (also with the arrow keys and space), speed slider, clickable timeline, and a checkbox that draws the true labels as rings.

How it relates to the C program:
- The initial rows are chosen with the **same libc `srand()`/`rand()`** as `gmm_init_indices()` (called through `ctypes`), so the starting point is the same.
- The server repeats the same EM as `PEMC-baseline-sequential-diagonal.c` or `PEMC-baseline-sequential-full.c` (depending on the covariance chosen; same floors and convergence test) in NumPy, recording every iteration. With full covariance the ellipses are tilted, because they follow the correlations between variables.
- It also runs the **real binary** of the chosen model with `-v` on the same data and the page compares the log-likelihood of every iteration (expandable section under the chart). A mismatch would mean that the page and the C code have diverged.
- The data have D dimensions: the plot is the projection on the first two principal components (the percentage of variance kept is shown), while EM uses all dimensions.


## Plots

Everything is regenerable from the logs: the report uses only the static plots produced by the two scripts in `src/plots-generate/` (rules, catalog and experiment matrix: [doc/plots.md](doc/plots.md); colors: [doc/color.md](doc/color.md)). The performance plots P1-P11 (sweeps, speedup, strong and weak scaling, placement, ranks vs threads, time breakdown, optimization gain, roofline, load imbalance) come from the cluster runs; the clustering plots C1-C6 (convergence, choice of K with BIC/AIC, PCA, centroids, cluster sizes, full vs diagonal) come from the sequential runs on a range of K. How to produce them: [doc/exec-cluster.md](doc/exec-cluster.md).

## Results

TODO

## Documentation

- [doc/exec-local.md](doc/exec-local.md): run locally (datasets, build, usage and options, log format, plots)
- [doc/exec-cluster.md](doc/exec-cluster.md): step-by-step procedure to run everything on the cluster (VPN, clone, datasets, modules, build, PBS jobs, plots)
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
