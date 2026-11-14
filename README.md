# ParallelExpectationMaximizationClustering

### How to run

[![Run on the cluster](https://img.shields.io/badge/Run_on_the-Cluster-blue?style=for-the-badge&logo=gnubash&logoColor=white)](doc/exec-cluster.md)
[![Visualize the algorithm](https://img.shields.io/badge/Visualize-the_Algorithm-purple?style=for-the-badge&logo=python&logoColor=white)](doc/exec-visual.md)

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
| `opt1-MPI-OpenMP` | first optimization of the baseline | diagonal written (same log-likelihood and iterations as the baseline on one rank; to be checked with several ranks on the cluster): E-step and M-step fused in one pass (no responsibility matrix) inside one parallel region per iteration; full not written yet |
| `opt2-MPI-OpenMP` | second optimization of the baseline | **empty, not written yet** |

All the versions of the same model must end with the same log-likelihood and the same number of iterations (same dataset, `-k` and `-s`).

## Repository structure

```
.
├── src/
│   ├── diagonal/          # Diagonal covariance: PEMC-<version>-diagonal.c (sequential, baseline, opt1, opt2)
│   ├── full/              # Full covariance (Cholesky): PEMC-<version>-full.c (sequential, baseline, opt1, opt2)
│   ├── common/            # Shared code (no MPI): dataset I/O, GMM core, CLI options, logging, perf model, evaluation, timing
│   ├── script/            # PBS launchers for the cluster + UCX wrapper for multi-node mpiexec
│   ├── dataset-managed/   # Python scripts that generate, download, verify and convert the datasets -> data/
│   ├── plots-generate/    # Python scripts that generate the performance and clustering plots -> results/plots/
│   └── visual-sequential/ # Local web visualization of the EM algorithm, step by step (see below)
├── include/               # Headers of the shared code, plus MPI-only helpers and timer macros (compile with -Iinclude -lm)
├── data/                  # NOT versioned: filled locally by src/dataset-managed/
│   ├── real/              #   real datasets, one folder each (raw files + converted .bin)
│   └── synthetic/         #   synthetic datasets, one folder per profile (see Data)
├── results/
│   ├── logs/              # Raw execution logs: <model>/<stage>/ for the scaling runs, plus sweep/, cluster/ and pbs/ (PBS output)
│   ├── clusters/          # Clustering results saved with --save, used by the clustering plots
│   ├── tables/            # Summary tables
│   └── plots/             # Generated plots (PDF)
├── doc/                   # Documentation: how to run, EM study notes, dataset docs, papers, task list, LaTeX report (see Documentation below)
├── makefile               # Build rules (make builds bin/, which is not versioned)
└── README.md
```

## Data

All the datasets are described in **[doc/datasets.md](doc/datasets.md)** (profiles, variables, sizes, file format, sources, licenses, preprocessing, verification). The whole `data/` folder is **not versioned** (listed in `.gitignore`): the datasets are generated (synthetic) or downloaded (real) on the machine where the experiments run, following [doc/exec-cluster.md](doc/exec-cluster.md). Every dataset is converted to the same binary format (header `int32` N, D + `float64` matrix in row-major order, labels in a separate file) and every column is standardized (z-score).

### Synthetic datasets

Three profiles generated from a known Gaussian mixture (true labels available, deterministic with seed 42). They are used for correctness checks and for the weak and strong scaling, because the real datasets are too small to show any scaling. Weak scaling: one file per number of processes $p \in \{1,2,4,8,16\}$ with $N = 250\,000 \cdot p$ rows; strong scaling: the `_p16` file. About 5 GB in total.

| | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| Clusters $K$ / variables $D$ | 4 / 10 | 5 / 20 | 6 / 50 |
| Covariance inside a cluster | diagonal | correlated | diagonal |
| What it tests | correctness, reference for scaling | overlapping clusters, diagonal model mismatch | rare group (1%), 2% outliers, high $D$ |

### Real datasets

Public datasets, mostly medical, used for the clustering quality (none has missing values in the numeric columns):

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

## Running the experiments

- **On the cluster:** [doc/exec-cluster.md](doc/exec-cluster.md) (VPN, clone, datasets, modules, build, PBS jobs, plots). One version of one model on all the datasets per command, to stay under the queue limit: `src/script/pemc-submit.sh <diagonal|full> <sequential|baseline|opt1|opt2>`.

## Visualization of the algorithm

A small local web page shows how EM works step by step on any dataset of `data/` (diagonal or full covariance): initial centroids, E-step, convergence test and M-step for every iteration, with the formulas filled with the real numbers and a check against the C binary. How to run it: [doc/exec-visual.md](doc/exec-visual.md).

## Plots

Everything is regenerable from the logs: the report uses only the static plots produced by the two scripts in `src/plots-generate/` (rules, catalog and experiment matrix: [doc/plots.md](doc/plots.md); colors: [doc/color.md](doc/color.md)). The performance plots P1-P11 (sweeps, speedup, strong and weak scaling, placement, ranks vs threads, time breakdown, optimization gain, roofline, load imbalance) come from the cluster runs; the clustering plots C1-C6 (convergence, choice of K with BIC/AIC, PCA, centroids, cluster sizes, full vs diagonal) come from the sequential runs on a range of K. How to produce them: [doc/exec-cluster.md](doc/exec-cluster.md).

## Results

TODO

## Documentation

- [doc/exec-visual.md](doc/exec-visual.md): run the step-by-step web visualization of the EM algorithm
- [doc/exec-cluster.md](doc/exec-cluster.md): step-by-step procedure to run everything on the cluster (VPN, clone, datasets, modules, build, PBS jobs, plots)
- [doc/todo.md](doc/todo.md): task list and workflow
- [doc/todo-jobs.md](doc/todo-jobs.md): checklist of the submission commands on the cluster (batch A and batch B)
- [doc/em-explained.md](doc/em-explained.md): study guide on the EM algorithm (formulas, notation, data, parallelism)
- [doc/datasets.md](doc/datasets.md): all the datasets (synthetic profiles and variables, sizes and work per rank, file format, real sources and licenses, preprocessing, BrainWeb, verification report), ready to be reused in the report
- [doc/papers.md](doc/papers.md): free papers for the state of the art (theory, parallel EM on one and several nodes, candidate optimizations); the PDFs are in `doc/papers/`, kept only locally
- [doc/latex/Group6-Report.tex](doc/latex/Group6-Report.tex): IEEE-style report
- [Presentation](https://docs.google.com/presentation/d/1nnzwo61ZBHoQlmlYfGVbIb39rVTOexQlnH01uqBfev0/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45): project presentation slides (Google Slides)
