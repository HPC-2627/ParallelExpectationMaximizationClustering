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

Each model has four versions, for a total of eight programs: `sequential` (single process, no parallelism: correctness reference and serial time for the speedup), `baseline-MPI-OpenMP` (first hybrid parallelization: rows split among MPI ranks, OpenMP threads inside each rank; the reference of the parallel tests) and two optimizations of the baseline, `opt1-MPI-OpenMP` and `opt2-MPI-OpenMP`. In both models `opt1` optimizes the **computation** and `opt2` the **communication** between the MPI processes. The details can differ because each model has its own bottleneck (the diagonal one is memory-bound, $O(NKD)$; the full one is compute-bound for large $D$, $O(NKD^2)$): `opt1` full contains, besides the fused pass of `opt1` diagonal, the blocking of the rows. Both `opt2` start from the baseline of their model and change only the communication, so each is compared with that baseline. Every optimization is compared only with versions of its own model.

#### Diagonal covariance ([`src/diagonal/`](src/diagonal/))

| Version | What it is |
|---|---|
| [`sequential`](src/diagonal/PEMC-baseline-sequential-diagonal.c) | correctness reference and serial time |
| [`baseline-MPI-OpenMP`](src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c) | hybrid MPI + OpenMP parallelization, two `MPI_Allreduce` per iteration |
| [`opt1-MPI-OpenMP`](src/diagonal/PEMC-opt1-MPI-OpenMP-diagonal.c) | **computation**: E-step and M-step fused in one pass (no responsibility matrix), one parallel region per iteration, a thread-local buffer per thread (no false sharing); communication as in the baseline |
| [`opt2-MPI-OpenMP`](src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c) | **communication**: ONE `MPI_Allreduce` per iteration instead of two (the log-likelihood travels in the same buffer as the M-step sums); computation identical to the baseline |

#### Full covariance ([`src/full/`](src/full/))

| Version | What it is |
|---|---|
| [`sequential`](src/full/PEMC-baseline-sequential-full.c) | correctness reference and serial time |
| [`baseline-MPI-OpenMP`](src/full/PEMC-baseline-MPI-OpenMP-full.c) | hybrid MPI + OpenMP parallelization (Cholesky), two `MPI_Allreduce` per iteration |
| [`opt1-MPI-OpenMP`](src/full/PEMC-opt1-MPI-OpenMP-full.c) | **computation, all the optimizations**: E-step and M-step fused in one pass (no responsibility matrix), one parallel region per iteration (the K Cholesky factorizations are also split among the threads), per-thread buffers without false sharing, and **blocks of B rows** (option `-b`, default 64) copied transposed into a per-thread buffer: the forward substitution runs 8 rows at a time (independent chains in one SIMD register, each element of `L_k` loaded once for 8 rows) and the sums `sxx` are accumulated in registers over the block and written once per block (matrix-vector -> matrix-matrix operations, Kwedlo 2023). Communication as in the baseline |
| [`opt2-MPI-OpenMP`](src/full/PEMC-opt2-MPI-OpenMP-full.c) | **communication** (same idea as the diagonal `opt2`): ONE `MPI_Allreduce` per iteration instead of two (the log-likelihood in the same buffer as the sums) and `sxx` sent as its lower triangle only (K·D(D+1)/2 values instead of K·D²); computation identical to the full baseline |

All the versions of the same model must end with the same log-likelihood and the same number of iterations (same dataset, `-k` and `-s`).

## Repository structure

```
.
├── src/
│   ├── diagonal/          # Diagonal covariance: PEMC-<version>-diagonal.c (sequential, baseline, opt1, opt2)
│   ├── full/              # Full covariance (Cholesky): PEMC-<version>-full.c (sequential, baseline, opt1, opt2)
│   ├── common/            # Shared code (no MPI): dataset I/O, GMM core, CLI options, logging, perf model, evaluation, timing
│   ├── script/            # PBS launchers for the cluster + UCX wrapper for multi-node mpiexec + hardware of the nodes
│   ├── dataset-managed/   # Python scripts that generate, download, verify and convert the datasets -> data/
│   ├── plots-generate/    # Python scripts that generate the performance and clustering plots -> results/plots/performance/ and results/plots/cluster/ (EM steps figure -> doc/img/)
│   └── visual-sequential/ # Local web visualization of the EM algorithm, step by step (see below)
├── include/               # Headers of the shared code, plus MPI-only helpers and timer macros (compile with -Iinclude -lm)
├── data/                  # NOT versioned: filled locally by src/dataset-managed/
│   ├── real/              #   real datasets, one folder each (raw files + converted .bin)
│   └── synthetic/         #   synthetic datasets, one folder per profile (see Data)
├── results/
│   ├── logs/              # Raw execution logs: <model>/<stage>/ for the scaling runs, plus sweep/, cluster/ and pbs/ (PBS output)
│   ├── clusters/          # Clustering results saved with --save, used by the clustering plots
│   ├── tables/            # Summary tables (clustering, hardware, node memory vs speed)
│   └── plots/             # Generated plots (PDF)
├── doc/                   # Documentation: how to run, EM study notes, dataset docs, papers, task list, LaTeX report (see Documentation below)
├── makefile               # Build rules (make builds bin/, which is not versioned)
└── README.md
```

## Data

Every dataset is converted to the same binary format (header `int32` N, D + `float64` matrix in row-major order, labels in a separate file) and every column is standardized (z-score).

- **Code:** [`src/dataset-managed/`](src/dataset-managed/) (generate, download, convert, verify)
- **How to run:** [exec-cluster.md, Phase 2: Datasets](doc/exec-cluster.md#phase-2-datasets)
- **Docs:** [doc/datasets.md](doc/datasets.md) (profiles, sources, format)
- **Output:** `data/` (not versioned, created on each machine)

### Synthetic datasets

Three profiles generated from a known Gaussian mixture (true labels available, deterministic with seed 42). They are used for correctness checks and for the weak and strong scaling, because the real datasets are too small to show any scaling. Weak scaling: one file per number of processes $p \in \{1,2,4,8,16\}$ with $N = 250\,000 \cdot p$ rows; strong scaling: the `_p16` file. About 5 GB in total.

| | `synth_clean` | `synth_overlap` | `synth_rare` |
|---|---|---|---|
| Clusters $K$ / variables $D$ | 4 / 10 | 5 / 20 | 6 / 50 |
| Covariance inside a cluster | diagonal | correlated | diagonal |
| What it tests | correctness, reference for scaling | overlapping clusters, diagonal model mismatch | rare group (1%), 2% outliers, high $D$ |

### Sweep datasets

The sweeps check in practice the cost of an iteration, $O(NKD)$ for the diagonal model and $O(NKD^2)$ for the full model, by changing **one parameter at a time** with $N = 250\,000$ and the layout fixed (1 node, 1 rank × 8 threads, `pack:excl`). Only the four baselines are run. On a log-log plot the expected slope is 1 in $K$ for both models, and 1 (diagonal) or 2 (full) in $D$. Only time is measured, not memory.

| Sweep | Dataset | Varies | Fixed |
|---|---|---|---|
| $K$ | `synth_clean_p01` (no new file) | $K \in \{2, 4, 8, 16, 32, 64\}$ | $D = 10$ |
| $D$ | `synth_sweep_d005` … `synth_sweep_d050` (same parameters as `synth_clean`, about 240 MB) | $D \in \{5, 10, 20, 35, 50\}$ | $K = 4$ |

Run with `src/script/pemc-submit.sh sweep` (8 jobs); the logs go to `results/logs/sweep/`. Details in [datasets.md, 3.7](doc/datasets.md#37-k-and-d-sweeps).

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

- **Code:** [`src/diagonal/`](src/diagonal/), [`src/full/`](src/full/), [`src/common/`](src/common/), [`include/`](include/) (shared code)
- **Scripts:** [`src/script/`](src/script/) (PBS submit + job), explained in [doc/script.md](doc/script.md)
- **How to run:** [doc/exec-cluster.md](doc/exec-cluster.md) (VPN, build, PBS jobs)
- **Output:** [`results/logs/`](results/logs/) (one folder per model and version)

### Cluster hardware

All the runs use the HPC3 cluster of the University of Trento (PBS Pro, queue `shortCPUQ`), only on nodes of type `cpu_type=Xeon6252N`.

| Item | Value |
|---|---|
| CPU | 4 × Intel Xeon Gold 6252N (Cascade Lake), 2.3 GHz (turbo 3.6 GHz, Intel catalog), AVX-512 |
| Cores | 96 per node (4 sockets × 24), no hyper-threading, 4 NUMA nodes (one per socket) |
| Caches | L1d 32 KB and L2 1 MB per core, L3 35.75 MB per socket |
| Memory | 256 GB to 2 TB DDR4 per node, 6 channels per socket (2, 4 or 6 populated) |
| Network | Ethernet 10 Gb/s |
| Nodes used | 40 (groups `c10`, `c11`, `c12`) |
| Software | Rocky Linux 9.8, GCC 12.3.0 (`-O3 -fopenmp`, no `-march`), Open MPI 4.1.5 + UCX 1.14.1 over TCP |

Collected with [`src/script/pemc-hwinfo.sh`](src/script/pemc-hwinfo.sh); details in [results/tables/hardware.md](results/tables/hardware.md).

Do the memory channels change the times? No: [results/tables/node-memory.md](results/tables/node-memory.md) puts the memory configuration of every node next to the speed of the MPI ranks that ran on it (ranks of the same run on 4- and 6-channel nodes within 1% on average; the imbalance between ranks depends on the code, not on the node type). Generated by [`src/tables/node-memory.py`](src/tables/node-memory.py) from `results/hwinfo/` (copied from the cluster, not versioned), the PBS outputs and the logs.

## Visualization of the algorithm

A small local web page shows how EM works step by step on any dataset of `data/` (diagonal or full covariance): initial centroids, E-step, convergence test and M-step for every iteration, with the formulas filled with the real numbers and a check against the C binary.

- **Code:** [`src/visual-sequential/`](src/visual-sequential/) (web page + Python server)
- **How to run:** [doc/exec-visual.md](doc/exec-visual.md) (local, in the browser)

## Plots

Everything is regenerable from the logs. The performance plots P1-P11 (sweeps, speedup, strong and weak scaling, placement, ranks vs threads, time breakdown, optimization gain, roofline, load imbalance) come from the cluster runs; the clustering plots C1-C6 (convergence, choice of K with BIC/AIC, PCA, centroids, cluster sizes, full vs diagonal) come from the sequential runs on a range of K.

- **Code:** [`src/plots-generate/`](src/plots-generate/) (performance P1-P11, clustering C1-C6)
- **How to run:** [exec-cluster.md, Phase 5: Plots](doc/exec-cluster.md#phase-5-plots)
- **Docs:** [doc/plots.md](doc/plots.md) (plot catalog)
- **Figures:** [`results/plots/`](results/plots/) (generated PDFs: `performance/` P1-P11, `cluster/` C1-C6)

## Results

TODO

## Documentation

- [doc/exec-visual.md](doc/exec-visual.md): run the step-by-step EM visualization
- [doc/exec-cluster.md](doc/exec-cluster.md): run everything on the cluster, from VPN to plots
- [doc/script.md](doc/script.md): the scripts in `src/script/` and their variables
- [doc/todo.md](doc/todo.md): task list and workflow
- [doc/em-explained.md](doc/em-explained.md): EM algorithm (formulas, notation, parallelism)
- [doc/datasets.md](doc/datasets.md): synthetic, sweep and real datasets (sources, format)
- [doc/papers_and_references.md](doc/papers_and_references.md): papers and references
- [doc/latex/Group6-Report.pdf](doc/latex/Group6-Report.pdf): IEEE-style report
- [Presentation](https://docs.google.com/presentation/d/1nnzwo61ZBHoQlmlYfGVbIb39rVTOexQlnH01uqBfev0/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45): slides (Google Slides)
