# Running locally

How to run the programs by hand on one machine (laptop or a single node, without PBS): download and conversion of the datasets, build, usage and options of the programs, log format and plots. The commands are the same ones used by the PBS scripts. The step-by-step procedure for the cluster is in [exec-cluster.md](exec-cluster.md); the step-by-step web visualization of the algorithm is in [exec-visual.md](exec-visual.md). Run everything from the root of the repository.

## Synthetic datasets

```bash
python3 src/dataset-managed/generate-synthetic-dataset.py            # about 5 GB in data/synthetic/
python3 src/dataset-managed/generate-synthetic-dataset.py --sweep    # D sweep files for plot P1 (K = 4, D = 5 ... 50)
python3 src/dataset-managed/generate-synthetic-dataset.py --help     # options (e.g. a smaller size)
```

It needs Python 3 with NumPy. The generation is deterministic (default seed 42). Details of the datasets: [datasets.md](datasets.md#3-synthetic-datasets).

## Real datasets: download, verification, conversion

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

### Verification

```bash
python3 src/dataset-managed/verify-real-datasets.py
```

It reports, for each dataset, the size, the number of numeric columns, the empty cells (NaN, empty strings, `?`, `NA`), the duplicate rows and the label distribution. The report is written to section 5 of [doc/datasets.md](datasets.md#5-verification-of-the-real-datasets), between the markers `BEGIN/END verify-real-datasets` (the rest of the file is not touched). A dataset is `COMPLETE` if its numeric columns have no empty cell, otherwise `DISCARD`.

### Conversion to binary

The raw CSV files are kept as served by the source. To use the real datasets with the same C loader of the synthetic ones, convert them:

```bash
python3 src/dataset-managed/convert-real-to-bin.py            # all the datasets
python3 src/dataset-managed/convert-real-to-bin.py heart_failure       # only some
```

For each dataset it writes in `data/real/<name>/`:
- `<name>.bin`: same format as the synthetic data (8-byte header with two `int` N and D, then $N \cdot D$ `double` row-major);
- `<name>_labels.bin`: one `int` per row (classes encoded as 0..C-1), only if the label is categorical (`parkinson` has continuous targets, so no labels);
- `<name>.json`: columns used and dropped, mean and standard deviation of the standardization, class mapping.

Preprocessing: only the numeric columns are kept (identifiers removed), constant columns are dropped, every column is standardized (z-score). A dataset with empty cells (`DISCARD`) would be skipped: nothing is imputed. `brainweb` is not CSV and has its own script (below).

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
make mpi-omp      # builds bin/PEMC-baseline-MPI-OpenMP-diagonal and bin/PEMC-baseline-MPI-OpenMP-full (mpicc -fopenmp)
make clean
```

The sources of the versions are in `src/diagonal/` and `src/full/`; the shared code in `src/common/` does not use MPI, so the same files are compiled by both rules. Every executable goes flat in `bin/` with the name of its source (`src/diagonal/PEMC-xxx.c` -> `bin/PEMC-xxx`). To add a version (e.g. `opt1`), append its name to `MPI_BINS` in the [makefile](../makefile). The `bin/` folder is not versioned.

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

Run the sequential versions from the root of the repository: the only file written is the log `results/logs/baseline-sequential-diagonal.log` (a block is appended at every run; the folder is created if missing).

| Option | Meaning |
|---|---|
| `-d, --dataset FILE` | dataset in binary format (header N, D + float64 row-major) |
| `-k, --clusters K` | number of clusters |
| `-n, --rows N` | use only the first N rows of the file (default: all) |
| `-i, --max-iter N` | maximum number of iterations (default 200) |
| `-t, --tol X` | relative tolerance on the log-likelihood (default 1e-6) |
| `-s, --seed S` | seed of the initialization (default 42): the same seed gives the same initialization in every version |
| `-r, --restarts R` | number of initializations, the best one is kept (default 1); the times are summed over all the restarts |
| `-w, --warmup N` | warm-up EM iterations done before the timed run and then discarded (default 3, 0 = off). They wake up the OpenMP threads, MPI and the caches; the result is identical with and without warm-up, and their time is not in `t_total` nor in the other times (it is logged as `t_warmup`) |
| `-l, --log FILE.log` | log file (default `results/logs/<name>.log`) |
| `--no-log` | do not write the log |
| `--save DIR` | save the clustering result (labels of every row, weights and means, log-likelihood curve) in `DIR`, see `include/cluster-out.h`. Sequential versions only; used by the clustering plots |
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
warmup_iters   : 3
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
t_warmup       : 0.231044
flops          : 3.1e+09
bytes          : 2.4e+09
rank_compute   : 0.61 0.60 0.63 0.62
t_iteration    : 0.079643
```

Times are in seconds. `hostname` and `cpu_model` identify the node (the nodes of the cluster are not all equal). `ari` (adjusted Rand index against the true labels) is `n/a` if the dataset has no labels. `t_warmup` is the time of the `warmup_iters` throwaway iterations, not included in `t_total`. `flops` and `bytes` are an analytic count of the floating-point operations and of the main-memory traffic of the run (`include/perfmodel.h`, not hardware counters), used for the roofline. `rank_compute` is the compute time (E-step + M-step, no MPI) of every MPI rank in rank order (`n/a` in the sequential versions), used for the load imbalance. `t_iteration` is `(t_estep + t_mstep + t_comm) / iterations` (only with one restart). To read it in Python, split the file on `==== run ====` and every line on the first `:`.

**Correctness check between versions:** run them with the same dataset, `-k` and `-s`: they must end with the same log-likelihood (up to rounding) and the same number of iterations. Compare only versions of the same model: `-diagonal` with `-diagonal`, `-full` with `-full`.

## Plots

Everything is regenerable from the logs: the report uses only the static plots produced by the two scripts in `src/plots-generate/` (rules, catalog and experiment matrix: [doc/plots.md](plots.md); colors: [doc/color.md](color.md)). Python 3 with `numpy`, `pandas` and `matplotlib` (`pip install numpy pandas matplotlib`). Run everything from the root of the repository. A plot whose data is missing is skipped and the reason is printed.

**Performance plots (P1-P11).** First run the experiments on the cluster ([exec-cluster.md](exec-cluster.md)), then:

```
python3 src/plots-generate/plots-performance.py                      # all the performance plots -> results/plots/
python3 src/plots-generate/plots-performance.py --plots P2 P6        # only some of them
python3 src/plots-generate/plots-performance.py --placement scatter:excl --large synth_clean_p16 --small cardiotocography
python3 src/plots-generate/plots-performance.py --peak-gflops 150 --peak-bw 60   # roofline P10 with the node ceilings
```

Main options: `--placement` (the placement fixed in every plot except P6; choose it after looking at P6, default `pack`), `--large` / `--small` (datasets of P2/P4/..., automatic if missing), `--version` (version of P6/P7/P11, default the most optimized present), `--models`, `--fmt pdf png`, `--logs`, `--out`. Speedup is always against the sequential version of the same model; opt1 and opt2 (P9) are compared with the parallel baseline MPI + OpenMP. The roofline ceilings (`--peak-gflops`, `--peak-bw`) are those of one node: measure the bandwidth with STREAM.

**Clustering plots (C1-C6 and table).** The clustering is the same in every version, so the runs are made with the sequential baseline (both models), on a range of K per dataset (`src/script/pemc-cluster.sh`, see [exec-cluster.md](exec-cluster.md)); at the true K the result is also saved with `--save`:

```
python3 src/plots-generate/plots-cluster.py                         # C1-C6 + results/tables/clustering-summary.csv
python3 src/plots-generate/plots-cluster.py --plots C2 C6 --datasets heart_failure parkinson
```

Output files, one per model (`diagonal` / `full`) in `results/plots/`: `P1-sweep`, `P2-speedup`, `P4-strong`, `P5-weak`, `P6-placement`, `P7-ranks-threads`, `P8-breakdown`, `P9-opt-gain`, `P10-roofline`, `P11-imbalance`, `C1-convergence`, `C2-choose-k`, `C3-pca`, `C4-centroids`, `C5-sizes` (for example `P2-speedup-full.pdf`); `C6-bic-full-vs-diagonal.pdf` compares the two models.

