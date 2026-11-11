# Running the experiments on the cluster

Step-by-step procedure to reproduce all the experiments on the University of Trento cluster (PBS scheduler, `shortCPUQ` queue).

---

## Phase 1: Access and preparation

### 1. Connect to the University VPN

Use GlobalProtect:

```text
vpn-mfa.icts.unitn.it
```

### 2. Access the cluster

```bash
ssh username@baldo.disi.unitn.it
```

### 3. Clone the repository

```bash
git clone https://github.com/HPC-2627/ParallelExpectationMaximizationClustering.git
cd ParallelExpectationMaximizationClustering
```

---

## Phase 2: Datasets

The `data/` folder is **not versioned**: after the clone it does not exist, so both the synthetic and the real datasets must be created on the cluster. Run everything from the root of the repository.

### Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn ucimlrepo brainweb-dl nibabel
```

### Synthetic datasets (about 5 GB)

```bash
python3 src/dataset-managed/generate-synthetic-dataset.py
python3 src/dataset-managed/generate-synthetic-dataset.py --sweep    # D sweep files for plot P1 (K = 4, D = 5 ... 50)
```

The generation is deterministic (seed 42): the files are identical to the ones generated locally. Details in [synthetic-datasets-info.md](synthetic-datasets-info.md).

### Real datasets (about 250 MB)

```bash
python3 src/dataset-managed/generate-real-datasets.py     # download into data/real/<name>/raw/
python3 src/dataset-managed/verify-real-datasets.py       # optional: check of the downloads
python3 src/dataset-managed/convert-real-to-bin.py        # CSV -> .bin
python3 src/dataset-managed/convert-brainweb-to-bin.py    # BrainWeb (NIfTI) -> .bin
```

If the compute nodes have no internet access, run the downloads on the login node (the job only reads the `.bin` files). Sources and licenses: [real-datasets-source.md](real-datasets-source.md).

---

## Phase 3: Compilation

### Load the environment

```bash
module purge
module load OpenMPI/4.1.5-GCC-12.3.0
```

The same module is loaded inside the PBS jobs ([pemc-job.sh](../src/script/pemc-job.sh)), and [mpi_wrapper.sh](../src/script/mpi_wrapper.sh) (UCX over Ethernet) is meant for Open MPI.

### Grant execution permissions

```bash
chmod +x src/script/*.sh
```

### Build the project

```bash
make clean
make all mpi-omp
```

This produces in `bin/` the 4 executables (sequential and MPI + OpenMP, diagonal and full). `bin/` is not versioned.

---

## Phase 4: Experiments (PBS)

Submit from the root of the repository:

```bash
src/script/pemc-submit.sh                # sequential + parallel
src/script/pemc-submit.sh sequential     # only the 2 sequential versions
src/script/pemc-submit.sh parallel       # only the 2 MPI + OpenMP baselines
src/script/pemc-submit.sh sweep          # K and D sweeps (plot P1)
DRY_RUN=1 src/script/pemc-submit.sh      # prints the qsub commands without submitting
```

One job = one version with one layout, and it runs **all the datasets** (10 real + 15 synthetic; `gene_rnaseq` is skipped with full covariance). The sequential jobs use 1 core on an exclusive node. For the MPI + OpenMP baselines the layout is `NODES x RANKS_PER_NODE x THREADS` (`ncpus` per node = `RANKS_PER_NODE x THREADS`, `mpiprocs` = `RANKS_PER_NODE`) and is combined with the 4 PBS placements:

| Placement | Meaning |
|---|---|
| `pack` | the chunks go on as few nodes as possible |
| `scatter` | every chunk on a different node |
| `pack:excl`, `scatter:excl` | same, but the nodes are reserved for the job (no other job disturbs the timing) |

With one node `pack` and `scatter` coincide, so only `pack` and `pack:excl` are submitted. The layouts, memory, walltime, repetitions and subset of datasets are variables at the top of [pemc-submit.sh](../src/script/pemc-submit.sh).

Clustering runs (sequential baseline, K range per dataset, input of plots C1-C6):

```bash
src/script/pemc-cluster.sh               # ONLY=heart KMAX=8 src/script/pemc-cluster.sh to test
```

It can also be submitted with `qsub`.

### Monitor the jobs

```bash
qstat -u $USER        # state of the jobs
qstat -f <job_id>     # details of one job
qdel <job_id>         # cancel a job
```

### Test a job without the queue

```bash
PBS_O_WORKDIR=$PWD VERSION=baseline-sequential-full ONLY=heart_failure src/script/pemc-job.sh
```

### Output

- logs: `results/logs/<version>_<placement>_<nodes>n<ranks>r<threads>t.log` (`<version>_seq.log` for the sequential ones); sweeps in `results/logs/sweep/`; clustering in `results/logs/cluster/` and `results/clusters/`;
- standard output/error of the jobs: `results/logs/pbs/<job>.out|.err` (not versioned).

---

## Phase 5: Plots

When all the jobs are finished, either generate the plots on the cluster (Python 3 with `numpy`, `pandas`, `matplotlib`) or copy the logs to the local machine. From the local machine:

```bash
scp -r username@baldo.disi.unitn.it:ParallelExpectationMaximizationClustering/results/logs results/
scp -r username@baldo.disi.unitn.it:ParallelExpectationMaximizationClustering/results/clusters results/
```

Then, from the root of the repository:

```bash
python3 src/plots-generate/plots-performance.py     # P1-P11 -> results/plots/
python3 src/plots-generate/plots-cluster.py         # C1-C6 + results/tables/clustering-summary.csv
```

Options, catalog and experiment matrix: [plots.md](plots.md).
