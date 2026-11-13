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

The generation is deterministic (seed 42): the files are identical to the ones generated locally. Details in [datasets.md](datasets.md#3-synthetic-datasets).

### Real datasets (about 250 MB)

```bash
python3 src/dataset-managed/generate-real-datasets.py     # download into data/real/<name>/raw/
python3 src/dataset-managed/verify-real-datasets.py       # optional: check of the downloads
python3 src/dataset-managed/convert-real-to-bin.py        # CSV -> .bin
python3 src/dataset-managed/convert-brainweb-to-bin.py    # BrainWeb (NIfTI) -> .bin
```

If the compute nodes have no internet access, run the downloads on the login node (the job only reads the `.bin` files). Sources and licenses: [datasets.md](datasets.md#41-sources-and-licenses).

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

This produces in `bin/` the executables of the versions written so far: the 2 sequential baselines (diagonal, full), the MPI + OpenMP baselines (diagonal, full) and `PEMC-opt1-MPI-OpenMP-diagonal`. The versions still empty (opt2, opt1 full) are not in the makefile yet. `bin/` is not versioned.

---

## Phase 4: Experiments (PBS)

Submit from the root of the repository.

**One stage at a time, both models, all the data.** One command = one version of one model (`diagonal` or `full`) on **all the datasets** (10 real + 15 synthetic; `gene_rnaseq` is skipped with the full model) with all the layouts below. The queue has a **per-user limit** on the jobs (about 30 between running and queued, counting also the jobs of other projects): with everything at once some `qsub` are rejected. So submit **one stage at a time**, and send the next one when the jobs of the previous one have finished (or at least started):

Every stage is submitted in **two batches** (`PLACES_LIST`), at most 8 jobs each:

| Batch | Placements | Jobs per parallel stage | Why separate |
|---|---|---|---|
| **A** | `pack`, `scatter` | 8 | they start as soon as some cores are free: these are the runs that give the results |
| **B** | `pack:excl`, `scatter:excl` | 8 | the nodes are reserved for the job (clean timing), but on a busy cluster they can stay queued for a long time (see the note below) |

The sequential stage is one job, already on a node used only by itself: it is submitted once, in batch A. Send **all of batch A first**, check `qstat -u $USER`, then batch B when there is room under the limit.

```bash
make clean && make all mpi-omp                       # build first (module OpenMPI loaded, see Phase 3)
DRY_RUN=1 src/script/pemc-submit.sh diagonal baseline   # check the qsub commands of a stage (nothing is sent)

# ---- batch A: pack + scatter ---------------------------------------------------------------
export PLACES_LIST="pack scatter"
src/script/pemc-submit.sh diagonal sequential        # 1 job
src/script/pemc-submit.sh diagonal baseline          # 8 jobs
src/script/pemc-submit.sh diagonal opt1              # 8 jobs
src/script/pemc-submit.sh diagonal opt2              # 8 jobs, when opt2 is written
src/script/pemc-submit.sh full sequential            # 1 job
src/script/pemc-submit.sh full baseline              # 8 jobs
src/script/pemc-submit.sh full opt1                  # 8 jobs, when opt1 full is written
src/script/pemc-submit.sh full opt2                  # 8 jobs, when opt2 full is written

# ---- batch B: pack:excl + scatter:excl (no sequential: it is already exclusive) -------------
export PLACES_LIST="pack:excl scatter:excl"
src/script/pemc-submit.sh diagonal baseline          # 8 jobs
src/script/pemc-submit.sh diagonal opt1              # 8 jobs
src/script/pemc-submit.sh diagonal opt2              # 8 jobs, when opt2 is written
src/script/pemc-submit.sh full baseline              # 8 jobs
src/script/pemc-submit.sh full opt1                  # 8 jobs, when opt1 full is written
src/script/pemc-submit.sh full opt2                  # 8 jobs, when opt2 full is written
```

Sweeps of K and D (plot P1: time per iteration vs K and vs D), one fixed layout (1 node, 1 rank x 8 threads, node used only by the job). They need the D sweep files made in Phase 2 (`generate-synthetic-dataset.py --sweep`); the logs go in `results/logs/sweep/`:

```bash
src/script/pemc-submit.sh sweep
```

One job = one version with one layout, and it runs **all the datasets** (10 real + 15 synthetic; `gene_rnaseq` is skipped with full covariance). The sequential jobs use 1 core on an exclusive node. For the MPI + OpenMP versions the layout is `NODES x RANKS_PER_NODE x THREADS` (`ncpus` per node = `RANKS_PER_NODE x THREADS`, `mpiprocs` = `RANKS_PER_NODE`) and is combined with the 4 PBS placements:

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

- logs: `results/logs/<model>/<stage>/<version>_<placement>_<nodes>n<ranks>r<threads>t.log` (`<version>_seq.log` for the sequential ones), with model = `diagonal` | `full` and stage = `sequential` | `baseline` | `opt1` | `opt2` (folders already in the repository); sweeps in `results/logs/sweep/`; clustering in `results/logs/cluster/` and `results/clusters/`;
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
