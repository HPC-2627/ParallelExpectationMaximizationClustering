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
ssh username@hpc3-login1.unitn.it
```

If `ssh` stays silent and never asks for the password, the VPN is not connected (or the host name is wrong): check with `ssh -v`.

### 3. Clone the repository

```bash
git clone https://github.com/HPC-2627/ParallelExpectationMaximizationClustering.git
cd ParallelExpectationMaximizationClustering
```

If the repository is already on the cluster, update it instead (the scripts change: always run the last version):

```bash
cd ParallelExpectationMaximizationClustering
git pull
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

The generation is deterministic (seed 42): the files are identical to the ones generated locally.

### Real datasets (about 250 MB to download, about 650 MB after the conversion)

```bash
python3 src/dataset-managed/generate-real-datasets.py     # download into data/real/<name>/raw/
python3 src/dataset-managed/verify-real-datasets.py       # optional: check of the downloads
python3 src/dataset-managed/convert-real-to-bin.py        # CSV -> .bin
python3 src/dataset-managed/convert-brainweb-to-bin.py    # BrainWeb (NIfTI) -> .bin
```

If the compute nodes have no internet access, run the downloads on the login node (the job only reads the `.bin` files). Sources and licenses: [datasets.md](datasets.md#31-sources-and-licenses).

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

This produces in `bin/` the executables of all the versions: `baseline-sequential` (diagonal and full) and, for each model, `baseline-MPI-OpenMP`, `opt1` and `opt2`. `bin/` is not versioned. All the versions use the same flags (`-O3`, no `-march`); `make mpi-omp ARCH=-march=cascadelake` adds architecture flags to all of them, for a separate experiment only.

---

## Phase 4: Experiments (PBS)

Submit from the root of the repository. What every script does: [script.md](script.md).

### What the scripts guarantee (guidelines for comparable times)

The cluster is **heterogeneous** (several Xeon generations, from 2014 to 2023) and the nodes are shared: without precautions every job lands on a different CPU and the times of `baseline-MPI-OpenMP`, opt1 and `baseline-sequential` cannot be compared. [pemc-submit.sh](../src/script/pemc-submit.sh) and [pemc-job.sh](../src/script/pemc-job.sh) take care of it:

| Guideline | How | Variable (top of `pemc-submit.sh`) |
|---|---|---|
| **Same CPU for every job** (`baseline-sequential` included) | PBS gives only nodes with `cpu_type=Xeon6252N` (87 nodes, the most common type; `pbsnodes -a` lists the others) | `NODE_FILTER` (default `cpu_type=Xeon6252N`, empty = any node) |
| **No log from a wrong CPU** | at start every job checks the CPU model of all its nodes; if one is different it stops **before** writing any log | `REQUIRE_CPU` (default `6252N`, empty = no check). Must name the same CPU as `NODE_FILTER` |
| **One core per OpenMP thread** | `mpiexec --map-by slot:PE=THREADS --bind-to core`: every rank gets `THREADS` cores. Without it Open MPI binds each rank to ONE core and all its threads share it (no speedup from the threads) | - |
| **Multi-node runs on all the nodes** | `--hostfile $PBS_NODEFILE`: Open MPI does not see the other nodes of the job by itself (same as Exercise 10) | - |
| **Repetitions** | every (dataset, layout) runs 3 times; the plots use the median, error bar min-max | `REPS` (default 3) |

### Before a new campaign: clean up

Two campaigns running at the same time write **into the same log files** (blocks mixed, double repetitions). Before submitting again:

```bash
qselect -u $USER | xargs -r qdel                       # cancel all my jobs (running and queued)
qstat -u $USER                                         # wait until nothing is left
find results/logs -name '*.log' -delete                # old logs (other CPU / old scripts)
rm -f results/logs/pbs/*.out results/logs/pbs/*.err
git pull                                               # last version of scripts and code
module load OpenMPI/4.1.5-GCC-12.3.0                   # needed by mpi-omp
make clean && make all mpi-omp                         # plain "make" builds ONLY baseline-sequential:
ls bin/                                                # the MPI jobs then fail at once (exe not found)
unset CONFIGS_LIST                                     # no leftover filter from a previous submission
```

### Submit

**One stage at a time, both models, all the data.**

One command = one version of one model (`diagonal` or `full`) on **all the datasets** (10 real + 15 synthetic; `gene_rnaseq` is skipped with the full model) with all the layouts below. 

The queue has a **per-user limit** on the jobs (about 30 between running and queued, counting also the jobs of other projects): with everything at once some `qsub` are rejected. 

So submit **one stage at a time**, and send the next one when the jobs of the previous one have finished (or at least started):

Every stage is submitted in **two batches** (`PLACES_LIST`), at most 8 jobs each:

| Batch | Placements | Jobs per parallel stage | Why separate |
|---|---|---|---|
| **A** | `pack`, `scatter` | 8 | they start as soon as some cores are free: these are the runs that give the results |
| **B** | `pack:excl`, `scatter:excl` | 8 | the nodes are reserved for the job (clean timing), but on a busy cluster they can stay queued for a long time (no Xeon6252N node is ever completely empty) |

The `sequential` stage (`baseline-sequential`) is one job, already on a node used only by itself: it is submitted once, in batch A. 

Send **all of batch A first**, check `qstat -u $USER`, then batch B when there is room under the limit.

```bash
make clean && make all mpi-omp                       # build first (module OpenMPI loaded, see Phase 3)

# ---- batch A: pack + scatter ---------------------------------------------------------------
export PLACES_LIST="pack scatter"
src/script/pemc-submit.sh diagonal sequential        # 1 job
src/script/pemc-submit.sh diagonal baseline          # 8 jobs
src/script/pemc-submit.sh diagonal opt1              # 8 jobs
src/script/pemc-submit.sh diagonal opt2              # 8 jobs
src/script/pemc-submit.sh full sequential            # 1 job
src/script/pemc-submit.sh full baseline              # 8 jobs
src/script/pemc-submit.sh full opt1                  # 8 jobs
src/script/pemc-submit.sh full opt2                  # 8 jobs

# ---- batch B: pack:excl + scatter:excl (no baseline-sequential: it is already exclusive) -------------
export PLACES_LIST="pack:excl scatter:excl"
src/script/pemc-submit.sh diagonal baseline          # 8 jobs
src/script/pemc-submit.sh diagonal opt1              # 8 jobs
src/script/pemc-submit.sh diagonal opt2              # 8 jobs
src/script/pemc-submit.sh full baseline              # 8 jobs
src/script/pemc-submit.sh full opt1                  # 8 jobs
src/script/pemc-submit.sh full opt2                  # 8 jobs
```

Sweeps of K and D (plot P1: time per iteration vs K and vs D), one fixed layout (1 node, 1 rank x 8 threads, node used only by the job). 

They need the D sweep files made in Phase 2 (`generate-synthetic-dataset.py --sweep`); the logs go in `results/logs/sweep/`:

```bash
src/script/pemc-submit.sh sweep                       # all 8 versions: 16 jobs (K and D)
```

The first campaign swept only `baseline-sequential` and `baseline-MPI-OpenMP`. The opt1/opt2 sweeps were added later, without re-running the others (the logs are appended: giving a version again would duplicate its runs):

```bash
SWEEP_VERSIONS="opt1-MPI-OpenMP-diagonal opt2-MPI-OpenMP-diagonal opt1-MPI-OpenMP-full opt2-MPI-OpenMP-full" \
DRY_RUN=1 src/script/pemc-submit.sh sweep               # check: 8 qsub (4 versions x K, D)
SWEEP_VERSIONS="opt1-MPI-OpenMP-diagonal opt2-MPI-OpenMP-diagonal opt1-MPI-OpenMP-full opt2-MPI-OpenMP-full" \
src/script/pemc-submit.sh sweep                         # 8 jobs, a few minutes each once running
```

`SWEEP_PLACE` changes the placement (default `pack:excl`, the same as the first sweeps; `pack` starts sooner but the new curves would then be on a shared node). P1 draws every version found in `results/logs/sweep/`, no change to the plot script.

After opt2 full was rewritten on `baseline-MPI-OpenMP` (2026-11-22), only its sweep was run again (2026-11-23, 2 jobs `o2f-swK`, `o2f-swD`, `pack:excl`). The old log was moved away first, because the logs are appended:

```bash
mv results/logs/sweep/opt2-MPI-OpenMP-full_pack_excl_1n1r8t.log old-logs/
SWEEP_VERSIONS="opt2-MPI-OpenMP-full" src/script/pemc-submit.sh sweep
grep -c '==== run' results/logs/sweep/opt2-MPI-OpenMP-full_pack_excl_1n1r8t.log   # 33 when done
```

#### Runs on 1, 2 and 4 cores of one node (speedup curves below 8 cores)

The default layouts start at 8 cores, and 8 -> 16 -> 32 cores also changes 1 -> 2 -> 4 nodes: without these runs P2/P3, P4 and P5 jump from 1 to 8 cores. They give:

- the **scaling inside one node** (1 -> 2 -> 4 -> 8 cores), separated from the network;
- every version **on 1 core** (`1:1:1`): the gain of the optimization alone, so the efficiency above 1 of opt1 full (P2, P5) can be split into optimization x parallelism;
- two more points of the **weak scaling** (`synth_overlap_p02` on 2 cores, `p04` on 4 cores, as in P5).

All the splits of 1, 2 and 4 cores, as for 8 cores:

| Cores | Layouts (`NODES:RANKS_PER_NODE:THREADS`) |
|---|---|
| 1 | `1:1:1` |
| 2 | `1:1:2`, `1:2:1` |
| 4 | `1:1:4`, `1:2:2`, `1:4:1` |

One node, so `scatter` = `pack`: 6 layouts = **6 jobs per version**, 36 in all (`baseline-MPI-OpenMP`, opt1, opt2 x 2 models). The logs are new files (`<version>_pack_1n1r1t.log`, ...): nothing is overwritten. The longest jobs are the 1-core ones of the full model: about 2.5-3.5 h on a shared node for all the datasets (`baseline-sequential` full takes 148 min on an exclusive node): the 6 h walltime is enough.

```bash
export CONFIGS_LIST="1:1:1 1:1:2 1:2:1 1:1:4 1:2:2 1:4:1"
export PLACES_LIST="pack"                               # pack only (see below)
DRY_RUN=1 src/script/pemc-submit.sh diagonal baseline   # check: prints the 6 qsub, submits nothing

src/script/pemc-submit.sh diagonal baseline          # 6 jobs
src/script/pemc-submit.sh diagonal opt1              # 6 jobs
src/script/pemc-submit.sh diagonal opt2              # 6 jobs
src/script/pemc-submit.sh full baseline              # 6 jobs
src/script/pemc-submit.sh full opt1                  # 6 jobs
src/script/pemc-submit.sh full opt2                  # 6 jobs

unset CONFIGS_LIST PLACES_LIST                       # back to the default layouts and placements
```

**Only `pack`, no exclusive runs for these layouts.** We did not run them with `pack:excl` because of the time: the exclusive jobs wait hours in the queue (no Xeon6252N node is ever completely free), and the plots use `pack` for all the core counts. The report says so in the Measurement Methodology.

Batch A is enough for the plots (`plots-performance.py` uses the placement `pack` by default); batch B only completes the set of placements. After "per-user limit" errors resume the same group with `SKIP=<jobs already submitted>`. When the jobs are done copy `results/logs/diagonal/` and `results/logs/full/` to the laptop and run `plots-performance.py` again: the new core counts are picked up from the logs, no change to the scripts (P7 keeps only the splits of the 8 cores of the node).

One job = one version with one layout, and it runs **all the datasets** (10 real + 15 synthetic; `gene_rnaseq` is skipped with full covariance). The `baseline-sequential` jobs use 1 core on an exclusive node. For the MPI + OpenMP versions the layout is `NODES x RANKS_PER_NODE x THREADS` (`ncpus` per node = `RANKS_PER_NODE x THREADS`, `mpiprocs` = `RANKS_PER_NODE`) and is combined with the 4 PBS placements:

| Placement | Meaning |
|---|---|
| `pack` | the chunks go on as few nodes as possible |
| `scatter` | every chunk on a different node |
| `pack:excl`, `scatter:excl` | same, but the nodes are reserved for the job (no other job disturbs the timing) |

With one node `pack` and `scatter` coincide, so only `pack` and `pack:excl` are submitted. Note: with `pack` PBS may put several chunks on the **same** node (e.g. a "2n2r4t" job with its 4 ranks on one node, communication through shared memory): for a real comparison across nodes use the `scatter` runs. 

The layouts, memory, walltime, repetitions, node type and subset of datasets are variables at the top of [pemc-submit.sh](../src/script/pemc-submit.sh).

Other options of `pemc-submit.sh` (environment variables, combined with the commands above):

```bash
CONFIGS_LIST="2:2:4 4:2:4" src/script/pemc-submit.sh diagonal opt1   # only these layouts (e.g. to resubmit failed jobs)
SKIP=4 src/script/pemc-submit.sh diagonal opt1                       # skip the first 4 jobs (after "per-user limit" errors)
REPS=1 src/script/pemc-submit.sh diagonal baseline                   # 1 repetition (quick test)
NODE_FILTER= REQUIRE_CPU= src/script/pemc-submit.sh diagonal baseline   # any node (times NOT comparable)
```

### Check the first jobs

Do not wait for the end of the campaign. A minute after the submission:

```bash
qstat -x -u $USER | grep " F "                        # jobs already finished: after a few seconds = failed
grep -l "error" results/logs/pbs/*.err                # why (e.g. "not 6252N", "not enough slots")
```

When the first job is finished (PBS writes `.out` and `.err` only at the end):

```bash
grep "CPU check" results/logs/pbs/*.out | head -3     # "every node has 6252N"
grep "bound to" results/logs/pbs/bd-pk-1x2x4.err      # every rank with THREADS cores marked B
tail -1 results/logs/pbs/*.out                        # "done, 0 failure(s)"
grep -c "^==== run" results/logs/diagonal/*/*.log     # progress: runs written (25 datasets x REPS per job)
```

If a rank shows only one `B`, the threads share one core: the times are wrong, cancel and check `pemc-job.sh`.

Clustering runs (`baseline-sequential`, K range per dataset, input of plots C1-C6):

```bash
rm -rf results/logs/cluster results/clusters   # only before a new launch: the runs are appended, not skipped
qsub src/script/pemc-cluster.sh                # 24 cores of a Xeon 6252N node, ~21 processes in parallel (~30-60 min)
ONLY=heart KMAX=8 src/script/pemc-cluster.sh   # to test: one dataset, one process, in this shell
```

Resources are in the `#PBS` lines of the script (24 cores, 48 GB, 6 h; `place=excl` is commented out, since no 6252N node is ever completely free). It starts one process per dataset and one per K for `brainweb` (alone it takes about 2 h). Progress: `tail -n 1 results/logs/cluster/run_*.txt` (`done, 0 failure(s)` = finished); at the end `pemc-cluster.o<id>` in the root of the repository ends with `all done, 0 process(es) with failures`.

### Monitor the jobs

PBS writes `.out` and `.err` only when a job ends, but the program appends one block to the log after every finished dataset: the logs show the progress while the jobs run.

**Finished, running and queued jobs of the current campaign, and runs written per log.** Set `FIRST` to the first job ID printed by `qsub` in this campaign (older jobs are not shown):

```bash
FIRST=514542
qstat -x -u $USER | awk -v first=$FIRST '$1+0 >= first {
  s=$10; n=$4
  if (s=="F") f=f"  "n"\n"; else if (s=="R") r=r"  "n" ("$11")\n"; else q=q"  "n"\n"
} END {
  printf "== FINISHED\n%s\n== RUNNING (elapsed time)\n%s\n== QUEUED\n%s\n", f, r, q
}'
echo "== RUNS WRITTEN (75 = 25 datasets x 3 repetitions)"
for f in results/logs/diagonal/*/*.log; do
  printf "  %-50s %3d / 75\n" "$(basename $f .log | sed 's/-MPI-OpenMP-diagonal//')" "$(grep -c '^==== run' $f)"
done
```

A finished job should have 75/75 runs (the full model skips `gene_rnaseq`: 72). A job that is finished with fewer runs, or after a few seconds, failed: see its `.out` / `.err` in `results/logs/pbs/`. Job names in `qstat` are truncated (`bd-pk-1x1*`); `qstat -w` prints them in full (with different columns).

**Last runs of every log** (dataset, CPU, time per iteration). The CPU must always be the one of `NODE_FILTER` (Gold 6252N):

```bash
for f in results/logs/diagonal/*/*.log; do
  echo "== $(basename $f .log)"
  awk '/^dataset/{d=$3} /^cpu_model/{sub(/^cpu_model *: /,""); c=$0} /^t_iteration/{print "   ", d, "|", c, "| t_it =", $3}' "$f" | tail -2
done
```

**Live view**, refreshed every 30 s (exit with `Ctrl+C`):

```bash
watch -n 30 'qstat -u $USER | tail -n +6; echo; grep -c "^==== run" results/logs/diagonal/*/*.log 2>/dev/null'
```

### Output

- logs: `results/logs/<model>/<stage>/<version>_<placement>_<nodes>n<ranks>r<threads>t.log` (`<version>_seq.log` for `baseline-sequential`), with model = `diagonal` | `full` and stage = `sequential` | `baseline` | `opt1` | `opt2` (folders already in the repository); sweeps in `results/logs/sweep/`; sweep over the block size of `opt1` full (`BLOCK=64 src/script/pemc-submit.sh full opt1`; all the measurements use B = 64) in `results/logs/block/<version>_<placement>_<nodes>n<ranks>r<threads>t_b<B>.log`; clustering in `results/logs/cluster/` and `results/clusters/`;
- standard output/error of the jobs: `results/logs/pbs/<job>.out|.err`;
- STREAM (roofs of P10): `results/logs/stream/`, see below.

---

## Hardware of the nodes (report table)

One small PBS job on each of the nodes used by the runs (groups `c10`, `c11`, `c12`) reads CPU, RAM and DIMMs, NUMA, caches, network and OS.
It fills the table "Hardware Infrastructure" of the report, because the nodes are not all the same (RAM and memory channels differ); summary in [results/tables/hardware.md](../results/tables/hardware.md).

```bash
src/script/pemc-hwinfo.sh submit      # one job per node
src/script/pemc-hwinfo.sh status      # repeat until "missing 0, queued 0"
src/script/pemc-hwinfo.sh software    # compiler, MPI, UCX
src/script/pemc-hwinfo.sh summary     # same on all the nodes vs different (RAM, DIMMs, NIC, kernel)
```

Raw outputs in `results/hwinfo/` (versioned); to copy new ones from the cluster to the local machine:

```bash
scp -r username@hpc3-login1.unitn.it:ParallelExpectationMaximizationClustering/results/hwinfo results/
```

---

## Roofs of the roofline (plot P10)

One exclusive job on a Xeon 6252N node measures the memory bandwidth with STREAM; the peak GFLOP/s are calculated. Details in [script.md, 3](script.md#3-roofs-of-the-roofline).

```bash
src/script/pemc-stream.sh submit                                                  # login node: downloads stream.c, submits the job
src/script/pemc-stream.sh summary > results/logs/stream/stream_<node>-summary.txt # when it is done
```

Run in the repository: `hpc3-c12-n49`, Triad 97.0 GB/s with 8 threads, peak 588.8 GFLOP/s (AVX-512) and 73.6 GFLOP/s (SSE2): the defaults of `plots-performance.py`, so P10 needs no flag.

---

## Phase 5: Plots

When all the jobs are finished, either generate the plots on the cluster (Python 3 with `numpy`, `pandas`, `matplotlib`) or copy the logs to the local machine. From the local machine:

```bash
scp -r username@hpc3-login1.unitn.it:ParallelExpectationMaximizationClustering/results/logs results/
scp -r username@hpc3-login1.unitn.it:ParallelExpectationMaximizationClustering/results/clusters results/
```

Then, from the root of the repository:

```bash
python3 src/plots-generate/plots-performance.py     # report plots P2-P5, P8-P11 -> results/plots/performance/
python3 src/plots-generate/plots-performance.py --plots P1 P6 P7 P12   # the other ones (only in the repository)
python3 src/plots-generate/plots-cluster.py         # C1-C6 -> results/plots/cluster/ (+ results/tables/clustering-summary.csv)
python3 src/tables/clustering-summary.py            # only the table results/tables/clustering-summary.csv
```

Options and catalog of the plots: [plots.md](plots.md).

---

## Usage of the programs

The PBS jobs ([pemc-job.sh](../src/script/pemc-job.sh)) call the executables in `bin/`; for a quick test they can also be run by hand from the repository root (login node or `qsub -I -l select=1:ncpus=8:mem=4gb:cpu_type=Xeon6252N -l walltime=00:20:00`). Times on the login node are not meaningful.

```bash
# baseline-sequential
./bin/PEMC-baseline-sequential-diagonal -d data/synthetic/synth_clean/synth_clean_p01.bin -k 4

# MPI + OpenMP: R ranks x T threads, each rank bound to T cores
module load OpenMPI/4.1.5-GCC-12.3.0
OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -n 2 --map-by slot:PE=4 --bind-to core \
  bin/PEMC-opt1-MPI-OpenMP-diagonal -d data/synthetic/synth_clean/synth_clean_p01.bin -k 4 --no-log
```

Without `--map-by slot:PE=T --bind-to core` all threads of a rank share one core. `-h` lists the options.

**$K$** = number of true classes (`synth_clean` 4, `synth_overlap` 5, `synth_rare` 6, `brainweb` 12, `cardiotocography` 10, `diabetes130` 3, `gene_rnaseq` 5, others 2; `parkinson` is unlabeled). Full covariance is $O(NKD^2)$: not feasible on `gene_rnaseq`; on big files start with `-n`.

| Option | Meaning |
|---|---|
| `-d FILE` | binary dataset (header N, D + float64 row-major) |
| `-k K` | number of clusters |
| `-n N` | use only the first N rows |
| `-i N` | max iterations (200) |
| `-t X` | relative log-likelihood tolerance (1e-6) |
| `-s S` | init seed (42), same init in every version |
| `-r R` | restarts, best kept; times summed (1) |
| `-w N` | untimed warm-up iterations (3), logged as `t_warmup` |
| `-b B` | block rows for `opt1` full (64) |
| `-l FILE` / `--no-log` | log file (`results/logs/<name>.log`) / no log |
| `--save DIR` | save labels, weights, means, loglik curve (`baseline-sequential` only) |
| `-v` | print loglik every iteration |

**Log** (`.log`): each run appends a `==== run ====` block of `key : value` lines (parse by splitting on the first `:`). Main keys:

- run info: `version`, `dataset`, `hostname`, `cpu_model` (compare only runs on the same CPU), `ranks`, `threads`, `N`, `D`, `K`, `iterations`, `converged`
- quality: `loglik`, `ari` (`n/a` without labels)
- times (s): `t_total`, `t_read`, `t_init`, `t_estep`, `t_mstep`, `t_comm`, `t_warmup` (not in `t_total`), `t_iteration` = (E+M+comm)/iterations. In opt1 E and M are fused into `t_estep`
- `flops`, `bytes`: analytic counts ([perfmodel.h](../include/perfmodel.h)) for the roofline
- `rank_compute`: per-rank compute time, for load imbalance

**Correctness check:** same dataset, `-k`, `-s` → same log-likelihood (up to rounding) and same iterations. Compare only `-diagonal` with `-diagonal`, `-full` with `-full`.
