# Scripts in `src/script/`

Bash scripts for the HPC3 cluster (PBS queue `shortCPUQ`), in two groups: the ones that **run the experiments** and the ones that **collect the hardware of the nodes**. Run them from the **root of the repository**; the full procedure is in [exec-cluster.md](exec-cluster.md).

- [1. Experiments](#1-experiments): `pemc-submit.sh`, `pemc-job.sh`, `mpi_wrapper.sh`, `pemc-cluster.sh`
- [2. Cluster hardware](#2-cluster-hardware): `pemc-hwinfo.sh`, `pemc-hwprobe.sh`
- [3. Roofs of the roofline](#3-roofs-of-the-roofline): `pemc-stream.sh`

---

## 1. Experiments

Performance runs (sequential, baseline, opt1, opt2, sweeps) and clustering runs. They need the binaries: `make all mpi-omp` first.

| Script | What it does | Run by |
|---|---|---|
| [`pemc-submit.sh`](../src/script/pemc-submit.sh) | submits the PBS jobs: one job per version × layout | you |
| [`pemc-job.sh`](../src/script/pemc-job.sh) | body of one job: runs one version on all the datasets with one layout | `qsub`, called by `pemc-submit.sh` |
| [`mpi_wrapper.sh`](../src/script/mpi_wrapper.sh) | forces UCX (OpenMPI transport) onto the cluster Ethernet interface, then starts the binary | `mpiexec`, on every rank |
| [`pemc-cluster.sh`](../src/script/pemc-cluster.sh) | sequential runs over a range of K for the clustering plots C1-C6, in parallel on 24 cores of one node | you, with `qsub` |

Chain of a parallel run:

```
pemc-submit.sh ──qsub──> pemc-job.sh ──mpiexec──> mpi_wrapper.sh ──exec──> bin/PEMC-<version>
```

### `pemc-submit.sh`

```bash
src/script/pemc-submit.sh <diagonal|full> [sequential|baseline|opt1|opt2]   # one model, one stage (or all written stages)
src/script/pemc-submit.sh [sequential|parallel|all]                        # baselines of both models
src/script/pemc-submit.sh sweep                                            # K and D sweeps for plot P1
```

- **Layouts** (`CONFIGS`, `NODES:RANKS_PER_NODE:THREADS`): `1:1:8 1:2:4 1:4:2 1:8:1` (one node, 8 cores) and `2:2:4 4:2:4` (16 and 32 cores).
- **Placements** (`PLACES`): `pack`, `scatter`, `pack:excl`, `scatter:excl`. With 1 node only `pack` and `pack:excl` are submitted. The sequential version always runs on 1 core with `pack:excl`.
- **Same CPU for every job:** `NODE_FILTER=cpu_type=Xeon6252N` in the `select`, so the times can be compared.
- **Only written versions** (`WRITTEN` list): a new version must be added there and to `MPI_BINS` in the makefile.
- PBS output (`.out`/`.err`) goes to `results/logs/pbs/`.

| Variable | Effect |
|---|---|
| `DRY_RUN=1` | only prints the `qsub` commands |
| `PLACES_LIST="pack scatter"` | only these placements |
| `CONFIGS_LIST="2:2:4 4:2:4"` | only these layouts |
| `REPS=1` | repetitions of every (dataset, layout), default 3 |
| `SKIP=4` | does not submit the first 4 jobs (resume after a "per-user limit" error) |
| `BLOCK=64` | rows per block of `opt1` full (sweep over B, logs in `results/logs/block/`) |
| `SWEEP_VERSIONS="..."` | with `sweep`: only these versions (default all 8; the logs are appended, so give only the new ones) |
| `SWEEP_PLACE=pack` | with `sweep`: placement of the sweep jobs (default `pack:excl`) |
| `NODE_FILTER=` / `REQUIRE_CPU=` | empty = any node / no CPU check |

### `pemc-job.sh`

Not submitted by hand: it receives the layout from `qsub` and the variables with `-v` (`VERSION`, `NODES`, `RANKS_PER_NODE`, `THREADS`, `PLACE`, `REPS`, `SETS`, `SYNTH_P`, `ONLY`, `SWEEP`, `BLOCK`, `REQUIRE_CPU`).

1. **CPU check:** every node must have `REQUIRE_CPU` (default `6252N`), otherwise the job stops **before** writing any log.
2. **Datasets:** all the real ones + `synth_*_p01..p16`, each with K = number of true classes. With `SWEEP=K` / `SWEEP=D` the list is replaced by the sweep. `gene_rnaseq` is skipped for the full model (does not fit in memory).
3. **Launch:** MPI versions with `mpiexec --hostfile $PBS_NODEFILE --map-by slot:PE=THREADS --bind-to core` through `mpi_wrapper.sh`; `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
4. **Logs:** `results/logs/<model>/<stage>/<version>_<place>_<N>n<R>r<T>t.log` (sequential: `_seq`); sweeps in `results/logs/sweep/`, block sweeps in `results/logs/block/`.

### `mpi_wrapper.sh`

Finds the network interface that reaches `192.168.108.1`, sets `UCX_TLS=tcp,self,sm` and `UCX_NET_DEVICES=<iface>`, prints the choice on stderr (in the `.err` file) and replaces itself with the program (`exec "$@"`). Without it multi-node runs fail.

### `pemc-cluster.sh`

```bash
qsub src/script/pemc-cluster.sh            # everything, in parallel on 24 cores of one node (~30-60 min)
ONLY=heart src/script/pemc-cluster.sh      # only the datasets whose name contains "heart", one process
```

Runs `PEMC-baseline-sequential-{diagonal,full}` for every K in `KMIN..KMAX` (default 2..10, plus the true K if larger), without warm-up.

- **PBS** (lines `#PBS` of the script): 24 cores of one node with `cpu_type=Xeon6252N`, 48 GB, 6 h. The times are not used by C1-C6, so the node can be shared. `place=excl` (whole node) is commented out: no 6252N node is ever completely free, and with it the job stays queued.
- **Parallel:** without `ONLY` and `KS` it starts one process per dataset, and one per K for `brainweb` (`SPLIT_K`: alone it takes about 2 h), about 21 processes, then waits for all of them. Output of each process: `results/logs/cluster/run_<dataset>[_k<K>].txt`, ending with `done, 0 failure(s)`. `PARALLEL=0` runs everything in one process.
- `results/logs/cluster/baseline-sequential-<model>.log`: one block per run (BIC/AIC vs K, C2, C6).
- `results/clusters/`: labels, means and log-likelihood curve, saved **only** for the true K (C1, C3, C4, C5).

Variables: `KMIN`, `KMAX`, `KS` (explicit list of K, replaces the range), `ONLY`, `SETS` (`real | synthetic | all`), `SPLIT_K`, `PARALLEL`.

The script does not skip runs already in the logs: before a new launch remove `results/logs/cluster/` and `results/clusters/`, otherwise the blocks are duplicated.

---

## 2. Cluster hardware

Hardware of the 40 nodes used by the experiments, for the table "Hardware Infrastructure" of the report and the table in the README. They need no binaries and no root. Summary of the results: [`results/tables/hardware.md`](../results/tables/hardware.md).

| Script | What it does | Run by |
|---|---|---|
| [`pemc-hwinfo.sh`](../src/script/pemc-hwinfo.sh) | one small job per node, status of the jobs, software versions, summary | you, on the login node |
| [`pemc-hwprobe.sh`](../src/script/pemc-hwprobe.sh) | body of one of those jobs: prints the hardware of the node where it runs | `qsub`, called by `pemc-hwinfo.sh` |

Chain:

```
pemc-hwinfo.sh submit ──qsub (host=<node>)──> pemc-hwprobe.sh ──> results/hwinfo/<node>.txt
pemc-hwinfo.sh summary ──reads──> results/hwinfo/*.txt   (summary written by hand in results/tables/hardware.md)
```

### `pemc-hwinfo.sh`

Collects the hardware of all the nodes used by the experiments, for the table "Hardware Infrastructure" of the report. Run it on the login node, from the root of the repository:

```bash
src/script/pemc-hwinfo.sh submit      # one job per node still without data (waits and retries if the queue is full)
src/script/pemc-hwinfo.sh status      # done / queued (with the reason) / missing, per node
src/script/pemc-hwinfo.sh software    # GCC, Open MPI, UCX -> results/hwinfo/software.txt
src/script/pemc-hwinfo.sh summary     # same on all the nodes vs different: RAM, DIMMs per socket, NIC, kernel
```

- **Nodes:** the list `NODES` in the script: the 40 nodes of all the runs (1 in `c10`, 18 in `c11`, 21 in `c12`), taken from the PBS outputs of both team members; `NODES_FILE=<file>` uses another list.
- **Output:** raw files in `results/hwinfo/` (not versioned); the summary used by the report is [`results/tables/hardware.md`](../results/tables/hardware.md).
- **Jobs:** `hw-<node>`, 1 core, 1 GB, 2 minutes, `-l select=...:host=<node>`, body `pemc-hwprobe.sh`; output in `results/hwinfo/<node>.txt`. A node is done when its file contains the line `MemTotal`.
- **Re-runs:** `submit` skips the nodes already done or still in the queue, so it can be run again at any time; when the per-user limit of the queue is reached it waits 30 s and tries again.
- **Long waits:** a job stays in the queue (`Insufficient amount of resource: host`) while other users' jobs fill its node; PBS gives no estimate (`qstat -T`). Leave it there or remove it with `qdel`.

### `pemc-hwprobe.sh`

Body of the jobs of `pemc-hwinfo.sh`; it can also be run by hand on one node. It prints one `key: value` line per item, without root:

- OS and kernel;
- CPU model, sockets, cores per socket, threads per core (hyper-threading), NUMA nodes, L1/L2/L3 caches;
- total RAM and RAM per NUMA node;
- every DIMM (label `PROC_<socket>_DIMM_<slot>` on `c10`/`c11`, `CPU_SrcID#<socket>_MC#_Chan#_DIMM#` on `c12`; size, type) from the EDAC driver, and their speed only if `dmidecode` is allowed (otherwise `not allowed`);
- frequency governor, driver and turbo (empty if Linux exposes no frequency driver);
- network interfaces that are up (speed, MTU, card model) and InfiniBand if present.

---

## 3. Roofs of the roofline

The two roofs of plot P10: the memory bandwidth of the node, **measured** with [STREAM](https://www.cs.virginia.edu/stream/), and the peak GFLOP/s, **calculated**. No EM run is needed: the points of P10 come from the `flops` and `bytes` keys of the logs that are already there.

### `pemc-stream.sh`

```bash
src/script/pemc-stream.sh submit     # downloads stream.c into bin/stream/ (login node), submits 1 job
src/script/pemc-stream.sh summary    # Triad GB/s and peak GFLOP/s per thread count, flags for P10
```

- **Job:** `stream`, a whole Xeon 6252N node (`ncpus=96`, `place=pack:excl`, so no other job uses the memory bandwidth), 20 minutes. The CPU is checked as in `pemc-job.sh`. STREAM is compiled on the node (GCC 12.3.0, `-O3 -march=native -fopenmp`), with arrays of 200M doubles (1.6 GB each, much larger than the 4 × 35.75 MB of L3).
- **Runs:** `OMP_PLACES=cores`, T = 1 2 4 8 16 24 48 96 threads with `OMP_PROC_BIND=close` (as in the EM runs), plus T = 8 and 96 with `spread` (threads spread over the 4 sockets).
- **Output:** `results/logs/stream/stream_<node>.txt`, the raw STREAM output (one block per T); PBS output in `results/logs/pbs/stream.{out,err}`.
- **Peak GFLOP/s** = T × 2.3 GHz (base clock) × FLOP/cycle per core, in double precision: 4 with SSE2 (what the binaries use: `-O3` without `-march`), 16 with AVX2+FMA, 32 with AVX-512 FMA (the machine peak, 2 FMA ports). It is an upper bound: with AVX-512 the clock drops below 2.3 GHz.
- **For P10:** the 1-node points use 8 cores, so `summary` prints the roofs for T = 8 (`close`), ready for `plots-performance.py --plots P10 --peak-bw <GB/s> --peak-gflops <GFLOP/s>`.

| Variable | Effect |
|---|---|
| `ARRAY_SIZE` | elements per array, default 200000000 |
| `NTIMES` | repetitions per kernel (STREAM keeps the best), default 20 |
| `THREADS_LIST` | thread counts, default `1 2 4 8 16 24 48 96` |
| `GHZ` | clock for the peak, default 2.3 |
| `NODE_FILTER=` / `REQUIRE_CPU=` | empty = any node / no CPU check |
