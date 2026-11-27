# Scripts in `src/script/`

Bash scripts for the HPC3 cluster (PBS queue `shortCPUQ`), run from the **root of the repository**. 

Full procedure in [exec-cluster.md](exec-cluster.md).

Plots (P1-P12, C1-C6) in [plots.md](plots.md).

Terms:

- **Version:** one program `bin/PEMC-<version>`: `baseline-sequential`, `baseline-MPI-OpenMP`, `opt1-MPI-OpenMP` (optimizes the computation), `opt2-MPI-OpenMP` (optimizes the communication), each for the `diagonal` and `full` model (8 in all). **Stage** = the version without the model: `sequential`, `baseline`, `opt1`, `opt2`.
- **Layout** `N:R:T`: N nodes, R MPI ranks per node, T OpenMP threads per rank (`2:2:4` = 16 cores).
- **Placement** (PBS `-l place=`): `pack` = as few nodes as possible (every `pack` job ended up on **one** 96-core node), `scatter` = one node per chunk (the only real multi-node runs), `:excl` = nodes reserved for the job.

| Script | What it does | Run by |
|---|---|---|
| [`pemc-submit.sh`](../src/script/pemc-submit.sh) | submits the performance jobs: one per version × layout × placement | you |
| [`pemc-job.sh`](../src/script/pemc-job.sh) | body of one job: one version, one layout, all the datasets | `qsub` |
| [`mpi_wrapper.sh`](../src/script/mpi_wrapper.sh) | forces UCX onto the cluster Ethernet interface, then starts the binary | `mpiexec`, every rank |
| [`pemc-cluster.sh`](../src/script/pemc-cluster.sh) | `baseline-sequential` over a range of K, for C1-C6 | you, with `qsub` |
| [`pemc-hwinfo.sh`](../src/script/pemc-hwinfo.sh) | hardware of the nodes: one job per node, status, summary | you, login node |
| [`pemc-hwprobe.sh`](../src/script/pemc-hwprobe.sh) | body of one of those jobs: prints the hardware of its node | `qsub` |
| [`pemc-stream.sh`](../src/script/pemc-stream.sh) | STREAM bandwidth and peak GFLOP/s: the roofs of P10 | you |

```
pemc-submit.sh ──qsub──> pemc-job.sh ──mpiexec──> mpi_wrapper.sh ──exec──> bin/PEMC-<version>
pemc-hwinfo.sh ──qsub (host=<node>)──> pemc-hwprobe.sh ──> results/hwinfo/<node>.txt
```

The experiments need the binaries: `make all mpi-omp` (`pemc-cluster.sh` only `make`).

---

## 1. Experiments

### `pemc-submit.sh`

```bash
src/script/pemc-submit.sh <diagonal|full> [sequential|baseline|opt1|opt2]   # one model, one stage (none = all)
src/script/pemc-submit.sh [sequential|parallel|all]                        # baselines of both models
src/script/pemc-submit.sh sweep                                            # K and D sweeps (P1)
```

- **Layouts** (`CONFIGS`): `1:1:8 1:2:4 1:4:2 1:8:1` (8 cores, 1 node), `2:2:4 4:2:4` (16 and 32 cores). The runs on 1, 2 and 4 cores and the other splits on 2 and 4 nodes (`2:8:1 2:4:2 4:8:1 4:4:2`) were submitted with `CONFIGS_LIST`/`PLACES_LIST` ([exec-cluster.md](exec-cluster.md)). One rank per node on several nodes (`2:1:8`, `4:1:8`) fails: Open MPI refuses the binding, every run is `FAILED`.
- **Placements** (`PLACES`): `pack`, `scatter`, `pack:excl`, `scatter:excl` (on 1 node only `pack` and `pack:excl`). The plots use only exclusive nodes: `pack:excl` for 1-8 cores (1 node) and `scatter:excl` for 16 and 32 cores (2 and 4 nodes); `pack` and `scatter` only in P6. `baseline-sequential`: 1 core, `pack:excl`.
- **Same CPU everywhere:** `NODE_FILTER=cpu_type=Xeon6252N` in the `select`; `pemc-job.sh`.
- **Sweeps** (`sweep`, 2 jobs per version): fixed layout `1:1:8`, `pack:excl` (`baseline-sequential` on 1 core). K sweep: `synth_clean_p01`, K = 2, 4, 8, 16, 32, 64. D sweep: `synth_sweep_d005`…`d050` (D = 5, 10, 20, 35, 50), K = 4, generated first with `generate-synthetic-dataset.py --sweep`. Logs in `results/logs/sweep/`, **appended**.
- **New versions** must be added to `WRITTEN` here and to `MPI_BINS` in the makefile.
- **PBS output:** `results/logs/pbs/<v>-<p>-<N>x<R>x<T>.{out,err}`, e.g. `bd-pkx-2x2x4` = `baseline-MPI-OpenMP-diagonal`, `pack:excl`, `2:2:4`. Versions `sd sf bd bf o1d o1f o2d o2f` (`d` diagonal, `f` full), placements `pk sc pkx scx`; `baseline-sequential` jobs are `sd-seq`/`sf-seq`, sweep jobs `<v>-swK`/`-swD`. In the repository only the jobs on 8-32 cores, `sd-seq`/`sf-seq` and `pemc-cluster.o551214`.

| Variable | Effect |
|---|---|
| `DRY_RUN=1` | only prints the `qsub` commands |
| `PLACES_LIST` / `CONFIGS_LIST` | only these placements / layouts |
| `REPS` | repetitions of every (dataset, layout), default 3 |
| `SKIP=4` | skips the first 4 jobs (resume after a "per-user limit" error) |
| `BLOCK=64` | rows per block of `opt1` full, passed as `-b` (logs in `results/logs/block/`). All the measurements use B = 64 |
| `SWEEP_VERSIONS` / `SWEEP_PLACE` | `sweep`: only these versions (default all 8) / placement (default `pack:excl`) |
| `NODE_FILTER=` / `REQUIRE_CPU=` | empty = any node / no CPU check |

### `pemc-job.sh`

Not run by hand: `qsub -v` passes layout and options.

1. **CPU check:** every node must have `REQUIRE_CPU` (`6252N`), else it stops before writing any log.
2. **Datasets:** all the real ones + `synth_*_p01…p16` (5 sizes), K = true classes; `gene_rnaseq` skipped for the full model (memory).
3. **Launch:** `mpiexec --hostfile $PBS_NODEFILE --map-by slot:PE=T --bind-to core` through `mpi_wrapper.sh`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`.
4. **Logs:** `results/logs/<model>/<stage>/<version>_<place>_<N>n<R>r<T>t.log` (`:` → `_`, e.g. `opt1-MPI-OpenMP-diagonal_pack_excl_1n2r4t.log`); `baseline-sequential`: `<version>_seq.log`.

### `mpi_wrapper.sh`

UCX (the Open MPI transport) may pick an interface that does not reach the other nodes, and multi-node runs fail. The wrapper finds the interface that reaches `192.168.108.1` (internal network), sets `UCX_TLS=tcp,self,sm` and `UCX_NET_DEVICES=<iface>`, prints it in the `.err` file and runs the program (`exec "$@"`).

### `pemc-cluster.sh`

```bash
qsub src/script/pemc-cluster.sh            # everything, 24 cores of one node (30-60 min)
ONLY=heart src/script/pemc-cluster.sh      # only the datasets whose name contains "heart"
```

- Runs `baseline-sequential` (both models) for K = `KMIN..KMAX` (2..10, plus the true K if larger), one process per dataset (per K for `brainweb`, `SPLIT_K`), about 21 in parallel.
- 24 cores, 48 GB, 6 h, node shared (times are not used by C1-C6). Output `pemc-cluster.o<jobid>` in the folder of the `qsub`: move it to `results/logs/pbs/` by hand.
- Output: `results/logs/cluster/baseline-sequential-<model>.log` (BIC/AIC vs K: C2, C6), `results/clusters/` (labels, means, log-likelihood, only for the true K: C1, C3-C5), `results/logs/cluster/run_*.txt` (one per process).
- Variables: `KMIN`, `KMAX`, `KS` (list of K), `ONLY`, `SETS`, `SPLIT_K`, `PARALLEL=0` (one process).
- Runs are appended: remove `results/logs/cluster/` and `results/clusters/` before a new launch.

---

## 2. Cluster hardware

For the "Hardware Infrastructure" table of the report and the README; summary in [`results/tables/hardware.md`](../results/tables/hardware.md). No binaries, no root.

```bash
src/script/pemc-hwinfo.sh submit      # one job per node without data (retries if the queue is full)
src/script/pemc-hwinfo.sh status      # done / queued (with the reason) / missing, per node
src/script/pemc-hwinfo.sh software    # GCC, Open MPI, UCX -> results/hwinfo/software.txt
src/script/pemc-hwinfo.sh summary     # what is the same on all the nodes and what differs
```

- **Nodes:** [`results/hwinfo/nodes.txt`](../results/hwinfo/nodes.txt), the 40 nodes of all the runs of both team members (recreated from `NODES` in the script if missing; `NODES_FILE=<file>` for another list).
- **Jobs:** `hw-<node>`, 1 core, 2 min, `host=<node>`; output `results/hwinfo/<node>.txt`, done when it contains `MemTotal`. `submit` skips nodes done or queued, so it can be rerun at any time.
- **`pemc-hwprobe.sh`** prints `key: value` lines: OS, kernel, CPU, sockets, cores, NUMA, caches, RAM, every DIMM (EDAC driver), frequency driver, network interfaces. 31 of the 40 files come from an earlier version without `avx512f`, RAM per NUMA node and frequency lines; all have what the report uses.

---

## 3. Roofs of the roofline

```bash
src/script/pemc-stream.sh submit     # downloads stream.c (login node), submits 1 job
src/script/pemc-stream.sh summary    # Triad GB/s and peak GFLOP/s per thread count, flags for P10
```

- **Bandwidth, measured:** STREAM on a whole exclusive Xeon 6252N node (`-O3 -march=native -fopenmp`, arrays of 1.6 GB ≫ L3), T = 1, 2, 4, 8, 16, 24, 48, 96 threads `close`, plus 8 and 96 `spread`. Output `results/logs/stream/stream_<node>.txt`.
- **Peak, calculated:** T × 2.3 GHz × FLOP/cycle: 4 with SSE2 (the binaries, `-O3` without `-march`), 32 with AVX-512 FMA (machine peak, an upper bound).
- **Run in the repository:** `hpc3-c12-n49` ([raw](../results/logs/stream/stream_hpc3-c12-n49.txt), [summary](../results/logs/stream/stream_hpc3-c12-n49-summary.txt)). With T = 8 (the 1-node points of P10): Triad 97.0 GB/s, peak 588.8 (AVX-512) and 73.6 (SSE2) GFLOP/s, the defaults of `plots-performance.py`, so P10 needs no flag.
- Variables: `ARRAY_SIZE` (200000000), `NTIMES` (20), `THREADS_LIST`, `GHZ` (2.3), `NODE_FILTER=` / `REQUIRE_CPU=`.
