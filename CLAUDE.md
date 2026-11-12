# HPC4DS — Course Guide

High Performance Computing for Data Science, University of Trento (DISI), Prof. Sandro Fiore.
Reference textbook: Peter Pacheco, *An Introduction to Parallel Programming* (Morgan Kaufmann) — Ch. 3 "Distributed Memory Programming with MPI" is the backbone of lectures 4–12.

This file guides how to write exercises and the course project in this repo, following the professor's slides in `teoria/`.

## Project: Parallel Expectation-Maximization (EM) Clustering

Team project (Group 6: Michael Bernasconi, Gabriele Fronzoni): hybrid MPI + OpenMP Gaussian-mixture EM, chosen from the clustering topics of Lecture 11. The main line uses a **diagonal covariance** model; a **full covariance** variant is a nice-to-have (different model, compare only within itself). Full README: `README.md`; task list: `doc/todo.md`.

### Current state (updated 2026-11-05)

**Done**
- Study material and papers: `doc/em-explained.md`, `doc/papers.md` (PDFs in `doc/papers/`, not versioned), `doc/project-guide.md`, `doc/question.md` (questions sent to the instructors).
- Shared timing code: `include/my-time-lib.h` + `src/common/my-time-lib.c` (timer macros, mean/median/std; compile with `-Iinclude -lm`).
- Datasets (scripts in `src/dataset-managed/`, everything under `data/` is **not versioned**):
  - Synthetic: `generate-synthetic-dataset.py`, 3 profiles (`synth_clean` K=4 D=10, `synth_overlap` K=5 D=20, `synth_rare` K=6 D=50), weak-scaling files `_p01 ... _p16` with N = 250 000 * p. Details in `doc/synthetic-datasets-info.md`. **Not generated locally yet** (`data/synthetic/` is missing: run the generator).
  - Real (12 datasets, mostly medical): `generate-real-datasets.py` (download), `verify-real-datasets.py` (writes `doc/real-datasets-info.md`; `cleveland` is DISCARD), `convert-real-to-bin.py` (CSV -> `.bin`), sources in `doc/real-datasets-source.md`. Downloaded and converted locally.
  - BrainWeb (subject 44, T1 only, N = 11 862 016 voxels, D = 1): `convert-brainweb-to-bin.py` does `.nii.gz -> .nii -> .bin`. Labels (crisp, 0.5 mm) are subsampled by 2 and placed on the T1 grid at an offset found by FFT cross-correlation of the brain masks (offset (0,19,38), eta^2 = 0.951). The NIfTI affines of these files are inconsistent and must not be used. Verified: header/size, z-score inversion, gunzip identity, label counts.
- Binary format (all datasets): int32 N, int32 D, float64[N*D] row-major little-endian; labels in a separate `<name>_labels.bin` (int32 per row); metadata in `<name>.json`. Columns are z-scored.

**Not started (all `.c` files in `src/` except the timing lib are empty)**
- `PEMC-baseline-sequential-diagonal.c` (correctness reference, serial time), `PEMC-baseline-MPI-OpenMP-diagonal.c`, `PEMC-opt1/2/3-MPI-OpenMP-diagonal.c` (the 3 optimizations are still to be chosen), `PEMC-baseline-MPI-OpenMP-full.c` (1 placeholder line).
- Data loader in C (shared code), log format (CSV), PBS launch scripts, `makefile` (empty).
- Experiments (`results/logs|tables|plots` are empty), plots (`results/plots-performance.py`, `results/plots-cluster.py`), report (`doc/latex/Group6-Report.tex`), README Build/Usage/Results sections (TODO).

**Next steps (in order, see `doc/todo.md`)**: generate synthetic data; C data loader; sequential baseline; MPI+OpenMP baseline checked against it (same final log-likelihood); first cluster run on 2 nodes; choose and implement the 3 optimizations; experiments (strong/weak scaling, single and multi node, 4 PBS placements); plots; report with the AI-usage statement.

### Repo layout

- `src/` — C sources `PEMC-<version>-<model>.c`; `src/common/` shared code; `src/dataset-managed/` Python scripts for datasets.
- `include/` — headers. `data/` — datasets (not versioned). `results/` — logs, tables, plots and plot scripts.
- `doc/` — study notes, dataset docs, `latex/` report. `teoria/` (course slides, read-only) is not in this clone.
- `exercise/` rules below apply to course exercises only; the project code lives in `src/`, with its own PBS scripts.

## Language & tooling

- **C** is the default language (C++ is allowed). All slide examples use `mpi.h`.
- **MPI only** for the base exercises; **OpenMP** is added later for hybrid parallelization in the project (recommended, not mandatory). No CUDA/pthreads/SIMD/Spark/Hadoop appear in this course.
- Cluster uses **PBS** as job scheduler, module `MPICH/3.4.2-GCC-10.3.0`, launched via `mpiexec`.
- Multi-node runs require a UCX wrapper script (`mpi_wrapper.sh`) that sets `UCX_TLS=tcp,self,sm` and `UCX_NET_DEVICES`, then `exec "$@"`. Always launch through it on multi-node jobs:
  ```bash
  mpiexec -n 16 ./mpi_wrapper.sh ./ring
  ```

## Code style (follow Pacheco's conventions, as used in slides)

- Decompose into small, named functions mirroring the book: `Get_input`, `Build_mpi_type`, `Trap`, `Read_vector`, `Print_vector`, `Mat_vect_mult`, etc.
- Comment every function parameter's direction: `/* in */`, `/* out */`, `/* in/out */`.
- Heavily comment code — the project report explicitly requires "properly commented" C.
- Prefer collectives (`MPI_Bcast`, `MPI_Reduce`, `MPI_Allreduce`, `MPI_Scatter`, `MPI_Gather`, `MPI_Allgather`) over manual `MPI_Send`/`MPI_Recv` loops once collectives have been introduced — manual loops are the "naive first version" the course deliberately improves on.

## Canonical running examples from lectures (use as templates)

1. **Ring token-passing** (open and closed ring) — basic `MPI_Send`/`MPI_Recv`, rank-based addressing with `(rank + 1) % size`.
2. **Trapezoidal rule integration** — serial → naive parallel (manual send/recv to rank 0) → tree-structured global sum → `MPI_Reduce` → `MPI_Bcast` for input distribution → derived MPI datatype (`MPI_Type_create_struct`) to broadcast `(a, b, n)` in one call.
3. **Vector sum / matrix-vector multiplication** — `MPI_Scatter` rows of `A`, `MPI_Allgather` vector `x`, local dot-product loop, `MPI_Gather` result `y`.
4. **Point-to-point bandwidth benchmark** — send doubling message sizes (1 byte to 1MB) between 2 processes, time with `MPI_Wtime`, compute MB/sec, plot vs. size on log-scale X axis. Explicitly flagged as relevant groundwork for the project's performance analysis.

## Timing & benchmarking rules

- Use `MPI_Wtime()` (preferred) or `gettimeofday()`.
- Always `MPI_Barrier` before starting the timer to avoid skew between ranks.
- Pattern for worst-case timing across ranks:
  ```c
  MPI_Barrier(comm);
  start = MPI_Wtime();
  /* code under test */
  finish = MPI_Wtime();
  local_elapsed = finish - start;
  MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  ```
  (`MPI_MAX` reduction gives the true wall-clock bound across processes.)
- Every benchmark must report: how timing was taken, max message/problem size tested, the bandwidth/throughput formula used, a statistical approach for accurate estimates (repeat runs, not a single sample), which PBS placement strategy was used, and an execution-time model `f(n)` as a function of problem size.

## PBS placement strategies (test all of these for any benchmark/project run)

```
#PBS -l select=2:ncpus=1:mem=2gb -l place=pack          # same node
#PBS -l select=2:ncpus=1:mem=2gb -l place=scatter       # different nodes
#PBS -l select=2:ncpus=1:mem=2gb -l place=pack:excl     # same node, exclusive
#PBS -l select=2:ncpus=1:mem=2gb -l place=scatter:excl  # different nodes, exclusive
```
Job script skeleton:
```bash
#!/bin/bash
#PBS -l select=4:ncpus=4:mem=2gb
#PBS -l walltime=0:00:30
#PBS -q shortCPUQ
module load MPICH/3.4.2-GCC-10.3.0
mpiexec -n 16 ./mpi_wrapper.sh ./<binary>
```

## Course project

### Scope
A parallel application using **MPI and OpenMP** (hybrid), developed **in a team**, chosen from the professor's fixed topic list (numerical algorithms, metaheuristics, sketches/streaming algorithms, factorizations, clustering, puzzle solvers, etc. — see Lecture 11 slides for the exact list). Discussed at an oral exam where **all team members must be present**.

### Mandatory report structure
1. **Introduction** — describe the assigned problem; analyze the sequential algorithm (pseudocode, examples, figures).
2. **Parallel design** — identify parallelism opportunities; survey state-of-the-art parallel strategies with citations; discuss alternative designs and why they help/hurt communication/sync cost; discuss hybrid (data-dependent) strategies too.
3. **Implementation** — commented C (or C++) code; may implement multiple strategies and compare pros/cons; include critical code snippets; **must link the repo**; MPI+OpenMP hybrid recommended.
4. **Performance and scalability analysis** — execution time, speedup, efficiency; **strong and weak scalability** where applicable; **single-node and multi-node benchmarks**.

### Evaluation criteria
Clarity of presentation; depth/correctness/originality of theoretical analysis; technical implementation quality and documentation; number/quality of parallel strategies compared; critical thinking in performance evaluation; **a good project must show significant speedup**. Report ≥10 pages. **An AI-usage statement is mandatory in every report.**

### Practical tips (from slides)
- Cite papers/sources that inspired the approach and the data sources used.
- If no suitable dataset exists, write a synthetic random data generator; test at small/medium/large scale and compare.
- If building on existing work, either: reproduce it and compare results/discuss differences; implement from pseudocode independently and compare against the original authors' findings; extend existing code with a clear delta; or port and compare two different implementations.

## When writing exercises or project code in this repo

- Exercises (if any) go under `exercise/`; project code under `src/`. Keep each program's `.c` and matching PBS `.sh` launcher together, named consistently.
- Always provide a PBS submission script alongside any new MPI program.
- Python dataset scripts: heavy comments, `/* in */`-style docs on helpers, idempotent, deterministic (seed 42), run from the repo root with a venv (`.venv/` is git-ignored).
- When asked to benchmark something, default to testing all four placement strategies and reporting MB/sec or time vs. problem size, log-scaled, matching the Lecture 12 P2P assignment format.
- Favor the book's function decomposition and in/out comment style over ad hoc code, even for throwaway exercises — it's what the professor's slides and the project rubric reward.
