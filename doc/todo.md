# TODO

Workflow, in order. Phases overlap on purpose: datasets are prepared while the code is written, and the report is written along the way, not at the end.

## 0. Organization
- [X] Send the email with the questions
- [X] Split the work between team members

## 1. Preparation
- [X] Find papers on parallel Expectation-Maximization clustering
- [X] Find papers where this was already done on a single node and on multiple nodes (to cite in the report)
- [X] Write Introduction and sequential algorithm in the report (two models, diagonal and full with Cholesky, 4 versions each, Algorithm 1 for both) and the Dataset section (`doc/latex/Group6-Report.tex`)
- [ ] Write the state of the art in the report (as soon as the papers are read)
- [X] Find about 10 real datasets (as many as possible medical) and write down their sources (to cite them), saved in `data/real/`
- [X] Write the generator (new script, location to decide) and create 3 synthetic datasets saved in `data/synthetic/`
- [X] Preprocessing of the datasets: normalization, missing values, common input format

## 2. Implementation
- [X] (mike) Shared code (reading datasets, timers, logging) in `include/`
- [X] (mike) Sequential C implementation: `src/diagonal/PEMC-baseline-sequential-diagonal.c` 
- [X] (mike) Define the log format (.log text file: version, dataset, ranks, threads, time per phase, iterations, log-likelihood) 
- [X] (mike) Define visual method to visualize `src/diagonal/PEMC-baseline-sequential-diagonal.c` 
- [X] (gabri) Baseline MPI + OpenMP: `src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c`, checked against `src/diagonal/PEMC-baseline-sequential-diagonal.c` (same final log-likelihood on a small dataset, correctness reference and serial time)
- [X] (mike) Baseline MPI + OpenMP, full covariance (equal in importance to the diagonal model; 4 versions per model: sequential, baseline, opt1, opt2): `src/full/PEMC-baseline-MPI-OpenMP-full.c` (variant of the diagonal baseline, checked against `src/full/PEMC-baseline-sequential-full.c`)
- [X] Define the PBS launch script (new file, location to decide)
- [X] First test run on the cluster (also on 2 nodes) to find environment problems early
- [X] Optimization 1, diagonal: `src/diagonal/PEMC-opt1-MPI-OpenMP-diagonal.c` (compared against `src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c`) --> to choose
- [X] Optimization 1, full covariance: `src/full/PEMC-opt1-MPI-OpenMP-full.c`: all the optimizations of the computation (fused E+M pass as in the diagonal opt1, plus blocks of B rows, option `-b`, default 64), compared against `src/full/PEMC-baseline-MPI-OpenMP-full.c`. Written; correctness checked locally (same iterations and ll)
- [X] Run opt1 full again on the cluster (16 jobs: `src/script/pemc-submit.sh full opt1`, first remove the old logs in `results/logs/full/opt1/` on the cluster)
- [X] Sweeps of K and D also for opt1/opt2 (plot P1: for which K and D the optimizations pay off; e.g. the gain of the blocks of opt1 full vs D): `SWEEP_VERSIONS="opt1-MPI-OpenMP-diagonal opt2-MPI-OpenMP-diagonal opt1-MPI-OpenMP-full opt2-MPI-OpenMP-full" src/script/pemc-submit.sh sweep` (8 jobs, `pack:excl`; commands in `doc/exec-cluster.md`, sweeps). Logs in `results/logs/sweep/`, P1 picks them up
- [X] (IMPORTANT) Runs on 1, 2 and 4 cores of one node (`CONFIGS_LIST="1:1:1 1:1:2 1:2:1 1:1:4 1:2:2 1:4:1"`, baseline, opt1, opt2, both models, **only `pack`**: 36 jobs; commands in `doc/exec-cluster.md`, "Runs on 1, 2 and 4 cores of one node"). Batch B (`pack:excl`) cancelled (queued for hours, the plots use `pack`): not needed. Status: 33/36 finished, the 3 full 1-core jobs still running; then check the run counts (75 diagonal, 72 full), copy the logs and regenerate the plots. Write in the report that these layouts are measured only with `pack`. Today P2, P4, P5 jump from 1 to 8 cores, and 8 -> 16 -> 32 cores also changes 1 -> 2 -> 4 nodes. They give: (1) the scaling inside one node, separated from the network; (2) opt1 on 1 core = the gain of the optimization alone, so the efficiency above 1 of opt1 full (P2, P5) can be split into optimization x parallelism; (3) more points for the weak scaling (`synth_clean_p02`, `p04` already exist)
- [X] Optimization 2, diagonal: `src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c` (compared against `src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c`) --> to choose
- [X] Optimization 2, full covariance: `src/full/PEMC-opt2-MPI-OpenMP-full.c`: communication between the MPI processes, same idea as the diagonal opt2 (one `MPI_Allreduce` per iteration, packed lower triangle of sxx), built on the full baseline and compared against it (2026-11-22: rebuilt from the baseline, before it was built on opt1 full). Correctness checked locally (same iterations, same log-likelihood at every iteration and same ARI as the baseline full, 1/2/4 ranks, converged and max_iter runs)
- [ ] Re-run opt2 full on the cluster: the logs in `results/logs/full/opt2/` (and the opt2 full sweep/block logs) come from the OLD version built on opt1 full and are stale (16 jobs: `PLACES_LIST="pack:excl scatter:excl" src/script/pemc-submit.sh full opt2`, then `PLACES_LIST="pack scatter"`)
- [ ] Compare opt2 full with the baseline full (expected gain on the small datasets on several nodes, ~1.0x on the large ones): plot P9 full, 32 cores. (Old results, opt2 built on opt1: synthetic opt2 = opt1; real: opt2 faster on cardiovascular, eeg_eye_state, heart_failure, parkinson)
- [X] Build rules for all the versions in `makefile`
- [ ] Check performances on course notes.

## 3. Experiments
Run all the versions (`src/diagonal/PEMC-baseline-sequential-diagonal.c`, `src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c`, `src/diagonal/PEMC-opt1-MPI-OpenMP-diagonal.c`, `src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c`) and, with the same importance but compared only among themselves (different model), the full-covariance ones (`src/full/PEMC-baseline-sequential-full.c`, `src/full/PEMC-baseline-MPI-OpenMP-full.c`, `src/full/PEMC-opt1-MPI-OpenMP-full.c`, `src/full/PEMC-opt2-MPI-OpenMP-full.c`) on the datasets in `data/real/` and `data/synthetic/`.
- [X] Single-node runs: threads, ranks and ranks × threads at a fixed number of cores
- [X] Multi-node runs, including PBS placement (`pack`, `scatter`, `pack:excl`, `scatter:excl`)
- [X] Strong scaling and weak scaling
- [X] Small / medium / large datasets
- [X] Total time to convergence and number of iterations, data reading and distribution time (in every log: `t_total`, `iterations`, `t_read`)
- [X] Scaling with K and D (time per iteration, plot P1): `src/script/pemc-submit.sh sweep`, logs in `results/logs/sweep/`; for D first generate `synth_sweep_dDDD` with `generate-synthetic-dataset.py --sweep`
- [X] Roofs of the roofline (plot P10): STREAM on a Xeon 6252N node (`src/script/pemc-stream.sh`, logs in `results/logs/stream/`): Triad 97.0 GB/s with 8 threads; peak 588.8 GFLOP/s (AVX-512) and 73.6 GFLOP/s (SSE2, as compiled), written as defaults in `plots-performance.py`
- [X] Clustering runs for C1-C6 (`qsub src/script/pemc-cluster.sh`): 209 runs (diagonal 109, full 100), no failure, logs in `results/logs/cluster/`, saved clusters in `results/clusters/`
- [X] Repeat each run several times (3 repetitions, warm-up discarded), raw logs in `results/logs/`
- [ ] Summary tables in `results/tables/`: `clustering-summary.csv` done (`src/tables/clustering-summary.py`) and `hardware.md`; no performance table yet

## 4. Plots and report
- [X] Performance plots in `src/plots-generate/plots-performance.py` (time, speedup, efficiency, scaling, communication breakdown, ...), output in `results/plots/performance/`
- [X] Clustering plots in `src/plots-generate/plots-cluster.py` (PCA, centroid heatmap, cluster sizes, log-likelihood, ...), output in `results/plots/cluster/`
- [ ] Abstract 
- [X] keywords of the report
- [X] Parallel design section (Foster, communication): data distribution by blocks of rows, one `MPI_Allreduce` of a contiguous buffer per iteration, per-thread buffers with padding, results equal to the sequential up to rounding, meaning of `t_comm` (includes load imbalance)
- [ ] (important) --> CHECK COSTO COMPUTAZIONE VS COMUNICAZIONE E COME LO HO MISURATO ETC....
- [ ] (IMPORTANT) --> example table slide 23 Lecture 19 --> Data dependence analysis (Lecture 19, "Data dependences and OpenMP"): new subsection of "Parallel Design Methodology" in the report, no code change expected. For every parallel loop of every version (baseline, opt1, opt2, diagonal and full) apply the 3 steps of the lecture:
  - detection: every variable written in the loop (scalars and arrays), including the functions called inside (`log_sum_exp`, `exp`, `log`: no `static` or global state)
  - classification: table as in slide 23 (memory location, earlier/later statement, loop-carried or not, flow / anti / output)
  - removal: how each one is handled (`reduction(+:ll)`, array reduction `reduction(+:acc[:len])` + `MPI_Allreduce`, per-thread slice of `logp`/`z` = privatization, `reduction(max:bad)`, `simd reduction` of the full opt versions)
  - non-removable dependences: the outer EM loop (iteration t+1 needs the parameters of t: recurrence, so the parallelism is inside one iteration) and the forward substitution of the full model (`z_a` needs `z_0..z_{a-1}`; opt1 goes around it with 8 independent rows in SIMD, as slide 34)
  - re-check opt1 and opt2 line by line; change the code only if a dependence turns out to be unhandled (data race)
  - [ ] tables of slides 23-24 for the 6 MPI+OpenMP versions: `results/tables/data-dependences/<version>.xlsx` (sheets Definitions, Loops, Dependences), generated by `src/tables/data-dependences.py` (line numbers found in the sources, rerun after changing the code; needs openpyxl, installed in `.venv`). No unhandled dependence found: no code change needed
  - [ ] write the subsection of the report from the tables (one summary table, not all the rows)
  - "Data Dependence Analysis" is already in the index terms of the report (IEEEkeywords): if this subsection is not written, remove it from there
- [X] Experimental setup section: hardware infrastructure (cluster, nodes, PBS placements); hardware of the 40 nodes collected with `src/script/pemc-hwinfo.sh`, summary in `results/tables/hardware.md`, same table in the README
- [ ] Check whether the times of the memory-bound runs depend on the node type (2, 4 or 6 memory channels populated; TODO in the report, Hardware Infrastructure)
- [ ] Implementation and performance evaluation sections of the report (`doc/latex/Group6-Report.tex`)
- [ ] Performance evaluation: compare opt1 and opt2 of both models against the baseline of their model (same model, same layouts and placements), and compare our results with the papers that use the same techniques (e.g. fused E/M per thread: Lee et al., arXiv:1606.02054, `doc/papers_and_references.md` #4.1; cache blocking / matrix formulation: Kwedlo 2023, #4.2; per-thread accumulators + reduction: Sanderson & Curtin, #2), discussing differences in setting (datasets, covariance type, hardware)
- [X] Report: description of opt1 and opt2 once chosen, and cite the data sources (tables already have the links)
- [ ] Discussion and conclusions
- [X] AI statement (mandatory): written at the top of the report (code help, search of sources, mathematical and grammatical correction); check that it is accurate before delivery
- [ ] Repo link (SET TO PUBLIC) and data sources cited
- [X] Report: insert the link to the execution on the cluster (TODO in Measurement Methodology, item "Execution on the cluster")
- [ ] Report: before delivery, check that every index term (IEEEkeywords) has its own part in the report; remove the ones that do not
- [X] Report: insert the link to the visualization of the algorithm (TODO at the end of the subsection "The EM algorithm for Gaussian mixtures", paragraph "Visualization of the algorithm")
- [ ] (IMPORTANT) --> compared performance for own algorithm with different paper with the same technique or differrentc ECC..
- [X] Report at least 10 pages
- [ ] IMPORTANT --> TODO: check if all link, references, papers and content correspondign with implementation and  C code

## Final checks
- [ ] Code audit (`src/`, `src/common/`, `include/`, `src/visual-sequential/`), version by version, before delivery:
  - every function, struct and non-obvious line is commented (what it does and why), in English
  - everything is used: no unused functions, parameters, options, macros, includes or files (for example the statistics in `my-time-lib.c` and `dataset_read_header()` if no version ends up calling them); delete what is not used
  - no hand-made replacements of standard functions (e.g. a custom random generator instead of `rand()`); use the standard library
  - no leftover portability or "just in case" code that the cluster does not need (we already removed the custom RNG, the architecture/endianness check on the datasets and `_POSIX_C_SOURCE`; look for similar cases: unneeded `#define`s, defensive checks, duplicated helpers)
  - no behaviour hidden in the code that is not in the README (options, defaults, file formats)
  - `make` from scratch with no warnings (`-Wall -Wextra`), and the visualization still matches the C binary after any change
- [ ] Code commented, `makefile` builds from scratch, `README.md` has build and usage instructions
- [ ] Both team members ready for the oral exam

## Nice to have
- [ ] Segmentation map visualization on a medical image dataset
- [ ] Interactive visualization for the presentation
- [X] Report: subsection "Validation of the clustering" (end of Experimental Setup & Dataset, `\label{sec:validation}`, TODOs in red): insert ALL the plots C1-C6 of `results/plots/cluster/`, one or two sentences each, at most about one page; do not spend much time on it (HPC project: it only shows that the fast code gives a correct clustering)
  - C1 convergence: iterations change with dataset and model, so the metric is t_it
  - C2 choose K: the control synth_clean_p01 finds the true K = 4
  - C3 PCA: clusters match the visible groups
  - C4 centroids (diagonal and full): clusters differ on a few variables
  - C5 sizes: close to the true classes on the synthetic data
  - C6 full vs diagonal: where the extra cost of full gives a better model († local optimum, ‡ D = 1)
  - limits (degenerate clusters, BIC minimum at K = 10, † and ‡) go in the Discussion
