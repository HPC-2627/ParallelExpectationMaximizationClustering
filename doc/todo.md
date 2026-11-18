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
- [ ] Sweep over B for opt1 full (`BLOCK=16|32|64|128 CONFIGS_LIST="1:1:8" PLACES_LIST="pack:excl" src/script/pemc-submit.sh full opt1`, logs in `results/logs/block/`): not run yet, all the logs have `block : 64`
- [ ] 1-core runs (`CONFIGS_LIST="1:1:1"`) for speedup and efficiency: no `1n1r1t` log yet
- [X] Optimization 2, diagonal: `src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c` (compared against `src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c`) --> to choose
- [X] Optimization 2, full covariance: `src/full/PEMC-opt2-MPI-OpenMP-full.c`: communication between the MPI processes, same idea as the diagonal opt2 (one `MPI_Allreduce` per iteration, packed lower triangle of sxx), built on opt1 full and compared against it. Written; correctness checked locally (same iterations, same log-likelihood at every iteration as opt1 full, 1/2/4 ranks)
- [X] Run opt2 full on the cluster (16 jobs: `PLACES_LIST="pack:excl scatter:excl" src/script/pemc-submit.sh full opt2`, then `PLACES_LIST="pack scatter"`)
- [ ] Compare opt2 full with opt1 full (expected gain on the small datasets on several nodes, ~1.0x on the large ones)
- [X] Build rules for all the versions in `makefile`
- [ ] Check performances on course notes.

## 3. Experiments
Run all the versions (`src/diagonal/PEMC-baseline-sequential-diagonal.c`, `src/diagonal/PEMC-baseline-MPI-OpenMP-diagonal.c`, `src/diagonal/PEMC-opt1-MPI-OpenMP-diagonal.c`, `src/diagonal/PEMC-opt2-MPI-OpenMP-diagonal.c`) and, with the same importance but compared only among themselves (different model), the full-covariance ones (`src/full/PEMC-baseline-sequential-full.c`, `src/full/PEMC-baseline-MPI-OpenMP-full.c`, `src/full/PEMC-opt1-MPI-OpenMP-full.c`, `src/full/PEMC-opt2-MPI-OpenMP-full.c`) on the datasets in `data/real/` and `data/synthetic/`.
- [X] Single-node runs: threads, ranks and ranks × threads at a fixed number of cores
- [X] Multi-node runs, including PBS placement (`pack`, `scatter`, `pack:excl`, `scatter:excl`)
- [X] Strong scaling and weak scaling
- [X] Small / medium / large datasets
- [X] Total time to convergence and number of iterations, data reading and distribution time (in every log: `t_total`, `iterations`, `t_read`)
- [ ] Scaling with K and D (time per iteration, plot P1): `src/script/pemc-submit.sh sweep`, logs in `results/logs/sweep/`; for D first generate `synth_sweep_dDDD` with `generate-synthetic-dataset.py --sweep`
- [ ] Point-to-point communication benchmark (latency and bandwidth): no code and no logs yet
- [X] Repeat each run several times (3 repetitions, warm-up discarded), raw logs in `results/logs/`
- [ ] Summary tables in `results/tables/` (empty: generated by the plot scripts)

## 4. Plots and report
- [ ] Performance plots in `src/plots-generate/plots-performance.py` (time, speedup, efficiency, scaling, communication breakdown, ...), output in `results/plots/`
- [ ] Clustering plots in `src/plots-generate/plots-cluster.py` (PCA, centroid heatmap, cluster sizes, log-likelihood, ...), output in `results/plots/`
- [ ] Abstract and keywords of the report (empty)
- [ ] Parallel design section (Foster, communication): data distribution by blocks of rows, one `MPI_Allreduce` of a contiguous buffer per iteration, per-thread buffers with padding, results equal to the sequential up to rounding, meaning of `t_comm` (includes load imbalance)
- [X] Experimental setup section: hardware infrastructure (cluster, nodes, PBS placements); hardware of the 40 nodes collected with `src/script/pemc-hwinfo.sh`, summary in `results/tables/hardware.md`, same table in the README
- [ ] Check whether the times of the memory-bound runs depend on the node type (2, 4 or 6 memory channels populated; TODO in the report, Hardware Infrastructure)
- [ ] Implementation and performance evaluation sections of the report (`doc/latex/Group6-Report.tex`)
- [ ] Performance evaluation: compare opt1 and opt2 against the baseline (same model, same layouts and placements), and compare our results with the papers that use the same techniques (e.g. fused E/M per thread: Lee et al., arXiv:1606.02054, `doc/papers.md` #4.1; cache blocking / matrix formulation: Kwedlo 2023, #4.2; per-thread accumulators + reduction: Sanderson & Curtin, #2), discussing differences in setting (datasets, covariance type, hardware)
- [ ] Report: description of opt1 and opt2 once chosen, and cite the data sources (tables already have the links)
- [ ] Discussion and conclusions
- [X] AI statement (mandatory): written at the top of the report (code help, search of sources, mathematical and grammatical correction); check that it is accurate before delivery
- [ ] Repo link (SET TO PUBLIC) and data sources cited
- [ ] Report: insert the link to the execution on the cluster (TODO in Measurement Methodology, item "Execution on the cluster")
- [X] Report: insert the link to the visualization of the algorithm (TODO at the end of the subsection "The EM algorithm for Gaussian mixtures", paragraph "Visualization of the algorithm")
- [ ] (IMPORTANT) --> compared performance for own algorithm with different paper with the same technique or differrentc ECC..
- [X] Report at least 10 pages

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
