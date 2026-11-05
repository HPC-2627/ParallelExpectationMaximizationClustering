# TODO

Workflow, in order. Phases overlap on purpose: datasets are prepared while the code is written, and the report is written along the way, not at the end.

## 0. Organization
- [X] Send the email with the questions (see [question.md](question.md))
- [X] Split the work between team members

## 1. Preparation
- [X] Find papers on parallel Expectation-Maximization clustering
- [X] Find papers where this was already done on a single node and on multiple nodes (to cite in the report)
- [ ] Write Introduction, sequential algorithm and state of the art in the report (`doc/latex/Group6-Report.tex`, as soon as the papers are read)
- [X] Find about 10 real datasets (as many as possible medical) and write down their sources (to cite them), saved in `data/real/`
- [X] Write the generator (new script, location to decide) and create 3 synthetic datasets saved in `data/synthetic/`
- [X] Preprocessing of the datasets: normalization, missing values, common input format

## 2. Implementation
- [X] (mike) Shared code (reading datasets, timers, logging) in `include/`
- [X] (mike) Sequential C implementation: `src/PEMC-baseline-sequential-diagonal.c` 
- [X] (mike) Define the log format (.log text file: version, dataset, ranks, threads, time per phase, iterations, log-likelihood) 
- [ ] (gabri) Baseline MPI + OpenMP: `src/PEMC-baseline-MPI-OpenMP-diagonal.c`, checked against `src/PEMC-baseline-sequential-diagonal.c` (same final log-likelihood on a small dataset, correctness reference and serial time)
- [ ] Define the PBS launch script (new file, location to decide)
- [ ] First test run on the cluster (also on 2 nodes) to find environment problems early
- [ ] Optimization 1: `src/PEMC-opt1-MPI-OpenMP-diagonal.c` (compared against `src/PEMC-baseline-MPI-OpenMP-diagonal.c`) --> to choose
- [ ] Optimization 2: `src/PEMC-opt2-MPI-OpenMP-diagonal.c` (compared against `src/PEMC-baseline-MPI-OpenMP-diagonal.c`) --> to choose
- [ ] Optimization 3: `src/PEMC-opt3-MPI-OpenMP-diagonal.c` (compared against `src/PEMC-baseline-MPI-OpenMP-diagonal.c`) --> to choose
- [ ] Build rules for all the versions in `makefile`
- [ ] Correctness test after every version, not only at the end

## 3. Experiments
Run all the versions (`src/PEMC-baseline-sequential-diagonal.c`, `src/PEMC-baseline-MPI-OpenMP-diagonal.c`, `src/PEMC-opt1-MPI-OpenMP-diagonal.c`, `src/PEMC-opt2-MPI-OpenMP-diagonal.c`, `src/PEMC-opt3-MPI-OpenMP-diagonal.c`) on the datasets in `data/real/` and `data/synthetic/`.
- [ ] Single-node runs: threads, ranks and ranks × threads at a fixed number of cores
- [ ] Multi-node runs, including PBS placement (`pack`, `scatter`, `pack:excl`, `scatter:excl`)
- [ ] Strong scaling and weak scaling
- [ ] Small / medium / large datasets
- [ ] Scaling with K and D (time per iteration), total time to convergence and number of iterations, data reading and distribution time
- [ ] Point-to-point communication benchmark (latency and bandwidth)
- [ ] Repeat each run several times (discard warm-up), keep the raw logs in `results/logs/` and the summary tables in `results/tables/`

## 4. Plots and report
- [ ] Performance plots in `results/plots-performance.py` (time, speedup, efficiency, scaling, communication breakdown, ...), output in `results/plots/`
- [ ] Clustering plots in `results/plots-cluster.py` (PCA, centroid heatmap, cluster sizes, log-likelihood, ...), output in `results/plots/`
- [ ] Parallel design, implementation and results sections of the report (`doc/latex/Group6-Report.tex`)
- [ ] Discussion and conclusions
- [ ] AI statement (mandatory)
- [ ] Repo link and data sources cited
- [ ] Report at least 10 pages

## Final checks
- [ ] Code commented, `makefile` builds from scratch, `README.md` has build and usage instructions
- [ ] Both team members ready for the oral exam

## Nice to have
- [ ] Full covariance variant: `src/PEMC-baseline-MPI-OpenMP-full.c` (after the diagonal baseline works; check it by running with 1 rank and 1 thread, compare only within the same model)
- [ ] Segmentation map visualization on a medical image dataset
- [ ] Interactive visualization for the presentation
