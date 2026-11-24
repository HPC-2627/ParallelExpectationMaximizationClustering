# TODO

Workflow, in order. Phases overlap on purpose: datasets are prepared while the code is written, and the report is written along the way, not at the end.

## 0. Organization
- [X] Send the email with the questions
- [X] Split the work between team members

## 1. Preparation
- [X] Find papers on parallel Expectation-Maximization clustering
- [X] Find papers where this was already done on a single node and on multiple nodes (to cite in the report)
- [X] Write Introduction and sequential algorithm in the report (two models, diagonal and full with Cholesky, 4 versions each, Algorithm 1 for both) and the Dataset section (`doc/latex/Group6-Report.tex`)
- [x] Write the state of the art in the report (as soon as the papers are read)
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
- [X] Re-run opt2 full on the cluster (new version built on the baseline): 22 jobs, every file with 72 runs, logs in `results/logs/full/opt2/` (the 6 jobs on 1, 2, 4 cores finished 2026-11-23)
- [X] Re-run the K and D sweeps of opt2 full (P1 full has no opt2 curve: the old sweep was of the opt2 built on opt1), so that P1 full has the same 4 curves as P1 diagonal. On the cluster: DONE 2026-11-24: 33 runs, same iterations and log-likelihood as the baseline at every point, P1 regenerated with 4 curves in both models
  - [X] move the old log first (the sweep logs are APPENDED): `mv results/logs/sweep/opt2-MPI-OpenMP-full_pack_excl_1n1r8t.log old-logs/`
  - [X] `SWEEP_VERSIONS="opt2-MPI-OpenMP-full" src/script/pemc-submit.sh sweep` (2 jobs `o2f-swK`, `o2f-swD`, `pack:excl`, 1 node 1 rank x 8 threads, as the other sweeps; dry run checked 2026-11-23). If still queued after about one day: `qdel` and relaunch with `SWEEP_PLACE=pack` (then write "opt2 full on a shared node" in the report)
  - [X] follow the jobs: `qstat -u $USER`; runs done: `grep -c '==== run' results/logs/sweep/opt2-MPI-OpenMP-full_pack_excl_1n1r8t.log` (goal 33 = 18 K sweep on `synth_clean_p01` + 15 D sweep on `synth_sweep_d*`); reason of a queued job: `qstat -f $(qselect -u $USER -N o2f-swK) | grep -i comment` (same with `o2f-swD`)
  - [X] when both jobs are gone from `qstat` and the count is 33, copy the log to the laptop: `scp michael.bernasconi@hpc3-login1.unitn.it:ParallelExpectationMaximizationClustering/results/logs/sweep/opt2-MPI-OpenMP-full_pack_excl_1n1r8t.log results/logs/sweep/`
  - [X] regenerate P1 (`python3 src/plots-generate/plots-performance.py --plots P1 --min-run 0`) and remove "Optimization 2 full not re-run" from the report (Performance Evaluation, "Figures not included")
- [X] Compare opt2 full with the baseline full (2026-11-23): P9 full and report, Section "Communication vs Computation Breakdown" (opt2 removes the collective cost that is not waiting, 6.5 -> 0.5 ms/iteration on 32 cores, but that is <= 2% of an iteration; the time is decided by the wait for the slowest rank)
- [X] Build rules for all the versions in `makefile`
- [X] Check performances on course notes.

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
- [X] Summary tables in `results/tables/`: `clustering-summary.csv` done (`src/tables/clustering-summary.py`) and `hardware.md`; no performance table yet

## 4. Plots and report
- [X] Performance plots in `src/plots-generate/plots-performance.py` (time, speedup, efficiency, scaling, communication breakdown, ...), output in `results/plots/performance/`
- [X] Clustering plots in `src/plots-generate/plots-cluster.py` (PCA, centroid heatmap, cluster sizes, log-likelihood, ...), output in `results/plots/cluster/`
- [X] Abstract: final version (2026-11-23), numbers checked against Performance Evaluation, Discussion and Conclusion
- [X] keywords of the report
- [X] Correct opt2 full everywhere it appeared with the old version (2026-11-23, checked against the code): report (list of the versions, subsection and pseudocode of Optimization 2 full, Measurement Methodology "What is measured"), `README.md`, `doc/papers_and_references.md`, `doc/exec-cluster.md`, comments of `args.h`, `logger.h`, `args.c`, `pemc-job.sh`, `pemc-submit.sh`. Still to do with the new data: correctness check, numbers, plots, Discussion (items below)
- [X] Opt2 references checked against the code (2026-11-23): Thakur et al. (#T) and BYU (#4.3a/#4.3b) for both opt2, both built on the baseline; no paper of opt1 (#4.1, #4.2, #4.4) is cited any more for opt2 full
- [X] PLOTS: only datasets whose runs all last at least 1 s (redone 2026-11-23 after the loss of 2026-11-22)
  - [X] `plots-performance.py`: option `--min-run` (default 1 s: drop the datasets with a run, any version and core count, shorter than that), grid P2/P3/P4/P8/P10 as ONE row (brainweb + synth_overlap_p08/p16 diagonal, brainweb + synth_rare_p08/p16 full), P9 in one panel (real first, dotted separator), short node labels (`1n`, `2n`, `4n`), P5 on `synth_overlap` (weak runs all > 6 s), P1/P6/P7/P12 out of the default list, no "1 s" text inside the figures; update `doc/plots.md`
  - [X] report: subsection "Datasets Shown and Duration of the Runs" at the start of Performance Evaluation (why synthetic: the public medical datasets are small; only runs >= 1 s to avoid showing noise, with the measured spread of the repetitions; the other datasets measured with the same procedure, logs in the repository with links, `--min-run 0` draws them all)
  - [X] report: every figure right below the paragraph that describes it (environment `figblock`: text + figure never split across pages), "Logs:" link to `results/logs/<model>/` in every caption
  - [X] sweep log of the OLD opt2 full: already removed from `results/logs/sweep/`
- [X] IMPORTANT :  example table slide 23 Lecture 19 --> Data dependence analysis (Lecture 19, "Data dependences and OpenMP"): new subsection of "Parallel Design Methodology" in the report, no code change expected. For every parallel loop of every version (baseline, opt1, opt2, diagonal and full) apply the 3 steps of the lecture: -- DONE 2026-11-23: subsection "Data Dependence Analysis" (end of Parallel Design Methodology), definitions + one table per MPI+OpenMP file in the format of slide 25, with a Removal column; line numbers written by hand from the sources (update them if the code changes). No unhandled dependence, no code change
  - [X] detection: every variable written in the loop (scalars and arrays), including the functions called inside (`log_sum_exp`, `exp`, `log`: no `static` or global state)
  - [X] classification: table as in slide 23 (memory location, earlier/later statement, loop-carried or not, flow / anti / output)
  - [X] removal: how each one is handled (`reduction(+:ll)`, array reduction `reduction(+:acc[:len])` + `MPI_Allreduce`, per-thread slice of `logp`/`z` = privatization, `reduction(max:bad)`, `simd reduction` of opt1 full)
  - [X] non-removable dependences: the outer EM loop (iteration t+1 needs the parameters of t: recurrence, so the parallelism is inside one iteration) and the forward substitution of the full model (`z_a` needs `z_0..z_{a-1}`; opt1 goes around it with 8 independent rows in SIMD, as slide 34)
  - [X] re-check opt1 and opt2 line by line; change the code only if a dependence turns out to be unhandled (data race)
  - [X] tables of slides 23-25 for the 6 MPI+OpenMP versions: written directly in the report (Tables "Data dependences of <file>"); the old plan with `results/tables/data-dependences/*.xlsx` and `src/tables/data-dependences.py` was not kept (those files are not in the repository)
  - [X] tables completed (2026-11-23): added the rows of `r` (E-step -> M-step, baseline and opt2 of both models), `R` and `Zt` reused as `RX` (opt1 full) and the Cholesky recurrence (opt1 full, lines of `src/common/gmm.c`); new paragraph "How to read the tables" and the four removal cases (privatization, reduction, end of a loop, not removable) explained in the text
  - [X] write the subsection of the report from the tables (one summary table, not all the rows)
  - [X] "Data Dependence Analysis" is in the index terms of the report and the subsection is written: keep it
- [X] Parallel design section (Foster, communication): data distribution by blocks of rows, one `MPI_Allreduce` of a contiguous buffer per iteration, per-thread buffers with padding, results equal to the sequential up to rounding, meaning of `t_comm` (includes load imbalance)
- [X] (important) Computation vs communication and how it is measured (2026-11-23): report, paragraph "What t_comm contains": t_comm split into the wait for the slowest rank (slowest minus fastest `rank_compute`) and the real cost of the collectives, per version, on 16 and 32 cores
- [X] Experimental setup section: hardware infrastructure (cluster, nodes, PBS placements); hardware of the 40 nodes collected with `src/script/pemc-hwinfo.sh`, summary in `results/tables/hardware.md`, same table in the README
- [X] Node type (2, 4 or 6 memory channels) vs times: MEASURED a posteriori (2026-11-23) with `results/hwinfo/` (copied from the cluster) + the node of every rank (lines `MCW rank` of the PBS `.err`) + `rank_compute`: `src/tables/node-memory.py` -> `results/tables/node-memory.md`. Result: the channels do NOT change the times (ranks on 4- and 6-channel nodes within 1%; same node slow for the baseline but not for opt1); the imbalance depends on the code (baseline/opt2 ranks 0.68-1.46 of the mean, opt1 0.94-1.03). Report updated (Hardware Infrastructure, Discussion on Optimization 2 and on the imbalance, Limits, Conclusion), README links the table.
- [X] Implementation and performance evaluation sections of the report (`doc/latex/Group6-Report.tex`), rewritten 2026-11-23
- [X] Performance evaluation: compare opt1 and opt2 against the baseline (P9, Discussion) and with the papers (2026-11-24): Discussion, subsection "Comparison with Related Work" (Sanderson & Curtin, Lee 2016, Lee 2020, Kwedlo with numbers measured in the same conditions; Machlica, BYU, Yin, Srivastava not compared numerically, with the reason). "3-8x faster than the baseline" qualified as 32 cores, with the one-node value 2.3-2.5x (abstract, P9 full, Conclusion, README)
- [X] Report: description of opt1 and opt2 once chosen, and cite the data sources (tables already have the links)
- [X] After the 6 opt2 full jobs: plots regenerated, Performance Evaluation and Discussion rewritten (2026-11-23), correctness check recomputed (9,849 runs, max relative difference 4.9e-9 diagonal / 7.7e-9 full), expected gain of opt2 connected to the figures (Discussion, "Optimization 2: What the Saved Communication Is Worth")
- [X] Conclusion (2026-11-23), with future work. Performance Evaluation also lists, with links, the figures not in the report (P1, P6, P7, P12, regenerated with `--min-run 0`)
- [X] Discussion (2026-11-23): when the parallelization pays off, Optimization 1 (computation is the bottleneck), Optimization 2 (what the saved communication is worth), rank/thread split and load imbalance, limits of the measurements
- [X] Report, Performance Evaluation (2026-11-23): tables "Datasets shown" and "Communication per iteration" removed, their content written in the text (list of the long datasets; wait / collectives / imbalance per version)
- [X] Report, Measurement Methodology (2026-11-23): correctness check recomputed with all the logs (5,025 diagonal + 4,824 full runs, max 4.9e-9 / 7.7e-9), "Stability" with the measured spread per placement (red TODO removed)
- [X] Plots P1, P6, P7, P12 regenerated with the current logs (`--plots P1 P6 P7 P12 --min-run 0`), linked in the report as figures not included
- [X] Old PBS outputs of the opt2 full built on opt1 (`results/logs/pbs/o2f-*`) replaced by the new ones, nested folder `results/logs/pbs/pbs/` removed
- [X] AI statement (mandatory): written at the top of the report (code help, search of sources, mathematical and grammatical correction); check that it is accurate before delivery
- [X] Repo link (SET TO PUBLIC) and data sources cited
- [X] Report: insert the link to the execution on the cluster (TODO in Measurement Methodology, item "Execution on the cluster")
- [X] Report: two parts still in red: "Choice of B=64" (Optimization 1 full: text is correct, make it black or shorten it) and the placement note in Measurement Methodology (verified, make it black). The node-type TODO is done
- [X] Commit and push the work of 2026-11-23: report (PDF + tex), `src/tables/node-memory.py`, `results/tables/node-memory.md`, `README.md`, `doc/todo.md` (and later the opt2 full sweep log + P1)
- [X] Report: before delivery, check that every index term (IEEEkeywords) has its own part in the report; remove the ones that do not (2026-11-24: all 13 have a part, none removed; "row-block partitioning" and "message aggregation" now named in the text)
- [X] Report: insert the link to the visualization of the algorithm (TODO at the end of the subsection "The EM algorithm for Gaussian mixtures", paragraph "Visualization of the algorithm")
- [X] (IMPORTANT) --> compare the performance of our algorithm with the papers that use the same techniques (same item as "Performance evaluation" above: done 2026-11-24)
- [X] Report at least 10 pages
- [X] IMPORTANT --> TODO: check if all link, references, papers and content correspondign with implementation and  C code (2026-11-24: algorithms, buffer sizes and the line numbers of the 6 dependence tables match the C code; all citations, refs, figures and repository links resolve; paper sections and numbers match the PDFs; run counts and correctness maxima recomputed from the logs; fixed: nodes used 40 -> 49, one quotation)

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
