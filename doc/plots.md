# Plots: guidelines

Rules to decide which plots go in the report and how to keep them readable. Scripts: `src/plots-generate/plots-performance.py` and `src/plots-generate/plots-cluster.py`, output in `results/plots/performance/` and `results/plots/cluster/`.

## 0. Golden rules

1. **One plot answers one question.** The question is written as the plot title in this file (see the catalog).
2. **One x axis, at most 4 lines.** The lines are the versions (sequential, baseline, opt1, opt2). If you need more lines, you need more plots.
3. **Everything else is fixed and declared in the caption**: model, dataset, placement, number of repetitions.
4. **The full grid goes in tables, not in plots.** All combinations live in `results/tables/` (CSV) and in the repo. The report shows only the plots below.
5. **Diagonal and full are never on the same performance plot** (different model, different cost). Use two panels, (a) diagonal and (b) full, same axes.
6. **Same style everywhere**: same color for the same version in every plot, same marker, same font, vector output (PDF).

## 1. The dimensions and how each one is handled

| Dimension | Values | How to handle it |
|---|---|---|
| Model | diagonal, full | Two panels (a)/(b) of the same figure |
| Version | sequential, baseline, opt1, opt2 | Lines (or bars) inside a plot. Sequential is only the reference (1 core) |
| Dataset | 10 real, 3 synthetic | Real: clustering quality and time to convergence. Synthetic: scaling, K, D (we control them). Per plot pick 1 small, 1 medium, 1 large, never all |
| Placement | pack, scatter, pack:excl, scatter:excl | **Experimental parameter, not a plot dimension by default.** One dedicated plot (P6). The scaling plots use `pack` on 1 node and `scatter:excl` on 2-4 nodes (`--placement`, `--multi-placement`): PBS puts every `pack` request on ONE 96-core node (checked in `results/logs/pbs/*.out`), so only `scatter` gives real multi-node runs. Only if P6 shows a visible effect, the key plots (P2, P4) are also drawn as a 2x2 grid, one panel per placement (see 1.1) |
| Ranks x threads | e.g. 1x16 ... 16x1 | One dedicated heatmap (P7), then fixed to the best split |
| Cores / nodes | 1 ... max | The x axis of the scaling plots |

### Placement procedure (the part that causes the mess)

1. Run only the placement experiment: 2 and 4 nodes, 1 large dataset, best version, 1 model, the 4 placements.
2. Plot P6 (grouped bars). This is the result "does placement matter?".
3. Pick the best placement. **All the other performance plots use only that one**, declared in every caption.
4. Only if the best placement changes with the number of nodes, add a speedup plot with one line per placement, **for the best version only**.

Same logic for ranks x threads: find the best split once (P7), then fix it.

### 1.1 Why placement is not simply split into 4 panels in every plot

`excl` means *exclusive* (the nodes are used only by our job), it is not a separate dimension of its own: it is the second factor of a 2x2 design (`pack` / `scatter` x shared / `excl`).

The alternative to the procedure above is to draw **every** plot as a 2x2 grid (small multiples), one panel per placement, with the version lines inside each panel:

|  | shared | excl |
|---|---|---|
| **pack** | `pack` | `pack:excl` |
| **scatter** | `scatter` | `scatter:excl` |

This is valid and respects the rule "one x axis, at most 4 lines" (the placement becomes the panel dimension, not a line). Reading it: along a row you see the effect of `excl`, along a column the effect of `pack` vs `scatter`. The axes must be shared (same x and y range in all the panels), otherwise the comparison is false.

Why it is NOT the default:
- **Number of panels.** With the two models (diagonal and full are never mixed) it is one 2x2 grid per model: 8 panels per plot. For 10 plots that is 80 panels. The report has limited space and nobody reads 80 panels.
- **Redundancy.** If placement has a small effect, the 4 panels are almost identical and occupy the space for nothing.
- **Cost of the experiments.** Every plot would need all the runs repeated for the 4 placements (4x the cluster time), instead of one placement fixed after P6.

What we do instead (compromise):
1. P6 shows the placement alone and answers "does it matter?". If the effect is negligible, the other plots use one placement and we are done.
2. If the effect is visible, the 2x2 grid is used only for the plots where it matters most: speedup (P2) and strong scaling (P4). The other plots keep the best placement.
3. The scripts can generate the 2x2 version of all the plots anyway; those go in the appendix or in the repo, not in the main text.

## 2. Performance plots (catalog)

| ID | Question | Type | x | y | Lines / groups | Fixed |
|---|---|---|---|---|---|---|
| P1 | How does time per iteration scale with K and D? | log-log lines, two panels (a) K (b) D, reference slope dashed | K, D | time / iteration | versions | one layout (1 node, 1 rank x 8 threads). K sweep: `synth_clean_p01` with K = 2..64; D sweep: `synth_sweep_dDDD`, K = 4. Data in `results/logs/sweep/` |
| P2 | How much faster than sequential? (speedup) | lines + ideal | cores | speedup | versions | one row of 3 panels (`--grid`): the real `brainweb` and two synthetic sizes, p08 and p16, of the family with the most work (`synth_overlap` for diagonal, `synth_rare` for full); placement |
| P3 | How efficient? | lines | cores | efficiency | versions | same grid as P2, own file `P3-efficiency-<model>` |
| P4 | Strong scaling | lines + ideal | cores | time (or speedup) | versions | N fixed; same grid as P2 |
| P5 | Weak scaling | lines | cores | efficiency = time of the SAME version on 1 core / time on p cores | versions | N / core fixed (`synth_overlap_pNN` on NN cores, `--family`); reference = each version on 1 core, not the sequential |
| P6 | Does PBS placement matter? | grouped bars | nodes (1, 2, 4) | time / iteration | 4 placements | most optimized version (opt2), 1 model, 1 large dataset |
| P7 | Best ranks x threads at fixed cores? | bars (one per split, fastest framed in red) | ranks x threads | time / iteration | none | 1 node, most optimized version (opt2) (with 8 cores only 4 splits exist: a ranks x threads table would be mostly empty) |
| P8 | Where does the time go? | stacked bars, each bar = 100% of one run, total time written above it | 1, 8, 32 cores x version | share of the run (%) | E-step, M-step, `t_comm`, I/O | same grid as P2 |
| P9 | What do opt1 and opt2 gain over the baseline? | bars normalized (**parallel baseline MPI-OpenMP = 1**, not the sequential), ratio written above every bar | dataset | time / baseline | opt1, opt2 | one panel per model, 32 cores; ALL the long datasets, real first, a dotted line before the synthetic ones |
| P10 | Is it compute or memory bound? | roofline | FLOP/byte | GFLOP/s | one point per version | model per file, 1 node. FLOP and bytes are an analytic count (`include/perfmodel.h`, keys `flops`, `bytes` of the log); roofs from `--peak-gflops` and `--peak-bw` (measure the node with STREAM) |
| P11 | Load imbalance between ranks | bars mean + min-max whisker, % = max/mean - 1 | nodes / ranks x threads | compute time per rank | none | most optimized version (opt2). Key `rank_compute` of the log (E-step + M-step of every rank). `t_comm` includes the imbalance, here it is separated |
| P12 | Does the speedup depend on the size of the dataset? | scatter, filled = real, empty = synthetic | rows N (log) | speedup over sequential | versions | ALL the datasets, two panels: 8 cores (1 node) and 32 cores (4 nodes) |

Clarity of every figure (P and C): bold title in plain words plus one line with model / dataset / placement, ONE legend above the panels with names in plain words, axes with their unit, drawn 7 in wide (4.4 in for the single-panel P5, P6, P7, P10); the report includes them at `width=0.82\textwidth`, P5 at `0.52\textwidth`, the clustering plots C1, C2, C6 at `0.85\textwidth`. Axis limits never cut a point (an efficiency above 1 stays visible). In the performance figures the real datasets come first (left), every panel title says "real" or "synthetic"; the x ticks of the scaling plots read `8` over `1n` (8 cores, 1 node). In the clustering figures (C1-C5) the real datasets are on top, the synthetic ones below, each block with its own header.

### 2.1 Which plots and which datasets go in the report

- **Only long runs.** A dataset is drawn only if EVERY one of its runs (all versions, all core counts) spends at least 1 s in the measured iterations, $t_E + t_M + t_{comm}$ (`--min-run`, default 1.0; `--min-run 0` draws all). Below that the spread of the repetitions is several percent and the comparison is mostly noise. The figures do not write the threshold: the report states it. With the current logs the long datasets are: diagonal `brainweb`, `synth_overlap_p08`, `synth_overlap_p16`, `synth_rare_p16`; full `brainweb`, `synth_overlap_p04/p08/p16`, `synth_rare_p02/p04/p08/p16`. All the other datasets stay in `results/logs/`.
- **Why so many synthetic datasets.** The public medical datasets we found are small (thousands of rows, runs of milliseconds); `brainweb` is the only real one long enough. The synthetic ones give runs of seconds to minutes.
- **Plots of the report** (default of `--plots`): P2, P3, P4, P5, P8, P9, P10, P11. P1, P6, P7 and P12 are not in the report: their runs are short (milliseconds) or say little; the script can still draw them (`--plots P1 P6 P7 P12`).
- P5 is checked on its own: its weak runs ($p$ times the rows on $p$ cores) are all longer than 1 s, even if the small sizes p01/p02 are not "long" as datasets.

## 3. Clustering plots (catalog)

The clustering result is the same in every version (we check the final log-likelihood), so these plots use **one version only**, the baseline.

| ID | Question | Type | Datasets | File |
|---|---|---|---|---|
| C1 | Does EM converge? Which model fits better? | log-likelihood per point vs iteration, one panel per dataset, diagonal and full in the same panel; `X` = stopped at the iteration limit | 4 (default: synth_overlap_p01, cardiotocography, heart_failure, synth_clean_p01) | `C1-convergence` |
| C2 | Which K is right? | BIC (thick) and AIC (dotted) vs K, both models, distance from the best value; star = best K of each curve, dotted line = true K. `synth_clean_p01` (pure Gaussian) is always added as the control case | 3 + control | `C2-choose-k` |
| C3 | Do the clusters look right? | PCA 2D, rows = true classes / found by diagonal / found by full, found clusters recolored after their true class (optimal one-to-one assignment), ARI and variance kept by the PCA in every column | 4 | `C3-pca` |
| C4 | What does each cluster look like? | heatmap of the centroids (variables x clusters), one shared color scale; clusters named after their true class, as in C3 and C5 | 2 with D <= 30 | `C4-centroids-<model>` |
| C5 | Are the clusters balanced? | bars of the share of points per class: truth, diagonal, full | same as C3 | `C5-sizes` |
| C6 | Diagonal or full, which fits better? | bars of BIC(full) - BIC(diagonal), colored by the model that wins. Hatched (†): log-likelihood of full < diagonal, i.e. a worse local optimum, not a model comparison. Gray (‡): D = 1, the two models are the same | all real | `C6-bic-full-vs-diagonal` |

Every figure has a bold title in plain words, ONE legend above the panels (never on the data), and the dataset name on every panel. The found clusters are paired one-to-one with the true classes (Hungarian algorithm on the overlap table), so a cluster has the same name and color in C3, C4 and C5. Datasets of C1-C5: `--datasets`, in order of preference.

Table instead of plot: the 10 real datasets with N, D, K, log-likelihood, BIC, iterations, for both models. It is written in `results/tables/clustering-summary.csv` by `src/tables/clustering-summary.py` (also by `plots-cluster.py`, same content).

How the data is produced: `src/script/pemc-cluster.sh` runs the sequential baseline (both models) on the datasets for every K of a range (default 2..10, plus the true K) and appends to `results/logs/cluster/`. Only at the true K it also passes `--save results/clusters` (`include/cluster-out.h`): labels of every row, weights and means, log-likelihood after every iteration. BIC and AIC are computed by the script from the log (`loglik`, N, D, K, model), with `p = (K-1) + K*D + K*D` free parameters for diagonal and `(K-1) + K*D + K*D*(D+1)/2` for full.

Notes: C3 draws a random sample of at most 5000 rows (PCA on the sample), and a second row colored by the true classes when the dataset has them. `brainweb` has D = 1, so it has no PCA. `gene_rnaseq` has no full covariance run (D = 20 264).

## 4. Experiment matrix (what to run on the cluster for each plot)

| Plots | Runs needed |
|---|---|
| P6 | 4 placements x {2, 4 nodes} x 1 dataset x 1 version x 1 model (x repetitions) |
| P7 | all ranks x threads splits on 1 node x 1 dataset x most optimized version (opt2) |
| P2, P3, P4, P8, P9 | 4 versions x 2 models x cores {1, 2, 4, ..., max} x 3 datasets, with the fixed placement and split |
| P5 | 4 versions x 2 models x cores {...} with N proportional to cores |
| P1 | `src/script/pemc-submit.sh sweep` (K sweep on `synth_clean_p01`, D sweep on the files made by `generate-synthetic-dataset.py --sweep`), versions, one fixed layout. Logs go in `results/logs/sweep/` |
| P10, P11 | no extra runs: `flops`, `bytes` and `rank_compute` are in every log written by the new binaries (rebuild and re-run the scaling jobs) |
| C1-C6 | `src/script/pemc-cluster.sh`: sequential baseline, diagonal and full, K range per dataset (C2, C6, table), `--save` at the true K (C1, C3, C4, C5). Logs in `results/logs/cluster/` |

Run P6 and P7 **first**: they fix the placement and the split for all the rest, so the other runs are not repeated.

## 5. Practical conventions

- Repetitions: at least 3 per point (discard warm-up), plot the median, error bars min-max.
- Speedup is always computed against the **sequential version of the same model**, not against the parallel one on 1 core. Say it in the caption.
- Log-log axes for time vs cores, linear for speedup and efficiency.
- A caption always states: model, dataset, placement, ranks x threads, repetitions.
- File names: `<ID>-<what>-<model>.pdf`, e.g. `P2-speedup-diagonal.pdf`.
- The interactive app (`src/visual-sequential`) is for exploration and the oral exam. The report uses only the static plots produced by the two scripts, so every figure can be regenerated.
