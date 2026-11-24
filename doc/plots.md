# Plots

Which plots go in the report and how they are drawn.

- Scripts: `src/plots-generate/plots-performance.py` and `src/plots-generate/plots-cluster.py` (`plots-em-steps.py`: figures of the EM steps, not in this catalog).
- Output: `results/plots/performance/` and `results/plots/cluster/`.
- Runs and commands that produce the data: [exec-cluster.md](exec-cluster.md).

## 1. Dimensions

| Dimension | Values | How it is handled |
|---|---|---|
| Model | diagonal, full | Separate panels / files |
| Version | `baseline-sequential`, `baseline-MPI-OpenMP`, `opt1`, `opt2` | Lines or bars. `baseline-sequential` is only the reference (1 core) |
| Dataset | 10 real, 3 synthetic families + D sweep (`synth_sweep_dDDD`, P1 only) | Real: clustering quality (and `brainweb` in the scaling plots). Synthetic: scaling, K, D. Only the long ones (2.1) |
| Placement | pack, scatter, pack:excl, scatter:excl | Own plot (P6), otherwise fixed: `pack` on 1 node, `scatter:excl` on 2-4 nodes (`--placement`, `--multi-placement`). PBS puts every `pack` request on ONE 96-core node (see `results/logs/pbs/*.out`), so only `scatter` gives real multi-node runs |
| Ranks x threads | 1x8 ... 8x1 (the 8 cores of one node) | Own plot (P7), then the fastest split for every core count |
| Cores / nodes | 1 ... max | x axis of the scaling plots |

## 2. Performance plots

Unless the Type says otherwise, there is one line per version.

| File (`<model>` = diagonal / full) | What it shows | Type | x | y | Fixed |
|---|---|---|---|---|---|
| `P1-sweep-<model>` | How time per iteration grows with K and D | log-log, panels (a) K (b) D, dashed reference slope | K, D | time / iteration | 1 node `pack:excl`, 1x8 (`baseline-sequential` on 1 core). K: `synth_clean_p01`, K = 2, 4, 8, 16, 32, 64; D: `synth_sweep_dDDD`, D = 5, 10, 20, 35, 50, K = 4. Logs in `results/logs/sweep/` |
| `P2-speedup-<model>` | How many times faster than `baseline-sequential` | lines + ideal | cores | speedup | 3 panels (`--grid`): `brainweb`, p08, p16 of `synth_overlap` (diagonal) / `synth_rare` (full) |
| `P3-efficiency-<model>` | How much of each added core is actually used | lines + ideal (= 1) | cores | efficiency | as P2 |
| `P4-strong-<model>` | How the time per iteration drops with more cores, same problem | lines + ideal | cores | time / iteration | N fixed, as P2 |
| `P5-weak-<model>` | Whether time stays flat when data grows with the cores | lines + ideal (= 1) | cores | weak-scaling efficiency (same version: time on 1 core / time on p cores) | `synth_overlap_pNN` on NN cores (`--family`) |
| `P6-placement-<model>` | Whether the PBS placement changes the time | grouped bars, one bar per placement | nodes (1, 2, 4) | time / iteration | most optimized version (opt2, `--version`), 1 large dataset (`--large`) |
| `P7-ranks-threads-<model>` | Which ranks x threads split is fastest on one node | bars, fastest framed in red | split | time / iteration | 1 node, opt2 (`--version`), as P6 |
| `P8-breakdown-<model>` | Where the time goes (E-step, M-step, communication, I/O) | stacked bars (100%), total time above | 1, 8, 32 cores x version | share (%) | as P2 |
| `P9-opt-gain-<model>` | How much opt1 and opt2 gain over `baseline-MPI-OpenMP` | bars opt1, opt2, `baseline-MPI-OpenMP` = 1, ratio above | dataset | time / `baseline-MPI-OpenMP` | largest core count shared by the versions (32), all long datasets (real first, dotted separator) |
| `P10-roofline-<model>` | Whether the code is compute or memory bound | roofline, one point per version | FLOP/byte | GFLOP/s (over E + M time) | as P2, 8 cores of 1 node. Analytic `flops`, `bytes` (`include/perfmodel.h`); roofs from `--peak-gflops`, `--peak-bw` (STREAM), dashed roof = peak as compiled (`--peak-gflops-compiled`, SSE2) |
| `P11-imbalance-<model>` | How unevenly the work is split between ranks | bars mean + min-max, % = max/mean - 1 | nodes / split | compute time per rank | opt2 (`--version`), as P6, key `rank_compute` (E + M per rank) |
| `P12-size-<model>` | Whether bigger datasets get more speedup | scatter, filled = real, empty = synthetic | N (log) | speedup | all long datasets (`--min-run 0`: all), panels 8 and 32 cores |

### 2.1 What goes in the report

- **Only long runs**: a dataset is drawn only if every one of its drawn runs (fastest split per core count) spends at least 1 s in $t_E + t_M + t_{comm}$ (`--min-run`, default 1.0; `0` draws all). Shorter runs are mostly noise. P5 is checked on its own (all its weak runs exceed 1 s).
- **Default `--plots`**: P2, P3, P4, P5, P8, P9, P10, P11 (in the report). P1, P6, P7, P12 have short runs or say little: `--plots P1 P6 P7 P12` draws them (repository only).

## 3. Clustering plots

The result is the same in every version (same final log-likelihood), so the plots and the table below use only the **`baseline-sequential`** version (`bin/PEMC-baseline-sequential-<model>`, 1 core).

| File | What it shows | Type | Datasets | Version |
|---|---|---|---|---|
| `C1-convergence` | Whether EM converges and which model reaches the higher log-likelihood | log-likelihood per point vs iteration, both models per panel; `X` = iteration limit | 4 (default: synth_overlap_p01, cardiotocography, heart_failure, synth_clean_p01) | `baseline-sequential` |
| `C2-choose-k` | Which number of clusters K the criteria pick, vs the true K | BIC (thick), AIC (dotted) vs K, distance from best; star = best K, dotted line = true K; `synth_clean_p01` always added as control | 2 real + 1 synthetic + control | `baseline-sequential` |
| `C3-pca` | The found clusters next to the true classes, in 2D | PCA 2D, one row per dataset, columns = true / diagonal / full, clusters recolored after their class, ARI and explained variance | 4 | `baseline-sequential` |
| `C4-centroids-<model>` | The mean value of every variable in every cluster | centroid heatmap (variables x clusters), shared scale | 1 real + 1 synthetic, D <= 30 | `baseline-sequential` |
| `C5-sizes` | Whether the found clusters have the same sizes as the true classes | bars of share per class: truth, diagonal, full | as C3 | `baseline-sequential` |
| `C6-bic-full-vs-diagonal` | Which covariance model fits each real dataset better | bars of BIC(full) - BIC(diagonal), colored by winner. † = full has lower log-likelihood (worse local optimum); ‡ = D = 1, same model | all real with both models (9: no `gene_rnaseq`) | `baseline-sequential` |

Clusters are matched one-to-one to the true classes (Hungarian algorithm), so names and colors agree across C3, C4, C5. Datasets of C1-C5: `--datasets`, in order of preference. C3 samples at most 5000 rows. `brainweb` (D = 1) has no PCA; `gene_rnaseq` has no full run (D = 20 264).

Table `results/tables/clustering-summary.csv` (numbers behind C1-C6, `baseline-sequential`): one row per dataset and model at the true K (real + `synth_clean_p01`, `synth_overlap_p01`; `parkinson` has no classes, K = 2 is arbitrary). Columns: version, N, D, K, log-likelihood, BIC, AIC, iterations, converged, ARI.

Data: `src/script/pemc-cluster.sh` runs `baseline-sequential`, both models, for K in a range (default 2..10 plus true K) into `results/logs/cluster/`; at the true K it adds `--save results/clusters` (`include/cluster-out.h`: labels, weights, means, log-likelihood per iteration). Free parameters for BIC/AIC: `(K-1) + 2KD` diagonal, `(K-1) + KD + KD(D+1)/2` full.

## 4. Conventions

- At least 3 repetitions per point (`REPS`), warm-up iterations excluded from the times; median with min-max error bars.
- Speedup is against **`baseline-sequential` of the same model**; say it in the caption.
- Axes: cores on a log2 scale; time and speedup on a log scale in the line plots (bars and ratios linear), efficiency linear.
- Size: drawn at final size, 7 in wide (single-panel P5, P6, P7, P10: 4.4 in); included at `0.82\textwidth`, P5 at `0.52\textwidth`.
- File names: `<ID>-<what>-<model>.pdf`, e.g. `P2-speedup-diagonal.pdf` (C plots with both models in one figure have no `<model>`).
- The interactive app (`src/visual-sequential`) is for exploration and to show how the algorithm works.
