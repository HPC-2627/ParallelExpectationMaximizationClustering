# Plots

Which plots go in the report and how they are drawn. 

Scripts: `src/plots-generate/plots-performance.py` and `src/plots-generate/plots-cluster.py`

Output in `results/plots/performance/` and `results/plots/cluster/`.

## 1. Dimensions

| Dimension | Values | How it is handled |
|---|---|---|
| Model | diagonal, full | Separate panels / files |
| Version | `baseline-sequential`, `baseline-MPI-OpenMP`, opt1, opt2 | Lines or bars. `baseline-sequential` is only the reference (1 core) |
| Dataset | 10 real, 3 synthetic families | Real: clustering quality. Synthetic: scaling, K, D. Only the long ones (2.1) |
| Placement | pack, scatter, pack:excl, scatter:excl | Own plot (P6), otherwise fixed: `pack` on 1 node, `scatter:excl` on 2-4 nodes (`--placement`, `--multi-placement`). PBS puts every `pack` request on ONE 96-core node (see `results/logs/pbs/*.out`), so only `scatter` gives real multi-node runs |
| Ranks x threads | e.g. 1x16 ... 16x1 | Own plot (P7), then fixed to the best split |
| Cores / nodes | 1 ... max | x axis of the scaling plots |

Placement is not drawn as a 2x2 grid (pack/scatter x shared/excl) in every plot: it would be 8 panels per plot with both models and 4x the cluster time. 

If P6 shows a visible effect, only P2 and P4 get the 2x2 grid (shared axes); the rest goes in the appendix.

## 2. Performance plots

Unless the Type says otherwise, there is one line per version.

| File (`<model>` = diagonal / full) | What it shows | Type | x | y | Fixed |
|---|---|---|---|---|---|
| `P1-sweep-<model>` | How time per iteration grows with K and D | log-log, panels (a) K (b) D, dashed reference slope | K, D | time / iteration | 1 node, 1x8. K: `synth_clean_p01`, K = 2..64; D: `synth_sweep_dDDD`, K = 4. Logs in `results/logs/sweep/` |
| `P2-speedup-<model>` | How many times faster than `baseline-sequential` | lines + ideal | cores | speedup | 3 panels (`--grid`): `brainweb`, p08, p16 of `synth_overlap` (diagonal) / `synth_rare` (full) |
| `P3-efficiency-<model>` | How much of each added core is actually used | lines | cores | efficiency | as P2 |
| `P4-strong-<model>` | How total time drops with more cores, same problem | lines + ideal | cores | time | N fixed, as P2 |
| `P5-weak-<model>` | Whether time stays flat when data grows with the cores | lines | cores | time of the same version on 1 core / time on p cores | `synth_overlap_pNN` on NN cores (`--family`) |
| `P6-placement-<model>` | Whether the PBS placement changes the time | grouped bars, one bar per placement | nodes (1, 2, 4) | time / iteration | opt2, 1 model, 1 large dataset |
| `P7-ranks-threads-<model>` | Which ranks x threads split is fastest on one node | bars, fastest framed in red | split | time / iteration | 1 node, opt2 |
| `P8-breakdown-<model>` | Where the time goes (E-step, M-step, communication, I/O) | stacked bars (100%), total above, segments E-step, M-step, `t_comm`, I/O | 1, 8, 32 cores x version | share (%) | as P2 |
| `P9-opt-gain-<model>` | How much opt1 and opt2 gain over `baseline-MPI-OpenMP` | bars opt1, opt2, `baseline-MPI-OpenMP` = 1, ratio above | dataset | time / `baseline-MPI-OpenMP` | 32 cores, all long datasets (real first, dotted separator) |
| `P10-roofline-<model>` | Whether the code is compute or memory bound | roofline, one point per version | FLOP/byte | GFLOP/s | 1 node. Analytic `flops`, `bytes` (`include/perfmodel.h`); roofs from `--peak-gflops`, `--peak-bw` (STREAM) |
| `P11-imbalance-<model>` | How unevenly the work is split between ranks | bars mean + min-max, % = max/mean - 1 | nodes / split | compute time per rank | opt2, key `rank_compute` (E + M per rank) |
| `P12-size-<model>` | Whether bigger datasets get more speedup | scatter, filled = real, empty = synthetic | N (log) | speedup | all datasets, panels 8 and 32 cores |



## 3. Clustering plots

The result is the same in every version (same final log-likelihood), so the plots and the table below use only the **`baseline-sequential`** version (`bin/PEMC-baseline-sequential-<model>`, 1 core).

| File | What it shows | Type | Datasets | Version |
|---|---|---|---|---|
| `C1-convergence` | Whether EM converges and which model reaches the higher log-likelihood | log-likelihood per point vs iteration, both models per panel; `X` = iteration limit | 4 (default: synth_overlap_p01, cardiotocography, heart_failure, synth_clean_p01) | `baseline-sequential` |
| `C2-choose-k` | Which number of clusters K the criteria pick, vs the true K | BIC (thick), AIC (dotted) vs K, distance from best; star = best K, dotted line = true K; `synth_clean_p01` always added as control | 3 + control | `baseline-sequential` |
| `C3-pca` | The found clusters next to the true classes, in 2D | PCA 2D, rows = true / diagonal / full, clusters recolored after their class, ARI and explained variance | 4 | `baseline-sequential` |
| `C4-centroids-<model>` | The mean value of every variable in every cluster | centroid heatmap (variables x clusters), shared scale | 2 with D <= 30 | `baseline-sequential` |
| `C5-sizes` | Whether the found clusters have the same sizes as the true classes | bars of share per class: truth, diagonal, full | as C3 | `baseline-sequential` |
| `C6-bic-full-vs-diagonal` | Which covariance model fits each real dataset better | bars of BIC(full) - BIC(diagonal), colored by winner. † = full has lower log-likelihood (worse local optimum); ‡ = D = 1, same model | all real | `baseline-sequential` |

Clusters are matched one-to-one to the true classes (Hungarian algorithm), so names and colors agree across C3, C4, C5. Datasets of C1-C5: `--datasets`, in order of preference. C3 samples at most 5000 rows. `brainweb` (D = 1) has no PCA; `gene_rnaseq` has no full run (D = 20 264).

Table (numbers behind C1-C6, `baseline-sequential`, one row per real dataset and model: version, N, D, K, log-likelihood, BIC, AIC, iterations, converged, ARI): `results/tables/clustering-summary.csv`.

Data: `src/script/pemc-cluster.sh` runs `baseline-sequential`, both models, for K in a range (default 2..10 plus true K) into `results/logs/cluster/`; at the true K it adds `--save results/clusters` (`include/cluster-out.h`: labels, weights, means, log-likelihood per iteration). Free parameters for BIC/AIC: `(K-1) + 2KD` diagonal, `(K-1) + KD + KD(D+1)/2` full.

## 4. Runs

Runs and commands: see [exec-cluster.md](exec-cluster.md).

## 5. Conventions

- At least 3 repetitions per point (warm-up discarded), median with min-max error bars.
- Speedup is against **`baseline-sequential` of the same model**; say it in the caption.
- Log-log for time vs cores, linear for speedup and efficiency.
- File names: `<ID>-<what>-<model>.pdf`, e.g. `P2-speedup-diagonal.pdf`.
- The interactive app (`src/visual-sequential`) is for exploration and to show how the algorithm works
