# Plots: guidelines

Rules to decide which plots go in the report and how to keep them readable. Scripts: `src/plots-generate/plots-performance.py` and `src/plots-generate/plots-cluster.py`, output in `results/plots/`.

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
| Placement | pack, scatter, pack:excl, scatter:excl | **Experimental parameter, not a plot dimension.** One dedicated plot (P6), then fixed to the best value everywhere else |
| Ranks x threads | e.g. 1x16 ... 16x1 | One dedicated heatmap (P7), then fixed to the best split |
| Cores / nodes | 1 ... max | The x axis of the scaling plots |

### Placement procedure (the part that causes the mess)

1. Run only the placement experiment: 2 and 4 nodes, 1 large dataset, best version, 1 model, the 4 placements.
2. Plot P6 (grouped bars). This is the result "does placement matter?".
3. Pick the best placement. **All the other performance plots use only that one**, declared in every caption.
4. Only if the best placement changes with the number of nodes, add a speedup plot with one line per placement, **for the best version only**.

Same logic for ranks x threads: find the best split once (P7), then fix it.

## 2. Performance plots (catalog)

| ID | Question | Type | x | y | Lines / groups | Fixed |
|---|---|---|---|---|---|---|
| P1 | How does time per iteration scale with K and D? | log-log lines | K (or D) | time / iteration | versions | dataset synthetic, placement |
| P2 | How much faster than sequential? (speedup) | lines + ideal | cores | speedup | versions | 1 large + 1 small dataset, placement |
| P3 | How efficient? | lines | cores | efficiency | versions | same as P2. Put it as a second panel of P2 |
| P4 | Strong scaling | lines + ideal | cores | time (or speedup) | versions | N fixed |
| P5 | Weak scaling | lines | cores | time (or efficiency) | versions | N / core fixed (generated synthetic) |
| P6 | Does PBS placement matter? | grouped bars | nodes (1, 2, 4) | time / iteration | 4 placements | best version, 1 model, 1 large dataset |
| P7 | Best ranks x threads at fixed cores? | heatmap | threads | ranks | color = time | 1 node, best version |
| P8 | Where does the time go? | stacked bars | ranks | time | E-step, M-step, `t_comm`, I/O | 1 version per panel or bars side by side |
| P9 | What do opt1 and opt2 gain over the baseline? | bars normalized (baseline = 1) | dataset | time / baseline | opt1, opt2 | model per panel, placement |
| P10 | Is it compute or memory bound? | roofline | FLOP/byte | GFLOPS | one point per version | model per panel, node. Optional |
| P11 | Load imbalance between ranks | bars min/mean/max | ranks | compute time per rank | none | Optional. `t_comm` includes imbalance |

Priority: P2(+P3), P4, P5, P6, P7, P8, P9 are essential. P1, P10, P11 only if there is time and space.

## 3. Clustering plots (catalog)

The clustering result is the same in every version (we check the final log-likelihood), so these plots use **one version only**, the baseline.

| ID | Question | Type | Datasets |
|---|---|---|---|
| C1 | Does EM converge? | log-likelihood vs iteration, one line per dataset | 3-4 real |
| C2 | Which K is right? | BIC and AIC vs K, minimum marked | 2-3 real |
| C3 | Do the clusters look right? | PCA 2D, colored by cluster, grid of panels | 3-4 real + 1 synthetic (known truth) |
| C4 | What does each cluster look like? | heatmap of the centroids | 1-2 real with small/medium D |
| C5 | Are the clusters balanced? | bars of cluster sizes | same as C3. Can be merged with C4 |
| C6 | Diagonal or full, which fits better? | bars of the BIC difference (full - diagonal) | all real |

Table instead of plot: the 10 real datasets with N, D, K, log-likelihood, BIC, iterations, for both models.

## 4. Experiment matrix (what to run on the cluster for each plot)

| Plots | Runs needed |
|---|---|
| P6 | 4 placements x {2, 4 nodes} x 1 dataset x 1 version x 1 model (x repetitions) |
| P7 | all ranks x threads splits on 1 node x 1 dataset x best version |
| P2, P3, P4, P8, P9 | 4 versions x 2 models x cores {1, 2, 4, ..., max} x 3 datasets, with the fixed placement and split |
| P5 | 4 versions x 2 models x cores {...} with N proportional to cores |
| P1 | synthetic with varying K and D, 4 versions, a fixed core count |
| C1-C6 | 1 run per dataset and model (baseline), saving the clustering output |

Run P6 and P7 **first**: they fix the placement and the split for all the rest, so the other runs are not repeated.

## 5. Practical conventions

- Repetitions: at least 3 per point (discard warm-up), plot the median, error bars min-max.
- Speedup is always computed against the **sequential version of the same model**, not against the parallel one on 1 core. Say it in the caption.
- Log-log axes for time vs cores, linear for speedup and efficiency.
- A caption always states: model, dataset, placement, ranks x threads, repetitions.
- File names: `<ID>-<what>-<model>.pdf`, e.g. `P2-speedup-diagonal.pdf`.
- The interactive app (`src/visual-sequential`) is for exploration and the oral exam. The report uses only the static plots produced by the two scripts, so every figure can be regenerated.
