#!/usr/bin/env python3
"""
plots-cluster.py

Clustering plots of the report (catalog C1-C6 of doc/plots.md, colors of doc/color.md) and the
summary table. The clustering result is the same in every version (the final log-likelihood is
checked), so everything here comes from the SEQUENTIAL BASELINE, diagonal and full.

Usage (from the root of the repository):
    qsub src/script/pemc-cluster.sh                     # 1. the runs, on the cluster (see that file)
    python3 src/plots-generate/plots-cluster.py         # 2. the plots, in results/plots/cluster/
    python3 src/plots-generate/plots-cluster.py --plots C2 C6 --datasets heart_failure parkinson

Inputs
    results/logs/cluster/*.log         one block per run and per K (format: include/logger.h)
                                       -> C2 (BIC/AIC vs K), C6, summary table
    results/clusters/<ds>_<model>_k<K>.{labels.bin,params.csv,loglik.csv}
                                       saved with --save at K = true K (include/cluster-out.h)
                                       -> C1 (convergence), C3 (PCA), C4 (centroids), C5 (sizes)
    data/real/<ds>/<ds>.bin, data/synthetic/...
                                       original data, only to project it with PCA (C3), and the
                                       true labels (<ds>_labels.bin or <ds>.labels.bin)

Plots (diagonal and full in the SAME figure, colors in MODEL_COLOR; only C4 has one file per model)
    C1 convergence   log-likelihood per point vs iteration, one panel per dataset, one line per model
    C2 choice of K   BIC and AIC vs K for both models, best K of each curve marked, true K dotted
    C3 PCA           2D projection: true classes, clusters of diagonal, clusters of full (3 rows);
                     the found clusters are recolored after the class they overlap most, ARI shown
    C4 centroids     heatmap of the cluster means (standardized variables), C4-centroids-<model>
    C5 sizes         share of the points per class: truth, diagonal, full (3 bars per class)
    C6 BIC full - diagonal at the true K, one bar per real dataset, colored by the winning model
Every figure: bold title in plain words, one legend above the panels, dataset name on every panel.
Table: results/tables/clustering-summary.csv (N, D, K, loglik, BIC, AIC, iterations, ARI).

Number of free parameters (for BIC = -2 loglik + p ln N, AIC = -2 loglik + 2 p):
    diagonal  (K - 1) + K D + K D            weights, means, variances
    full      (K - 1) + K D + K D (D + 1) / 2   weights, means, covariance matrices
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Root of the repository (this file is in <root>/src/plots-generate/): the default input and
# output folders are relative to it, so the script works from any current directory
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------
# Style: one role per color (doc/color.md)
# ---------------------------------------------------------------------------------------------
# Clusters / classes: categorical, in this order, each class with its own marker
CLASS_COLOR = ["#ff0000", "#22C55E", "#69d2ff", "#dca500", "#912603", "#006326",
               "#002fb0", "#7904d3", "#502B00", "#e44aff", "#ff8d35", "#2c79ff"]
CLASS_MARKER = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p"]
TEXT, IDEAL, GRID, MISSING = "#171e26", "#6B7280", "#E5E7EB", "#656B75"
# The two covariance models, when they share a figure (C1, C2, C3, C5, C6): one color each, and
# also a different line style / marker, so they stay distinct in grayscale
MODEL_COLOR = {"diagonal": "#0E7490", "full": "#C2410C"}       # teal / burnt orange
MODEL_LS = {"diagonal": "-", "full": "--"}
MODEL_MARKER = {"diagonal": "o", "full": "s"}
MODEL_LABEL = {"diagonal": "diagonal covariance", "full": "full covariance"}
TRUTH_COLOR = "#9CA3AF"                                          # true classes (C5): neutral gray
MODELS = ["diagonal", "full"]

# Same look as plots-performance.py: serif font, light gray grid under the data, no top/right
# frame, vector output with real fonts (fonttype 42)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# Number of true classes of every dataset = the K of the saved runs (same list as the scripts;
# parkinson has no classes: K = 2 is an arbitrary choice)
TRUE_K = {"brainweb": 12, "cardiotocography": 10, "cardiovascular": 2, "cdc_diabetes": 2,
          "diabetes130": 3, "eeg_eye_state": 2, "gene_rnaseq": 5, "heart_failure": 2,
          "parkinson": 2, "statlog_heart": 2, "synth_clean_p01": 4, "synth_overlap_p01": 5}
CONTROL = "synth_clean_p01"  # pure Gaussian data: the reference case of C2
REAL = ["brainweb", "cardiotocography", "cardiovascular", "cdc_diabetes", "diabetes130",
        "eeg_eye_state", "gene_rnaseq", "heart_failure", "parkinson", "statlog_heart"]


# ---------------------------------------------------------------------------------------------
# Reading of the logs and of the saved results
# ---------------------------------------------------------------------------------------------
def n_params(model, K, D):
    """Number of free parameters of the mixture (used by BIC and AIC): the K-1 independent
    weights, the K*D means and the variances (diagonal: K*D) or the covariance matrices
    (full: K matrices, each with D(D+1)/2 distinct entries because it is symmetric)."""
    return (K - 1) + K * D + (K * D if model == "diagonal" else K * D * (D + 1) / 2)


def load_logs(logdir):
    """Reads every *.log of logdir into a table with one row per run, and adds BIC and AIC."""
    rows = []  # one dict per run
    for path in sorted(logdir.glob("*.log")):
        run = None  # block being read
        # the extra "====" line at the end makes the loop hand over the last block too
        for line in path.read_text(errors="replace").splitlines() + ["==== end"]:
            if line.startswith("===="):
                if run:
                    rows.append(run)
                run = {}
            elif run is not None and ":" in line:
                k, v = line.split(":", 1)
                run[k.strip()] = v.strip()
    if not rows:
        sys.exit(f"no run found in {logdir}/*.log (run src/script/pemc-cluster.sh first)")
    df = pd.DataFrame(rows)
    # text -> numbers ("n/a" becomes NaN)
    for c in ("N", "D", "K", "iterations", "converged", "loglik", "ari"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["model"] = df["version"].str.rsplit("-", n=1).str[1]
    df["dataset"] = df["dataset"].map(lambda p: Path(p).stem)
    # BIC = -2 loglik + p ln(N), AIC = -2 loglik + 2 p: the lower, the better; they penalize
    # the number of parameters p, BIC more strongly when N is large
    df["bic"] = -2 * df.loglik + df.apply(lambda r: n_params(r.model, r.K, r.D), axis=1) * np.log(df.N)
    df["aic"] = -2 * df.loglik + 2 * df.apply(lambda r: n_params(r.model, r.K, r.D), axis=1)
    # the runs are deterministic: if a (dataset, model, K) was run twice, keep the last
    return df.drop_duplicates(["dataset", "model", "K"], keep="last").reset_index(drop=True)


def data_path(ds, data_root):
    """Path of the .bin file of a dataset name (real or synthetic, see the repository layout)."""
    if ds.startswith("synth_"):
        return data_root / "synthetic" / ds.rsplit("_p", 1)[0] / f"{ds}.bin"
    return data_root / "real" / ds / f"{ds}.bin"


def true_labels(ds, data_root):
    """True classes of the dataset, or None. Real: <ds>_labels.bin, synthetic: <ds>.labels.bin."""
    p = data_path(ds, data_root)
    for q in (p.with_name(p.stem + "_labels.bin"), p.with_name(p.stem + ".labels.bin")):
        if q.exists():
            return np.fromfile(q, dtype="<i4")
    return None


def saved(clusters, ds, model, K):
    """(labels, params DataFrame, loglik curve or None) of a saved run, or None if missing."""
    base = clusters / f"{ds}_{model}_k{K}"
    if not base.with_suffix(".labels.bin").exists():
        return None
    lab = np.fromfile(str(base) + ".labels.bin", dtype="<i4")
    par = pd.read_csv(str(base) + ".params.csv")
    ll = pd.read_csv(str(base) + ".loglik.csv") if Path(str(base) + ".loglik.csv").exists() else None
    return lab, par, ll


def pick(available, wanted, n):
    """The first n datasets of 'wanted' that are available; if there are none, any n available."""
    got = [d for d in wanted if d in available][:n]
    return got or sorted(available)[:n]


def save(fig, out, name, fmts):
    """Writes the figure in every requested format and frees its memory."""
    for f in fmts:
        fig.savefig(out / f"{name}.{f}")
    plt.close(fig)
    print("wrote", out / f"{name}.{fmts[0]}")


def need(cond, why):
    """Stops the current plot with a readable reason if its data is missing (main() prints it
    and goes on with the next one)."""
    if not cond:
        raise LookupError(why)


def class_color(labels):
    """Color of every label: the 12 class colors in order; -1 (outlier) = gray."""
    return np.where(labels < 0, MISSING, np.array(CLASS_COLOR)[labels % 12])


# ---- common layout: the same look in every figure (doc/plots.md, section 5) ----
def title(fig, text):
    """Bold title in plain words: what the figure shows. If the figure has a top_legend, the
    title goes just ABOVE it (call top_legend first): title, legend, panels, from the top."""
    leg = getattr(fig, "_top_legend", None)
    if leg is None:
        fig.suptitle(text, fontweight="bold", fontsize=10)
        return
    fig.canvas.draw()  # final layout and size of the legend, in figure coordinates
    top = leg.get_window_extent().transformed(fig.transFigure.inverted()).y1
    fig.text(0.5, top + 0.012, text, ha="center", va="bottom", fontweight="bold", fontsize=10)


def panel_label(ax, text, loc="upper left"):
    """Name of the panel written INSIDE it, bold, on a white box (it never hides the data
    behind it completely)."""
    x, ha = (0.02, "left") if "left" in loc else (0.98, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5, fontweight="bold",
            color=TEXT, zorder=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5))


def top_legend(fig, handles, labels, ncol=None, handlelength=2.2):
    """ONE legend for the whole figure, boxed, horizontal, just above the panels (never on the
    data). It sits outside the area of the panels: savefig(bbox_inches="tight") keeps it in the
    file. A tuple of handles is drawn side by side (e.g. a few class markers for one entry)."""
    from matplotlib.legend_handler import HandlerTuple
    fig._top_legend = fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol or len(labels),
        frameon=True, fontsize=7.5, handlelength=handlelength, columnspacing=1.4, edgecolor=GRID,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.9)})


# Width of the text of the report (IEEEtran, one column): every figure is drawn at its FINAL size,
# so LaTeX does not shrink it and the 7-8 pt labels stay 7-8 pt in the PDF
FIG_W = 7.0


def panel_grid(n, row_h, width_ratios=None, ncols=2):
    """n panels in a grid with ncols columns (2: four panels = 2 x 2), FIG_W wide, row_h inches
    per row. width_ratios (one per panel) sets the width of the panels of each row; an incomplete
    last row keeps the same column widths. Returns (fig, list of the n axes, ncols)."""
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceiling division
    fig = plt.figure(figsize=(FIG_W, row_h * nrows + 0.3), layout="constrained")
    gs = fig.add_gridspec(nrows, 1)
    axs = []
    for r in range(nrows):
        idx = list(range(r * ncols, min(n, (r + 1) * ncols)))
        wr = None
        if width_ratios is not None:
            wr = [width_ratios[i] for i in idx]
            wr += [float(np.mean(wr))] * (ncols - len(idx))  # empty slot of an incomplete row
        sub = gs[r].subgridspec(1, ncols, width_ratios=wr)
        axs += [fig.add_subplot(sub[0, c]) for c in range(len(idx))]
    return fig, axs, ncols


def true_k_runs(ctx, ds):
    """Runs of the logs at the true K of ds, one row per model (index = model)."""
    g = ctx.logs[(ctx.logs.dataset == ds) & (ctx.logs.K == TRUE_K[ds])]
    return g.set_index("model")


def hungarian(cost):
    """Optimal assignment of a SQUARE cost matrix (Hungarian / Kuhn-Munkres algorithm, O(n^3),
    version with potentials): returns col[i] = column given to row i, with the minimum total
    cost. Written here because the scipy of the system does not match the installed numpy."""
    n = cost.shape[0]
    INF = float("inf")
    u, v = np.zeros(n + 1), np.zeros(n + 1)  # potentials of rows and columns
    p = np.zeros(n + 1, dtype=int)           # p[j] = row assigned to column j (1-based, 0 = none)
    way = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv = np.full(n + 1, INF)
        used = np.zeros(n + 1, dtype=bool)
        while p[j0] != 0:                     # grow an alternating path until a free column
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
        while j0:                             # flip the path
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    col = np.zeros(n, dtype=int)
    for j in range(1, n + 1):
        col[p[j] - 1] = j - 1
    return col


def match_map(found, truth):
    """New name of every found cluster: the true class it is paired with. EM numbers its
    clusters in an arbitrary order (cluster 0 of EM has nothing to do with class 0), so they are
    paired one-to-one with the classes maximizing the total number of shared points (optimal
    assignment on the table found cluster x true class). Clusters left without a class (more
    clusters than classes) get new ids after the classes. Rows with truth < 0 (outliers) are
    ignored. Returns an array: new id of found cluster i."""
    m = truth >= 0
    kf, kt = found.max() + 1, max(truth.max() + 1, 1)
    n = max(kf, kt)
    C = np.zeros((n, n))  # square: padded with empty rows / columns
    np.add.at(C, (found[m], truth[m]), 1)
    col = hungarian(-C)   # maximize the overlap = minimize its opposite
    new, extra = np.zeros(kf, dtype=int), kt
    for i in range(kf):
        if col[i] < kt:
            new[i] = col[i]
        else:             # paired with a padding column: no class left for this cluster
            new[i], extra = extra, extra + 1
    return new


def match_to_truth(found, truth):
    """Found labels renamed after the paired true class (match_map): the same color / position
    means the same group as in the true classes."""
    return match_map(found, truth)[found]


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def c1_convergence(ctx, models):
    """C1: log-likelihood per point at every EM iteration, one panel per dataset, the two models
    in the same panel (same y scale: the higher curve fits the data better)."""
    avail = {d for d in TRUE_K if any(saved(ctx.clusters, d, m, TRUE_K[d]) for m in models)}
    dss = pick(avail, ctx.args.datasets, 4)
    need(dss, "no saved run")
    fig, axs, nc = panel_grid(len(dss), row_h=2.25)
    stopped = False  # some run hit the iteration limit: one more legend entry
    for ax, ds in zip(axs, dss):
        runs = true_k_runs(ctx, ds)
        lows = []
        for m in models:
            s = saved(ctx.clusters, ds, m, TRUE_K[ds])
            if s is None or s[2] is None or len(s[2]) < 2 or m not in runs.index:
                continue
            ll, N = s[2], runs.loc[m, "N"]
            # divided by N: the datasets have very different sizes, the value per point does not
            y = ll.loglik.to_numpy() / N
            ax.plot(ll.iteration, y, color=MODEL_COLOR[m], ls=MODEL_LS[m], lw=1.5,
                    marker=MODEL_MARKER[m], ms=3.5, markevery=max(1, len(y) // 8))
            if not runs.loc[m, "converged"]:  # stopped at the maximum number of iterations
                ax.plot(ll.iteration.iloc[-1], y[-1], marker="X", ms=8, color=MODEL_COLOR[m],
                        mec=TEXT, mew=0.6, ls="", zorder=5)
                stopped = True
            lows.append(y[min(1, len(y) - 1)])
        # iteration 0 (random start) is far below everything else and would squash the curves:
        # the y axis starts from the value after the first iteration
        top = ax.get_ylim()[1]
        if lows:
            bottom = min(lows)
            ax.set_ylim(bottom - 0.05 * (top - bottom), top)
        ax.set_title(f"{ds}  (K = {TRUE_K[ds]})", fontsize=8, fontweight="bold")
        ax.set_xlabel("EM iteration")
    for ax in axs[::nc]:  # left column
        ax.set_ylabel("log-likelihood per point\n(higher = better; from iteration 1)")
    h = [plt.Line2D([], [], color=MODEL_COLOR[m], ls=MODEL_LS[m], marker=MODEL_MARKER[m], ms=4)
         for m in models]
    lab = [MODEL_LABEL[m] for m in models]
    if stopped:
        h.append(plt.Line2D([], [], color=IDEAL, marker="X", ms=7, mec=TEXT, mew=0.6, ls=""))
        lab.append("stopped at the iteration limit (not converged)")
    top_legend(fig, h, lab)
    title(fig, "EM convergence: log-likelihood at every iteration (K = number of true classes)")
    save(fig, ctx.out, "C1-convergence", ctx.fmts)


def c2_choose_k(ctx, models):
    """C2: BIC and AIC for every K, both models in the same panel. Every criterion is shown as
    the distance from its best value over BOTH models: 0 = the best (model, K) of the panel."""
    d = ctx.logs[ctx.logs.model.isin(models)]
    avail = {x for x, g in d.groupby("dataset") if g.K.nunique() >= 3}  # a curve needs >= 3 K
    dss = pick(avail, ctx.args.datasets, 3)
    # synth_clean is pure Gaussian data with K = 4: the CONTROL case where BIC must find the true
    # K. It is always shown (as the last panel), so the real datasets can be judged against it
    if CONTROL in avail and CONTROL not in dss:
        dss.append(CONTROL)
    need(dss, "no dataset with a range of K")
    fig, axs, nc = panel_grid(len(dss), row_h=2.4)
    for ax, ds in zip(axs, dss):
        g = d[d.dataset == ds]
        span = 1.0
        for crit, lw, alpha in (("bic", 1.6, 1.0), ("aic", 1.0, 0.55)):
            best = g[crit].min()  # best value of this criterion over both models
            for m in models:
                gm = g[g.model == m].sort_values("K")
                if gm.empty:
                    continue
                y = gm[crit] - best
                span = max(span, y.max())
                ax.plot(gm.K, y, color=MODEL_COLOR[m], lw=lw, alpha=alpha,
                        ls=MODEL_LS[m] if crit == "bic" else ":", marker=MODEL_MARKER[m],
                        ms=3.5 if crit == "bic" else 2.5)
                i = y.to_numpy().argmin()  # best K of this (criterion, model)
                ax.plot(gm.K.iloc[i], y.iloc[i], marker="*", ms=12 if crit == "bic" else 9,
                        color=MODEL_COLOR[m], mec=TEXT, mew=0.6, ls="", alpha=alpha, zorder=5)
        ax.axvline(TRUE_K[ds], color=IDEAL, ls=(0, (1, 2)), lw=1.2)
        # symlog: linear near 0, logarithmic beyond (the values span many orders of magnitude)
        ax.set_yscale("symlog", linthresh=max(1.0, 0.01 * span))
        ax.set_ylim(bottom=0)
        ax.set_xticks(sorted(g.K.unique()))
        ax.set_xlabel("number of clusters K")
        ax.set_title(ds + ("  (control: pure Gaussian)" if ds == CONTROL else ""),
                     fontsize=8, fontweight="bold")
    for ax in axs[::nc]:  # left column
        ax.set_ylabel("criterion minus its best value\n(lower = better)")
    h, lab = [], []
    for m in models:
        h.append(plt.Line2D([], [], color=MODEL_COLOR[m], ls=MODEL_LS[m], marker=MODEL_MARKER[m],
                            ms=4, lw=1.6))
        lab.append(f"BIC, {MODEL_LABEL[m]}")
    for m in models:
        h.append(plt.Line2D([], [], color=MODEL_COLOR[m], ls=":", marker=MODEL_MARKER[m], ms=3,
                            lw=1.0, alpha=0.55))
        lab.append(f"AIC, {MODEL_LABEL[m]}")
    h += [plt.Line2D([], [], color=TEXT, marker="*", ms=10, ls=""),
          plt.Line2D([], [], color=IDEAL, ls=(0, (1, 2)), lw=1.2)]
    lab += ["best K of each curve", "true number of classes"]
    top_legend(fig, h, lab, ncol=3)
    title(fig, "Choice of K: BIC and AIC for every number of clusters")
    save(fig, ctx.out, "C2-choose-k", ctx.fmts)


def project_pca(X, max_points, seed=0):
    """2D PCA of a random subsample (at most max_points rows). Returns (coords, row indices,
    share of the variance carried by the two components)."""
    rng = np.random.default_rng(seed)  # fixed seed: the same sample in every run of the script
    n = X.shape[0]
    idx = np.sort(rng.choice(n, size=min(n, max_points), replace=False))
    Z = np.asarray(X[idx], dtype=float)  # only these rows are read from the (memory-mapped) file
    Z -= Z.mean(axis=0)                  # PCA works on centered data
    # rows of vt = principal directions (eigenvectors of the covariance), the first two carry
    # the largest variance; projecting on them gives the 2D coordinates
    _, sv, vt = np.linalg.svd(Z, full_matrices=False)
    explained = (sv[:2] ** 2).sum() / (sv ** 2).sum()  # singular values^2 = variance per direction
    return Z @ vt[:2].T, idx, explained


def scatter_classes(ax, xy, labels):
    """Scatter with one marker and one color per class (the info never depends on hue only)."""
    for k in sorted(set(labels)):
        m = labels == k
        col = MISSING if k < 0 else CLASS_COLOR[k % 12]
        mk = "x" if k < 0 else CLASS_MARKER[k % 12]
        ax.scatter(xy[m, 0], xy[m, 1], s=5, color=col, marker=mk, alpha=0.65, linewidths=0.4)
    # Axis limits at the 0.5-99.5 percentiles (+ margin): a few outliers must not squash the
    # picture (they are still drawn if inside, otherwise cut by the frame)
    for axis, setter in ((0, ax.set_xlim), (1, ax.set_ylim)):
        lo, hi = np.percentile(xy[:, axis], [0.5, 99.5])
        pad = 0.08 * (hi - lo + 1e-12)
        setter(lo - pad, hi + pad)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def c3_pca(ctx, models):
    """C3: the data projected on its first 2 principal components (PCA), the same points drawn
    three times: colored by true class (top row), by the clusters of the diagonal model and by
    the clusters of the full model. The found clusters are renamed after the true class they
    overlap most (match_to_truth), so the SAME color means the same group in every row."""
    avail = {d for d in TRUE_K if all(saved(ctx.clusters, d, m, TRUE_K[d]) for m in models)}
    dss = []  # datasets that can be drawn
    for ds in pick(avail, ctx.args.datasets, 4):
        p = data_path(ds, ctx.data)
        if p.exists() and np.fromfile(p, dtype="<i4", count=2)[1] >= 2:  # PCA needs D >= 2
            dss.append(ds)
    need(dss, "no saved run with D >= 2 and its data file")
    rows = ["truth"] + list(models)
    side = (FIG_W - 0.6) / len(dss)  # panels about square
    fig, axs = plt.subplots(len(rows), len(dss), figsize=(FIG_W, side * len(rows) + 0.5),
                            squeeze=False, layout="constrained")
    for j, ds in enumerate(dss):
        N, D = np.fromfile(data_path(ds, ctx.data), dtype="<i4", count=2)  # header of the .bin
        # memory map: the file is not loaded in RAM, only the sampled rows are read
        X = np.memmap(data_path(ds, ctx.data), dtype="<f8", mode="r", offset=8, shape=(N, D))
        xy, idx, expl = project_pca(X, ctx.args.max_points)  # same sample and projection in every row
        truth = true_labels(ds, ctx.data)
        truth = truth if truth is not None and len(truth) == N else None
        runs = true_k_runs(ctx, ds)
        axs[0, j].set_title(f"{ds}\nK={TRUE_K[ds]}  N={N:,}  D={D}\n"
                            f"PCA: {100 * expl:.0f}% of variance", fontsize=7, fontweight="bold")
        for i, r in enumerate(rows):
            ax = axs[i, j]
            if r == "truth":
                if truth is None:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "no true labels", ha="center", va="center", color=MISSING)
                    continue
                scatter_classes(ax, xy, truth[idx])
                continue
            lab = saved(ctx.clusters, ds, r, TRUE_K[ds])[0]
            if truth is not None:
                lab = match_to_truth(lab, truth)
            scatter_classes(ax, xy, lab[idx])
            if r in runs.index and not np.isnan(runs.loc[r, "ari"]):
                panel_label(ax, f"ARI = {runs.loc[r, 'ari']:.2f}", loc="lower right")
            for s in ax.spines.values():  # frame in the color of the model: which row is which
                s.set_visible(True)
                s.set_color(MODEL_COLOR[r])
                s.set_linewidth(1.6)
    axs[0, 0].set_ylabel("true classes", fontweight="bold")
    for i, m in enumerate(models, start=1):
        axs[i, 0].set_ylabel(f"found clusters\n{MODEL_LABEL[m]}", color=MODEL_COLOR[m], fontweight="bold")
    # legend: what the colors mean (they are classes, different in every dataset) and the ARI
    h = [plt.Line2D([], [], color=CLASS_COLOR[k], marker=CLASS_MARKER[k], ls="", ms=5) for k in range(4)]
    h.append(plt.Line2D([], [], color="none"))
    top_legend(fig, [tuple(h[:4]), h[4]],
               ["one color and marker per class (a found cluster takes the color of the class it overlaps most)",
                "ARI: agreement with the true classes, 1 = perfect, 0 = random"],
               ncol=1, handlelength=5.5)
    title(fig, f"Clusters in 2D (PCA, {ctx.args.max_points:,} random points): true classes vs found")
    save(fig, ctx.out, "C3-pca", ctx.fmts)


def variable_names(ds, D, data_root):
    """Column names from the dataset json (real datasets); x1..xD if not available."""
    j = data_path(ds, data_root).with_suffix(".json")
    try:
        cols = json.loads(j.read_text())["columns"]
        if len(cols) == D:
            return list(cols)
    except (OSError, KeyError, ValueError):
        pass
    return [f"x{i + 1}" for i in range(D)]


def c4_centroids(ctx, model):
    avail = {d for d in TRUE_K if saved(ctx.clusters, d, model, TRUE_K[d])}
    cand = [d for d in avail if saved(ctx.clusters, d, model, TRUE_K[d])[1].shape[1] - 2 <= 30]
    dss = pick(set(cand), ctx.args.datasets, 2)  # small / medium D only: a heatmap must be readable
    need(dss, "no saved run with D <= 30")
    # Data of every panel. Clusters ordered by weight, the largest on the left. The matrix is
    # transposed (one ROW per variable, one COLUMN per cluster): the variable names, long, stay
    # horizontal on the y axis and the x axis only carries the short cluster labels.
    panels = []
    for ds in dss:
        lab, par, _ = saved(ctx.clusters, ds, model, TRUE_K[ds])
        par = par.sort_values("pi", ascending=False)
        M = par[[c for c in par.columns if c.startswith("mu_")]].to_numpy().T  # D x K
        names = [n.replace("_", " ") for n in variable_names(ds, M.shape[0], ctx.data)]
        # cluster names: the paired true class (match_map), as in C3 and C5, so the same group has
        # the same name in every figure; without true labels the numbers of EM are kept
        truth = true_labels(ds, ctx.data)
        if truth is not None and len(truth) == len(lab):
            mp = match_map(lab, truth)
            name = [f"{mp[int(c)]}" + ("*" if mp[int(c)] >= TRUE_K[ds] else "") for c in par.cluster]
        else:
            name = [f"{int(c)}" for c in par.cluster]
        xlab = [f"{n}\n{100 * p:.0f}%" for n, p in zip(name, par.pi)]
        panels.append((ds, M, names, xlab))
    # ONE color scale for all the panels (comparable colors, one colorbar), symmetric around 0
    # (white = mean of the variable); clipped at +-2 standard deviations so that a few extreme
    # means do not wash out all the others
    LIM = 2.0
    lim = max(1e-6, min(LIM, max(np.abs(M).max() for _, M, _, _ in panels)))
    maxD = max(M.shape[0] for _, M, _, _ in panels)
    # panel width proportional to the number of clusters (plus room for the variable names)
    fig, axs = plt.subplots(1, len(panels), squeeze=False, layout="constrained",
                            figsize=(FIG_W, 1.1 + 0.16 * maxD),
                            gridspec_kw={"width_ratios": [M.shape[1] + 3 for _, M, _, _ in panels]})
    for ax, (ds, M, names, xlab) in zip(axs[0], panels):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
        ax.grid(False)
        ax.set_yticks(range(M.shape[0]), names, fontsize=7)
        ax.set_xticks(range(M.shape[1]), xlab, fontsize=6.5)
        ax.tick_params(length=0)
        ax.set_xlabel("cluster, named after its true class\n(share of the points)")
        ax.set_title(f"{ds}  (K = {M.shape[1]}, D = {M.shape[0]})", fontweight="bold", fontsize=9)
        # thin white lines between the cells: the clusters stay distinct even with similar colors
        ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    clip = f" (clipped at $\\pm${lim:g})" if lim == LIM else ""
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.85, aspect=25, pad=0.02,
                 label=f"standardized mean{clip}\nred: above average, blue: below")
    fig.suptitle(f"Cluster centroids: mean of every variable in each cluster ({model} covariance)",
                 fontweight="bold", fontsize=10)
    save(fig, ctx.out, f"C4-centroids-{model}", ctx.fmts)


def c5_sizes(ctx, models):
    """C5: share of the points in every cluster, next to the share of every true class. For each
    class three bars: the true class (gray), the cluster of the diagonal model and the cluster of
    the full model matched to it (match_to_truth). A good clustering has the colored bars as high
    as the gray one. Without true labels the clusters are only sorted by size."""
    avail = {d for d in TRUE_K if all(saved(ctx.clusters, d, m, TRUE_K[d]) for m in models)}
    dss = pick(avail, ctx.args.datasets, 4)
    need(dss, "no saved run")
    width = [max(TRUE_K[ds], 3) + 2 for ds in dss]  # panel width ~ number of classes
    fig, axs, nc = panel_grid(len(dss), row_h=2.3, width_ratios=width)
    series = [("truth", TRUTH_COLOR, None)] + [(m, MODEL_COLOR[m], None) for m in models]
    bw = 0.8 / len(series)
    extra, matched = False, False  # for the note under the figure
    for ax, ds in zip(axs, dss):
        truth = true_labels(ds, ctx.data)
        lab = {m: saved(ctx.clusters, ds, m, TRUE_K[ds])[0] for m in models}
        if truth is not None and len(truth) == len(next(iter(lab.values()))):
            lab = {m: match_to_truth(l, truth) for m, l in lab.items()}
            share = {"truth": np.bincount(truth[truth >= 0])}
            xname = "true class (number)"
            matched = True
        else:
            truth = None
            lab = {m: np.argsort(np.argsort(-np.bincount(l)))[l] for m, l in lab.items()}  # 0 = largest
            share = {}
            xname = "cluster, largest first\n(no true labels)"
        for m in models:
            share[m] = np.bincount(lab[m])
        n = max(len(v) for v in share.values())
        # order of the classes on the x axis: the largest true class (or cluster) first
        ref = share["truth"] if truth is not None else share[models[0]]
        order = np.argsort(-np.pad(ref, (0, n - len(ref))))
        for s, (name, col, _) in enumerate(series):
            if name not in share:
                continue
            v = np.pad(share[name], (0, n - len(share[name])))
            ax.bar(np.arange(n) + (s - (len(series) - 1) / 2) * bw, 100 * v[order] / v.sum(), bw,
                   color=col, edgecolor=TEXT, linewidth=0.5)
        # a cluster matched to no class (more clusters than classes) is marked with "*"
        ax.set_xticks(range(n), [str(c) if c < TRUE_K[ds] else f"{c}*" for c in order], fontsize=7)
        extra = extra or n > TRUE_K[ds]
        ax.set_xlabel(xname)  # x axis named under EVERY panel
        ax.set_title(ds, fontsize=8, fontweight="bold")
    for ax in axs[::nc]:  # left column
        ax.set_ylabel("share of the points (%)")
    # note under the figure: how a found cluster is put above a true class
    note = []
    if matched:
        note.append("each found cluster is shown above the true class it shares most points with")
    if extra:
        note.append("* = found cluster that matches no class")
    if note:
        fig.supxlabel("; ".join(note), fontsize=7, color=IDEAL)
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=TEXT, lw=0.5) for _, c, _ in series]
    lab = ["true classes"] + [f"found clusters, {MODEL_LABEL[m]}" for m in models]
    top_legend(fig, h, lab)
    title(fig, "Cluster sizes: share of the points in every cluster vs every true class")
    save(fig, ctx.out, "C5-sizes", ctx.fmts)


def c6_model_choice(ctx, models=None):
    """C6: BIC(full) - BIC(diagonal) at the true K, one bar per real dataset. Below 0 the full
    covariance fits better even after paying for its extra parameters; above 0 the diagonal one
    is enough. The bar takes the color of the model that wins.
    Two cases are NOT a comparison of the models and are marked:
      - D = 1: with one variable the two models are the same model (same parameters), the
        difference only says that the two runs stopped in different points (gray bar);
      - log-likelihood of full < diagonal: full contains diagonal (a full matrix with zero
        covariances), so at the best solution its log-likelihood cannot be lower; a lower value
        means a worse local optimum of EM (one start per run), not a worse model (hatched bar)."""
    rows = []  # (dataset, D, BIC difference, kind)
    for ds in REAL:
        runs = true_k_runs(ctx, ds)
        if "diagonal" in runs.index and "full" in runs.index:
            d, f = runs.loc["diagonal"], runs.loc["full"]
            kind = "same" if int(d.D) == 1 else ("local" if f.loglik < d.loglik else "ok")
            rows.append((ds, int(d.D), f.bic - d.bic, kind))
    need(rows, "no real dataset with both models")
    rows.sort(key=lambda r: r[2])  # most favorable to the full model on the left
    fig, ax = plt.subplots(figsize=(FIG_W, 3.4), layout="constrained")
    v = np.array([r[2] for r in rows])
    for i, (ds, D, x, kind) in enumerate(rows):
        col = MISSING if kind == "same" else MODEL_COLOR["full" if x < 0 else "diagonal"]
        ax.bar(i, x, 0.7, color=col, edgecolor=TEXT, linewidth=0.6,
               hatch="////" if kind == "local" else None, alpha=0.75 if kind != "ok" else 1.0)
    ax.axhline(0, color=TEXT, lw=0.9)
    # symlog: linear near 0 (|difference| < 100) and logarithmic beyond, because the differences
    # span many orders of magnitude and have both signs
    ax.set_yscale("symlog", linthresh=100)
    for i, x in enumerate(v):  # the value on every bar, outside its end
        ax.text(i, x * 1.6 if abs(x) > 100 else x + (40 if x >= 0 else -40), f"{x:+.2g}",
                ha="center", va="bottom" if x >= 0 else "top", fontsize=6.5, color=TEXT)
    ax.set_ylim(-abs(v.min()) * 12 if v.min() < 0 else -100, abs(v.max()) * 12 if v.max() > 0 else 100)
    mark = {"ok": "", "local": " †", "same": " ‡"}
    ax.set_xticks(range(len(rows)), [f"{r[0]}{mark[r[3]]}\n(D = {r[1]})" for r in rows], rotation=40,
                  ha="right", fontsize=7)
    ax.set_ylabel("BIC(full) − BIC(diagonal)\n(log scale beyond ±100)")
    kinds = {r[3] for r in rows}
    h = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLOR[m], ec=TEXT, lw=0.5) for m in ("full", "diagonal")]
    lab = ["full covariance fits better (BIC difference < 0)",
           "diagonal covariance is enough (BIC difference > 0)"]
    if "local" in kinds:
        h.append(plt.Rectangle((0, 0), 1, 1, fc="white", ec=TEXT, lw=0.5, hatch="////"))
        lab.append("† log-likelihood of full < diagonal: a worse local optimum of EM, not a worse model")
    if "same" in kinds:
        h.append(plt.Rectangle((0, 0), 1, 1, fc=MISSING, ec=TEXT, lw=0.5, alpha=0.75))
        lab.append("‡ D = 1: the two models are the same, the difference is only between two runs")
    top_legend(fig, h, lab, ncol=1)
    title(fig, "Diagonal or full covariance? BIC difference at the true K, per real dataset")
    save(fig, ctx.out, "C6-bic-full-vs-diagonal", ctx.fmts)


def summary_table(ctx):
    """Table of the datasets at the true K, both models (instead of a plot)."""
    rows = []
    for ds, K in TRUE_K.items():
        for model in MODELS:
            g = ctx.logs[(ctx.logs.dataset == ds) & (ctx.logs.model == model) & (ctx.logs.K == K)]
            if len(g):
                r = g.iloc[0]
                rows.append(dict(dataset=ds, model=model, N=int(r.N), D=int(r.D), K=int(r.K),
                                 loglik=r.loglik, BIC=r.bic, AIC=r.aic,
                                 iterations=int(r.iterations), converged=int(r.converged), ARI=r.ari))
    need(rows, "no run at the true K")
    ctx.tables.mkdir(parents=True, exist_ok=True)
    path = ctx.tables / "clustering-summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)


PLOTS = {"C1": c1_convergence, "C2": c2_choose_k, "C3": c3_pca, "C4": c4_centroids,
         "C5": c5_sizes, "C6": c6_model_choice}


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs/cluster")
    ap.add_argument("--clusters", type=Path, default=ROOT / "results/clusters")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "results/plots/cluster")
    ap.add_argument("--tables", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--plots", nargs="+", default=list(PLOTS) + ["table"], help="C1..C6 and/or table")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--datasets", nargs="+",
                    default=["synth_overlap_p01", "cardiotocography", "heart_failure", "synth_clean_p01",
                             "eeg_eye_state", "statlog_heart"],
                    help="datasets for C1-C5, in order of preference (the first ones available are used)")
    ap.add_argument("--max-points", type=int, default=5000, help="rows drawn in the PCA of C3")
    ap.add_argument("--fmt", nargs="+", default=["pdf"], help="output formats (pdf, png, ...)")
    args = ap.parse_args()

    # ---- data: all the runs of the cluster logs, with BIC and AIC ----
    args.out.mkdir(parents=True, exist_ok=True)
    logs = load_logs(args.logs)
    ctx = argparse.Namespace(args=args, logs=logs, clusters=args.clusters, data=args.data,
                             out=args.out, tables=args.tables, fmts=args.fmt)
    print(f"{len(logs)} runs, datasets: {sorted(set(logs.dataset))}")

    # ---- draw: C4 once per model (one heatmap per model); C1, C2, C3, C5, C6 and the table
    # once, with both models in the same figure (they compare diagonal and full) ----
    for pid in args.plots:
        try:
            if pid == "table":
                summary_table(ctx)
            elif pid == "C4":
                for model in args.models:
                    try:
                        c4_centroids(ctx, model)
                    except LookupError as e:
                        print(f"skip {pid} {model}: {e}")
            elif pid in PLOTS:
                PLOTS[pid](ctx, args.models)
            else:
                print(f"skip {pid}: unknown")
        except LookupError as e:
            print(f"skip {pid}: {e}")


if __name__ == "__main__":
    main()
