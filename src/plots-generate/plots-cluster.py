#!/usr/bin/env python3
"""
plots-cluster.py

Clustering plots of the report (catalog C1-C6 of doc/plots.md, colors of doc/color.md) and the
summary table. The clustering result is the same in every version (the final log-likelihood is
checked), so everything here comes from the SEQUENTIAL BASELINE, diagonal and full.

Usage (from the root of the repository):
    src/script/pemc-cluster.sh                          # 1. the runs (see that file)
    python3 src/plots-generate/plots-cluster.py         # 2. the plots, in results/plots/
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

Plots (one file per model, <ID>-<what>-<model>.pdf; C6 compares the two models)
    C1 convergence   progress of the log-likelihood vs iteration, one line per dataset
    C2 choice of K   BIC and AIC minus their minimum vs K, minimum marked, true K dotted
    C3 PCA           2D projection colored by found cluster (top) and by true class (bottom)
    C4 centroids     heatmap of the cluster means (standardized variables)
    C5 sizes         bars of the cluster sizes
    C6 BIC full - diagonal at the true K, one bar per dataset
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


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def c1_convergence(ctx, model):
    avail = {d for d in TRUE_K if saved(ctx.clusters, d, model, TRUE_K[d])}
    dss = pick(avail, ctx.args.datasets, 4)
    need(dss, "no saved run")
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for i, ds in enumerate(dss):  # dataset i = the i-th color / marker of the cluster list
        ll = saved(ctx.clusters, ds, model, TRUE_K[ds])[2]
        if ll is None or len(ll) < 2:
            continue
        y = ll.loglik.to_numpy()
        # the datasets have log-likelihoods of very different size: each curve is rescaled
        # between its first and its last value, so only the SHAPE of the convergence is compared
        span = y[-1] - y[0]
        prog = (y - y[0]) / span if span != 0 else np.ones_like(y)  # 0 = start, 1 = final value
        ax.plot(ll.iteration, prog, color=CLASS_COLOR[i], marker=CLASS_MARKER[i], ms=4, lw=1.3,
                markevery=max(1, len(y) // 10), label=f"{ds} (K={TRUE_K[ds]})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("log-likelihood progress (0 = start, 1 = final)")
    ax.set_title(f"C1  convergence of EM, {model}")
    ax.legend()
    save(fig, ctx.out, f"C1-convergence-{model}", ctx.fmts)


def c2_choose_k(ctx, model):
    d = ctx.logs[ctx.logs.model == model]
    # only datasets with at least 3 values of K make a curve
    avail = {x for x, g in d.groupby("dataset") if g.K.nunique() >= 3}
    dss = pick(avail, ctx.args.datasets, 3)
    need(dss, "no dataset with a range of K")
    fig, axs = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 3.0), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        g = d[d.dataset == ds].sort_values("K")
        for col, ls, mk, c, name in (("bic", "-", "o", TEXT, "BIC"), ("aic", "--", "s", IDEAL, "AIC")):
            y = g[col] - g[col].min()  # difference from the minimum: panels are comparable
            # (the absolute value of BIC depends on N and on the scale of the data)
            ax.plot(g.K, y, ls=ls, marker=mk, color=c, ms=4, lw=1.3, label=name)
            i = y.to_numpy().argmin()  # position of the best K for this criterion
            ax.plot(g.K.iloc[i], 0, marker="*", color=c, ms=13, mec=TEXT, mew=0.6, ls="")
        if ds in TRUE_K:  # the number of classes of the labels, for reference
            ax.axvline(TRUE_K[ds], color=IDEAL, ls=":", lw=1)
        ax.set_yscale("symlog", linthresh=max(1.0, 0.01 * (g.bic.max() - g.bic.min())))
        ax.set_ylim(bottom=0)  # both curves are >= 0 by construction: no negative half-axis
        ax.set_xlabel("K")
        ax.set_title(ds)
    axs[0, 0].set_ylabel("criterion - minimum (lower is better)")
    axs[0, 0].legend()
    fig.suptitle(f"C2  choice of K, {model} (star = minimum, dotted = true K)", y=1.04)
    save(fig, ctx.out, f"C2-choose-k-{model}", ctx.fmts)


def project_pca(X, labels_list, max_points, seed=0):
    """2D PCA of a random subsample (at most max_points rows). Returns (coords, row indices)."""
    rng = np.random.default_rng(seed)  # fixed seed: the same sample in every run of the script
    n = X.shape[0]
    idx = np.sort(rng.choice(n, size=min(n, max_points), replace=False))
    Z = np.asarray(X[idx], dtype=float)  # only these rows are read from the (memory-mapped) file
    Z -= Z.mean(axis=0)                  # PCA works on centered data
    # rows of vt = principal directions (eigenvectors of the covariance), the first two carry
    # the largest variance; projecting on them gives the 2D coordinates
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ vt[:2].T, idx


def scatter_classes(ax, xy, labels):
    """Scatter with one marker and one color per class (the info never depends on hue only)."""
    for k in sorted(set(labels)):
        m = labels == k
        col = MISSING if k < 0 else CLASS_COLOR[k % 12]
        mk = "x" if k < 0 else CLASS_MARKER[k % 12]
        ax.scatter(xy[m, 0], xy[m, 1], s=7, color=col, marker=mk, alpha=0.65, linewidths=0.5)
    # Axis limits at the 0.5-99.5 percentiles (+ margin): a few outliers must not squash the
    # picture (they are still drawn if inside, otherwise cut by the frame)
    for axis, setter in ((0, ax.set_xlim), (1, ax.set_ylim)):
        lo, hi = np.percentile(xy[:, axis], [0.5, 99.5])
        pad = 0.08 * (hi - lo + 1e-12)
        setter(lo - pad, hi + pad)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def c3_pca(ctx, model):
    avail = {d for d in TRUE_K if saved(ctx.clusters, d, model, TRUE_K[d])}
    dss = []  # datasets that can be drawn
    for ds in pick(avail, ctx.args.datasets, 6):
        p = data_path(ds, ctx.data)
        if p.exists() and np.fromfile(p, dtype="<i4", count=2)[1] >= 2:  # PCA needs D >= 2
            dss.append(ds)
    need(dss, "no saved run with D >= 2 and its data file")
    fig, axs = plt.subplots(2, len(dss), figsize=(2.7 * len(dss), 5.2), squeeze=False)
    for j, ds in enumerate(dss):
        N, D = np.fromfile(data_path(ds, ctx.data), dtype="<i4", count=2)  # header of the .bin
        # memory map: the file is not loaded in RAM, only the sampled rows are read
        X = np.memmap(data_path(ds, ctx.data), dtype="<f8", mode="r", offset=8, shape=(N, D))
        lab = saved(ctx.clusters, ds, model, TRUE_K[ds])[0]  # cluster found for every row
        xy, idx = project_pca(X, None, ctx.args.max_points)
        scatter_classes(axs[0, j], xy, lab[idx])
        axs[0, j].set_title(f"{ds}\nK={TRUE_K[ds]}, N={N}, D={D}", fontsize=8)
        truth = true_labels(ds, ctx.data)  # same sample and same projection: the panels are comparable
        if truth is not None and len(truth) == N:
            scatter_classes(axs[1, j], xy, truth[idx])
        else:
            axs[1, j].axis("off")
            axs[1, j].text(0.5, 0.5, "no true labels", ha="center", va="center", color=MISSING)
    axs[0, 0].set_ylabel("found clusters")
    axs[1, 0].set_ylabel("true classes")
    fig.suptitle(f"C3  PCA 2D, {model} (random sample of at most {ctx.args.max_points} rows)", y=1.0)
    save(fig, ctx.out, f"C3-pca-{model}", ctx.fmts)


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
    fig, axs = plt.subplots(1, len(dss), figsize=(4.6 * len(dss), 3.6), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        # clusters ordered by weight, the largest on top
        par = saved(ctx.clusters, ds, model, TRUE_K[ds])[1].sort_values("pi", ascending=False)
        # M: one row per cluster, one column per variable (the "mu_*" columns of params.csv)
        M = par[[c for c in par.columns if c.startswith("mu_")]].to_numpy()
        lim = max(1e-6, min(3.0, np.abs(M).max()))  # symmetric around 0: 0 is the neutral color
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.grid(False)
        names = variable_names(ds, M.shape[1], ctx.data)
        ax.set_xticks(range(M.shape[1]), names, rotation=90, fontsize=7)
        ax.set_yticks(range(M.shape[0]), [f"{int(c)} ({100 * p:.0f}%)" for c, p in zip(par.cluster, par.pi)],
                      fontsize=7)
        ax.set_ylabel("cluster (weight)")
        ax.set_title(ds)
        fig.colorbar(im, ax=ax, label="mean (standardized)")
    fig.suptitle(f"C4  cluster centroids, {model}", y=1.02)
    save(fig, ctx.out, f"C4-centroids-{model}", ctx.fmts)


def c5_sizes(ctx, model):
    avail = {d for d in TRUE_K if saved(ctx.clusters, d, model, TRUE_K[d])}
    dss = pick(avail, ctx.args.datasets, 6)
    need(dss, "no saved run")
    fig, axs = plt.subplots(1, len(dss), figsize=(2.6 * len(dss), 2.8), squeeze=False)
    for ax, ds in zip(axs[0], dss):
        lab = saved(ctx.clusters, ds, model, TRUE_K[ds])[0]
        sizes = np.bincount(lab, minlength=TRUE_K[ds])
        order = np.argsort(-sizes)  # largest first; the color follows the cluster index
        ax.bar(range(len(sizes)), 100 * sizes[order] / sizes.sum(), color=class_color(order),
               edgecolor=TEXT, linewidth=0.6)
        ax.set_xticks(range(len(sizes)), [str(c) for c in order], fontsize=7)
        ax.set_xlabel("cluster")
        ax.set_title(ds, fontsize=8)
    axs[0, 0].set_ylabel("rows (%)")
    fig.suptitle(f"C5  cluster sizes, {model}", y=1.03)
    save(fig, ctx.out, f"C5-sizes-{model}", ctx.fmts)


def c6_model_choice(ctx, _model=None):
    """BIC(full) - BIC(diagonal) at the true K. Negative = the full covariance fits better even
    after paying for its extra parameters."""
    rows = []  # (dataset, BIC difference)
    for ds in REAL:
        g = ctx.logs[(ctx.logs.dataset == ds) & (ctx.logs.K == TRUE_K[ds])]
        a, b = g[g.model == "diagonal"], g[g.model == "full"]
        if len(a) and len(b):
            rows.append((ds, b.bic.iloc[0] - a.bic.iloc[0]))
    need(rows, "no real dataset with both models")
    rows.sort(key=lambda r: r[1])  # most favorable to the full model on the left
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * len(rows) + 1.5), 3.2))
    ax.bar(range(len(rows)), [r[1] for r in rows], color=GRID, edgecolor=TEXT, linewidth=0.8)
    ax.axhline(0, color=TEXT, lw=0.8)
    # symlog: linear near 0 (|difference| < 100) and logarithmic beyond, because the differences
    # span many orders of magnitude and have both signs
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("BIC(full) - BIC(diagonal)\n(negative: full is better)")
    ax.set_title("C6  diagonal or full covariance, at the true K")
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
    ap.add_argument("--out", type=Path, default=ROOT / "results/plots")
    ap.add_argument("--tables", type=Path, default=ROOT / "results/tables")
    ap.add_argument("--plots", nargs="+", default=list(PLOTS) + ["table"], help="C1..C6 and/or table")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--datasets", nargs="+",
                    default=["cardiotocography", "heart_failure", "eeg_eye_state", "statlog_heart",
                             "synth_clean_p01"],
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

    # ---- draw: C1-C5 once per model, C6 and the table once (they use both models) ----
    for pid in args.plots:
        try:
            if pid == "table":
                summary_table(ctx)
            elif pid == "C6":  # compares the two models: one file
                c6_model_choice(ctx)
            elif pid in PLOTS:
                for model in args.models:
                    try:
                        PLOTS[pid](ctx, model)
                    except LookupError as e:
                        print(f"skip {pid} {model}: {e}")
            else:
                print(f"skip {pid}: unknown")
        except LookupError as e:
            print(f"skip {pid}: {e}")


if __name__ == "__main__":
    main()
