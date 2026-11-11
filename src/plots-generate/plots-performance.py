#!/usr/bin/env python3
"""
plots-performance.py

Performance plots of the report (catalog P1-P11 of doc/plots.md, colors of doc/color.md).
Reads all the logs in results/logs/*.log and writes PDF files in results/plots/.

Usage (from the root of the repository):
    python3 src/plots-generate/plots-performance.py                 # all the plots
    python3 src/plots-generate/plots-performance.py --plots P2 P6   # only some
    python3 src/plots-generate/plots-performance.py --placement scatter:excl --large synth_clean_p16

Where the data comes from
    One log file per PBS job: results/logs/<VERSION>_<TAG>.log, TAG = "seq" (sequential) or
    "<placement>_<N>n<R>r<T>t" (e.g. pack_excl_2n2r4t = pack:excl, 2 nodes, 2 ranks/node,
    4 threads/rank). The placement is NOT in the log block, so it is read from the file name.
    Block format: see include/logger.h.

Rules (doc/plots.md)
    - Time = t_iteration (time per iteration), median over the repetitions, error bar = min-max.
    - Speedup is always against the SEQUENTIAL version of the same model and dataset.
    - For every number of cores the best ranks x threads split is used (P7 shows the splits).
    - One placement fixed for all the plots except P6 (--placement, default pack).
    - Diagonal and full are never on the same plot: one file per model.
    - File names: <ID>-<what>-<model>.pdf

Reference of the optimizations
    opt1 / opt2 are always compared with the PARALLEL BASELINE (baseline-MPI-OpenMP-<model>) on
    the same cores, dataset and placement (P9), NOT with the sequential version. The sequential
    version is only the reference of the speedup (P2/P3) and of the weak-scaling efficiency (P5).

Extra data in the logs (see include/logger.h and include/perfmodel.h)
    P1  sweeps of K and D: results/logs/sweep/*.log, made with "pemc-submit.sh sweep".
        They are a separate folder because the same dataset appears with several K.
    P10 roofline: "flops" and "bytes" are an ANALYTIC count (not hardware counters), so every
        version of the same model is measured with the same yardstick.
    P11 load imbalance: "rank_compute" = compute seconds of every MPI rank.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------
# Style: one role per color (doc/color.md). Never pick colors by hand elsewhere in this file.
# ---------------------------------------------------------------------------------------------
VERSION_COLOR = {"sequential": "#6B7280", "baseline": "#2579ff", "opt1": "#ff2e2e", "opt2": "#38ffca"}
VERSION_MARKER = {"sequential": "s", "baseline": "o", "opt1": "^", "opt2": "D"}
VERSION_ORDER = ["sequential", "baseline", "opt1", "opt2"]

PLACE_COLOR = {"pack": "#627fff", "pack:excl": "#0600be", "scatter": "#ffab76", "scatter:excl": "#af3200"}
PLACE_ORDER = ["pack", "pack:excl", "scatter", "scatter:excl"]

STEP_COLOR = {"E-step": "#6fff2c", "M-step": "#3415ff", "t_comm": "#ff0000", "I/O": "#efc000"}
STEP_COL = {"E-step": "t_estep", "M-step": "t_mstep", "t_comm": "t_comm", "I/O": "t_read"}

TEXT, IDEAL, GRID, MISSING = "#171e26", "#6B7280", "#E5E7EB", "#656B75"
MODELS = ["diagonal", "full"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42,  # real fonts in the PDF (editable, searchable)
})


# ---------------------------------------------------------------------------------------------
# Reading of the logs
# ---------------------------------------------------------------------------------------------
TAG_RE = re.compile(r"^(pack|scatter)(_excl)?_(\d+)n(\d+)r(\d+)t$")
NUMERIC = ["ranks", "threads", "N", "D", "K", "restarts", "iterations", "converged", "loglik",
           "ari", "t_total", "t_read", "t_init", "t_estep", "t_mstep", "t_comm", "t_warmup",
           "flops", "bytes"]


def parse_blocks(path):
    """Yields one dict per '==== run ====' block ('key : value' lines, split at the first ':')."""
    run = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("===="):
            if run:
                yield run
            run = {}
        elif run is not None and ":" in line:
            k, v = line.split(":", 1)
            run[k.strip()] = v.strip()
    if run:
        yield run


def parse_rank_compute(text):
    """'0.61 0.60 0.63' -> array of the compute time of every rank; None if n/a or missing."""
    try:
        v = np.array([float(x) for x in str(text).split()])
    except ValueError:
        return None
    return v if v.size else None


def load_logs(logdir, required=True):
    """Reads every *.log of logdir (not the subfolders) into one row per run."""
    rows = []
    for path in sorted(logdir.glob("*.log")):  # not the pbs/ subfolder (stdout/stderr of the jobs)
        tag = path.stem.split("_", 1)[1] if "_" in path.stem else "seq"
        m = TAG_RE.match(tag)
        for b in parse_blocks(path):
            b["file"] = path.name
            if m:
                b["placement"] = m.group(1) + (":excl" if m.group(2) else "")
                b["nodes"] = int(m.group(3))
            else:  # "seq" or a log without tag: assume a sequential/single-node run
                b["placement"], b["nodes"] = "seq", 1
            rows.append(b)
    if not rows:
        if required:
            sys.exit(f"no run found in {logdir}/*.log")
        return None
    df = pd.DataFrame(rows)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["rc"] = df["rank_compute"].map(parse_rank_compute) if "rank_compute" in df else None
    df["model"] = df["version"].str.rsplit("-", n=1).str[1]
    base = df["version"].str.rsplit("-", n=1).str[0]
    df["label"] = np.where(base.str.contains("sequential"), "sequential", base.str.split("-").str[0])
    df["dataset"] = df["dataset"].map(lambda p: Path(p).stem)
    df["cores"] = (df["ranks"] * df["threads"]).astype(int)
    # time per iteration, recomputed (the log writes n/a when restarts > 1)
    df["t_iter"] = np.where((df["restarts"] == 1) & (df["iterations"] > 0),
                            (df["t_estep"] + df["t_mstep"] + df["t_comm"]) / df["iterations"], np.nan)
    return df


def aggregate(df):
    """Median / min / max over the repetitions of every (version, model, dataset, layout)."""
    key = ["label", "model", "dataset", "K", "D", "placement", "nodes", "ranks", "threads", "cores"]
    g = df.dropna(subset=["t_iter"]).groupby(key)
    a = g["t_iter"].agg(t="median", lo="min", hi="max", reps="count")
    for c in ("t_estep", "t_mstep", "t_comm", "t_read", "N", "flops", "bytes"):
        a[c] = g[c].median()
    return a.reset_index()


def best_split(agg, placement):
    """Placement fixed (sequential always kept); for every core count the best ranks x threads."""
    a = agg[(agg.placement == placement) | (agg.label == "sequential")]
    key = ["label", "model", "dataset", "K", "cores"]
    return a.loc[a.groupby(key)["t"].idxmin()].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def versions_in(d):
    return [v for v in VERSION_ORDER if v in set(d.label)]


def pick_datasets(best, model, large, small):
    d = best[(best.model == model) & (best.label != "sequential")]
    if d.empty:
        return []
    n = d.groupby("dataset")["N"].first().sort_values()
    big = large if large in n.index else n.index[-1]
    sml = small if small in n.index else n.index[0]
    return [big] if big == sml else [big, sml]


def best_version(agg):
    for v in ("opt2", "opt1", "baseline"):
        if v in set(agg.label):
            return v
    return None


def cores_axis(ax, cores):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(cores))))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("cores")


def line(ax, x, y, lo, hi, v):
    ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=VERSION_COLOR[v],
                marker=VERSION_MARKER[v], ms=5, lw=1.5, capsize=2, label=v)


def ideal(ax, x, y, label="ideal"):
    ax.plot(x, y, ls="--", color=IDEAL, lw=1, label=label)


def save(fig, out, name, fmts):
    for f in fmts:
        fig.savefig(out / f"{name}.{f}")
    plt.close(fig)
    print("wrote", out / f"{name}.{fmts[0]}")


def need(cond, why):
    if not cond:
        raise LookupError(why)


# ---------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------
def p2_p3_speedup(ctx, model):
    """P2 speedup (row 1) + P3 efficiency (row 2); columns = large / small dataset."""
    best = ctx.best
    dss = pick_datasets(best, model, ctx.args.large, ctx.args.small)
    need(dss, "no parallel data")
    fig, axs = plt.subplots(2, len(dss), figsize=(3.6 * len(dss), 5.2), squeeze=False)
    for j, ds in enumerate(dss):
        d = best[(best.model == model) & (best.dataset == ds)]
        seq = d[d.label == "sequential"]
        need(not seq.empty, f"no sequential run for {model}/{ds}")
        t_seq = seq.t.iloc[0]
        allc = []
        for v in versions_in(d):
            s = d[d.label == v].sort_values("cores")
            if v == "sequential":
                s = s.assign(cores=1)
            c, sp = s.cores.to_numpy(), t_seq / s.t.to_numpy()
            lo, hi = t_seq / s.hi.to_numpy(), t_seq / s.lo.to_numpy()
            allc += list(c)
            line(axs[0, j], c, sp, lo, hi, v)
            line(axs[1, j], c, sp / c, lo / c, hi / c, v)
        c = np.array(sorted(set(allc)))
        ideal(axs[0, j], c, c)
        ideal(axs[1, j], c, np.ones_like(c))
        axs[0, j].set_yscale("log", base=2)
        axs[0, j].yaxis.set_major_formatter(ScalarFormatter())
        axs[0, j].set_title(f"{ds} (N={int(d.N.max())})")
        axs[0, j].set_ylabel("speedup vs sequential")
        axs[1, j].set_ylabel("efficiency")
        axs[1, j].set_ylim(0, 1.15)
        for r in (0, 1):
            cores_axis(axs[r, j], c)
    axs[0, 0].legend()
    fig.suptitle(f"P2/P3  speedup and efficiency, {model}, placement {ctx.args.placement}", y=1.0)
    save(fig, ctx.out, f"P2-speedup-{model}", ctx.fmts)


def p4_strong(ctx, model):
    best = ctx.best
    dss = pick_datasets(best, model, ctx.args.large, ctx.args.small)
    need(dss, "no parallel data")
    fig, axs = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 3.0), squeeze=False)
    for j, ds in enumerate(dss):
        d = best[(best.model == model) & (best.dataset == ds)]
        allc = []
        for v in versions_in(d):
            s = d[d.label == v].sort_values("cores")
            c = np.ones(len(s), int) if v == "sequential" else s.cores.to_numpy()
            allc += list(c)
            line(axs[0, j], c, s.t.to_numpy(), s.lo.to_numpy(), s.hi.to_numpy(), v)
        seq = d[d.label == "sequential"]
        if not seq.empty:
            c = np.array(sorted(set(allc)))
            ideal(axs[0, j], c, seq.t.iloc[0] / c)
        axs[0, j].set_xscale("log", base=2)
        axs[0, j].set_yscale("log")
        axs[0, j].set_title(f"{ds} (N={int(d.N.max())})")
        axs[0, j].set_ylabel("time / iteration (s)")
        cores_axis(axs[0, j], allc)
    axs[0, 0].legend()
    fig.suptitle(f"P4  strong scaling, {model}, placement {ctx.args.placement}", y=1.04)
    save(fig, ctx.out, f"P4-strong-{model}", ctx.fmts)


def p5_weak(ctx, model):
    """Weak scaling: synthetic <family>_pNN has N proportional to NN, run on NN cores."""
    best, fam = ctx.best, ctx.args.family
    d = best[(best.model == model) & best.dataset.str.startswith(fam + "_p")].copy()
    need(not d.empty, f"no {fam}_pNN dataset")
    d["p"] = d.dataset.str.extract(r"_p(\d+)$")[0].astype(int)
    ref = d[(d.label == "sequential") & (d.p == 1)]
    need(not ref.empty, f"no sequential {fam}_p01")
    t1 = ref.t.iloc[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    cs = [1]
    for v in versions_in(d):
        s = d[(d.label == v) & (d.cores == d.p)].sort_values("cores")  # N per core constant
        if v == "sequential":
            s = s.assign(cores=1)
        if s.empty:
            continue
        c = s.cores.to_numpy()
        cs += list(c)
        line(ax, c, t1 / s.t.to_numpy(), t1 / s.hi.to_numpy(), t1 / s.lo.to_numpy(), v)
    ideal(ax, sorted(set(cs)), np.ones(len(set(cs))))
    cores_axis(ax, cs)
    ax.set_ylabel("weak-scaling efficiency")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"P5  weak scaling, {model}, {fam}, placement {ctx.args.placement}")
    ax.legend()
    save(fig, ctx.out, f"P5-weak-{model}", ctx.fmts)


def p6_placement(ctx, model):
    """Grouped bars: nodes x 4 placements. Best version, large dataset."""
    a = ctx.agg
    v = ctx.args.version or best_version(a)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = a[(a.label == v) & (a.model == model) & (a.dataset == ds[0]) & (a.placement != "seq")]
    need(not d.empty, f"no data for {v}/{model}/{ds[0]}")
    # same number of cores per node is NOT guaranteed: for each (placement, nodes) take the best split
    d = d.loc[d.groupby(["placement", "nodes"])["t"].idxmin()]
    nodes = sorted(d.nodes.unique())
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    w = 0.2
    for i, pl in enumerate(PLACE_ORDER):
        s = d[d.placement == pl].set_index("nodes").reindex(nodes)
        x = np.arange(len(nodes)) + (i - 1.5) * w
        err = [(s.t - s.lo).fillna(0), (s.hi - s.t).fillna(0)]
        ax.bar(x, s.t, w, yerr=err, color=PLACE_COLOR[pl], edgecolor=TEXT, linewidth=0.6, capsize=2,
               hatch="//" if "excl" in pl else None, label=pl)
    ax.set_xticks(range(len(nodes)), [str(n) for n in nodes])
    ax.set_xlabel("nodes")
    ax.set_ylabel("time / iteration (s)")
    ax.set_title(f"P6  placement, {v} {model}, {ds[0]}")
    ax.legend(ncols=2)
    save(fig, ctx.out, f"P6-placement-{model}", ctx.fmts)


def p7_heatmap(ctx, model):
    """Heatmap ranks x threads on ONE node (best version, large dataset)."""
    a = ctx.agg
    v = ctx.args.version or best_version(a)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = a[(a.label == v) & (a.model == model) & (a.dataset == ds[0]) & (a.nodes == 1)
          & (a.placement.isin([ctx.args.placement, ctx.args.placement.split(":")[0]]))]
    need(not d.empty, "no single-node runs")
    pv = d.pivot_table(index="ranks", columns="threads", values="t", aggfunc="min").sort_index(ascending=False)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(MISSING)
    fig, ax = plt.subplots(figsize=(1.0 + 0.8 * pv.shape[1], 0.9 + 0.6 * pv.shape[0]))
    im = ax.imshow(np.ma.masked_invalid(pv.values), cmap=cmap, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [str(c) for c in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [str(r) for r in pv.index])
    ax.grid(False)
    lo, hi = np.nanmin(pv.values), np.nanmax(pv.values)
    for (i, j), val in np.ndenumerate(pv.values):
        if not np.isnan(val):  # white text on dark cells, black on light ones
            ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=8,
                    color="white" if (val - lo) / (hi - lo + 1e-12) < 0.55 else "black")
    ax.set_xlabel("threads per rank")
    ax.set_ylabel("MPI ranks")
    ax.set_title(f"P7  {v} {model}, {ds[0]}, 1 node")
    fig.colorbar(im, ax=ax, label="time / iteration (s)")
    save(fig, ctx.out, f"P7-ranks-threads-{model}", ctx.fmts)


def p8_breakdown(ctx, model):
    """Stacked bars E-step / M-step / t_comm / I/O. One panel per version, x = cores."""
    best = ctx.best
    d = best[best.model == model]
    vs = versions_in(d)
    need(vs, "no data")
    ds = pick_datasets(best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = d[d.dataset == ds[0]]
    fig, axs = plt.subplots(1, len(vs), figsize=(3.2 * len(vs), 3.0), squeeze=False, sharey=True)
    for ax, v in zip(axs[0], vs):
        s = d[d.label == v].sort_values("cores")
        x = np.arange(len(s))
        bottom = np.zeros(len(s))
        for name, col in STEP_COL.items():
            y = s[col].fillna(0).to_numpy()
            ax.bar(x, y, 0.65, bottom=bottom, color=STEP_COLOR[name], edgecolor=TEXT, linewidth=0.5,
                   label=name)
            bottom += y
        ax.set_xticks(x, [str(c) for c in s.cores])
        ax.set_xlabel("cores")
        ax.set_title(v)
    axs[0, 0].set_ylabel("time, whole run (s)")
    axs[0, 0].legend()
    fig.suptitle(f"P8  where the time goes, {model}, {ds[0]}, placement {ctx.args.placement}", y=1.04)
    save(fig, ctx.out, f"P8-breakdown-{model}", ctx.fmts)


def p9_opt_gain(ctx, model):
    """P9: opt1/opt2 time divided by the time of the PARALLEL BASELINE (baseline-MPI-OpenMP, = 1),
    at the largest core count shared by the versions, same dataset and placement. The sequential
    version is NOT the reference here. Below 1 = faster than the baseline."""
    best = ctx.best
    d = best[best.model == model]
    need("baseline" in set(d.label), "no baseline")
    opts = [v for v in ("opt1", "opt2") if v in set(d.label)]
    need(opts, "no opt1/opt2 data yet")
    base = d[d.label == "baseline"]
    cores = max(set(base.cores) & set(d[d.label.isin(opts)].cores))
    d = d[d.cores == cores]
    dss = sorted(set(d[d.label == "baseline"].dataset))
    fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(dss) + 2), 3.2))
    w = 0.8 / len(opts)
    for i, v in enumerate(opts):
        y = []
        for ds in dss:
            tb = d[(d.label == "baseline") & (d.dataset == ds)].t
            tv = d[(d.label == v) & (d.dataset == ds)].t
            y.append(tv.iloc[0] / tb.iloc[0] if len(tb) and len(tv) else np.nan)
        ax.bar(np.arange(len(dss)) + (i - (len(opts) - 1) / 2) * w, y, w, color=VERSION_COLOR[v],
               edgecolor=TEXT, linewidth=0.6, label=v)
    ax.axhline(1, ls="--", color=IDEAL, lw=1, label="baseline")
    ax.set_xticks(range(len(dss)), dss, rotation=45, ha="right")
    ax.set_ylabel("time / baseline time")
    ax.set_title(f"P9  gain over the baseline, {model}, {cores} cores, placement {ctx.args.placement}")
    ax.legend()
    save(fig, ctx.out, f"P9-opt-gain-{model}", ctx.fmts)

def p1_sweeps(ctx, model):
    """P1: time per iteration vs K (panel a) and vs D (panel b), log-log, one line per version.
    Data: results/logs/sweep/ (one fixed layout, see pemc-submit.sh sweep)."""
    sw = ctx.sweep
    need(sw is not None, "no sweep logs (run: pemc-submit.sh sweep)")
    d = sw[sw.model == model]
    need(not d.empty, f"no sweep run for {model}")
    is_d = d.dataset.str.startswith("synth_sweep_d")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, sub, col, name, slope in ((axs[0], d[~is_d], "K", "K (clusters)", 1.0),
                                      (axs[1], d[is_d], "D", "D (variables)",
                                       2.0 if model == "full" else 1.0)):
        first = None
        for v in versions_in(sub):
            s_ = sub[sub.label == v]
            s_ = s_.loc[s_.groupby(col)["t"].idxmin()].sort_values(col)  # one layout: 1 row per x
            if s_.empty:
                continue
            line(ax, s_[col].to_numpy(), s_.t.to_numpy(), s_.lo.to_numpy(), s_.hi.to_numpy(), v)
            first = first or (s_[col].iloc[0], s_.t.iloc[0], s_[col].to_numpy())
        if first:  # reference slope anchored at the first point of the first version
            x0, y0, xs = first
            ideal(ax, xs, y0 * (xs / x0) ** slope,
                  label=f"$\\propto {col}^{{{slope:g}}}$" if slope != 1 else f"$\\propto {col}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(name)
        ax.set_ylabel("time / iteration (s)")
        ax.legend()
    axs[0].set_title("(a) vs K")
    axs[1].set_title("(b) vs D")
    fig.suptitle(f"P1  cost of an iteration, {model}", y=1.03)
    save(fig, ctx.out, f"P1-sweep-{model}", ctx.fmts)


def p10_roofline(ctx, model):
    """P10 roofline of ONE node: one point per version. x = FLOP / byte, y = GFLOP/s.
    FLOP and bytes are the analytic count of include/perfmodel.h. The ceilings are the measured
    peaks of the node, given with --peak-gflops and --peak-bw (e.g. from a STREAM run and from
    cores x GHz x FLOP/cycle); without them only the points are drawn."""
    a = ctx.agg
    ds = pick_datasets(ctx.best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = a[(a.model == model) & (a.dataset == ds[0]) & (a.nodes == 1)
          & (a.placement.isin([ctx.args.placement, "seq"])) & a.flops.notna() & a.bytes.notna()]
    need(not d.empty, "no single-node run with flops/bytes (re-run with the new binaries)")
    d = d.loc[d.groupby("label")["t"].idxmin()]  # best split of every version
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs = np.logspace(-1, 2, 100)
    if ctx.args.peak_gflops and ctx.args.peak_bw:
        ax.plot(xs, np.minimum(ctx.args.peak_gflops, ctx.args.peak_bw * xs), color=TEXT, lw=1.2,
                label="roofline")
        ax.axhline(ctx.args.peak_gflops, color=IDEAL, ls="--", lw=0.8)
    for _, r in d.iterrows():
        v = r.label
        ax.plot(r.flops / r.bytes, r.flops / (r.t_estep + r.t_mstep) / 1e9, ls="",
                marker=VERSION_MARKER[v], color=VERSION_COLOR[v], ms=8, mec=TEXT, mew=0.6,
                label=f"{v} ({int(r.cores)} cores)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("performance (GFLOP/s)")
    ax.set_title(f"P10  roofline, {model}, {ds[0]}, 1 node")
    ax.legend()
    save(fig, ctx.out, f"P10-roofline-{model}", ctx.fmts)


def p11_imbalance(ctx, model):
    """P11 load imbalance: mean compute time per rank (bar) with min-max (whisker) for every
    ranks x threads layout of the best version. The % above a bar is (max / mean - 1): the time
    the other ranks wait for the slowest one (it is hidden inside t_comm in the other plots)."""
    df = ctx.df
    v = ctx.args.version or best_version(ctx.agg)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = df[(df.label == v) & (df.model == model) & (df.dataset == ds[0]) & (df.ranks > 1)
           & (df.placement == ctx.args.placement) & df.rc.notna()]
    need(not d.empty, "no multi-rank run with rank_compute (re-run with the new binaries)")
    d = d.assign(lo=d.rc.map(np.min), mean=d.rc.map(np.mean), hi=d.rc.map(np.max))
    g = d.groupby(["nodes", "ranks", "threads"])[["lo", "mean", "hi"]].median().reset_index()
    g = g.sort_values(["nodes", "ranks", "threads"])
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(max(4.2, 0.9 * len(g) + 1.5), 3.2))
    ax.bar(x, g["mean"], 0.6, color=VERSION_COLOR[v], edgecolor=TEXT, linewidth=0.6,
           yerr=[g["mean"] - g.lo, g.hi - g["mean"]], capsize=3)
    for xi, r in zip(x, g.itertuples()):
        ax.text(xi, r.hi, f"{100 * (r.hi / r.mean - 1):.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{int(r.nodes)}n\n{int(r.ranks)}x{int(r.threads)}" for r in g.itertuples()])
    ax.set_xlabel("nodes / ranks x threads")
    ax.set_ylabel("compute time per rank (s)")
    ax.set_title(f"P11  load imbalance, {v} {model}, {ds[0]}")
    ax.set_ylim(0, g.hi.max() * 1.15)
    save(fig, ctx.out, f"P11-imbalance-{model}", ctx.fmts)


PLOTS = {"P1": p1_sweeps, "P2": p2_p3_speedup, "P3": p2_p3_speedup, "P4": p4_strong, "P5": p5_weak,
         "P6": p6_placement, "P7": p7_heatmap, "P8": p8_breakdown, "P9": p9_opt_gain, "P10": p10_roofline,
         "P11": p11_imbalance}


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/plots")
    ap.add_argument("--plots", nargs="+", default=["P1", "P2", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"],
                    help="IDs to draw (P3 is the second row of P2)")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--placement", default="pack", choices=PLACE_ORDER,
                    help="placement fixed for every plot except P6 (the best one found with P6)")
    ap.add_argument("--version", help="version for P6/P7 (default: the most optimized one present)")
    ap.add_argument("--large", default="synth_clean_p16", help="large dataset (auto: max N if missing)")
    ap.add_argument("--small", default="cardiotocography", help="small dataset (auto: min N if missing)")
    ap.add_argument("--family", default="synth_clean", help="synthetic family for the weak scaling P5")
    ap.add_argument("--peak-gflops", type=float,
                    help="P10: peak GFLOP/s of ONE node (the flat roof of the roofline)")
    ap.add_argument("--peak-bw", type=float,
                    help="P10: memory bandwidth of ONE node in GB/s (e.g. STREAM triad)")
    ap.add_argument("--fmt", nargs="+", default=["pdf"], help="output formats (pdf, png, ...)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load_logs(args.logs)
    agg = aggregate(df)
    best = best_split(agg, args.placement)
    sweep_df = load_logs(args.logs / "sweep", required=False)  # P1 only
    sweep = aggregate(sweep_df) if sweep_df is not None else None
    ctx = argparse.Namespace(args=args, df=df, agg=agg, best=best, sweep=sweep, out=args.out,
                             fmts=args.fmt)
    print(f"{len(df)} runs, versions: {sorted(set(df.label))}, placement: {args.placement}")

    done = set()
    for pid in args.plots:
        fn = PLOTS.get(pid)
        if fn is None:
            print(f"skip {pid}: not implemented")
            continue
        for model in args.models:
            if (fn, model) in done:
                continue
            done.add((fn, model))
            try:
                fn(ctx, model)
            except LookupError as e:
                print(f"skip {pid} {model}: {e}")


if __name__ == "__main__":
    main()
