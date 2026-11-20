#!/usr/bin/env python3
"""
plots-performance.py

Performance plots of the report (catalog P1-P11 of doc/plots.md).
Reads all the logs in results/logs/<model>/<stage>/*.log and writes PDF files in results/plots/performance/.

Usage (from the root of the repository):
    python3 src/plots-generate/plots-performance.py                 # all the plots
    python3 src/plots-generate/plots-performance.py --plots P2 P6   # only some
    python3 src/plots-generate/plots-performance.py --placement scatter:excl --large synth_clean_p16

Where the data comes from
    One log file per PBS job: results/logs/<model>/<stage>/<VERSION>_<TAG>.log
    (model = diagonal | full, stage = sequential | baseline | opt1 | opt2), TAG = "seq" (sequential) or
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
    - Clarity (same rules as plots-cluster.py): bold title in plain words + one line with model,
      dataset and placement; ONE legend above the panels with names in plain words ("pack,
      exclusive node", "MPI communication + waiting for the other ranks"); axes with their unit.
    - Size: drawn at the final size of the report, FIG_W = 7 in (include with width=\textwidth)
      or HALF_W = 4.4 in for the single-panel plots P5, P6, P7, P10 (width=0.62\textwidth),
      so the labels stay 7-8 pt in the PDF.
    - Real and synthetic datasets in the same figure (P2/P3, P4, P9): two blocks, the REAL
      datasets on top and the SYNTHETIC ones below, each with its own header.

Reference of the optimizations
    opt1 / opt2 are always compared with the PARALLEL BASELINE (baseline-MPI-OpenMP-<model>) on
    the same cores, dataset and placement (P9), NOT with the sequential version. The sequential
    version is only the reference of the speedup (P2/P3). The weak-scaling efficiency (P5) of
    every version uses the SAME version on 1 core as reference, so it shows the scaling only.

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
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, ScalarFormatter

# Root of the repository (this file is in <root>/src/plots-generate/): the default input and
# output folders are relative to it, so the script works from any current directory
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------
# Style: one role per color. Never pick colors by hand elsewhere in this file.
# ---------------------------------------------------------------------------------------------
VERSION_COLOR = {"sequential": "#6B7280", "baseline": "#2579ff", "opt1": "#ff2e2e", "opt2": "#38ffca"}
VERSION_MARKER = {"sequential": "s", "baseline": "o", "opt1": "^", "opt2": "D"}
VERSION_ORDER = ["sequential", "baseline", "opt1", "opt2"]

# pack = green, scatter = purple; light = shared node(s), dark = exclusive node(s) (also hatched)
PLACE_COLOR = {"pack": "#a7e3a1", "pack:excl": "#1e6b2a", "scatter": "#d6c3f5", "scatter:excl": "#4c2a85"}
PLACE_ORDER = ["pack", "pack:excl", "scatter", "scatter:excl"]

STEP_COLOR = {"E-step": "#6fff2c", "M-step": "#3415ff", "t_comm": "#ff0000", "I/O": "#efc000"}
STEP_COL = {"E-step": "t_estep", "M-step": "t_mstep", "t_comm": "t_comm", "I/O": "t_read"}

TEXT, IDEAL, GRID, MISSING = "#171e26", "#6B7280", "#E5E7EB", "#656B75"
MODELS = ["diagonal", "full"]

# Same look in every figure: serif font, light gray grid under the data, no top/right frame,
# vector output with real fonts (fonttype 42) so the text of the PDF can be searched and edited
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
# Tag of a parallel log file name, e.g. "pack_excl_2n2r4t" = placement pack:excl, 2 nodes, 2 ranks
# per node, 4 threads per rank (written by src/script/pemc-job.sh). Groups: placement, "_excl" or
# nothing, nodes, ranks per node, threads.
TAG_RE = re.compile(r"^(pack|scatter)(_excl)?_(\d+)n(\d+)r(\d+)t$")
# Log keys that are numbers (everything else stays text); "n/a" values become NaN
NUMERIC = ["ranks", "threads", "N", "D", "K", "restarts", "iterations", "converged", "loglik",
           "ari", "t_total", "t_read", "t_init", "t_estep", "t_mstep", "t_comm", "t_warmup",
           "flops", "bytes"]


def parse_blocks(path):
    """Yields one dict per '==== run ====' block ('key : value' lines, split at the first ':')."""
    run = None  # block being read (None before the first "====" line)
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("===="):  # a new block starts: hand over the previous one
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


def load_logs(logdir, required=True, pattern="*/*/*.log"):
    """Reads every log matching pattern (relative to logdir) into one row per run.
    Default: results/logs/<model>/<stage>/*.log; not sweep/, cluster/ or pbs/ (they are one level
    only). The sweeps use pattern "*.log" on their own folder."""
    rows = []  # one dict per run, over all the files
    for path in sorted(logdir.glob(pattern)):
        # file name = <VERSION>_<TAG>.log: the version has no "_", so the first "_" splits them.
        # The placement and the number of nodes are NOT in the block: they come from the tag
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
            sys.exit(f"no run found in {logdir}/{pattern}")
        return None
    df = pd.DataFrame(rows)
    # text -> numbers (errors="coerce": "n/a" or a missing key becomes NaN)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    # compute time of every rank, as an array per run (P11)
    df["rc"] = df["rank_compute"].map(parse_rank_compute) if "rank_compute" in df else None
    # "baseline-MPI-OpenMP-full" -> model "full"; base = "baseline-MPI-OpenMP"
    df["model"] = df["version"].str.rsplit("-", n=1).str[1]
    base = df["version"].str.rsplit("-", n=1).str[0]
    # label = the version as shown in the plots: sequential | baseline | opt1 | opt2
    df["label"] = np.where(base.str.contains("sequential"), "sequential", base.str.split("-").str[0])
    df["dataset"] = df["dataset"].map(lambda p: Path(p).stem)  # path -> file name without ".bin"
    # total cores of the run (ranks is the TOTAL number of MPI processes, threads is per rank)
    df["cores"] = (df["ranks"] * df["threads"]).astype(int)
    # time per iteration, recomputed (the log writes n/a when restarts > 1)
    df["t_iter"] = np.where((df["restarts"] == 1) & (df["iterations"] > 0),
                            (df["t_estep"] + df["t_mstep"] + df["t_comm"]) / df["iterations"], np.nan)
    return df


def aggregate(df):
    """Median / min / max over the repetitions of every (version, model, dataset, layout)."""
    key = ["label", "model", "dataset", "K", "D", "placement", "nodes", "ranks", "threads", "cores"]
    g = df.dropna(subset=["t_iter"]).groupby(key)  # runs without t_iter (restarts > 1) are skipped
    # t = median time per iteration, lo / hi = min / max (error bars), reps = number of repetitions
    a = g["t_iter"].agg(t="median", lo="min", hi="max", reps="count")
    for c in ("t_estep", "t_mstep", "t_comm", "t_read", "N", "flops", "bytes"):
        a[c] = g[c].median()
    return a.reset_index()


def best_split(agg, placement):
    """Placement fixed (sequential always kept); for every core count the best ranks x threads."""
    a = agg[(agg.placement == placement) | (agg.label == "sequential")]
    # one row per (version, model, dataset, K, cores): the one with the smallest time among the
    # ranks x threads splits that give those cores
    key = ["label", "model", "dataset", "K", "cores"]
    return a.loc[a.groupby(key)["t"].idxmin()].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def versions_in(d):
    """Versions present in the table d, in the fixed plotting order (sequential first)."""
    return [v for v in VERSION_ORDER if v in set(d.label)]


def pick_datasets(best, model, large, small):
    """[large, small] datasets to draw for a model. The requested names are used if they are in
    the data, otherwise the dataset with the largest / smallest N among the parallel runs.
    Returns one name only if large and small coincide, and [] if there is no parallel run."""
    d = best[(best.model == model) & (best.label != "sequential")]
    if d.empty:
        return []
    n = d.groupby("dataset")["N"].first().sort_values()  # datasets ordered by size
    big = large if large in n.index else n.index[-1]
    sml = small if small in n.index else n.index[0]
    return [big] if big == sml else [big, sml]


def best_version(agg):
    """The most optimized version present in the data (opt2 > opt1 > baseline), or None."""
    for v in ("opt2", "opt1", "baseline"):
        if v in set(agg.label):
            return v
    return None


# ---- names in plain words, for legends and titles (the logs keep the short ones) ----
VERSION_LABEL = {"sequential": "sequential (1 core)", "baseline": "baseline MPI + OpenMP",
                 "opt1": "opt1 (faster computation)", "opt2": "opt2 (less communication)"}
PLACE_LABEL = {"pack": "pack, shared node", "pack:excl": "pack, exclusive node",
               "scatter": "scatter, shared nodes", "scatter:excl": "scatter, exclusive nodes"}
STEP_LABEL = {"E-step": "E-step (computation)", "M-step": "M-step (computation)",
              "t_comm": "MPI communication + waiting for the other ranks", "I/O": "reading the dataset"}
MODEL_TXT = {"diagonal": "diagonal covariance", "full": "full covariance"}

# Figures are drawn at their FINAL size, so LaTeX does not shrink them and the 7-8 pt labels stay
# 7-8 pt in the PDF: FIG_W = width of the text of the report (IEEEtran, one column), to include
# with width=\textwidth; HALF_W = single-panel plots (P5, P6, P7, P10), width=0.62\textwidth
FIG_W, HALF_W = 7.0, 4.4


def place_txt(p):
    """'placement pack, shared node' for the subtitles."""
    return f"placement {PLACE_LABEL.get(p, p)}"


def title(fig, text, sub=None):
    """Bold title in plain words (what the figure shows) and an optional second line in normal
    font (model, dataset, placement), ABOVE everything else of the figure (legend included):
    from the top, title, subtitle, legend, panels."""
    # lines wrapped to the width of the figure (about 12 bold 10 pt and 15 normal 8 pt
    # characters per inch): a long title must not make the figure wider than the page
    text = textwrap.fill(text, int(fig.get_figwidth() * 12))
    sub = textwrap.fill(sub, int(fig.get_figwidth() * 15)) if sub else sub
    fig.canvas.draw()  # final layout: positions and sizes are known
    r = fig.canvas.get_renderer()
    top = fig.get_tightbbox(r).y1 / fig.get_figheight()  # top of the drawn content (fraction)
    if sub:
        t = fig.text(0.5, top + 0.008, sub, ha="center", va="bottom", fontsize=8, color=TEXT,
                     linespacing=1.3)
        top = t.get_window_extent(r).transformed(fig.transFigure.inverted()).y1
    fig.text(0.5, top + 0.006, text, ha="center", va="bottom", fontsize=10, fontweight="bold",
             linespacing=1.3)


def top_legend(fig, handles, labels, ncol=None):
    """ONE legend for the whole figure, boxed, above the panels (never on the data). It sits
    outside the area of the panels: savefig(bbox_inches="tight") keeps it in the file."""
    fig._top_legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                                 ncol=ncol or len(labels), frameon=True, fontsize=7.5,
                                 handlelength=2.2, columnspacing=1.4, edgecolor=GRID)


def legend_from(fig, axs, ncol=None):
    """top_legend with the entries of all the panels, each label once, in order of appearance."""
    seen = {}
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    # matplotlib lists plain lines before error bars: the reference lines (ideal, reference,
    # baseline = 1) would come first, so they are moved to the end
    ref = ("ideal", "reference", "baseline MPI + OpenMP (=")
    labels = sorted(seen, key=lambda l: l.startswith(ref))  # stable: the rest keeps its order
    top_legend(fig, [seen[l] for l in labels], labels, ncol)


def panel_name(ax, text):
    """Name of a panel, above it, bold."""
    ax.set_title(text, fontsize=8.5, fontweight="bold")


def panel_grid(n, row_h, ncols=2):
    """n panels in a grid with ncols columns (4 panels = 2 x 2), FIG_W wide, row_h inches per
    row. Returns (fig, list of the n axes, ncols); empty cells of the last row are hidden."""
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceiling division
    fig, axs = plt.subplots(nrows, ncols, figsize=(FIG_W, row_h * nrows + 0.2), squeeze=False,
                            layout="constrained")
    axs = list(axs.ravel())
    for ax in axs[n:]:
        ax.set_visible(False)
    return fig, axs[:n], ncols


def is_synth(ds):
    """True for the synthetic datasets (generated), False for the real ones."""
    return ds.startswith("synth_")


def kind_of(ds):
    """'Real dataset' or 'Synthetic dataset', for the block headers."""
    return "Synthetic dataset" if is_synth(ds) else "Real dataset"


def blocks(fig, n, heights=None):
    """The figure split in n blocks stacked vertically (subfigures): the REAL datasets go in the
    top block(s), the SYNTHETIC ones below, every block with its own header (block_header)."""
    subs = fig.subfigures(n, 1, height_ratios=heights)
    return [subs] if n == 1 else list(subs)


def block_header(sf, head):
    """Header of a block of panels (e.g. 'Real dataset: cardiotocography'): bold, on the left."""
    sf.suptitle(head, x=0.0, ha="left", fontsize=9, fontweight="bold", color=TEXT)


def nodes_of(d):
    """{cores: nodes} of the parallel runs in d (with the fixed layouts every core count runs
    on one number of nodes: 8 cores = 1 node, 16 = 2, 32 = 4)."""
    p = d[d.label != "sequential"]
    return {int(c): int(n) for c, n in zip(p.cores, p.nodes)}


def cores_axis(ax, cores, nodes=None):
    """x axis in cores: log base 2, with ticks exactly at the core counts that were measured.
    nodes {cores: nodes}: the number of nodes is written under every core count, because the
    step from 1 to 2 nodes (network communication) explains most drops of the curves."""
    ticks = sorted(set(int(c) for c in cores))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    if nodes:
        ax.set_xticks(ticks, [f"{c}\n{nodes[c]} node{'s' if nodes[c] > 1 else ''}" if c in nodes
                              else str(c) for c in ticks], fontsize=6.8)
        ax.set_xlabel("number of cores (nodes)")
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("number of cores")


def line(ax, x, y, lo, hi, v):
    """Line of version v (its color and marker) with an error bar from lo to hi at every point."""
    ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=VERSION_COLOR[v],
                marker=VERSION_MARKER[v], ms=4.5, lw=1.4, capsize=2, mec=TEXT, mew=0.4,
                label=VERSION_LABEL[v])


def ideal(ax, x, y, label="ideal (perfect scaling)"):
    """Reference line (ideal speedup, efficiency 1, ...): gray and dashed, never a version color."""
    ax.plot(x, y, ls="--", color=IDEAL, lw=1, label=label)


def save(fig, out, name, fmts):
    """Writes the figure in every requested format (pdf, png, ...) and frees its memory."""
    for f in fmts:
        fig.savefig(out / f"{name}.{f}")
    plt.close(fig)
    print("wrote", out / f"{name}.{fmts[0]}")


def need(cond, why):
    """Stops the current plot with a readable reason if its data is missing (main() prints it
    and goes on with the next plot)."""
    if not cond:
        raise LookupError(why)


# ---------------------------------------------------------------------------------------------
# Plots. Every figure: bold title in plain words + one line with model / dataset / placement,
# ONE legend above the panels, axes named with their unit (doc/plots.md, section 5)
# ---------------------------------------------------------------------------------------------
def p2_p3_speedup(ctx, model):
    """P2 speedup (left) + P3 efficiency (right), one block per dataset: the REAL (small) dataset
    on top, the SYNTHETIC (large) one below."""
    best = ctx.best
    dss = sorted(pick_datasets(best, model, ctx.args.large, ctx.args.small), key=is_synth)
    need(dss, "no parallel data")
    fig = plt.figure(figsize=(FIG_W, 2.45 * len(dss) + 0.2), layout="constrained")
    left = []
    for sf, ds in zip(blocks(fig, len(dss)), dss):
        ax_s, ax_e = sf.subplots(1, 2)
        left.append(ax_s)
        d = best[(best.model == model) & (best.dataset == ds)]
        seq = d[d.label == "sequential"]
        need(not seq.empty, f"no sequential run for {model}/{ds}")
        t_seq = seq.t.iloc[0]  # time per iteration of the sequential run: reference of the speedup
        allc = []  # every core count drawn in this block (for the ticks and the ideal line)
        eff_max = 1.0  # highest efficiency drawn (above 1 = superlinear speedup)
        for v in versions_in(d):
            s = d[d.label == v].sort_values("cores")
            if v == "sequential":
                s = s.assign(cores=1)
            c, sp = s.cores.to_numpy(), t_seq / s.t.to_numpy()
            # a larger time gives a smaller speedup: the error bar limits swap
            lo, hi = t_seq / s.hi.to_numpy(), t_seq / s.lo.to_numpy()
            allc += list(c)
            line(ax_s, c, sp, lo, hi, v)
            line(ax_e, c, sp / c, lo / c, hi / c, v)  # efficiency = speedup / cores
            eff_max = max(eff_max, np.nanmax(hi / c))
        c = np.array(sorted(set(allc)))
        ideal(ax_s, c, c)                 # ideal speedup = number of cores
        ideal(ax_e, c, np.ones_like(c))   # ideal efficiency = 1
        ax_s.set_yscale("log", base=2)
        ax_s.yaxis.set_major_formatter(ScalarFormatter())
        # never cut a point: an efficiency above 1 must stay visible (opt1 also changes the
        # computation, blocks of rows and SIMD, so it can beat the sequential version per core)
        ax_e.set_ylim(0, eff_max * 1.1 + 0.05)
        for ax in (ax_s, ax_e):
            cores_axis(ax, c, nodes_of(d))
        ax_s.set_ylabel("speedup over sequential\n(log scale)")
        ax_e.set_ylabel("parallel efficiency\n(speedup / cores)")
        panel_name(ax_s, "speedup")
        panel_name(ax_e, "efficiency")
        block_header(sf, f"{kind_of(ds)}: {ds}  (N = {int(d.N.max()):,} points)")
    legend_from(fig, left, ncol=3)
    title(fig, "Speedup and parallel efficiency against the sequential version",
          f"{MODEL_TXT[model]}, {place_txt(ctx.args.placement)}; "
          "for every core count the fastest ranks × threads split")
    save(fig, ctx.out, f"P2-speedup-{model}", ctx.fmts)


def p4_strong(ctx, model):
    """P4 strong scaling: time per iteration vs cores, the dataset (N) fixed. One block per
    dataset: the REAL (small) one on top, the SYNTHETIC (large) one below."""
    best = ctx.best
    dss = sorted(pick_datasets(best, model, ctx.args.large, ctx.args.small), key=is_synth)
    need(dss, "no parallel data")
    fig = plt.figure(figsize=(FIG_W, 2.3 * len(dss) + 0.2), layout="constrained")
    axs = []
    for sf, ds in zip(blocks(fig, len(dss)), dss):
        ax = sf.subplots()
        axs.append(ax)
        d = best[(best.model == model) & (best.dataset == ds)]
        allc = []
        for v in versions_in(d):
            s = d[d.label == v].sort_values("cores")
            # the sequential version always runs on 1 core
            c = np.ones(len(s), int) if v == "sequential" else s.cores.to_numpy()
            allc += list(c)
            line(ax, c, s.t.to_numpy(), s.lo.to_numpy(), s.hi.to_numpy(), v)
        seq = d[d.label == "sequential"]
        if not seq.empty:
            c = np.array(sorted(set(allc)))
            ideal(ax, c, seq.t.iloc[0] / c, label="ideal (sequential time / cores)")
        ax.set_yscale("log")
        cores_axis(ax, allc, nodes_of(d))
        ax.set_ylabel("time per iteration (s)\n(log scale)")
        block_header(sf, f"{kind_of(ds)}: {ds}  (N = {int(d.N.max()):,} points)")
    legend_from(fig, axs, ncol=3)
    title(fig, "Strong scaling: time per iteration with the size of the dataset fixed",
          f"{MODEL_TXT[model]}, {place_txt(ctx.args.placement)}")
    save(fig, ctx.out, f"P4-strong-{model}", ctx.fmts)


def p5_weak(ctx, model):
    """Weak scaling: synthetic <family>_pNN has N proportional to NN, run on NN cores."""
    best, fam = ctx.best, ctx.args.family
    d = best[(best.model == model) & best.dataset.str.startswith(fam + "_p")].copy()
    need(not d.empty, f"no {fam}_pNN dataset")
    d["p"] = d.dataset.str.extract(r"_p(\d+)$")[0].astype(int)  # size factor = cores of the weak run
    fig, ax = plt.subplots(figsize=(HALF_W, 3.0), layout="constrained")
    cs = [1]  # core counts drawn (for the ticks and the ideal line)
    eff_max = 1.0  # highest efficiency drawn: the y axis never cuts a point
    for v in versions_in(d):
        # Weak-scaling efficiency of a version = its OWN time on 1 core (smallest problem, p01)
        # / its time on p cores with p x the data. The reference is the same version, NOT the
        # sequential one: otherwise a faster algorithm (opt1/opt2 full, ~2x the sequential) would
        # start above 1 and the plot would mix "faster algorithm" with "scales well" (that gain
        # is in P2 and P9). The sequential version has no curve here (it never runs on p > 1).
        if v == "sequential":
            continue
        ref = d[(d.label == v) & (d.p == 1) & (d.cores == 1)]
        if ref.empty:
            print(f"  P5 {model}: {v} has no 1-core run on {fam}_p01, skipped")
            continue
        t1 = ref.t.iloc[0]
        s = d[(d.label == v) & (d.cores == d.p)].sort_values("cores")  # N per core constant
        c = s.cores.to_numpy()
        cs += list(c)
        line(ax, c, t1 / s.t.to_numpy(), t1 / s.hi.to_numpy(), t1 / s.lo.to_numpy(), v)
        eff_max = max(eff_max, np.nanmax(t1 / s.lo.to_numpy()))
    ideal(ax, sorted(set(cs)), np.ones(len(set(cs))), label="ideal (efficiency 1)")
    cores_axis(ax, cs, nodes_of(d[d.cores == d.p]))
    ax.set_xlabel("number of cores p (nodes); dataset with p × the rows")
    ax.set_ylabel("weak-scaling efficiency\n(same version: time on 1 core\n/ time on p cores)")
    ax.set_ylim(0, eff_max * 1.1 + 0.05)
    legend_from(fig, [ax], ncol=2)
    title(fig, "Weak scaling: the data per core is kept constant",
          f"{MODEL_TXT[model]}, {fam}_pNN on NN cores, {place_txt(ctx.args.placement)}")
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
    fig, ax = plt.subplots(figsize=(HALF_W, 3.1), layout="constrained")
    w = 0.2  # width of one bar: 4 bars per group of nodes
    for i, pl in enumerate(PLACE_ORDER):
        # reindex puts NaN where a placement has no run for that number of nodes (e.g. scatter on
        # 1 node does not exist): that bar is simply not drawn
        s = d[d.placement == pl].set_index("nodes").reindex(nodes)
        x = np.arange(len(nodes)) + (i - 1.5) * w
        err = [(s.t - s.lo).fillna(0), (s.hi - s.t).fillna(0)]
        ax.bar(x, s.t, w, yerr=err, color=PLACE_COLOR[pl], edgecolor=TEXT, linewidth=0.6, capsize=2,
               hatch="//" if "excl" in pl else None, label=PLACE_LABEL[pl])
    ax.set_xticks(range(len(nodes)), [f"{n} node" + ("s" if n > 1 else "") for n in nodes])
    ax.set_xlabel("number of nodes (on 1 node, scatter = pack)")
    ax.set_ylabel("time per iteration (s)")
    legend_from(fig, [ax], ncol=2)
    title(fig, "Effect of the PBS placement on the time per iteration",
          f"{VERSION_LABEL[v]}, {MODEL_TXT[model]}, {ds[0]}; bars = median, whiskers = min-max")
    save(fig, ctx.out, f"P6-placement-{model}", ctx.fmts)


def p7_heatmap(ctx, model):
    """P7: the ranks x threads splits of the cores of ONE node (best version, large dataset), one
    bar each: time per iteration (median, whiskers min-max), the fastest one marked. With 8 cores
    the splits are 8x1, 4x2, 2x4, 1x8 (a ranks x threads table would be mostly empty)."""
    a = ctx.agg
    v = ctx.args.version or best_version(a)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = a[(a.label == v) & (a.model == model) & (a.dataset == ds[0]) & (a.nodes == 1)
          & (a.placement.isin([ctx.args.placement, ctx.args.placement.split(":")[0]]))]
    need(not d.empty, "no single-node runs")
    # only the splits of ALL the cores of the node (8): the runs on 1, 2, 4 cores of one node (for
    # the speedup curves) are not splits of the same cores and must not be compared here
    d = d[d.cores == d.cores.max()]
    # one bar per split (several placements for the same split: the fastest), from all-MPI to
    # all-OpenMP
    d = d.loc[d.groupby(["ranks", "threads"])["t"].idxmin()].sort_values("ranks", ascending=False)
    x = np.arange(len(d))
    best_i = int(np.argmin(d.t.to_numpy()))
    fig, ax = plt.subplots(figsize=(HALF_W, 3.0), layout="constrained")
    ax.bar(x, d.t, 0.6, yerr=[d.t - d.lo, d.hi - d.t], capsize=3, color=VERSION_COLOR[v],
           edgecolor=[("#ff2e2e" if i == best_i else TEXT) for i in x],
           linewidth=[(2.2 if i == best_i else 0.6) for i in x])
    for xi, t in zip(x, d.t):  # the value on every bar
        ax.text(xi, t / 2, f"{t:.3g} s", ha="center", va="center", fontsize=7.5, color=TEXT)
    ax.set_xticks(x, [f"{int(r)} ranks\n× {int(t)} thr." for r, t in zip(d.ranks, d.threads)],
                  fontsize=7.5)
    ax.set_xlabel(f"MPI ranks × OpenMP threads per rank ({int(d.cores.iloc[0])} cores, 1 node)")
    ax.set_ylabel("time per iteration (s)\n(lower = better)")
    top_legend(fig, [plt.Rectangle((0, 0), 1, 1, fc=VERSION_COLOR[v], ec="#ff2e2e", lw=2.2),
                     plt.Line2D([], [], color=TEXT, marker="_", ms=9, mew=1.2, lw=1.0)],
               ["fastest split", "min-max of the repetitions"])
    title(fig, "Ranks × threads on one node: which split is the fastest?",
          f"{VERSION_LABEL[v]}, {MODEL_TXT[model]}, {ds[0]}")
    save(fig, ctx.out, f"P7-ranks-threads-{model}", ctx.fmts)


def p8_breakdown(ctx, model):
    """Stacked bars E-step / M-step / t_comm / I/O. One panel per version (2 x 2), x = cores."""
    best = ctx.best
    d = best[best.model == model]
    vs = versions_in(d)
    need(vs, "no data")
    ds = pick_datasets(best, model, ctx.args.large, ctx.args.small)
    need(ds, "no parallel data")
    d = d[d.dataset == ds[0]]
    fig, axs, nc = panel_grid(len(vs), row_h=2.4)
    for ax, v in zip(axs, vs):
        s = d[d.label == v].sort_values("cores")
        x = np.arange(len(s))          # one bar per core count
        bottom = np.zeros(len(s))      # height reached by the stack so far
        for name, col in STEP_COL.items():
            y = s[col].fillna(0).to_numpy()
            ax.bar(x, y, 0.6, bottom=bottom, color=STEP_COLOR[name], edgecolor=TEXT, linewidth=0.5,
                   label=STEP_LABEL[name])
            bottom += y
        for xi, tot in zip(x, bottom):  # total time above every bar
            ax.text(xi, tot, f"{tot:.2f} s", ha="center", va="bottom", fontsize=7, color=TEXT)
        # own y scale per panel (the sequential run is 10-40x longer and would flatten the others);
        # room above the bars for the labels
        ax.set_ylim(0, (bottom.max() if len(bottom) else 1) * 1.15)
        ax.set_xticks(x, [str(c) for c in s.cores])
        ax.set_xlabel("number of cores")
        panel_name(ax, VERSION_LABEL[v])
    for ax in axs[::nc]:
        ax.set_ylabel("time of the whole run (s)")
    legend_from(fig, axs, ncol=2)
    title(fig, "Where the time goes: time of the whole run, split by phase",
          f"{MODEL_TXT[model]}, {ds[0]}, {place_txt(ctx.args.placement)}; "
          "every panel has its own time scale (read the total above the bars)")
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
    # compare at the largest core count that has both the baseline and at least one optimization
    cores = max(set(base.cores) & set(d[d.label.isin(opts)].cores))
    d = d[d.cores == cores]
    every = sorted(set(d[d.label == "baseline"].dataset))
    # two blocks: the REAL datasets on top, the SYNTHETIC ones below
    groups = [(h, l) for h, l in (("Real datasets", [x for x in every if not is_synth(x)]),
                                  ("Synthetic datasets", [x for x in every if is_synth(x)])) if l]
    fig = plt.figure(figsize=(FIG_W, 2.7 * len(groups) + 0.2), layout="constrained")
    axs = []
    w = 0.8 / len(opts)
    for sf, (head, dss) in zip(blocks(fig, len(groups)), groups):
        ax = sf.subplots()
        axs.append(ax)
        for i, v in enumerate(opts):
            y = []  # time / baseline time for every dataset (NaN if one of the two runs is missing)
            for ds in dss:
                tb = d[(d.label == "baseline") & (d.dataset == ds)].t
                tv = d[(d.label == v) & (d.dataset == ds)].t
                y.append(tv.iloc[0] / tb.iloc[0] if len(tb) and len(tv) else np.nan)
            ax.bar(np.arange(len(dss)) + (i - (len(opts) - 1) / 2) * w, y, w, color=VERSION_COLOR[v],
                   edgecolor=TEXT, linewidth=0.6, label=VERSION_LABEL[v])
        ax.axhline(1, ls="--", color=IDEAL, lw=1, label="baseline MPI + OpenMP (= 1)")
        ax.set_xticks(range(len(dss)), dss, rotation=40, ha="right", fontsize=7)
        ax.set_xlim(-0.6, len(dss) - 0.4)
        ax.set_ylabel("time / baseline time\n(below 1 = faster)")
        block_header(sf, head)
    legend_from(fig, axs)
    title(fig, "Gain of the optimizations over the parallel baseline, per dataset",
          f"{MODEL_TXT[model]}, {cores} cores, {place_txt(ctx.args.placement)}")
    save(fig, ctx.out, f"P9-opt-gain-{model}", ctx.fmts)


def p1_sweeps(ctx, model):
    """P1: time per iteration vs K (panel a) and vs D (panel b), log-log, one line per version.
    Data: results/logs/sweep/ (one fixed layout, see pemc-submit.sh sweep)."""
    sw = ctx.sweep
    need(sw is not None, "no sweep logs (run: pemc-submit.sh sweep)")
    d = sw[sw.model == model]
    need(not d.empty, f"no sweep run for {model}")
    is_d = d.dataset.str.startswith("synth_sweep_d")
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.0), layout="constrained")
    for ax, sub, col, name, slope in ((axs[0], d[~is_d], "K", "number of clusters K", 1.0),
                                      (axs[1], d[is_d], "D", "number of variables D",
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
                  label=(f"reference: time $\\propto {col}^{{{slope:g}}}$" if slope != 1
                         else f"reference: time $\\propto {col}$"))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sorted(set(sub[col].astype(int)))))  # measured values
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set_xlabel(name + " (log scale)")
    axs[0].set_ylabel("time per iteration (s)\n(log scale)")
    panel_name(axs[0], "(a) growing K  (synth_clean_p01)")
    panel_name(axs[1], "(b) growing D  (synth_sweep_dDDD, K = 4)")
    legend_from(fig, axs, ncol=2)
    title(fig, "Cost of one EM iteration as the number of clusters and of variables grows",
          f"{MODEL_TXT[model]}; the dashed lines show the expected growth")
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
    # the sequential version runs on 1 core: against the roofs of 8 cores it would look 8x worse
    # than it is, so only the parallel versions (all on the 8 cores of one node) are drawn
    d = d[d.label != "sequential"]
    need(not d.empty, "no parallel single-node run with flops/bytes")
    fig, ax = plt.subplots(figsize=(HALF_W, 3.5), layout="constrained")
    xs = np.logspace(-1, 2, 100)
    gf, bw, gfc = ctx.args.peak_gflops, ctx.args.peak_bw, ctx.args.peak_gflops_compiled
    if gf and bw:
        ax.plot(xs, np.minimum(gf, bw * xs), color=TEXT, lw=1.3, label="roofline (limit of the node)")
        if gfc:  # the real limit of OUR binaries: -O3 without -march = SSE2, no FMA
            ax.plot(xs, np.minimum(gfc, bw * xs), color=TEXT, lw=1.0, ls="--",
                    label=f"limit as compiled (SSE2, {gfc:.0f} GFLOP/s)")
        ridge = gf / bw  # intensity where the two limits meet
        ax.axvline(ridge, color=IDEAL, ls=":", lw=0.9)
        # the two roofs and the two regions, written on the plot
        ax.text(xs[-1], gf * 1.15, f"peak computation {gf:.0f} GFLOP/s", ha="right", va="bottom",
                fontsize=7, color=TEXT)
        bw_label = ax.text(0.13, bw * 0.13 * 1.25, f"memory bandwidth {bw:.0f} GB/s", ha="left",
                           va="bottom", fontsize=7, color=TEXT, rotation_mode="anchor")
        ax.text(ridge / 1.15, 0.02, "← memory bound", ha="right", va="bottom", fontsize=7,
                color=IDEAL, transform=ax.get_xaxis_transform())
        ax.text(ridge * 1.15, 0.02, "compute bound →", ha="left", va="bottom", fontsize=7,
                color=IDEAL, transform=ax.get_xaxis_transform())
    # Versions with the same computation fall on the same point (diagonal: baseline = opt2; full:
    # opt1 = opt2): the markers are drawn concentric, the larger below, so all of them stay visible
    size = {"baseline": 13, "opt1": 13, "opt2": 7.5}
    for _, r in sorted(d.iterrows(), key=lambda kv: -size[kv[1].label]):
        v = r.label
        ax.plot(r.flops / r.bytes, r.flops / (r.t_estep + r.t_mstep) / 1e9, ls="",
                marker=VERSION_MARKER[v], color=VERSION_COLOR[v], ms=size[v], mec=TEXT, mew=0.7,
                label=f"{VERSION_LABEL[v]}, {int(r.cores)} core" + ("s" if r.cores > 1 else ""))
    ax.set_xscale("log")
    ax.set_yscale("log")
    # room under the lowest point for the "memory bound / compute bound" labels
    pts = d.flops / (d.t_estep + d.t_mstep) / 1e9
    ax.set_ylim(bottom=min(pts.min(), bw * 0.1 if bw else pts.min()) / 6)
    ax.set_xlabel("arithmetic intensity (FLOP per byte from memory)")
    ax.set_ylabel("performance (GFLOP/s)")
    legend_from(fig, [ax], ncol=2)
    if gf and bw:  # the bandwidth label follows the slope of the roof, as drawn on the page
        fig.canvas.draw()
        (x1, y1), (x2, y2) = ax.transData.transform([(0.1, bw * 0.1), (1.0, bw * 1.0)])
        bw_label.set_rotation(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    title(fig, "Roofline: is EM limited by memory or by computation?",
          f"{MODEL_TXT[model]}, {ds[0]}, 8 cores of 1 node; "
          "overlapping points are drawn one inside the other")
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
    fig, ax = plt.subplots(figsize=(min(FIG_W, max(HALF_W, 0.85 * len(g) + 1.8)), 3.2),
                           layout="constrained")
    ax.bar(x, g["mean"], 0.6, color=VERSION_COLOR[v], edgecolor=TEXT, linewidth=0.6,
           yerr=[g["mean"] - g.lo, g.hi - g["mean"]], capsize=3)
    for xi, r in zip(x, g.itertuples()):
        ax.text(xi, r.hi, f"+{100 * (r.hi / r.mean - 1):.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{int(r.nodes)} node{'s' if r.nodes > 1 else ''}\n{int(r.ranks)} × {int(r.threads)}"
                      for r in g.itertuples()], fontsize=7)
    ax.set_xlabel("nodes, MPI ranks × OpenMP threads per rank")
    ax.set_ylabel("computation time of a rank (s)")
    ax.set_ylim(0, g.hi.max() * 1.18)
    h = [plt.Rectangle((0, 0), 1, 1, fc=VERSION_COLOR[v], ec=TEXT, lw=0.6),
         plt.Line2D([], [], color=TEXT, marker="_", ms=9, mew=1.2, ls="-", lw=1.0),
         plt.Line2D([], [], color="none")]
    top_legend(fig, h, ["mean over the ranks", "fastest to slowest rank",
                        "+% = extra time of the slowest rank (the others wait)"], ncol=3)
    title(fig, "Load imbalance between MPI ranks",
          f"{VERSION_LABEL[v]}, {MODEL_TXT[model]}, {ds[0]}, {place_txt(ctx.args.placement)}")
    save(fig, ctx.out, f"P11-imbalance-{model}", ctx.fmts)


PLOTS = {"P1": p1_sweeps, "P2": p2_p3_speedup, "P3": p2_p3_speedup, "P4": p4_strong, "P5": p5_weak,
         "P6": p6_placement, "P7": p7_heatmap, "P8": p8_breakdown, "P9": p9_opt_gain, "P10": p10_roofline,
         "P11": p11_imbalance}


# ---------------------------------------------------------------------------------------------
def main():
    # ---- command line ----
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/plots/performance")
    ap.add_argument("--plots", nargs="+", default=["P1", "P2", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11"],
                    help="IDs to draw (P3 = the efficiency panels of the P2 figure)")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--placement", default="pack", choices=PLACE_ORDER,
                    help="placement fixed for every plot except P6 (the best one found with P6)")
    ap.add_argument("--version", help="version for P6/P7 (default: the most optimized one present)")
    ap.add_argument("--large", default="synth_clean_p16", help="large dataset (auto: max N if missing)")
    ap.add_argument("--small", default="cardiotocography", help="small dataset (auto: min N if missing)")
    ap.add_argument("--family", default="synth_clean", help="synthetic family for the weak scaling P5")
    # Default roofs of P10, for the 8 cores of the 1-node runs on a Xeon Gold 6252N
    # (src/script/pemc-stream.sh summary, results/logs/stream/):
    #   peak = 8 cores x 2.3 GHz x 32 FLOP/cycle (AVX-512, 2 FMA ports) = 588.8 GFLOP/s
    #   bandwidth = STREAM Triad with 8 threads (OMP_PROC_BIND=close) = 97.0 GB/s
    ap.add_argument("--peak-gflops", type=float, default=588.8,
                    help="P10: peak GFLOP/s of the cores used (the flat roof of the roofline) [588.8]")
    ap.add_argument("--peak-bw", type=float, default=97.0,
                    help="P10: memory bandwidth in GB/s, STREAM Triad [97.0]")
    # Second, lower roof: what OUR binaries can reach. They are built with -O3 and no -march, so
    # the compiler uses SSE2 (2 doubles, no FMA): 8 cores x 2.3 GHz x 4 FLOP/cycle = 73.6 GFLOP/s
    ap.add_argument("--peak-gflops-compiled", type=float, default=73.6,
                    help="P10: peak GFLOP/s of the binaries as compiled (SSE2) [73.6]; 0 = not drawn")
    ap.add_argument("--fmt", nargs="+", default=["pdf"], help="output formats (pdf, png, ...)")
    args = ap.parse_args()

    # ---- data: logs -> one row per run -> one row per layout (median over the repetitions) ->
    # one row per core count (best ranks x threads split) ----
    args.out.mkdir(parents=True, exist_ok=True)
    df = load_logs(args.logs)
    agg = aggregate(df)
    best = best_split(agg, args.placement)
    sweep_df = load_logs(args.logs / "sweep", required=False, pattern="*.log")  # P1 only
    sweep = aggregate(sweep_df) if sweep_df is not None else None
    # everything the plot functions need, in one object (they are called as plot(ctx, model))
    ctx = argparse.Namespace(args=args, df=df, agg=agg, best=best, sweep=sweep, out=args.out,
                             fmts=args.fmt)
    print(f"{len(df)} runs, versions: {sorted(set(df.label))}, placement: {args.placement}")

    # ---- draw: every requested plot, for every model. P2 and P3 are the same function (one
    # figure with two rows), so "done" avoids drawing it twice if both IDs are requested ----
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
            except LookupError as e:  # missing data (see need()): skip this plot, not the others
                print(f"skip {pid} {model}: {e}")


if __name__ == "__main__":
    main()
