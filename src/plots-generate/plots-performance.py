#!/usr/bin/env python3
"""
plots-performance.py

Performance plots of the report (catalog P1-P12 of doc/plots.md; the default set is the one of
the report: P2, P3, P4, P5, P8, P9, P10, P11).
Reads all the logs in results/logs/<model>/<stage>/*.log and writes PDF files in results/plots/performance/.

Usage (from the root of the repository):
    python3 src/plots-generate/plots-performance.py                 # the plots of the report
    python3 src/plots-generate/plots-performance.py --plots P2 P6   # only some
    python3 src/plots-generate/plots-performance.py --placement scatter:excl --large synth_clean_p16
    python3 src/plots-generate/plots-performance.py --grid brainweb synth_rare_p04 synth_rare_p16
    python3 src/plots-generate/plots-performance.py --min-run 0     # also the short datasets

Where the data comes from
    One log file per PBS job: results/logs/<model>/<stage>/<VERSION>_<TAG>.log
    (model = diagonal | full, stage = sequential | baseline | opt1 | opt2), TAG = "seq" (baseline-sequential) or
    "<placement>_<N>n<R>r<T>t" (e.g. pack_excl_2n2r4t = pack:excl, 2 nodes, 2 ranks/node,
    4 threads/rank). The placement is NOT in the log block, so it is read from the file name.
    Block format: see include/logger.h.

Rules (doc/plots.md)
    - Time = t_iteration (time per iteration), median over the repetitions, error bar = min-max.
    - Speedup is always against baseline-sequential of the same model and dataset.
    - For every number of cores the best ranks x threads split WITH AT LEAST 2 RANKS PER NODE is
      used (see hybrid_splits; P7 shows all the splits of one node, 1 x 8 included).
    - Placement: only exclusive nodes. 1-8 cores on 1 node with --placement (default pack:excl),
      16 and 32 cores on 2 and 4 nodes with --multi-placement (default scatter:excl). PBS puts
      every pack request on ONE node (the nodes have 96 cores), so only scatter gives real
      multi-node runs (see best_split). P6 compares the four placements, with the same split.
    - Diagonal and full are never on the same plot: one file per model.
    - File names: <ID>-<what>-<model>.pdf
    - Clarity (same rules as plots-cluster.py): bold title in plain words + one line with model,
      dataset and placement ("1-8 cores (1 node): pack exclusive; 16-32 cores (2-4 nodes): scatter
      exclusive"); ONE legend above the panels with names in plain words ("pack exclusive", "MPI
      communication + waiting for the other ranks"); axes with their unit.
    - Size: FIG_W = 7 in, HALF_W = 4.4 in for the single-panel plots P5, P6, P7, P10; the report
      includes them at width=0.82\textwidth (P5 at 0.52\textwidth).
    - Only LONG datasets (--min-run, default 1 s): a dataset is drawn only if EVERY one of its
      runs (all versions, all core counts) spends at least min-run seconds in the measured
      iterations (t_estep + t_mstep + t_comm). Shorter runs are dominated by noise (spread of the
      repetitions ~6% below 10 ms); their logs stay in results/logs/. P5 is checked on its own
      (its weak runs are all long). The figures do not write the threshold: the report does.
    - Datasets: P2, P3, P4, P8, P10 are one row of panels (--grid, default GRID_DEFAULT: the
      real brainweb and two large synthetic ones); P9 shows ALL the long datasets, real first;
      P6, P7, P11 one large synthetic dataset (--large): they measure the machine (placement,
      splits, balance), not the data.

Reference of the optimizations
    opt1 / opt2 are always compared with baseline-MPI-OpenMP-<model> on
    the same cores, dataset and placement (P9), NOT with baseline-sequential. baseline-sequential
    is only the reference of the speedup (P2/P3). The weak-scaling efficiency (P5) of
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
            else:  # "seq" or a log without tag: assume a baseline-sequential/single-node run
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
    # label = short key of the version (= stage folder): sequential | baseline | opt1 | opt2
    # (baseline-sequential, baseline-MPI-OpenMP, opt1, opt2; plain names in VERSION_LABEL)
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
    # length of the measured part of the run (all the iterations, no reading / warm-up): the
    # filter --min-run of the datasets (long_datasets)
    a["t_run"] = a.t_estep + a.t_mstep + a.t_comm
    return a.reset_index()


def long_datasets(best, min_run):
    """{(model, dataset)} whose SHORTEST run (over all the versions and core counts in best)
    lasts at least min_run seconds: the only datasets drawn, the others are mostly noise."""
    g = best.groupby(["model", "dataset"])["t_run"].min()
    return set(g[g >= min_run].index)


def hybrid_splits(a):
    """Drops the splits with ONE rank per node and several threads (1 x 2, 1 x 4, 1 x 8 on one
    node). They cannot be compared on every core count: on 2 and 4 nodes the 1-rank-per-node
    layouts failed (Open MPI refuses the binding, see src/script/pemc-job.sh), so for the same
    rule on every core count they are left out everywhere. 1 x 1 (one core) is kept."""
    return a[~((a.ranks / a.nodes == 1) & (a.threads > 1))]


def best_split(agg, placement="pack:excl", multi="scatter:excl"):
    """For every core count the best ranks x threads split with at least 2 ranks per node (see
    hybrid_splits; the baseline-sequential runs always kept). All the points are on exclusive
    nodes: runs on ONE node use `placement` (default pack:excl), runs on several nodes use
    `multi` (default scatter:excl). The reason: PBS puts all the chunks of a pack (or pack:excl) request on ONE
    node when it has enough cores (the nodes have 96), so the "2 / 4 node" pack runs are really
    16 / 32 cores of one node (checked in results/logs/pbs/*.out, "nodes assigned by PBS"). Only
    scatter places the chunks on different nodes, so it is the only real multi-node measure; the
    exclusive variant is used because on shared nodes the spread of the repetitions is 22%; on ONE
    shared node (pack) the memory-bound versions are also up to 22% slower than on an exclusive
    one (pack:excl), which is why pack:excl is the default on one node."""
    agg = hybrid_splits(agg)
    a = agg[((agg.nodes == 1) & (agg.placement == placement))
            | ((agg.nodes > 1) & (agg.placement == multi)) | (agg.label == "sequential")]
    # one row per (version, model, dataset, K, cores): the one with the smallest time among the
    # ranks x threads splits that give those cores
    key = ["label", "model", "dataset", "K", "cores"]
    return a.loc[a.groupby(key)["t"].idxmin()].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def versions_in(d):
    """Versions present in the table d, in the fixed plotting order (baseline-sequential first)."""
    return [v for v in VERSION_ORDER if v in set(d.label)]


# Datasets of the grid figures (P2, P3, P4, P8, P10), one panel each in one row (--grid). Only
# datasets whose runs all last >= 1 s (long_datasets): the real one is brainweb (11.9 M rows, the
# only real dataset that long; the other real ones finish in milliseconds), the synthetic ones
# are two sizes (2 M and 4 M rows) of the family with the most work for the model: synth_overlap
# (K = 5, D = 20) for diagonal, synth_rare (K = 6, D = 50) for full. One family per model, so
# between the two synthetic panels only N changes.
GRID_DEFAULT = {"diagonal": ["brainweb", "synth_overlap_p08", "synth_overlap_p16"],
                "full": ["brainweb", "synth_rare_p08", "synth_rare_p16"]}
# dataset of the one-dataset plots P6, P7, P11: the large synthetic one of the grid
LARGE_DEFAULT = {"diagonal": "synth_overlap_p16", "full": "synth_rare_p16"}


def large_of(ctx, model):
    """Dataset of the one-dataset plots (P6, P7, P11): --large, or the default of the model."""
    return ctx.args.large or LARGE_DEFAULT[model]


BREAKDOWN_CORES = [1, 8, 32]  # core counts of the bars of P8
VERSION_SHORT = {"sequential": "seq", "baseline": "base", "opt1": "opt1", "opt2": "opt2"}


def dataset_grid(ctx, model, row_h):
    """Figure with ONE row of panels, one per dataset of --grid (default GRID_DEFAULT), real
    first. A dataset without parallel runs or shorter than --min-run (not in ctx.best) is
    skipped with a message. Returns (fig, cells): one cell per drawn dataset (ax, ds, d = its
    rows of ctx.best, row, col)."""
    data = ctx.best[ctx.best.model == model]
    names = ctx.args.grid or GRID_DEFAULT[model]
    names = sorted(names, key=is_synth)  # real first (stable: the rest keeps its order)
    keep = [ds for ds in names if (data[data.dataset == ds].label != "sequential").sum() > 0]
    for ds in names:
        if ds not in keep:
            print(f"  {model}: {ds} not drawn (no parallel run, or a run shorter than --min-run)")
    need(keep, f"none of the grid datasets has long parallel runs for {model}")
    fig, axs = plt.subplots(1, len(keep), figsize=(FIG_W, row_h + 0.35), layout="constrained",
                            squeeze=False)
    cells = []
    for c, (ax, ds) in enumerate(zip(axs[0], keep)):
        d = data[data.dataset == ds]
        kind = "synthetic" if is_synth(ds) else "real"
        ax.set_title(f"{kind}: {ds}\n(N = {int(d.N.max()):,})", fontsize=7.8, fontweight="bold")
        cells.append(argparse.Namespace(ax=ax, ds=ds, d=d, row=0, col=c))
    return fig, cells


def grid_axes(cell, cores, ylabel):
    """Cores axis of a grid panel; y label only on the first panel of the row."""
    cores_axis(cell.ax, cores, nodes_of(cell.d))
    cell.ax.set_ylabel(ylabel if cell.col == 0 else "")


def fmt_s(t):
    """Seconds in a short readable form: 1009 s, 135 s, 1.08 s, 38.2 ms, 950 µs."""
    if t >= 1000:
        return f"{t:.0f} s"
    if t >= 1:
        return f"{t:.3g} s"
    if t >= 1e-3:
        return f"{t * 1e3:.3g} ms"
    return f"{t * 1e6:.3g} µs"


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
    """The most optimized version present in the data (opt2 > opt1 > baseline-MPI-OpenMP), or None."""
    for v in ("opt2", "opt1", "baseline"):
        if v in set(agg.label):
            return v
    return None


# ---- names in plain words, for legends and titles (the logs keep the short ones) ----
VERSION_LABEL = {"sequential": "baseline-sequential (1 core)", "baseline": "baseline-MPI-OpenMP",
                 "opt1": "opt1 (faster computation)", "opt2": "opt2 (less communication)"}
PLACE_LABEL = {"pack": "pack shared", "pack:excl": "pack exclusive",
               "scatter": "scatter shared", "scatter:excl": "scatter exclusive"}
STEP_LABEL = {"E-step": "E-step (computation)", "M-step": "M-step (computation)",
              "t_comm": "MPI communication + waiting for the other ranks", "I/O": "reading the dataset"}
MODEL_TXT = {"diagonal": "diagonal covariance", "full": "full covariance"}

# Figure sizes: FIG_W = full figures, HALF_W = single-panel plots (P5, P6, P7, P10). The report
# includes them at width=0.82\textwidth (P5 at 0.52\textwidth), see doc/plots.md.
FIG_W, HALF_W = 7.0, 4.4


def place_txt(p):
    """'8 cores (1 node): pack exclusive' for the subtitles of the one-node plots."""
    return f"8 cores (1 node): {PLACE_LABEL.get(p, p)}"


def scale_txt(args):
    """Placements of the scaling curves, for the subtitles (see best_split)."""
    return (f"1-8 cores (1 node): {PLACE_LABEL.get(args.placement, args.placement)}\n"
            f"16-32 cores (2-4 nodes): {PLACE_LABEL.get(args.multi_placement, args.multi_placement)}")


def title(fig, text, sub=None):
    """Bold title in plain words (what the figure shows) and an optional second line in normal
    font (model, dataset, placement), ABOVE everything else of the figure (legend included):
    from the top, title, subtitle, legend, panels."""
    # lines wrapped to the width of the figure (about 12 bold 10 pt and 15 normal 8 pt
    # characters per inch): a long title must not make the figure wider than the page
    text = textwrap.fill(text, int(fig.get_figwidth() * 12))
    # the subtitle keeps its line breaks (the placement goes on lines of its own: pack, then scatter)
    if sub:
        sub = "\n".join(textwrap.fill(s, int(fig.get_figwidth() * 15)) for s in sub.split("\n"))
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
    # baseline-MPI-OpenMP = 1) would come first, so they are moved to the end
    ref = ("ideal", "reference", "baseline-MPI-OpenMP (=")
    labels = sorted(seen, key=lambda l: l.startswith(ref))  # stable: the rest keeps its order
    top_legend(fig, [seen[l] for l in labels], labels, ncol)


def panel_name(ax, text):
    """Name of a panel, above it, bold."""
    ax.set_title(text, fontsize=8.5, fontweight="bold")




def is_synth(ds):
    """True for the synthetic datasets (generated), False for the real ones."""
    return ds.startswith("synth_")








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
        # "8\n1n" = 8 cores on 1 node (short: "1 node" under every tick overlaps)
        ax.set_xticks(ticks, [f"{c}\n{nodes[c]}n" if c in nodes else str(c) for c in ticks],
                      fontsize=6.8)
        ax.set_xlabel("number of cores (n = nodes)")
    else:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("number of cores")


def line(ax, x, y, lo, hi, v, ms=4.5, lw=1.4, zorder=2):
    """Line of version v (its color and marker) with an error bar from lo to hi at every point.
    ms / lw / zorder: larger and lower for a line that another one covers (P1)."""
    ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], color=VERSION_COLOR[v],
                marker=VERSION_MARKER[v], ms=ms, lw=lw, capsize=2, mec=TEXT, mew=0.4,
                label=VERSION_LABEL[v], zorder=zorder)


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
# ONE legend above the panels, axes named with their unit (doc/plots.md, section 4)
# ---------------------------------------------------------------------------------------------
def p2_speedup(ctx, model):
    """P2 speedup against baseline-sequential, one row of dataset panels (dataset_grid)."""
    fig, cells = dataset_grid(ctx, model, 2.4)
    for c in cells:
        seq = c.d[c.d.label == "sequential"]
        need(not seq.empty, f"no baseline-sequential run for {model}/{c.ds}")
        t_seq = seq.t.iloc[0]  # time per iteration of the baseline-sequential run: reference of the speedup
        allc = []
        for v in versions_in(c.d):
            s = c.d[c.d.label == v].sort_values("cores")
            if v == "sequential":
                s = s.assign(cores=1)
            cc = s.cores.to_numpy()
            allc += list(cc)
            # a larger time gives a smaller speedup: the error bar limits swap
            line(c.ax, cc, t_seq / s.t.to_numpy(), t_seq / s.hi.to_numpy(), t_seq / s.lo.to_numpy(), v)
        cc = np.array(sorted(set(allc)))
        ideal(c.ax, cc, cc)  # ideal speedup = number of cores
        c.ax.set_yscale("log", base=2)
        c.ax.yaxis.set_major_formatter(ScalarFormatter())
        grid_axes(c, allc, "speedup over baseline-sequential\n(log scale)")
    legend_from(fig, [c.ax for c in cells], ncol=3)
    title(fig, "Speedup against baseline-sequential",
          f"{MODEL_TXT[model]}; fastest split with at least 2 ranks per node"
          f"\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P2-speedup-{model}", ctx.fmts)


def p3_efficiency(ctx, model):
    """P3 parallel efficiency = speedup / cores, same grid as P2."""
    fig, cells = dataset_grid(ctx, model, 2.4)
    for c in cells:
        seq = c.d[c.d.label == "sequential"]
        need(not seq.empty, f"no baseline-sequential run for {model}/{c.ds}")
        t_seq = seq.t.iloc[0]
        allc, eff_max = [], 1.0  # never cut a point: an efficiency above 1 must stay visible
        for v in versions_in(c.d):
            s = c.d[c.d.label == v].sort_values("cores")
            if v == "sequential":
                s = s.assign(cores=1)
            cc = s.cores.to_numpy()
            allc += list(cc)
            sp, lo, hi = t_seq / s.t.to_numpy(), t_seq / s.hi.to_numpy(), t_seq / s.lo.to_numpy()
            line(c.ax, cc, sp / cc, lo / cc, hi / cc, v)
            eff_max = max(eff_max, np.nanmax(hi / cc))
        cc = np.array(sorted(set(allc)))
        ideal(c.ax, cc, np.ones_like(cc, dtype=float))  # ideal efficiency = 1
        c.ax.set_ylim(0, eff_max * 1.1 + 0.05)
        grid_axes(c, allc, "parallel efficiency\n(speedup / cores)")
    legend_from(fig, [c.ax for c in cells], ncol=3)
    title(fig, "Parallel efficiency against baseline-sequential",
          f"{MODEL_TXT[model]}; efficiency = speedup / cores\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P3-efficiency-{model}", ctx.fmts)


def p4_strong(ctx, model):
    """P4 strong scaling: time per iteration vs cores with the dataset fixed, same grid as P2."""
    fig, cells = dataset_grid(ctx, model, 2.4)
    for c in cells:
        allc = []
        for v in versions_in(c.d):
            s = c.d[c.d.label == v].sort_values("cores")
            cc = np.ones(len(s), int) if v == "sequential" else s.cores.to_numpy()
            allc += list(cc)
            line(c.ax, cc, s.t.to_numpy(), s.lo.to_numpy(), s.hi.to_numpy(), v)
        seq = c.d[c.d.label == "sequential"]
        if not seq.empty:
            cc = np.array(sorted(set(allc)))
            ideal(c.ax, cc, seq.t.iloc[0] / cc, label="ideal (baseline-sequential time / cores)")
        c.ax.set_yscale("log")
        grid_axes(c, allc, "time per iteration (s)\n(log scale)")
    legend_from(fig, [c.ax for c in cells], ncol=3)
    title(fig, "Strong scaling: time per iteration with the size of the dataset fixed",
          f"{MODEL_TXT[model]}\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P4-strong-{model}", ctx.fmts)


def p5_weak(ctx, model):
    """Weak scaling: synthetic <family>_pNN has N proportional to NN, run on NN cores."""
    # ctx.best_all: the small sizes (p01, p02) are below --min-run as datasets (their runs on
    # 16-32 cores are short), but the WEAK runs (p on p cores) all have the same work per core
    # as p01 on 1 core: they are checked one by one below
    best, fam = ctx.best_all, ctx.args.family
    d = best[(best.model == model) & best.dataset.str.startswith(fam + "_p")].copy()
    need(not d.empty, f"no {fam}_pNN dataset")
    d["p"] = d.dataset.str.extract(r"_p(\d+)$")[0].astype(int)  # size factor = cores of the weak run
    weak = d[(d.label != "sequential") & (d.cores == d.p)]
    short = weak[weak.t_run < ctx.args.min_run]
    if not short.empty:
        print(f"  P5 {model}: {len(short)} weak runs shorter than --min-run "
              f"({short.t_run.min():.2f} s): the curve is noisy")
    fig, ax = plt.subplots(figsize=(HALF_W, 3.0), layout="constrained")
    cs = [1]  # core counts drawn (for the ticks and the ideal line)
    eff_max = 1.0  # highest efficiency drawn: the y axis never cuts a point
    for v in versions_in(d):
        # Weak-scaling efficiency of a version = its OWN time on 1 core (smallest problem, p01)
        # / its time on p cores with p x the data. The reference is the same version, NOT the
        # baseline-sequential: otherwise a faster algorithm (opt1 full, ~2x baseline-sequential) would
        # start above 1 and the plot would mix "faster algorithm" with "scales well" (that gain
        # is in P2 and P9). baseline-sequential has no curve here (it never runs on p > 1).
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
    ax.set_xlabel("number of cores p (n = nodes); dataset with p × the rows")
    ax.set_ylabel("weak-scaling efficiency\n(same version: time on 1 core\n/ time on p cores)")
    ax.set_ylim(0, eff_max * 1.1 + 0.05)
    legend_from(fig, [ax], ncol=2)
    title(fig, "Weak scaling: the data per core is kept constant",
          f"{MODEL_TXT[model]}, {fam}_pNN on NN cores\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P5-weak-{model}", ctx.fmts)


def p6_placement(ctx, model):
    """Grouped bars: nodes x 4 placements. Most optimized version (opt2), large dataset."""
    a = ctx.agg
    v = ctx.args.version or best_version(a)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, large_of(ctx, model), ctx.args.small)
    need(ds, "no parallel data")
    d = a[(a.label == v) & (a.model == model) & (a.dataset == ds[0]) & (a.placement != "seq")]
    need(not d.empty, f"no data for {v}/{model}/{ds[0]}")
    # the SAME split for every placement and number of nodes: 2 ranks x 4 threads per node, the
    # only one measured with all four placements (the best split would compare different splits)
    d = d[(d.ranks == 2 * d.nodes) & (d.threads == 4)]
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
    # the nodes REQUESTED: pack puts all the chunks on ONE 96-core node, only scatter really uses
    # 2 or 4 nodes (see best_split)
    ax.set_xlabel("nodes requested (pack: always 1 node; scatter: 2 or 4)")
    ax.set_ylabel("time per iteration (s)")
    legend_from(fig, [ax], ncol=2)
    title(fig, "Effect of the PBS placement on the time per iteration",
          f"{v}, {model} model, {ds[0]}, split 2 × 4 per node")
    save(fig, ctx.out, f"P6-placement-{model}", ctx.fmts)


def p7_heatmap(ctx, model):
    """P7: the ranks x threads splits of the cores of ONE node (most optimized version, opt2, large dataset), one
    bar each: time per iteration (median, whiskers min-max), the fastest one marked. With 8 cores
    the splits are 8x1, 4x2, 2x4, 1x8 (a ranks x threads table would be mostly empty)."""
    a = ctx.agg
    v = ctx.args.version or best_version(a)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, large_of(ctx, model), ctx.args.small)
    need(ds, "no parallel data")
    d = a[(a.label == v) & (a.model == model) & (a.dataset == ds[0]) & (a.nodes == 1)
          & (a.placement == ctx.args.placement)]
    need(not d.empty, "no single-node runs")
    # only the splits of ALL the cores of the node (8): the runs on 1, 2, 4 cores of one node (for
    # the speedup curves) are not splits of the same cores and must not be compared here
    d = d[d.cores == d.cores.max()]
    # one bar per split, from all-MPI to all-OpenMP (1 x 8 is shown here, but the scaling plots
    # do not use it, see hybrid_splits)
    d = d.sort_values("ranks", ascending=False)
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
          f"{VERSION_LABEL[v]}, {MODEL_TXT[model]}, {ds[0]}\n{place_txt(ctx.args.placement)}")
    save(fig, ctx.out, f"P7-ranks-threads-{model}", ctx.fmts)


def p8_breakdown(ctx, model):
    """P8 where the time goes, same grid as P2. In every panel one bar per (cores, version) for
    the cores of BREAKDOWN_CORES: the bar is the whole run split by phase (E-step, M-step,
    communication, reading) as a SHARE of its time, so datasets of milliseconds and of minutes
    can be compared; the total time of the run is written above the bar."""
    fig, cells = dataset_grid(ctx, model, 2.6)
    w = 0.22  # width of one bar; the bars of one core count are side by side
    for c in cells:
        ax, d = c.ax, c.d
        ticks, names, x0 = [], [], 0.0
        for cores in BREAKDOWN_CORES:
            vs = [v for v in VERSION_ORDER if not d[(d.label == v) & (d.cores == cores)].empty]
            if not vs:
                continue
            xs = x0 + (np.arange(len(vs)) - (len(vs) - 1) / 2) * w
            for xb, v in zip(xs, vs):
                r = d[(d.label == v) & (d.cores == cores)].iloc[0]
                vals = [0.0 if pd.isna(r[col]) else float(r[col]) for col in STEP_COL.values()]
                tot = sum(vals)
                bottom = 0.0
                for (name, _), val in zip(STEP_COL.items(), vals):
                    h = 100 * val / tot if tot > 0 else 0.0
                    ax.bar(xb, h, w * 0.9, bottom=bottom, color=STEP_COLOR[name], edgecolor=TEXT,
                           linewidth=0.4, label=STEP_LABEL[name])
                    bottom += h
                ax.text(xb, 102, fmt_s(tot), rotation=90, ha="center", va="bottom", fontsize=5.6,
                        color=TEXT)
                ticks.append(xb)
                names.append(VERSION_SHORT[v])
            # the core count under its group of bars, below the names of the versions
            ax.annotate(f"{cores} core" + ("s" if cores > 1 else ""), xy=(x0, 0),
                        xycoords=("data", "axes fraction"), xytext=(0, -25),
                        textcoords="offset points", ha="center", va="top", fontsize=7)
            x0 += 1.0
        ax.set_xticks(ticks, names, fontsize=6, rotation=90)
        ax.set_ylim(0, 140)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("share of the run (%)" if c.col == 0 else "")
    legend_from(fig, [c.ax for c in cells], ncol=2)
    title(fig, "Where the time goes: the whole run split by phase",
          f"{MODEL_TXT[model]}; seq = baseline-sequential, base = baseline-MPI-OpenMP; total time above"
          f"\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P8-breakdown-{model}", ctx.fmts)


def p9_opt_gain(ctx, model):
    """P9: opt1/opt2 time divided by the time of baseline-MPI-OpenMP (= 1),
    at the largest core count shared by the versions, same dataset and placement. baseline-sequential
    is NOT the reference here. Below 1 = faster than baseline-MPI-OpenMP."""
    best = ctx.best
    d = best[best.model == model]
    need("baseline" in set(d.label), "no baseline-MPI-OpenMP run")
    opts = [v for v in ("opt1", "opt2") if v in set(d.label)]
    need(opts, "no opt1/opt2 data yet")
    base = d[d.label == "baseline"]
    # compare at the largest core count that has both baseline-MPI-OpenMP and at least one optimization
    cores = max(set(base.cores) & set(d[d.label.isin(opts)].cores))
    d = d[d.cores == cores]
    every = sorted(set(d[d.label == "baseline"].dataset))
    # ONE panel (with the long datasets only one is real: two blocks would give it a whole row):
    # the REAL datasets first, a dotted line, then the SYNTHETIC ones
    real = [x for x in every if not is_synth(x)]
    dss = real + [x for x in every if is_synth(x)]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0), layout="constrained")
    w = 0.8 / len(opts)
    for i, v in enumerate(opts):
        y = []  # time / baseline-MPI-OpenMP time for every dataset (NaN if one of the two runs is missing)
        for ds in dss:
            tb = d[(d.label == "baseline") & (d.dataset == ds)].t
            tv = d[(d.label == v) & (d.dataset == ds)].t
            y.append(tv.iloc[0] / tb.iloc[0] if len(tb) and len(tv) else np.nan)
        x = np.arange(len(dss)) + (i - (len(opts) - 1) / 2) * w
        ax.bar(x, y, w, color=VERSION_COLOR[v], edgecolor=TEXT, linewidth=0.6, label=VERSION_LABEL[v])
        for xi, yi in zip(x, y):  # the ratio above every bar (opt1 full is a very short bar)
            if not np.isnan(yi):
                ax.text(xi, yi, f"{yi:.2f}", ha="center", va="bottom", fontsize=5.8, color=TEXT)
    ax.axhline(1, ls="--", color=IDEAL, lw=1, label="baseline-MPI-OpenMP (= 1)")
    if real and len(real) < len(dss):
        ax.axvline(len(real) - 0.5, ls=":", color=IDEAL, lw=1)
    ax.set_xticks(range(len(dss)), [f"{x} (real)" if not is_synth(x) else x for x in dss],
                  rotation=30, ha="right", fontsize=7)
    ax.set_xlim(-0.6, len(dss) - 0.4)
    ax.set_xlabel("dataset")
    ax.set_ylabel("time / baseline-MPI-OpenMP time\n(below 1 = faster)")
    legend_from(fig, [ax])
    title(fig, "Gain of the optimizations over baseline-MPI-OpenMP, per dataset",
          f"{MODEL_TXT[model]}\n{cores} cores ({cores // 8} nodes): "
          f"{PLACE_LABEL.get(ctx.args.multi_placement, ctx.args.multi_placement)}")
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
            # opt2 does the same computation as baseline-MPI-OpenMP (both models), so their lines
            # coincide: baseline-MPI-OpenMP is drawn thicker and below, opt2 thinner on top (as the
            # markers of P10)
            big = v == "baseline"
            line(ax, s_[col].to_numpy(), s_.t.to_numpy(), s_.lo.to_numpy(), s_.hi.to_numpy(), v,
                 ms=8 if big else 4.5, lw=3.2 if big else 1.4, zorder=2 if big else 3)
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
          f"{MODEL_TXT[model]}, 1 rank × 8 threads (baseline-sequential: 1 core)"
          f"\n8 cores (1 node): {PLACE_LABEL['pack:excl']}")
    save(fig, ctx.out, f"P1-sweep-{model}", ctx.fmts)


def p10_roofline(ctx, model):
    """P10 roofline of ONE node, same grid as P2: one panel per dataset with the versions on the
    8 cores of one node. x = FLOP / byte, y = GFLOP/s. FLOP and bytes are the analytic count of
    include/perfmodel.h. The roofs are the peaks of the node (--peak-gflops, --peak-bw) and the
    peak of the binaries as compiled (--peak-gflops-compiled, SSE2)."""
    a = ctx.agg
    gf, bw, gfc = ctx.args.peak_gflops, ctx.args.peak_bw, ctx.args.peak_gflops_compiled
    fig, cells = dataset_grid(ctx, model, 2.6)
    xs = np.logspace(-1, 2, 100)
    rot = []  # (axis, label of the bandwidth roof): rotated along the roof once the layout is known
    drawn = []
    for c in cells:
        ax = c.ax
        d = a[(a.model == model) & (a.dataset == c.ds) & (a.nodes == 1) & (a.cores == 8)
              & (a.placement == ctx.args.placement) & (a.label != "sequential")
              & a.flops.notna() & a.bytes.notna()]
        if d.empty:
            ax.set_visible(False)
            continue
        drawn.append(c)
        # the roofs are those of the 8 cores of one node: only the 8-core runs are compared
        d = hybrid_splits(d)  # same split rule as the scaling plots
        d = d.loc[d.groupby("label")["t"].idxmin()]  # best split of every version
        if gf and bw:
            ax.plot(xs, np.minimum(gf, bw * xs), color=TEXT, lw=1.1, label="roofline (limit of the node)")
            if gfc:
                ax.plot(xs, np.minimum(gfc, bw * xs), color=TEXT, lw=0.9, ls="--",
                        label=f"limit as compiled (SSE2, {gfc:.0f} GFLOP/s)")
            ridge = gf / bw
            ax.axvline(ridge, color=IDEAL, ls=":", lw=0.8)
            ax.text(xs[-1], gf * 1.15, f"peak {gf:.0f} GFLOP/s", ha="right", va="bottom",
                    fontsize=5.8, color=TEXT)
            rot.append((ax, ax.text(0.13, bw * 0.13 * 1.3, f"bandwidth {bw:.0f} GB/s", ha="left",
                                    va="bottom", fontsize=5.8, color=TEXT, rotation_mode="anchor")))
            ax.text(ridge / 1.15, 0.03, "← memory bound", ha="right", va="bottom", fontsize=5.6,
                    color=IDEAL, transform=ax.get_xaxis_transform())
            ax.text(ridge * 1.15, 0.03, "compute bound →", ha="left", va="bottom", fontsize=5.6,
                    color=IDEAL, transform=ax.get_xaxis_transform())
        # versions with the same computation fall on the same point: concentric markers
        size = {"baseline": 11, "opt1": 11, "opt2": 6.5}
        for _, r in sorted(d.iterrows(), key=lambda kv: -size[kv[1].label]):
            v = r.label
            ax.plot(r.flops / r.bytes, r.flops / (r.t_estep + r.t_mstep) / 1e9, ls="",
                    marker=VERSION_MARKER[v], color=VERSION_COLOR[v], ms=size[v], mec=TEXT, mew=0.6,
                    label=f"{VERSION_LABEL[v]}, 8 cores")
        ax.set_xscale("log")
        ax.set_yscale("log")
        pts = d.flops / (d.t_estep + d.t_mstep) / 1e9
        ax.set_ylim(min(pts.min(), bw * 0.1 if bw else pts.min()) / 6, (gf or pts.max()) * 4)
        ax.set_xlabel("FLOP per byte (log)")
        ax.set_ylabel("GFLOP/s (log)" if c.col == 0 else "")
    need(drawn, "no 8-core single-node run with flops/bytes")
    legend_from(fig, [c.ax for c in drawn], ncol=2)
    fig.canvas.draw()  # final layout: the bandwidth labels follow the slope of the roof
    for ax, lab in rot:
        (x1, y1), (x2, y2) = ax.transData.transform([(0.1, bw * 0.1), (1.0, bw * 1.0)])
        lab.set_rotation(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    title(fig, "Roofline: is EM limited by memory or by computation?",
          f"{MODEL_TXT[model]}; overlapping points are drawn one inside the other"
          f"\n{place_txt(ctx.args.placement)}")
    save(fig, ctx.out, f"P10-roofline-{model}", ctx.fmts)


def p12_size(ctx, model):
    """P12 speedup against the size of the dataset, ALL the datasets: one point per (dataset,
    version), on 8 cores (1 node) and on 32 cores (4 nodes). Filled markers = real datasets,
    empty = synthetic. It shows that the speedup depends on the size, real or synthetic."""
    best = ctx.best[ctx.best.model == model]
    seq = best[best.label == "sequential"].set_index("dataset").t
    need(not seq.empty, "no baseline-sequential run")
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.0), layout="constrained", sharey=True)
    for ax, cores in zip(axs, (8, 32)):
        drawn = False
        for v in ("baseline", "opt1", "opt2"):
            s = best[(best.label == v) & (best.cores == cores) & best.dataset.isin(seq.index)]
            if s.empty:
                continue
            drawn = True
            sp = seq.reindex(s.dataset).to_numpy() / s.t.to_numpy()
            real = ~s.dataset.map(is_synth).to_numpy()
            ax.plot(s.N.to_numpy()[real], sp[real], ls="", marker=VERSION_MARKER[v],
                    color=VERSION_COLOR[v], mec=TEXT, mew=0.4, ms=5, label=VERSION_LABEL[v])
            ax.plot(s.N.to_numpy()[~real], sp[~real], ls="", marker=VERSION_MARKER[v], mfc="none",
                    mec=VERSION_COLOR[v], mew=1.2, ms=5)
        if not drawn:
            ax.set_visible(False)
            continue
        ax.axhline(cores, ls="--", color=IDEAL, lw=1, label="ideal (speedup = cores)")
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("rows of the dataset N (log scale)")
        n = {8: 1, 32: 4}[cores]
        panel_name(ax, f"{cores} cores ({n} node{'s' if n > 1 else ''})")
    axs[0].set_ylabel("speedup over baseline-sequential\n(log scale)")
    seen = {}
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    hs = [seen[l] for l in seen] + [
        plt.Line2D([], [], ls="", marker="o", color=TEXT, ms=5),
        plt.Line2D([], [], ls="", marker="o", mfc="none", mec=TEXT, mew=1.2, ms=5)]
    top_legend(fig, hs, list(seen) + ["real dataset", "synthetic dataset"], ncol=3)
    title(fig, "Speedup against the size of the dataset (all the datasets)",
          f"{MODEL_TXT[model]}; filled = real dataset, empty = synthetic"
          f"\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P12-size-{model}", ctx.fmts)


def p11_imbalance(ctx, model):
    """P11 load imbalance: mean compute time per rank (bar) with min-max (whisker) for every
    ranks x threads layout of the most optimized version (opt2). The % above a bar is (max / mean - 1): the time
    the other ranks wait for the slowest one (it is hidden inside t_comm in the other plots)."""
    df = ctx.df
    v = ctx.args.version or best_version(ctx.agg)
    need(v, "no parallel version")
    ds = pick_datasets(ctx.best, model, large_of(ctx, model), ctx.args.small)
    need(ds, "no parallel data")
    d = df[(df.label == v) & (df.model == model) & (df.dataset == ds[0]) & (df.ranks > 1)
           & (((df.nodes == 1) & (df.placement == ctx.args.placement))
              | ((df.nodes > 1) & (df.placement == ctx.args.multi_placement))) & df.rc.notna()]
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
          f"{VERSION_LABEL[v]}, {MODEL_TXT[model]}, {ds[0]}\n{scale_txt(ctx.args)}")
    save(fig, ctx.out, f"P11-imbalance-{model}", ctx.fmts)


PLOTS = {"P1": p1_sweeps, "P2": p2_speedup, "P3": p3_efficiency, "P4": p4_strong, "P5": p5_weak,
         "P6": p6_placement, "P7": p7_heatmap, "P8": p8_breakdown, "P9": p9_opt_gain, "P10": p10_roofline,
         "P11": p11_imbalance, "P12": p12_size}


# ---------------------------------------------------------------------------------------------
def main():
    # ---- command line ----
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/plots/performance")
    # P1 (sweeps), P6 (placement), P7 (ranks x threads) and P12 (speedup vs size) are not in the
    # report: their runs are short (milliseconds) and say little; they can still be drawn
    ap.add_argument("--plots", nargs="+", default=["P2", "P3", "P4", "P5", "P8", "P9", "P10", "P11"],
                    help="IDs to draw (all: P1 ... P12)")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--placement", default="pack:excl", choices=PLACE_ORDER,
                    help="placement of the runs on ONE node (every plot except P6) [pack:excl]")
    ap.add_argument("--multi-placement", default="scatter:excl", choices=PLACE_ORDER,
                    help="placement of the runs on 2-4 nodes (pack would put them on ONE node) "
                         "[scatter:excl]")
    ap.add_argument("--version", help="version for P6/P7 (default: the most optimized one present)")
    # Datasets. The grid figures (P2, P3, P4, P8, P10) show one row of datasets; the one-dataset
    # plots (P6, P7, P11) the large synthetic one.
    # (defaults and the reason of the choice: GRID_DEFAULT, LARGE_DEFAULT)
    ap.add_argument("--grid", nargs="+", default=None, metavar="DATASET",
                    help="datasets of the grid figures, both models (default: brainweb + "
                         "synth_overlap_p08/p16 for diagonal, brainweb + synth_rare_p08/p16 for full)")
    ap.add_argument("--min-run", type=float, default=1.0,
                    help="draw only the datasets whose runs ALL last at least this many seconds "
                         "(t_estep + t_mstep + t_comm); 0 = all the datasets [1.0]")
    ap.add_argument("--large", default=None,
                    help="dataset of the one-dataset plots P6, P7, P11 (default: the large synthetic "
                         "one of the grid, synth_overlap_p16 diagonal / synth_rare_p16 full)")
    ap.add_argument("--small", default="cardiotocography", help="fallback small dataset")
    ap.add_argument("--family", default="synth_overlap",
                    help="synthetic family for the weak scaling P5 [synth_overlap]")
    # Default roofs of P10, for the 8 cores of the 1-node runs on a Xeon Gold 6252N
    # (src/script/pemc-stream.sh, run on hpc3-c12-n49: results/logs/stream/stream_hpc3-c12-n49-summary.txt):
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
    best = best_split(agg, args.placement, args.multi_placement)
    # only the long datasets (--min-run): best_all keeps all of them for P5 (see p5_weak)
    best_all = best
    keep = long_datasets(best, args.min_run)
    is_long = lambda t: [(m, ds) in keep for m, ds in zip(t.model, t.dataset)]
    best, agg = best[is_long(best)], agg[is_long(agg)]
    for m in args.models:
        print(f"{m}: datasets with all the runs >= {args.min_run:g} s: "
              f"{sorted(ds for mm, ds in keep if mm == m)}")
    sweep_df = load_logs(args.logs / "sweep", required=False, pattern="*.log")  # P1 only
    sweep = aggregate(sweep_df) if sweep_df is not None else None
    # everything the plot functions need, in one object (they are called as plot(ctx, model))
    ctx = argparse.Namespace(args=args, df=df, agg=agg, best=best, best_all=best_all, sweep=sweep,
                             out=args.out, fmts=args.fmt)
    print(f"{len(df)} runs, versions: {sorted(set(df.label))}, placement: {args.placement} on 1 node, "
          f"{args.multi_placement} on 2-4 nodes")

    # ---- draw: every requested plot, for every model ("done": an ID requested twice is drawn
    # once) ----
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
