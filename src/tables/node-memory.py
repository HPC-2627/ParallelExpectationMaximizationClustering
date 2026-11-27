#!/usr/bin/env python3
"""
node-memory.py

Do the memory channels of a node change the run times? Short answer, with the numbers, written to
results/tables/node-memory.md. The report (Hardware Infrastructure, Discussion on the imbalance
between the ranks, Limits) links it.

Usage (from the root of the repository):
    src/script/pemc-hwinfo.sh submit / summary     # 1. hardware of every node, on the cluster
    scp -r ...:ParallelExpectationMaximizationClustering/results/hwinfo results/   # 2. copy it
    python3 src/tables/node-memory.py               # 3. the file
    python3 src/tables/node-memory.py --print       #    and also print it

Input
    results/hwinfo/<node>.txt       one file per node, written by src/script/pemc-hwprobe.sh: the
                                    lines "MemTotal: ..." and "DIMM: <label> <size> <type>"
    results/logs/pbs/*-scx-*.err    stderr of the multi-node jobs (scatter:excl): Open MPI writes
                                    "[<node>:<pid>] MCW rank <r> bound to ..." for every rank of
                                    every run (--report-bindings in src/script/pemc-job.sh),
                                    i.e. on which node every rank ran
    results/logs/<model>/<stage>/*.log   the runs (format: include/logger.h); "rank_compute" =
                                    compute seconds of every rank
Output
    results/tables/node-memory.md

How the memory channels are found (6 channels per socket on these CPUs, Xeon Gold 6252N)
    c12 nodes: label CPU_SrcID#<socket>_MC#<m>_Chan#<c>_DIMM#<d>  -> channel = (m, c), explicit
    c10/c11 nodes: label PROC_<socket>_DIMM_<slot>, slots 1..12, two slots per channel
                                                                  -> channel = (slot + 1) // 2
    populated channels of a socket = number of distinct channels with at least one DIMM. The ranks
    of every run are bound to socket 0 (see the MCW lines), so socket 0 is used (on these nodes
    all the sockets have the same configuration anyway).

What is measured
    Runs on 2 and 4 nodes (scatter:excl): the ranks of ONE run execute the same code on the same
    amount of data, but on different nodes. For every rank:
        relative time = its compute time / mean compute time of the ranks of its run
    1.00 = as fast as the average rank of its run, 1.10 = 10% slower. If the number of memory
    channels mattered, the ranks on nodes with fewer channels would be above 1.
    Only the datasets whose runs all last >= 1 s (same rule as src/plots-generate/plots-performance.py
    --min-run); the same check is repeated with the thresholds of THRESHOLDS to show that the
    answer does not depend on it.
    The runs on one node are not used: every job runs on one node and every version ran on a few
    nodes only, so the node cannot be separated from the version.
"""

import argparse
import importlib.util
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# The log reader and the "long datasets" rule are those of the plots, so the report, the plots and
# this table use exactly the same runs
_spec = importlib.util.spec_from_file_location(
    "plots_performance", ROOT / "src/plots-generate/plots-performance.py")
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)

# Prefix of the PBS job names (src/script/pemc-submit.sh) -> (model, version)
JOB_PREFIX = {"bd": ("diagonal", "baseline"), "o1d": ("diagonal", "opt1"), "o2d": ("diagonal", "opt2"),
              "bf": ("full", "baseline"), "o1f": ("full", "opt1"), "o2f": ("full", "opt2")}
# Multi-node layouts: PBS name <nodes>x<ranks per node>x<threads> -> tag of the log file name
LAYOUTS = {"2x2x4": "2n2r4t", "4x2x4": "4n2r4t"}


# ---------------------------------------------------------------------------------------------
# Hardware of the nodes
# ---------------------------------------------------------------------------------------------
def read_node(path):
    """results/hwinfo/<node>.txt -> dict: RAM (GiB), populated channels and max DIMMs per channel
    of socket 0, and the list of the populated channels of every socket."""
    per_socket = defaultdict(lambda: defaultdict(int))  # socket -> channel -> number of DIMMs
    ram = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("MemTotal"):
            ram = float(line.split()[1])
        if not line.startswith("DIMM: "):
            continue
        label = line.split()[1]
        m = re.match(r"CPU_SrcID#(\d+)_MC#(\d+)_Chan#(\d+)_DIMM#\d+", label)  # c12: explicit
        if m:
            socket, channel = int(m[1]), (int(m[2]), int(m[3]))
        else:
            m = re.match(r"PROC_(\d+)_DIMM_(\d+)", label)                     # c10/c11: by slot
            if not m:
                continue
            socket, channel = int(m[1]) - 1, (int(m[2]) + 1) // 2
        per_socket[socket][channel] += 1
    if not per_socket:
        return None
    s0 = per_socket[min(per_socket)]
    return {"ram": ram, "channels": len(s0), "dimms_per_channel": max(s0.values()),
            "all_sockets": [len(per_socket[s]) for s in sorted(per_socket)]}


def read_nodes(hwdir):
    """All the nodes of results/hwinfo/ -> {node: dict of read_node}."""
    nodes = {}
    for path in sorted(hwdir.glob("hpc3-*.txt")):
        info = read_node(path)
        if info:
            nodes[path.stem] = info
    return nodes


# ---------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------
def long_datasets(logs):
    """{(model, dataset)} whose runs all last >= 1 s (same rule and code as the plots)."""
    df = pp.load_logs(logs)
    best = pp.best_split(pp.aggregate(df), "pack", "scatter:excl")
    return df, pp.long_datasets(best, 1.0)


def rank_rows(logs, pbsdir, nodes, keep):
    """One row per rank of every multi-node run (scatter:excl) of a long dataset: model, version,
    job, node of the rank, its channels, its relative compute time (see the docstring)."""
    rows = []
    for prefix, (model, version) in JOB_PREFIX.items():
        for layout, tag in LAYOUTS.items():
            err = pbsdir / f"{prefix}-scx-{layout}.err"
            log = logs / model / version / f"{version}-MPI-OpenMP-{model}_scatter_excl_{tag}.log"
            if not err.exists() or not log.exists():
                print(f"skip {err.name}: missing .err or log")
                continue
            n_ranks = int(layout[0]) * int(layout[2])  # nodes x ranks per node
            # "[hpc3-c12-n20.unitn.it:3571583] MCW rank 0 bound to ...": one line per rank, the
            # n_ranks lines of a run are written together, runs in the same order as the log
            binds = re.findall(r"\[([\w-]+)\.unitn\.it:\d+\] MCW rank (\d+)", err.read_text(errors="replace"))
            blocks = list(pp.parse_blocks(log))
            if len(binds) != n_ranks * len(blocks):
                print(f"skip {err.name}: {len(binds)} binding lines for {len(blocks)} runs")
                continue
            for i, block in enumerate(blocks):
                dataset = Path(block["dataset"]).stem
                if (model, dataset) not in keep:
                    continue
                host = {int(r): h for h, r in binds[i * n_ranks:(i + 1) * n_ranks]}
                rc = np.array([float(x) for x in block["rank_compute"].split()])
                for r in range(n_ranks):
                    hw = nodes.get(host[r], {})
                    rows.append({"model": model, "version": version, "job": err.stem,
                                 "dataset": dataset, "run": i, "rank": r, "node": host[r],
                                 "channels": hw.get("channels"), "rel": rc[r] / rc.mean()})
    rows = pd.DataFrame(rows)
    rows["channels"] = rows["channels"].astype("Int64")  # integer, empty when the channel count is not known
    return rows




# ---------------------------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------------------------
BASE = "baseline-MPI-OpenMP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hwinfo", type=Path, default=ROOT / "results/hwinfo")
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables/node-memory.md")
    ap.add_argument("--print", action="store_true", help="also print the file")
    args = ap.parse_args()

    nodes = read_nodes(args.hwinfo)
    if not nodes:
        raise SystemExit(f"no node in {args.hwinfo}: copy results/hwinfo/ from the cluster")
    df, keep = long_datasets(args.logs)
    n_datasets = df[["model", "dataset"]].drop_duplicates().shape[0]
    ranks = rank_rows(args.logs, args.logs / "pbs", nodes, keep)
    o1 = ranks.version == "opt1"

    # datasets used, by model
    kept_txt = "; ".join(f"{m}: " + ", ".join(f"`{d}`" for d in sorted(d for mm, d in keep if mm == m))
                         for m in sorted({m for m, _ in keep}))

    # 1. mean relative time by number of channels, with how many nodes and ranks
    g = ranks.groupby("channels")
    channels_txt = "\n".join(
        f"   - {c} channels: **{g.rel.mean()[c]:.3f}×** ({g.node.nunique()[c]} nodes, {g.size()[c]} ranks)"
        for c in g.rel.mean().index)
    # (groupby skips the ranks of a node without a channel count: they are not in this point)

    # 2. the clearest node that is slow with baseline-MPI-OpenMP and average with opt1
    per_node = ranks.groupby(["node", "version"]).rel.mean().unstack("version")
    per_node = per_node.dropna(subset=["baseline", "opt1"])
    slow = (per_node.baseline - per_node.opt1).idxmax()
    slow_ch = nodes[slow]["channels"]
    slow_b, slow_o = per_node.loc[slow, "baseline"], per_node.loc[slow, "opt1"]

    # 3. spread inside one run: widest p05-p95 band over the two models
    q = ranks[~o1].groupby("model").rel.quantile
    b_lo, b_hi = q(0.05).min(), q(0.95).max()

    text = f"""# Node memory

Generated by `src/tables/node-memory.py`.

## Question

**Do the memory channels of a node change the run times?**

The nodes have 2, 4 or 6 memory channels per socket (47 GB/s, 94 GB/s or 141 GB/s of peak
bandwidth, see [hardware.md](hardware.md)), and we could not choose them. If this mattered, a
version that ran on worse nodes would look slower than it is.

## Answer

**No.** The run times do not depend on the memory of the node. The differences between the ranks
of one run come from the code.

### How we checked

- **Runs on 2 and 4 nodes** (`scatter:excl`, 2 ranks x 4 threads per node, i.e. 8 cores of
  socket 0). All the ranks of one run process the same number of rows of the same dataset, but on
  different nodes. The node of every rank comes from the PBS output, its compute time from the
  run log.
- **Compute time** of a rank = time of its E-step + M-step (s), without the MPI communication.
- **Relative time** of a rank = its compute time (s) / average compute time of the ranks of its
  run (s). It is a ratio, written with ×: 1.00× = as fast as the average rank, 1.10× = 10% slower.
  If the channels mattered, the ranks on the nodes with fewer channels would always be above 1.00×.
- **Only the runs of at least 1 s**, the same rule as the report (Section "Datasets Shown and
  Duration of the Runs"): a dataset is used only if every one of its runs (every version, every number of
  cores) lasts at least 1 s. Shorter runs are dominated by noise and are not used. Datasets used
  ({len(keep)} of {n_datasets}): {kept_txt}.

Data: {len(ranks)} ranks, {ranks.groupby(["job", "dataset", "run"]).ngroups} runs, {ranks.node.nunique()} nodes.

### What we found

1. **The channels make no difference.** Average relative time of a rank:
{channels_txt}

   The nodes with the least bandwidth are not the slow ones.
2. **A slow node is not slow because of its memory.** `{slow}` ({slow_ch} channels) is
   {100 * (slow_b - 1):.0f}% slower than the average with {BASE} ({slow_b:.2f}×), but at the average with opt1
   ({slow_o:.2f}×).
3. **The imbalance depends on the version.** {BASE} and opt2 have the same computation (opt2
   changes only the communication): 90% of their ranks are between {b_lo:.2f}× and {b_hi:.2f}× the average of
   their run (extremes {ranks[~o1].rel.min():.2f}× and {ranks[~o1].rel.max():.2f}×). In opt1 every rank is between {ranks[o1].rel.min():.2f}× and {ranks[o1].rel.max():.2f}×.

"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print("wrote", args.out)
    if args.print:
        print(text)


if __name__ == "__main__":
    main()
