#!/usr/bin/env python3
"""
node-memory.py

Does the memory configuration of a node change the times? Table of all the nodes used by the
experiments (RAM, populated memory channels, DIMMs per channel) next to how fast the MPI ranks
that ran on them were, written to results/tables/node-memory.md. The report (Hardware
Infrastructure, and the Discussion on the load imbalance) links this table.

Usage (from the root of the repository):
    src/script/pemc-hwinfo.sh submit / summary     # 1. hardware of every node, on the cluster
    scp -r ...:ParallelExpectationMaximizationClustering/results/hwinfo results/   # 2. copy it
    python3 src/tables/node-memory.py               # 3. the table
    python3 src/tables/node-memory.py --print       #    and also print it

Input
    results/hwinfo/<node>.txt       one file per node, written by src/script/pemc-hwprobe.sh: the
                                    lines "MemTotal: ..." and "DIMM: <label> <size> <type>"
                                    (versioned; new nodes: copy them from the cluster)
    results/logs/pbs/*-scx-*.err    stderr of the multi-node jobs (scatter:excl): Open MPI writes
                                    "[<node>:<pid>] MCW rank <r> bound to ..." for every rank of
                                    every run (option --report-bindings of src/script/pemc-job.sh),
                                    i.e. on which node every rank ran
    results/logs/<model>/<stage>/*.log   the runs (format: include/logger.h); "rank_compute" =
                                    compute seconds of every rank, "hostname" = node of the run
Output
    results/tables/node-memory.md

How the memory channels are found (6 channels per socket on these CPUs, Xeon Gold 6252N)
    c12 nodes: label CPU_SrcID#<socket>_MC#<m>_Chan#<c>_DIMM#<d>  -> channel = (m, c), explicit
    c10/c11 nodes: label PROC_<socket>_DIMM_<slot>, slots 1..12, two slots per channel
                                                                  -> channel = (slot + 1) // 2
    populated channels of a socket = number of distinct channels with at least one DIMM. The ranks
    of every run are bound to socket 0 (see the MCW lines), so the table uses socket 0 (on these
    nodes all the sockets have the same configuration anyway).

What is compared (only the "long" datasets of the report, whose runs all last >= 1 s, the same
rule of src/plots-generate/plots-performance.py --min-run)
    1. Runs on several nodes (scatter:excl, 16 and 32 cores): the ranks of ONE run execute the same
       code on the same amount of data, but on different nodes. For every rank:
           relative time = its compute time / mean compute time of the ranks of its run
       1.00 = as fast as the average rank of its run, 1.10 = 10% slower. If the number of memory
       channels mattered, the ranks on nodes with fewer channels would be above 1.
    2. Runs on one node, 8 cores: baseline-MPI-OpenMP and Optimization 2 have the same computation (only
       the communication differs), so their compute time per iteration on the same dataset and
       layout can be compared between the nodes that ran them:
           relative time = compute time per iteration / median of the same (model, dataset,
                           ranks, threads, placement) over baseline-MPI-OpenMP and Optimization 2 jobs
       Every job runs on ONE node and every version ran on a few nodes only: here the node is
       mixed with the version, so this part is only an indication.
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
    rows["channels"] = rows["channels"].astype("Int64")  # integer, empty for a node without data
    return rows


def single_node_rows(df, nodes, keep):
    """Runs on one node, 8 cores, baseline-MPI-OpenMP and Optimization 2 (same computation): relative compute
    time per iteration of every run (see the docstring)."""
    d = df[(df.nodes == 1) & (df.cores == 8) & df.label.isin(["baseline", "opt2"])].copy()
    d = d[[(m, ds) in keep for m, ds in zip(d.model, d.dataset)]]
    d["node"] = d.hostname.str.replace(".unitn.it", "", regex=False)
    d["channels"] = d.node.map(lambda n: nodes.get(n, {}).get("channels")).astype("Int64")
    d["t"] = (d.t_estep + d.t_mstep) / d.iterations
    key = ["model", "dataset", "ranks", "threads", "placement"]
    d["rel"] = d.t / d.groupby(key).t.transform("median")
    return d


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------
def md_table(df):
    """DataFrame -> markdown table (no extra dependency)."""
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


def fmt(x, nd=3):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hwinfo", type=Path, default=ROOT / "results/hwinfo")
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables/node-memory.md")
    ap.add_argument("--print", action="store_true", help="also print the table")
    args = ap.parse_args()

    nodes = read_nodes(args.hwinfo)
    if not nodes:
        raise SystemExit(f"no node in {args.hwinfo}: copy results/hwinfo/ from the cluster")
    df, keep = long_datasets(args.logs)
    ranks = rank_rows(args.logs, args.logs / "pbs", nodes, keep)
    single = single_node_rows(df, nodes, keep)
    ranks["family"] = np.where(ranks.version == "opt1", "opt1", "baseline/opt2")

    # ---- 1. one row per node ----
    per_node = []
    for n, hw in sorted(nodes.items()):
        r = ranks[ranks.node == n]
        s = single[single.node == n]
        per_node.append({
            "node": n, "group": n.split("-")[1], "RAM (GiB)": f"{hw['ram']:.0f}",
            "channels / socket": hw["channels"], "DIMMs / channel": hw["dimms_per_channel"],
            "multi-node ranks": len(r),
            "rel. time, baseline-MPI-OpenMP/opt2": fmt(r[r.family == "baseline/opt2"].rel.mean()),
            "rel. time, opt1": fmt(r[r.family == "opt1"].rel.mean()),
            "single-node runs": len(s), "rel. time, single node": fmt(s.rel.mean())})
    per_node = pd.DataFrame(per_node)

    # ---- 2. multi-node: by version and number of channels ----
    by_ch = (ranks.groupby(["model", "version", "channels"]).rel
             .agg(ranks="size", mean="mean", min="min", max="max").reset_index())
    for c in ("mean", "min", "max"):
        by_ch[c] = by_ch[c].map(fmt)

    # ---- 3. multi-node: every job, its nodes and how fast their ranks were ----
    by_job = (ranks.groupby(["job", "node", "channels"]).rel.mean().map(fmt).reset_index()
              .rename(columns={"rel": "mean rel. time"}))

    # ---- 4. spread of the ranks inside one run, by family ----
    spread = (ranks.groupby(["model", "family"]).rel
              .agg(ranks="size", p05=lambda s: s.quantile(0.05), p95=lambda s: s.quantile(0.95),
                   min="min", max="max").reset_index())
    for c in ("p05", "p95", "min", "max"):
        spread[c] = spread[c].map(fmt)

    # ---- 5. single node: by number of channels ----
    single_ch = (single.groupby(["model", "label", "channels"]).rel
                 .agg(runs="size", nodes=lambda s: single.loc[s.index, "node"].nunique(), mean="mean")
                 .reset_index())
    single_ch["mean"] = single_ch["mean"].map(fmt)

    text = f"""# Memory configuration of the nodes and speed of the runs

Generated by `src/tables/node-memory.py` (do not edit by hand). Question: does the number of
populated memory channels of a node (2, 4 or 6 per socket, see `hardware.md`) change the times?
Only the datasets whose runs all last at least 1 s (the "long" datasets of the report).

**Relative time** of a rank = its compute time / mean compute time of the ranks of the same run
(1.00 = average rank of its run, 1.10 = 10% slower). The ranks of one run execute the same code on
the same amount of data, on different nodes: if the channels mattered, the ranks on the nodes with
fewer channels would be above 1.

## 1. Every node

`multi-node ranks`: ranks of the runs on 2 and 4 nodes (`scatter:excl`) that ran on the node.
`single-node runs`: runs on 8 cores of one node, baseline-MPI-OpenMP and Optimization 2 (same computation),
relative to the median of the same dataset and layout.

{md_table(per_node)}

## 2. Runs on several nodes, by number of channels

{md_table(by_ch)}

## 3. Runs on several nodes, every job (PBS name: version-scx-nodes x ranks per node x threads)

{md_table(by_job)}

## 4. Spread of the ranks inside one run

5th and 95th percentile, minimum and maximum of the relative time of the ranks.

{md_table(spread)}

## 5. Runs on one node, by number of channels

Every job runs on one node and every version ran on a few nodes only, so here the node is mixed
with the version: only an indication.

{md_table(single_ch)}
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print("wrote", args.out)
    if args.print:
        print(text)


if __name__ == "__main__":
    main()
