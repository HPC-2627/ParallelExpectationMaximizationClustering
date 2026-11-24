#!/usr/bin/env python3
"""
clustering-summary.py

Summary table of the clustering runs (the table of doc/plots.md, section 3, instead of a plot):
one row per (dataset, model) at K = number of true classes, with N, D, K, final log-likelihood,
BIC, AIC, iterations, convergence and ARI. The clustering result is the same in every version, so
the table uses baseline-sequential (bin/PEMC-baseline-sequential-<model>), diagonal and full.

Usage (from the root of the repository):
    qsub src/script/pemc-cluster.sh                      # 1. the runs, on the cluster
    python3 src/tables/clustering-summary.py             # 2. the table
    python3 src/tables/clustering-summary.py --print     #    and also print it

Input
    results/logs/cluster/*.log      one block per run and per K (format: include/logger.h)
Output
    results/tables/clustering-summary.csv
        dataset, model, N, D, K, loglik, BIC, AIC, iterations, converged, ARI
        converged = 0: EM stopped at the maximum number of iterations (200) before the tolerance.
        ARI empty: the dataset has no true labels (parkinson).

BIC = -2 loglik + p ln N, AIC = -2 loglik + 2 p (the lower, the better), with p free parameters:
    diagonal  (K - 1) + K D + K D               weights, means, variances
    full      (K - 1) + K D + K D (D + 1) / 2   weights, means, covariance matrices
Same computation as src/plots-generate/plots-cluster.py (plots C2 and C6).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["diagonal", "full"]

# Number of true classes of every dataset = the K of the table (same list as pemc-cluster.sh;
# parkinson has no classes: K = 2 is an arbitrary choice). The order is the order of the rows.
TRUE_K = {"brainweb": 12, "cardiotocography": 10, "cardiovascular": 2, "cdc_diabetes": 2,
          "diabetes130": 3, "eeg_eye_state": 2, "gene_rnaseq": 5, "heart_failure": 2,
          "parkinson": 2, "statlog_heart": 2, "synth_clean_p01": 4, "synth_overlap_p01": 5}


def n_params(model, K, D):
    """Free parameters of the mixture: K-1 weights, K*D means, and the variances (diagonal: K*D)
    or the covariance matrices (full: K symmetric matrices, D(D+1)/2 distinct entries each)."""
    return (K - 1) + K * D + (K * D if model == "diagonal" else K * D * (D + 1) / 2)


def load_logs(logdir):
    """Every *.log of logdir -> one row per run ("key : value" blocks separated by "====")."""
    rows = []
    for path in sorted(logdir.glob("*.log")):
        run = None  # block being read
        # the extra "====" line at the end hands over the last block too
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
    for c in ("N", "D", "K", "iterations", "converged", "loglik", "ari"):  # "n/a" -> NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["model"] = df["version"].str.rsplit("-", n=1).str[1]       # baseline-sequential-<model>
    df["dataset"] = df["dataset"].map(lambda p: Path(p).stem)     # data/.../<name>.bin -> <name>
    p = df.apply(lambda r: n_params(r.model, r.K, r.D), axis=1)
    df["bic"] = -2 * df.loglik + p * np.log(df.N)
    df["aic"] = -2 * df.loglik + 2 * p
    # the runs are deterministic: if a (dataset, model, K) was run twice, keep the last
    return df.drop_duplicates(["dataset", "model", "K"], keep="last").reset_index(drop=True)


def summary(logs):
    """One row per (dataset, model) at the true K; the pairs without a run are reported."""
    rows, missing = [], []
    for ds, K in TRUE_K.items():
        for model in MODELS:
            g = logs[(logs.dataset == ds) & (logs.model == model) & (logs.K == K)]
            if g.empty:
                missing.append(f"{ds} {model}")
                continue
            r = g.iloc[0]
            rows.append(dict(dataset=ds, model=model, version=r.version.rsplit("-", 1)[0], N=int(r.N), D=int(r.D), K=int(r.K),
                             loglik=r.loglik, BIC=r.bic, AIC=r.aic,
                             iterations=int(r.iterations), converged=int(r.converged), ARI=r.ari))
    return pd.DataFrame(rows), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, default=ROOT / "results/logs/cluster")
    ap.add_argument("--out", type=Path, default=ROOT / "results/tables/clustering-summary.csv")
    ap.add_argument("--print", action="store_true", help="also print the table")
    args = ap.parse_args()

    table, missing = summary(load_logs(args.logs))
    if table.empty:
        sys.exit("no run at the true K")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(table)} rows)")
    if missing:  # gene_rnaseq full is expected: it is never run (does not fit in memory)
        print("no run at the true K for:", ", ".join(missing))
    if args.print:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
