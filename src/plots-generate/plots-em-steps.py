#!/usr/bin/env python3
"""
plots-em-steps.py

Static figure of the report that shows how EM works, in a few steps: one run of the web
visualization (doc/exec-visual.md) on synth_clean (diagonal covariance, K = 4, first 300 rows,
seed 12), drawn at the initialization, after a few iterations and at convergence.
It is the figure "Visualization of the algorithm" of the report (fig:em-steps).

The EM is NOT reimplemented here: the trace comes from run_trace() of
src/visual-sequential/server.py (same maths, same initial rows and same convergence test as
src/diagonal/PEMC-baseline-sequential-diagonal.c and src/full/PEMC-baseline-sequential-full.c).

Usage (from the root of the repository):
    python3 src/plots-generate/plots-em-steps.py        # -> doc/img/em-steps.pdf (+ .png)
    python3 src/plots-generate/plots-em-steps.py --model full --steps 0 1 2 5
    python3 src/plots-generate/plots-em-steps.py --plane pca     # same plane as the web page

Inputs
    data/synthetic/synth_clean/synth_clean_p01.bin    the dataset (any .bin of data/ with --path)

Output
    doc/img/em-steps.pdf    vector version, included by the report (doc/latex/Group6-Report.tex)
    doc/img/em-steps.png    raster version (200 dpi), for the markdown docs

Every panel shows, for iteration t (t = number of M-steps done so far):
    points     colored by their responsibilities r_ik of the E-step of iteration t (mix of the
               cluster colors); in the first panel they are gray, nothing is assigned yet
    crosses    the means mu_k used by that E-step; the thin line is the path they followed
    ellipses   2-sigma contour of each cluster (covariance projected on the plane)
    title      the log-likelihood L of that E-step

The plane of the picture (EM itself always uses all D dimensions):
    sep  (default) the 2D projection where the two closest clusters found at convergence are as
         far apart as possible. With D = 10 the clusters of synth_clean are well separated, but
         in the PCA plane two of them overlap: this plane shows the separation EM really finds.
    lda  the Fisher LDA plane of the clusters found at convergence
    pca  the first two principal components (the plane of the web page)
Only the picture changes: the EM run is the same in the three cases.
"""

import argparse
import importlib.util
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------

# Repository root (this file is in src/plots-generate/)
ROOT = Path(__file__).resolve().parent.parent.parent
# The figure goes with the other images of the docs, where the report and the .md files find it
OUT = ROOT / "doc" / "img"

# ---------------------------------------------------------------------------------------------
# EM of the web visualization
# ---------------------------------------------------------------------------------------------

# server.py is a script, not a package (and its folder has a '-' in the name): it is loaded
# from its path. Importing it does NOT start the server (that happens only under __main__).
_spec = importlib.util.spec_from_file_location("em_server", ROOT / "src" / "visual-sequential" / "server.py")
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)

# run_trace() also runs the real C binary to compare the log-likelihoods: not needed for a
# picture (and it would fail if `make` was not run), so it is replaced by a function that does nothing
em.run_c_binary = lambda *a, **k: None

# snapshot() stores the parameters already projected on the PCA plane of the web page. To draw
# them on ANY plane we also need the D-dimensional ones: the original function is wrapped so that
# every snapshot also keeps a copy of mu (K x D) and of the variances (K x D, diagonal model) or
# covariance matrices (K x D x D, full model). The copy is needed because the M-step updates
# mu and var in place.
_snapshot = em.snapshot


def _snapshot_raw(pi, mu, var, mean, V):
    out = _snapshot(pi, mu, var, mean, V)
    out["mu_raw"], out["var_raw"] = mu.copy(), var.copy()
    return out


em.snapshot = _snapshot_raw

# ---------------------------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------------------------

# Cluster colors of doc/color.md (C3-C5), in the same order
CLASS_COLOR = ["#ff0000", "#22C55E", "#69d2ff", "#dca500", "#912603", "#006326",
               "#002fb0", "#7904d3", "#502B00", "#e44aff", "#ff8d35", "#2c79ff"]
# TEXT = text and axes, GRAY = points not assigned yet (initialization), GRID = background grid
TEXT, GRAY, GRID = "#171e26", "#9CA3AF", "#E5E7EB"

# Same look as plots-performance.py and plots-cluster.py: serif font, light gray grid under the
# data, no top/right frame, vector output with real fonts (fonttype 42)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

# Axis labels of each plane. The axes of "sep" have no meaning of their own (they are just the
# two directions found by the search), so the tick labels are hidden in every case.
AXIS = {"pca": ("PC1", "PC2"), "lda": ("LD1", "LD2"), "sep": ("direction 1", "direction 2")}


# ---------------------------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------------------------

def ellipse(ax, mu, c, color, nsig=2.0):
    """Draws the nsig-sigma contour of a 2D Gaussian with mean mu and covariance c = [xx, xy, yy].
    The axes of the ellipse are the eigenvectors of the covariance and the half-lengths are
    nsig * sqrt(eigenvalue)."""
    C = np.array([[c[0], c[1]], [c[1], c[2]]])
    w, V = np.linalg.eigh(C)              # eigenvalues in ascending order: w[1] is the largest
    w = np.maximum(w, 1e-12)              # guard against tiny negative values from rounding
    angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))   # direction of the major axis
    ax.add_patch(Ellipse(mu, 2 * nsig * np.sqrt(w[1]), 2 * nsig * np.sqrt(w[0]), angle=angle,
                         fill=False, edgecolor=color, linewidth=1.3))


# ---------------------------------------------------------------------------------------------
# Choice of the plane (only for the picture)
# ---------------------------------------------------------------------------------------------

def lda_plane(X, y):
    """D x 2 orthonormal basis of the Fisher LDA plane of the groups y: the two directions with
    the largest ratio between-group variance / within-group variance (top-2 eigenvectors of
    Sw^-1 Sb)."""
    m = X.mean(axis=0)
    Sw = np.zeros((X.shape[1], X.shape[1]))   # within-group scatter
    Sb = np.zeros_like(Sw)                    # between-group scatter
    for c in np.unique(y):
        A = X[y == c]
        mc = A.mean(axis=0)
        Sw += (A - mc).T @ (A - mc)
        Sb += len(A) * np.outer(mc - m, mc - m)
    # tiny ridge on Sw so that the solve never fails (e.g. a constant variable)
    w, V = np.linalg.eig(np.linalg.solve(Sw + 1e-9 * np.eye(len(Sw)), Sb))
    V = np.real(V[:, np.argsort(-np.real(w))[:2]])
    return np.linalg.qr(V)[0]                 # orthonormal basis: no distortion of the picture


def min_separation(P, y):
    """Separation of the two CLOSEST groups of y in the projection P (n x 2). For each pair of
    groups: distance between the two means divided by the sum of the standard deviations of the
    two groups along the line that joins them (> 1 roughly means that the two clouds do not
    overlap). Returns the minimum over all the pairs."""
    best = np.inf
    groups = [P[y == c] for c in np.unique(y)]
    for A, B in itertools.combinations(groups, 2):
        d = B.mean(axis=0) - A.mean(axis=0)
        u = d / np.linalg.norm(d)             # unit vector from mean A to mean B
        best = min(best, np.linalg.norm(d) / ((A @ u).std() + (B @ u).std()))
    return best


def sep_plane(X, y, seed=0, n_random=5000, n_refine=3000):
    """D x 2 orthonormal basis of the plane where the two closest groups of y are as far apart as
    possible (maximizes min_separation). LDA maximizes an AVERAGE separation and can still leave
    two groups on top of each other, so a direct search is used:
      1. candidates: the LDA plane and n_random random planes; the best one is kept;
      2. local refinement: n_refine small random rotations of the best plane, keeping only the
         improvements; the size of the step is halved every 500 attempts.
    The generator has a fixed seed, so the figure is always the same."""
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=0)
    # 1. starting point: the LDA plane, then random orthonormal planes (QR of a Gaussian matrix)
    best = lda_plane(X, y)
    best_s = min_separation(Xc @ best, y)
    for _ in range(n_random):
        V = np.linalg.qr(rng.normal(size=(X.shape[1], 2)))[0]
        s = min_separation(Xc @ V, y)
        if s > best_s:
            best, best_s = V, s
    # 2. local search around the best plane (perturb, re-orthonormalize, keep if better)
    step = 0.3
    for i in range(n_refine):
        V = np.linalg.qr(best + step * rng.normal(size=best.shape))[0]
        s = min_separation(Xc @ V, y)
        if s > best_s:
            best, best_s = V, s
        if i % 500 == 499:
            step /= 2
    return best


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main():
    # --- parameters: the defaults are the run of the report ---
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default="data/synthetic/synth_clean/synth_clean_p01.bin",
                    help="dataset (.bin under data/), relative to the root of the repository")
    ap.add_argument("--model", default="diagonal", choices=em.MODELS)
    ap.add_argument("-k", type=int, default=4, help="number of clusters")
    ap.add_argument("-n", type=int, default=300, help="number of rows used (the first n)")
    ap.add_argument("--seed", type=int, default=12, help="seed of the initial rows (as the C program)")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 3, -1],
                    help="iterations to draw, one panel each (0 = initialization, -1 = last)")
    ap.add_argument("--plane", default="sep", choices=["sep", "lda", "pca"],
                    help="sep: plane where the two closest clusters found at convergence are as "
                         "far apart as possible; lda: Fisher LDA plane of those clusters; "
                         "pca: first two principal components (as the web page)")
    ap.add_argument("--out", default=str(OUT / "em-steps.pdf"),
                    help="output file (.pdf); a .png with the same name is also written")
    a = ap.parse_args()

    # --- one complete EM run, recorded iteration by iteration (same function as the web page).
    #     iters[t]["params"] = parameters used by the E-step of iteration t (t = 0: initial ones),
    #     iters[t]["logp"]   = log(pi_k N(x_i | mu_k, Sigma_k)) of that E-step, iters[t]["ll"] = L ---
    tr = em.run_trace({"path": a.path, "model": a.model, "k": a.k, "n": a.n, "seed": a.seed})
    iters = tr["iters"]
    K = tr["K"]
    full = a.model == "full"

    # --- the same n rows used by run_trace (memmap reads only them; offset 8 = skip the header) ---
    X = np.array(np.memmap(em.ROOT / a.path, dtype=np.float64, mode="r", offset=8, shape=(tr["N"], tr["D"])))
    mean = X.mean(axis=0)

    # --- plane of the picture (D x 2 orthonormal basis V) ---
    if a.plane == "pca":
        V = np.linalg.svd(X - mean, full_matrices=False)[2][:2].T
    else:
        # lda and sep need groups: the hard assignment at convergence (cluster with the largest r_ik)
        y = np.array(iters[-1]["logp"]).argmax(axis=1)
        V = lda_plane(X, y) if a.plane == "lda" else sep_plane(X, y)
    P = (X - mean) @ V                                               # the points in the plane
    expl = ((P - P.mean(axis=0)) ** 2).sum() / ((X - mean) ** 2).sum()   # fraction of variance kept

    def proj(params):
        """Means (K x 2) and covariances [xx, xy, yy] of the clusters projected on the plane:
        mu -> V^T (mu - mean), Sigma -> V^T Sigma V (diagonal model: Sigma = diag(var))."""
        mu2 = (params["mu_raw"] - mean) @ V
        cov = []
        for k in range(K):
            S = params["var_raw"][k] if full else np.diag(params["var_raw"][k])
            C = V.T @ S @ V
            cov.append([C[0, 0], C[0, 1], C[1, 1]])
        return mu2, cov

    cols = np.array([matplotlib.colors.to_rgb(c) for c in CLASS_COLOR[:K]])   # K x 3 (RGB)
    steps = [s if s >= 0 else len(iters) - 1 for s in a.steps]                 # -1 -> last iteration

    # same axis limits in every panel (computed from all the points), so the movement of the
    # means from one panel to the next is visible
    pad = 0.08 * (P.max(axis=0) - P.min(axis=0))
    lo, hi = P.min(axis=0) - pad, P.max(axis=0) + pad

    # --- one panel per requested iteration ---
    fig, axs = plt.subplots(1, len(steps), figsize=(1.9 * len(steps) + 0.4, 2.3), sharex=True, sharey=True)
    for j, (ax, t) in enumerate(zip(np.atleast_1d(axs), steps)):
        rec = iters[t]

        # responsibilities of this E-step, r_ik = exp(logp_ik - log sum_k exp(logp_ik)),
        # computed with the max subtracted (same log-sum-exp trick as the C code)
        logp = np.array(rec["logp"])
        r = np.exp(logp - logp.max(axis=1, keepdims=True))
        r /= r.sum(axis=1, keepdims=True)

        # color of each point = mix of the cluster colors weighted by r_ik (a point shared by two
        # clusters gets an intermediate color); gray at the initialization, nothing assigned yet
        pc = GRAY if t == 0 else np.clip(r @ cols, 0, 1)
        ax.scatter(P[:, 0], P[:, 1], s=6, c=pc, alpha=0.75, linewidths=0)

        # means, their path from the initialization up to iteration t, and the 2-sigma ellipses
        path = np.array([proj(iters[s]["params"])[0] for s in range(t + 1)])   # (t+1) x K x 2
        mus, covs = proj(rec["params"])
        for k in range(K):
            if t > 0:
                ax.plot(path[:, k, 0], path[:, k, 1], color=CLASS_COLOR[k], lw=0.8, alpha=0.8)
            ellipse(ax, mus[k], covs[k], CLASS_COLOR[k])
            ax.scatter(*mus[k], marker="X", s=55, color=CLASS_COLOR[k], edgecolors="white",
                       linewidths=0.7, zorder=5)

        # title: (a), (b), ... + iteration + log-likelihood of this E-step
        if t == 0:
            title = "(a) initialization"
        else:
            last = t == len(iters) - 1
            title = f"({chr(97 + j)}) iteration {t}" + (" (converged)" if last and rec.get("converged") else "")
        ax.set_title(f"{title}\n$\\mathcal{{L}}={rec['ll']:.0f}$")

        # the coordinates in the plane have no physical meaning: no tick labels
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_xlabel(AXIS[a.plane][0])
    np.atleast_1d(axs)[0].set_ylabel(AXIS[a.plane][1])

    # --- output: PDF for the report, PNG for the markdown docs ---
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"{out}  ({len(iters) - 1} iterations, {100 * expl:.0f}% of the variance in the {a.plane} plane)")


if __name__ == "__main__":
    main()
