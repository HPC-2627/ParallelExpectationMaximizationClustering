#!/usr/bin/env python3
"""
Local server for the EM visualization (src/visual-sequential/index.html).

  python3 src/visual-sequential/server.py  ->  http://127.0.0.1:8000

What it does for every run requested by the page:
  1. reads the first n rows of a .bin dataset (header int32 N, int32 D + float64 row-major);
  2. chooses the K initial rows with the SAME libc srand()/rand() calls used by gmm_init_indices()
     in src/common/gmm.c (through ctypes), so the starting point is the one of the C program;
  3. runs the same EM as src/PEMC-baseline-sequential-diagonal.c or
     src/PEMC-baseline-sequential-full.c (model=diagonal | full: same floors, same convergence
     test) and records every iteration;
  4. runs the REAL binary bin/PEMC-baseline-sequential-<model> with -v on the same data and
     returns its log-likelihood per iteration, so the page can show that the two agree.
The data have D dimensions: for the picture they are projected on the first two principal
components (the EM itself works in the full D dimensions).
"""
import ctypes
import json
import math
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

# ---------------------------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------------------------

# Repository root (this file is in src/visual-sequential/); data/ and bin/ are looked for from here
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
MODELS = ("diagonal", "full")


def binary(model):
    """Path of the C program of the given model."""
    return ROOT / "bin" / f"PEMC-baseline-sequential-{model}"
PAGE = Path(__file__).resolve().parent / "index.html"

# Same constants as include/common.h:
#   LOG_2PI  = log(2*pi), constant term of the log-density of a Gaussian
#   VAR_FLOOR = minimum variance (avoids division by zero / log(0) in the E-step)
#   NK_EMPTY = below this sum of responsibilities a cluster is considered empty
#   PI_FLOOR = minimum weight of a cluster (avoids log(0))
LOG_2PI = 1.8378770664093453
VAR_FLOOR, NK_EMPTY, PI_FLOOR = 1e-6, 1e-10, 1e-12
MAX_ROWS = 3000  # the trace is sent as JSON: keep it light

# The C library of the machine: calling ITS srand()/rand() gives exactly the numbers the C
# program gets (same libc, same algorithm), so the initial centroids are the same.
libc = ctypes.CDLL("libc.so.6")
libc.srand.argtypes = [ctypes.c_uint]
libc.rand.restype = ctypes.c_int


# ---------------------------------------------------------------------------------------------
# Reading the datasets
# ---------------------------------------------------------------------------------------------

def labels_path(p: Path):
    """Labels file of a dataset (<name>.labels.bin or <name>_labels.bin), or None."""
    for cand in (p.with_name(p.stem + ".labels.bin"), p.with_name(p.stem + "_labels.bin")):
        if cand.exists():
            return cand
    return None


def read_header(p: Path):
    """(N, D) from the 8-byte header of a .bin dataset (two int32; the data follow as float64)."""
    h = np.fromfile(p, dtype=np.int32, count=2)
    return int(h[0]), int(h[1])


def list_datasets():
    """All the .bin datasets under data/ with N, D and a suggested K (number of true classes)."""
    out = []
    for p in sorted(DATA.rglob("*.bin")):
        if "labels" in p.name:  # skip the label files, they are not datasets
            continue
        N, D = read_header(p)
        lp = labels_path(p)
        K = 2  # default when the dataset has no labels
        if lp:
            # K = number of distinct classes; only the first rows are read to stay fast
            lab = np.fromfile(lp, dtype=np.int32, count=min(N, 200000))
            K = int(len(np.unique(lab)))
        out.append({"path": str(p.relative_to(ROOT)), "N": N, "D": D, "K": K, "labels": bool(lp)})
    return out


# ---------------------------------------------------------------------------------------------
# EM (same maths as src/PEMC-baseline-sequential-diagonal.c, written with NumPy)
# ---------------------------------------------------------------------------------------------

def init_indices(N, K, seed):
    """Same as gmm_init_indices() in gmm.c: K DISTINCT row indices in [0, N), drawn with the
    libc rand() seeded by srand(seed)."""
    libc.srand(seed & 0xFFFFFFFF)
    idx = []
    while len(idx) < K:
        # rand() gives 31 bits: two calls are combined into 62 bits (order: high part first,
        # as in gmm.c) so that datasets with more than 2^31 rows can be indexed.
        hi = libc.rand()
        lo = libc.rand()
        cand = (((hi << 31) ^ lo) & 0xFFFFFFFFFFFFFFFF) % N
        if cand not in idx:  # rejection sampling: keep only indices not drawn yet
            idx.append(cand)
    return idx


def e_step(X, pi, mu, var):
    """DIAGONAL model. E-step: returns the log-densities logp[i,k] = log(pi_k N(x_i|mu_k,var_k)), their
    log-sum-exp per row (lse) and the log-likelihood ll = sum_i lse[i].
    The responsibilities are r_ik = exp(logp_ik - lse_i) (computed where needed)."""
    N, D = X.shape
    K = len(pi)
    # part of the log-density that depends only on the cluster (computed once per iteration)
    c = np.log(pi) - 0.5 * (D * LOG_2PI + np.log(var).sum(axis=1))
    logp = np.empty((N, K))
    for k in range(K):
        diff = X - mu[k]
        # squared Mahalanobis distance with diagonal covariance: sum_d (x_d - mu_d)^2 / var_d
        logp[:, k] = c[k] - 0.5 * ((diff * diff) @ (1.0 / var[k]))
    # log-sum-exp: subtract the row maximum before exponentiating (no underflow)
    m = logp.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True)))[:, 0]
    return logp, lse, float(lse.sum())


def m_step(X, logp, lse, pi, mu, var):
    """DIAGONAL model. M-step: updates pi, mu, var IN PLACE using the responsibilities as weights.
    Returns N_k (sum of the responsibilities of each cluster) and the number of empty clusters."""
    N = X.shape[0]
    r = np.exp(logp - lse[:, None])      # responsibilities, shape (N, K)
    # the three sums over the rows (what the parallel versions combine with MPI_Allreduce)
    Nk = r.sum(axis=0)                   # sum_i r_ik
    sx = r.T @ X                         # sum_i r_ik x_i
    sx2 = r.T @ (X * X)                  # sum_i r_ik x_i^2
    empty = 0
    for k in range(len(pi)):
        if Nk[k] < NK_EMPTY:
            # empty cluster: mean and variance stay as before (dividing by ~0 would give NaN)
            empty += 1
            pi[k] = PI_FLOOR
            continue
        mean = sx[k] / Nk[k]
        v = sx2[k] / Nk[k] - mean * mean   # variance = E[x^2] - E[x]^2
        mu[k] = mean
        var[k] = np.maximum(v, VAR_FLOOR)
        pi[k] = max(Nk[k] / N, PI_FLOOR)
    return Nk, empty


def cholesky_jitter(S):
    """Cholesky factor of S; if S is not positive definite a growing jitter is added to the
    diagonal (starting from VAR_FLOOR, x10), exactly like e_step() of the full C program."""
    jitter = 0.0
    while True:
        try:
            return np.linalg.cholesky(S + jitter * np.eye(len(S)))
        except np.linalg.LinAlgError:
            jitter = VAR_FLOOR if jitter == 0.0 else jitter * 10.0
            if jitter > 1e6:
                raise ValueError("covariance not positive definite")


def e_step_full(X, pi, mu, cov):
    """FULL model. Same outputs as e_step(). Sigma_k = L L^T; log det = 2 sum log L_dd and the
    squared Mahalanobis distance is |L^-1 (x - mu)|^2."""
    N, D = X.shape
    K = len(pi)
    logp = np.empty((N, K))
    for k in range(K):
        L = cholesky_jitter(cov[k])
        log_det = 2.0 * np.log(np.diag(L)).sum()
        z = (X - mu[k]) @ np.linalg.inv(L).T      # rows: L^-1 (x_i - mu_k)
        logp[:, k] = np.log(pi[k]) - 0.5 * (D * LOG_2PI + log_det) - 0.5 * (z * z).sum(axis=1)
    m = logp.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True)))[:, 0]
    return logp, lse, float(lse.sum())


def m_step_full(X, logp, lse, pi, mu, cov):
    """FULL model. M-step IN PLACE: Sigma_k = sum_i r_ik x x^T / N_k - mu mu^T (+ VAR_FLOOR on
    the diagonal). Returns N_k and the number of empty clusters."""
    N, D = X.shape
    r = np.exp(logp - lse[:, None])
    Nk = r.sum(axis=0)
    sx = r.T @ X
    empty = 0
    for k in range(len(pi)):
        if Nk[k] < NK_EMPTY:
            empty += 1
            pi[k] = PI_FLOOR
            continue
        mean = sx[k] / Nk[k]
        sxx = (X * r[:, k:k + 1]).T @ X            # sum_i r_ik x_i x_i^T
        S = sxx / Nk[k] - np.outer(mean, mean)
        S = (S + S.T) / 2.0 + VAR_FLOOR * np.eye(D)
        mu[k] = mean
        cov[k] = S
        pi[k] = max(Nk[k] / N, PI_FLOOR)
    return Nk, empty


def snapshot(pi, mu, var, mean, V):
    """Parameters as the page needs them, in the 2D plane of the picture.
    mean, V = mean of the data and the D x 2 matrix of the first two principal components.
    var is K x D (diagonal model) or K x D x D (full model: the covariance matrices).
      mu   -> projected means (K x 2)
      cov  -> projected covariance of each cluster V^T Sigma V, as [xx, xy, yy]
      var_mean -> average variance over the D variables (a single number per cluster)"""
    mu2 = (mu - mean) @ V
    full = var.ndim == 3
    cov = []
    for k in range(len(pi)):
        C = V.T @ (var[k] @ V if full else var[k][:, None] * V)
        cov.append([C[0, 0], C[0, 1], C[1, 1]])
    vm = np.trace(var, axis1=1, axis2=2) / var.shape[2] if full else var.mean(axis=1)
    return {"pi": pi.tolist(), "mu": mu2.round(5).tolist(), "cov": np.round(cov, 6).tolist(),
            "var_mean": vm.round(5).tolist()}


# ---------------------------------------------------------------------------------------------
# Cross-check with the real C program
# ---------------------------------------------------------------------------------------------

def run_c_binary(model, path, K, n, seed, max_iter, tol):
    """Runs bin/PEMC-baseline-sequential-<model> with the same settings and -v, and parses
    the log-likelihood printed at every iteration. Returns None if the binary is missing
    (run `make`) or fails. The page compares these values with its own."""
    exe = binary(model)
    if not exe.exists():
        return None
    cmd = [str(exe), "-d", path, "-k", str(K), "-n", str(n), "-s", str(seed),
           "-i", str(max_iter), "-t", repr(tol), "--no-log", "-v"]  # --no-log: do not touch the log file
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return None
    # lines like "  iteration   3  log-likelihood -1234.5  change +1.0e+01"
    lls = [float(m.group(1)) for m in re.finditer(r"iteration\s+\d+\s+log-likelihood\s+(\S+)", out)]
    # final summary line "log-likelihood  : -1234.5"
    m = re.search(r"log-likelihood\s*:\s*(\S+)", out)
    return {"ll": lls, "final_ll": float(m.group(1)) if m else None}


# ---------------------------------------------------------------------------------------------
# One complete run: this is what /api/run returns
# ---------------------------------------------------------------------------------------------

def run_trace(q):
    """Runs EM on the first n rows and returns the full trace for the page (see index.html).
    q = query parameters of the request (path, model, k, n, seed, max_iter, tol)."""
    # --- parameters, with the same defaults as the C program ---
    rel = q["path"]
    p = (ROOT / rel).resolve()
    # safety: only .bin files inside data/ can be read (no path tricks like ../)
    if DATA not in p.parents or p.suffix != ".bin" or not p.exists():
        raise ValueError("invalid dataset")
    model = q.get("model", "diagonal")
    if model not in MODELS:
        raise ValueError("invalid model")
    full = model == "full"
    K = int(q.get("k", 3))
    seed = int(q.get("seed", 42))
    max_iter = int(q.get("max_iter", 200))
    tol = float(q.get("tol", 1e-6))

    # --- data: only the first n rows (same as the -n option of the C program) ---
    N_file, D = read_header(p)
    n = min(int(q.get("n", 400)), N_file, MAX_ROWS)
    if K > n:
        raise ValueError("K larger than the number of rows")
    # memmap reads from the file only the n rows needed (offset 8 = skip the header)
    X = np.array(np.memmap(p, dtype=np.float64, mode="r", offset=8, shape=(n, D)))
    lp = labels_path(p)
    truth = np.fromfile(lp, dtype=np.int32, count=n).tolist() if lp else None

    # --- PCA for the picture only: the means and covariances are projected on the first two
    #     principal components (EM itself always uses all D dimensions) ---
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    V = Vt[:2].T                            # D x 2: directions of largest variance
    if V.shape[1] < 2:                      # D = 1 (brainweb): the second axis is flat
        V = np.hstack([V, np.zeros((D, 2 - V.shape[1]))])
    expl = float((S[:2] ** 2).sum() / max((S ** 2).sum(), 1e-300))   # fraction of variance kept
    pts = ((X - mean) @ V).round(5).tolist()                         # the points in the plane

    # --- initialization (gmm_init_indices + gmm_init_from_rows): means = K chosen rows,
    #     variance 1, equal weights ---
    idx = init_indices(n, K, seed)
    pi = np.full(K, 1.0 / K)
    mu = X[idx].copy()
    var = np.tile(np.eye(D), (K, 1, 1)) if full else np.ones((K, D))   # Sigma = I / var = 1
    e_fn, m_fn = (e_step_full, m_step_full) if full else (e_step, m_step)

    # --- main EM loop, same structure as run_em() in the C file: E-step, convergence test on
    #     the log-likelihood, M-step. One record per iteration is stored for the page. ---
    iters, ll_prev = [], -math.inf
    done = False
    for it in range(max_iter):
        logp, lse, ll = e_fn(X, pi, mu, var)
        # "params" = parameters USED by this E-step; logp lets the page recompute the
        # responsibilities of every point (rounded to keep the JSON small)
        rec = {"ll": ll, "params": snapshot(pi, mu, var, mean, V), "logp": logp.round(3).tolist(),
               "converged": False}
        # same test as gmm_converged(): |ll - ll_prev| <= tol * |ll| (never at the first iteration)
        if math.isfinite(ll_prev) and abs(ll - ll_prev) <= tol * abs(ll):
            rec["converged"] = True
            iters.append(rec)
            done = True
            break
        Nk, empty = m_fn(X, logp, lse, pi, mu, var)
        rec["Nk"], rec["empty"] = Nk.round(4).tolist(), empty
        rec["next"] = snapshot(pi, mu, var, mean, V)   # parameters AFTER the M-step
        iters.append(rec)
        ll_prev = ll
    if not done:  # same extra E-step as run_em() when max_iter is reached
        logp, lse, ll = e_fn(X, pi, mu, var)
        iters.append({"ll": ll, "params": snapshot(pi, mu, var, mean, V), "logp": logp.round(3).tolist(),
                      "converged": False, "final": True})

    return {"path": rel, "model": model, "N": n, "N_file": N_file, "D": D, "K": K, "seed": seed, "tol": tol,
            "max_iter": max_iter, "points": pts, "truth": truth, "init_idx": idx,
            "explained": expl, "iters": iters, "c_binary": run_c_binary(model, rel, K, n, seed, max_iter, tol)}


# ---------------------------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    """Three routes: / (the page), /api/datasets (list), /api/run (trace of one EM run)."""

    def _send(self, code, body, ctype):
        """Sends a complete response (status, headers, body)."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/datasets":
                self._send(200, json.dumps(list_datasets()).encode(), "application/json")
            elif u.path == "/api/run":
                # parse_qs gives lists ({"k": ["3"]}): keep the first value of each parameter
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                self._send(200, json.dumps(run_trace(q)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as e:  # any error is returned as JSON and shown by the page
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")

    def log_message(self, *a):
        pass  # no line per request on the terminal


if __name__ == "__main__":
    # optional argument: port (default 8000). Only 127.0.0.1: the server is not reachable
    # from other machines.
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"EM visualization: http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
