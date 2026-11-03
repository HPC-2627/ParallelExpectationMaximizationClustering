#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator of the synthetic datasets (medical domain) for the weak scaling of the EM.

CHARACTERISTICS
Generates 3 datasets of fake "patients", each one with a different profile:
  synth_clean   K=4  D=10  well separated, balanced clusters, diagonal covariance
                           (it satisfies the assumptions of the model: converges in few iterations)
  synth_overlap K=5  D=20  overlapping clusters, correlated variables, some skewed variables
                           (glucose, creatinine...), slightly unbalanced weights
  synth_rare    K=6  D=50  a rare subgroup (1%), 2% of anomalous values (outliers),
                           high D (tests numerical stability and a different amount of computation)

-----------------------------------------------------------------------------------------------------------------
WHY SYNTHETIC DATA AND WHY THESE THREE PROFILES
- Real medical datasets are almost always small (a few hundred rows): with so little data the
  communication weighs more than the computation and no scaling can be observed. For the weak
  scaling N must be able to grow at will, which is possible only with generated data.
- Generating the data means knowing the TRUTH: the true subgroup of every patient and the true
  parameters. It is therefore possible to check that all the versions (sequential, baseline,
  optimizations) give the same result and to measure how well the EM finds the groups.
- The three profiles are chosen to cover different situations (see PROFILES): the "easy" case
  that satisfies the model, the "realistic" case (overlapping and correlated) and the "hard"
  case (rare group, outliers, high D). K and D also change, because the cost of an iteration is
  roughly proportional to N*K*D, so different computation regimes are observed.

-------------------------------------------------------------------------------------------------------------------
GENERATION
Every patient is drawn from a MIXTURE OF GAUSSIANS (the same family of models that the EM tries
to reconstruct):
  1. the subgroup k is chosen with probability pi_k (the weights);
  2. the point is drawn from a Gaussian with mean mu_k and covariance Sigma_k;
  3. (only for some profiles) the variables are transformed into clinical units: skewness and
     physiological limits;
  4. the columns are standardized (z-score), because the algorithm works on normalized data.
The true means and covariances are saved separately, so it is possible to check how well the EM
finds them.

------------------------------------------------------------------------------------------------------------------
WEAK SCALING
In the weak scaling the size of the problem grows with the resources: with p processes
N = N0 * p rows are used (N0 = rows per process).
For every profile a SEPARATE FILE is created for each number of processes p, for example
synth_clean_p01.bin, synth_clean_p02.bin, synth_clean_p04.bin, ...: each file contains exactly
N0 * p rows, so the program only has to be given the right file, without having to remember how
many rows to use. It is the clearest choice for the repository and for the explanation.
Rank r reads rows from r*N0 to (r+1)*N0 - 1 of the file: every rank always has N0 rows.
The files of the same profile are "nested": the rows are drawn from a single random stream and
written to all the files that require them, so the file for p=2 contains the rows of the file
for p=1 plus other N0 rows, and so on. Going from p to 2p only the AMOUNT of data changes and not
the sample, so the comparison between the sizes is cleaner.
Cost: the files of a profile together take about twice the size of the largest one
(1+2+4+8+16 = 31 versus 16), that is about 5 GB in total for the three profiles with the
default parameters.

-------------------------------------------------------------------------------------------------------------------------
FILE FORMAT (in data/synthetic/<profile>/, one folder per profile:
  synth_clean/, synth_overlap/, synth_rare/)
  <name>_pPP.bin         (PP = number of processes, two digits: p01, p02, p04, p08, p16)
                     header: 2 32-bit integers (N, D), then N*D double numbers (8 bytes)
                     in row-major order (row 0, row 1, ...), little-endian.
                     In C:  fread(&N,4,1,f); fread(&D,4,1,f); fread(X, 8, N*D, f);
  <name>_pPP.labels.bin  N 32-bit integers: the true subgroup of every row (-1 = outlier).
                     Used only to check the results, NOT by the EM.
  <name>.json        ONE per profile: profile parameters, names of the variables, means and
                     standard deviations used for the standardization (the same for all the
                     files of the profile), and for every p the name of the file, N, counts
                     and empirical means of the clusters.

Why this format:
  - binary and not text (CSV): it is read with a single fread, without converting text into
    numbers, and the file is much smaller (8 bytes per value instead of 15-20 characters);
  - double (64 bit) and not float: the EM sums many numbers and computes exp/log, the double
    precision avoids losses of accuracy that would make the parallel versions differ;
  - row-major order: the points of a rank are a contiguous block of the file, so every rank can
    read only its own rows (fseek or MPI-IO) without anyone reading everything;
  - little-endian: it is the byte order of the x86 processors of the cluster nodes;
  - 8-byte header (two int32): it is enough to describe the file and it keeps the doubles
    aligned to 8 bytes;
  - labels in a separate file: the EM must not be able to "see" the truth and the data file
    remains a clean matrix of numbers only;
  - JSON: readable text that documents the dataset and contains the truth for the checks.

--------------------------------------------------------------------------------------------------------------------------
REPRODUCIBILITY
With the same seed and the same parameters identical files are obtained: every block of rows
uses a random generator derived from (seed, profile, block number).

USAGE
  python3 src/dataset-managed/generate-synthetic-dataset.py                       # the 3 profiles, default parameters
  python3 src/dataset-managed/generate-synthetic-dataset.py --n-per-rank 100000 --procs 1 2 4 8
  python3 src/dataset-managed/generate-synthetic-dataset.py --profiles synth_clean
"""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------
# GENERAL PARAMETERS
# ----------------------------------------------------------------------------------------

# Rows of the "pilot" sample used to estimate the mean and the standard deviation of every
# column. They are needed to standardize ALL the blocks (and all the files) with the same
# constants (otherwise every block would have a different normalization) without re-reading
# the file after writing it. 200 000 rows are enough: the error on the mean is about
# 1/sqrt(200000), that is 0.2% of the standard deviation.
PILOT_N = 200_000

# Standard deviation (in the latent space) of the noise used for the outliers. With 4.0 the
# outliers fall on average 3-4 standard deviations away from the center of the data: they are
# clearly anomalous (measurement artifacts, transcription errors) but not absurd.
OUTLIER_SCALE = 4.0

# Number of latent factors of the model used to create the correlations between variables.
# Few factors (3) produce a low-rank correlation structure, typical of clinical data: many
# variables (pressure, BMI, glucose...) depend on a few common "reasons" (age, weight,
# metabolic state).
N_FACTORS = 3

# Clinical variables: (name, mean, standard deviation, minimum, maximum, skewed)
#  - mean and standard deviation are typical values of an adult population;
#  - minimum and maximum are the plausible physiological limits (used only if the profile has
#    clip=True): they avoid impossible values, for example a negative pressure;
#  - skewed=True marks the variables with a long right tail (glucose, creatinine...): in some
#    profiles they are generated with a log-normal distribution, because in real data these
#    measurements have many normal values and a few very high ones.
# There are 20 variables: enough for the profiles with D <= 20. For a higher D generic columns
# 'biomarker_XX' are added (see variable_info).
CLINICAL_VARIABLES = [
    ("age",           55.0, 15.0,  18.0, 100.0, False),   # years
    ("bmi",           27.0,  5.0,  14.0,  60.0, False),   # kg/m^2
    ("sys_bp",       130.0, 20.0,  80.0, 230.0, False),   # mmHg, systolic pressure
    ("dia_bp",        80.0, 11.0,  40.0, 130.0, False),   # mmHg, diastolic pressure
    ("glucose",      105.0, 30.0,  50.0, 400.0, True),    # mg/dL
    ("hba1c",          5.8,  1.0,   4.0,  14.0, True),    # %
    ("cholesterol",  200.0, 40.0,  90.0, 400.0, False),   # mg/dL
    ("hdl",           52.0, 14.0,  20.0, 110.0, False),   # mg/dL
    ("creatinine",     1.0,  0.3,   0.3,   6.0, True),    # mg/dL
    ("heart_rate",    75.0, 12.0,  40.0, 150.0, False),   # beats/min
    ("ldl",          120.0, 35.0,  30.0, 300.0, False),   # mg/dL
    ("triglycerides",140.0, 70.0,  30.0, 800.0, True),    # mg/dL
    ("albumin",        4.2,  0.4,   2.0,   5.5, False),   # g/dL
    ("hemoglobin",    13.5,  1.8,   6.0,  19.0, False),   # g/dL
    ("wbc",            7.0,  2.0,   2.0,  25.0, True),    # 10^3/uL, white blood cells
    ("platelets",    250.0, 60.0,  50.0, 600.0, False),   # 10^3/uL
    ("sodium",       140.0,  3.0, 120.0, 160.0, False),   # mmol/L
    ("potassium",      4.2,  0.4,   2.5,   6.5, False),   # mmol/L
    ("alt",           28.0, 15.0,   5.0, 200.0, True),    # U/L, liver enzyme
    ("crp",            3.0,  3.0,   0.1, 100.0, True),    # mg/L, C-reactive protein
]

# The three profiles. Keys:
#   K, D         number of clusters and of variables
#   weights      share of patients of every subgroup (sum = 1)
#   delta        distance (in units of standard deviation) between the means of two clusters:
#                6 = very well separated, 2.5 = overlapping
#   cov          "diag" = diagonal covariance, "corr" = correlated variables inside the cluster
#   std_range    ("diag" only) interval from which the standard deviation of each variable is drawn
#   corr_strength ("corr" only) intensity of the correlations
#   outliers     fraction of anomalous points (label -1)
#   skew         True = "skewed" variables generated as log-normal
#   clip         True = values limited to the physiological interval
PROFILES = {
    # EASY CASE, it satisfies the assumptions of the model (Gaussians with diagonal covariance).
    #  - K=4, D=10: small dimensions, little computation per point (memory-bound regime);
    #  - equal weights: no group dominates, no group is rare;
    #  - delta=6: very well separated groups, the EM converges in few iterations and finds them
    #    almost all: it is the reference profile to check correctness and for the scaling;
    #  - std_range=(1,1): same standard deviation in all the variables, the simplest case;
    #  - no outliers, no skewness, no limits: pure Gaussian data.
    "synth_clean": dict(
        K=4, D=10, weights=[0.25, 0.25, 0.25, 0.25], delta=6.0,
        cov="diag", std_range=(1.0, 1.0), outliers=0.0, skew=False, clip=False),

    # REALISTIC CASE: real clinical data have overlapping groups and correlated variables.
    #  - K=5, D=20: more work per point than the easy case;
    #  - decreasing weights (30%...10%): groups of different sizes, as in real populations;
    #  - delta=2.5: overlapping groups, the EM converges slowly and can make mistakes (the BIC
    #    also has a real job in choosing K);
    #  - cov="corr", corr_strength=0.5: correlated variables inside every cluster (age with
    #    pressure, BMI with glucose): the diagonal model is therefore slightly WRONG, and this
    #    must be stated in the report as a limit;
    #  - skew=True, clip=True: skewed and limited variables, like the real ones.
    "synth_overlap": dict(
        K=5, D=20, weights=[0.30, 0.25, 0.20, 0.15, 0.10], delta=2.5,
        cov="corr", corr_strength=0.5, outliers=0.0, skew=True, clip=True),

    # HARD CASE: rare group, outliers, many variables.
    #  - K=6, D=50: high D, so more computation per point (regime closer to compute-bound);
    #  - weights with a group at 1%: a "rare" group (rare disease) that the EM struggles to keep
    #    and that can make the variance collapse (tests the regularization epsilon);
    #  - delta=4: intermediate separation;
    #  - std_range=(0.6,1.4): different standard deviations across variables and clusters;
    #  - outliers=2%: measurement errors or artifacts, marked with label -1;
    #  - clip=True: extreme values are limited to the physiological interval.
    # NOTE: the imbalance of the cluster sizes does NOT unbalance the work between the ranks
    # (the cost of a point is the same for all): this profile tests numerical stability, not
    # load balancing.
    "synth_rare": dict(
        K=6, D=50, weights=[0.40, 0.25, 0.18, 0.12, 0.04, 0.01], delta=4.0,
        cov="diag", std_range=(0.6, 1.4), outliers=0.02, skew=False, clip=True),
}


# ----------------------------------------------------------------------------------------
# VARIABLES
# ----------------------------------------------------------------------------------------

def variable_info(j):
    """Returns (name, mean, std. dev., minimum, maximum, skewed) of column j.
    The first columns are the clinical variables; the others are generic 'biomarker_XX'
    (mean 0, standard deviation 1, no limits) to reach D variables."""
    if j < len(CLINICAL_VARIABLES):
        return CLINICAL_VARIABLES[j]
    return (f"biomarker_{j - len(CLINICAL_VARIABLES) + 1:02d}", 0.0, 1.0, -np.inf, np.inf, False)


# ----------------------------------------------------------------------------------------
# MODEL: true means and covariances of every subgroup
# ----------------------------------------------------------------------------------------

def build_model(p, rng):
    """Draws ONCE the true parameters of the profile.

    Returns:
      mu  (K, D)    mean of every cluster
      L   (K, D, D) Cholesky factor of the covariance of every cluster (Sigma_k = L L^T);
                    to draw a point it is enough to compute mu_k + L_k @ z, with z standard Gaussian.
    """
    K, D = p["K"], p["D"]

    # Means: every coordinate is Gaussian with standard deviation s = delta / sqrt(2 D).
    # Chosen because the distance between two means drawn in this way is about
    # sqrt(2 D) * s = delta (in standard deviations inside the cluster), whatever D is:
    # so "delta" has the same meaning (separation of the groups) for all the profiles.
    mu = rng.normal(0.0, p["delta"] / np.sqrt(2.0 * D), size=(K, D))

    L = np.zeros((K, D, D))
    for k in range(K):
        if p["cov"] == "corr":
            # Random correlation matrix with a factor model:
            #   C = I + strength * B B^T   (always positive definite, so it has a Cholesky factor)
            # and then normalized to a correlation (diagonal = 1). Every cluster has its own,
            # so the groups differ also in shape and not only in position.
            B = rng.standard_normal((D, N_FACTORS))
            C = np.eye(D) + p["corr_strength"] * (B @ B.T)
            d = np.sqrt(np.diag(C))
            R = C / np.outer(d, d)
            L[k] = np.linalg.cholesky(R)
        else:
            # Diagonal covariance: one standard deviation per variable and per cluster.
            # It is still stored as a diagonal DxD matrix, so the code that draws the points
            # is the same for both types of covariance.
            s = rng.uniform(p["std_range"][0], p["std_range"][1], size=D)
            L[k] = np.diag(s)
    return mu, L


def to_clinical_units(Z, p):
    """Transforms the 'latent' values Z (Gaussian) into values with clinical units.

    - normal variable:  value = mean + std * z
    - skewed variable:  value = mean * exp((std / mean) * z)   (log-normal, always positive,
                        with a long right tail: like real glucose and creatinine)
    - if clip=True the values are limited to the physiological interval.
    """
    X = np.empty_like(Z)
    for j in range(Z.shape[1]):
        _, mean, sd, minimum, maximum, skewed = variable_info(j)
        if p["skew"] and skewed:
            col = mean * np.exp((sd / mean) * Z[:, j])
        else:
            col = mean + sd * Z[:, j]
        if p["clip"]:
            col = np.clip(col, minimum, maximum)
        X[:, j] = col
    return X


def generate_block(p, mu, L, rng, n):
    """Generates n patients. Returns (X in clinical units, true labels)."""
    K, D = p["K"], p["D"]

    # 1) subgroup of every patient, drawn with probability pi_k
    labels = rng.choice(K, size=n, p=np.asarray(p["weights"])).astype(np.int32)

    # 2) point drawn from the Gaussian of its subgroup: mu_k + z @ L_k^T
    #    (done for the groups of rows of the same cluster, much faster than row by row)
    Z = np.empty((n, D))
    for k in range(K):
        idx = np.flatnonzero(labels == k)
        if idx.size:
            Z[idx] = mu[k] + rng.standard_normal((idx.size, D)) @ L[k].T

    # 3) outliers: some points are replaced by wide noise and marked with -1
    if p["outliers"] > 0:
        mask = rng.random(n) < p["outliers"]
        Z[mask] = rng.normal(0.0, OUTLIER_SCALE, size=(int(mask.sum()), D))
        labels[mask] = -1

    # 4) conversion into clinical units (skewness and physiological limits)
    return to_clinical_units(Z, p), labels


# ----------------------------------------------------------------------------------------
# GENERATION OF A PROFILE AND WRITING ON DISK
# ----------------------------------------------------------------------------------------

def update_statistics(sums, counts, Xn, labels):
    """Adds the rows Xn (already standardized) to the sums per cluster, later used to compute
    the true empirical means of every subgroup."""
    for k in np.unique(labels):
        sel = labels == k
        sums[int(k)] += Xn[sel].sum(axis=0)
        counts[int(k)] += int(sel.sum())


def generate_profile(name, index, p, args, out_dir):
    K, D = p["K"], p["D"]
    N0 = args.n_per_rank
    procs = sorted(set(args.procs))              # numbers of processes of the weak scaling

    # One subfolder per profile (data/synthetic/synth_clean/, synth_overlap/, synth_rare/):
    # the files of different profiles are not mixed and every folder is self-contained.
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)

    N_max = N0 * procs[-1]                       # rows of the largest file
    if N_max >= 2**31:
        # N is written as a 32-bit integer in the header: it cannot exceed 2^31 - 1
        raise SystemExit(f"{name}: too many rows ({N_max}) for a 32-bit integer")
    total_gb = sum(8 + N0 * pp * D * 8 for pp in procs) / 1e9
    print(f"\n=== {name}: K={K}, D={D}, {len(procs)} files (p = {', '.join(map(str, procs))}), "
          f"up to {N_max} rows, about {total_gb:.2f} GB in total")

    # Independent and reproducible random generators, derived from (seed, profile, purpose):
    #   purpose 0 -> parameters of the model; purpose 1 -> pilot sample; purpose 2+i -> block i.
    # So every block has its own stream of random numbers and the result does not depend on the
    # order in which the profiles are generated.
    rng_model = np.random.default_rng([args.seed, index, 0])
    mu, L = build_model(p, rng_model)

    # Pilot sample: used to compute the mean and the standard deviation of every column, used to
    # standardize all the blocks (and all the files) with the SAME constants.
    rng_pilot = np.random.default_rng([args.seed, index, 1])
    X_pilot, _ = generate_block(p, mu, L, rng_pilot, PILOT_N)
    mean = X_pilot.mean(axis=0)
    std = X_pilot.std(axis=0)
    std[std == 0] = 1.0                          # avoids divisions by zero

    start_time = time.time()
    with ExitStack() as stack:
        # A data file and a labels file for every p, opened together. The file for p contains
        # the first N0 * p rows: every generated block is written to all the files that
        # require it, so the files are nested (see WEAK SCALING at the top).
        files = {}
        for pp in procs:
            n_pp = N0 * pp
            name_pp = f"{name}_p{pp:02d}"
            fb = stack.enter_context(open(folder / f"{name_pp}.bin", "wb"))
            fl = stack.enter_context(open(folder / f"{name_pp}.labels.bin", "wb"))
            # Header: N and D as little-endian 32-bit integers ("<i4")
            fb.write(np.array([n_pp, D], dtype="<i4").tobytes())
            files[pp] = dict(
                name=name_pp, N=n_pp, fb=fb, fl=fl,
                sums={k: np.zeros(D) for k in range(-1, K)},      # sums per cluster
                counts={k: 0 for k in range(-1, K)},
            )

        row, block = 0, 0                        # first global row of the block, block number
        while row < N_max:
            n = min(args.chunk, N_max - row)
            rng = np.random.default_rng([args.seed, index, 2 + block])
            X, labels = generate_block(p, mu, L, rng, n)
            Xn = (X - mean) / std                        # standardization (z-score)

            for pp in procs:
                f = files[pp]
                # rows of this block that belong to the file of p (the first f["N"] of the stream)
                m = min(n, f["N"] - row)
                if m <= 0:
                    continue
                f["fb"].write(Xn[:m].astype("<f8").tobytes())       # little-endian double, row-major
                f["fl"].write(labels[:m].astype("<i4").tobytes())   # little-endian int32
                update_statistics(f["sums"], f["counts"], Xn[:m], labels[:m])

            row += n
            block += 1
            print(f"    {row:>10}/{N_max} rows ({time.time() - start_time:.1f} s)", end="\r")
    print()

    # One JSON per profile with parameters and information useful to who uses the datasets
    variable_names = [variable_info(j)[0] for j in range(D)]
    info = dict(
        name=name, K=K, D=D, seed=args.seed,
        profile={k: v for k, v in p.items()},
        variables=variable_names,
        standardization=dict(mean=mean.tolist(), std=std.tolist()),
        rows_per_process=N0,
        files={
            str(pp): dict(
                data_file=files[pp]["name"] + ".bin",
                labels_file=files[pp]["name"] + ".labels.bin",
                N=files[pp]["N"],
                counts_per_cluster={str(k): v for k, v in files[pp]["counts"].items() if v > 0},
                empirical_means_standardized={
                    str(k): (files[pp]["sums"][k] / files[pp]["counts"][k]).tolist()
                    for k in files[pp]["sums"] if files[pp]["counts"][k] > 0},
            )
            for pp in procs
        },
        note="the file of p processes has N0*p rows; rank r reads rows from r*N0 to (r+1)*N0-1",
    )
    with open(folder / f"{name}.json", "w") as f:
        json.dump(info, f, indent=2)

    for pp in procs:
        print(f"  p={pp:<2} {files[pp]['name']}.bin")
        verify_file(folder / f"{files[pp]['name']}.bin", files[pp]["N"], D)


def verify_file(path_bin, N, D):
    """Re-reads the file just written and checks header, size and statistics.
    It is used to discover immediately a writing error (truncated file, wrong format)."""
    expected = 8 + N * D * 8
    actual = path_bin.stat().st_size
    header = np.fromfile(path_bin, dtype="<i4", count=2)
    X = np.memmap(path_bin, dtype="<f8", mode="r", offset=8, shape=(N, D))
    sample = np.asarray(X[: min(N, 100_000)])
    ok = (actual == expected) and header[0] == N and header[1] == D
    print(f"    check: header=({header[0]}, {header[1]}), size {'OK' if ok else 'WRONG'}, "
          f"column means in [{sample.mean(axis=0).min():+.2f}, {sample.mean(axis=0).max():+.2f}], "
          f"std in [{sample.std(axis=0).min():.2f}, {sample.std(axis=0).max():.2f}]")
    if not ok:
        raise SystemExit("the written file does not have the expected size")


# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------

def main():
    # Default output folder: <repo>/data/synthetic (the script is in <repo>/src/dataset-managed)
    default_out = Path(__file__).resolve().parents[2] / "data" / "synthetic"

    ap = argparse.ArgumentParser(description="Generates the synthetic datasets for the weak scaling")
    ap.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES),
                    help="profiles to generate (default: all)")
    # 250 000 rows per process: enough work per rank for the computation to dominate the
    # communication (a rough estimate is a few tenths of a second per iteration, depending on
    # the profile; it has not been measured), but with files that are still manageable
    # (from 320 MB to 1.6 GB for the case with p = 16).
    ap.add_argument("--n-per-rank", type=int, default=250_000,
                    help="rows per process, N0 (default: 250000)")
    # Powers of 2 up to 16: they are the typical configurations (a node with 16 cores or
    # several nodes). Change the values if more or fewer ranks can be used on the cluster.
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="numbers of processes of the weak scaling (default: 1 2 4 8 16)")
    # Fixed seed: the same parameters always give the same files (repeatable experiments).
    ap.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    # 500 000 rows at a time: a block with D=50 takes about 200 MB of RAM. Larger blocks use
    # too much memory, smaller ones slow down the generation.
    ap.add_argument("--chunk", type=int, default=500_000,
                    help="rows generated at a time, to avoid filling the RAM (default: 500000)")
    ap.add_argument("--out-dir", type=Path, default=default_out,
                    help=f"output folder (default: {default_out})")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Weak scaling: rows used with p processes ->",
          ", ".join(f"p={pp}: {args.n_per_rank * pp}" for pp in args.procs))
    for index, name in enumerate(PROFILES):
        if name in args.profiles:
            generate_profile(name, index, PROFILES[name], args, args.out_dir)
    print("\nDone. One folder per profile in:", args.out_dir)


if __name__ == "__main__":
    main()
