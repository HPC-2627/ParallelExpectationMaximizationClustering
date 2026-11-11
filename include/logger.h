#ifndef LOGGER_H
#define LOGGER_H

/*
 * logger.h
 *
 * Text log (.log) of the experiments: one block per run, appended at the END of the file
 * (previous runs are not deleted).
 *
 * The keys are the same for all the versions, so a single script
 * (src/plots-generate/plots-performance.py) can read all the logs to make the performance plots
 * (speedup, efficiency, roofline, load imbalance, ...).
 *
 * Example of a block (the comments on the right are NOT in the file):
 *
 *   ==== run ====
 *   timestamp      : 2026-11-05T09:53:41Z          date and time in UTC
 *   version        : baseline-sequential-diagonal
 *   dataset        : data/synthetic/synth_clean/synth_clean_p01.bin
 *   hostname       : HP-Victus                      node where the run was made
 *   cpu_model      : 12th Gen Intel(R) Core(TM) i5-12450H
 *   ranks          : 1                              MPI processes
 *   threads        : 1                              OpenMP threads per process
 *   N              : 250000                         rows used
 *   D              : 10                             variables
 *   K              : 4                              clusters
 *   restarts       : 1                              initializations tried
 *   warmup_iters   : 3                              warm-up iterations, discarded from the times
 *   iterations     : 12                             of the best run
 *   converged      : 1                              1 = tolerance reached, 0 = max iterations
 *   loglik         : -3087811.89369                 final log-likelihood
 *   ari            : 0.914663                       n/a if the dataset has no labels
 *   t_total        : 0.990871                       seconds: read + init + EM + labels
 *   t_read         : 0.031663                       seconds spent reading the data
 *   t_init         : 2e-06                          seconds spent in the initialization
 *   t_estep        : 0.793119                       seconds, summed over the iterations
 *   t_mstep        : 0.162592                       seconds, summed over the iterations
 *   t_comm         : 0                              seconds of communication (0 if sequential)
 *   t_warmup       : 0.231044                       seconds of the warm-up (not in t_total)
 *   flops          : 3.1e+09                        analytic FLOP of the timed run (perfmodel.h)
 *   bytes          : 2.4e+09                        analytic main-memory traffic (perfmodel.h)
 *   rank_compute   : 0.61 0.60 0.63 0.62            compute seconds (E-step + M-step, no MPI) of
 *                                                   EACH rank, in rank order; n/a if sequential
 *   t_iteration    : 0.079643                       (t_estep + t_mstep + t_comm) / iterations
 *
 * Every line is "key : value": it is readable by eye and easy to parse (just split at the
 * first ':'). Runs are separated by an empty line. hostname and cpu_model identify the node,
 * because the nodes of the cluster are not all equal and the times depend on the machine.
 * In the MPI versions only rank 0 writes the block. The functions of this file do not use MPI.
 */

/* Everything that must be written in the log for ONE run. It is filled by the main program. */
typedef struct
{
    const char *version; /* name of the version (e.g. baseline-sequential-diagonal) */
    const char *dataset; /* path of the dataset */
    int ranks;           /* MPI processes (1 in the sequential version) */
    int threads;         /* OpenMP threads per process (1 in the sequential version) */
    int N, D, K;         /* rows used, variables, clusters */
    int restarts;        /* number of initializations tried */
    int warmup_iters;    /* warm-up EM iterations done before the timed run (0 = none) */
    int iterations;      /* EM iterations of the best run */
    int converged;       /* 1 = tolerance reached, 0 = maximum number of iterations */
    double loglik;       /* final log-likelihood */
    double ari;          /* adjusted Rand index; negative (< -1.5) or NaN if not available */
    double t_total;      /* whole program: read + initialization + EM + labels (s) */
    double t_read;       /* reading of the dataset (s) */
    double t_init;       /* initialization (s) */
    double t_estep;      /* E-step, summed over the iterations (s) */
    double t_mstep;      /* M-step, summed over the iterations (s) */
    double t_comm;       /* communication between processes (s; 0 in the sequential version) */
    double t_warmup;     /* warm-up iterations (s): NOT included in t_total nor in the other times */
    double flops;        /* analytic FLOP of the timed run, see perfmodel.h (<= 0 = not available) */
    double bytes;        /* analytic main-memory traffic in bytes (<= 0 = not available) */
    const double *rank_compute; /* compute time of every rank (s), array of n_rank_compute values
                                   in rank order; NULL if not available (sequential versions) */
    int n_rank_compute;  /* length of rank_compute (= number of MPI ranks) */
} RunStats;

/* Appends a block to the .log file at 'path'. Missing folders of the path are created.
 * Returns 0 if ok, -1 on error (for example file not writable). */
int log_run(const char *path, const RunStats *s);

#endif
