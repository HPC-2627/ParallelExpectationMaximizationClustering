#ifndef ARGS_H
#define ARGS_H

/*
 * args.h
 *
 * Command-line arguments, shared by all the versions.
 *
 * Why shared: if every version (sequential, MPI, optimized) accepts EXACTLY the same options,
 * the same script (for example a PBS script for the cluster) can launch any version by changing
 * only the name of the executable, and the results are comparable.
 *
 * Typical use in the main program:
 *     RunArgs a;
 *     args_defaults(&a, "version-name");      // 1. default values
 *     int rc = args_parse(argc, argv, &a);    // 2. override with the command line
 *     if (rc != 0) return rc > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
 */

#include <stdint.h>

/* All the settings of a run. The fields are filled by args_defaults() and then modified by
 * args_parse() according to the options given by the user. */
typedef struct
{
    const char *dataset;  /* path of the .bin file with the data (required, option -d) */
    int K;                /* number of clusters to look for (required, option -k) */
    long n_rows;          /* how many rows to use, taken from the START of the file (-1 = all) */
    int max_iter;         /* maximum number of EM iterations (safety net: if convergence is not
                             reached, it stops here anyway) */
    double tol;           /* RELATIVE tolerance on the log-likelihood to stop: see
                             gmm_converged() in gmm.h */
    uint64_t seed;        /* seed of the random generator of the initialization. Same seed =
                             same starting point = reproducible result */
    int restarts;         /* number of different initializations (seed, seed+1, ...): the run
                             with the best log-likelihood is kept */
    int warmup;           /* warm-up EM iterations done before the timed run and then discarded
                             (0 = no warm-up). See the warm-up pass in main() of every version */
    int block;            /* rows per block of the blocked versions (opt2 full, option -b);
                             0 = default of the version. The other versions ignore it */
    const char *save_dir; /* folder where the clustering result (labels, means, log-likelihood
                             curve) is saved, see cluster-out.h (NULL = do not save, option --save) */
    int verbose;          /* 1 = print the log-likelihood of every iteration */
    const char *log_path; /* .log file to which the block of results is appended (NULL = no
                             log). After args_parse() it holds the final effective path */
    int no_log;           /* 1 = the user asked not to write the log (--no-log) */
    char log_buf[512];    /* memory that hosts the DEFAULT log path
                             ("results/logs/<name>.log"): in that case log_path points inside
                             it. It lives in the struct to avoid calling malloc */
    const char *name;     /* name of the version, written in the log and used in the default
                             log file name (can be changed with --name) */
} RunArgs;

/* Sets the default values of all the fields.
 *   a             structure to initialize
 *   default_name  name of the version (e.g. "baseline-sequential-diagonal") */
void args_defaults(RunArgs *a, const char *default_name);

/* Reads the command line and fills *a. Return values:
 *    0  everything ok, the program can proceed
 *    1  help was requested (-h): the help has already been printed, exit with SUCCESS
 *   -1  error (missing or invalid argument): the message is already on stderr, exit with error
 * If the user gives neither -l nor --no-log, the log is written to results/logs/<name>.log.
 * The path is RELATIVE to the current directory: the program must therefore be launched from
 * the root of the repository. */
int args_parse(int argc, char **argv, RunArgs *a);

/* Prints the help message (list of options) on stderr. 'prog' is the name of the executable
 * (argv[0]), shown in the "usage:" line. */
void args_usage(const char *prog);

#endif
