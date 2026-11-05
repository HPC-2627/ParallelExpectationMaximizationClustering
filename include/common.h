#ifndef COMMON_H
#define COMMON_H

/*
 * common.h
 *
 * Numerical constants and utility macros shared by all the versions of the program
 * (sequential, MPI + OpenMP baseline, optimizations).
 *
 * The constants below are "numerical safeguards": they prevent EM from producing infinity (inf)
 * or "not a number" (NaN) in edge cases. They are defined in a single place because ALL the
 * versions must use the same values: if one version used a different threshold, the final
 * log-likelihoods would no longer be comparable.
 */

#include <stdio.h>
#include <stdlib.h>

/* ------------------------------------------------------------------------------------------
 * Include guard: the first two lines of the file (#ifndef / #define) and the last one (#endif)
 * prevent this file from being included twice in the same translation unit, which would cause
 * redefinition errors.
 * ------------------------------------------------------------------------------------------ */

/* log(2*pi) = 1.8378770664...
 * The log-density of a Gaussian with D variables contains the term  -(D/2) * log(2*pi).
 * It is a constant: it is written once instead of calling log(2*M_PI) at every use.
 * (M_PI is not part of the C standard; writing the value avoids portability problems.) */
#define LOG_2PI 1.8378770664093453

/* VAR_FLOOR: minimum allowed variance of a variable inside a cluster ("regularization").
 * Problem: if a cluster captures a few identical points (or a constant variable), the variance
 * computed by the M-step is 0. In the E-step the density contains 1/var and log(var): with
 * var = 0 there would be a division by zero and the log-likelihood would become infinite,
 * breaking the algorithm. Enforcing var >= VAR_FLOOR prevents this.
 * The value 1e-6 is tiny compared to the typical variance of standardized data
 * (z-score: mean 0, variance ~1), so it does not alter the results in normal cases. */
#define VAR_FLOOR 1e-6

/* NK_EMPTY: threshold below which a cluster is considered EMPTY.
 * N_k = sum of the responsibilities of cluster k = "how many rows" the cluster owns.
 * If N_k < NK_EMPTY the cluster has practically no points: computing its mean and variance
 * would require dividing by ~0 (NaN result). In that case the M-step leaves mean and
 * variances as in the previous iteration and skips the cluster. */
#define NK_EMPTY 1e-10

/* PI_FLOOR: minimum mixing weight of a cluster.
 * The E-step contains log(pi_k): if pi_k were exactly 0 it would give -infinity. An empty
 * cluster therefore gets this tiny (but > 0) weight instead of 0. */
#define PI_FLOOR 1e-12

/* ------------------------------------------------------------------------------------------
 * DIE(...): prints "error: <message>" on stderr and terminates the program with an error code.
 *
 * It is used like printf:  DIE("file %s not found", name);
 * The first argument MUST be a string literal (in quotes), because the preprocessor
 * concatenates it with "error: " (concatenation of adjacent string literals in C).
 *
 * The "do { ... } while (0)" structure is the standard way to write a multi-statement macro:
 * it makes it usable as a single statement even inside an if/else without braces, and it
 * requires the final semicolon like a normal call.
 * ------------------------------------------------------------------------------------------ */
#define DIE(...)                          \
    do                                    \
    {                                     \
        fprintf(stderr, "error: " __VA_ARGS__); \
        fprintf(stderr, "\n");            \
        exit(EXIT_FAILURE);               \
    } while (0)

/* ------------------------------------------------------------------------------------------
 * CHECK_ALLOC(ptr): terminates with a clear message if malloc/calloc returned NULL
 * (out of memory). Without the check, the program would segfault at the first use of the
 * pointer, without explaining the cause.
 * __FILE__ and __LINE__ are replaced by the preprocessor with the name of the source file and
 * the line number where the macro was written: they tell which allocation failed.
 * ------------------------------------------------------------------------------------------ */
#define CHECK_ALLOC(ptr)                                          \
    do                                                            \
    {                                                             \
        if ((ptr) == NULL)                                        \
            DIE("out of memory (%s:%d)", __FILE__, __LINE__);     \
    } while (0)

#endif
