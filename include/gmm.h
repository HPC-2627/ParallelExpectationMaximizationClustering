#ifndef GMM_H
#define GMM_H

/*
 * gmm.h
 *
 * GMM = Gaussian Mixture Model.
 * This file defines the parameters of the model with DIAGONAL covariance and the functions
 * shared by all the versions (allocation, initialization, log-sum-exp, convergence test).
 *
 * The loops of the E-step and of the M-step are NOT here: they are written in every version,
 * because they are exactly what the optimizations modify (loop fusion, vectorization, ...).
 *
 * MODEL
 *   p(x) = sum_k pi_k * N(x | mu_k, diag(var_k)),     k = 0..K-1
 * i.e. the density of a point x is a weighted sum of K Gaussians. "Diagonal covariance" means
 * that inside every cluster the D variables are independent: every variable has its own mean
 * and variance, but there are no correlations between variables.
 *
 * MEMORY LAYOUT (all the arrays are doubles, in row-major order):
 *   pi[k]            mixing weight of cluster k (share of the data of the cluster; sum = 1)
 *   mu[k*D + d]      mean of variable d in cluster k
 *   var[k*D + d]     variance of variable d in cluster k
 */

#include <stdint.h>

typedef struct
{
    int K;       /* number of clusters */
    int D;       /* number of variables */
    double *pi;  /* K values */
    double *mu;  /* K*D values */
    double *var; /* K*D values */
} GMMParams;

/* Allocates (and zeroes) the arrays of p for K clusters and D variables. Terminates the
 * program if memory is missing. */
void gmm_alloc(GMMParams *p, int K, int D);

/* Releases the arrays of p and sets them to NULL. */
void gmm_free(GMMParams *p);

/* Copies all the parameters of src into dst. dst must already be allocated with the same K
 * and D. */
void gmm_copy(GMMParams *dst, const GMMParams *src);

/*
 * INITIALIZATION, identical in all the versions
 * (EM finds a local maximum: starting from different points it can reach different results,
 * so to compare the versions they must all start from the same point).
 *
 * Step 1: gmm_init_indices() chooses K DISTINCT rows of the dataset (indices in [0, N)) with
 *         rand() seeded with the given seed. Only N, K and the seed are needed, not the data:
 *         every MPI rank can therefore compute the same indices without communicating (then
 *         the rank that owns a chosen row sends it to the others).
 *           N     total number of rows of the dataset
 *           K     number of clusters = how many rows to choose
 *           seed  seed of the generator
 *           idx   (output) array of K distinct indices
 * Step 2: gmm_init_from_rows() builds the parameters from the K chosen rows (rows, K*D values,
 *         row k at rows[k*D ...]):
 *           mu_k = chosen row k,    var = 1,    pi = 1/K.
 *         Variance 1 is the right order of magnitude because the data are standardized
 *         (z-score: mean 0, variance 1); this way no pass over the data is needed (nor
 *         communication between the processes).
 */
void gmm_init_indices(uint64_t N, int K, uint64_t seed, uint64_t *idx);
void gmm_init_from_rows(GMMParams *p, const double *rows);

/* log(sum_k exp(v[k])) for a vector v of K values, computed without overflow/underflow.
 * The trick: the maximum m is subtracted from all the terms, log(sum exp(v)) = m + log(sum exp(v-m)).
 * This way the largest term becomes exp(0) = 1 and the sum is never 0 nor infinite.
 * Computing it directly with exp(v[k]) would give 0 for very negative v (log-densities with
 * large D are hundreds of units below zero) and therefore log(0) = -inf. */
double log_sum_exp(const double *v, int K);

/* Convergence test on the log-likelihood: |ll - ll_prev| <= tol * |ll|
 * (the change is RELATIVE, so the threshold does not depend on the scale of the data).
 * Returns 1 if converged, 0 otherwise. It always returns 0 if ll_prev is not a finite number,
 * i.e. at the first iteration, when there is no previous value yet. */
int gmm_converged(double ll, double ll_prev, double tol);

#endif
